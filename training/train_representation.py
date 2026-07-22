"""Train BP or greedy layer-wise Hebbian representations with exact resume."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from torch import nn

from data.mnist import build_mnist_dataloaders
from learning_rules import build_trainer
from models import ConvAutoencoder
from schemas import load_config, validate_config
from utils.checkpointing import (
    config_fingerprint,
    file_sha256,
    load_resume_checkpoint,
    restore_loader_generator,
    restore_rng_state,
    save_epoch_checkpoint,
    utc_now,
)
from utils.reproducibility import git_provenance, set_global_seed, state_dict_checksum
from utils.results import (
    append_metric,
    create_run_directory,
    initialize_run_status,
    read_metadata,
    read_run_status,
    update_run_status,
    write_json,
    write_metadata,
    write_resolved_config,
)


def _elapsed(started: float, offset: float) -> float:
    return offset + time.perf_counter() - started


def _checkpoint_progress(
    *,
    stage: str,
    active_layer: str,
    completed_epoch: int,
    global_epoch: int,
    samples_seen: int,
    steps_completed: int,
    wall_time_sec: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "active_layer": active_layer,
        "completed_epoch": completed_epoch,
        "global_epoch": global_epoch,
        "samples_seen": samples_seen,
        "steps_completed": steps_completed,
        "wall_time_sec": wall_time_sec,
        **extra,
    }


def _record_status(run_dir: Path, progress: dict[str, Any], **extra: Any) -> None:
    update_run_status(
        run_dir,
        stage=progress["stage"],
        active_layer=progress["active_layer"],
        completed_epoch=progress["completed_epoch"],
        global_epoch=progress["global_epoch"],
        samples_seen=progress["samples_seen"],
        steps_completed=progress["steps_completed"],
        wall_time_sec=progress["wall_time_sec"],
        **extra,
    )


def _metric_progress(progress: dict[str, Any], archive_name: str) -> dict[str, Any]:
    return {
        "checkpoint_id": Path(archive_name).stem,
        "global_epoch": progress["global_epoch"],
        "step": progress["steps_completed"],
        "samples_seen": progress["samples_seen"],
        "wall_time_sec": progress["wall_time_sec"],
    }


def _train_bp(
    model,
    trainer,
    loaders,
    config: dict,
    run_dir: Path,
    *,
    resume_payload: dict[str, Any] | None,
    stop_after_global_epoch: int | None,
) -> bool:
    if resume_payload is None:
        completed_epoch = 0
        global_epoch = 0
        samples_seen = 0
        steps_completed = 0
        wall_offset = 0.0
        best_validation_loss = float("inf")
        best_epoch = 0
    else:
        progress = resume_payload["progress"]
        completed_epoch = int(progress["completed_epoch"])
        global_epoch = int(progress["global_epoch"])
        samples_seen = int(progress["samples_seen"])
        steps_completed = int(progress["steps_completed"])
        wall_offset = float(progress["wall_time_sec"])
        best_validation_loss = float(progress["best_validation_loss"])
        best_epoch = int(progress["best_epoch"])
        trainer.optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    started = time.perf_counter()

    for epoch in range(completed_epoch + 1, config["training"]["bp_epochs"] + 1):
        train_metrics = trainer.run_epoch(loaders["train"], training=True)
        validation_metrics = trainer.run_epoch(loaders["validation"], training=False)
        global_epoch += 1
        samples_seen += train_metrics.num_samples
        steps_completed += len(loaders["train"])
        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        torch.save(model.state_dict(), run_dir / "model_last.pt")
        archive_name = f"bp_representation_epoch_{epoch:03d}.pt"
        progress = _checkpoint_progress(
            stage="representation",
            active_layer="all",
            completed_epoch=epoch,
            global_epoch=global_epoch,
            samples_seen=samples_seen,
            steps_completed=steps_completed,
            wall_time_sec=_elapsed(started, wall_offset),
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
        )
        for split, metrics in (("train", train_metrics), ("validation", validation_metrics)):
            append_metric(
                run_dir,
                {
                    "stage": "representation",
                    "split": split,
                    "epoch": epoch,
                    "dataset_passes": epoch,
                    "reconstruction_loss": metrics.loss,
                    "num_samples": metrics.num_samples,
                    **_metric_progress(progress, archive_name),
                },
            )
        save_epoch_checkpoint(
            run_dir,
            archive_name=archive_name,
            config=config,
            rule="bp",
            stage="representation",
            model=model,
            optimizer=trainer.optimizer,
            progress=progress,
            train_loader=loaders["train"],
        )
        _record_status(run_dir, progress, status="running", checkpoint=archive_name)
        print(
            f"epoch={epoch:02d} train_mse={train_metrics.loss:.6f} "
            f"validation_mse={validation_metrics.loss:.6f}",
            flush=True,
        )
        if stop_after_global_epoch is not None and global_epoch >= stop_after_global_epoch:
            _record_status(run_dir, progress, status="paused", checkpoint=archive_name)
            return False

    write_json(
        run_dir,
        "training_summary.json",
        {"best_validation_mse": best_validation_loss, "best_epoch": best_epoch},
        overwrite=True,
    )
    return True


@torch.no_grad()
def _decoder_validation(model, loader, criterion, device: torch.device) -> tuple[float, int]:
    model.encoder.eval()
    model.decoder.eval()
    total_loss = 0.0
    total_samples = 0
    for images, _, _ in loader:
        images = images.to(device)
        reconstruction = model.decode(model.encode(images))
        loss = criterion(reconstruction, images)
        total_loss += float(loss.item()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples, total_samples


def _train_frozen_decoder(
    model,
    loaders,
    config: dict,
    run_dir: Path,
    device,
    *,
    resume_payload: dict[str, Any] | None,
    base_progress: dict[str, Any],
    stop_after_global_epoch: int | None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    bp = config["backprop"]
    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=bp["lr"],
        betas=tuple(bp["betas"]),
        weight_decay=bp["weight_decay"],
    )
    criterion = nn.MSELoss(reduction="mean")
    if resume_payload is not None and resume_payload["stage"] == "decoder":
        progress = resume_payload["progress"]
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        completed_epoch = int(progress["completed_epoch"])
        global_epoch = int(progress["global_epoch"])
        samples_seen = int(progress["samples_seen"])
        steps_completed = int(progress["steps_completed"])
        wall_offset = float(progress["wall_time_sec"])
        best_validation_loss = float(progress["best_validation_loss"])
        best_epoch = int(progress["best_epoch"])
        encoder_hash_before = progress["encoder_hash_before_decoder"]
    else:
        completed_epoch = 0
        global_epoch = int(base_progress["global_epoch"])
        samples_seen = int(base_progress["samples_seen"])
        steps_completed = int(base_progress["steps_completed"])
        wall_offset = float(base_progress["wall_time_sec"])
        best_validation_loss = float("inf")
        best_epoch = 0
        encoder_hash_before = state_dict_checksum(model.encoder)
    started = time.perf_counter()

    for epoch in range(completed_epoch + 1, config["training"]["decoder_epochs"] + 1):
        model.encoder.eval()
        model.decoder.train()
        total_loss = 0.0
        total_samples = 0
        for images, _, _ in loaders["train"]:
            images = images.to(device)
            with torch.no_grad():
                latent = model.encode(images)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.decode(latent), images)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * images.shape[0]
            total_samples += images.shape[0]
        train_loss = total_loss / total_samples
        validation_loss, validation_samples = _decoder_validation(
            model, loaders["validation"], criterion, device
        )
        global_epoch += 1
        samples_seen += total_samples
        steps_completed += len(loaders["train"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        torch.save(model.state_dict(), run_dir / "model_last.pt")
        archive_name = f"decoder_epoch_{epoch:03d}.pt"
        progress = _checkpoint_progress(
            stage="decoder",
            active_layer="decoder",
            completed_epoch=epoch,
            global_epoch=global_epoch,
            samples_seen=samples_seen,
            steps_completed=steps_completed,
            wall_time_sec=_elapsed(started, wall_offset),
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            encoder_hash_before_decoder=encoder_hash_before,
            layer_summaries=base_progress.get("layer_summaries", {}),
        )
        for split, loss_value, sample_count in (
            ("train", train_loss, total_samples),
            ("validation", validation_loss, validation_samples),
        ):
            append_metric(
                run_dir,
                {
                    "stage": "decoder",
                    "split": split,
                    "epoch": epoch,
                    "dataset_passes": epoch,
                    "reconstruction_loss": loss_value,
                    "num_samples": sample_count,
                    **_metric_progress(progress, archive_name),
                },
            )
        save_epoch_checkpoint(
            run_dir,
            archive_name=archive_name,
            config=config,
            rule="hebbian",
            stage="decoder",
            model=model,
            optimizer=optimizer,
            progress=progress,
            train_loader=loaders["train"],
        )
        _record_status(run_dir, progress, status="running", checkpoint=archive_name)
        print(
            f"decoder epoch={epoch:02d} train_mse={train_loss:.6f} "
            f"validation_mse={validation_loss:.6f}",
            flush=True,
        )
        if stop_after_global_epoch is not None and global_epoch >= stop_after_global_epoch:
            _record_status(run_dir, progress, status="paused", checkpoint=archive_name)
            return {}, False, progress

    encoder_hash_after = state_dict_checksum(model.encoder)
    if encoder_hash_after != encoder_hash_before:
        raise RuntimeError("Hebbian encoder changed during frozen decoder training")
    summary = {
        "best_decoder_epoch": best_epoch,
        "best_decoder_validation_mse": best_validation_loss,
        "encoder_hash_before_decoder": encoder_hash_before,
        "encoder_hash_after_decoder": encoder_hash_after,
        "encoder_unchanged_during_decoder_training": True,
    }
    return summary, True, progress


def _train_hebbian(
    model,
    trainer,
    loaders,
    config: dict,
    run_dir: Path,
    device,
    *,
    resume_payload: dict[str, Any] | None,
    stop_after_global_epoch: int | None,
) -> bool:
    epochs_per_layer = config["training"]["hebbian_epochs_per_layer"]
    if resume_payload is None:
        global_epoch = samples_seen = steps_completed = 0
        wall_offset = 0.0
        layer_summaries: dict[str, dict[str, Any]] = {}
        resume_layer = trainer.layer_names[0]
        resume_epoch = 0
    else:
        progress = resume_payload["progress"]
        global_epoch = int(progress["global_epoch"])
        samples_seen = int(progress["samples_seen"])
        steps_completed = int(progress["steps_completed"])
        wall_offset = float(progress["wall_time_sec"])
        layer_summaries = progress.get("layer_summaries", {})
        resume_layer = progress["active_layer"]
        resume_epoch = int(progress["completed_epoch"])
    started = time.perf_counter()

    if resume_payload is None or resume_payload["stage"] == "hebbian_encoder":
        resume_index = trainer.layer_names.index(resume_layer)
        for layer_index, layer_name in enumerate(trainer.layer_names):
            if layer_index < resume_index:
                continue
            first_epoch = resume_epoch + 1 if layer_index == resume_index else 1
            for epoch in range(first_epoch, epochs_per_layer + 1):
                diagnostics = trainer.train_layer_epoch(loaders["train"], layer_name)
                global_epoch += 1
                samples_seen += diagnostics.num_samples
                steps_completed += len(loaders["train"])
                if epoch == epochs_per_layer:
                    checkpoint_path = run_dir / f"encoder_{layer_name}_end.pt"
                    torch.save(model.encoder.state_dict(), checkpoint_path)
                    layer_summaries[layer_name] = {
                        "checkpoint": checkpoint_path.name,
                        "encoder_hash": state_dict_checksum(model.encoder),
                        "last_epoch": {
                            "update_norm": diagnostics.update_norm,
                            "activation_mean": diagnostics.activation_mean,
                            "activation_sparsity": diagnostics.activation_sparsity,
                            "active_neuron_ratio": diagnostics.active_neuron_ratio,
                            "winner_entropy": diagnostics.winner_entropy,
                            "max_winner_share": diagnostics.max_winner_share,
                            "collapse_detected": diagnostics.collapse_detected,
                        },
                    }
                archive_name = f"hebbian_{layer_name}_epoch_{epoch:03d}.pt"
                progress = _checkpoint_progress(
                    stage="hebbian_encoder",
                    active_layer=layer_name,
                    completed_epoch=epoch,
                    global_epoch=global_epoch,
                    samples_seen=samples_seen,
                    steps_completed=steps_completed,
                    wall_time_sec=_elapsed(started, wall_offset),
                    layer_summaries=layer_summaries,
                )
                append_metric(
                    run_dir,
                    {
                        "stage": "hebbian_encoder",
                        "split": "train",
                        "layer": layer_name,
                        "epoch": epoch,
                        "dataset_passes": epoch,
                        "update_norm": diagnostics.update_norm,
                        "weight_norm_mean": diagnostics.weight_norm_mean,
                        "weight_norm_std": diagnostics.weight_norm_std,
                        "preactivation_mean": diagnostics.preactivation_mean,
                        "preactivation_std": diagnostics.preactivation_std,
                        "activation_mean": diagnostics.activation_mean,
                        "activation_variance": diagnostics.activation_variance,
                        "activation_sparsity": diagnostics.activation_sparsity,
                        "active_neuron_ratio": diagnostics.active_neuron_ratio,
                        "winner_entropy": diagnostics.winner_entropy,
                        "max_winner_share": diagnostics.max_winner_share,
                        "collapse_detected": diagnostics.collapse_detected,
                        "num_samples": diagnostics.num_samples,
                        **_metric_progress(progress, archive_name),
                    },
                )
                save_epoch_checkpoint(
                    run_dir,
                    archive_name=archive_name,
                    config=config,
                    rule="hebbian",
                    stage="hebbian_encoder",
                    model=model,
                    optimizer=None,
                    progress=progress,
                    train_loader=loaders["train"],
                )
                _record_status(run_dir, progress, status="running", checkpoint=archive_name)
                print(
                    f"layer={layer_name} epoch={epoch:02d} "
                    f"update_norm={diagnostics.update_norm:.6f} "
                    f"active={diagnostics.active_neuron_ratio:.4f} "
                    f"max_winner_share={diagnostics.max_winner_share:.4f} "
                    f"collapse={diagnostics.collapse_detected}",
                    flush=True,
                )
                if stop_after_global_epoch is not None and global_epoch >= stop_after_global_epoch:
                    _record_status(run_dir, progress, status="paused", checkpoint=archive_name)
                    return False
            resume_epoch = 0

        torch.save(model.encoder.state_dict(), run_dir / "encoder_hebbian.pt")
        base_progress = progress
        decoder_resume = None
    else:
        base_progress = resume_payload["progress"]
        decoder_resume = resume_payload

    decoder_summary, complete, final_progress = _train_frozen_decoder(
        model,
        loaders,
        config,
        run_dir,
        device,
        resume_payload=decoder_resume,
        base_progress=base_progress,
        stop_after_global_epoch=stop_after_global_epoch,
    )
    if not complete:
        return False
    write_json(
        run_dir,
        "hebbian_training_summary.json",
        {
            "rule": "competitive_oja_topk",
            "winner_fraction": config["hebbian"]["winner_fraction"],
            "learning_rate": config["hebbian"]["lr"],
            "layer_lrs": config["hebbian"].get("layer_lrs", {}),
            "epochs_per_layer": epochs_per_layer,
            "layer_order": list(trainer.layer_names),
            "final_encoder_hash": state_dict_checksum(model.encoder),
            "layers": final_progress.get("layer_summaries", layer_summaries),
            **decoder_summary,
        },
        overwrite=True,
    )
    return True


def _new_run_metadata(
    run_dir: Path, config: dict[str, Any], model: ConvAutoencoder, device: torch.device
) -> dict[str, Any]:
    manifest = Path(config["data"]["split_manifest"])
    architecture = model.architecture_metadata()
    return {
        "version": config["version"],
        "experiment_id": f"{config['version']}_{config['data']['dataset'].lower()}_{config['model']['architecture']}",
        "run_id": run_dir.name,
        "created_at_utc": utc_now(),
        "config_sha256": config_fingerprint(config),
        "split_manifest": str(manifest),
        "split_manifest_sha256": file_sha256(manifest) if manifest.exists() else None,
        "learning_rule": config["training"]["learning_rule"],
        "model_type": "convolutional_autoencoder",
        "architecture_id": config["model"]["architecture"],
        "seed": int(config["training"]["seed"]),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "initial_state_hash": state_dict_checksum(model),
        **git_provenance(str(Path.cwd())),
        **architecture,
    }


def train_config(
    config: dict[str, Any],
    *,
    loaders=None,
    run_root: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
    stop_after_global_epoch: int | None = None,
) -> Path:
    """Run a resolved config; ``stop_after_global_epoch`` simulates preemption."""

    validate_config(config)
    seed = int(config["training"]["seed"])
    rule = config["training"]["learning_rule"]
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaders is None:
        loaders = build_mnist_dataloaders(config, seed=seed)
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
    trainer = build_trainer(model, config, device)

    resume_payload = None
    if resume_run_dir is None:
        root = config["results"]["root"] if run_root is None else run_root
        run_dir = create_run_directory(root, rule=rule, seed=seed)
        write_resolved_config(run_dir, config)
        write_metadata(run_dir, _new_run_metadata(run_dir, config, model, device))
        initialize_run_status(
            run_dir,
            {
                "status": "running",
                "stage": "not_started",
                "active_layer": "",
                "completed_epoch": 0,
                "global_epoch": 0,
                "samples_seen": 0,
                "steps_completed": 0,
                "wall_time_sec": 0.0,
                "resume_count": 0,
                "checkpoint": None,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "error": None,
            },
        )
        if rule == "bp":
            initial_progress = _checkpoint_progress(
                stage="representation",
                active_layer="all",
                completed_epoch=0,
                global_epoch=0,
                samples_seen=0,
                steps_completed=0,
                wall_time_sec=0.0,
                best_validation_loss=float("inf"),
                best_epoch=0,
            )
            initial_optimizer = trainer.optimizer
        else:
            initial_progress = _checkpoint_progress(
                stage="hebbian_encoder",
                active_layer=trainer.layer_names[0],
                completed_epoch=0,
                global_epoch=0,
                samples_seen=0,
                steps_completed=0,
                wall_time_sec=0.0,
                layer_summaries={},
            )
            initial_optimizer = None
        save_epoch_checkpoint(
            run_dir,
            archive_name="initial_state.pt",
            config=config,
            rule=rule,
            stage=initial_progress["stage"],
            model=model,
            optimizer=initial_optimizer,
            progress=initial_progress,
            train_loader=loaders["train"],
        )
        _record_status(
            run_dir,
            initial_progress,
            status="running",
            checkpoint="initial_state.pt",
        )
    else:
        run_dir = Path(resume_run_dir)
        status = read_run_status(run_dir)
        if status["status"] == "completed":
            return run_dir
        metadata = read_metadata(run_dir)
        if metadata["config_sha256"] != config_fingerprint(config):
            raise RuntimeError("Resume config does not match run metadata")
        resume_payload = load_resume_checkpoint(run_dir, config, rule=rule)
        model.load_state_dict(resume_payload["model_state_dict"])
        restore_rng_state(resume_payload["rng_state"])
        restore_loader_generator(loaders["train"], resume_payload["train_loader_generator_state"])
        update_run_status(
            run_dir,
            status="running",
            resume_count=int(status.get("resume_count", 0)) + 1,
            error=None,
        )

    try:
        if rule == "bp":
            complete = _train_bp(
                model,
                trainer,
                loaders,
                config,
                run_dir,
                resume_payload=resume_payload,
                stop_after_global_epoch=stop_after_global_epoch,
            )
        elif rule == "hebbian":
            complete = _train_hebbian(
                model,
                trainer,
                loaders,
                config,
                run_dir,
                device,
                resume_payload=resume_payload,
                stop_after_global_epoch=stop_after_global_epoch,
            )
        else:
            raise ValueError(f"Unsupported learning rule: {rule}")
    except Exception as error:
        update_run_status(run_dir, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    if complete:
        update_run_status(run_dir, status="completed", completed_at_utc=utc_now(), error=None)
    return run_dir


def train(
    config_path: str | Path | None = None,
    *,
    resume_run_dir: str | Path | None = None,
    stop_after_global_epoch: int | None = None,
) -> Path:
    if resume_run_dir is not None:
        resolved_path = Path(resume_run_dir) / "config_resolved.yaml"
        with resolved_path.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle)
        if config_path is not None:
            requested = load_config(config_path)
            if config_fingerprint(requested) != config_fingerprint(resolved):
                raise RuntimeError("Provided config differs from the run's resolved config")
        config = resolved
    elif config_path is not None:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or resume_run_dir is required")
    return train_config(
        config,
        resume_run_dir=resume_run_dir,
        stop_after_global_epoch=stop_after_global_epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--stop-after-global-epoch", type=int)
    args = parser.parse_args()
    if not args.config and not args.resume_run_dir:
        parser.error("one of --config or --resume-run-dir is required")
    print(
        train(
            args.config,
            resume_run_dir=args.resume_run_dir,
            stop_after_global_epoch=args.stop_after_global_epoch,
        ).resolve()
    )


if __name__ == "__main__":
    main()
