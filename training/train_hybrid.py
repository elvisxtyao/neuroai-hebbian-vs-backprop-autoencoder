"""Resumable layer-rule training for the finite hybrid depth ablation."""

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
from learning_rules.hebbian import HebbianTrainer
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
    initialize_run_status,
    read_metadata,
    read_run_status,
    update_run_status,
    write_json,
    write_metadata,
    write_resolved_config,
)


RULE = "hybrid_depth_ablation"
LAYERS = ("enc1", "enc2", "enc3")


def _layer_hashes(model: ConvAutoencoder) -> dict[str, str]:
    return {
        layer: state_dict_checksum(getattr(model.encoder, layer))
        for layer in LAYERS
    }


def _parameter_manifest(
    model: ConvAutoencoder,
    layer_rules: dict[str, str],
) -> tuple[dict[str, Any], list[nn.Parameter]]:
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_parameters: list[nn.Parameter] = []
    for layer in LAYERS:
        trainable = layer_rules[layer] == "bp"
        for name, parameter in getattr(model.encoder, layer).named_parameters():
            full_name = f"encoder.{layer}.{name}"
            parameter.requires_grad_(trainable)
            (trainable_names if trainable else frozen_names).append(full_name)
            if trainable:
                trainable_parameters.append(parameter)
    for name, parameter in model.decoder.named_parameters():
        parameter.requires_grad_(True)
        trainable_names.append(f"decoder.{name}")
        trainable_parameters.append(parameter)
    manifest = {
        "schema_version": "hybrid-parameter-manifest-v1",
        "encoder_layer_rules": layer_rules,
        "bp_trainable_parameter_names": trainable_names,
        "bp_frozen_parameter_names": frozen_names,
        "bp_trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        "bp_frozen_parameter_count": sum(
            parameter.numel()
            for layer in LAYERS
            if layer_rules[layer] != "bp"
            for parameter in getattr(model.encoder, layer).parameters()
        ),
        "frozen_layer_hashes_before_bp": {
            layer: state_dict_checksum(getattr(model.encoder, layer))
            for layer in LAYERS
            if layer_rules[layer] != "bp"
        },
        "trainable_layer_hashes_before_bp": {
            layer: state_dict_checksum(getattr(model.encoder, layer))
            for layer in LAYERS
            if layer_rules[layer] == "bp"
        },
        "decoder_hash_before_bp": state_dict_checksum(model.decoder),
    }
    optimizer_ids = {
        id(parameter)
        for group_parameter in trainable_parameters
        for parameter in (group_parameter,)
    }
    expected_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if optimizer_ids != expected_ids:
        raise RuntimeError("BP optimizer parameter manifest is incomplete")
    return manifest, trainable_parameters


def _evaluate_reconstruction(
    model: ConvAutoencoder,
    loader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            loss = criterion(model(images), images)
            total += float(loss.item()) * images.shape[0]
            samples += images.shape[0]
    return total / samples, samples


def _train_bp_epoch(
    model: ConvAutoencoder,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    samples = 0
    for images, _, _ in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), images)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item()) * images.shape[0]
        samples += images.shape[0]
    return total / samples, samples


def _progress(
    *,
    stage: str,
    active_layer: str,
    completed_epoch: int,
    global_epoch: int,
    samples_seen: int,
    steps_completed: int,
    wall_time_sec: float,
    best_validation_mse: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "active_layer": active_layer,
        "completed_epoch": completed_epoch,
        "global_epoch": global_epoch,
        "samples_seen": samples_seen,
        "steps_completed": steps_completed,
        "wall_time_sec": wall_time_sec,
        "best_validation_mse": best_validation_mse,
        "best_epoch": best_epoch,
    }


def _metadata(
    run_dir: Path,
    config: dict[str, Any],
    model: ConvAutoencoder,
    device: torch.device,
) -> dict[str, Any]:
    split = Path(config["data"]["split_manifest"])
    return {
        "schema_version": "hybrid-depth-run-metadata-v1",
        "version": config["version"],
        "experiment_id": (
            "hybrid_hhb_confirmation"
            if config["hybrid"].get("confirmation_stage") == "stage2d"
            else (
                "stage3_formal_core"
                if config["hybrid"].get("confirmation_stage") == "stage3_core"
                else "hybrid_depth_ablation_seed42"
            )
        ),
        "run_id": run_dir.name,
        "method_id": config["hybrid"]["method_id"],
        "learning_rule": "hybrid",
        "encoder_layer_rules": config["hybrid"]["encoder_layer_rules"],
        "created_at_utc": utc_now(),
        "config_sha256": config_fingerprint(config),
        "split_manifest": str(split),
        "split_manifest_sha256": file_sha256(split),
        "seed": int(config["training"]["seed"]),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "initial_state_hash": state_dict_checksum(model),
        "initial_encoder_hash": state_dict_checksum(model.encoder),
        "initial_decoder_hash": state_dict_checksum(model.decoder),
        "initial_layer_hashes": _layer_hashes(model),
        "test_samples_accessed": 0,
        **git_provenance(str(Path.cwd())),
        **model.architecture_metadata(),
    }


def train_hybrid_config(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    loaders=None,
    resume: bool = True,
    stop_after_global_epoch: int | None = None,
) -> Path:
    validate_config(config)
    if config["training"]["learning_rule"] != "hybrid":
        raise ValueError("train_hybrid_config requires learning_rule=hybrid")
    seed = int(config["training"]["seed"])
    confirmation_stage = config["hybrid"].get("confirmation_stage")
    allowed_seeds = (
        {43, 44}
        if confirmation_stage == "stage2d"
        else ({0, 1, 2, 3, 4} if confirmation_stage == "stage3_core" else {42})
    )
    if seed not in allowed_seeds:
        stage_name = (
            "Stage 2D"
            if confirmation_stage == "stage2d"
            else ("Stage 3" if confirmation_stage == "stage3_core" else "Stage 2C")
        )
        raise ValueError(
            f"{stage_name} hybrid training requires one of "
            f"{sorted(allowed_seeds)}, got {seed}"
        )
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaders is None:
        loaders = build_mnist_dataloaders(config, seed=seed, include_test=False)
    if "test" in loaders:
        raise RuntimeError("Hybrid validation-only training must not construct a test loader")

    run_dir = Path(run_dir)
    layer_rules = dict(config["hybrid"]["encoder_layer_rules"])
    hebbian_layers = [layer for layer in LAYERS if layer_rules[layer] == "hebbian"]
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed).to(device)
    hebbian = HebbianTrainer(model, config, device)
    payload: dict[str, Any] | None = None

    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"Run already exists: {run_dir}")
        status = read_run_status(run_dir)
        if status["status"] == "completed":
            return run_dir
        metadata = read_metadata(run_dir)
        if metadata["config_sha256"] != config_fingerprint(config):
            raise RuntimeError("Resume config does not match metadata")
        payload = load_resume_checkpoint(run_dir, config, rule=RULE)
        model.load_state_dict(payload["model_state_dict"])
        restore_rng_state(payload["rng_state"])
        restore_loader_generator(
            loaders["train"], payload["train_loader_generator_state"]
        )
        update_run_status(
            run_dir,
            status="running",
            resume_count=int(status.get("resume_count", 0)) + 1,
            error=None,
        )
    else:
        run_dir.mkdir(parents=True)
        write_resolved_config(run_dir, config)
        write_metadata(run_dir, _metadata(run_dir, config, model, device))
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
                "test_samples_accessed": 0,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "error": None,
            },
        )
        initial_stage = "hybrid_hebbian" if hebbian_layers else "hybrid_bp_joint"
        initial_layer = hebbian_layers[0] if hebbian_layers else "joint"
        initial_progress = _progress(
            stage=initial_stage,
            active_layer=initial_layer,
            completed_epoch=0,
            global_epoch=0,
            samples_seen=0,
            steps_completed=0,
            wall_time_sec=0.0,
            best_validation_mse=float("inf"),
            best_epoch=0,
        )
        save_epoch_checkpoint(
            run_dir,
            archive_name="initial_state.pt",
            config=config,
            rule=RULE,
            stage=initial_stage,
            model=model,
            optimizer=None,
            progress=initial_progress,
            train_loader=loaders["train"],
        )
        update_run_status(
            run_dir,
            **initial_progress,
            checkpoint="initial_state.pt",
        )

    if payload is None:
        global_epoch = samples_seen = steps_completed = 0
        wall_offset = 0.0
        resume_stage = "hybrid_hebbian" if hebbian_layers else "hybrid_bp_joint"
        resume_layer = hebbian_layers[0] if hebbian_layers else "joint"
        resume_epoch = 0
        best_validation_mse = float("inf")
        best_epoch = 0
    else:
        previous = payload["progress"]
        global_epoch = int(previous["global_epoch"])
        samples_seen = int(previous["samples_seen"])
        steps_completed = int(previous["steps_completed"])
        wall_offset = float(previous["wall_time_sec"])
        resume_stage = payload["stage"]
        resume_layer = previous["active_layer"]
        resume_epoch = int(previous["completed_epoch"])
        best_validation_mse = float(previous["best_validation_mse"])
        best_epoch = int(previous["best_epoch"])
    started = time.perf_counter()

    try:
        if resume_stage == "hybrid_hebbian":
            resume_index = hebbian_layers.index(resume_layer)
            for layer_index, layer in enumerate(hebbian_layers):
                if layer_index < resume_index:
                    continue
                first_epoch = resume_epoch + 1 if layer_index == resume_index else 1
                for epoch in range(
                    first_epoch,
                    int(config["training"]["hebbian_epochs_per_layer"]) + 1,
                ):
                    diagnostics = hebbian.train_layer_epoch(loaders["train"], layer)
                    global_epoch += 1
                    samples_seen += diagnostics.num_samples
                    steps_completed += len(loaders["train"])
                    wall_time = wall_offset + time.perf_counter() - started
                    progress = _progress(
                        stage="hybrid_hebbian",
                        active_layer=layer,
                        completed_epoch=epoch,
                        global_epoch=global_epoch,
                        samples_seen=samples_seen,
                        steps_completed=steps_completed,
                        wall_time_sec=wall_time,
                        best_validation_mse=best_validation_mse,
                        best_epoch=best_epoch,
                    )
                    archive = f"hybrid_hebbian_{layer}_epoch_{epoch:03d}.pt"
                    append_metric(
                        run_dir,
                        {
                            "stage": "hybrid_hebbian",
                            "split": "train",
                            "layer": layer,
                            "checkpoint_id": Path(archive).stem,
                            "epoch": epoch,
                            "global_epoch": global_epoch,
                            "step": steps_completed,
                            "samples_seen": samples_seen,
                            "dataset_passes": epoch,
                            "wall_time_sec": wall_time,
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
                        },
                    )
                    save_epoch_checkpoint(
                        run_dir,
                        archive_name=archive,
                        config=config,
                        rule=RULE,
                        stage="hybrid_hebbian",
                        model=model,
                        optimizer=None,
                        progress=progress,
                        train_loader=loaders["train"],
                    )
                    update_run_status(
                        run_dir,
                        status="running",
                        **progress,
                        checkpoint=archive,
                    )
                    print(
                        f"method={config['hybrid']['method_id']} "
                        f"layer={layer} epoch={epoch:02d} "
                        f"update_norm={diagnostics.update_norm:.6f}",
                        flush=True,
                    )
                    if (
                        stop_after_global_epoch is not None
                        and global_epoch >= stop_after_global_epoch
                    ):
                        update_run_status(run_dir, status="paused")
                        return run_dir
                torch.save(
                    model.encoder.state_dict(),
                    run_dir / f"encoder_{layer}_end.pt",
                )
                resume_epoch = 0

        parameter_manifest, trainable_parameters = _parameter_manifest(
            model, layer_rules
        )
        manifest_path = run_dir / "trainable_frozen_parameter_manifest.json"
        if manifest_path.exists():
            existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if (
                existing["bp_trainable_parameter_names"]
                != parameter_manifest["bp_trainable_parameter_names"]
            ):
                raise RuntimeError("Resume parameter manifest changed")
            frozen_hashes = existing["frozen_layer_hashes_before_bp"]
        else:
            write_json(
                run_dir,
                "trainable_frozen_parameter_manifest.json",
                parameter_manifest,
            )
            frozen_hashes = parameter_manifest["frozen_layer_hashes_before_bp"]

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=float(config["backprop"]["lr"]),
            betas=tuple(config["backprop"]["betas"]),
            weight_decay=float(config["backprop"]["weight_decay"]),
        )
        first_bp_epoch = 1
        if payload is not None and resume_stage == "hybrid_bp_joint":
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            first_bp_epoch = resume_epoch + 1
        else:
            best_validation_mse = float("inf")
            best_epoch = 0

        bp_epochs = int(config["training"]["bp_epochs"])
        for epoch in range(first_bp_epoch, bp_epochs + 1):
            train_mse, train_samples = _train_bp_epoch(
                model, loaders["train"], optimizer, device
            )
            validation_mse, validation_samples = _evaluate_reconstruction(
                model, loaders["validation"], device
            )
            for layer, expected_hash in frozen_hashes.items():
                if state_dict_checksum(getattr(model.encoder, layer)) != expected_hash:
                    raise RuntimeError(f"Frozen layer changed during BP stage: {layer}")
            global_epoch += 1
            samples_seen += train_samples
            steps_completed += len(loaders["train"])
            wall_time = wall_offset + time.perf_counter() - started
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            torch.save(model.state_dict(), run_dir / "model_last.pt")
            progress = _progress(
                stage="hybrid_bp_joint",
                active_layer="joint",
                completed_epoch=epoch,
                global_epoch=global_epoch,
                samples_seen=samples_seen,
                steps_completed=steps_completed,
                wall_time_sec=wall_time,
                best_validation_mse=best_validation_mse,
                best_epoch=best_epoch,
            )
            archive = f"hybrid_bp_joint_epoch_{epoch:03d}.pt"
            for split, loss, count in (
                ("train", train_mse, train_samples),
                ("validation", validation_mse, validation_samples),
            ):
                append_metric(
                    run_dir,
                    {
                        "stage": "hybrid_bp_joint",
                        "split": split,
                        "layer": ",".join(
                            layer
                            for layer in LAYERS
                            if layer_rules[layer] == "bp"
                        )
                        or "decoder_only",
                        "checkpoint_id": Path(archive).stem,
                        "epoch": epoch,
                        "global_epoch": global_epoch,
                        "step": steps_completed,
                        "samples_seen": samples_seen,
                        "dataset_passes": epoch,
                        "wall_time_sec": wall_time,
                        "reconstruction_loss": loss,
                        "num_samples": count,
                    },
                )
            save_epoch_checkpoint(
                run_dir,
                archive_name=archive,
                config=config,
                rule=RULE,
                stage="hybrid_bp_joint",
                model=model,
                optimizer=optimizer,
                progress=progress,
                train_loader=loaders["train"],
            )
            update_run_status(
                run_dir,
                status="running",
                **progress,
                checkpoint=archive,
            )
            print(
                f"method={config['hybrid']['method_id']} epoch={epoch:02d} "
                f"train_mse={train_mse:.6f} validation_mse={validation_mse:.6f}",
                flush=True,
            )
            if (
                stop_after_global_epoch is not None
                and global_epoch >= stop_after_global_epoch
            ):
                update_run_status(run_dir, status="paused")
                return run_dir

        best_state = torch.load(
            run_dir / "model_best.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(best_state)
        frozen_after = {
            layer: state_dict_checksum(getattr(model.encoder, layer))
            for layer in frozen_hashes
        }
        if frozen_after != frozen_hashes:
            raise RuntimeError("Best checkpoint violates frozen-layer hashes")
        final_validation_mse, validation_samples = _evaluate_reconstruction(
            model, loaders["validation"], device
        )
        write_json(
            run_dir,
            "hybrid_training_summary.json",
            {
                "schema_version": "hybrid-training-summary-v1",
                "method_id": config["hybrid"]["method_id"],
                "encoder_layer_rules": layer_rules,
                "best_bp_epoch": best_epoch,
                "best_validation_reconstruction_mse": final_validation_mse,
                "validation_samples": validation_samples,
                "global_epochs": global_epoch,
                "samples_seen": samples_seen,
                "steps_completed": steps_completed,
                "wall_time_sec": wall_offset + time.perf_counter() - started,
                "initial_state_hash": read_metadata(run_dir)["initial_state_hash"],
                "initial_decoder_hash": read_metadata(run_dir)[
                    "initial_decoder_hash"
                ],
                "final_model_hash": state_dict_checksum(model),
                "final_encoder_hash": state_dict_checksum(model.encoder),
                "frozen_layer_hashes_before_bp": frozen_hashes,
                "frozen_layer_hashes_after_bp": frozen_after,
                "frozen_layers_unchanged": True,
                "test_samples_accessed": 0,
            },
            overwrite=True,
        )
        update_run_status(
            run_dir,
            status="completed",
            completed_at_utc=utc_now(),
            test_samples_accessed=0,
            error=None,
        )
        return run_dir
    except Exception as error:
        update_run_status(
            run_dir,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def train_hybrid(
    config_path: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    return train_hybrid_config(load_config(config_path), run_dir=run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(train_hybrid(args.config, run_dir=args.run_dir).resolve())


if __name__ == "__main__":
    main()
