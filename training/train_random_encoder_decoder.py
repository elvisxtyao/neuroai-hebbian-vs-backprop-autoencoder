"""Train only the decoder behind a paired, permanently random encoder."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from data.mnist import build_mnist_dataloaders
from evaluation.compare_reconstructions import (
    evaluate_model,
    reconstruct_selected,
    select_stratified_samples,
)
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
    read_run_status,
    update_run_status,
    write_json,
    write_metadata,
    write_resolved_config,
)


PROTOCOL = "random_frozen_encoder_bp_decoder"


@torch.no_grad()
def _validation_loss(model, loader, criterion, device) -> tuple[float, int]:
    model.eval()
    total = 0.0
    samples = 0
    for images, _, _ in loader:
        images = images.to(device)
        loss = criterion(model.decode(model.encode(images)), images)
        total += float(loss.item()) * images.shape[0]
        samples += images.shape[0]
    return total / samples, samples


def _load_model(path: Path, config: dict, device) -> ConvAutoencoder:
    model = ConvAutoencoder(config["model"]["latent_dim"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def _tensor_to_image(tensor: torch.Tensor, size: int = 112) -> Image.Image:
    array = np.uint8(tensor.detach().cpu().squeeze().clamp(0, 1).numpy() * 255)
    return Image.fromarray(array, mode="L").resize((size, size), Image.Resampling.NEAREST)


def _save_grid(
    path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
) -> None:
    columns = (
        ("Original", images),
        ("Untrained AE", reconstructions["untrained"]),
        ("Random enc + trained dec", reconstructions["random_encoder_trained_decoder"]),
        ("Hebbian enc", reconstructions["hebbian"]),
        ("BP AE", reconstructions["bp"]),
    )
    cell, gap, left, header = 112, 8, 128, 34
    canvas = Image.new(
        "L",
        (left + len(columns) * (cell + gap) + gap, header + len(images) * (cell + gap)),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (title, _) in enumerate(columns):
        draw.text((left + gap + column * (cell + gap), 10), title, fill=0, font=font)
    for row in range(len(images)):
        y = header + row * (cell + gap)
        draw.text(
            (8, y + cell // 2),
            f"digit={int(labels[row])} id={int(sample_ids[row])}",
            fill=0,
            font=font,
        )
        for column, (_, values) in enumerate(columns):
            canvas.paste(
                _tensor_to_image(values[row], cell),
                (left + gap + column * (cell + gap), y),
            )
    canvas.save(path)


def _analyze(
    run_dir: Path,
    config: dict,
    loaders,
    device,
    *,
    bp_run: Path | None,
    hebbian_run: Path | None,
    samples_per_class: int,
    sample_seed: int,
) -> dict[str, Any]:
    initial_payload = torch.load(
        run_dir / "checkpoints" / "initial_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    initial_model = ConvAutoencoder(config["model"]["latent_dim"])
    initial_model.load_state_dict(initial_payload["model_state_dict"])
    initial_model = initial_model.to(device).eval()
    models: dict[str, ConvAutoencoder] = {
        "untrained": initial_model,
        "random_encoder_trained_decoder": _load_model(
            run_dir / "model_best.pt", config, device
        ),
    }
    if bp_run is not None:
        models["bp"] = _load_model(bp_run / "model_best.pt", config, device)
    if hebbian_run is not None:
        models["hebbian"] = _load_model(
            hebbian_run / "model_best.pt", config, device
        )
    metrics = {
        name: evaluate_model(model, loaders["test"], device=device)
        for name, model in models.items()
    }
    before = metrics["untrained"]["mse"]
    decoder_only = metrics["random_encoder_trained_decoder"]["mse"]
    comparison = {
        "absolute_mse_reduction": before - decoder_only,
        "relative_mse_reduction": 1.0 - decoder_only / before,
        "decoder_substantially_compensates_random_encoder": decoder_only <= 0.5 * before,
    }
    checksums = {
        "initial_encoder": state_dict_checksum(initial_model.encoder),
        "trained_encoder": state_dict_checksum(
            models["random_encoder_trained_decoder"].encoder
        ),
        "initial_decoder": state_dict_checksum(initial_model.decoder),
        "trained_decoder": state_dict_checksum(
            models["random_encoder_trained_decoder"].decoder
        ),
    }
    if "hebbian" in metrics:
        comparison.update(
            {
                "mse_gap_vs_hebbian": decoder_only - metrics["hebbian"]["mse"],
                "mse_ratio_vs_hebbian": decoder_only / metrics["hebbian"]["mse"],
                "decoder_fully_matches_hebbian_encoder": decoder_only
                <= metrics["hebbian"]["mse"],
            }
        )
    if "bp" in metrics:
        comparison.update(
            {
                "mse_gap_vs_bp": decoder_only - metrics["bp"]["mse"],
                "mse_ratio_vs_bp": decoder_only / metrics["bp"]["mse"],
                "decoder_fully_matches_bp_encoder": decoder_only <= metrics["bp"]["mse"],
            }
        )

    for name in ("untrained", "random_encoder_trained_decoder"):
        append_metric(
            run_dir,
            {
                "stage": "random_encoder_reconstruction_final",
                "split": "test",
                "layer": name,
                "reconstruction_loss": metrics[name]["mse"],
                "num_samples": metrics[name]["num_samples"],
            },
        )

    if samples_per_class > 0 and {"bp", "hebbian"}.issubset(models):
        images, labels, sample_ids = select_stratified_samples(
            loaders["test"].dataset,
            samples_per_class=samples_per_class,
            seed=sample_seed,
        )
        selected = {
            name: reconstruct_selected(model, images, device) for name, model in models.items()
        }
        _save_grid(run_dir / "decoder_compensation_grid.png", images, labels, sample_ids, selected)
        write_json(
            run_dir,
            "decoder_compensation_samples.json",
            {
                "sample_seed": sample_seed,
                "samples_per_class": samples_per_class,
                "sample_ids": sample_ids.tolist(),
                "labels": labels.tolist(),
            },
            overwrite=True,
        )
    return {"metrics": metrics, "comparison": comparison, "checksums": checksums}


def train_random_decoder_config(
    config: dict[str, Any],
    *,
    loaders=None,
    run_root: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
    bp_run: str | Path | None = None,
    hebbian_run: str | Path | None = None,
    samples_per_class: int = 2,
    sample_seed: int = 23,
    stop_after_epoch: int | None = None,
) -> Path:
    validate_config(config)
    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaders is None:
        loaders = build_mnist_dataloaders(config, seed=seed, download=False)
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=config["backprop"]["lr"],
        betas=tuple(config["backprop"]["betas"]),
        weight_decay=config["backprop"]["weight_decay"],
    )
    criterion = nn.MSELoss(reduction="mean")

    if resume_run_dir is None:
        root = config["results"]["root"] if run_root is None else run_root
        run_dir = create_run_directory(root, rule="random_decoder", seed=seed)
        write_resolved_config(run_dir, config)
        manifest = Path(config["data"]["split_manifest"])
        write_metadata(
            run_dir,
            {
                "version": config["version"],
                "experiment_id": f"{config['version']}_mnist_{config['model']['architecture']}",
                "run_id": run_dir.name,
                "created_at_utc": utc_now(),
                "config_sha256": config_fingerprint(config),
                "split_manifest_sha256": file_sha256(manifest),
                "learning_rule": PROTOCOL,
                "model_type": "convolutional_autoencoder_control",
                "architecture_id": config["model"]["architecture"],
                "latent_dim": config["model"]["latent_dim"],
                "seed": seed,
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "initial_state_hash": state_dict_checksum(model),
                "initial_encoder_hash": state_dict_checksum(model.encoder),
                "initial_decoder_hash": state_dict_checksum(model.decoder),
                **git_provenance(str(Path.cwd())),
                **model.architecture_metadata(),
            },
        )
        initial_status = {
            "status": "running",
            "stage": "random_encoder_decoder",
            "active_layer": "decoder",
            "completed_epoch": 0,
            "global_epoch": 0,
            "samples_seen": 0,
            "steps_completed": 0,
            "wall_time_sec": 0.0,
            "resume_count": 0,
            "checkpoint": "initial_state.pt",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "error": None,
        }
        initialize_run_status(run_dir, initial_status)
        progress = {
            "stage": "random_encoder_decoder",
            "active_layer": "decoder",
            "completed_epoch": 0,
            "global_epoch": 0,
            "samples_seen": 0,
            "steps_completed": 0,
            "wall_time_sec": 0.0,
            "best_validation_loss": float("inf"),
            "best_epoch": 0,
            "encoder_hash": state_dict_checksum(model.encoder),
        }
        save_epoch_checkpoint(
            run_dir,
            archive_name="initial_state.pt",
            config=config,
            rule=PROTOCOL,
            stage="random_encoder_decoder",
            model=model,
            optimizer=optimizer,
            progress=progress,
            train_loader=loaders["train"],
        )
    else:
        run_dir = Path(resume_run_dir)
        status = read_run_status(run_dir)
        if status["status"] == "completed":
            return run_dir
        payload = load_resume_checkpoint(run_dir, config, rule=PROTOCOL)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        progress = payload["progress"]
        restore_rng_state(payload["rng_state"])
        restore_loader_generator(loaders["train"], payload["train_loader_generator_state"])
        update_run_status(
            run_dir,
            status="running",
            resume_count=int(status.get("resume_count", 0)) + 1,
            error=None,
        )

    encoder_hash = progress["encoder_hash"]
    completed_epoch = int(progress["completed_epoch"])
    samples_seen = int(progress["samples_seen"])
    steps_completed = int(progress["steps_completed"])
    wall_offset = float(progress["wall_time_sec"])
    best_validation_loss = float(progress["best_validation_loss"])
    best_epoch = int(progress["best_epoch"])
    started = time.perf_counter()
    try:
        for epoch in range(completed_epoch + 1, config["training"]["decoder_epochs"] + 1):
            model.encoder.eval()
            model.decoder.train()
            total = 0.0
            samples = 0
            for images, _, _ in loaders["train"]:
                images = images.to(device)
                with torch.no_grad():
                    latent = model.encode(images)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model.decode(latent), images)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().item()) * images.shape[0]
                samples += images.shape[0]
            train_loss = total / samples
            validation_loss, validation_samples = _validation_loss(
                model, loaders["validation"], criterion, device
            )
            samples_seen += samples
            steps_completed += len(loaders["train"])
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            torch.save(model.state_dict(), run_dir / "model_last.pt")
            progress = {
                "stage": "random_encoder_decoder",
                "active_layer": "decoder",
                "completed_epoch": epoch,
                "global_epoch": epoch,
                "samples_seen": samples_seen,
                "steps_completed": steps_completed,
                "wall_time_sec": wall_offset + time.perf_counter() - started,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "encoder_hash": encoder_hash,
            }
            archive_name = f"random_decoder_epoch_{epoch:03d}.pt"
            for split, value, count in (
                ("train", train_loss, samples),
                ("validation", validation_loss, validation_samples),
            ):
                append_metric(
                    run_dir,
                    {
                        "stage": "random_encoder_decoder",
                        "split": split,
                        "layer": "decoder",
                        "checkpoint_id": Path(archive_name).stem,
                        "epoch": epoch,
                        "global_epoch": epoch,
                        "step": steps_completed,
                        "samples_seen": samples_seen,
                        "dataset_passes": epoch,
                        "wall_time_sec": progress["wall_time_sec"],
                        "reconstruction_loss": value,
                        "num_samples": count,
                    },
                )
            save_epoch_checkpoint(
                run_dir,
                archive_name=archive_name,
                config=config,
                rule=PROTOCOL,
                stage="random_encoder_decoder",
                model=model,
                optimizer=optimizer,
                progress=progress,
                train_loader=loaders["train"],
            )
            update_run_status(
                run_dir,
                status="running",
                completed_epoch=epoch,
                global_epoch=epoch,
                samples_seen=samples_seen,
                steps_completed=steps_completed,
                wall_time_sec=progress["wall_time_sec"],
                checkpoint=archive_name,
            )
            print(
                f"epoch={epoch:02d} train_mse={train_loss:.8f} "
                f"validation_mse={validation_loss:.8f}",
                flush=True,
            )
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                update_run_status(run_dir, status="paused", checkpoint=archive_name)
                return run_dir
        if state_dict_checksum(model.encoder) != encoder_hash:
            raise RuntimeError("Random encoder changed while training decoder")
        analysis = _analyze(
            run_dir,
            config,
            loaders,
            device,
            bp_run=None if bp_run is None else Path(bp_run),
            hebbian_run=None if hebbian_run is None else Path(hebbian_run),
            samples_per_class=samples_per_class,
            sample_seed=sample_seed,
        )
        write_json(
            run_dir,
            "random_encoder_decoder_summary.json",
            {
                "protocol": PROTOCOL,
                "encoder_trained": False,
                "decoder_trained_with_backprop": True,
                "encoder_hash_before": encoder_hash,
                "encoder_hash_after": state_dict_checksum(model.encoder),
                "encoder_unchanged": state_dict_checksum(model.encoder) == encoder_hash,
                "best_epoch": best_epoch,
                "best_validation_mse": best_validation_loss,
                **analysis,
            },
            overwrite=True,
        )
        update_run_status(
            run_dir,
            status="completed",
            completed_at_utc=utc_now(),
            error=None,
        )
    except Exception as error:
        update_run_status(
            run_dir,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    return run_dir


def train_random_decoder(
    config_path: str | Path | None,
    *,
    resume_run_dir: str | Path | None = None,
    bp_run: str | Path | None = None,
    hebbian_run: str | Path | None = None,
    samples_per_class: int = 2,
    stop_after_epoch: int | None = None,
) -> Path:
    if resume_run_dir is not None:
        with (Path(resume_run_dir) / "config_resolved.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            config = yaml.safe_load(handle)
        if config_path is not None and config_fingerprint(load_config(config_path)) != config_fingerprint(config):
            raise RuntimeError("Provided config differs from the baseline run config")
    elif config_path is not None:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or resume_run_dir is required")
    return train_random_decoder_config(
        config,
        resume_run_dir=resume_run_dir,
        bp_run=bp_run,
        hebbian_run=hebbian_run,
        samples_per_class=samples_per_class,
        stop_after_epoch=stop_after_epoch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--bp-run")
    parser.add_argument("--hebbian-run")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if not args.config and not args.resume_run_dir:
        parser.error("one of --config or --resume-run-dir is required")
    print(
        train_random_decoder(
            args.config,
            resume_run_dir=args.resume_run_dir,
            bp_run=args.bp_run,
            hebbian_run=args.hebbian_run,
            samples_per_class=args.samples_per_class,
            stop_after_epoch=args.stop_after_epoch,
        ).resolve()
    )


if __name__ == "__main__":
    main()
