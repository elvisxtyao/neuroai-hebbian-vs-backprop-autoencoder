"""Resumable paired decoder training with the complete encoder frozen."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn

from data.mnist import build_mnist_dataloaders
from models import ConvAutoencoder, autoencoder_from_config
from schemas import load_config, validate_config
from utils.checkpointing import (
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
    read_run_status,
    update_run_status,
    write_json,
    write_metadata,
    write_resolved_config,
)


RULE = "standardized_decoder"


def _evaluate(model: ConvAutoencoder, loader, device: torch.device) -> tuple[float, int]:
    model.eval()
    total_squared_error = 0.0
    total_pixels = 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            reconstruction = model.decoder(model.encoder(images))
            total_squared_error += float(
                (reconstruction - images).square().sum().item()
            )
            total_pixels += images.numel()
    return total_squared_error / total_pixels, total_pixels


def _train_epoch(
    model: ConvAutoencoder,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, int, int]:
    model.encoder.eval()
    model.decoder.train()
    criterion = nn.MSELoss(reduction="mean")
    total = 0.0
    samples = 0
    steps = 0
    for images, _, _ in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            latent = model.encoder(images)
        reconstruction = model.decoder(latent)
        loss = criterion(reconstruction, images)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item()) * images.shape[0]
        samples += images.shape[0]
        steps += 1
    return total / samples, samples, steps


def _progress(
    *,
    epoch: int,
    samples_seen: int,
    steps_completed: int,
    wall_time_sec: float,
    best_validation_mse: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "stage": RULE,
        "active_layer": "decoder",
        "completed_epoch": epoch,
        "global_epoch": epoch,
        "samples_seen": samples_seen,
        "steps_completed": steps_completed,
        "wall_time_sec": wall_time_sec,
        "best_validation_mse": best_validation_mse,
        "best_epoch": best_epoch,
    }


def train_standardized_decoder_config(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    loaders=None,
    resume: bool = True,
    stop_after_epoch: int | None = None,
) -> Path:
    """Train a fresh paired decoder while preserving the completed encoder."""

    validate_config(config)
    decoder_config = config.get("standardized_decoder")
    if not isinstance(decoder_config, dict):
        raise ValueError("Missing standardized_decoder config")
    if decoder_config.get("optimizer") != "adam":
        raise ValueError("Standardized decoder optimizer must be Adam")
    if decoder_config.get("loss") != "mse_pixel_mean":
        raise ValueError("Standardized decoder loss must be mse_pixel_mean")
    if decoder_config.get("validation_selection") != "min_reconstruction_mse":
        raise ValueError("Standardized decoder selection must use validation MSE")

    run_dir = Path(run_dir)
    source_checkpoint = run_dir / "model_best.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Missing completed encoder checkpoint: {source_checkpoint}")
    standard_dir = run_dir / "standardized_decoder"
    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaders is None:
        loaders = build_mnist_dataloaders(config, seed=seed, include_test=False)
    if "test" in loaders:
        raise RuntimeError("Standardized decoder must not construct a test loader")

    paired_model = autoencoder_from_config(config, seed=seed)
    source_state = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True
    )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source_state.items()
        if key.startswith("encoder.")
    }
    paired_model.encoder.load_state_dict(encoder_state)
    for parameter in paired_model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in paired_model.decoder.parameters():
        parameter.requires_grad_(True)
    paired_model.to(device)
    encoder_hash_before = state_dict_checksum(paired_model.encoder)
    decoder_initial_hash = state_dict_checksum(paired_model.decoder)
    optimizer = torch.optim.Adam(
        paired_model.decoder.parameters(),
        lr=float(decoder_config["lr"]),
        betas=tuple(float(value) for value in decoder_config["betas"]),
        weight_decay=float(decoder_config["weight_decay"]),
    )

    payload: dict[str, Any] | None = None
    if standard_dir.exists():
        if not resume:
            raise FileExistsError(f"Standardized decoder run exists: {standard_dir}")
        status = read_run_status(standard_dir)
        if status["status"] == "completed":
            return standard_dir
        payload = load_resume_checkpoint(standard_dir, config, rule=RULE)
        paired_model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        restore_rng_state(payload["rng_state"])
        restore_loader_generator(
            loaders["train"], payload["train_loader_generator_state"]
        )
        update_run_status(
            standard_dir,
            status="running",
            resume_count=int(status.get("resume_count", 0)) + 1,
            error=None,
        )
    else:
        standard_dir.mkdir(parents=True)
        write_resolved_config(standard_dir, config)
        write_metadata(
            standard_dir,
            {
                "schema_version": "standardized-decoder-metadata-v1",
                "experiment_id": "hybrid_hhb_confirmation",
                "run_id": f"{run_dir.name}_standardized_decoder",
                "model_type": "autoencoder",
                "learning_rule": RULE,
                "architecture_id": config["model"]["architecture"],
                "latent_dim": int(config["model"]["latent_dim"]),
                "seed": seed,
                "method_id": config["hybrid"]["method_id"],
                "source_run_dir": str(run_dir),
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": file_sha256(source_checkpoint),
                "encoder_hash_before": encoder_hash_before,
                "decoder_initial_hash": decoder_initial_hash,
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "test_samples_accessed": 0,
                **git_provenance(str(Path.cwd())),
            },
        )
        initialize_run_status(
            standard_dir,
            {
                "status": "running",
                "stage": RULE,
                "active_layer": "decoder",
                "completed_epoch": 0,
                "global_epoch": 0,
                "samples_seen": 0,
                "steps_completed": 0,
                "wall_time_sec": 0.0,
                "best_validation_mse": float("inf"),
                "best_epoch": 0,
                "resume_count": 0,
                "checkpoint": None,
                "test_samples_accessed": 0,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "error": None,
            },
        )
        initial_progress = _progress(
            epoch=0,
            samples_seen=0,
            steps_completed=0,
            wall_time_sec=0.0,
            best_validation_mse=float("inf"),
            best_epoch=0,
        )
        save_epoch_checkpoint(
            standard_dir,
            archive_name="initial_state.pt",
            config=config,
            rule=RULE,
            stage=RULE,
            model=paired_model,
            optimizer=optimizer,
            progress=initial_progress,
            train_loader=loaders["train"],
        )
        update_run_status(
            standard_dir,
            **initial_progress,
            checkpoint="initial_state.pt",
        )

    if payload is None:
        completed_epoch = 0
        samples_seen = 0
        steps_completed = 0
        wall_offset = 0.0
        best_validation_mse = float("inf")
        best_epoch = 0
    else:
        previous = payload["progress"]
        completed_epoch = int(previous["completed_epoch"])
        samples_seen = int(previous["samples_seen"])
        steps_completed = int(previous["steps_completed"])
        wall_offset = float(previous["wall_time_sec"])
        best_validation_mse = float(previous["best_validation_mse"])
        best_epoch = int(previous["best_epoch"])

    started = time.perf_counter()
    try:
        for epoch in range(
            completed_epoch + 1, int(decoder_config["epochs"]) + 1
        ):
            train_mse, train_samples, train_steps = _train_epoch(
                paired_model, loaders["train"], optimizer, device
            )
            validation_mse, validation_pixels = _evaluate(
                paired_model, loaders["validation"], device
            )
            if state_dict_checksum(paired_model.encoder) != encoder_hash_before:
                raise RuntimeError("Standardized decoder mutated the encoder")
            samples_seen += train_samples
            steps_completed += train_steps
            wall_time = wall_offset + time.perf_counter() - started
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_epoch = epoch
                torch.save(
                    paired_model.decoder.state_dict(),
                    standard_dir / "decoder_best.pt",
                )
            torch.save(
                paired_model.decoder.state_dict(),
                standard_dir / "decoder_last.pt",
            )
            append_metric(
                standard_dir,
                {
                    "stage": RULE,
                    "split": "train",
                    "layer": "decoder",
                    "checkpoint_id": f"standardized_decoder_epoch_{epoch:03d}",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "step": steps_completed,
                    "samples_seen": samples_seen,
                    "dataset_passes": epoch,
                    "wall_time_sec": wall_time,
                    "reconstruction_loss": train_mse,
                    "num_samples": train_samples,
                },
            )
            append_metric(
                standard_dir,
                {
                    "stage": RULE,
                    "split": "validation",
                    "layer": "decoder",
                    "checkpoint_id": f"standardized_decoder_epoch_{epoch:03d}",
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "step": steps_completed,
                    "samples_seen": samples_seen,
                    "dataset_passes": epoch,
                    "wall_time_sec": wall_time,
                    "reconstruction_loss": validation_mse,
                    "num_samples": validation_pixels,
                },
            )
            progress = _progress(
                epoch=epoch,
                samples_seen=samples_seen,
                steps_completed=steps_completed,
                wall_time_sec=wall_time,
                best_validation_mse=best_validation_mse,
                best_epoch=best_epoch,
            )
            archive = f"standardized_decoder_epoch_{epoch:03d}.pt"
            save_epoch_checkpoint(
                standard_dir,
                archive_name=archive,
                config=config,
                rule=RULE,
                stage=RULE,
                model=paired_model,
                optimizer=optimizer,
                progress=progress,
                train_loader=loaders["train"],
            )
            update_run_status(
                standard_dir,
                status="running",
                **progress,
                checkpoint=archive,
            )
            print(
                f"method={config['hybrid']['method_id']} seed={seed} "
                f"standardized_decoder epoch={epoch:02d} "
                f"train_mse={train_mse:.6f} "
                f"validation_mse={validation_mse:.6f}",
                flush=True,
            )
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                update_run_status(standard_dir, status="paused")
                return standard_dir

        best_path = standard_dir / "decoder_best.pt"
        if not best_path.exists():
            raise RuntimeError("Standardized decoder produced no best checkpoint")
        paired_model.decoder.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True)
        )
        final_validation_mse, validation_pixels = _evaluate(
            paired_model, loaders["validation"], device
        )
        encoder_hash_after = state_dict_checksum(paired_model.encoder)
        if encoder_hash_before != encoder_hash_after:
            raise RuntimeError("Standardized decoder final encoder checksum changed")
        write_json(
            standard_dir,
            "standardized_decoder_summary.json",
            {
                "schema_version": "standardized-decoder-summary-v1",
                "method_id": config["hybrid"]["method_id"],
                "seed": seed,
                "optimizer": "adam",
                "learning_rate": float(decoder_config["lr"]),
                "betas": [float(value) for value in decoder_config["betas"]],
                "weight_decay": float(decoder_config["weight_decay"]),
                "epochs": int(decoder_config["epochs"]),
                "selection": decoder_config["validation_selection"],
                "best_epoch": best_epoch,
                "best_validation_reconstruction_mse": final_validation_mse,
                "validation_pixels": validation_pixels,
                "samples_seen": samples_seen,
                "steps_completed": steps_completed,
                "wall_time_sec": wall_offset + time.perf_counter() - started,
                "decoder_initial_hash": decoder_initial_hash,
                "decoder_best_hash": state_dict_checksum(paired_model.decoder),
                "decoder_checkpoint_sha256": file_sha256(best_path),
                "encoder_hash_before": encoder_hash_before,
                "encoder_hash_after": encoder_hash_after,
                "encoder_unchanged": True,
                "test_samples_accessed": 0,
            },
            overwrite=True,
        )
        update_run_status(
            standard_dir,
            status="completed",
            completed_at_utc=utc_now(),
            test_samples_accessed=0,
            error=None,
        )
        return standard_dir
    except Exception as error:
        update_run_status(
            standard_dir,
            status="failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        train_standardized_decoder_config(
            config,
            run_dir=args.run_dir,
        ).resolve()
    )


if __name__ == "__main__":
    main()
