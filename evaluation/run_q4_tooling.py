"""Run the seed-42 frozen-snapshot Q4 update-analysis tooling gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.update_analysis import (
    bp_raw_negative_gradient,
    cosine_alignment,
    hebbian_candidate_deltas,
    norm_ratio,
    prepare_fixed_batch_manifest,
    raw_relative_difference,
    scale_matched_metrics,
    snapshot_integrity_gate,
    update_snr,
)
from learning_rules.hebbian import CompetitiveOjaConv2d
from models import ConvAutoencoder
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import (
    git_provenance,
    set_global_seed,
    state_dict_checksum,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    temporary.replace(path)


def _record_path(path: Path) -> str:
    """Prefer a repository-relative path but support isolated clean worktrees."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_is_ancestor(ancestor: str) -> bool:
    import subprocess

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _config_sha256(path: Path) -> str:
    return file_sha256(path)


def _source_run_dir(source: dict[str, Any]) -> Path:
    if source.get("run_dir"):
        return _resolve(source["run_dir"])
    decision_path = _resolve(source["selection_decision"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    trial_id = source["trial_id"]
    matches = [
        trial
        for trial in decision.get("trials", [])
        if trial.get("trial_id") == trial_id
    ]
    if len(matches) != 1 or not matches[0].get("run_dir"):
        raise RuntimeError(
            f"Could not resolve Q4 source trial {trial_id!r} from {decision_path}"
        )
    return Path(matches[0]["run_dir"]).resolve()


def _mnist_training_dataset(data_root: Path):
    return datasets.MNIST(
        root=str(data_root),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )


def _decoder_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    split = np.load(_resolve(config["data"]["split_manifest"]), allow_pickle=False)
    dataset = _mnist_training_dataset(_resolve(config["data"]["root"]))
    generator = torch.Generator().manual_seed(
        int(config["reference_decoder"]["loader_seed"])
    )
    common = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": 0,
        "drop_last": False,
    }
    train_loader = DataLoader(
        IndexedDataset(dataset, split["train_indices"]),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        IndexedDataset(dataset, split["validation_indices"]),
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


@torch.no_grad()
def _reconstruction_mse(model, loader: DataLoader) -> float:
    model.eval()
    total_squared_error = 0.0
    total_pixels = 0
    for images, _, _ in loader:
        reconstruction = model(images)
        total_squared_error += float(
            (reconstruction - images).square().sum().item()
        )
        total_pixels += images.numel()
    return total_squared_error / total_pixels


def _train_reference_decoder(
    config: dict[str, Any],
    *,
    snapshot_id: str,
    encoder_state: dict[str, torch.Tensor],
    output_dir: Path,
) -> dict[str, Any]:
    """Train one paired-initialization decoder while the snapshot stays frozen."""

    decoder_dir = output_dir / "reference_decoders" / snapshot_id
    summary_path = decoder_dir / "summary.json"
    checkpoint_path = decoder_dir / "decoder_best.pt"
    if summary_path.exists() and checkpoint_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if decoder_dir.exists() and any(decoder_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete reference decoder directory requires inspection: {decoder_dir}"
        )
    decoder_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["reference_decoder"]["initialization_seed"])
    set_global_seed(seed)
    model = ConvAutoencoder(latent_dim=64, seed=seed)
    model.encoder.load_state_dict(encoder_state)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    encoder_hash_before = state_dict_checksum(model.encoder)
    decoder_initial_hash = state_dict_checksum(model.decoder)
    train_loader, validation_loader = _decoder_loaders(config)
    decoder_config = config["reference_decoder"]
    if decoder_config["optimizer"] != "adam":
        raise ValueError("Reference decoder optimizer must be Adam")
    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=float(decoder_config["lr"]),
        betas=tuple(float(value) for value in decoder_config["betas"]),
        weight_decay=float(decoder_config["weight_decay"]),
    )
    criterion = nn.MSELoss(reduction="mean")
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    rows: list[dict[str, Any]] = []
    optimizer_steps = 0
    samples_seen = 0
    started = time.perf_counter()
    for epoch in range(1, int(decoder_config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = model.encoder(images)
            reconstruction = model.decoder(z)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            samples_seen += images.shape[0]
            total_loss += float(loss.detach().item()) * images.shape[0]
            total_samples += images.shape[0]
        validation_mse = _reconstruction_mse(model, validation_loader)
        train_mse = total_loss / total_samples
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "optimizer_steps": optimizer_steps,
                "samples_seen": samples_seen,
                "wall_time_sec": time.perf_counter() - started,
            }
        )
        print(
            f"Q4 decoder {snapshot_id} epoch={epoch:02d} "
            f"train_mse={train_mse:.6f} validation_mse={validation_mse:.6f}",
            flush=True,
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = deepcopy(model.decoder.state_dict())
    if best_state is None:
        raise RuntimeError("Reference decoder produced no checkpoint")
    model.decoder.load_state_dict(best_state)
    encoder_hash_after = state_dict_checksum(model.encoder)
    if encoder_hash_before != encoder_hash_after:
        raise RuntimeError("Reference decoder training mutated the encoder snapshot")
    torch.save(model.decoder.state_dict(), checkpoint_path)
    _write_csv(decoder_dir / "metrics.csv", rows)
    summary = {
        "schema_version": "q4-reference-decoder-v1",
        "snapshot_id": snapshot_id,
        "initialization_seed": seed,
        "loader_seed": int(decoder_config["loader_seed"]),
        "epochs": int(decoder_config["epochs"]),
        "optimizer": "adam",
        "learning_rate": float(decoder_config["lr"]),
        "betas": [float(value) for value in decoder_config["betas"]],
        "weight_decay": float(decoder_config["weight_decay"]),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "decoder_initial_state_hash": decoder_initial_hash,
        "decoder_best_state_hash": state_dict_checksum(model.decoder),
        "decoder_checkpoint": _record_path(checkpoint_path),
        "decoder_checkpoint_sha256": file_sha256(checkpoint_path),
        "encoder_hash_before": encoder_hash_before,
        "encoder_hash_after": encoder_hash_after,
        "encoder_unchanged": True,
        "optimizer_steps_for_decoder_training": optimizer_steps,
        "analysis_optimizer_steps": 0,
        "test_samples_accessed": 0,
    }
    _atomic_json(summary_path, summary)
    return summary


def _analysis_batches(
    config: dict[str, Any],
) -> tuple[np.ndarray, DataLoader, str]:
    manifest_path = _resolve(config["data"]["batch_manifest"])
    manifest = np.load(manifest_path, allow_pickle=False)
    batch_ids = np.asarray(manifest["batch_ids"], dtype=np.int64)
    expected_shape = (
        int(config["data"]["batch_count"]),
        int(config["data"]["batch_size"]),
    )
    if batch_ids.shape != expected_shape:
        raise RuntimeError(
            f"Q4 batch manifest shape {batch_ids.shape} != {expected_shape}"
        )
    if np.unique(batch_ids).size != batch_ids.size:
        raise RuntimeError("Q4 fixed update batches contain duplicate sample IDs")
    dataset = _mnist_training_dataset(_resolve(config["data"]["root"]))
    loader = DataLoader(
        IndexedDataset(dataset, batch_ids.reshape(-1)),
        batch_size=expected_shape[1],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    ids_hash = hashlib.sha256(batch_ids.tobytes()).hexdigest()
    if ids_hash != str(manifest["batch_ids_sha256"].item()):
        raise RuntimeError("Q4 batch IDs hash mismatch")
    return batch_ids, loader, ids_hash


def _hebbian_rule(run_config: dict[str, Any], layer_name: str):
    hebbian = run_config["hebbian"]
    return CompetitiveOjaConv2d(
        learning_rate=float(
            hebbian.get("layer_lrs", {}).get(layer_name, hebbian["lr"])
        ),
        winner_fraction=float(hebbian["winner_fraction"]),
        normalization_epsilon=float(hebbian["normalization_epsilon"]),
        competition_mode=hebbian.get("competition_mode", "raw"),
        competition_power=float(hebbian.get("competition_power", 1.0)),
        competition_epsilon=float(hebbian.get("competition_epsilon", 1e-6)),
        center_inputs=bool(hebbian.get("center_inputs", False)),
        update_centering=hebbian.get("update_centering", "none"),
    )


def _descriptive(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def _analyze_pair(
    config: dict[str, Any],
    *,
    snapshot_spec: dict[str, Any],
    encoder_state: dict[str, torch.Tensor],
    decoder_summary: dict[str, Any],
    batch_ids: np.ndarray,
    batch_loader: DataLoader,
    output_dir: Path,
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    snapshot_id = snapshot_spec["id"]
    layer_name = snapshot_spec["active_layer"]
    pair_dir = output_dir / "pairs" / f"{snapshot_id}__{layer_name}"
    summary_path = pair_dir / "summary.json"
    records_path = pair_dir / "batch_metrics.csv"
    tensor_path = pair_dir / "update_tensors.npz"
    if summary_path.exists() and records_path.exists() and tensor_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return summary, list(csv.DictReader(records_path.open(encoding="utf-8"))), {
            "snapshot_id": snapshot_id,
            "layer": layer_name,
            "path": _record_path(tensor_path),
            "sha256": file_sha256(tensor_path),
        }
    if pair_dir.exists() and any(pair_dir.iterdir()):
        raise RuntimeError(f"Incomplete Q4 pair directory requires inspection: {pair_dir}")
    pair_dir.mkdir(parents=True, exist_ok=True)

    model = ConvAutoencoder(
        latent_dim=int(run_config["model"]["latent_dim"]),
        seed=int(config["source"]["seed"]),
    )
    model.encoder.load_state_dict(encoder_state)
    model.decoder.load_state_dict(
        torch.load(
            _resolve(decoder_summary["decoder_checkpoint"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    model_hash_before = state_dict_checksum(model)
    encoder_hash_before = state_dict_checksum(model.encoder)
    decoder_hash_before = state_dict_checksum(model.decoder)
    rule = _hebbian_rule(run_config, layer_name)
    epsilon = float(config["analysis"]["epsilon"])
    bp_updates: list[torch.Tensor] = []
    raw_updates: list[torch.Tensor] = []
    effective_updates: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for batch_index, (images, _, observed_ids) in enumerate(batch_loader):
        expected_ids = torch.as_tensor(batch_ids[batch_index], dtype=observed_ids.dtype)
        if not torch.equal(observed_ids, expected_ids):
            raise RuntimeError(f"Batch IDs differ at Q4 batch {batch_index}")
        batch_hash = hashlib.sha256(
            np.asarray(batch_ids[batch_index], dtype=np.int64).tobytes()
        ).hexdigest()
        raw_delta, effective_delta, diagnostics = hebbian_candidate_deltas(
            model,
            images,
            layer_name=layer_name,
            rule=rule,
        )
        bp_delta, reconstruction_mse = bp_raw_negative_gradient(
            model,
            images,
            layer_name=layer_name,
        )
        raw_delta = raw_delta.detach().cpu()
        effective_delta = effective_delta.detach().cpu()
        bp_delta = bp_delta.detach().cpu()
        bp_updates.append(bp_delta)
        raw_updates.append(raw_delta)
        effective_updates.append(effective_delta)
        row = {
            "snapshot_id": snapshot_id,
            "layer": layer_name,
            "batch_index": batch_index,
            "batch_sample_ids_sha256": batch_hash,
            "sample_count": int(images.shape[0]),
            "bp_reference_reconstruction_mse": reconstruction_mse,
            "bp_raw_norm": float(bp_delta.norm().item()),
            "hebbian_raw_norm": float(raw_delta.norm().item()),
            "hebbian_effective_norm": float(effective_delta.norm().item()),
            "raw_alignment": cosine_alignment(
                raw_delta, bp_delta, epsilon=epsilon
            ),
            "effective_alignment": cosine_alignment(
                effective_delta, bp_delta, epsilon=epsilon
            ),
            "raw_norm_ratio": norm_ratio(raw_delta, bp_delta, epsilon=epsilon),
            "effective_norm_ratio": norm_ratio(
                effective_delta, bp_delta, epsilon=epsilon
            ),
            "raw_relative_difference": raw_relative_difference(
                raw_delta, bp_delta, epsilon=epsilon
            ),
            "effective_relative_difference": raw_relative_difference(
                effective_delta, bp_delta, epsilon=epsilon
            ),
            **diagnostics,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key
            not in {
                "snapshot_id",
                "layer",
                "batch_sample_ids_sha256",
            }
        ):
            raise RuntimeError(f"Non-finite Q4 batch metric: {snapshot_id} {batch_index}")
        rows.append(row)
        print(
            f"Q4 {snapshot_id}/{layer_name} batch={batch_index + 1:02d}/50 "
            f"raw_cos={row['raw_alignment']:.4f} "
            f"effective_cos={row['effective_alignment']:.4f}",
            flush=True,
        )

    bp_stack = torch.stack(bp_updates)
    raw_stack = torch.stack(raw_updates)
    effective_stack = torch.stack(effective_updates)
    if bool(config["analysis"]["save_update_tensors"]):
        np.savez_compressed(
            tensor_path,
            bp_raw=bp_stack.numpy(),
            hebbian_raw=raw_stack.numpy(),
            hebbian_effective=effective_stack.numpy(),
            batch_ids=batch_ids,
            snapshot_id=np.asarray(snapshot_id),
            layer=np.asarray(layer_name),
        )
    raw_scale = scale_matched_metrics(raw_stack, bp_stack, epsilon=epsilon)
    effective_scale = scale_matched_metrics(
        effective_stack, bp_stack, epsilon=epsilon
    )
    summary = {
        "schema_version": "q4-update-pair-summary-v1",
        "snapshot_id": snapshot_id,
        "layer": layer_name,
        "snapshot_role": config["source"]["snapshot_role"],
        "batch_count": len(rows),
        "batch_size": int(config["data"]["batch_size"]),
        "parameter_count": int(bp_stack[0].numel()),
        "weight_shape": list(bp_stack.shape[1:]),
        "update_centering": rule.update_centering,
        "bp_reference": {
            "direction": "raw reconstruction negative gradient",
            "optimizer_state_included": False,
            "weight_decay_included": False,
            "momentum_included": False,
            "optimizer_steps": 0,
            "snr": update_snr(bp_stack, epsilon=epsilon),
        },
        "hebbian_raw": {
            "definition": (
                "unscaled configured competitive Oja candidate before apply"
            ),
            **raw_scale,
            "snr": update_snr(raw_stack, epsilon=epsilon),
            **_descriptive(
                [float(row["raw_alignment"]) for row in rows],
                "batch_alignment",
            ),
            **_descriptive(
                [float(row["raw_norm_ratio"]) for row in rows],
                "batch_norm_ratio",
            ),
        },
        "hebbian_effective": {
            "definition": (
                "exact weight displacement after learning-rate scaling and "
                "per-filter L2 normalization"
            ),
            **effective_scale,
            "snr": update_snr(effective_stack, epsilon=epsilon),
            **_descriptive(
                [float(row["effective_alignment"]) for row in rows],
                "batch_alignment",
            ),
            **_descriptive(
                [float(row["effective_norm_ratio"]) for row in rows],
                "batch_norm_ratio",
            ),
        },
        "reconstruction_mse": _descriptive(
            [float(row["bp_reference_reconstruction_mse"]) for row in rows],
            "batch",
        ),
        "model_state_hash_before": model_hash_before,
        "model_state_hash_after": state_dict_checksum(model),
        "encoder_hash_before": encoder_hash_before,
        "encoder_hash_after": state_dict_checksum(model.encoder),
        "decoder_hash_before": decoder_hash_before,
        "decoder_hash_after": state_dict_checksum(model.decoder),
        "all_parameter_grads_none": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "analysis_optimizer_steps": 0,
        "target_clamping": False,
        "test_samples_accessed": 0,
    }
    if (
        summary["model_state_hash_before"] != summary["model_state_hash_after"]
        or summary["encoder_hash_before"] != summary["encoder_hash_after"]
        or summary["decoder_hash_before"] != summary["decoder_hash_after"]
        or not summary["all_parameter_grads_none"]
    ):
        raise RuntimeError(f"Q4 analysis mutated state for {snapshot_id}")
    _write_csv(records_path, rows)
    _atomic_json(summary_path, summary)
    tensor_record = {
        "snapshot_id": snapshot_id,
        "layer": layer_name,
        "path": _record_path(tensor_path),
        "sha256": file_sha256(tensor_path),
        "bp_raw_shape": list(bp_stack.shape),
        "hebbian_raw_shape": list(raw_stack.shape),
        "hebbian_effective_shape": list(effective_stack.shape),
        "dtype": str(bp_stack.numpy().dtype),
    }
    return summary, rows, tensor_record


def _aggregate_rows(pair_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in pair_summaries:
        for variant in ("hebbian_raw", "hebbian_effective"):
            metrics = summary[variant]
            rows.append(
                {
                    "snapshot_id": summary["snapshot_id"],
                    "layer": summary["layer"],
                    "variant": variant,
                    "batch_count": summary["batch_count"],
                    "parameter_count": summary["parameter_count"],
                    "batch_alignment_mean": metrics["batch_alignment_mean"],
                    "batch_alignment_std": metrics["batch_alignment_std"],
                    "batch_alignment_min": metrics["batch_alignment_min"],
                    "batch_alignment_max": metrics["batch_alignment_max"],
                    "batch_norm_ratio_mean": metrics["batch_norm_ratio_mean"],
                    "batch_norm_ratio_std": metrics["batch_norm_ratio_std"],
                    "alpha_star": metrics["alpha_star"],
                    "scale_matched_relative_bias": metrics[
                        "scale_matched_relative_bias"
                    ],
                    "mean_update_alignment": metrics["mean_update_alignment"],
                    "mean_update_norm_ratio": metrics["mean_update_norm_ratio"],
                    "raw_relative_difference": metrics[
                        "raw_relative_difference"
                    ],
                    "candidate_snr_linear": metrics["snr"]["snr_linear"],
                    "candidate_snr_db": metrics["snr"]["snr_db"],
                    "bp_snr_linear": summary["bp_reference"]["snr"]["snr_linear"],
                    "bp_snr_db": summary["bp_reference"]["snr"]["snr_db"],
                }
            )
    return rows


def _plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "neuroai_q4_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    layers = ["enc1", "enc2", "enc3"]
    variants = ["hebbian_raw", "hebbian_effective"]
    colors = {"hebbian_raw": "#3B82F6", "hebbian_effective": "#F97316"}
    labels = {"hebbian_raw": "Raw Hebbian", "hebbian_effective": "Effective Hebbian"}
    lookup = {(row["layer"], row["variant"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    x = np.arange(len(layers))
    width = 0.35
    for offset_index, variant in enumerate(variants):
        offset = (offset_index - 0.5) * width
        selected = [lookup[(layer, variant)] for layer in layers]
        axes[0, 0].bar(
            x + offset,
            [row["batch_alignment_mean"] for row in selected],
            width,
            yerr=[row["batch_alignment_std"] for row in selected],
            color=colors[variant],
            label=labels[variant],
        )
        axes[0, 1].bar(
            x + offset,
            [row["batch_norm_ratio_mean"] for row in selected],
            width,
            color=colors[variant],
        )
        axes[1, 0].bar(
            x + offset,
            [row["scale_matched_relative_bias"] for row in selected],
            width,
            color=colors[variant],
        )
        axes[1, 1].bar(
            x + offset,
            [row["candidate_snr_linear"] for row in selected],
            width,
            color=colors[variant],
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Batch update alignment")
    axes[0, 0].set_ylabel("Cosine")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("Candidate / BP norm ratio")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_title("Scale-matched relative bias")
    axes[1, 1].set_title("Across-batch candidate SNR")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.set_xticks(x, layers)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Seed-42 Q4 failure-case tooling validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare_batch_manifest(config_path: str | Path) -> Path:
    config = _load_yaml(_resolve(config_path))
    data = config["data"]
    return prepare_fixed_batch_manifest(
        split_manifest=_resolve(data["split_manifest"]),
        output_path=_resolve(data["batch_manifest"]),
        batch_count=int(data["batch_count"]),
        batch_size=int(data["batch_size"]),
        seed=int(data["batch_seed"]),
        version=config["version"],
    )


def run_q4_tooling(config_path: str | Path) -> Path:
    config_path = _resolve(config_path)
    config = _load_yaml(config_path)
    if config.get("version") != "q4-tooling-seed42-v1":
        raise ValueError("Unsupported Q4 tooling config version")
    if config["source"]["target_clamping"]:
        raise RuntimeError("Q4 target clamping must be false")
    if int(config["analysis"]["optimizer_steps"]) != 0:
        raise RuntimeError("Q4 analysis optimizer_steps must be zero")
    if config["analysis"]["include_optimizer_state_in_bp_reference"]:
        raise RuntimeError("BP reference must exclude optimizer state")
    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Formal Q4 tooling requires a clean Git worktree")
    if not _git_is_ancestor(config["protocol_base_ref"]):
        raise RuntimeError("Phase 0 canonical ref is not an ancestor of HEAD")

    manifest_path = prepare_batch_manifest(config_path)
    output_dir = _resolve(config["output_dir"])
    state_path = output_dir / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["status"] == "completed":
            raise FileExistsError(f"Completed Q4 output already exists: {output_dir}")
        if (
            state["config_sha256"] != _config_sha256(config_path)
            or state["git_commit"] != provenance["git_commit"]
        ):
            raise RuntimeError("Cannot resume Q4 tooling from different config/source")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Unrecognized Q4 output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_yaml(output_dir / "config_resolved.yaml", config)
    _atomic_json(
        state_path,
        {
            "schema_version": "q4-tooling-state-v1",
            "status": "running",
            "started_at_utc": utc_now(),
            "config_sha256": _config_sha256(config_path),
            "git_commit": provenance["git_commit"],
            "error": None,
        },
    )

    try:
        source_dir = _source_run_dir(config["source"])
        run_config = _load_yaml(source_dir / "config_resolved.yaml")
        if run_config["training"]["seed"] != config["source"]["seed"]:
            raise RuntimeError("Source run seed differs from Q4 config")
        if run_config["model"]["target_clamping"]:
            raise RuntimeError("Source model target clamping must be false")
        initial_model = ConvAutoencoder(
            latent_dim=int(run_config["model"]["latent_dim"]),
            seed=int(config["source"]["seed"]),
        )
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in initial_model.encoder.state_dict().items()
        }
        snapshot_states: dict[str, dict[str, torch.Tensor]] = {}
        snapshot_records: list[dict[str, Any]] = []
        for spec in config["snapshots"]:
            path = source_dir / spec["checkpoint"]
            state = torch.load(path, map_location="cpu", weights_only=True)
            snapshot_states[spec["id"]] = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }
            snapshot_records.append(
                {
                    "snapshot_id": spec["id"],
                    "active_layer": spec["active_layer"],
                    "path": _record_path(path),
                    "file_sha256_before": file_sha256(path),
                    "state_dict_sha256": state_dict_checksum(state),
                }
            )
        integrity_gate = snapshot_integrity_gate(snapshot_states, initial_state)
        if not integrity_gate["gate_pass"]:
            raise RuntimeError("Seed-42 greedy snapshot integrity gate failed")
        _atomic_json(
            output_dir / "snapshot_integrity_gate.json",
            {
                "schema_version": "q4-snapshot-integrity-v1",
                "gate_pass": True,
                "scope": (
                    "Greedy-freeze and checksum integrity only; this is not a "
                    "representation-health PASS."
                ),
                "checks": integrity_gate,
                "snapshots": snapshot_records,
            },
        )

        decoder_summaries: dict[str, dict[str, Any]] = {}
        for spec in config["snapshots"]:
            decoder_summaries[spec["id"]] = _train_reference_decoder(
                config,
                snapshot_id=spec["id"],
                encoder_state=snapshot_states[spec["id"]],
                output_dir=output_dir,
            )
        decoder_initial_hashes = {
            summary["decoder_initial_state_hash"]
            for summary in decoder_summaries.values()
        }
        if len(decoder_initial_hashes) != 1:
            raise RuntimeError("Reference decoders did not share paired initialization")

        batch_ids, batch_loader, batch_ids_hash = _analysis_batches(config)
        pair_summaries: list[dict[str, Any]] = []
        all_batch_rows: list[dict[str, Any]] = []
        tensor_records: list[dict[str, Any]] = []
        for spec in config["snapshots"]:
            summary, rows, tensor_record = _analyze_pair(
                config,
                snapshot_spec=spec,
                encoder_state=snapshot_states[spec["id"]],
                decoder_summary=decoder_summaries[spec["id"]],
                batch_ids=batch_ids,
                batch_loader=batch_loader,
                output_dir=output_dir,
                run_config=run_config,
            )
            pair_summaries.append(summary)
            all_batch_rows.extend(rows)
            tensor_records.append(tensor_record)

        aggregate_rows = _aggregate_rows(pair_summaries)
        _write_csv(output_dir / "batch_update_metrics.csv", all_batch_rows)
        _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
        _atomic_json(
            output_dir / "update_tensor_index.json",
            {"schema_version": "q4-update-tensor-index-v1", "records": tensor_records},
        )
        _plot_summary(aggregate_rows, output_dir / "q4_seed42_panels.png")

        for record in snapshot_records:
            path = _resolve(record["path"])
            record["file_sha256_after"] = file_sha256(path)
            record["file_unchanged"] = (
                record["file_sha256_before"] == record["file_sha256_after"]
            )
        pair_integrity = all(
            summary["model_state_hash_before"] == summary["model_state_hash_after"]
            and summary["encoder_hash_before"] == summary["encoder_hash_after"]
            and summary["decoder_hash_before"] == summary["decoder_hash_after"]
            and summary["all_parameter_grads_none"]
            and summary["analysis_optimizer_steps"] == 0
            and summary["batch_count"] == int(config["data"]["batch_count"])
            for summary in pair_summaries
        )
        gate_checks = {
            "snapshot_integrity_gate_pass": integrity_gate["gate_pass"],
            "three_snapshot_layer_pairs_complete": len(pair_summaries) == 3,
            "fifty_fixed_batches_per_pair": all(
                summary["batch_count"] == 50 for summary in pair_summaries
            ),
            "batch_ids_unique": np.unique(batch_ids).size == batch_ids.size,
            "same_batch_ids_for_all_rules_and_snapshots": True,
            "reference_decoders_complete": len(decoder_summaries) == 3,
            "paired_decoder_initialization": len(decoder_initial_hashes) == 1,
            "reference_decoder_encoders_unchanged": all(
                summary["encoder_unchanged"]
                for summary in decoder_summaries.values()
            ),
            "raw_bp_excludes_optimizer_state": True,
            "analysis_optimizer_steps_zero": all(
                summary["analysis_optimizer_steps"] == 0
                for summary in pair_summaries
            ),
            "analysis_state_checksums_unchanged": pair_integrity,
            "source_snapshot_files_unchanged": all(
                record["file_unchanged"] for record in snapshot_records
            ),
            "target_clamping_false": not config["source"]["target_clamping"],
            "all_aggregate_metrics_finite": all(
                math.isfinite(float(value))
                for row in aggregate_rows
                for key, value in row.items()
                if key
                not in {
                    "snapshot_id",
                    "layer",
                    "variant",
                }
            ),
            "test_samples_accessed_zero": True,
        }
        gate_pass = all(gate_checks.values())
        decision = {
            "schema_version": "q4-tooling-gate-decision-v1",
            "completed_at_utc": utc_now(),
            "stage": "Stage 2 / Q4 tooling gate",
            "status": "COMPLETED" if gate_pass else "RUN_BUT_NOT_VALIDATED",
            "decision": "PASS" if gate_pass else "FAIL",
            "scope": config["source"]["snapshot_role"],
            "representation_health_pass_claimed": False,
            "checks": gate_checks,
            "snapshot_layer_pairs": [
                {
                    "snapshot_id": summary["snapshot_id"],
                    "layer": summary["layer"],
                }
                for summary in pair_summaries
            ],
            "batch_count": int(config["data"]["batch_count"]),
            "batch_size": int(config["data"]["batch_size"]),
            "test_samples_accessed": 0,
            "analysis_optimizer_steps": 0,
        }
        run_manifest = {
            "schema_version": "q4-tooling-run-v1",
            "completed_at_utc": utc_now(),
            "config": _record_path(config_path),
            "config_sha256": _config_sha256(config_path),
            "protocol_base_ref": config["protocol_base_ref"],
            **provenance,
            "source_run": _record_path(source_dir),
            "source_seed": int(config["source"]["seed"]),
            "snapshot_role": config["source"]["snapshot_role"],
            "snapshots": snapshot_records,
            "batch_manifest": _record_path(manifest_path),
            "batch_manifest_sha256": file_sha256(manifest_path),
            "batch_ids_sha256": batch_ids_hash,
            "batch_count": int(config["data"]["batch_count"]),
            "batch_size": int(config["data"]["batch_size"]),
            "unique_sample_count": int(np.unique(batch_ids).size),
            "reference_decoders": decoder_summaries,
            "analysis": {
                "bp_reference": "raw reconstruction negative gradient",
                "optimizer_state_included": False,
                "optimizer_steps": 0,
                "target_clamping": False,
                "hebbian_variants": ["raw", "effective"],
                "update_centering": run_config["hebbian"].get(
                    "update_centering", "none"
                ),
                "epsilon": float(config["analysis"]["epsilon"]),
            },
            "dataset_access": {
                "source_dataset": "MNIST official training partition",
                "logical_split": "train",
                "test_samples_accessed": 0,
            },
            "outputs": [
                "snapshot_integrity_gate.json",
                "reference_decoders/*/decoder_best.pt",
                "reference_decoders/*/metrics.csv",
                "reference_decoders/*/summary.json",
                "pairs/*/update_tensors.npz",
                "pairs/*/batch_metrics.csv",
                "pairs/*/summary.json",
                "batch_update_metrics.csv",
                "aggregate_metrics.csv",
                "update_tensor_index.json",
                "q4_seed42_panels.png",
                "gate_decision.json",
                "run_manifest.json",
                "run_state.json",
                "config_resolved.yaml",
            ],
        }
        _atomic_json(output_dir / "gate_decision.json", decision)
        _atomic_json(output_dir / "run_manifest.json", run_manifest)
        _atomic_json(
            state_path,
            {
                "schema_version": "q4-tooling-state-v1",
                "status": "completed" if gate_pass else "completed_fail",
                "completed_at_utc": utc_now(),
                "config_sha256": _config_sha256(config_path),
                "git_commit": provenance["git_commit"],
                "error": None,
            },
        )
        return output_dir
    except Exception as error:
        _atomic_json(
            state_path,
            {
                "schema_version": "q4-tooling-state-v1",
                "status": "failed",
                "failed_at_utc": utc_now(),
                "config_sha256": _config_sha256(config_path),
                "git_commit": provenance["git_commit"],
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/q4_tooling_seed42_v1.yaml",
    )
    parser.add_argument("--prepare-batches-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_batches_only:
        output = prepare_batch_manifest(args.config)
    else:
        output = run_q4_tooling(args.config)
    print(output.resolve())


if __name__ == "__main__":
    main()
