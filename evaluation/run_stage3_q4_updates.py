"""Formal five-seed frozen-snapshot update analysis for Stage 3 / Q4."""

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

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

from data.mnist import IndexedDataset
from evaluation.run_q4_tooling import _analyze_pair, _analysis_batches
from evaluation.update_analysis import (
    bp_raw_negative_gradient,
    snapshot_integrity_gate,
    update_snr,
)
from models import ConvAutoencoder, autoencoder_from_config
from utils.checkpointing import atomic_torch_save, file_sha256, utc_now
from utils.reproducibility import set_global_seed, state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]
METHOD_LABELS = {
    "full_bp": "BBB",
    "full_hebbian": "HHH",
    "hybrid_hhb": "HHB",
    "hybrid_hbb": "HBB",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _training_dataset_and_indices(config: dict):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root=str(resolve(config["data"]["root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    split = np.load(resolve(config["data"]["split_manifest"]), allow_pickle=False)
    return dataset, split["train_indices"], split["validation_indices"]


@torch.no_grad()
def _reconstruction_mse(model, loader) -> float:
    squared_error = 0.0
    pixels = 0
    model.eval()
    for images, _, _ in loader:
        reconstructed = model(images)
        squared_error += float((reconstructed - images).square().sum().item())
        pixels += images.numel()
    return squared_error / pixels


def train_reference_decoder(
    config: dict,
    *,
    seed: int,
    snapshot_id: str,
    encoder_state: dict[str, torch.Tensor],
    output_dir: Path,
) -> dict:
    """Epoch-boundary resumable paired decoder for one unique snapshot."""

    decoder_dir = output_dir / "reference_decoders" / snapshot_id
    decoder_dir.mkdir(parents=True, exist_ok=True)
    summary_path = decoder_dir / "summary.json"
    best_path = decoder_dir / "decoder_best.pt"
    if summary_path.exists() and best_path.exists():
        return read_json(summary_path)

    spec = config["reference_decoder"]
    model = autoencoder_from_config(config, seed=seed)
    model.encoder.load_state_dict(encoder_state)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_hash = state_dict_checksum(model.encoder)
    decoder_initial_hash = state_dict_checksum(model.decoder)
    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=float(spec["lr"]),
        betas=tuple(float(value) for value in spec["betas"]),
        weight_decay=float(spec["weight_decay"]),
    )
    start_epoch = 0
    best_epoch = 0
    best_validation = float("inf")
    best_state = None
    samples_seen = 0
    optimizer_steps = 0
    rows: list[dict] = []
    resume_path = decoder_dir / "resume_checkpoint.pt"
    if resume_path.exists():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload["snapshot_id"] != snapshot_id or payload["seed"] != seed:
            raise RuntimeError("reference decoder resume identity mismatch")
        model.decoder.load_state_dict(payload["decoder_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["completed_epoch"])
        best_epoch = int(payload["best_epoch"])
        best_validation = float(payload["best_validation_mse"])
        best_state = payload["best_decoder_state_dict"]
        samples_seen = int(payload["samples_seen"])
        optimizer_steps = int(payload["optimizer_steps"])
        rows = list(payload["rows"])

    dataset, train_ids, validation_ids = _training_dataset_and_indices(config)
    validation_loader = DataLoader(
        IndexedDataset(dataset, validation_ids),
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.MSELoss(reduction="mean")
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, int(spec["epochs"]) + 1):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        train_loader = DataLoader(
            IndexedDataset(dataset, train_ids),
            batch_size=int(config["data"]["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        model.train()
        total = 0.0
        count = 0
        for images, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = model.encoder(images)
            loss = criterion(model.decoder(z), images)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * images.shape[0]
            count += images.shape[0]
            samples_seen += images.shape[0]
            optimizer_steps += 1
        validation_mse = _reconstruction_mse(model, validation_loader)
        rows.append(
            {
                "epoch": epoch,
                "train_mse": total / count,
                "validation_mse": validation_mse,
                "samples_seen": samples_seen,
                "optimizer_steps": optimizer_steps,
                "wall_time_sec_this_process": time.perf_counter() - started,
            }
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = deepcopy(model.decoder.state_dict())
        atomic_torch_save(
            {
                "schema_version": "stage3-q4-reference-resume-v1",
                "snapshot_id": snapshot_id,
                "seed": seed,
                "completed_epoch": epoch,
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_validation_mse": best_validation,
                "best_decoder_state_dict": best_state,
                "samples_seen": samples_seen,
                "optimizer_steps": optimizer_steps,
                "rows": rows,
            },
            resume_path,
        )
        print(
            f"seed={seed} reference={snapshot_id} epoch={epoch:02d} "
            f"validation_mse={validation_mse:.6f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("reference decoder has no best state")
    model.decoder.load_state_dict(best_state)
    if state_dict_checksum(model.encoder) != encoder_hash:
        raise RuntimeError("reference decoder training mutated encoder")
    torch.save(model.decoder.state_dict(), best_path)
    write_csv(decoder_dir / "metrics.csv", rows)
    summary = {
        "schema_version": "stage3-q4-reference-decoder-v1",
        "snapshot_id": snapshot_id,
        "seed": seed,
        "epochs": int(spec["epochs"]),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "decoder_initial_state_hash": decoder_initial_hash,
        "decoder_best_state_hash": state_dict_checksum(model.decoder),
        "decoder_checkpoint": str(best_path.resolve()),
        "decoder_checkpoint_sha256": file_sha256(best_path),
        "encoder_hash_before": encoder_hash,
        "encoder_hash_after": state_dict_checksum(model.encoder),
        "encoder_unchanged": True,
        "samples_seen": samples_seen,
        "optimizer_steps_for_decoder_training": optimizer_steps,
        "analysis_optimizer_steps": 0,
        "test_samples_accessed": 0,
        "resume_checkpoint": str(resume_path.resolve()),
    }
    atomic_json(summary_path, summary)
    return summary


def analyze_bp_final(
    *,
    model: ConvAutoencoder,
    method: str,
    layer: str,
    seed: int,
    batch_ids: np.ndarray,
    loader,
    output_dir: Path,
    epsilon: float,
) -> dict:
    pair_dir = output_dir / "bp_final" / method / layer
    summary_path = pair_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    pair_dir.mkdir(parents=True, exist_ok=True)
    before = state_dict_checksum(model)
    updates = []
    rows = []
    for batch_index, (images, _, observed_ids) in enumerate(loader):
        expected = torch.as_tensor(batch_ids[batch_index], dtype=observed_ids.dtype)
        if not torch.equal(expected, observed_ids):
            raise RuntimeError("Q4 final-BP batch IDs differ")
        update, mse = bp_raw_negative_gradient(model, images, layer_name=layer)
        update = update.cpu()
        updates.append(update)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "layer": layer,
                "batch_index": batch_index,
                "batch_sample_ids_sha256": hashlib.sha256(
                    batch_ids[batch_index].tobytes()
                ).hexdigest(),
                "bp_raw_norm": float(update.norm().item()),
                "reconstruction_mse": mse,
            }
        )
    stack = torch.stack(updates)
    np.savez_compressed(
        pair_dir / "bp_raw_updates.npz",
        bp_raw=stack.numpy(),
        batch_ids=batch_ids,
    )
    write_csv(pair_dir / "batch_metrics.csv", rows)
    snr = update_snr(stack, epsilon=epsilon)
    after = state_dict_checksum(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("final BP analysis mutated model")
    summary = {
        "schema_version": "stage3-q4-final-bp-v1",
        "seed": seed,
        "method": method,
        "layer": layer,
        "batch_count": len(rows),
        "weight_shape": list(stack.shape[1:]),
        "mean_update_norm": float(
            np.mean([row["bp_raw_norm"] for row in rows])
        ),
        "sd_update_norm": float(
            np.std([row["bp_raw_norm"] for row in rows], ddof=1)
        ),
        "snr": snr,
        "model_hash_before": before,
        "model_hash_after": after,
        "all_parameter_grads_none": True,
        "analysis_optimizer_steps": 0,
        "test_samples_accessed": 0,
    }
    atomic_json(summary_path, summary)
    return summary


def _formal_rows(
    config: dict,
    pair_summaries: dict[tuple[int, str], dict],
    bp_summaries: dict[tuple[int, str, str], dict],
) -> list[dict]:
    rows = []
    for seed in config["seeds"]:
        for layer, spec in config["shared_hebbian_snapshots"].items():
            summary = pair_summaries[(seed, layer)]
            for method_id in spec["shared_methods"]:
                for variant in ("hebbian_raw", "hebbian_effective"):
                    metrics = summary[variant]
                    rows.append(
                        {
                            "seed": seed,
                            "method_id": method_id,
                            "method": METHOD_LABELS[method_id],
                            "layer": layer,
                            "rule": variant,
                            "snapshot": f"{layer}_end",
                            "alignment": metrics["mean_update_alignment"],
                            "norm_ratio": metrics["mean_update_norm_ratio"],
                            "alpha_star": metrics["alpha_star"],
                            "scale_matched_bias": metrics[
                                "scale_matched_relative_bias"
                            ],
                            "update_snr_linear": metrics["snr"]["snr_linear"],
                            "update_snr_db": metrics["snr"]["snr_db"],
                            "matched_bp_snr_linear": summary["bp_reference"]["snr"][
                                "snr_linear"
                            ],
                            "mean_update_norm": None,
                        }
                    )
        for method_id, layers in config["bp_final_layers"].items():
            for layer in layers:
                summary = bp_summaries[(seed, method_id, layer)]
                rows.append(
                    {
                        "seed": seed,
                        "method_id": method_id,
                        "method": METHOD_LABELS[method_id],
                        "layer": layer,
                        "rule": "bp_raw",
                        "snapshot": "final_validation_selected",
                        "alignment": 1.0,
                        "norm_ratio": 1.0,
                        "alpha_star": 1.0,
                        "scale_matched_bias": 0.0,
                        "update_snr_linear": summary["snr"]["snr_linear"],
                        "update_snr_db": summary["snr"]["snr_db"],
                        "matched_bp_snr_linear": summary["snr"]["snr_linear"],
                        "mean_update_norm": summary["mean_update_norm"],
                    }
                )
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["method"], row["layer"], row["rule"]) for row in rows})
    metrics = (
        "alignment",
        "norm_ratio",
        "alpha_star",
        "scale_matched_bias",
        "update_snr_linear",
        "update_snr_db",
        "matched_bp_snr_linear",
    )
    for method, layer, rule in keys:
        selected = [
            row
            for row in rows
            if (row["method"], row["layer"], row["rule"])
            == (method, layer, rule)
        ]
        result = {"method": method, "layer": layer, "rule": rule}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sd"] = float(values.std(ddof=1))
        output.append(result)
    return output


def exploratory_correlations(output_dir: Path, rows: list[dict]) -> list[dict]:
    q1 = list(
        csv.DictReader(
            (
                resolve(
                    "results/formal/phase0_v1_1/stage3_q1_complete/"
                    "per_seed_complete.csv"
                )
            ).open(encoding="utf-8")
        )
    )
    q2 = list(
        csv.DictReader(
            (
                resolve(
                    "results/formal/phase0_v1_1/stage3_q2_representation/"
                    "per_seed_layer_metrics.csv"
                )
            ).open(encoding="utf-8")
        )
    )
    q3 = list(
        csv.DictReader(
            (
                resolve(
                    "results/formal/phase0_v1_1/stage3_q3_noise/"
                    "per_seed_condition_metrics.csv"
                )
            ).open(encoding="utf-8")
        )
    )
    performance = {
        (int(row["seed"]), row["method"]): float(row["accuracy"])
        for row in q1
        if row["method"] in {"HHH", "HHB", "HBB"}
    }
    rank = {
        (int(row["seed"]), row["method"]): float(row["effective_rank"])
        for row in q2
        if row["layer"] == "z" and row["method"] in {"HHH", "HHB", "HBB"}
    }
    gaussian_drop = {
        (int(row["seed"]), row["method"]): float(
            row["accuracy_absolute_degradation"]
        )
        for row in q3
        if row["noise_type"] == "gaussian"
        and float(row["severity"]) == 0.4
        and row["method"] in {"HHH", "HHB", "HBB"}
    }
    aggregate = {}
    for row in rows:
        if row["rule"] != "hebbian_effective":
            continue
        aggregate.setdefault((row["seed"], row["method"]), []).append(row)
    joined = []
    for key, selected in aggregate.items():
        joined.append(
            {
                "seed": key[0],
                "method": key[1],
                "mean_effective_alignment": float(
                    np.mean([row["alignment"] for row in selected])
                ),
                "mean_effective_snr": float(
                    np.mean([row["update_snr_linear"] for row in selected])
                ),
                "test_accuracy": performance[key],
                "z_effective_rank": rank[key],
                "gaussian_0_4_accuracy_drop": gaussian_drop[key],
            }
        )
    write_csv(output_dir / "cross_metric_join.csv", joined)
    correlations = []
    for update_metric in ("mean_effective_alignment", "mean_effective_snr"):
        for outcome in (
            "test_accuracy",
            "z_effective_rank",
            "gaussian_0_4_accuracy_drop",
        ):
            coefficient, pvalue = spearmanr(
                [row[update_metric] for row in joined],
                [row[outcome] for row in joined],
            )
            correlations.append(
                {
                    "update_metric": update_metric,
                    "outcome": outcome,
                    "spearman_rho": float(coefficient),
                    "two_sided_pvalue_descriptive_only": float(pvalue),
                    "row_count": len(joined),
                    "nonindependence_warning": (
                        "Shared Hebbian prefixes duplicate some update records; "
                        "exploratory association only."
                    ),
                }
            )
    return correlations


def plot_summary(output_dir: Path, summary: list[dict]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    hebbian = [row for row in summary if row["rule"] == "hebbian_effective"]
    labels = [f"{row['method']}-{row['layer']}" for row in hebbian]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, title in (
        (axes[0, 0], "alignment", "Effective Hebbian–BP alignment"),
        (axes[0, 1], "norm_ratio", "Effective/BP norm ratio"),
        (axes[1, 0], "scale_matched_bias", "Scale-matched relative bias"),
        (axes[1, 1], "update_snr_linear", "Effective Hebbian SNR"),
    ):
        values = [row[f"{metric}_mean"] for row in hebbian]
        errors = [row[f"{metric}_sd"] for row in hebbian]
        ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=3)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "hebbian_update_mechanisms.png", dpi=160)
    fig.savefig(figures / "hebbian_update_mechanisms.pdf")
    plt.close(fig)

    bp = [row for row in summary if row["rule"] == "bp_raw"]
    labels = [f"{row['method']}-{row['layer']}" for row in bp]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(len(labels)),
        [row["update_snr_linear_mean"] for row in bp],
        yerr=[row["update_snr_linear_sd"] for row in bp],
        capsize=3,
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylabel("Raw BP gradient SNR")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "bp_gradient_snr.png", dpi=160)
    fig.savefig(figures / "bp_gradient_snr.pdf")
    plt.close(fig)


def run(config_path: Path, *, stop_after_seed: int | None = None) -> Path:
    protocol_path = resolve(config_path)
    config = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if config["version"] != "stage3-q4-updates-v1":
        raise ValueError("unsupported Q4 protocol")
    source_root = resolve(config["source_results_root"])
    if read_json(source_root / "freeze_gate.json")["decision"] != "PASS":
        raise RuntimeError("Stage 3 source freeze gate did not pass")
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_ids, batch_loader, batch_hash = _analysis_batches(config)
    pair_summaries = {}
    bp_summaries = {}
    integrity = {}

    for seed in config["seeds"]:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        states = {}
        source_files = {}
        hhh_dir = source_root / "runs" / f"seed_{seed}" / "full_hebbian"
        hhh_config = yaml.safe_load(
            (hhh_dir / "config_resolved.yaml").read_text(encoding="utf-8")
        )
        initial = autoencoder_from_config(
            hhh_config, seed=seed
        ).encoder.state_dict()
        for layer, spec in config["shared_hebbian_snapshots"].items():
            path = hhh_dir / spec["checkpoint"]
            states[f"{layer}_end"] = torch.load(
                path, map_location="cpu", weights_only=True
            )
            source_files[layer] = {
                "path": str(path),
                "sha256_before": file_sha256(path),
            }
            for method_id in spec["shared_methods"]:
                matched = (
                    source_root
                    / "runs"
                    / f"seed_{seed}"
                    / method_id
                    / spec["checkpoint"]
                )
                if file_sha256(matched) != source_files[layer]["sha256_before"]:
                    raise RuntimeError(
                        f"shared Hebbian snapshot mismatch: seed={seed} "
                        f"layer={layer} method={method_id}"
                    )
        snapshot_gate = snapshot_integrity_gate(states, initial)
        if not snapshot_gate["gate_pass"]:
            raise RuntimeError(f"snapshot integrity failed for seed {seed}")

        analysis_config = {
            **config,
            "model": hhh_config["model"],
            "source": {
                "seed": seed,
                "snapshot_role": "formal_shared_hebbian_layer_end",
            },
            "reference_decoder": {
                **config["reference_decoder"],
                "initialization_seed": seed,
                "loader_seed": seed,
            },
        }
        for layer in ("enc1", "enc2", "enc3"):
            snapshot_id = f"{layer}_end"
            decoder = train_reference_decoder(
                analysis_config,
                seed=seed,
                snapshot_id=snapshot_id,
                encoder_state=states[snapshot_id],
                output_dir=seed_dir,
            )
            summary, _, _ = _analyze_pair(
                analysis_config,
                snapshot_spec={
                    "id": snapshot_id,
                    "active_layer": layer,
                },
                encoder_state=states[snapshot_id],
                decoder_summary=decoder,
                batch_ids=batch_ids,
                batch_loader=batch_loader,
                output_dir=seed_dir / "shared_hebbian",
                run_config=hhh_config,
            )
            pair_summaries[(seed, layer)] = summary

        for method_id, layers in config["bp_final_layers"].items():
            run_dir = source_root / "runs" / f"seed_{seed}" / method_id
            resolved = yaml.safe_load(
                (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
            )
            model = autoencoder_from_config(resolved, seed=seed)
            model.load_state_dict(
                torch.load(
                    run_dir / "model_best.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            model.eval()
            for layer in layers:
                bp_summaries[(seed, method_id, layer)] = analyze_bp_final(
                    model=model,
                    method=method_id,
                    layer=layer,
                    seed=seed,
                    batch_ids=batch_ids,
                    loader=batch_loader,
                    output_dir=seed_dir,
                    epsilon=float(config["analysis"]["epsilon"]),
                )
        for layer, record in source_files.items():
            record["sha256_after"] = file_sha256(record["path"])
            record["unchanged"] = (
                record["sha256_before"] == record["sha256_after"]
            )
        integrity[str(seed)] = {
            "snapshot_gate": snapshot_gate,
            "source_files": source_files,
            "all_source_files_unchanged": all(
                record["unchanged"] for record in source_files.values()
            ),
            "test_samples_accessed": 0,
            "analysis_optimizer_steps": 0,
        }
        atomic_json(seed_dir / "integrity.json", integrity[str(seed)])
        if stop_after_seed is not None and seed >= stop_after_seed:
            atomic_json(
                output_dir / "paused_status.json",
                {
                    "status": "paused_at_seed_boundary",
                    "last_completed_seed": seed,
                    "completed_at_utc": utc_now(),
                },
            )
            return output_dir

    rows = _formal_rows(config, pair_summaries, bp_summaries)
    summary = summarize_rows(rows)
    write_csv(output_dir / "per_seed_layer_update_metrics.csv", rows)
    write_csv(output_dir / "method_layer_update_summary.csv", summary)
    if config["analysis"].get("exploratory_correlations", True):
        correlations = exploratory_correlations(output_dir, rows)
        write_csv(output_dir / "exploratory_correlations.csv", correlations)
    plot_summary(output_dir, summary)
    atomic_json(
        output_dir / "integrity.json",
        {
            "schema_version": "stage3-q4-integrity-v1",
            "completed_at_utc": utc_now(),
            "batch_ids_sha256": batch_hash,
            "fixed_batch_count": 50,
            "fixed_batch_size": 128,
            "seed_count": 5,
            "per_seed": integrity,
            "all_source_files_unchanged": all(
                value["all_source_files_unchanged"]
                for value in integrity.values()
            ),
            "analysis_optimizer_steps": 0,
            "test_samples_accessed": 0,
            "formal_update_rows": len(rows),
            "all_metrics_finite": all(
                math.isfinite(float(row[key]))
                for row in rows
                for key in (
                    "alignment",
                    "norm_ratio",
                    "alpha_star",
                    "scale_matched_bias",
                    "update_snr_linear",
                    "update_snr_db",
                )
            ),
        },
    )
    atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "stage3-q4-run-manifest-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "source_freeze_gate": "PASS",
            "seeds": config["seeds"],
            "shared_snapshot_policy": config["shared_hebbian_snapshots"],
            "bp_final_layers": config["bp_final_layers"],
            "formal_update_rows": len(rows),
            "training_performed": "paired reference decoders only",
            "analysis_optimizer_steps": 0,
            "test_samples_accessed": 0,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage3_q4_updates_v1.yaml"),
    )
    parser.add_argument("--stop-after-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.config, stop_after_seed=args.stop_after_seed).resolve())


if __name__ == "__main__":
    main()
