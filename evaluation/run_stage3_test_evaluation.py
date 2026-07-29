"""One-time frozen test evaluation for the Stage 3 formal core matrix."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from data.mnist import build_mnist_dataloaders
from evaluation.metrics import classification_metrics
from models import ConvAutoencoder, LinearProbe, autoencoder_from_config
from schemas import load_config
from utils.reproducibility import state_dict_checksum
from utils.results import write_json


METHODS = ("full_bp", "full_hebbian", "hybrid_hhb", "hybrid_hbb", "full_random")
SEEDS = (0, 1, 2, 3, 4)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_freeze_gate(results_root: Path) -> dict[str, Any]:
    gate_path = results_root / "freeze_gate.json"
    if not gate_path.exists():
        raise RuntimeError("Stage 3 freeze gate is missing; refusing test access")
    gate = _read_json(gate_path)
    if gate.get("decision") != "PASS":
        raise RuntimeError("Stage 3 freeze gate did not pass; refusing test access")
    checks = gate.get("global_checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise RuntimeError("Stage 3 freeze gate contains a failed global check")
    if int(gate.get("test_samples_accessed", -1)) != 0:
        raise RuntimeError("Pre-test gate does not report zero test access")
    return gate


def _load_frozen_components(run_dir: Path, config: dict[str, Any], device: torch.device):
    seed = int(config["training"]["seed"])
    latent_dim = int(config["model"]["latent_dim"])
    system = autoencoder_from_config(config, seed=seed)
    system.load_state_dict(
        torch.load(run_dir / "model_best.pt", map_location="cpu", weights_only=True)
    )
    standardized = autoencoder_from_config(config, seed=seed)
    standardized.load_state_dict(system.state_dict())
    standardized.decoder.load_state_dict(
        torch.load(
            run_dir / "standardized_decoder" / "decoder_best.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    probe = LinearProbe(
        latent_dim,
        epsilon=float(config["probe"]["standardization_epsilon"]),
    )
    probe.load_state_dict(
        torch.load(run_dir / "linear_probe.pt", map_location="cpu", weights_only=True)
    )
    for component in (system, standardized, probe):
        component.to(device).eval()
        for parameter in component.parameters():
            parameter.requires_grad_(False)
    return system, standardized, probe


@torch.no_grad()
def evaluate_frozen_components(
    system: ConvAutoencoder,
    standardized: ConvAutoencoder,
    probe: LinearProbe,
    loader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    hashes_before = {
        "system": state_dict_checksum(system),
        "standardized": state_dict_checksum(standardized),
        "probe": state_dict_checksum(probe),
    }
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    system_squared_error = 0.0
    standardized_squared_error = 0.0
    num_pixels = 0
    criterion = nn.MSELoss(reduction="sum")

    for images, batch_labels, batch_ids in loader:
        images = images.to(device)
        z = system.encode(images)
        logits = probe(z)
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        labels.append(batch_labels.cpu())
        sample_ids.append(torch.as_tensor(batch_ids).cpu())
        system_reconstruction = system.decode(z)
        standardized_reconstruction = standardized.decode(z)
        system_squared_error += float(criterion(system_reconstruction, images).item())
        standardized_squared_error += float(
            criterion(standardized_reconstruction, images).item()
        )
        num_pixels += images.numel()

    labels_tensor = torch.cat(labels)
    probabilities_tensor = torch.cat(probabilities)
    ids_tensor = torch.cat(sample_ids).long()
    if ids_tensor.unique().numel() != ids_tensor.numel():
        raise RuntimeError("Test sample IDs are not unique")
    labels_np = labels_tensor.numpy()
    probabilities_np = probabilities_tensor.numpy()
    metrics = classification_metrics(
        labels_np,
        probabilities_np.argmax(axis=1),
        probabilities_np,
    )
    hashes_after = {
        "system": state_dict_checksum(system),
        "standardized": state_dict_checksum(standardized),
        "probe": state_dict_checksum(probe),
    }
    if hashes_before != hashes_after:
        raise RuntimeError("Frozen model or probe changed during test evaluation")
    return {
        **metrics,
        "system_reconstruction_mse": system_squared_error / num_pixels,
        "standardized_reconstruction_mse": standardized_squared_error / num_pixels,
        "num_samples": int(labels_tensor.numel()),
        "num_pixels": int(num_pixels),
        "sample_id_min": int(ids_tensor.min().item()),
        "sample_id_max": int(ids_tensor.max().item()),
        "sample_ids_unique": True,
        "checksums_before": hashes_before,
        "checksums_after": hashes_after,
        "frozen_components_unchanged": True,
    }


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    )
    by_method: dict[str, Any] = {}
    for method in METHODS:
        rows = [row for row in records if row["method_id"] == method]
        by_method[method] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            by_method[method][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "bootstrap_95_ci": _bootstrap_ci(values),
                "values": values.tolist(),
            }

    contrasts = (
        ("HHB_minus_HHH", "hybrid_hhb", "full_hebbian"),
        ("HBB_minus_HHB", "hybrid_hbb", "hybrid_hhb"),
        ("BBB_minus_HHB", "full_bp", "hybrid_hhb"),
    )
    paired: dict[str, Any] = {}
    lookup = {(row["seed"], row["method_id"]): row for row in records}
    for name, left, right in contrasts:
        paired[name] = {}
        for metric in metrics:
            values = np.asarray(
                [lookup[(seed, left)][metric] - lookup[(seed, right)][metric] for seed in SEEDS],
                dtype=np.float64,
            )
            paired[name][metric] = {
                "mean_difference": float(values.mean()),
                "sd_difference": float(values.std(ddof=1)),
                "paired_bootstrap_95_ci": _bootstrap_ci(values),
                "paired_differences": values.tolist(),
            }
    return {
        "by_method": by_method,
        "paired_contrasts": paired,
        "pending_matched_controls": {
            "HBB_minus_RBB": "not_available_in_stage3_core",
            "HHB_minus_RRB": "not_available_in_stage3_core",
        },
    }


def run(results_root: Path, *, device: torch.device | None = None) -> Path:
    gate = require_freeze_gate(results_root)
    output_dir = results_root / "test_evaluation"
    if output_dir.exists():
        raise FileExistsError(
            "One-time Stage 3 test evaluation already exists; refusing a second access"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            for method in METHODS:
                run_dir = results_root / "runs" / f"seed_{seed}" / method
                config = load_config(run_dir / "config_resolved.yaml")
                system, standardized, probe = _load_frozen_components(
                    run_dir, config, device
                )
                loader = build_mnist_dataloaders(
                    config,
                    seed=seed,
                    download=False,
                    include_test=True,
                )["test"]
                metrics = evaluate_frozen_components(
                    system, standardized, probe, loader, device=device
                )
                record = {
                    "schema_version": "stage3-one-time-test-run-v1",
                    "seed": seed,
                    "method_id": method,
                    "git_commit": _read_json(run_dir / "metadata.json")["git_commit"],
                    "split": "test",
                    "test_access_ordinal": 1,
                    **metrics,
                }
                write_json(
                    output_dir,
                    f"seed_{seed}_{method}.json",
                    record,
                )
                records.append(record)
                print(
                    f"seed={seed} method={method} "
                    f"accuracy={metrics['accuracy']:.4f} "
                    f"system_mse={metrics['system_reconstruction_mse']:.6f} "
                    f"standardized_mse={metrics['standardized_reconstruction_mse']:.6f}"
                )
    except Exception:
        write_json(
            output_dir,
            "FAILED.json",
            {
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_records": len(records),
            },
        )
        raise

    summary = {
        "schema_version": "stage3-one-time-test-summary-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_gate_decision": gate["decision"],
        "source_git_commit": gate.get("source_git_commit", records[0]["git_commit"]),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "records_complete": len(records) == len(METHODS) * len(SEEDS),
        "total_checkpoint_test_evaluations": len(records),
        "test_samples_per_checkpoint": records[0]["num_samples"],
        "each_checkpoint_accessed_once": all(
            row["test_access_ordinal"] == 1 for row in records
        ),
        **_summarize(records),
    }
    write_json(output_dir, "summary.json", summary)
    with (output_dir / "per_run_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = [
            "seed",
            "method_id",
            "accuracy",
            "macro_f1",
            "classification_ce",
            "system_reconstruction_mse",
            "standardized_reconstruction_mse",
            "num_samples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    return output_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/formal/phase0_v1_1/stage3_core",
    )
    args = parser.parse_args()
    print(run(Path(args.results_root)).resolve())


if __name__ == "__main__":
    main()
