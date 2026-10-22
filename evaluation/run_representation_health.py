"""Run the validation-only Stage 1 representation-health gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.representation_health import (
    assess_layer_health,
    compute_layer_health,
    select_balanced_validation_ids,
)
from evaluation.representations import extract_representations
from models import ConvAutoencoder
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("h1", "h2", "z")


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
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_is_ancestor(ancestor: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def prepare_subset_manifest(config: dict[str, Any]) -> tuple[Path, np.ndarray, np.ndarray]:
    subset = config["subset"]
    split_path = _resolve(subset["split_manifest"])
    output_path = _resolve(subset["manifest"])
    split = np.load(split_path, allow_pickle=False)
    training = datasets.MNIST(
        root=str(_resolve(subset["data_root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    labels = np.asarray(training.targets, dtype=np.int64)
    sample_ids, sample_labels = select_balanced_validation_ids(
        split["validation_indices"],
        labels,
        samples_per_class=int(subset["samples_per_class"]),
        seed=int(subset["seed"]),
    )
    split_sha256 = file_sha256(split_path)

    if output_path.exists():
        existing = np.load(output_path, allow_pickle=False)
        required = {
            "sample_ids",
            "labels",
            "seed",
            "samples_per_class",
            "source_split_sha256",
            "version",
        }
        if not required.issubset(existing.files):
            raise RuntimeError("Existing Stage 1 subset manifest is incomplete")
        if (
            not np.array_equal(existing["sample_ids"], sample_ids)
            or not np.array_equal(existing["labels"], sample_labels)
            or int(existing["seed"].item()) != int(subset["seed"])
            or str(existing["source_split_sha256"].item()) != split_sha256
            or str(existing["version"].item()) != config["version"]
        ):
            raise RuntimeError("Existing Stage 1 subset manifest does not reproduce")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            sample_ids=sample_ids,
            labels=sample_labels,
            seed=np.asarray(subset["seed"], dtype=np.int64),
            samples_per_class=np.asarray(
                subset["samples_per_class"], dtype=np.int64
            ),
            source_split_sha256=np.asarray(split_sha256),
            version=np.asarray(config["version"]),
        )
    return output_path, sample_ids, sample_labels


def _validation_loader(
    config: dict[str, Any], sample_ids: np.ndarray
) -> DataLoader:
    training = datasets.MNIST(
        root=str(_resolve(config["subset"]["data_root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        IndexedDataset(training, sample_ids),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def run_gate(config_path: str | Path, *, prepare_only: bool = False) -> Path:
    config_path = _resolve(config_path)
    config = _load_yaml(config_path)
    if config.get("version") != "representation-health-v1":
        raise ValueError("Unsupported representation-health config version")
    manifest_path, sample_ids, sample_labels = prepare_subset_manifest(config)
    if prepare_only:
        return manifest_path

    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Formal Stage 1 requires a clean Git worktree")
    if not _git_is_ancestor(config["protocol_base_ref"]):
        raise RuntimeError("Phase 0 canonical ref is not an ancestor of HEAD")

    output_dir = _resolve(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 1 output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = _validation_loader(config, sample_ids)
    device = torch.device("cpu")
    thresholds = config["thresholds"]
    metric_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    gate_layers: dict[str, Any] = {}

    for checkpoint_spec in config["checkpoints"]:
        run_dir = _resolve(checkpoint_spec["run_dir"])
        checkpoint_path = run_dir / checkpoint_spec.get(
            "checkpoint", "model_best.pt"
        )
        run_config = _load_yaml(run_dir / "config_resolved.yaml")
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        model = ConvAutoencoder(
            int(run_config["model"]["latent_dim"]),
            seed=int(checkpoint_spec["seed"]),
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        checksum_before = state_dict_checksum(model)
        extracted = extract_representations(
            model,
            loader,
            device=device,
            layers=LAYERS,
        )
        if not torch.equal(
            extracted["sample_id"],
            torch.as_tensor(sample_ids, dtype=extracted["sample_id"].dtype),
        ):
            raise RuntimeError("Extracted sample IDs differ from frozen manifest order")
        if not torch.equal(
            extracted["label"],
            torch.as_tensor(sample_labels, dtype=extracted["label"].dtype),
        ):
            raise RuntimeError("Extracted labels differ from frozen manifest")
        checksum_after = state_dict_checksum(model)
        if checksum_before != checksum_after:
            raise RuntimeError("Representation extraction mutated a checkpoint")

        checkpoint_record = {
            "checkpoint_id": checkpoint_spec["id"],
            "role": checkpoint_spec["role"],
            "rule": checkpoint_spec["rule"],
            "seed": int(checkpoint_spec["seed"]),
            "run_dir": str(run_dir.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "state_dict_sha256_before": checksum_before,
            "state_dict_sha256_after": checksum_after,
            "source_config_sha256": metadata["config_sha256"],
            "source_initial_state_hash": metadata["initial_state_hash"],
        }
        input_records.append(checkpoint_record)

        for layer in LAYERS:
            metrics, counts, shares = compute_layer_health(
                extracted[layer],
                winner_fraction=float(config["winner_fraction"]),
                activation_epsilon=float(config["activation_epsilon"]),
                variance_epsilon=float(config["variance_epsilon"]),
            )
            assessment = assess_layer_health(metrics, thresholds)
            row = {
                "checkpoint_id": checkpoint_spec["id"],
                "role": checkpoint_spec["role"],
                "rule": checkpoint_spec["rule"],
                "seed": int(checkpoint_spec["seed"]),
                "layer": layer,
                **metrics,
                "expected_wta_sparsity_consistent": assessment[
                    "expected_wta_sparsity_consistent"
                ],
                "pathological_winner_concentration": assessment[
                    "pathological_winner_concentration"
                ],
                "representation_degeneracy": assessment[
                    "representation_degeneracy"
                ],
                "pathological_collapse": assessment["pathological_collapse"],
                "gate_pass": assessment["gate_pass"],
                "failed_checks": ";".join(assessment["failed_checks"]),
            }
            metric_rows.append(row)
            for unit, (count, share) in enumerate(zip(counts.tolist(), shares.tolist())):
                winner_rows.append(
                    {
                        "checkpoint_id": checkpoint_spec["id"],
                        "rule": checkpoint_spec["rule"],
                        "seed": int(checkpoint_spec["seed"]),
                        "layer": layer,
                        "unit": unit,
                        "winner_count": int(count),
                        "winner_share": share,
                    }
                )
            if checkpoint_spec["role"] == "primary_gate":
                gate_layers[layer] = {"metrics": metrics, **assessment}

    if set(gate_layers) != set(LAYERS):
        raise RuntimeError("Exactly one primary gate checkpoint is required")
    gate_pass = all(layer["gate_pass"] for layer in gate_layers.values())
    decision = {
        "schema_version": "representation-health-decision-v1",
        "completed_at_utc": utc_now(),
        "decision": "PASS" if gate_pass else "FAIL",
        "gate_pass": gate_pass,
        "primary_checkpoint_id": next(
            item["id"]
            for item in config["checkpoints"]
            if item["role"] == "primary_gate"
        ),
        "interpretation": (
            "All layerwise health checks passed."
            if gate_pass
            else "At least one layer failed preregistered representation-health checks; "
            "Stage 1B is required before Stage 2."
        ),
        "thresholds": thresholds,
        "layers": gate_layers,
    }
    run_manifest = {
        "schema_version": "representation-health-run-v1",
        "completed_at_utc": utc_now(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "protocol_base_ref": config["protocol_base_ref"],
        **provenance,
        "subset_manifest": str(manifest_path.relative_to(ROOT)),
        "subset_manifest_sha256": file_sha256(manifest_path),
        "source_split_sha256": file_sha256(
            _resolve(config["subset"]["split_manifest"])
        ),
        "sample_count": int(sample_ids.size),
        "class_counts": {
            str(class_id): int((sample_labels == class_id).sum())
            for class_id in range(10)
        },
        "sample_ids_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "dataset_access": {
            "source_dataset": "MNIST official training partition",
            "logical_split": "validation",
            "test_samples_accessed": 0,
        },
        "inputs": input_records,
        "outputs": [
            "health_metrics.csv",
            "winner_frequencies.csv",
            "gate_decision.json",
            "run_manifest.json",
        ],
    }
    _write_csv(output_dir / "health_metrics.csv", metric_rows)
    _write_csv(output_dir / "winner_frequencies.csv", winner_rows)
    _atomic_json(output_dir / "gate_decision.json", decision)
    _atomic_json(output_dir / "run_manifest.json", run_manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/representation_health_v1.yaml",
    )
    parser.add_argument("--prepare-manifest-only", action="store_true")
    args = parser.parse_args()
    output = run_gate(args.config, prepare_only=args.prepare_manifest_only)
    print(output.resolve())


if __name__ == "__main__":
    main()
