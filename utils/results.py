"""Atomic, resumable experiment records shared by all training stages."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METRIC_FIELDS = [
    "experiment_id",
    "run_id",
    "git_commit",
    "model_type",
    "learning_rule",
    "architecture_id",
    "latent_dim",
    "seed",
    "stage",
    "split",
    "layer",
    "checkpoint_id",
    "epoch",
    "global_epoch",
    "step",
    "samples_seen",
    "dataset_passes",
    "wall_time_sec",
    "reconstruction_loss",
    "classification_ce",
    "accuracy",
    "macro_f1",
    "update_norm",
    "weight_norm_mean",
    "weight_norm_std",
    "preactivation_mean",
    "preactivation_std",
    "activation_mean",
    "activation_variance",
    "activation_sparsity",
    "active_neuron_ratio",
    "winner_entropy",
    "max_winner_share",
    "collapse_detected",
    "num_samples",
]

METRIC_UNIQUE_KEY = ("stage", "split", "layer", "epoch")


def create_run_directory(root: str | Path, *, rule: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{timestamp}_{rule}_seed{seed}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def write_resolved_config(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "config_resolved.yaml"
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    path = run_dir / "metadata.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite metadata: {path}")
    _atomic_json(path, metadata)


def read_metadata(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    run_dir: Path,
    filename: str,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    path = run_dir / filename
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    _atomic_json(path, payload)


def initialize_run_status(run_dir: Path, status: dict[str, Any]) -> None:
    path = run_dir / "run_status.json"
    if path.exists():
        raise FileExistsError(f"Run status already exists: {path}")
    _atomic_json(path, status)


def read_run_status(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "run_status.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_run_status(run_dir: Path, **changes: Any) -> dict[str, Any]:
    status = read_run_status(run_dir)
    status.update(changes)
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_dir / "run_status.json", status)
    return status


def _metric_static_fields(run_dir: Path) -> dict[str, Any]:
    metadata = read_metadata(run_dir)
    return {
        "experiment_id": metadata.get("experiment_id", ""),
        "run_id": metadata.get("run_id", run_dir.name),
        "git_commit": metadata.get("git_commit", ""),
        "model_type": metadata.get("model_type", "autoencoder"),
        "learning_rule": metadata.get("learning_rule", ""),
        "architecture_id": metadata.get(
            "architecture_id", metadata.get("architecture", "")
        ),
        "latent_dim": metadata.get("latent_dim", ""),
        "seed": metadata.get("seed", ""),
    }


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _atomic_write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
    os.replace(temporary, path)


def append_metric(run_dir: Path, row: dict[str, Any]) -> None:
    """Atomically upsert one metric row.

    The unique epoch key makes replay after an interruption idempotent: an
    epoch can be recomputed without duplicating its result row.
    """

    path = run_dir / "metrics.csv"
    enriched = {**_metric_static_fields(run_dir), **row}
    unknown = set(enriched) - set(METRIC_FIELDS)
    if unknown:
        raise ValueError(f"Unknown metric fields: {sorted(unknown)}")
    normalized = {field: enriched.get(field, "") for field in METRIC_FIELDS}
    key = tuple(str(normalized[field]) for field in METRIC_UNIQUE_KEY)
    rows = _read_metric_rows(path)
    replaced = False
    for index, existing in enumerate(rows):
        existing_key = tuple(str(existing.get(field, "")) for field in METRIC_UNIQUE_KEY)
        if existing_key == key:
            rows[index] = normalized
            replaced = True
            break
    if not replaced:
        rows.append(normalized)
    _atomic_write_metrics(path, rows)


def remove_metric_stages(run_dir: Path, stages: set[str]) -> None:
    """Remove rows only for stages that are about to be recomputed."""

    path = run_dir / "metrics.csv"
    if not path.exists():
        return
    rows = [row for row in _read_metric_rows(path) if row.get("stage") not in stages]
    _atomic_write_metrics(path, rows)
