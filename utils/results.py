"""Minimal, non-overwriting result storage."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METRIC_FIELDS = [
    "stage",
    "split",
    "layer",
    "epoch",
    "global_epoch",
    "reconstruction_loss",
    "classification_ce",
    "accuracy",
    "macro_f1",
    "update_norm",
    "weight_norm_mean",
    "weight_norm_std",
    "activation_mean",
    "activation_sparsity",
    "active_neuron_ratio",
    "winner_entropy",
    "num_samples",
]


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


def write_resolved_config(run_dir: Path, config: dict[str, Any]) -> None:
    with (run_dir / "config_resolved.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    with (run_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def write_json(run_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    with (run_dir / filename).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_metric(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "metrics.csv"
    exists = path.exists()
    unknown = set(row) - set(METRIC_FIELDS)
    if unknown:
        raise ValueError(f"Unknown metric fields: {sorted(unknown)}")
    normalized = {field: row.get(field, "") for field in METRIC_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def remove_metric_stages(run_dir: Path, stages: set[str]) -> None:
    """Remove prior rows for stages that are about to be recomputed.

    Representation training and probe training share one metrics file.  A probe
    rerun should replace only its own rows while preserving the encoder and
    decoder training history.
    """
    path = run_dir / "metrics.csv"
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] not in stages]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
