from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from evaluation.analyze_stage3_q1 import (
    bootstrap_mean_ci,
    paired_effect,
    probe_curve,
    reconstruction_curve,
)


HEADER = [
    "stage",
    "split",
    "epoch",
    "global_epoch",
    "samples_seen",
    "wall_time_sec",
    "reconstruction_loss",
    "accuracy",
]


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_paired_effect_uses_seed_level_differences() -> None:
    result = paired_effect([0.9, 0.8, 0.7], [0.8, 0.7, 0.6])
    assert result["mean"] == pytest.approx(0.1)
    assert math.isinf(result["cohens_dz"])
    assert result["cohens_dz"] > 0


def test_paired_effect_zero_difference_has_zero_effect() -> None:
    result = paired_effect([1.0, 2.0], [1.0, 2.0])
    assert result["mean"] == 0.0
    assert result["cohens_dz"] == 0.0


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0])
    second = bootstrap_mean_ci([1.0, 2.0, 3.0])
    assert first == second


def test_reconstruction_curve_uses_validation_nonblank_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path,
        [
            {
                "stage": "hebbian",
                "split": "train",
                "epoch": 1,
                "global_epoch": 1,
                "samples_seen": 10,
                "wall_time_sec": 1,
                "reconstruction_loss": "",
                "accuracy": "",
            },
            {
                "stage": "decoder",
                "split": "validation",
                "epoch": 1,
                "global_epoch": 2,
                "samples_seen": 20,
                "wall_time_sec": 2,
                "reconstruction_loss": 0.2,
                "accuracy": "",
            },
            {
                "stage": "decoder",
                "split": "validation",
                "epoch": 2,
                "global_epoch": 3,
                "samples_seen": 30,
                "wall_time_sec": 3,
                "reconstruction_loss": 0.1,
                "accuracy": "",
            },
        ],
    )
    curve = reconstruction_curve(path)
    assert [row["reconstruction_mse"] for row in curve] == [0.2, 0.1]


def test_probe_curve_excludes_final_selection_row(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_metrics(
        path,
        [
            {
                "stage": "linear_probe",
                "split": "validation",
                "epoch": 1,
                "global_epoch": 1,
                "samples_seen": 10,
                "wall_time_sec": 1,
                "reconstruction_loss": "",
                "accuracy": 0.8,
            },
            {
                "stage": "linear_probe_final",
                "split": "validation",
                "epoch": 1,
                "global_epoch": 1,
                "samples_seen": 10,
                "wall_time_sec": 1,
                "reconstruction_loss": "",
                "accuracy": 0.8,
            },
        ],
    )
    assert probe_curve(path) == [
        {
            "epoch": 1,
            "samples_seen": 10,
            "wall_time_sec": 1.0,
            "accuracy": 0.8,
        }
    ]
