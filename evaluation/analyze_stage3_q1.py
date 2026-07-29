"""Aggregate the frozen Stage 3 clean-performance and matched-control results.

This module never loads a dataset or a checkpoint.  It only consumes immutable
training/test artifacts that were produced after the relevant technical gates
passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "full_bp": "BBB",
    "full_hebbian": "HHH",
    "hybrid_hhb": "HHB",
    "hybrid_hbb": "HBB",
    "full_random": "Random",
    "random_hbb": "RBB",
    "random_rrb": "RRB",
}
METHOD_ORDER = ("BBB", "HHH", "HHB", "HBB", "Random", "RBB", "RRB")
PRIMARY_CONTRASTS = (
    ("HHB_minus_HHH", "HHB", "HHH"),
    ("HBB_minus_HHB", "HBB", "HHB"),
    ("BBB_minus_HHB", "BBB", "HHB"),
    ("HBB_minus_RBB", "HBB", "RBB"),
    ("HHB_minus_RRB", "HHB", "RRB"),
)
TEST_METRICS = (
    "accuracy",
    "macro_f1",
    "classification_ce",
    "system_reconstruction_mse",
    "standardized_reconstruction_mse",
)
BUDGET_METRICS = (
    "reconstruction_aulc",
    "system_samples_seen",
    "system_wall_time_sec",
    "standardized_decoder_samples_seen",
    "standardized_decoder_wall_time_sec",
    "probe_samples_seen",
    "probe_wall_time_sec",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = 2026,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, array.size), replace=True)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples.mean(axis=1), (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def summarize(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    ci = bootstrap_mean_ci(array)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_95_ci": list(ci),
    }


def paired_effect(left: Iterable[float], right: Iterable[float]) -> dict:
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("paired arrays must have identical one-dimensional shape")
    differences = left_array - right_array
    sd = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    mean = float(differences.mean())
    zero_tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(differences)))
    )
    if sd <= zero_tolerance:
        dz = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    else:
        dz = mean / sd
    return {
        **summarize(differences),
        "cohens_dz": dz,
    }


def reconstruction_curve(metrics_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("reconstruction_loss", "").strip()
            if row.get("split") != "validation" or not value:
                continue
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "global_epoch": int(row["global_epoch"]),
                    "samples_seen": int(row["samples_seen"]),
                    "wall_time_sec": float(row["wall_time_sec"]),
                    "reconstruction_mse": float(value),
                }
            )
    if not rows:
        raise ValueError(f"no validation reconstruction curve in {metrics_csv}")
    # The system outcome is selected from its final decoder/BP stage.  Local
    # Hebbian layer diagnostics have blank reconstruction_loss and are excluded.
    return rows


def probe_curve(metrics_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "linear_probe" or row.get("split") != "validation":
                continue
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "samples_seen": int(row["samples_seen"]),
                    "wall_time_sec": float(row["wall_time_sec"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    if not rows:
        raise ValueError(f"no validation probe curve in {metrics_csv}")
    return rows


def _read_test_rows(root: Path) -> list[dict]:
    path = root / "test_evaluation" / "per_run_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_dir(root: Path, seed: int, method_id: str) -> Path:
    return root / "runs" / f"seed_{seed}" / method_id


def collect_records(core_root: Path, controls_root: Path) -> list[dict]:
    records: list[dict] = []
    for root in (core_root, controls_root):
        for test_row in _read_test_rows(root):
            seed = int(test_row["seed"])
            method_id = test_row["method_id"]
            run_dir = _run_dir(root, seed, method_id)
            status = read_json(run_dir / "run_status.json")
            standardized_status = read_json(
                run_dir / "standardized_decoder" / "run_status.json"
            )
            reconstruction = reconstruction_curve(run_dir / "metrics.csv")
            probe = probe_curve(run_dir / "metrics.csv")
            record = {
                "seed": seed,
                "method_id": method_id,
                "method": METHOD_LABELS[method_id],
                **{
                    metric: float(test_row[metric])
                    for metric in TEST_METRICS
                },
                "reconstruction_aulc": float(
                    np.mean([row["reconstruction_mse"] for row in reconstruction])
                ),
                "system_samples_seen": int(status["samples_seen"]),
                "system_wall_time_sec": float(status["wall_time_sec"]),
                "standardized_decoder_samples_seen": int(
                    standardized_status["samples_seen"]
                ),
                "standardized_decoder_wall_time_sec": float(
                    standardized_status["wall_time_sec"]
                ),
                "probe_samples_seen": int(probe[-1]["samples_seen"]),
                "probe_wall_time_sec": float(probe[-1]["wall_time_sec"]),
                "samples_to_threshold": None,
                "samples_to_threshold_status": "NA_NOT_PREREGISTERED",
                "_reconstruction_curve": reconstruction,
                "_probe_curve": probe,
            }
            records.append(record)
    records.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), row["seed"]))
    expected = {(method, seed) for method in METHOD_ORDER for seed in range(5)}
    observed = {(row["method"], row["seed"]) for row in records}
    if observed != expected:
        raise ValueError(
            f"incomplete Stage 3 Q1 matrix: missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    return records


def _public_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def build_summary(records: list[dict]) -> dict:
    by_method: dict[str, dict] = {}
    for method in METHOD_ORDER:
        rows = [row for row in records if row["method"] == method]
        by_method[method] = {
            metric: summarize(row[metric] for row in rows)
            for metric in TEST_METRICS + BUDGET_METRICS
        }

    contrasts: dict[str, dict] = {}
    for contrast_id, left, right in PRIMARY_CONTRASTS:
        left_rows = {
            row["seed"]: row for row in records if row["method"] == left
        }
        right_rows = {
            row["seed"]: row for row in records if row["method"] == right
        }
        contrasts[contrast_id] = {
            "left": left,
            "right": right,
            "metrics": {
                metric: paired_effect(
                    [left_rows[seed][metric] for seed in range(5)],
                    [right_rows[seed][metric] for seed in range(5)],
                )
                for metric in TEST_METRICS + BUDGET_METRICS
            },
        }

    versus_random: dict[str, dict] = {}
    random_rows = {
        row["seed"]: row for row in records if row["method"] == "Random"
    }
    for method in ("BBB", "HHH", "HHB", "HBB"):
        method_rows = {
            row["seed"]: row for row in records if row["method"] == method
        }
        versus_random[f"{method}_minus_Random"] = {
            metric: paired_effect(
                [method_rows[seed][metric] for seed in range(5)],
                [random_rows[seed][metric] for seed in range(5)],
            )
            for metric in TEST_METRICS
        }

    return {
        "schema_version": "stage3-q1-complete-summary-v1",
        "methods": list(METHOD_ORDER),
        "seeds": list(range(5)),
        "statistics": {
            "paired_unit": "seed",
            "bootstrap_seed": 2026,
            "bootstrap_resamples": 10_000,
            "confidence_level": 0.95,
            "effect_size": "paired Cohens dz",
        },
        "samples_to_threshold": {
            "status": "NA_NOT_PREREGISTERED",
            "reason": (
                "No absolute threshold was frozen before the formal runs; "
                "post-hoc threshold selection is prohibited."
            ),
        },
        "by_method": by_method,
        "primary_contrasts": contrasts,
        "versus_full_random": versus_random,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output_dir: Path, records: list[dict], summary: dict) -> None:
    public = [_public_record(row) for row in records]
    _write_csv(output_dir / "per_seed_complete.csv", public, list(public[0]))

    method_rows = []
    for method in METHOD_ORDER:
        row = {"method": method}
        for metric in TEST_METRICS + BUDGET_METRICS:
            stats = summary["by_method"][method][metric]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sd"] = stats["sd"]
            row[f"{metric}_ci_low"] = stats["bootstrap_95_ci"][0]
            row[f"{metric}_ci_high"] = stats["bootstrap_95_ci"][1]
        method_rows.append(row)
    _write_csv(output_dir / "method_summary.csv", method_rows, list(method_rows[0]))

    contrast_rows = []
    for contrast_id, contrast in summary["primary_contrasts"].items():
        for metric, stats in contrast["metrics"].items():
            contrast_rows.append(
                {
                    "contrast": contrast_id,
                    "left": contrast["left"],
                    "right": contrast["right"],
                    "metric": metric,
                    "mean_difference": stats["mean"],
                    "sd_difference": stats["sd"],
                    "ci_low": stats["bootstrap_95_ci"][0],
                    "ci_high": stats["bootstrap_95_ci"][1],
                    "cohens_dz": stats["cohens_dz"],
                }
            )
    _write_csv(
        output_dir / "paired_contrasts.csv",
        contrast_rows,
        list(contrast_rows[0]),
    )


def _method_color(method: str) -> str:
    return {
        "BBB": "#2463A3",
        "HHH": "#D95F02",
        "HHB": "#1B9E77",
        "HBB": "#66A61E",
        "Random": "#777777",
        "RBB": "#7570B3",
        "RRB": "#E7298A",
    }[method]


def plot_results(output_dir: Path, records: list[dict], summary: dict) -> None:
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in METHOD_ORDER:
        rows = [row for row in records if row["method"] == method]
        curves = np.asarray(
            [
                [point["reconstruction_mse"] for point in row["_reconstruction_curve"]]
                for row in rows
            ],
            dtype=np.float64,
        )
        x = np.arange(1, curves.shape[1] + 1)
        axes[0].plot(x, curves.mean(axis=0), label=method, color=_method_color(method))
        axes[0].fill_between(
            x,
            curves.mean(axis=0) - curves.std(axis=0, ddof=1),
            curves.mean(axis=0) + curves.std(axis=0, ddof=1),
            color=_method_color(method),
            alpha=0.12,
        )
        probe_curves = np.asarray(
            [
                [point["accuracy"] for point in row["_probe_curve"]]
                for row in rows
            ],
            dtype=np.float64,
        )
        probe_x = np.arange(1, probe_curves.shape[1] + 1)
        axes[1].plot(
            probe_x,
            probe_curves.mean(axis=0),
            label=method,
            color=_method_color(method),
        )
    axes[0].set(
        xlabel="System decoder/joint epoch",
        ylabel="Validation reconstruction MSE",
        title="System reconstruction learning curves",
    )
    axes[0].set_yscale("log")
    axes[1].set(
        xlabel="Frozen probe epoch",
        ylabel="Validation accuracy",
        title="Frozen linear-probe learning curves",
    )
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png")
    fig.savefig(output_dir / "learning_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order = ("HHH", "HHB", "HBB", "BBB")
    x = np.arange(len(order))
    for axis, metric, title in (
        (axes[0], "accuracy", "Classification"),
        (axes[1], "standardized_reconstruction_mse", "Recoverable information"),
    ):
        means = [summary["by_method"][method][metric]["mean"] for method in order]
        sds = [summary["by_method"][method][metric]["sd"] for method in order]
        axis.errorbar(x, means, yerr=sds, marker="o", capsize=4, color="#222222")
        axis.set_xticks(x, order)
        axis.set_title(title)
        axis.set_ylabel(
            "Test accuracy" if metric == "accuracy" else "Standardized decoder MSE"
        )
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Hebbian-depth dose: HHH → HHB → HBB → BBB")
    fig.tight_layout()
    fig.savefig(output_dir / "hebbian_depth_dose.png")
    fig.savefig(output_dir / "hebbian_depth_dose.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, (hybrid, control, title) in zip(
        axes,
        (
            ("HBB", "RBB", "One-layer Hebbian prefix"),
            ("HHB", "RRB", "Two-layer Hebbian prefix"),
        ),
    ):
        hybrid_values = summary["by_method"][hybrid]["accuracy"]["values"]
        control_values = summary["by_method"][control]["accuracy"]["values"]
        for seed, (hybrid_value, control_value) in enumerate(
            zip(hybrid_values, control_values)
        ):
            axis.plot(
                [0, 1],
                [control_value, hybrid_value],
                marker="o",
                alpha=0.8,
                label=f"seed {seed}",
            )
        axis.set_xticks([0, 1], [control, hybrid])
        axis.set_ylabel("Test accuracy")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "matched_prefix_controls.png")
    fig.savefig(output_dir / "matched_prefix_controls.pdf")
    plt.close(fig)


def run(core_root: Path, controls_root: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; Q1 formal aggregation is immutable"
        )
    for root in (core_root, controls_root):
        gate = read_json(root / "freeze_gate.json")
        test_summary = read_json(root / "test_evaluation" / "summary.json")
        if gate["decision"] != "PASS":
            raise RuntimeError(f"freeze gate did not pass: {root}")
        if not test_summary["records_complete"]:
            raise RuntimeError(f"test records incomplete: {root}")

    records = collect_records(core_root, controls_root)
    summary = build_summary(records)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "summary.json", summary)
    write_tables(output_dir, records, summary)
    plot_results(output_dir, records, summary)
    write_json(
        output_dir / "provenance.json",
        {
            "schema_version": "stage3-q1-provenance-v1",
            "core_root": str(core_root.resolve()),
            "controls_root": str(controls_root.resolve()),
            "input_mode": "immutable artifacts only",
            "datasets_loaded": False,
            "checkpoints_loaded": False,
            "test_access_increment": 0,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.core_root, args.controls_root, args.output_dir).resolve())


if __name__ == "__main__":
    main()
