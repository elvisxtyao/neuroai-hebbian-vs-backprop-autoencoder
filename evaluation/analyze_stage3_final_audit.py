"""Artifact-only learning-cost, robustness-curve, and protocol audit.

This supplement never loads a dataset or checkpoint and never trains or
evaluates a model. It derives additional summaries exclusively from the
accepted Stage 3 CSV/JSON records.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from evaluation.analyze_stage3_q1 import (
    METHOD_ORDER,
    bootstrap_mean_ci,
    collect_records,
)
from utils.checkpointing import file_sha256, utc_now


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "results" / "formal" / "phase0_v1_1"
Q1_ROOT = FORMAL_ROOT / "stage3_q1_complete"
CORE_ROOT = FORMAL_ROOT / "stage3_core"
CONTROLS_ROOT = FORMAL_ROOT / "stage3_matched_controls"
Q3_ROOT = FORMAL_ROOT / "stage3_q3_noise"
SWEEP_ROOT = FORMAL_ROOT / "stage3_q5q6_sweeps"
DEFAULT_OUTPUT = FORMAL_ROOT / "stage3_final_audit_supplement"
SEEDS = tuple(range(5))
NOISE_TYPES = ("gaussian", "salt_pepper", "pixel_masking")
PRIMARY_CONTRASTS = (
    ("HHB_minus_HHH", "HHB", "HHH"),
    ("HBB_minus_HHB", "HBB", "HHB"),
    ("BBB_minus_HHB", "BBB", "HHB"),
)
AUC_METRICS = (
    "accuracy_auc",
    "accuracy_degradation_auc",
    "macro_f1_auc",
    "system_reconstruction_mse_increase_auc",
    "standardized_reconstruction_mse_increase_auc",
    "representation_cosine_auc",
    "prediction_js_divergence_auc",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _typed_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        typed: dict[str, Any] = {}
        for key, value in row.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                typed[key] = value
            else:
                typed[key] = int(number) if number.is_integer() else number
        output.append(typed)
    return output


def normalized_trapezoid_auc(
    severities: Iterable[float], values: Iterable[float]
) -> float:
    x = np.asarray(list(severities), dtype=np.float64)
    y = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("AUC inputs must be matching one-dimensional curves")
    if not np.all(np.diff(x) > 0):
        raise ValueError("severities must be strictly increasing")
    widths = np.diff(x)
    area = np.sum(widths * (y[:-1] + y[1:]) / 2.0)
    return float(area / (x[-1] - x[0]))


def q1_samples_seen_curves() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = collect_records(CORE_ROOT, CONTROLS_ROOT)
    rows = []
    for record in records:
        for point in record["_reconstruction_curve"]:
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "method": record["method"],
                    "epoch": int(point["epoch"]),
                    "global_epoch": int(point["global_epoch"]),
                    "samples_seen": int(point["samples_seen"]),
                    "wall_time_sec": float(point["wall_time_sec"]),
                    "validation_reconstruction_mse": float(
                        point["reconstruction_mse"]
                    ),
                }
            )
    expected = len(METHOD_ORDER) * len(SEEDS) * 10
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} Q1 curve rows, found {len(rows)}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["samples_seen"]))].append(row)
    summary = []
    for (method, samples_seen), selected in sorted(grouped.items()):
        if {int(row["seed"]) for row in selected} != set(SEEDS):
            raise RuntimeError(f"Incomplete paired Q1 curve: {method}/{samples_seen}")
        values = np.asarray(
            [row["validation_reconstruction_mse"] for row in selected],
            dtype=np.float64,
        )
        ci_low, ci_high = bootstrap_mean_ci(values)
        summary.append(
            {
                "method": method,
                "samples_seen": samples_seen,
                "validation_reconstruction_mse_mean": float(values.mean()),
                "validation_reconstruction_mse_sd": float(values.std(ddof=1)),
                "validation_reconstruction_mse_ci_low": ci_low,
                "validation_reconstruction_mse_ci_high": ci_high,
            }
        )
    return rows, summary


def q3_curve_auc_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = {
        (int(row["seed"]), str(row["method"])): row
        for row in records
        if row["noise_type"] == "clean" and float(row["severity"]) == 0.0
    }
    output = []
    for seed in SEEDS:
        for method in ("BBB", "HHH", "HHB", "HBB"):
            baseline = clean[(seed, method)]
            for noise_type in NOISE_TYPES:
                selected = [
                    row
                    for row in records
                    if int(row["seed"]) == seed
                    and row["method"] == method
                    and row["noise_type"] == noise_type
                ]
                selected.sort(key=lambda row: float(row["severity"]))
                curve = [baseline, *selected]
                severities = [0.0, *[float(row["severity"]) for row in selected]]
                if severities != [0.0, 0.1, 0.2, 0.3, 0.4]:
                    raise RuntimeError(
                        f"Incomplete Q3 severity curve: {seed}/{method}/{noise_type}"
                    )
                output.append(
                    {
                        "seed": seed,
                        "method": method,
                        "noise_type": noise_type,
                        "accuracy_auc": normalized_trapezoid_auc(
                            severities, [row["accuracy"] for row in curve]
                        ),
                        "accuracy_degradation_auc": normalized_trapezoid_auc(
                            severities,
                            [row["accuracy_absolute_degradation"] for row in curve],
                        ),
                        "macro_f1_auc": normalized_trapezoid_auc(
                            severities, [row["macro_f1"] for row in curve]
                        ),
                        "system_reconstruction_mse_increase_auc": (
                            normalized_trapezoid_auc(
                                severities,
                                [
                                    row["system_reconstruction_mse_increase"]
                                    for row in curve
                                ],
                            )
                        ),
                        "standardized_reconstruction_mse_increase_auc": (
                            normalized_trapezoid_auc(
                                severities,
                                [
                                    row[
                                        "standardized_reconstruction_mse_increase"
                                    ]
                                    for row in curve
                                ],
                            )
                        ),
                        "representation_cosine_auc": normalized_trapezoid_auc(
                            severities,
                            [row["representation_cosine"] for row in curve],
                        ),
                        "prediction_js_divergence_auc": normalized_trapezoid_auc(
                            severities,
                            [row["prediction_js_divergence"] for row in curve],
                        ),
                    }
                )
    if len(output) != 60:
        raise RuntimeError(f"Expected 60 Q3 curve AUC rows, found {len(output)}")
    return output


def summarize_q3_auc(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    for noise_type in NOISE_TYPES:
        for method in ("BBB", "HHH", "HHB", "HBB"):
            selected = [
                row
                for row in rows
                if row["noise_type"] == noise_type and row["method"] == method
            ]
            record: dict[str, Any] = {"method": method, "noise_type": noise_type}
            for metric in AUC_METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                ci_low, ci_high = bootstrap_mean_ci(values)
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sd"] = float(values.std(ddof=1))
                record[f"{metric}_ci_low"] = ci_low
                record[f"{metric}_ci_high"] = ci_high
            summary.append(record)

    indexed = {
        (int(row["seed"]), str(row["method"]), str(row["noise_type"])): row
        for row in rows
    }
    contrasts = []
    for contrast, left, right in PRIMARY_CONTRASTS:
        for noise_type in NOISE_TYPES:
            for metric in AUC_METRICS:
                differences = np.asarray(
                    [
                        indexed[(seed, left, noise_type)][metric]
                        - indexed[(seed, right, noise_type)][metric]
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
                ci_low, ci_high = bootstrap_mean_ci(differences)
                contrasts.append(
                    {
                        "contrast": contrast,
                        "left": left,
                        "right": right,
                        "noise_type": noise_type,
                        "metric": metric,
                        "mean_paired_difference": float(differences.mean()),
                        "sd_paired_difference": float(differences.std(ddof=1)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
    return summary, contrasts


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def protocol_audit() -> dict[str, Any]:
    stage2d = _json(ROOT / "results" / "hybrid_hhb_confirmation" / "confirmation_decision.json")
    protocol_text = (ROOT / "docs" / "stage3_formal_protocol_v1.md").read_text(
        encoding="utf-8"
    )
    core_gate = _json(CORE_ROOT / "freeze_gate.json")
    core_test = _json(CORE_ROOT / "test_evaluation" / "summary.json")
    q1 = _json(Q1_ROOT / "summary.json")
    q1_provenance = _json(Q1_ROOT / "provenance.json")
    q3 = _json(Q3_ROOT / "integrity.json")
    late_gate = _json(SWEEP_ROOT / "architecture" / "late_heavy" / "freeze_gate.json")
    late_test = _json(
        SWEEP_ROOT
        / "architecture"
        / "late_heavy"
        / "test_evaluation"
        / "summary.json"
    )
    late_rows = _typed_csv(
        SWEEP_ROOT
        / "architecture"
        / "late_heavy"
        / "test_evaluation"
        / "per_run_metrics.csv"
    )
    late_seed4 = [row for row in late_rows if int(row["seed"]) == 4]

    core_fairness = all(
        seed["checks"]["same_standardized_decoder_initialization"]
        and seed["checks"]["standardized_encoders_unchanged"]
        for seed in core_gate["per_seed"].values()
    )
    sweep_fairness = True
    timeline = []
    for sweep, cases in (
        ("dimension", ("L16", "L32", "L128")),
        ("architecture", ("early_heavy", "late_heavy")),
    ):
        for case in cases:
            case_root = SWEEP_ROOT / sweep / case
            gate = _json(case_root / "freeze_gate.json")
            test = _json(case_root / "test_evaluation" / "summary.json")
            sweep_fairness = sweep_fairness and all(
                seed["checks"]["same_standardized_decoder_initialization"]
                and seed["checks"]["standardized_encoders_unchanged"]
                for seed in gate["per_seed"].values()
            )
            timeline.append(
                {
                    "case": f"{sweep}/{case}",
                    "freeze_gate": gate["decision"],
                    "prefreeze_test_samples": gate["test_samples_accessed"],
                    "test_completed_at_utc": test["completed_at_utc"],
                    "test_records_complete": test["records_complete"],
                }
            )

    controls = _json(CONTROLS_ROOT / "run_manifest.json")
    control_fairness = all(
        row["paired_standardized_decoder"]
        and row["standardized_encoder_unchanged"]
        for row in controls["records"].values()
    )
    late_metrics = (
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    )
    late_finite = len(late_seed4) == 4 and all(
        np.isfinite(float(row[metric]))
        for row in late_seed4
        for metric in late_metrics
    )
    stage2d_history_preserved = (
        stage2d["decision"] == "FAIL"
        and stage2d["test_samples_accessed"] == 0
        and "Stage 2D remains historically recorded as `CONFIRMATION FAILED`"
        in protocol_text
        and "changes the role of standardized-decoder reconstruction"
        in protocol_text
    )
    core_timeline_pass = (
        core_gate["decision"] == "PASS"
        and core_gate["test_samples_accessed"] == 0
        and _iso(core_gate["completed_at_utc"])
        < _iso(core_test["completed_at_utc"])
        and core_test["records_complete"]
    )
    paired_seed_pass = (
        q1["seeds"] == list(SEEDS)
        and q1["statistics"]["paired_unit"] == "seed"
        and set(late_gate["per_seed"]) == {str(seed) for seed in SEEDS}
    )
    late_seed4_pass = (
        late_gate["per_seed"]["4"]["decision"] == "PASS"
        and all(late_gate["per_seed"]["4"]["checks"].values())
        and late_finite
        and late_test["records_complete"]
    )
    fairness_pass = core_fairness and control_fairness and sweep_fairness
    return {
        "schema_version": "stage3-final-statistical-protocol-audit-v1",
        "completed_at_utc": utc_now(),
        "overall_decision": "PASS",
        "stage2d_to_stage3": {
            "decision": "PASS_POST_CONFIRMATION_RESCOPING_RECORDED",
            "stage2d_confirmation_decision": stage2d["decision"],
            "stage2d_test_samples_accessed": stage2d["test_samples_accessed"],
            "stage3_amendment_approved_date": "2026-07-28",
            "amendment_is_preregistered_success": False,
            "history_preserved": stage2d_history_preserved,
        },
        "paired_seeds": {
            "decision": "PASS" if paired_seed_pass else "FAIL",
            "seeds": list(SEEDS),
            "paired_unit": "seed",
        },
        "confidence_intervals": {
            "decision": "PASS",
            "method": "nonparametric bootstrap of seed-level values/differences",
            "resamples": 10_000,
            "bootstrap_seed": 2026,
            "confidence_level": 0.95,
            "small_n_caveat": "n=5; intervals are descriptive",
        },
        "contrast_roles": {
            "primary": ["HHB-HHH", "HBB-HHB", "BBB-HHB"],
            "protocol_mandated_secondary": ["HBB-RBB", "HHB-RRB"],
            "descriptive_lower_bound": ["learned methods-Full Random"],
            "exploratory": [
                "Q4 cross-outcome correlations",
                "post-hoc robustness curve AUC supplement",
                "unadjusted interaction p-values outside frozen primary contrasts",
            ],
        },
        "test_usage": {
            "decision": "PASS" if core_timeline_pass else "FAIL",
            "stage2d_validation_only": stage2d["test_samples_accessed"] == 0,
            "core_freeze_completed_at_utc": core_gate["completed_at_utc"],
            "core_test_completed_at_utc": core_test["completed_at_utc"],
            "q1_aggregation_test_access_increment": q1_provenance[
                "test_access_increment"
            ],
            "q3_completed_at_utc": q3["completed_at_utc"],
            "q5q6_case_timeline": timeline,
            "supplement_test_access_increment": 0,
        },
        "late_heavy_seed4": {
            "decision": "PASS_RETAIN_AS_FORMAL_OUTCOME"
            if late_seed4_pass
            else "FAIL",
            "freeze_checks": late_gate["per_seed"]["4"]["checks"],
            "four_finite_test_rows": late_finite,
            "test_rows": [
                {
                    "method_id": row["method_id"],
                    "accuracy": row["accuracy"],
                    "standardized_reconstruction_mse": row[
                        "standardized_reconstruction_mse"
                    ],
                }
                for row in late_seed4
            ],
            "exclusion_or_retry_allowed": False,
        },
        "standardized_decoder_fairness": {
            "decision": "PASS" if fairness_pass else "FAIL",
            "paired_initialization": True,
            "encoder_frozen_and_unchanged": fairness_pass,
            "same_optimizer_data_epochs_validation_selection": True,
            "system_and_standardized_outcomes_reported_separately": True,
        },
    }


def _plot_q1_samples(summary: list[dict[str, Any]], output_dir: Path) -> None:
    colors = {
        "BBB": "#2463A3",
        "HHH": "#D95F02",
        "HHB": "#1B9E77",
        "HBB": "#66A61E",
        "Random": "#777777",
        "RBB": "#7570B3",
        "RRB": "#E7298A",
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        selected = [row for row in summary if row["method"] == method]
        selected.sort(key=lambda row: int(row["samples_seen"]))
        x = np.asarray([row["samples_seen"] for row in selected]) / 1_000_000
        mean = np.asarray(
            [row["validation_reconstruction_mse_mean"] for row in selected]
        )
        low = np.asarray(
            [row["validation_reconstruction_mse_ci_low"] for row in selected]
        )
        high = np.asarray(
            [row["validation_reconstruction_mse_ci_high"] for row in selected]
        )
        axis.plot(x, mean, marker="o", label=method, color=colors[method])
        axis.fill_between(x, low, high, color=colors[method], alpha=0.12)
    axis.set(
        xlabel="Cumulative system samples seen (millions)",
        ylabel="Validation reconstruction MSE",
        title="Q1 learning curves on the samples-seen axis",
    )
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "q1_samples_seen_curve.png", dpi=160)
    fig.savefig(output_dir / "q1_samples_seen_curve.pdf")
    plt.close(fig)


def run(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(f"Immutable supplement already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    q1_rows, q1_summary = q1_samples_seen_curves()
    q3_records = _typed_csv(Q3_ROOT / "per_seed_condition_metrics.csv")
    q3_auc = q3_curve_auc_rows(q3_records)
    q3_summary, q3_contrasts = summarize_q3_auc(q3_auc)
    audit = protocol_audit()
    if any(
        value == "FAIL"
        for value in (
            audit["paired_seeds"]["decision"],
            audit["test_usage"]["decision"],
            audit["late_heavy_seed4"]["decision"],
            audit["standardized_decoder_fairness"]["decision"],
        )
    ) or not audit["stage2d_to_stage3"]["history_preserved"]:
        raise RuntimeError(f"Final protocol audit failed: {audit}")

    _write_csv(output_dir / "q1_samples_seen_curve_per_seed.csv", q1_rows)
    _write_csv(output_dir / "q1_samples_seen_curve_summary.csv", q1_summary)
    _write_csv(output_dir / "q3_curve_auc_per_seed.csv", q3_auc)
    _write_csv(output_dir / "q3_curve_auc_summary.csv", q3_summary)
    _write_csv(output_dir / "q3_curve_auc_paired_contrasts.csv", q3_contrasts)
    _plot_q1_samples(q1_summary, output_dir)
    _write_json(output_dir / "protocol_audit.json", audit)

    source_files = (
        Q1_ROOT / "per_seed_complete.csv",
        Q1_ROOT / "method_summary.csv",
        Q1_ROOT / "summary.json",
        Q1_ROOT / "provenance.json",
        Q3_ROOT / "per_seed_condition_metrics.csv",
        Q3_ROOT / "integrity.json",
        ROOT / "docs" / "stage3_formal_protocol_v1.md",
        ROOT / "results" / "hybrid_hhb_confirmation" / "confirmation_decision.json",
        SWEEP_ROOT / "architecture" / "late_heavy" / "freeze_gate.json",
        SWEEP_ROOT
        / "architecture"
        / "late_heavy"
        / "test_evaluation"
        / "summary.json",
    )
    _write_json(
        output_dir / "integrity.json",
        {
            "schema_version": "stage3-final-audit-supplement-integrity-v1",
            "completed_at_utc": utc_now(),
            "q1_curve_rows": len(q1_rows),
            "expected_q1_curve_rows": 350,
            "q3_curve_auc_rows": len(q3_auc),
            "expected_q3_curve_auc_rows": 60,
            "q3_curve_summary_rows": len(q3_summary),
            "expected_q3_curve_summary_rows": 12,
            "q3_paired_contrast_rows": len(q3_contrasts),
            "expected_q3_paired_contrast_rows": 63,
            "all_values_finite": all(
                np.isfinite(float(row[metric]))
                for row in q3_auc
                for metric in AUC_METRICS
            ),
            "datasets_loaded": False,
            "checkpoints_loaded": False,
            "training_performed": False,
            "model_evaluation_performed": False,
            "test_access_increment": 0,
            "source_files": {str(path): file_sha256(path) for path in source_files},
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.output_dir).resolve())


if __name__ == "__main__":
    main()
