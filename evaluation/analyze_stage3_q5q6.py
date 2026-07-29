"""Aggregate the completed Q5/Q6 dimension and architecture experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import f as f_distribution

from evaluation.analyze_stage3_q1 import bootstrap_mean_ci
from evaluation.run_stage3_q2_representation import linear_cka
from training.run_stage3_q5q6_sweeps import METHODS, ROOT, SEEDS, validate_protocol
from utils.checkpointing import file_sha256, utc_now
from utils.results import write_json


LABELS = {
    "full_bp": "BBB",
    "full_hebbian": "HHH",
    "hybrid_hhb": "HHB",
    "hybrid_hbb": "HBB",
}
METHOD_LABELS = tuple(LABELS.values())
LAYERS = ("h1", "h2", "z")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value: str) -> float | int | str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return value
    return int(result) if result.is_integer() else result


def _typed(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {key: _number(value) for key, value in row.items()} for row in rows
    ]


def _case_dirs(protocol: dict[str, Any]) -> dict[tuple[str, str], Path]:
    output = Path(protocol["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    return {
        (sweep, case): output / sweep / case
        for sweep in ("dimension", "architecture")
        for case, spec in protocol[f"{sweep}_cases"].items()
        if spec["source"] == "new_formal_run"
    }


def _require_complete(protocol: dict[str, Any]) -> dict[tuple[str, str], Path]:
    directories = _case_dirs(protocol)
    for (sweep, case), directory in directories.items():
        gate = _json(directory / "freeze_gate.json")
        test = _json(directory / "test_evaluation" / "summary.json")
        representation = _json(directory / "representation" / "integrity.json")
        noise = _json(directory / "noise" / "integrity.json")
        checks = {
            "freeze": gate.get("decision") == "PASS",
            "test": test.get("records_complete") is True,
            "representation": (
                representation.get("record_count") == 20
                and representation.get("metric_rows") == 60
                and representation.get("all_checkpoints_unchanged") is True
            ),
            "noise": (
                noise.get("checkpoint_count") == 20
                and noise.get("metric_row_count") == 260
                and noise.get("all_components_unchanged") is True
            ),
        }
        if sweep == "architecture":
            update = _json(directory / "update_mechanisms" / "integrity.json")
            checks["update_mechanisms"] = (
                update.get("seed_count") == 5
                and update.get("formal_update_rows") == 90
                and update.get("all_source_files_unchanged") is True
                and update.get("analysis_optimizer_steps") == 0
                and update.get("test_samples_accessed") == 0
            )
        if not all(checks.values()):
            raise RuntimeError(f"Incomplete {sweep}/{case}: {checks}")
    return directories


def _performance_rows(
    protocol: dict[str, Any],
    directories: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    core_path = (
        ROOT
        / "results"
        / "formal"
        / "phase0_v1_1"
        / "stage3_core"
        / "test_evaluation"
        / "per_run_metrics.csv"
    )
    core = [
        row
        for row in _typed(_csv(core_path))
        if row["method_id"] in METHODS
    ]
    rows = []
    for sweep, case in (("dimension", "L64"), ("architecture", "balanced")):
        for row in core:
            rows.append(
                {
                    "sweep": sweep,
                    "case": case,
                    "seed": row["seed"],
                    "method_id": row["method_id"],
                    "method": LABELS[row["method_id"]],
                    **{
                        key: row[key]
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "classification_ce",
                            "system_reconstruction_mse",
                            "standardized_reconstruction_mse",
                        )
                    },
                }
            )
    for (sweep, case), directory in directories.items():
        for row in _typed(
            _csv(directory / "test_evaluation" / "per_run_metrics.csv")
        ):
            rows.append(
                {
                    "sweep": sweep,
                    "case": case,
                    "seed": row["seed"],
                    "method_id": row["method_id"],
                    "method": LABELS[row["method_id"]],
                    **{
                        key: row[key]
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "classification_ce",
                            "system_reconstruction_mse",
                            "standardized_reconstruction_mse",
                        )
                    },
                }
            )
    return rows


def _representation_rows(
    directories: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    core_path = (
        ROOT
        / "results"
        / "formal"
        / "phase0_v1_1"
        / "stage3_q2_representation"
        / "per_seed_layer_metrics.csv"
    )
    core = [
        row
        for row in _typed(_csv(core_path))
        if row["method_id"] in METHODS
    ]
    rows = []
    for sweep, case in (("dimension", "L64"), ("architecture", "balanced")):
        for row in core:
            rows.append({"sweep": sweep, "case": case, **row})
    for (sweep, case), directory in directories.items():
        rows.extend(
            _typed(
                _csv(
                    directory
                    / "representation"
                    / "per_seed_layer_metrics.csv"
                )
            )
        )
    return rows


def _noise_rows(
    directories: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    core_path = (
        ROOT
        / "results"
        / "formal"
        / "phase0_v1_1"
        / "stage3_q3_noise"
        / "per_seed_condition_metrics.csv"
    )
    core = [
        row
        for row in _typed(_csv(core_path))
        if row["method_id"] in METHODS
    ]
    rows = []
    for sweep, case in (("dimension", "L64"), ("architecture", "balanced")):
        for row in core:
            rows.append({"sweep": sweep, "case": case, **row})
    for (sweep, case), directory in directories.items():
        rows.extend(
            _typed(
                _csv(directory / "noise" / "per_seed_condition_metrics.csv")
            )
        )
    return rows


def _update_rows(
    directories: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    core_path = (
        ROOT
        / "results"
        / "formal"
        / "phase0_v1_1"
        / "stage3_q4_updates"
        / "per_seed_layer_update_metrics.csv"
    )
    rows = [
        {"sweep": "architecture", "case": "balanced", **row}
        for row in _typed(_csv(core_path))
    ]
    for case in ("early_heavy", "late_heavy"):
        rows.extend(
            {
                "sweep": "architecture",
                "case": case,
                **row,
            }
            for row in _typed(
                _csv(
                    directories[("architecture", case)]
                    / "update_mechanisms"
                    / "per_seed_layer_update_metrics.csv"
                )
            )
        )
    return rows


def _case_order(sweep: str) -> tuple[str, ...]:
    return (
        ("L16", "L32", "L64", "L128")
        if sweep == "dimension"
        else ("early_heavy", "balanced", "late_heavy")
    )


def _baseline(sweep: str) -> str:
    return "L64" if sweep == "dimension" else "balanced"


def _summaries(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    group_extra: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("sweep", "case", "method") + group_extra
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, selected in sorted(grouped.items()):
        record = dict(zip(keys, group))
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            ci = bootstrap_mean_ci(values)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
            record[f"{metric}_ci_low"] = ci[0]
            record[f"{metric}_ci_high"] = ci[1]
        output.append(record)
    return output


def _sensitivity_and_relative(
    summary: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    group_extra: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_keys = ("sweep", "method") + group_extra
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    sensitivity = []
    relative = []
    for group, rows in sorted(grouped.items()):
        base_record = dict(zip(group_keys, group))
        sweep = str(base_record["sweep"])
        indexed = {row["case"]: row for row in rows}
        baseline_case = _baseline(sweep)
        for metric in metrics:
            field = f"{metric}_mean"
            baseline = float(indexed[baseline_case][field])
            values = np.asarray(
                [float(indexed[case][field]) for case in _case_order(sweep)]
            )
            sensitivity.append(
                {
                    **base_record,
                    "metric": metric,
                    "baseline_case": baseline_case,
                    "baseline_value": baseline,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "sensitivity": float(
                        (values.max() - values.min())
                        / (abs(baseline) + 1e-12)
                    ),
                }
            )
            for case in _case_order(sweep):
                value = float(indexed[case][field])
                relative.append(
                    {
                        **base_record,
                        "case": case,
                        "metric": metric,
                        "value": value,
                        "baseline_case": baseline_case,
                        "baseline_value": baseline,
                        "absolute_change": value - baseline,
                        "relative_change": (value - baseline)
                        / (abs(baseline) + 1e-12),
                    }
                )
    return sensitivity, relative


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    interaction: bool,
) -> np.ndarray:
    methods = METHOD_LABELS
    cases = _case_order(str(rows[0]["sweep"]))
    columns = [np.ones(len(rows))]
    for seed in SEEDS[1:]:
        columns.append(np.asarray([row["seed"] == seed for row in rows]))
    for method in methods[1:]:
        columns.append(np.asarray([row["method"] == method for row in rows]))
    for case in cases[1:]:
        columns.append(np.asarray([row["case"] == case for row in rows]))
    if interaction:
        for method in methods[1:]:
            for case in cases[1:]:
                columns.append(
                    np.asarray(
                        [
                            row["method"] == method and row["case"] == case
                            for row in rows
                        ]
                    )
                )
    return np.column_stack(columns).astype(np.float64)


def _interaction_test(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    y = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    reduced = _design_matrix(rows, interaction=False)
    full = _design_matrix(rows, interaction=True)
    reduced_residual = y - reduced @ np.linalg.lstsq(
        reduced, y, rcond=None
    )[0]
    full_residual = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    sse_reduced = float(reduced_residual @ reduced_residual)
    sse_full = float(full_residual @ full_residual)
    rank_reduced = int(np.linalg.matrix_rank(reduced))
    rank_full = int(np.linalg.matrix_rank(full))
    numerator_df = rank_full - rank_reduced
    denominator_df = len(rows) - rank_full
    numerator = max(sse_reduced - sse_full, 0.0) / numerator_df
    denominator = sse_full / denominator_df
    statistic = numerator / max(denominator, 1e-30)
    return {
        "metric": metric,
        "f_statistic": statistic,
        "numerator_df": numerator_df,
        "denominator_df": denominator_df,
        "p_value": float(
            f_distribution.sf(statistic, numerator_df, denominator_df)
        ),
        "model": "seed_block + method + case + method_x_case",
    }


def _paired_contrasts(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    output = []
    contrasts = (
        ("HHB_minus_HHH", "HHB", "HHH"),
        ("HBB_minus_HHB", "HBB", "HHB"),
        ("BBB_minus_HHB", "BBB", "HHB"),
    )
    for sweep in ("dimension", "architecture"):
        for case in _case_order(sweep):
            selected = [
                row
                for row in rows
                if row["sweep"] == sweep and row["case"] == case
            ]
            indexed = {
                (row["seed"], row["method"]): row for row in selected
            }
            for name, left, right in contrasts:
                for metric in metrics:
                    differences = np.asarray(
                        [
                            float(indexed[(seed, left)][metric])
                            - float(indexed[(seed, right)][metric])
                            for seed in SEEDS
                        ]
                    )
                    ci = bootstrap_mean_ci(differences)
                    output.append(
                        {
                            "sweep": sweep,
                            "case": case,
                            "contrast": name,
                            "metric": metric,
                            "mean_paired_difference": float(
                                differences.mean()
                            ),
                            "sd_paired_difference": float(
                                differences.std(ddof=1)
                            ),
                            "ci_low": ci[0],
                            "ci_high": ci[1],
                        }
                    )
    return output


def _compensation(
    representation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            row["sweep"],
            row["case"],
            row["seed"],
            row["method"],
            row["layer"],
        ): row
        for row in representation
    }
    output = []
    for sweep in ("dimension", "architecture"):
        for case in _case_order(sweep):
            for seed in SEEDS:
                for method in METHOD_LABELS:
                    h2 = indexed[(sweep, case, seed, method, "h2")]
                    z = indexed[(sweep, case, seed, method, "z")]
                    output.append(
                        {
                            "sweep": sweep,
                            "case": case,
                            "seed": seed,
                            "method": method,
                            "effective_rank_z_minus_h2": float(
                                z["effective_rank"]
                            )
                            - float(h2["effective_rank"]),
                            "effective_rank_z_over_h2": float(
                                z["effective_rank"]
                            )
                            / max(float(h2["effective_rank"]), 1e-12),
                            "linear_probe_z_minus_h2": float(
                                z["linear_probe_cv_accuracy"]
                            )
                            - float(h2["linear_probe_cv_accuracy"]),
                            "separability_z_over_h2": float(
                                z["between_within_scatter_ratio"]
                            )
                            / max(
                                float(
                                    h2["between_within_scatter_ratio"]
                                ),
                                1e-12,
                            ),
                        }
                    )
    return output


def _embedding_path(
    directories: dict[tuple[str, str], Path],
    *,
    case: str,
    seed: int,
    method: str,
    layer: str,
) -> Path:
    if case == "balanced":
        return (
            ROOT
            / "results"
            / "formal"
            / "phase0_v1_1"
            / "stage3_q2_representation"
            / "embeddings"
            / f"seed_{seed}_{method}_{layer}_pca.npz"
        )
    return (
        directories[("architecture", case)]
        / "representation"
        / "embeddings"
        / f"seed_{seed}_{method}_{layer}_pca.npz"
    )


def _architecture_cka(
    directories: dict[tuple[str, str], Path],
) -> list[dict[str, Any]]:
    output = []
    for case in ("early_heavy", "late_heavy"):
        for method in METHOD_LABELS:
            for layer in LAYERS:
                values = []
                for seed in SEEDS:
                    with np.load(
                        _embedding_path(
                            directories,
                            case=case,
                            seed=seed,
                            method=method,
                            layer=layer,
                        )
                    ) as candidate:
                        left = candidate["embedding"]
                    with np.load(
                        _embedding_path(
                            directories,
                            case="balanced",
                            seed=seed,
                            method=method,
                            layer=layer,
                        )
                    ) as baseline:
                        right = baseline["embedding"]
                    values.append(linear_cka(left, right))
                output.append(
                    {
                        "case": case,
                        "reference_case": "balanced",
                        "method": method,
                        "layer": layer,
                        "mean_pca50_linear_cka": float(np.mean(values)),
                        "sd_pca50_linear_cka": float(
                            np.std(values, ddof=1)
                        ),
                    }
                )
    return output


def _plot_interactions(
    output_dir: Path,
    summary: list[dict[str, Any]],
    *,
    metric: str,
    filename: str,
    ylabel: str,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, sweep in zip(axes, ("dimension", "architecture")):
        cases = _case_order(sweep)
        for method in METHOD_LABELS:
            indexed = {
                row["case"]: row
                for row in summary
                if row["sweep"] == sweep and row["method"] == method
            }
            axis.errorbar(
                np.arange(len(cases)),
                [indexed[case][f"{metric}_mean"] for case in cases],
                yerr=[indexed[case][f"{metric}_sd"] for case in cases],
                marker="o",
                capsize=3,
                label=method,
            )
        axis.set_xticks(np.arange(len(cases)), cases, rotation=20)
        axis.set_title(sweep)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / f"{filename}.png", dpi=160)
    fig.savefig(figures / f"{filename}.pdf")
    plt.close(fig)


def run(protocol_path: str | Path) -> Path:
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    directories = _require_complete(protocol)
    output_root = Path(protocol["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root / "analysis"
    if output_dir.exists():
        raise FileExistsError("Immutable Q5/Q6 analysis already exists")
    output_dir.mkdir(parents=True)

    performance = _performance_rows(protocol, directories)
    representation = _representation_rows(directories)
    noise = _noise_rows(directories)
    updates = _update_rows(directories)
    performance_metrics = (
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    )
    performance_summary = _summaries(
        performance, metrics=performance_metrics
    )
    representation_metrics = (
        "effective_rank",
        "stable_rank",
        "winner_coverage_ratio",
        "winner_entropy",
        "max_winner_share",
        "linear_probe_cv_accuracy",
        "knn_cv_accuracy",
        "between_within_scatter_ratio",
        "silhouette_score",
    )
    representation_summary = _summaries(
        representation,
        metrics=representation_metrics,
        group_extra=("layer",),
    )
    severe_noise = [
        row
        for row in noise
        if row["noise_type"] != "clean"
        and float(row["severity"]) == 0.4
    ]
    noise_metrics = (
        "accuracy",
        "accuracy_absolute_degradation",
        "system_reconstruction_mse_increase",
        "standardized_reconstruction_mse_increase",
        "representation_cosine",
        "prediction_js_divergence",
    )
    noise_summary = _summaries(
        severe_noise,
        metrics=noise_metrics,
        group_extra=("noise_type",),
    )
    update_metrics = (
        "alignment",
        "norm_ratio",
        "alpha_star",
        "scale_matched_bias",
        "update_snr_linear",
        "matched_bp_snr_linear",
    )
    update_summary = _summaries(
        updates,
        metrics=update_metrics,
        group_extra=("layer", "rule"),
    )
    performance_sensitivity, performance_relative = (
        _sensitivity_and_relative(
            performance_summary, metrics=performance_metrics
        )
    )
    representation_sensitivity, representation_relative = (
        _sensitivity_and_relative(
            representation_summary,
            metrics=representation_metrics,
            group_extra=("layer",),
        )
    )
    noise_sensitivity, noise_relative = _sensitivity_and_relative(
        noise_summary,
        metrics=noise_metrics,
        group_extra=("noise_type",),
    )
    update_sensitivity, update_relative = _sensitivity_and_relative(
        update_summary,
        metrics=update_metrics,
        group_extra=("layer", "rule"),
    )
    interactions = []
    for sweep in ("dimension", "architecture"):
        selected = [
            row for row in performance if row["sweep"] == sweep
        ]
        for metric in performance_metrics:
            interactions.append(
                {
                    "sweep": sweep,
                    "domain": "performance",
                    **_interaction_test(selected, metric),
                }
            )
        for layer in LAYERS:
            selected = [
                row
                for row in representation
                if row["sweep"] == sweep and row["layer"] == layer
            ]
            for metric in (
                "effective_rank",
                "linear_probe_cv_accuracy",
                "between_within_scatter_ratio",
                "winner_entropy",
            ):
                interactions.append(
                    {
                        "sweep": sweep,
                        "domain": f"representation_{layer}",
                        **_interaction_test(selected, metric),
                    }
                )
        for noise_type in ("gaussian", "salt_pepper", "pixel_masking"):
            selected = [
                row
                for row in severe_noise
                if row["sweep"] == sweep
                and row["noise_type"] == noise_type
            ]
            for metric in (
                "accuracy",
                "accuracy_absolute_degradation",
            ):
                interactions.append(
                    {
                        "sweep": sweep,
                        "domain": f"noise_{noise_type}",
                        **_interaction_test(selected, metric),
                    }
                )

    compensation = _compensation(representation)
    architecture_cka = _architecture_cka(directories)
    representation_lookup = {
        (
            row["case"],
            row["seed"],
            row["method"],
            row["layer"],
        ): row
        for row in representation
        if row["sweep"] == "architecture"
    }
    update_representation_join = []
    for row in updates:
        key = (row["case"], row["seed"], row["method"], row["layer"])
        representation_row = representation_lookup[key]
        update_representation_join.append(
            {
                "case": row["case"],
                "seed": row["seed"],
                "method": row["method"],
                "layer": row["layer"],
                "rule": row["rule"],
                "alignment": row["alignment"],
                "update_snr_linear": row["update_snr_linear"],
                "winner_entropy": representation_row["winner_entropy"],
                "winner_coverage_ratio": representation_row[
                    "winner_coverage_ratio"
                ],
                "effective_rank": representation_row["effective_rank"],
                "linear_probe_cv_accuracy": representation_row[
                    "linear_probe_cv_accuracy"
                ],
            }
        )
    _write_csv(output_dir / "performance_per_seed.csv", performance)
    _write_csv(output_dir / "performance_summary.csv", performance_summary)
    _write_csv(output_dir / "representation_per_seed_layer.csv", representation)
    _write_csv(
        output_dir / "representation_summary.csv", representation_summary
    )
    _write_csv(output_dir / "noise_severity_0_4_per_seed.csv", severe_noise)
    _write_csv(output_dir / "noise_severity_0_4_summary.csv", noise_summary)
    _write_csv(output_dir / "architecture_update_per_seed.csv", updates)
    _write_csv(output_dir / "architecture_update_summary.csv", update_summary)
    _write_csv(
        output_dir / "sensitivity.csv",
        performance_sensitivity
        + representation_sensitivity
        + noise_sensitivity
        + update_sensitivity,
    )
    _write_csv(
        output_dir / "relative_to_baseline.csv",
        performance_relative
        + representation_relative
        + noise_relative
        + update_relative,
    )
    _write_csv(output_dir / "interaction_tests.csv", interactions)
    _write_csv(
        output_dir / "paired_performance_contrasts.csv",
        _paired_contrasts(performance, performance_metrics),
    )
    _write_csv(output_dir / "compensation_metrics.csv", compensation)
    _write_csv(
        output_dir / "architecture_cross_case_cka.csv", architecture_cka
    )
    _write_csv(
        output_dir / "architecture_update_representation_join.csv",
        update_representation_join,
    )
    _plot_interactions(
        output_dir,
        performance_summary,
        metric="accuracy",
        filename="classification_interactions",
        ylabel="Test accuracy",
    )
    _plot_interactions(
        output_dir,
        performance_summary,
        metric="standardized_reconstruction_mse",
        filename="standardized_reconstruction_interactions",
        ylabel="Standardized decoder MSE",
    )
    z_summary = [
        row for row in representation_summary if row["layer"] == "z"
    ]
    _plot_interactions(
        output_dir,
        z_summary,
        metric="effective_rank",
        filename="z_effective_rank_interactions",
        ylabel="z effective rank",
    )
    source_files = [
        directory / component
        for directory in directories.values()
        for component in (
            "freeze_gate.json",
            "test_evaluation/summary.json",
            "representation/integrity.json",
            "noise/integrity.json",
        )
    ]
    source_files.extend(
        directory / "update_mechanisms" / "integrity.json"
        for (sweep, _), directory in directories.items()
        if sweep == "architecture"
    )
    write_json(
        output_dir,
        "integrity.json",
        {
            "schema_version": "stage3-q5q6-analysis-integrity-v1",
            "completed_at_utc": utc_now(),
            "performance_rows": len(performance),
            "expected_performance_rows": 140,
            "representation_rows": len(representation),
            "expected_representation_rows": 420,
            "severe_noise_rows": len(severe_noise),
            "expected_severe_noise_rows": 420,
            "architecture_update_rows": len(updates),
            "expected_architecture_update_rows": 270,
            "all_values_finite": all(
                np.isfinite(float(row[metric]))
                for row in performance
                for metric in performance_metrics
            )
            and all(
                np.isfinite(float(row[metric]))
                for row in representation
                for metric in representation_metrics
            )
            and all(
                np.isfinite(float(row[metric]))
                for row in severe_noise
                for metric in noise_metrics
            ),
            "source_files": {
                str(path): file_sha256(path) for path in source_files
            },
            "performance_gate_applied": False,
            "test_used_for_selection": False,
        },
        overwrite=True,
    )
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-q5q6-analysis-run-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "methods": LABELS,
            "seeds": list(SEEDS),
            "dimension_cases": list(_case_order("dimension")),
            "architecture_cases": list(_case_order("architecture")),
            "training_performed": False,
        },
        overwrite=True,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/stage3_q5q6_sweeps_v1.yaml",
    )
    args = parser.parse_args()
    print(run(args.config).resolve())


if __name__ == "__main__":
    main()
