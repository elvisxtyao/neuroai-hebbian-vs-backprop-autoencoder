"""Build the v1.0-final publication figures from the compact evidence bundle only.

This script is deliberately artifact-only. It verifies the Phase 1 release bundle,
reads only small accepted CSV tables beneath release/v1.0-final, and refuses to
overwrite an existing output directory. It never imports training/model/dataset code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "v1.0-final"
DEFAULT_OUTPUT = ROOT / "figures" / "final"
FORMAL_SEEDS = {0, 1, 2, 3, 4}
METHOD_ORDER = ["BBB", "HBB", "HHB", "HHH", "RBB", "RRB", "Random"]
CORE_METHODS = ["BBB", "HBB", "HHB", "HHH"]
FIG1_METHODS = ["BBB", "HBB", "HHB", "HHH", "Random"]
COLORS = {
    "BBB": "#0072B2",
    "HBB": "#009E73",
    "HHB": "#E69F00",
    "HHH": "#D55E00",
    "RBB": "#CC79A7",
    "RRB": "#56B4E9",
    "Random": "#666666",
}
MARKERS = {
    "BBB": "o",
    "HBB": "D",
    "HHB": "s",
    "HHH": "^",
    "RBB": "P",
    "RRB": "X",
    "Random": "v",
}
LINESTYLES = {
    "BBB": "-",
    "HBB": "-.",
    "HHB": "--",
    "HHH": ":",
    "RBB": "-.",
    "RRB": "--",
    "Random": ":",
}

SOURCES = {
    "q1_seed": "data/q1/per_seed_complete.csv",
    "q1_summary": "data/q1/method_summary.csv",
    "q1_contrasts": "data/q1/paired_contrasts.csv",
    "q1_curve_seed": "data/audit/q1_samples_seen_curve_per_seed.csv",
    "q1_curve_summary": "data/audit/q1_samples_seen_curve_summary.csv",
    "q2_seed": "data/q2/per_seed_layer_metrics.csv",
    "q2_summary": "data/q2/method_layer_summary.csv",
    "q3_seed": "data/q3/per_seed_condition_metrics.csv",
    "q3_summary": "data/q3/condition_summary.csv",
    "q4_seed": "data/q4/per_seed_layer_update_metrics.csv",
    "q4_summary": "data/q4/method_layer_update_summary.csv",
    "q56_perf_seed": "data/q5q6/performance_per_seed.csv",
    "q56_perf_summary": "data/q5q6/performance_summary.csv",
    "q56_repr_seed": "data/q5q6/representation_per_seed_layer.csv",
    "q56_repr_summary": "data/q5q6/representation_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(relative: str) -> list[dict[str, str]]:
    path = RELEASE / relative
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty plotted-values table: {path.name}")
    selected = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def seed(row: dict[str, str]) -> int:
    return int(row["seed"])


def validate_seed_matrix(rows: Iterable[dict[str, str]], group_fields: Sequence[str]) -> None:
    groups: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].add(seed(row))
    incomplete = {group: values for group, values in groups.items() if values != FORMAL_SEEDS}
    if incomplete:
        raise ValueError(f"Incomplete formal seed groups: {incomplete}")


def verify_release() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    sys.path.insert(0, str(ROOT))
    from scripts.verify_final_release import verify

    result = verify(RELEASE)
    if result["formal_seeds"] != sorted(FORMAL_SEEDS) or result["test_access_increment"] != 0:
        raise ValueError("Release verifier returned an unexpected protocol boundary")
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    source_records = {record["bundle_path"]: record for record in manifest["bundle_files"]}
    missing = sorted(set(SOURCES.values()) - set(source_records))
    if missing:
        raise ValueError(f"Figure source absent from Phase 1 manifest: {missing}")
    for relative in SOURCES.values():
        if sha256(RELEASE / relative) != source_records[relative]["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    return manifest, source_records


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def legend_handles(methods: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.7,
            markersize=4.5,
            label=method,
        )
        for method in methods
    ]


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=360, metadata={"Software": "Microlearning v1.0-final artifact-only figure builder"})
    fig.savefig(
        pdf,
        metadata={
            "Title": stem,
            "Author": "Microlearning final release",
            "Creator": "scripts/plot_final_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def jittered_x(base: float, seed_value: int, width: float = 0.10) -> float:
    return base + (seed_value - 2) * width / 2


def figure1(output: Path, values: Path) -> dict[str, object]:
    full_per_seed = read_csv(SOURCES["q1_seed"])
    full_summary = read_csv(SOURCES["q1_summary"])
    validate_seed_matrix(full_per_seed, ["method"])
    if len(full_per_seed) != 35 or {row["method"] for row in full_per_seed} != set(METHOD_ORDER):
        raise ValueError("Q1 formal matrix is not the frozen 35-row design")
    per_seed = [row for row in full_per_seed if row["method"] in FIG1_METHODS]
    summary = [row for row in full_summary if row["method"] in FIG1_METHODS]
    write_csv(values / "fig1_per_seed.csv", per_seed)
    write_csv(values / "fig1_summary.csv", summary)
    summary_by_method = {row["method"]: row for row in summary}
    seed_by_method = defaultdict(list)
    for row in per_seed:
        seed_by_method[row["method"]].append(row)

    metrics = [
        ("accuracy", "Test accuracy", False),
        ("standardized_reconstruction_mse", "Standardized decoder MSE", True),
        ("system_reconstruction_mse", "System reconstruction MSE", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), constrained_layout=True)
    for idx, (metric, ylabel, log_scale) in enumerate(metrics):
        ax = axes[idx]
        for x, method in enumerate(FIG1_METHODS):
            rows = sorted(seed_by_method[method], key=seed)
            ys = [f(row, metric) for row in rows]
            ax.scatter(
                [jittered_x(x, seed(row)) for row in rows],
                ys,
                s=22,
                facecolor="white",
                edgecolor=COLORS[method],
                marker=MARKERS[method],
                linewidth=0.9,
                zorder=3,
            )
            summary_row = summary_by_method[method]
            mean = f(summary_row, f"{metric}_mean")
            low = f(summary_row, f"{metric}_ci_low")
            high = f(summary_row, f"{metric}_ci_high")
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - low], [high - mean]],
                color=COLORS[method],
                marker=MARKERS[method],
                markerfacecolor=COLORS[method],
                markeredgecolor="white",
                markersize=6,
                capsize=3,
                linewidth=1.8,
                zorder=4,
            )
        ax.set_xticks(range(len(FIG1_METHODS)), FIG1_METHODS, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.76, 0.93)
        panel_label(ax, chr(ord("A") + idx))
    save_figure(fig, output, "fig1_performance")
    return {"sources": [SOURCES["q1_seed"], SOURCES["q1_summary"]], "plotted_rows": [25, 5]}


def figure2(output: Path, values: Path) -> dict[str, object]:
    q1_seed = read_csv(SOURCES["q1_seed"])
    contrasts = read_csv(SOURCES["q1_contrasts"])
    curve_seed = [row for row in read_csv(SOURCES["q1_curve_seed"]) if row["method"] in CORE_METHODS]
    curve_summary = [row for row in read_csv(SOURCES["q1_curve_summary"]) if row["method"] in CORE_METHODS]
    if len(curve_seed) != 200:
        raise ValueError("Complete frozen trajectory data are unavailable for the four formal methods")
    validate_seed_matrix(curve_seed, ["method", "samples_seen"])
    counts = defaultdict(int)
    for row in curve_seed:
        counts[(row["method"], seed(row))] += 1
    if set(counts.values()) != {10} or len(counts) != 20:
        raise ValueError("Samples-seen trajectories must contain exactly ten observed points per method/seed")

    per_lookup = {(row["method"], seed(row)): row for row in q1_seed}
    pairs = [("HBB - RBB", "HBB", "RBB", "HBB_minus_RBB"), ("HHB - RRB", "HHB", "RRB", "HHB_minus_RRB")]
    paired_rows: list[dict[str, object]] = []
    for label, left, right, contrast_name in pairs:
        for seed_value in sorted(FORMAL_SEEDS):
            paired_rows.append(
                {
                    "contrast": contrast_name,
                    "label": label,
                    "seed": seed_value,
                    "left": left,
                    "right": right,
                    "accuracy_difference": f(per_lookup[(left, seed_value)], "accuracy")
                    - f(per_lookup[(right, seed_value)], "accuracy"),
                }
            )
    paired_summary = [
        row
        for row in contrasts
        if row["contrast"] in {pair[3] for pair in pairs} and row["metric"] == "accuracy"
    ]
    if len(paired_summary) != 2:
        raise ValueError("Frozen paired prefix-control accuracy contrasts are incomplete")
    cost_rows = [row for row in q1_seed if row["method"] in CORE_METHODS]
    write_csv(values / "fig2_paired_seed_effects.csv", paired_rows)
    write_csv(values / "fig2_paired_summary.csv", paired_summary)
    write_csv(values / "fig2_samples_seen_per_seed.csv", curve_seed)
    write_csv(values / "fig2_samples_seen_summary.csv", curve_summary)
    write_csv(values / "fig2_training_cost_per_seed.csv", cost_rows)

    fig = plt.figure(figsize=(11.2, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.45, 1.0])
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]

    summary_lookup = {row["contrast"]: row for row in paired_summary}
    for x, (_, _, _, contrast_name) in enumerate(pairs):
        rows = [row for row in paired_rows if row["contrast"] == contrast_name]
        color = COLORS["HBB"] if x == 0 else COLORS["HHB"]
        marker = MARKERS["HBB"] if x == 0 else MARKERS["HHB"]
        ax_a.scatter(
            [jittered_x(x, int(row["seed"]), 0.14) for row in rows],
            [float(row["accuracy_difference"]) for row in rows],
            facecolor="white",
            edgecolor=color,
            marker=marker,
            s=26,
            linewidth=1,
            zorder=3,
        )
        srow = summary_lookup[contrast_name]
        mean, low, high = f(srow, "mean_difference"), f(srow, "ci_low"), f(srow, "ci_high")
        ax_a.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color=color, marker=marker, capsize=4, linewidth=2)
    ax_a.axhline(0, color="#333333", linewidth=0.8)
    ax_a.set_xticks([0, 1], ["HBB - RBB", "HHB - RRB"], rotation=15, ha="right")
    ax_a.set_ylabel("Paired accuracy difference")
    panel_label(ax_a, "A")

    curve_summary_lookup = {(row["method"], row["samples_seen"]): row for row in curve_summary}
    for method in CORE_METHODS:
        for seed_value in sorted(FORMAL_SEEDS):
            rows = sorted(
                [row for row in curve_seed if row["method"] == method and seed(row) == seed_value],
                key=lambda row: f(row, "samples_seen"),
            )
            ax_b.plot(
                [f(row, "samples_seen") / 1e6 for row in rows],
                [f(row, "validation_reconstruction_mse") for row in rows],
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=0.65,
                alpha=0.18,
            )
        rows = sorted([row for row in curve_summary if row["method"] == method], key=lambda row: f(row, "samples_seen"))
        xs = np.array([f(row, "samples_seen") / 1e6 for row in rows])
        means = np.array([f(row, "validation_reconstruction_mse_mean") for row in rows])
        lows = np.array([f(row, "validation_reconstruction_mse_ci_low") for row in rows])
        highs = np.array([f(row, "validation_reconstruction_mse_ci_high") for row in rows])
        ax_b.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.10, linewidth=0)
        ax_b.plot(xs, means, color=COLORS[method], linestyle=LINESTYLES[method], marker=MARKERS[method], markersize=3.2, linewidth=1.8)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Observed system samples seen (millions)")
    ax_b.set_ylabel("Validation reconstruction MSE")
    panel_label(ax_b, "B")
    ax_b.legend(handles=legend_handles(CORE_METHODS), ncol=2, loc="upper right")

    for x, method in enumerate(CORE_METHODS):
        rows = sorted([row for row in cost_rows if row["method"] == method], key=seed)
        xs = [f(row, "system_samples_seen") / 1e6 for row in rows]
        ys = [f(row, "system_wall_time_sec") / 60 for row in rows]
        for row, x_value, y_value in zip(rows, xs, ys):
            ax_c.scatter(
                x_value,
                y_value,
                s=27,
                facecolor="white",
                edgecolor=COLORS[method],
                marker=MARKERS[method],
                linewidth=1,
                zorder=3,
            )
        ax_c.scatter(np.mean(xs), np.mean(ys), s=55, color=COLORS[method], marker=MARKERS[method], edgecolor="white", linewidth=0.7, zorder=4)
        ax_c.text(np.mean(xs) + 0.035, np.mean(ys), method, color=COLORS[method], fontsize=7.5, va="center")
    ax_c.set_xlabel("System samples seen (millions)")
    ax_c.set_ylabel("System wall-clock (minutes)")
    ax_c.set_xlim(0.35, 2.25)
    panel_label(ax_c, "C")
    save_figure(fig, output, "fig2_prefix_value_training_cost")
    return {
        "sources": [SOURCES["q1_seed"], SOURCES["q1_contrasts"], SOURCES["q1_curve_seed"], SOURCES["q1_curve_summary"]],
        "plotted_rows": [10, 2, 200, len(curve_summary), len(cost_rows)],
        "trajectory_complete": True,
        "interpolation_performed": False,
    }


def figure3(output: Path, values: Path) -> dict[str, object]:
    per_seed = [row for row in read_csv(SOURCES["q2_seed"]) if row["method"] in CORE_METHODS]
    summary = [row for row in read_csv(SOURCES["q2_summary"]) if row["method"] in CORE_METHODS]
    if len(per_seed) != 60 or len(summary) != 12:
        raise ValueError("Q2 core layer matrix is incomplete")
    validate_seed_matrix(per_seed, ["method", "layer"])
    write_csv(values / "fig3_per_seed.csv", per_seed)
    write_csv(values / "fig3_summary.csv", summary)
    layers = ["h1", "h2", "z"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.35), constrained_layout=True)
    metrics = [("effective_rank", "Effective rank"), ("linear_probe_cv_accuracy", "Frozen linear-probe CV accuracy")]
    for panel, (metric, ylabel) in enumerate(metrics):
        ax = axes[panel]
        for method in CORE_METHODS:
            for seed_value in sorted(FORMAL_SEEDS):
                rows = {row["layer"]: row for row in per_seed if row["method"] == method and seed(row) == seed_value}
                ax.plot(x, [f(rows[layer], metric) for layer in layers], color=COLORS[method], linestyle=LINESTYLES[method], linewidth=0.7, alpha=0.18)
                ax.scatter(x, [f(rows[layer], metric) for layer in layers], color=COLORS[method], marker=MARKERS[method], s=9, alpha=0.20)
            rows = {row["layer"]: row for row in summary if row["method"] == method}
            means = [f(rows[layer], f"{metric}_mean") for layer in layers]
            sds = [f(rows[layer], f"{metric}_sd") for layer in layers]
            ax.errorbar(
                x,
                means,
                yerr=sds,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markerfacecolor="white",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.8,
                zorder=4,
            )
        ax.set_xticks(x, ["h1", "h2", "z"])
        ax.set_xlabel("Encoder representation")
        ax.set_ylabel(ylabel)
        if metric == "linear_probe_cv_accuracy":
            ax.set_ylim(0.82, 0.94)
        panel_label(ax, chr(ord("A") + panel))
    axes[0].legend(handles=legend_handles(CORE_METHODS), loc="upper left", ncol=2)
    save_figure(fig, output, "fig3_layerwise_representation")
    return {"sources": [SOURCES["q2_seed"], SOURCES["q2_summary"]], "plotted_rows": [60, 12], "uncertainty": "mean +/- sample SD"}


def figure4(output: Path, values: Path) -> dict[str, object]:
    per_seed = [
        row
        for row in read_csv(SOURCES["q4_seed"])
        if row["method"] == "HHH" and row["rule"] == "hebbian_effective"
    ]
    summary = [
        row
        for row in read_csv(SOURCES["q4_summary"])
        if row["method"] == "HHH" and row["rule"] == "hebbian_effective"
    ]
    if len(per_seed) != 15 or len(summary) != 3:
        raise ValueError("Canonical HHH effective-update rows are incomplete")
    validate_seed_matrix(per_seed, ["method", "layer", "rule"])
    write_csv(values / "fig4_per_seed.csv", per_seed)
    write_csv(values / "fig4_summary.csv", summary)
    layers = ["enc1", "enc2", "enc3"]
    x = np.arange(3)
    metrics = [
        ("alignment", "Cosine alignment", False),
        ("scale_matched_bias", "Scale-matched bias", False),
        ("norm_ratio", "Norm ratio to matched BP", True),
        ("update_snr_linear", "Update SNR (linear)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    summary_lookup = {row["layer"]: row for row in summary}
    for panel, (metric, ylabel, log_scale) in enumerate(metrics):
        ax = axes.flat[panel]
        for layer_index, layer in enumerate(layers):
            rows = sorted([row for row in per_seed if row["layer"] == layer], key=seed)
            ax.scatter(
                [jittered_x(layer_index, seed(row), 0.14) for row in rows],
                [f(row, metric) for row in rows],
                s=25,
                facecolor="white",
                edgecolor=COLORS["HHH"],
                marker=MARKERS["HHH"],
                linewidth=1,
                zorder=3,
            )
            srow = summary_lookup[layer]
            mean, sd = f(srow, f"{metric}_mean"), f(srow, f"{metric}_sd")
            lower = max(mean - sd, np.finfo(float).tiny) if log_scale else mean - sd
            ax.errorbar(
                layer_index,
                mean,
                yerr=[[mean - lower], [sd]],
                color=COLORS["HHH"],
                marker=MARKERS["HHH"],
                capsize=3,
                linewidth=1.8,
                zorder=4,
            )
        ax.plot(x, [f(summary_lookup[layer], f"{metric}_mean") for layer in layers], color=COLORS["HHH"], linestyle=LINESTYLES["HHH"], linewidth=1.3)
        ax.set_xticks(x, ["Enc1", "Enc2", "Enc3"])
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        if metric in {"alignment", "scale_matched_bias"}:
            ax.set_ylim(-0.08, 1.08)
        panel_label(ax, chr(ord("A") + panel))
    save_figure(fig, output, "fig4_update_mechanism")
    return {
        "sources": [SOURCES["q4_seed"], SOURCES["q4_summary"]],
        "plotted_rows": [15, 3],
        "canonical_path": "HHH hebbian_effective only; shared HBB/HHB prefix duplicates omitted",
        "uncertainty": "mean +/- sample SD",
    }


def figure5(output: Path, values: Path) -> dict[str, object]:
    all_seed = read_csv(SOURCES["q3_seed"])
    all_summary = read_csv(SOURCES["q3_summary"])
    clean_seed = {(row["method"], seed(row)): row for row in all_seed if row["noise_type"] == "clean" and row["method"] in CORE_METHODS}
    clean_summary = {row["method"]: row for row in all_summary if row["noise_type"] == "clean" and row["method"] in CORE_METHODS}
    noise_types = ["gaussian", "salt_pepper", "pixel_masking"]
    plotted_seed: list[dict[str, object]] = []
    plotted_summary: list[dict[str, object]] = []
    for noise_type in noise_types:
        for method in CORE_METHODS:
            for seed_value in sorted(FORMAL_SEEDS):
                source = clean_seed[(method, seed_value)]
                plotted_seed.append({**source, "noise_type": noise_type, "severity": "0.0", "source_condition": "clean"})
            source_summary = clean_summary[method]
            plotted_summary.append({**source_summary, "noise_type": noise_type, "severity": "0.0", "source_condition": "clean"})
        plotted_seed.extend(
            {**row, "source_condition": row["noise_type"]}
            for row in all_seed
            if row["method"] in CORE_METHODS and row["noise_type"] == noise_type
        )
        plotted_summary.extend(
            {**row, "source_condition": row["noise_type"]}
            for row in all_summary
            if row["method"] in CORE_METHODS and row["noise_type"] == noise_type
        )
    validate_seed_matrix(plotted_seed, ["method", "noise_type", "severity"])
    write_csv(values / "fig5_per_seed_curves.csv", plotted_seed)
    write_csv(values / "fig5_summary_curves.csv", plotted_summary)

    labels = {"gaussian": "Gaussian noise", "salt_pepper": "Salt-and-pepper", "pixel_masking": "Pixel masking"}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), constrained_layout=True, sharey=True)
    for panel, noise_type in enumerate(noise_types):
        ax = axes[panel]
        for method in CORE_METHODS:
            for seed_value in sorted(FORMAL_SEEDS):
                rows = sorted(
                    [row for row in plotted_seed if row["noise_type"] == noise_type and row["method"] == method and seed(row) == seed_value],
                    key=lambda row: f(row, "severity"),
                )
                ax.plot(
                    [f(row, "severity") for row in rows],
                    [f(row, "accuracy") for row in rows],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=0.65,
                    alpha=0.17,
                )
            rows = sorted(
                [row for row in plotted_summary if row["noise_type"] == noise_type and row["method"] == method],
                key=lambda row: f(row, "severity"),
            )
            xs = np.array([f(row, "severity") for row in rows])
            means = np.array([f(row, "accuracy_mean") for row in rows])
            lows = np.array([f(row, "accuracy_ci_low") for row in rows])
            highs = np.array([f(row, "accuracy_ci_high") for row in rows])
            ax.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.10, linewidth=0)
            ax.plot(xs, means, color=COLORS[method], linestyle=LINESTYLES[method], marker=MARKERS[method], markersize=3.5, linewidth=1.8)
        ax.set_xlabel("Corruption severity")
        ax.set_title(labels[noise_type], pad=5)
        ax.set_ylim(0.0, 0.96)
        panel_label(ax, chr(ord("A") + panel))
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(handles=legend_handles(CORE_METHODS), loc="lower left", ncol=2)
    save_figure(fig, output, "fig5_robustness")
    return {
        "sources": [SOURCES["q3_seed"], SOURCES["q3_summary"]],
        "plotted_rows": [len(plotted_seed), len(plotted_summary)],
        "clean_rows_reused_at_severity_zero": True,
    }


def figure6(output: Path, values: Path) -> dict[str, object]:
    perf_seed = read_csv(SOURCES["q56_perf_seed"])
    perf_summary = read_csv(SOURCES["q56_perf_summary"])
    repr_seed = [row for row in read_csv(SOURCES["q56_repr_seed"]) if row["layer"] == "z"]
    repr_summary = [row for row in read_csv(SOURCES["q56_repr_summary"]) if row["layer"] == "z"]
    if len(perf_seed) != 140 or len(perf_summary) != 28 or len(repr_seed) != 140 or len(repr_summary) != 28:
        raise ValueError("Q5/Q6 dimension-architecture matrix is incomplete")
    validate_seed_matrix(perf_seed, ["sweep", "case", "method"])
    validate_seed_matrix(repr_seed, ["sweep", "case", "method", "layer"])
    late_seed4 = [row for row in perf_seed if row["sweep"] == "architecture" and row["case"] == "late_heavy" and seed(row) == 4]
    if len(late_seed4) != 4:
        raise ValueError("Late-heavy seed 4 is not fully preserved")
    write_csv(values / "fig6_per_seed_performance.csv", perf_seed)
    write_csv(values / "fig6_summary_performance.csv", perf_summary)
    write_csv(values / "fig6_per_seed_z_rank.csv", repr_seed)
    write_csv(values / "fig6_summary_z_rank.csv", repr_summary)

    configurations = [
        ("dimension", ["L16", "L32", "L64", "L128"], ["16", "32", "64", "128"]),
        ("architecture", ["early_heavy", "balanced", "late_heavy"], ["Early", "Balanced", "Late"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), constrained_layout=True)
    for col, (sweep, cases, labels) in enumerate(configurations):
        for row_index, (seed_rows, summary_rows, metric, ylabel) in enumerate(
            [
                (perf_seed, perf_summary, "accuracy", "Test accuracy"),
                (repr_seed, repr_summary, "effective_rank", "z effective rank"),
            ]
        ):
            ax = axes[row_index, col]
            x = np.arange(len(cases))
            for method in CORE_METHODS:
                for seed_value in sorted(FORMAL_SEEDS):
                    lookup = {
                        row["case"]: row
                        for row in seed_rows
                        if row["sweep"] == sweep and row["method"] == method and seed(row) == seed_value
                    }
                    values_y = [f(lookup[case], metric) for case in cases]
                    width = 1.05 if sweep == "architecture" and seed_value == 4 else 0.65
                    alpha = 0.38 if sweep == "architecture" and seed_value == 4 else 0.15
                    ax.plot(x, values_y, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=width, alpha=alpha)
                    ax.scatter(
                        x,
                        values_y,
                        facecolor="white" if seed_value != 4 else COLORS[method],
                        edgecolor=COLORS[method],
                        marker=MARKERS[method],
                        s=14 if seed_value != 4 else 22,
                        linewidth=0.7,
                        alpha=0.35 if seed_value != 4 else 0.8,
                    )
                lookup = {
                    row["case"]: row
                    for row in summary_rows
                    if row["sweep"] == sweep and row["method"] == method
                }
                means = np.array([f(lookup[case], f"{metric}_mean") for case in cases])
                lows = np.array([f(lookup[case], f"{metric}_ci_low") for case in cases])
                highs = np.array([f(lookup[case], f"{metric}_ci_high") for case in cases])
                ax.errorbar(
                    x,
                    means,
                    yerr=[means - lows, highs - means],
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    marker=MARKERS[method],
                    markerfacecolor="white",
                    markersize=4.2,
                    capsize=2.5,
                    linewidth=1.65,
                    zorder=4,
                )
            ax.set_xticks(x, labels)
            ax.set_xlabel("Latent dimension" if sweep == "dimension" else "Architecture allocation")
            ax.set_ylabel(ylabel)
            if metric == "accuracy":
                ax.set_ylim(0, 0.96)
            panel_label(ax, chr(ord("A") + row_index * 2 + col))
            if sweep == "architecture" and row_index == 0:
                ax.text(0.98, 0.04, "filled markers: seed 4 retained", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444444")
    axes[0, 0].legend(handles=legend_handles(CORE_METHODS), ncol=2, loc="lower left")
    save_figure(fig, output, "fig6_dimension_architecture")
    return {
        "sources": [SOURCES["q56_perf_seed"], SOURCES["q56_perf_summary"], SOURCES["q56_repr_seed"], SOURCES["q56_repr_summary"]],
        "plotted_rows": [140, 28, 140, 28],
        "late_heavy_seed4_rows": 4,
        "interpretation": "descriptive variability; no interaction proof",
    }


def method_design(output: Path) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(10.2, 4.0), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    methods = [
        ("BBB", ["BP", "BP", "BP"]),
        ("HBB", ["Hebb", "BP", "BP"]),
        ("HHB", ["Hebb", "Hebb", "BP"]),
        ("HHH", ["Hebb", "Hebb", "Hebb"]),
        ("RBB", ["Random", "BP", "BP"]),
        ("RRB", ["Random", "Random", "BP"]),
    ]
    y_positions = np.linspace(3.55, 0.55, len(methods))
    for (method, rules), y in zip(methods, y_positions):
        ax.text(0.35, y, method, color=COLORS[method], fontweight="bold", va="center", ha="left")
        previous_x = 1.15
        for index, rule in enumerate(rules):
            x = 1.65 + index * 1.35
            color = {"BP": "#E6F2F8", "Hebb": "#FFF0D6", "Random": "#EEEEEE"}[rule]
            box = FancyBboxPatch((x, y - 0.22), 0.95, 0.44, boxstyle="round,pad=0.04", facecolor=color, edgecolor=COLORS[method], linewidth=1)
            ax.add_patch(box)
            ax.text(x + 0.475, y, f"Enc{index + 1}\n{rule}", ha="center", va="center", fontsize=7.5)
            ax.add_patch(FancyArrowPatch((previous_x, y), (x, y), arrowstyle="-|>", mutation_scale=8, color="#777777", linewidth=0.8))
            previous_x = x + 0.95
        ax.add_patch(FancyArrowPatch((previous_x, y), (6.15, y), arrowstyle="-|>", mutation_scale=8, color="#777777", linewidth=0.8))
        decoder = FancyBboxPatch((6.15, y - 0.22), 1.25, 0.44, boxstyle="round,pad=0.04", facecolor="#F1EAF7", edgecolor="#6A51A3", linewidth=1)
        ax.add_patch(decoder)
        ax.text(6.775, y, "BP decoder", ha="center", va="center", fontsize=7.5)
        ax.add_patch(FancyArrowPatch((7.4, y), (8.05, y), arrowstyle="-|>", mutation_scale=8, color="#777777", linewidth=0.8))
        probe = FancyBboxPatch((8.05, y - 0.22), 1.45, 0.44, boxstyle="round,pad=0.04", facecolor="#E8F4EA", edgecolor="#238B45", linewidth=1)
        ax.add_patch(probe)
        ax.text(8.775, y, "Frozen probe", ha="center", va="center", fontsize=7.5)
    ax.text(1.65, 3.98, "Encoder learning-rule allocation", fontsize=9, fontweight="bold", ha="left")
    ax.text(6.15, 3.98, "Shared evaluation path", fontsize=9, fontweight="bold", ha="left")
    save_figure(fig, output, "method_design")
    return {"sources": [], "design_only": True, "formal_hero_figure": False}


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    manifest, source_records = verify_release()
    configure_style()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".final-figures-", dir=output.parent))
    try:
        values = temp / "plotted_values"
        values.mkdir()
        records = {
            "fig1_performance": figure1(temp, values),
            "fig2_prefix_value_training_cost": figure2(temp, values),
            "fig3_layerwise_representation": figure3(temp, values),
            "fig4_update_mechanism": figure4(temp, values),
            "fig5_robustness": figure5(temp, values),
            "fig6_dimension_architecture": figure6(temp, values),
            "method_design": method_design(temp),
        }
        for record in records.values():
            record["source_hashes"] = {
                relative: source_records[relative]["sha256"] for relative in record["sources"]
            }
        generated_files = sorted(path for path in temp.rglob("*") if path.is_file())
        output_hashes = {path.relative_to(temp).as_posix(): sha256(path) for path in generated_files}
        source_manifest = {
            "release_id": manifest["release_id"],
            "audited_source_commit": manifest["audited_source_commit"],
            "builder": {
                "path": "scripts/plot_final_figures.py",
                "sha256": sha256(Path(__file__)),
            },
            "formal_seeds": sorted(FORMAL_SEEDS),
            "statistical_unit": "paired seed",
            "bootstrap": {"resamples": 10000, "seed": 2026, "confidence_level": 0.95},
            "scientific_actions": {
                "datasets_loaded": False,
                "checkpoints_loaded": False,
                "training_performed": False,
                "model_evaluation_performed": False,
                "interpolation_performed": False,
                "test_access_increment": 0,
            },
            "figures": records,
            "generated_file_hashes": output_hashes,
        }
        (temp / "source_manifest.json").write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.replace(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == RELEASE or RELEASE in output.parents:
        raise ValueError("Figure output cannot be inside the frozen release bundle")
    build(output)
    print(json.dumps({"decision": "PASS", "output": str(output), "formal_hero_figures": 6}, indent=2))


if __name__ == "__main__":
    main()
