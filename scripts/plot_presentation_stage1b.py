"""Generate presentation figures from existing Q1 and Stage 1B artifacts.

This script is read-only with respect to experiment outputs. It auto-discovers
the completed result files below ``results/`` and writes figures plus a source
manifest to ``figures/presentation_stage1b/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "bp": "#3568A8",
    "hebbian": "#E07A2D",
    "random": "#7A7A7A",
    "baseline": "#222222",
    "v1": "#3568A8",
    "v2": "#E07A2D",
}
MARKERS = {"baseline": "*", "v1": "o", "v2": "^"}
LAYERS = ("h1", "h2", "z")
HEALTH_METRICS = (
    "winner_coverage_ratio",
    "winner_entropy",
    "normalized_effective_rank",
    "max_winner_share",
)
DISPLAY_NAMES = {
    "rms_power_0p5_wta_0p10": "v1 RMS p=.5, WTA=.10",
    "rms_power_1p0_wta_0p10": "v1 RMS p=1.0, WTA=.10",
    "standardized_wta_0p10": "v1 standardized, WTA=.10",
    "standardized_wta_0p20": "v1 standardized, WTA=.20",
    "centered_rms_lr_0p0005": "v2 centered RMS, lr=5e-4",
    "centered_rms_lr_0p001": "v2 centered RMS, lr=1e-3",
    "centered_standardized_lr_0p0005": (
        "v2 centered standardized, lr=5e-4"
    ),
    "centered_standardized_lr_0p001": (
        "v2 centered standardized, lr=1e-3"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing results/ (default: inferred).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: figures/presentation_stage1b).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_match(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        rendered = "\n".join(f"  - {path}" for path in paths) or "  (none)"
        raise RuntimeError(
            f"Expected exactly one {description}; found {len(paths)}:\n{rendered}"
        )
    return paths[0]


def discover_sources(repo_root: Path) -> dict[str, Any]:
    results_root = repo_root / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    q1_candidates = []
    for path in results_root.rglob("q1_summary.json"):
        payload = read_json(path)
        if payload.get("schema_version") == "q1-clean-summary-v1":
            q1_candidates.append(path)
    q1_summary_path = unique_match(q1_candidates, "Q1 clean summary")
    q1_summary = read_json(q1_summary_path)
    q1_table_path = q1_summary_path.with_name("q1_run_table.csv")
    if not q1_table_path.is_file():
        raise FileNotFoundError(f"Missing paired Q1 table: {q1_table_path}")

    baseline_selection_candidates = []
    for path in (results_root / "tuning").rglob("selection_decision.json"):
        payload = read_json(path)
        if (
            payload.get("schema_version") == "validation-tuning-decision-v1"
            and "best_hebbian" in payload
        ):
            baseline_selection_candidates.append(path)
    baseline_selection_path = unique_match(
        baseline_selection_candidates, "validation-tuning baseline selection"
    )
    baseline_selection = read_json(baseline_selection_path)

    baseline_health_candidates = []
    for path in (results_root / "formal").rglob("gate_decision.json"):
        payload = read_json(path)
        if (
            payload.get("schema_version") == "representation-health-decision-v1"
            and payload.get("primary_checkpoint_id") == "hebbian_selected_seed42"
        ):
            baseline_health_candidates.append(path)
    baseline_health_path = unique_match(
        baseline_health_candidates, "raw Hebbian representation-health gate"
    )
    baseline_health = read_json(baseline_health_path)

    stage1b_selection_paths = []
    for path in (results_root / "tuning").glob(
        "stage1b_*/selection_decision.json"
    ):
        payload = read_json(path)
        if payload.get("schema_version") == "stage1b-selection-v1":
            stage1b_selection_paths.append(path)
    if len(stage1b_selection_paths) != 2:
        raise RuntimeError(
            "Expected exactly two Stage 1B selection records (v1 and v2); "
            f"found {len(stage1b_selection_paths)}"
        )

    return {
        "q1_summary_path": q1_summary_path,
        "q1_summary": q1_summary,
        "q1_table_path": q1_table_path,
        "baseline_selection_path": baseline_selection_path,
        "baseline_selection": baseline_selection,
        "baseline_health_path": baseline_health_path,
        "baseline_health": baseline_health,
        "stage1b_selection_paths": sorted(stage1b_selection_paths),
    }


def load_q1_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    included = [
        row
        for row in rows
        if int(row["seed"]) in (0, 1) and row["rule"] in {"bp", "hebbian", "random"}
    ]
    expected = {(seed, rule) for seed in (0, 1) for rule in ("bp", "hebbian", "random")}
    observed = {(int(row["seed"]), row["rule"]) for row in included}
    if observed != expected:
        raise RuntimeError(f"Incomplete paired Q1 rows: expected {expected}, got {observed}")
    return included


def group_from_stage1b_path(path: Path) -> str:
    name = path.parent.name
    if name.endswith("_v1"):
        return "v1"
    if name.endswith("_v2"):
        return "v2"
    raise RuntimeError(f"Cannot infer Stage 1B group from {path}")


def build_stage1b_points(sources: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_health = sources["baseline_health"]
    baseline_selection = sources["baseline_selection"]
    points = [
        {
            "name": "raw_baseline",
            "display_name": "raw baseline",
            "group": "baseline",
            "validation_accuracy": float(
                baseline_selection["best_hebbian"]["validation_accuracy"]
            ),
            "z_normalized_effective_rank": float(
                baseline_health["layers"]["z"]["metrics"][
                    "normalized_effective_rank"
                ]
            ),
            "z_winner_coverage_ratio": float(
                baseline_health["layers"]["z"]["metrics"][
                    "winner_coverage_ratio"
                ]
            ),
            "health_path": sources["baseline_health_path"],
            "selection_path": sources["baseline_selection_path"],
            "health_payload": baseline_health,
        }
    ]

    for selection_path in sources["stage1b_selection_paths"]:
        selection = read_json(selection_path)
        group = group_from_stage1b_path(selection_path)
        health_root = selection_path.parent / "health"
        for trial in selection["trials"]:
            trial_id = trial["trial_id"]
            health_path = health_root / f"{trial_id}.json"
            if not health_path.is_file():
                raise FileNotFoundError(f"Missing Stage 1B health result: {health_path}")
            health = read_json(health_path)
            z_metrics = health["layers"]["z"]["metrics"]
            points.append(
                {
                    "name": trial_id,
                    "display_name": DISPLAY_NAMES[trial_id],
                    "group": group,
                    "validation_accuracy": float(trial["validation_accuracy"]),
                    "z_normalized_effective_rank": float(
                        z_metrics["normalized_effective_rank"]
                    ),
                    "z_winner_coverage_ratio": float(
                        z_metrics["winner_coverage_ratio"]
                    ),
                    "health_path": health_path,
                    "selection_path": selection_path,
                    "health_payload": health,
                }
            )

    if len(points) != 9:
        raise RuntimeError(f"Expected baseline + 8 Stage 1B points; got {len(points)}")
    return points


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
        metadata={"Software": "plot_presentation_stage1b.py"},
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "plot_presentation_stage1b.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def plot_accuracy(
    q1_summary: dict[str, Any],
    q1_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    rules = ("bp", "hebbian", "random")
    labels = ("BP", "Hebbian", "Random")
    means = np.array(
        [q1_summary["rules"][rule]["test_accuracy"]["mean"] for rule in rules]
    ) * 100
    sds = np.array(
        [q1_summary["rules"][rule]["test_accuracy"]["sd"] for rule in rules]
    ) * 100

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    x = np.arange(len(rules))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=7,
        width=0.64,
        color=[COLORS[rule] for rule in rules],
        edgecolor="#202020",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        zorder=2,
    )

    seed_offsets = {0: -0.055, 1: 0.055}
    for seed in (0, 1):
        values = [
            float(
                next(
                    row["test_accuracy"]
                    for row in q1_rows
                    if int(row["seed"]) == seed and row["rule"] == rule
                )
            )
            * 100
            for rule in rules
        ]
        ax.plot(
            x + seed_offsets[seed],
            values,
            color="#FFFFFF",
            linewidth=1.2,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            x + seed_offsets[seed],
            values,
            s=52,
            facecolor="#FFFFFF",
            edgecolor="#202020",
            linewidth=1.1,
            zorder=4,
        )

    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in means], padding=11)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Preliminary classification accuracy\nPaired seeds n=2")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "preliminary_classification_accuracy")


def plot_reconstruction(
    q1_summary: dict[str, Any],
    q1_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    rules = ("bp", "hebbian")
    labels = ("BP", "Hebbian")
    means = np.array(
        [
            q1_summary["rules"][rule]["test_reconstruction_mse"]["mean"]
            for rule in rules
        ]
    )
    sds = np.array(
        [
            q1_summary["rules"][rule]["test_reconstruction_mse"]["sd"]
            for rule in rules
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    x = np.arange(len(rules))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=7,
        width=0.58,
        color=[COLORS[rule] for rule in rules],
        edgecolor="#202020",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        zorder=2,
    )
    seed_offsets = {0: -0.045, 1: 0.045}
    for seed in (0, 1):
        values = [
            float(
                next(
                    row["test_reconstruction_mse"]
                    for row in q1_rows
                    if int(row["seed"]) == seed and row["rule"] == rule
                )
            )
            for rule in rules
        ]
        ax.scatter(
            x + seed_offsets[seed],
            values,
            s=52,
            facecolor="#FFFFFF",
            edgecolor="#202020",
            linewidth=1.1,
            zorder=4,
        )

    ax.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=11)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(means + sds) * 1.24)
    ax.set_ylabel("Test reconstruction MSE")
    ax.set_title("Preliminary reconstruction error\nPaired seeds n=2")
    clean_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "preliminary_reconstruction_mse")


def wrapped_label(name: str) -> str:
    return "\n".join(
        textwrap.wrap(name, width=25, break_long_words=True, break_on_hyphens=False)
    )


def scatter_offsets(metric: str) -> dict[str, tuple[int, int]]:
    if metric == "z_normalized_effective_rank":
        return {
            "raw_baseline": (-82, 18),
            "rms_power_0p5_wta_0p10": (18, -38),
            "rms_power_1p0_wta_0p10": (32, -43),
            "standardized_wta_0p10": (32, 20),
            "standardized_wta_0p20": (-160, -8),
            "centered_rms_lr_0p0005": (18, -22),
            "centered_standardized_lr_0p0005": (18, 13),
            "centered_rms_lr_0p001": (-155, 18),
            "centered_standardized_lr_0p001": (18, -21),
        }
    return {
        "raw_baseline": (-90, 18),
        "rms_power_0p5_wta_0p10": (16, -27),
        "rms_power_1p0_wta_0p10": (-225, -24),
        "standardized_wta_0p10": (-225, 25),
        "standardized_wta_0p20": (-225, -56),
        "centered_rms_lr_0p0005": (18, -24),
        "centered_standardized_lr_0p0005": (-205, 14),
        "centered_rms_lr_0p001": (-178, 18),
        "centered_standardized_lr_0p001": (18, -20),
    }


def plot_tradeoff(
    points: list[dict[str, Any]],
    metric: str,
    xlabel: str,
    stem: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 8.2))
    for group in ("baseline", "v1", "v2"):
        group_points = [point for point in points if point["group"] == group]
        ax.scatter(
            [point[metric] for point in group_points],
            [point["validation_accuracy"] * 100 for point in group_points],
            s=190 if group == "baseline" else 110,
            marker=MARKERS[group],
            color=COLORS[group],
            edgecolor="white",
            linewidth=1.1,
            zorder=7 if group == "baseline" else 4,
        )

    offsets = scatter_offsets(metric)
    for point in points:
        ax.annotate(
            wrapped_label(point["display_name"]),
            (point[metric], point["validation_accuracy"] * 100),
            xytext=offsets[point["name"]],
            textcoords="offset points",
            fontsize=11,
            arrowprops={
                "arrowstyle": "-",
                "color": "#8A8A8A",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 4,
            },
            zorder=5,
        )

    ax.axhline(
        88.63,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Accuracy floor (88.63%)",
        zorder=1,
    )
    if metric == "z_normalized_effective_rank":
        ax.axvline(
            0.10,
            color="#7C3F98",
            linestyle=":",
            linewidth=1.5,
            label="Normalized-rank threshold (0.10)",
            zorder=1,
        )
        ax.set_xlim(0, 0.11)
    else:
        ax.axvline(
            0.50,
            color="#7C3F98",
            linestyle=":",
            linewidth=1.5,
            label="Coverage threshold (0.50)",
            zorder=1,
        )
        ax.set_xlim(0, 1.05)

    ax.set_ylim(12, 96)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Stage 1B representation–performance trade-off\nValidation-only seed 42; no candidate passed")
    clean_axes(ax)
    category_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[group],
            color="none",
            markerfacecolor=COLORS[group],
            markeredgecolor="white",
            markersize=12,
            label=label,
        )
        for group, label in (
            ("baseline", "Raw baseline"),
            ("v1", "v1 candidates"),
            ("v2", "v2 candidates"),
        )
    ]
    threshold_handles, threshold_labels = ax.get_legend_handles_labels()
    ax.legend(
        category_handles + threshold_handles,
        [handle.get_label() for handle in category_handles] + threshold_labels,
        loc="center right",
        frameon=False,
        ncol=1,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def health_matrix(point: dict[str, Any]) -> np.ndarray:
    layers = point["health_payload"]["layers"]
    return np.array(
        [
            [float(layers[layer]["metrics"][metric]) for metric in HEALTH_METRICS]
            for layer in LAYERS
        ],
        dtype=float,
    )


def plot_health_heatmap(
    baseline: dict[str, Any],
    best_repair: dict[str, Any],
    output_dir: Path,
) -> None:
    matrices = [health_matrix(baseline), health_matrix(best_repair)]
    titles = [
        "Raw Hebbian baseline",
        (
            "Best rank-repair candidate (still FAIL)\n"
            f"{best_repair['display_name']}"
        ),
    ]
    column_labels = (
        "Winner\ncoverage ↑",
        "Winner\nentropy ↑",
        "Normalized\neffective rank ↑",
        "Max winner\nshare ↓",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    image = None
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(column_labels)), column_labels)
        ax.set_yticks(np.arange(len(LAYERS)), LAYERS)
        ax.set_title(title, pad=14, fontsize=16)
        ax.tick_params(length=0)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.48 else "#111111",
                    fontsize=13,
                    fontweight="bold",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert image is not None
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.14, top=0.79, wspace=0.20)
    colorbar_axis = fig.add_axes([0.88, 0.18, 0.018, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Metric value (0–1)", fontsize=14)
    fig.suptitle(
        "Representation-health comparison across encoder layers\nValidation-only seed 42",
        y=1.02,
    )
    save_figure(fig, output_dir, "representation_health_heatmap")


def write_provenance(
    repo_root: Path,
    output_dir: Path,
    sources: dict[str, Any],
    q1_rows: list[dict[str, str]],
    points: list[dict[str, Any]],
    best_repair: dict[str, Any],
) -> None:
    baseline = next(point for point in points if point["group"] == "baseline")
    source_paths = {
        sources["q1_summary_path"],
        sources["q1_table_path"],
        sources["baseline_selection_path"],
        sources["baseline_health_path"],
        *(point["selection_path"] for point in points if point["group"] != "baseline"),
        *(point["health_path"] for point in points if point["group"] != "baseline"),
    }
    manifest = {
        "schema_version": "presentation-stage1b-sources-v1",
        "generated_by": str(Path(__file__).relative_to(repo_root)),
        "figures": [
            "preliminary_classification_accuracy",
            "preliminary_reconstruction_mse",
            "stage1b_tradeoff_effective_rank",
            "stage1b_tradeoff_winner_coverage",
            "representation_health_heatmap",
        ],
        "paired_seeds": [0, 1],
        "best_repair_candidate": best_repair["name"],
        "best_repair_selection_rule": (
            "maximum z normalized effective rank; validation accuracy as tie-break"
        ),
        "best_repair_gate_pass": bool(
            best_repair["health_payload"].get("health_pass", False)
        ),
        "source_files": [
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256(path),
            }
            for path in sorted(source_paths)
        ],
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    q1_summary = sources["q1_summary"]
    for rule in ("bp", "hebbian", "random"):
        rows.append(
            {
                "figure": "preliminary_classification_accuracy",
                "series": rule,
                "layer": "",
                "metric": "test_accuracy_mean",
                "value": q1_summary["rules"][rule]["test_accuracy"]["mean"],
                "source": str(sources["q1_summary_path"].relative_to(repo_root)),
            }
        )
        rows.append(
            {
                "figure": "preliminary_classification_accuracy",
                "series": rule,
                "layer": "",
                "metric": "test_accuracy_sd",
                "value": q1_summary["rules"][rule]["test_accuracy"]["sd"],
                "source": str(sources["q1_summary_path"].relative_to(repo_root)),
            }
        )
    for rule in ("bp", "hebbian"):
        rows.append(
            {
                "figure": "preliminary_reconstruction_mse",
                "series": rule,
                "layer": "",
                "metric": "test_reconstruction_mse_mean",
                "value": q1_summary["rules"][rule]["test_reconstruction_mse"]["mean"],
                "source": str(sources["q1_summary_path"].relative_to(repo_root)),
            }
        )
        rows.append(
            {
                "figure": "preliminary_reconstruction_mse",
                "series": rule,
                "layer": "",
                "metric": "test_reconstruction_mse_sd",
                "value": q1_summary["rules"][rule]["test_reconstruction_mse"]["sd"],
                "source": str(sources["q1_summary_path"].relative_to(repo_root)),
            }
        )
    for q1_row in q1_rows:
        for metric in ("test_accuracy", "test_reconstruction_mse"):
            if q1_row[metric]:
                rows.append(
                    {
                        "figure": "preliminary_seed_values",
                        "series": f"{q1_row['rule']}_seed{q1_row['seed']}",
                        "layer": "",
                        "metric": metric,
                        "value": q1_row[metric],
                        "source": str(sources["q1_table_path"].relative_to(repo_root)),
                    }
                )
    for point in points:
        for metric in (
            "validation_accuracy",
            "z_normalized_effective_rank",
            "z_winner_coverage_ratio",
        ):
            source_path = (
                point["selection_path"]
                if metric == "validation_accuracy"
                else point["health_path"]
            )
            rows.append(
                {
                    "figure": "stage1b_tradeoff",
                    "series": point["name"],
                    "layer": "z",
                    "metric": metric,
                    "value": point[metric],
                    "source": str(source_path.relative_to(repo_root)),
                }
            )
    for point in (baseline, best_repair):
        matrix = health_matrix(point)
        for layer_index, layer in enumerate(LAYERS):
            for metric_index, metric in enumerate(HEALTH_METRICS):
                rows.append(
                    {
                        "figure": "representation_health_heatmap",
                        "series": point["name"],
                        "layer": layer,
                        "metric": metric,
                        "value": matrix[layer_index, metric_index],
                        "source": str(point["health_path"].relative_to(repo_root)),
                    }
                )

    with (output_dir / "plotted_values.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure", "series", "layer", "metric", "value", "source"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "figures" / "presentation_stage1b"
    )
    configure_style()
    sources = discover_sources(repo_root)
    q1_rows = load_q1_rows(sources["q1_table_path"])
    points = build_stage1b_points(sources)

    baseline = next(point for point in points if point["group"] == "baseline")
    candidates = [point for point in points if point["group"] != "baseline"]
    best_repair = max(
        candidates,
        key=lambda point: (
            point["z_normalized_effective_rank"],
            point["validation_accuracy"],
        ),
    )

    plot_accuracy(sources["q1_summary"], q1_rows, output_dir)
    plot_reconstruction(sources["q1_summary"], q1_rows, output_dir)
    plot_tradeoff(
        points,
        metric="z_normalized_effective_rank",
        xlabel="Normalized effective rank at z",
        stem="stage1b_tradeoff_effective_rank",
        output_dir=output_dir,
    )
    plot_tradeoff(
        points,
        metric="z_winner_coverage_ratio",
        xlabel="Winner coverage ratio at z",
        stem="stage1b_tradeoff_winner_coverage",
        output_dir=output_dir,
    )
    plot_health_heatmap(baseline, best_repair, output_dir)
    write_provenance(
        repo_root,
        output_dir,
        sources,
        q1_rows,
        points,
        best_repair,
    )
    print(f"Generated presentation figures in {output_dir}")
    print(f"Best rank-repair candidate: {best_repair['name']} (still gate FAIL)")


if __name__ == "__main__":
    main()
