"""Plot Hebbian training diagnostics and the paired BP comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def plot_hebbian_diagnostics(run_dir: Path) -> Path:
    rows = _rows(run_dir)
    hebbian = [row for row in rows if row["stage"] == "hebbian_encoder"]
    decoder = [
        row
        for row in rows
        if row["stage"] == "decoder" and row["split"] == "validation"
    ]
    colors = {"enc1": "#2f6f9f", "enc2": "#df8f2d", "enc3": "#3d9970"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    definitions = (
        ("update_norm", "Candidate update norm", True),
        ("active_neuron_ratio", "Active-neuron ratio", False),
        ("winner_entropy", "Normalized winner entropy", False),
    )
    for axis, (field, title, log_scale) in zip(axes.flat[:3], definitions):
        for layer in ("enc1", "enc2", "enc3"):
            selected = [row for row in hebbian if row["layer"] == layer]
            axis.plot(
                [_number(row, "epoch") for row in selected],
                [_number(row, field) for row in selected],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=layer,
                color=colors[layer],
            )
        axis.set_title(title)
        axis.set_xlabel("Layer-local epoch")
        axis.grid(alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
        axis.legend(frameon=False)

    axis = axes.flat[3]
    axis.plot(
        [_number(row, "epoch") for row in decoder],
        [_number(row, "reconstruction_loss") for row in decoder],
        marker="o",
        color="#8b4f96",
    )
    axis.set_title("Frozen-encoder decoder validation MSE")
    axis.set_xlabel("Decoder epoch")
    axis.grid(alpha=0.25)
    fig.suptitle("Hebbian encoder and decoder diagnostics (seed 0)", fontsize=14)
    output = run_dir / "hebbian_training_diagnostics.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _final_test(rows: list[dict[str, str]], stage: str) -> dict[str, str]:
    matches = [
        row for row in rows if row["stage"] == stage and row["split"] == "test"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one test row for {stage}, found {len(matches)}")
    return matches[0]


def plot_bp_comparison(hebbian_run: Path, bp_run: Path) -> Path:
    names = ["BP", "Hebbian"]
    run_rows = [_rows(bp_run), _rows(hebbian_run)]
    reconstruction = [
        _number(_final_test(rows, "reconstruction_final"), "reconstruction_loss")
        for rows in run_rows
    ]
    probe_rows = [_final_test(rows, "linear_probe_final") for rows in run_rows]
    accuracy = [_number(row, "accuracy") for row in probe_rows]
    macro_f1 = [_number(row, "macro_f1") for row in probe_rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    colors = ["#376996", "#cf7b31"]
    axes[0].bar(names, reconstruction, color=colors)
    axes[0].set_ylabel("Pixel-mean MSE (lower is better)")
    axes[0].set_title("Test reconstruction")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(reconstruction):
        axes[0].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    positions = range(len(names))
    width = 0.35
    axes[1].bar([p - width / 2 for p in positions], accuracy, width, label="Accuracy", color="#4c78a8")
    axes[1].bar([p + width / 2 for p in positions], macro_f1, width, label="Macro-F1", color="#72b7b2")
    axes[1].set_xticks(list(positions), names)
    axes[1].set_ylim(0.8, 0.94)
    axes[1].set_title("Frozen linear probe on test set")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    for index, value in enumerate(accuracy):
        axes[1].text(index - width / 2, value, f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(macro_f1):
        axes[1].text(index + width / 2, value, f"{100 * value:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase0-v1 single-seed BP–Hebbian comparison", fontsize=13)
    output = hebbian_run / "bp_hebbian_seed0_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hebbian-run", required=True, type=Path)
    parser.add_argument("--bp-run", required=True, type=Path)
    args = parser.parse_args()
    print(plot_hebbian_diagnostics(args.hebbian_run).resolve())
    print(plot_bp_comparison(args.hebbian_run, args.bp_run).resolve())


if __name__ == "__main__":
    main()
