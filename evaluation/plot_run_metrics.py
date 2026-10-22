"""Plot BP/Hebbian training diagnostics and their paired comparison."""

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


def plot_bp_diagnostics(run_dir: Path) -> Path:
    """Plot joint-AE and frozen-linear-probe learning curves for a BP run."""

    rows = _rows(run_dir)
    representation = [row for row in rows if row["stage"] == "representation"]
    train = [row for row in representation if row["split"] == "train"]
    validation = [row for row in representation if row["split"] == "validation"]
    probe = [
        row
        for row in rows
        if row["stage"] == "linear_probe" and row["split"] == "validation"
    ]
    if not train or not validation:
        raise ValueError("BP run has no representation train/validation history")
    if not probe:
        raise ValueError("BP run has no linear-probe validation history")

    train_by_epoch = {int(_number(row, "epoch")): row for row in train}
    validation_by_epoch = {int(_number(row, "epoch")): row for row in validation}
    common_epochs = sorted(set(train_by_epoch) & set(validation_by_epoch))
    train_mse = [
        _number(train_by_epoch[epoch], "reconstruction_loss")
        for epoch in common_epochs
    ]
    validation_mse = [
        _number(validation_by_epoch[epoch], "reconstruction_loss")
        for epoch in common_epochs
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(common_epochs, train_mse, marker="o", label="Train", color="#2f6f9f")
    axis.plot(
        common_epochs,
        validation_mse,
        marker="o",
        label="Validation",
        color="#df8f2d",
    )
    axis.set_title("Joint autoencoder reconstruction MSE")
    axis.set_xlabel("AE epoch")
    axis.set_ylabel("Pixel-mean MSE")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[0, 1]
    initial_validation_mse = validation_mse[0]
    axis.plot(
        common_epochs,
        [
            100.0 * (initial_validation_mse - value) / initial_validation_mse
            for value in validation_mse
        ],
        marker="o",
        color="#8b4f96",
    )
    axis.set_title("Validation reconstruction improvement")
    axis.set_xlabel("AE epoch")
    axis.set_ylabel("MSE reduction from epoch 1 (%)")
    axis.grid(alpha=0.25)

    probe_epochs = [_number(row, "epoch") for row in probe]
    axis = axes[1, 0]
    axis.plot(
        probe_epochs,
        [_number(row, "accuracy") for row in probe],
        marker="o",
        markersize=3,
        label="Accuracy",
        color="#3d9970",
    )
    axis.plot(
        probe_epochs,
        [_number(row, "macro_f1") for row in probe],
        marker="o",
        markersize=3,
        label="Macro-F1",
        color="#cf7b31",
    )
    axis.set_title("Frozen linear-probe validation metrics")
    axis.set_xlabel("Probe epoch")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.plot(
        probe_epochs,
        [_number(row, "classification_ce") for row in probe],
        marker="o",
        markersize=3,
        color="#b44c64",
    )
    axis.set_title("Frozen linear-probe validation CE")
    axis.set_xlabel("Probe epoch")
    axis.set_ylabel("Cross-entropy")
    axis.grid(alpha=0.25)

    fig.suptitle("Backpropagation training diagnostics (seed 0)", fontsize=14)
    output = run_dir / "bp_training_diagnostics.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _stage_rows(
    rows: list[dict[str, str]], *, stage: str, split: str
) -> list[dict[str, str]]:
    selected = [
        row for row in rows if row.get("stage") == stage and row.get("split") == split
    ]
    return sorted(selected, key=lambda row: _number(row, "epoch"))


def plot_paired_training_diagnostics(hebbian_run: Path, bp_run: Path) -> Path:
    """Compare only diagnostics whose definitions exist for both protocols."""

    bp_rows = _rows(bp_run)
    hebbian_rows = _rows(hebbian_run)
    bp_reconstruction = _stage_rows(
        bp_rows, stage="representation", split="validation"
    )
    hebbian_reconstruction = _stage_rows(
        hebbian_rows, stage="decoder", split="validation"
    )
    bp_probe = _stage_rows(bp_rows, stage="linear_probe", split="validation")
    hebbian_probe = _stage_rows(
        hebbian_rows, stage="linear_probe", split="validation"
    )
    groups = {
        "BP reconstruction": bp_reconstruction,
        "Hebbian reconstruction": hebbian_reconstruction,
        "BP probe": bp_probe,
        "Hebbian probe": hebbian_probe,
    }
    missing = [name for name, selected in groups.items() if not selected]
    if missing:
        raise ValueError(f"Missing histories required for paired plot: {missing}")

    colors = {"BP": "#2f6f9f", "Hebbian": "#df8f2d"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.12, top=0.89, hspace=0.38, wspace=0.26
    )

    axis = axes[0, 0]
    reconstruction_groups = (
        ("BP", bp_reconstruction),
        ("Hebbian", hebbian_reconstruction),
    )
    for name, selected in reconstruction_groups:
        axis.plot(
            [_number(row, "epoch") for row in selected],
            [_number(row, "reconstruction_loss") for row in selected],
            marker="o",
            linewidth=1.8,
            label=name,
            color=colors[name],
        )
    axis.set_title("Validation reconstruction MSE")
    axis.set_xlabel("Protocol-local reconstruction epoch")
    axis.set_ylabel("Pixel-mean MSE")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for name, selected in reconstruction_groups:
        values = [_number(row, "reconstruction_loss") for row in selected]
        initial = values[0]
        axis.plot(
            [_number(row, "epoch") for row in selected],
            [100.0 * (initial - value) / initial for value in values],
            marker="o",
            linewidth=1.8,
            label=name,
            color=colors[name],
        )
    axis.set_title("Within-protocol reconstruction improvement")
    axis.set_xlabel("Protocol-local reconstruction epoch")
    axis.set_ylabel("MSE reduction from epoch 1 (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    probe_groups = (("BP", bp_probe), ("Hebbian", hebbian_probe))
    for axis, field, title in (
        (axes[1, 0], "accuracy", "Frozen-probe validation accuracy"),
        (axes[1, 1], "macro_f1", "Frozen-probe validation macro-F1"),
    ):
        for name, selected in probe_groups:
            axis.plot(
                [_number(row, "epoch") for row in selected],
                [_number(row, field) for row in selected],
                linewidth=1.7,
                label=name,
                color=colors[name],
            )
        axis.set_title(title)
        axis.set_xlabel("Probe epoch")
        axis.set_ylabel("Score")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle("BP–Hebbian common-metric training diagnostics (seed 0)", fontsize=14)
    fig.text(
        0.5,
        0.025,
        (
            "Reconstruction context: BP joint AE vs Hebbian frozen encoder + BP "
            "decoder (not a pure learning-rule comparison).\n"
            "Frozen-probe curves use the same protocol and are directly comparable."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    output = hebbian_run / "bp_hebbian_training_diagnostics.png"
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
    print(plot_bp_diagnostics(args.bp_run).resolve())
    print(plot_paired_training_diagnostics(args.hebbian_run, args.bp_run).resolve())
    print(plot_bp_comparison(args.hebbian_run, args.bp_run).resolve())


if __name__ == "__main__":
    main()
