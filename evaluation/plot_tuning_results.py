"""Plot validation-only tuning results without reading test metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get("status") != "completed" for row in rows):
        raise RuntimeError("Tuning table contains incomplete logical trials")
    return rows


def plot_tuning_results(trial_table: str | Path, output: str | Path) -> Path:
    rows = _rows(Path(trial_table))
    by_id = {row["trial_id"]: row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    lr_ids = ["hebb_lr_0p0001", "hebb_lr_0p0005", "hebb_lr_0p001", "hebb_lr_0p005"]
    axes[0, 0].plot(
        ["1e-4", "5e-4", "1e-3", "5e-3"],
        [100 * float(by_id[key]["validation_accuracy"]) for key in lr_ids],
        marker="o",
    )
    axes[0, 0].set_title("Hebbian learning rate (L=64, WTA=0.20)")

    wta_ids = ["hebb_wta_0p05", "hebb_wta_0p1", "hebb_wta_0p2"]
    axes[0, 1].plot(
        ["0.05", "0.10", "0.20"],
        [100 * float(by_id[key]["validation_accuracy"]) for key in wta_ids],
        marker="o",
    )
    axes[0, 1].set_title("Hebbian winner fraction (lr=5e-4, L=64)")

    latent_ids = ["hebb_latent_16", "hebb_latent_64", "hebb_latent_128"]
    axes[1, 0].plot(
        ["16", "64", "128"],
        [100 * float(by_id[key]["validation_accuracy"]) for key in latent_ids],
        marker="o",
    )
    axes[1, 0].set_title("Hebbian dimension screen (Q5 evidence)")

    for weight_decay, label in (("0", "wd=0"), ("0p0001", "wd=1e-4")):
        ids = [
            f"bp_lr_0p0003_wd_{weight_decay}",
            f"bp_lr_0p001_wd_{weight_decay}",
            f"bp_lr_0p003_wd_{weight_decay}",
            f"bp_lr_0p01_wd_{weight_decay}",
        ]
        axes[1, 1].plot(
            ["3e-4", "1e-3", "3e-3", "1e-2"],
            [100 * float(by_id[key]["validation_accuracy"]) for key in ids],
            marker="o",
            label=label,
        )
    axes[1, 1].set_title("BP learning rate and weight decay (L=64)")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_ylabel("Validation accuracy (%)")
        axis.set_xlabel("Candidate")
        axis.grid(alpha=0.25)
    figure.suptitle("Validation-only tuning, seed 42 (no test metrics)")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_tuning_results(args.trial_table, args.output).resolve())


if __name__ == "__main__":
    main()
