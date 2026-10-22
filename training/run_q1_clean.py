"""Resumable five-seed Q1 clean-performance experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from evaluation.evaluate_reconstruction import evaluate_config
from models import ConvAutoencoder
from schemas import load_config, validate_config
from training.train_linear_probe import train_linear_probe_config
from training.train_representation import train_config
from utils.checkpointing import config_fingerprint, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum
from utils.results import (
    create_run_directory,
    initialize_run_status,
    read_metadata,
    read_run_status,
    write_metadata,
    write_resolved_config,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _metric_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _has_metric(run_dir: Path, *, stage: str, split: str) -> bool:
    return any(
        row.get("stage") == stage and row.get("split") == split
        for row in _metric_rows(run_dir)
    )


def _final_metric(run_dir: Path, *, stage: str, split: str) -> dict[str, str]:
    matches = [
        row
        for row in _metric_rows(run_dir)
        if row.get("stage") == stage and row.get("split") == split
    ]
    if not matches:
        raise RuntimeError(f"Missing {stage}/{split} metric: {run_dir}")
    return matches[-1]


def _bootstrap_ci(
    differences: np.ndarray, *, seed: int, resamples: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _curve_summary(
    rows: list[dict[str, str]], *, threshold: float
) -> dict[str, float | int | None]:
    curve = sorted(rows, key=lambda row: int(row["epoch"]))
    accuracies = np.asarray([float(row["accuracy"]) for row in curve])
    samples = np.asarray([float(row["samples_seen"]) for row in curve])
    walls = np.asarray([float(row["wall_time_sec"]) for row in curve])

    def normalized_auc(x: np.ndarray) -> float:
        if len(x) < 2 or x[-1] == x[0]:
            return float(accuracies.mean())
        return float(np.trapezoid(accuracies, x=x) / (x[-1] - x[0]))

    reached = np.flatnonzero(accuracies >= threshold)
    first = None if len(reached) == 0 else int(reached[0])
    return {
        "epoch_aulc": float(accuracies.mean()),
        "samples_aulc": normalized_auc(samples),
        "wall_time_aulc": normalized_auc(walls),
        "samples_to_threshold": None if first is None else int(samples[first]),
        "wall_time_to_threshold_sec": None if first is None else float(walls[first]),
        "probe_wall_time_sec": float(walls[-1]),
        "probe_samples_seen": int(samples[-1]),
    }


class Q1Runner:
    def __init__(self, manifest_path: Path, output_dir: Path | None = None) -> None:
        self.manifest_path = manifest_path.resolve()
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = yaml.safe_load(handle)
        self.output_dir = (
            Path(self.manifest["output_root"]) if output_dir is None else output_dir
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.output_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.state_path = self.output_dir / "q1_state.json"
        if self.state_path.exists():
            with self.state_path.open(encoding="utf-8") as handle:
                self.state = json.load(handle)
        else:
            self.state = {
                "schema_version": "q1-clean-state-v1",
                "manifest_sha256": config_fingerprint(self.manifest),
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "status": "running",
                "runs": {},
            }
            self._save()
        if self.state["manifest_sha256"] != config_fingerprint(self.manifest):
            raise RuntimeError("Q1 manifest changed after execution started")
        self.bp_base = load_config(
            (self.manifest_path.parent / self.manifest["bp_config"]).resolve()
        )
        self.hebbian_base = load_config(
            (self.manifest_path.parent / self.manifest["hebbian_config"]).resolve()
        )
        if self.bp_base["model"] != self.hebbian_base["model"]:
            raise RuntimeError("Q1 BP and Hebbian model configs must be identical")

    def _save(self) -> None:
        self.state["updated_at_utc"] = utc_now()
        _atomic_json(self.state_path, self.state)

    def _config(self, rule: str, seed: int) -> dict:
        config = deepcopy(self.bp_base if rule == "bp" else self.hebbian_base)
        config["training"]["seed"] = seed
        validate_config(config)
        return config

    def _discover(self, fingerprint: str, rule: str) -> Path | None:
        candidates: list[tuple[int, str, Path]] = []
        for run_dir in self.runs_dir.iterdir():
            metadata_path = run_dir / "metadata.json"
            status_path = run_dir / "run_status.json"
            if not metadata_path.exists() or not status_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                metadata.get("config_sha256") == fingerprint
                and metadata.get("learning_rule") == rule
            ):
                candidates.append(
                    (
                        int(status.get("global_epoch", 0)),
                        str(status.get("created_at_utc", "")),
                        run_dir,
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda value: (-value[0], value[1]))
        return candidates[0][2]

    def _register(self, key: str, run_dir: Path, config: dict) -> None:
        self.state["runs"][key] = {
            "run_dir": str(run_dir.resolve()),
            "config_sha256": config_fingerprint(config),
            "representation": "running",
            "probe": "pending",
            "reconstruction": "pending",
        }
        self._save()

    def _representation_run(self, rule: str, seed: int) -> Path:
        key = f"{rule}_seed{seed}"
        config = self._config(rule, seed)
        entry = self.state["runs"].get(key)
        if entry is None:
            discovered = self._discover(config_fingerprint(config), rule)
            if discovered is not None:
                self._register(key, discovered, config)
                run_dir = train_config(config, resume_run_dir=discovered)
            else:
                run_dir = train_config(
                    config,
                    run_root=self.runs_dir,
                    on_run_created=lambda created: self._register(
                        key, created, config
                    ),
                )
        else:
            run_dir = Path(entry["run_dir"])
            if read_run_status(run_dir)["status"] != "completed":
                run_dir = train_config(config, resume_run_dir=run_dir)
        self.state["runs"][key]["representation"] = "completed"
        self._save()

        if not _has_metric(run_dir, stage="linear_probe_final", split="test"):
            train_linear_probe_config(config, run_dir, validation_only=False)
        self.state["runs"][key]["probe"] = "completed"
        self._save()

        if not _has_metric(run_dir, stage="reconstruction_final", split="test"):
            evaluate_config(config, run_dir)
        self.state["runs"][key]["reconstruction"] = "completed"
        self._save()
        return run_dir

    def _random_probe_run(self, seed: int) -> Path:
        key = f"random_seed{seed}"
        config = self._config("bp", seed)
        entry = self.state["runs"].get(key)
        if entry is None:
            run_dir = create_run_directory(
                self.runs_dir, rule="random_encoder", seed=seed
            )
            model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
            write_resolved_config(run_dir, config)
            write_metadata(
                run_dir,
                {
                    "version": config["version"],
                    "experiment_id": "q1-clean-v1",
                    "run_id": run_dir.name,
                    "created_at_utc": utc_now(),
                    "config_sha256": config_fingerprint(config),
                    "learning_rule": "random_encoder",
                    "model_type": "random_encoder_control",
                    "architecture_id": config["model"]["architecture"],
                    "latent_dim": config["model"]["latent_dim"],
                    "seed": seed,
                    "protocol": config.get("protocol", {}),
                    "device": "cpu",
                    "initial_state_hash": state_dict_checksum(model),
                    **git_provenance(str(Path.cwd())),
                    **model.architecture_metadata(),
                },
            )
            torch.save(model.state_dict(), run_dir / "model_best.pt")
            torch.save(model.state_dict(), run_dir / "model_last.pt")
            initialize_run_status(
                run_dir,
                {
                    "status": "completed",
                    "stage": "random_encoder",
                    "active_layer": "none",
                    "completed_epoch": 0,
                    "global_epoch": 0,
                    "samples_seen": 0,
                    "steps_completed": 0,
                    "wall_time_sec": 0.0,
                    "resume_count": 0,
                    "checkpoint": "model_best.pt",
                    "created_at_utc": utc_now(),
                    "updated_at_utc": utc_now(),
                    "error": None,
                },
            )
            self.state["runs"][key] = {
                "run_dir": str(run_dir.resolve()),
                "config_sha256": config_fingerprint(config),
                "representation": "not_trained",
                "probe": "pending",
                "reconstruction": "not_applicable",
            }
            self._save()
        else:
            run_dir = Path(entry["run_dir"])
        if not _has_metric(run_dir, stage="linear_probe_final", split="test"):
            train_linear_probe_config(config, run_dir, validation_only=False)
        self.state["runs"][key]["probe"] = "completed"
        self._save()
        return run_dir

    def _validate_pairing(self, seed: int) -> None:
        hashes = {
            rule: read_metadata(
                Path(self.state["runs"][f"{rule}_seed{seed}"]["run_dir"])
            )["initial_state_hash"]
            for rule in ("bp", "hebbian")
        }
        random_hash = read_metadata(
            Path(self.state["runs"][f"random_seed{seed}"]["run_dir"])
        )["initial_state_hash"]
        if len({*hashes.values(), random_hash}) != 1:
            raise RuntimeError(f"Paired initial-state mismatch for seed {seed}")

    def _analyze(self, analyzed_seeds: list[int] | None = None) -> None:
        seeds = [
            int(seed)
            for seed in (
                self.manifest["paired_seeds"]
                if analyzed_seeds is None
                else analyzed_seeds
            )
        ]
        if not seeds:
            raise ValueError("At least one completed seed is required for analysis")
        threshold = float(self.manifest["classification_threshold"])
        run_rows: list[dict[str, Any]] = []
        for seed in seeds:
            for rule in ("bp", "hebbian", "random"):
                run_dir = Path(self.state["runs"][f"{rule}_seed{seed}"]["run_dir"])
                final = _final_metric(
                    run_dir, stage="linear_probe_final", split="test"
                )
                validation = _final_metric(
                    run_dir, stage="linear_probe_final", split="validation"
                )
                probe_curve = [
                    row
                    for row in _metric_rows(run_dir)
                    if row.get("stage") == "linear_probe"
                    and row.get("split") == "validation"
                ]
                curve = _curve_summary(probe_curve, threshold=threshold)
                training_stage = (
                    "representation" if rule == "bp" else "hebbian_encoder"
                )
                training_rows = [
                    row
                    for row in _metric_rows(run_dir)
                    if row.get("stage") == training_stage
                    and row.get("split") in {"train", "validation"}
                ]
                if rule == "random":
                    representation_samples = 0
                    representation_wall = 0.0
                else:
                    representation_samples = max(
                        int(float(row["samples_seen"]))
                        for row in training_rows
                        if row.get("samples_seen")
                    )
                    representation_wall = max(
                        float(row["wall_time_sec"])
                        for row in training_rows
                        if row.get("wall_time_sec")
                    )
                run_status = read_run_status(run_dir)
                reconstruction = (
                    None
                    if rule == "random"
                    else float(
                        _final_metric(
                            run_dir, stage="reconstruction_final", split="test"
                        )["reconstruction_loss"]
                    )
                )
                run_rows.append(
                    {
                        "seed": seed,
                        "rule": rule,
                        "run_dir": str(run_dir),
                        "test_accuracy": float(final["accuracy"]),
                        "test_macro_f1": float(final["macro_f1"]),
                        "test_ce": float(final["classification_ce"]),
                        "validation_accuracy": float(validation["accuracy"]),
                        "test_reconstruction_mse": reconstruction,
                        "representation_samples_seen": representation_samples,
                        "representation_wall_time_sec": representation_wall,
                        "model_training_samples_seen": int(
                            run_status.get("samples_seen", 0)
                        ),
                        "model_training_wall_time_sec": float(
                            run_status.get("wall_time_sec", 0.0)
                        ),
                        **curve,
                    }
                )

        run_table = self.output_dir / "q1_run_table.csv"
        with run_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)

        metrics = ("test_accuracy", "test_macro_f1", "test_ce")
        paired_rows: list[dict[str, Any]] = []
        for seed in seeds:
            values = {
                row["rule"]: row
                for row in run_rows
                if int(row["seed"]) == int(seed)
            }
            paired_rows.append(
                {
                    "seed": seed,
                    **{
                        f"hebbian_minus_bp_{metric}": values["hebbian"][metric]
                        - values["bp"][metric]
                        for metric in metrics
                    },
                    "hebbian_minus_bp_test_reconstruction_mse": values[
                        "hebbian"
                    ]["test_reconstruction_mse"]
                    - values["bp"]["test_reconstruction_mse"],
                }
            )
        paired_path = self.output_dir / "paired_differences.csv"
        with paired_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

        summary: dict[str, Any] = {
            "schema_version": "q1-clean-summary-v1",
            "completed_at_utc": utc_now(),
            "analysis_status": (
                "complete"
                if seeds
                == [int(seed) for seed in self.manifest["paired_seeds"]]
                else "preliminary"
            ),
            "paired_seeds": seeds,
            "planned_paired_seeds": [
                int(seed) for seed in self.manifest["paired_seeds"]
            ],
            "classification_threshold": threshold,
            "rules": {},
            "paired_differences_hebbian_minus_bp": {},
        }
        numeric_fields = (
            "test_accuracy",
            "test_macro_f1",
            "test_ce",
            "test_reconstruction_mse",
            "epoch_aulc",
            "samples_aulc",
            "wall_time_aulc",
            "representation_samples_seen",
            "representation_wall_time_sec",
            "model_training_samples_seen",
            "model_training_wall_time_sec",
            "probe_wall_time_sec",
        )
        for rule in ("bp", "hebbian", "random"):
            rule_rows = [row for row in run_rows if row["rule"] == rule]
            summary["rules"][rule] = {}
            for field in numeric_fields:
                values = [
                    float(row[field])
                    for row in rule_rows
                    if row[field] is not None
                ]
                if values:
                    summary["rules"][rule][field] = {
                        "mean": float(np.mean(values)),
                        "sd": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1
                            else None
                        ),
                    }
            threshold_values = [
                row["samples_to_threshold"]
                for row in rule_rows
                if row["samples_to_threshold"] is not None
            ]
            summary["rules"][rule]["threshold_reached_seeds"] = len(
                threshold_values
            )
            if threshold_values:
                summary["rules"][rule]["samples_to_threshold_mean"] = float(
                    np.mean(threshold_values)
                )

        for field in (
            "test_accuracy",
            "test_macro_f1",
            "test_ce",
            "test_reconstruction_mse",
        ):
            differences = np.asarray(
                [
                    row[f"hebbian_minus_bp_{field}"]
                    for row in paired_rows
                ],
                dtype=np.float64,
            )
            low, high = _bootstrap_ci(
                differences,
                seed=int(self.manifest["bootstrap_seed"]),
                resamples=int(self.manifest["bootstrap_resamples"]),
            )
            summary["paired_differences_hebbian_minus_bp"][field] = {
                "mean": float(differences.mean()),
                "sd": (
                    float(differences.std(ddof=1))
                    if len(differences) > 1
                    else None
                ),
                "bootstrap_95_ci": [low, high],
            }
        _atomic_json(self.output_dir / "q1_summary.json", summary)
        self._plot(run_rows, seeds)
        self._plot_learning_curves(seeds)

    def _plot(self, run_rows: list[dict[str, Any]], seeds: list[int]) -> None:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(self.output_dir / ".matplotlib-cache")
        )
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for rule, color in (("bp", "tab:blue"), ("hebbian", "tab:orange")):
            values = [
                next(
                    row["test_accuracy"]
                    for row in run_rows
                    if row["rule"] == rule and int(row["seed"]) == int(seed)
                )
                for seed in seeds
            ]
            axes[0].plot(seeds, np.asarray(values) * 100, marker="o", label=rule)
        axes[0].set_title("Paired test accuracy")
        axes[0].set_xlabel("Seed")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        labels = ["BP", "Hebbian", "Random"]
        means = [
            np.mean(
                [row["test_accuracy"] for row in run_rows if row["rule"] == rule]
            )
            * 100
            for rule in ("bp", "hebbian", "random")
        ]
        errors = [
            np.std(
                [row["test_accuracy"] for row in run_rows if row["rule"] == rule],
                ddof=1,
            )
            * 100
            for rule in ("bp", "hebbian", "random")
        ]
        axes[1].bar(
            labels,
            means,
            yerr=errors,
            capsize=5,
            color=["tab:blue", "tab:orange", "tab:gray"],
        )
        axes[1].axhline(10, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Mean test accuracy ± SD")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(axis="y", alpha=0.25)
        figure.suptitle(
            "Q1 clean performance, paired seeds "
            + ", ".join(str(seed) for seed in seeds)
        )
        figure.tight_layout()
        figure.savefig(self.output_dir / "q1_clean_accuracy.png", dpi=180)
        plt.close(figure)

    def _plot_learning_curves(self, seeds: list[int]) -> None:
        import matplotlib.pyplot as plt

        curves: dict[str, list[list[dict[str, str]]]] = {
            rule: [] for rule in ("bp", "hebbian", "random")
        }
        for rule in curves:
            for seed in seeds:
                run_dir = Path(
                    self.state["runs"][f"{rule}_seed{seed}"]["run_dir"]
                )
                rows = [
                    row
                    for row in _metric_rows(run_dir)
                    if row.get("stage") == "linear_probe"
                    and row.get("split") == "validation"
                ]
                curves[rule].append(
                    sorted(rows, key=lambda row: int(row["epoch"]))
                )

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        specifications = (
            ("epoch", "Probe epoch"),
            ("samples_seen", "Probe samples seen (millions)"),
            ("wall_time_sec", "Probe wall time (seconds)"),
        )
        colors = {
            "bp": "tab:blue",
            "hebbian": "tab:orange",
            "random": "tab:gray",
        }
        for axis, (x_key, x_label) in zip(axes, specifications):
            for rule, rule_curves in curves.items():
                x_arrays = [
                    np.asarray([float(row[x_key]) for row in curve])
                    for curve in rule_curves
                ]
                y_arrays = [
                    np.asarray([float(row["accuracy"]) for row in curve]) * 100
                    for curve in rule_curves
                ]
                common_low = max(values[0] for values in x_arrays)
                common_high = min(values[-1] for values in x_arrays)
                if x_key == "epoch":
                    grid = np.arange(common_low, common_high + 1)
                elif x_key == "samples_seen":
                    grid = x_arrays[0][
                        (x_arrays[0] >= common_low)
                        & (x_arrays[0] <= common_high)
                    ]
                else:
                    grid = np.linspace(common_low, common_high, 100)
                aligned = np.vstack(
                    [
                        np.interp(grid, x_values, y_values)
                        for x_values, y_values in zip(x_arrays, y_arrays)
                    ]
                )
                mean = aligned.mean(axis=0)
                sd = (
                    aligned.std(axis=0, ddof=1)
                    if aligned.shape[0] > 1
                    else np.zeros_like(mean)
                )
                display_grid = grid / 1_000_000 if x_key == "samples_seen" else grid
                axis.plot(display_grid, mean, color=colors[rule], label=rule)
                axis.fill_between(
                    display_grid,
                    mean - sd,
                    mean + sd,
                    color=colors[rule],
                    alpha=0.16,
                )
            axis.axhline(
                float(self.manifest["classification_threshold"]) * 100,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel("Validation accuracy (%)")
            axis.grid(alpha=0.25)
        axes[0].legend()
        figure.suptitle(
            f"Frozen-probe learning curves, mean ± SD (n={len(seeds)} seeds)"
        )
        figure.tight_layout()
        figure.savefig(self.output_dir / "q1_learning_curves.png", dpi=180)
        plt.close(figure)

    def run(self, stop_after_seed: int | None = None) -> Path:
        planned_seeds = [int(seed) for seed in self.manifest["paired_seeds"]]
        if stop_after_seed is not None and stop_after_seed not in planned_seeds:
            raise ValueError(
                f"stop_after_seed={stop_after_seed} is not in {planned_seeds}"
            )
        completed_seeds: list[int] = []
        for seed in planned_seeds:
            for rule in ("bp", "hebbian"):
                print(f"starting rule={rule} seed={seed}", flush=True)
                self._representation_run(rule, int(seed))
            if self.manifest.get("random_encoder_control", True):
                self._random_probe_run(int(seed))
            self._validate_pairing(int(seed))
            completed_seeds.append(int(seed))
            if stop_after_seed is not None and int(seed) == stop_after_seed:
                break
        self._analyze(completed_seeds)
        if completed_seeds == planned_seeds:
            self.state["status"] = "completed"
            self.state["completed_at_utc"] = utc_now()
            self.state.pop("paused_after_seed", None)
            self.state.pop("paused_at_utc", None)
        else:
            self.state["status"] = "paused"
            self.state["paused_after_seed"] = completed_seeds[-1]
            self.state["paused_at_utc"] = utc_now()
        self._save()
        return self.output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/experiments/q1_clean_v1.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stop-after-seed", type=int)
    args = parser.parse_args()
    runner = Q1Runner(
        Path(args.manifest),
        None if args.output_dir is None else Path(args.output_dir),
    )
    print(runner.run(stop_after_seed=args.stop_after_seed))


if __name__ == "__main__":
    main()
