"""Sequential, resumable validation-only tuning for BP and Hebbian models."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from schemas import load_config, validate_config
from training.train_linear_probe import train_linear_probe_config
from training.train_representation import train_config
from utils.checkpointing import config_fingerprint, utc_now


@dataclass
class TrialResult:
    trial_id: str
    rule: str
    stage: str
    config_sha256: str
    run_dir: str
    status: str
    validation_accuracy: float | None
    validation_macro_f1: float | None
    validation_ce: float | None
    reused_from: str | None
    error: str | None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(temporary, path)


def _read_validation_result(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    result = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") == "linear_probe_final" and row.get("split") == "validation":
                result = {
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "classification_ce": float(row["classification_ce"]),
                }
    return result


def _assert_no_test_metrics(run_dir: Path) -> None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        test_rows = [row for row in csv.DictReader(handle) if row.get("split") == "test"]
    if test_rows:
        raise RuntimeError(f"Validation-only trial contains {len(test_rows)} test rows: {run_dir}")


def _select(results: list[TrialResult]) -> TrialResult:
    successful = [result for result in results if result.status == "completed"]
    if not successful:
        raise RuntimeError("No successful trial is available for selection")
    return min(
        successful,
        key=lambda result: (
            -float(result.validation_accuracy),
            float(result.validation_ce),
            result.trial_id,
        ),
    )


def _write_trial_table(path: Path, results: list[TrialResult]) -> None:
    fields = list(TrialResult.__dataclass_fields__)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    os.replace(temporary, path)


class ValidationTuner:
    def __init__(self, manifest_path: Path, output_dir: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = yaml.safe_load(handle)
        base_path = (self.manifest_path.parent / self.manifest["base_config"]).resolve()
        self.base = load_config(base_path, validate=False)
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.output_dir / "runs"
        self.configs_dir = self.output_dir / "configs"
        self.state_path = self.output_dir / "tuning_state.json"
        if self.state_path.exists():
            with self.state_path.open(encoding="utf-8") as handle:
                self.state = json.load(handle)
        else:
            self.state = {
                "schema_version": "validation-tuning-state-v1",
                "manifest": str(self.manifest_path),
                "manifest_sha256": config_fingerprint(self.manifest),
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "physical_trials": {},
                "logical_results": {},
                "status": "running",
            }
            _atomic_json(self.state_path, self.state)
        if self.state["manifest_sha256"] != config_fingerprint(self.manifest):
            raise RuntimeError("Tuning manifest changed after the run started")

    def _save_state(self) -> None:
        self.state["updated_at_utc"] = utc_now()
        _atomic_json(self.state_path, self.state)

    def _discover_run(self, fingerprint: str) -> Path | None:
        """Recover a run created before the runner could register it."""

        candidates: list[tuple[int, str, Path]] = []
        if not self.runs_dir.exists():
            return None
        for run_dir in self.runs_dir.iterdir():
            metadata_path = run_dir / "metadata.json"
            status_path = run_dir / "run_status.json"
            if not metadata_path.exists() or not status_path.exists():
                continue
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("config_sha256") != fingerprint:
                continue
            with status_path.open(encoding="utf-8") as handle:
                status = json.load(handle)
            if status.get("status") == "failed" and "Superseded duplicate" in str(
                status.get("error")
            ):
                continue
            candidates.append(
                (
                    int(status.get("global_epoch", 0)),
                    str(status.get("created_at_utc", "")),
                    run_dir,
                )
            )
        if not candidates:
            return None
        # Prefer most progress; for ties preserve the earliest/original run.
        candidates.sort(key=lambda value: (-value[0], value[1]))
        return candidates[0][2]

    def _register_physical(
        self,
        fingerprint: str,
        run_dir: Path,
        *,
        trial_id: str,
        rule: str,
    ) -> None:
        self.state["physical_trials"][fingerprint] = {
            "run_dir": str(run_dir.resolve()),
            "first_trial_id": trial_id,
            "rule": rule,
        }
        self._save_state()

    def hebbian_config(self, *, lr: float, winner: float, latent_dim: int) -> dict:
        config = deepcopy(self.base)
        config["training"]["learning_rule"] = "hebbian"
        config["training"]["seed"] = int(self.manifest["tuning_seed"])
        config["training"]["encoder_only_tuning"] = True
        config["model"]["latent_dim"] = int(latent_dim)
        config["hebbian"]["lr"] = float(lr)
        config["hebbian"]["winner_fraction"] = float(winner)
        config["hebbian"].pop("layer_lrs", None)
        config["hebbian"]["update"] = "oja"
        config["hebbian"]["filter_l2_normalize"] = True
        validate_config(config)
        return config

    def bp_config(self, *, lr: float, weight_decay: float) -> dict:
        config = deepcopy(self.base)
        config["training"]["learning_rule"] = "bp"
        config["training"]["seed"] = int(self.manifest["tuning_seed"])
        config["backprop"]["lr"] = float(lr)
        config["backprop"]["weight_decay"] = float(weight_decay)
        validate_config(config)
        return config

    def execute(self, trial_id: str, rule: str, stage: str, config: dict) -> TrialResult:
        if trial_id in self.state["logical_results"]:
            saved = self.state["logical_results"][trial_id]
            result = TrialResult(**saved)
            if result.status == "completed":
                return result
        config_path = self.configs_dir / f"{trial_id}.yaml"
        _write_yaml(config_path, config)
        fingerprint = config_fingerprint(config)
        physical = self.state["physical_trials"].get(fingerprint)
        reused_from = None
        try:
            if physical is None:
                discovered = self._discover_run(fingerprint)
                if discovered is not None:
                    run_dir = discovered
                    self._register_physical(
                        fingerprint,
                        run_dir,
                        trial_id=trial_id,
                        rule=rule,
                    )
                    run_dir = train_config(config, resume_run_dir=run_dir)
                else:
                    run_dir = train_config(
                        config,
                        run_root=self.runs_dir,
                        on_run_created=lambda created: self._register_physical(
                            fingerprint,
                            created,
                            trial_id=trial_id,
                            rule=rule,
                        ),
                    )
            else:
                run_dir = Path(physical["run_dir"])
                reused_from = physical["first_trial_id"]
                if _read_validation_result(run_dir) is None:
                    run_dir = train_config(config, resume_run_dir=run_dir)
            metric = _read_validation_result(run_dir)
            if metric is None:
                train_linear_probe_config(config, run_dir, validation_only=True)
                metric = _read_validation_result(run_dir)
            if metric is None:
                raise RuntimeError("Probe completed without a final validation row")
            _assert_no_test_metrics(run_dir)
            result = TrialResult(
                trial_id=trial_id,
                rule=rule,
                stage=stage,
                config_sha256=fingerprint,
                run_dir=str(run_dir.resolve()),
                status="completed",
                validation_accuracy=metric["accuracy"],
                validation_macro_f1=metric["macro_f1"],
                validation_ce=metric["classification_ce"],
                reused_from=reused_from,
                error=None,
            )
        except Exception as error:
            result = TrialResult(
                trial_id=trial_id,
                rule=rule,
                stage=stage,
                config_sha256=fingerprint,
                run_dir="" if physical is None else physical["run_dir"],
                status="failed",
                validation_accuracy=None,
                validation_macro_f1=None,
                validation_ce=None,
                reused_from=reused_from,
                error=f"{type(error).__name__}: {error}",
            )
        self.state["logical_results"][trial_id] = asdict(result)
        self._save_state()
        _write_trial_table(
            self.output_dir / "trial_table.csv",
            [TrialResult(**value) for value in self.state["logical_results"].values()],
        )
        print(
            f"trial={trial_id} status={result.status} "
            f"validation_accuracy={result.validation_accuracy} reused_from={reused_from}",
            flush=True,
        )
        return result

    def run(self) -> Path:
        hebbian_manifest = self.manifest["hebbian"]
        fixed_winner = float(hebbian_manifest["fixed_winner_fraction_for_lr"])
        fixed_latent = int(hebbian_manifest["fixed_latent_dim"])

        lr_results = []
        for lr in hebbian_manifest["learning_rates"]:
            trial_id = f"hebb_lr_{float(lr):g}".replace(".", "p")
            lr_results.append(
                self.execute(
                    trial_id,
                    "hebbian",
                    "hebbian_lr",
                    self.hebbian_config(lr=float(lr), winner=fixed_winner, latent_dim=fixed_latent),
                )
            )
        best_lr_result = _select(lr_results)
        best_lr = float(
            yaml.safe_load((self.configs_dir / f"{best_lr_result.trial_id}.yaml").read_text(encoding="utf-8"))["hebbian"]["lr"]
        )

        winner_results = []
        for winner in hebbian_manifest["winner_fractions"]:
            trial_id = f"hebb_wta_{float(winner):g}".replace(".", "p")
            winner_results.append(
                self.execute(
                    trial_id,
                    "hebbian",
                    "hebbian_winner_fraction",
                    self.hebbian_config(lr=best_lr, winner=float(winner), latent_dim=fixed_latent),
                )
            )
        best_winner_result = _select(winner_results)
        best_winner = float(
            yaml.safe_load((self.configs_dir / f"{best_winner_result.trial_id}.yaml").read_text(encoding="utf-8"))["hebbian"]["winner_fraction"]
        )

        latent_results = []
        for latent_dim in hebbian_manifest["coarse_latent_dims"]:
            trial_id = f"hebb_latent_{int(latent_dim)}"
            latent_results.append(
                self.execute(
                    trial_id,
                    "hebbian",
                    "hebbian_latent_dim",
                    self.hebbian_config(lr=best_lr, winner=best_winner, latent_dim=int(latent_dim)),
                )
            )
        best_hebbian_dimension_result = _select(latent_results)
        # Q1 must keep one shared forward architecture. BP tuning is performed
        # at the frozen L=64 architecture, so the primary Hebbian config is the
        # best L=64 winner-fraction trial. The dimension winner is retained as
        # a Q5 screen, not substituted into the main BP/Hebbian comparison.
        best_hebbian_result = best_winner_result

        bp_results = []
        for lr in self.manifest["backprop"]["learning_rates"]:
            for weight_decay in self.manifest["backprop"]["weight_decays"]:
                trial_id = (
                    f"bp_lr_{float(lr):g}_wd_{float(weight_decay):g}".replace(".", "p")
                )
                bp_results.append(
                    self.execute(
                        trial_id,
                        "bp",
                        "bp_lr_weight_decay",
                        self.bp_config(lr=float(lr), weight_decay=float(weight_decay)),
                    )
                )
        best_bp_result = _select(bp_results)

        physical_counts = {"hebbian": 0, "bp": 0}
        for value in self.state["physical_trials"].values():
            physical_counts[value["rule"]] += 1
        maximum = int(self.manifest["max_unique_trials_per_rule"])
        if physical_counts != {"hebbian": maximum, "bp": maximum}:
            raise RuntimeError(f"Unexpected physical trial counts: {physical_counts}")
        for physical in self.state["physical_trials"].values():
            _assert_no_test_metrics(Path(physical["run_dir"]))

        best_hebbian_config = yaml.safe_load(
            (self.configs_dir / f"{best_hebbian_result.trial_id}.yaml").read_text(encoding="utf-8")
        )
        best_hebbian_config["training"].pop("encoder_only_tuning", None)
        best_bp_config = yaml.safe_load(
            (self.configs_dir / f"{best_bp_result.trial_id}.yaml").read_text(encoding="utf-8")
        )
        selected_dir = Path("configs/selected")
        _write_yaml(selected_dir / "hebbian_validation_selected.yaml", best_hebbian_config)
        _write_yaml(selected_dir / "bp_validation_selected.yaml", best_bp_config)
        decision = {
            "schema_version": "validation-tuning-decision-v1",
            "completed_at_utc": utc_now(),
            "selection_split": "validation",
            "test_metric_rows": 0,
            "physical_trial_counts": physical_counts,
            "stabilization": hebbian_manifest["stabilization"],
            "stabilization_trial_count": 0,
            "best_hebbian": asdict(best_hebbian_result),
            "best_hebbian_dimension_screen": asdict(
                best_hebbian_dimension_result
            ),
            "best_bp": asdict(best_bp_result),
            "selected_config_hashes": {
                "hebbian": config_fingerprint(best_hebbian_config),
                "bp": config_fingerprint(best_bp_config),
            },
            "selection_rule": "max validation accuracy; then min validation CE; then lexical trial id",
            "latent_search_note": hebbian_manifest["note"],
            "architecture_constraint": (
                "Primary BP/Hebbian configs both use latent_dim=64. The L=128 "
                "Hebbian winner is retained only as a Q5 dimension-screen result."
            ),
        }
        _atomic_json(self.output_dir / "selection_decision.json", decision)
        self.state["status"] = "completed"
        self.state["selection_decision"] = str(
            (self.output_dir / "selection_decision.json").resolve()
        )
        self._save_state()
        return self.output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/tuning/validation_tuning_v1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tuning/validation_tuning_v1",
    )
    args = parser.parse_args()
    tuner = ValidationTuner(Path(args.manifest), Path(args.output_dir))
    print(tuner.run())


if __name__ == "__main__":
    main()
