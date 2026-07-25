"""Resumable validation-only Stage 1B Hebbian repair selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluation.representation_health import (
    assess_layer_health,
    compute_layer_health,
)
from evaluation.representations import extract_representations
from evaluation.run_representation_health import (
    _validation_loader,
    prepare_subset_manifest,
)
from models import ConvAutoencoder
from schemas import load_config, validate_config
from training.train_linear_probe import train_linear_probe_config
from training.train_representation import train_config
from utils.checkpointing import config_fingerprint, file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("h1", "h2", "z")


@dataclass(frozen=True)
class Stage1BResult:
    trial_id: str
    config_sha256: str
    run_dir: str
    competition_mode: str
    competition_power: float
    winner_fraction: float
    validation_accuracy: float | None
    validation_macro_f1: float | None
    validation_ce: float | None
    health_pass: bool
    accuracy_pass: bool
    eligible: bool
    h1_effective_rank: float | None
    h2_effective_rank: float | None
    z_effective_rank: float | None
    h1_winner_coverage: float | None
    h2_winner_coverage: float | None
    z_winner_coverage: float | None
    status: str
    error: str | None
    update_centering: str = "none"


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(temporary, path)


def _write_table(path: Path, results: list[Stage1BResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(Stage1BResult.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    os.replace(temporary, path)


def _validation_metric(run_dir: Path) -> dict[str, float] | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    result = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["stage"] == "linear_probe_final"
                and row["split"] == "validation"
            ):
                result = {
                    "accuracy": float(row["accuracy"]),
                    "macro_f1": float(row["macro_f1"]),
                    "classification_ce": float(row["classification_ce"]),
                }
            if row["split"] == "test":
                raise RuntimeError(f"Stage 1B trial contains a test row: {run_dir}")
    return result


def select_eligible(results: list[Stage1BResult]) -> Stage1BResult | None:
    eligible = [
        result
        for result in results
        if result.status == "completed" and result.eligible
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda result: (
            -float(result.validation_accuracy),
            float(result.validation_ce),
            result.trial_id,
        ),
    )


class Stage1BRunner:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.manifest = yaml.safe_load(handle)
        if self.manifest.get("version") not in {
            "stage1b-homeostasis-v1",
            "stage1b-centered-v2",
            "output-filter-centering-mechanism-v1",
        }:
            raise ValueError("Unsupported Stage 1B manifest version")
        if self.manifest.get("version") == "output-filter-centering-mechanism-v1":
            candidates = self.manifest.get("candidates", [])
            if len(candidates) != 1:
                raise ValueError(
                    "Output-filter centering experiment must freeze one candidate"
                )
            if candidates[0].get("update_centering") != "output_filters":
                raise ValueError("The frozen candidate must center output filters")
        manifest_dir = self.manifest_path.parent
        self.base = load_config(
            _resolve(manifest_dir, self.manifest["base_config"]),
            validate=False,
        )
        health_path = _resolve(manifest_dir, self.manifest["health_config"])
        with health_path.open(encoding="utf-8") as handle:
            self.health_config = yaml.safe_load(handle)
        self.output_dir = _resolve(ROOT, self.manifest["output_dir"])
        self.runs_dir = self.output_dir / "runs"
        self.configs_dir = self.output_dir / "configs"
        self.health_dir = self.output_dir / "health"
        self.state_path = self.output_dir / "stage1b_state.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_snapshot = self.output_dir / "manifest_resolved.yaml"
        if manifest_snapshot.exists():
            existing_manifest = yaml.safe_load(
                manifest_snapshot.read_text(encoding="utf-8")
            )
            if config_fingerprint(existing_manifest) != config_fingerprint(
                self.manifest
            ):
                raise RuntimeError(
                    "Saved validation manifest differs from requested manifest"
                )
        else:
            _atomic_yaml(manifest_snapshot, self.manifest)
        provenance = git_provenance(str(ROOT))
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            if provenance["git_worktree_dirty"]:
                raise RuntimeError("Stage 1B must start from a clean Git worktree")
            self.state = {
                "schema_version": "stage1b-state-v1",
                "manifest": str(self.manifest_path),
                "manifest_sha256": config_fingerprint(self.manifest),
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "status": "running",
                **provenance,
                "trials": {},
            }
            _atomic_json(self.state_path, self.state)
        if self.state["manifest_sha256"] != config_fingerprint(self.manifest):
            raise RuntimeError("Stage 1B manifest changed after execution started")

        subset_path, self.sample_ids, self.sample_labels = prepare_subset_manifest(
            self.health_config
        )
        self.subset_path = subset_path
        self.loader = _validation_loader(self.health_config, self.sample_ids)
        self.minimum_accuracy = float(
            self.manifest["baseline_validation_accuracy"]
        ) - float(self.manifest["noninferiority_margin"])

    def _save(self) -> None:
        self.state["updated_at_utc"] = utc_now()
        _atomic_json(self.state_path, self.state)

    def candidate_config(self, candidate: dict[str, Any]) -> dict[str, Any]:
        config = deepcopy(self.base)
        fixed = self.manifest["fixed"]
        config["training"]["learning_rule"] = "hebbian"
        config["training"]["seed"] = int(self.manifest["tuning_seed"])
        config["training"]["encoder_only_tuning"] = True
        config["training"]["hebbian_epochs_per_layer"] = int(
            fixed["epochs_per_layer"]
        )
        config["model"]["latent_dim"] = int(fixed["latent_dim"])
        config["hebbian"]["lr"] = float(
            candidate.get("learning_rate", fixed["learning_rate"])
        )
        config["hebbian"]["winner_fraction"] = float(
            candidate["winner_fraction"]
        )
        config["hebbian"]["competition_mode"] = candidate["competition_mode"]
        config["hebbian"]["competition_power"] = float(
            candidate["competition_power"]
        )
        config["hebbian"]["competition_epsilon"] = 1e-6
        config["hebbian"]["center_inputs"] = bool(
            candidate.get("center_inputs", False)
        )
        config["hebbian"]["update_centering"] = candidate.get(
            "update_centering", "none"
        )
        config["hebbian"]["filter_l2_normalize"] = bool(
            fixed["filter_l2_normalize"]
        )
        config["hebbian"]["update"] = fixed["update"]
        config["hebbian"].pop("layer_lrs", None)
        validate_config(config)
        return config

    def _discover(self, fingerprint: str) -> Path | None:
        if not self.runs_dir.exists():
            return None
        for run_dir in self.runs_dir.iterdir():
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config_sha256") == fingerprint:
                return run_dir
        return None

    def _register(
        self,
        trial_id: str,
        run_dir: Path,
        fingerprint: str,
    ) -> None:
        self.state["trials"][trial_id] = {
            "status": "running",
            "run_dir": str(run_dir.resolve()),
            "config_sha256": fingerprint,
        }
        self._save()

    def _health(
        self,
        trial_id: str,
        run_dir: Path,
        config: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        model = ConvAutoencoder(
            int(config["model"]["latent_dim"]),
            seed=int(config["training"]["seed"]),
        )
        checkpoint = run_dir / "model_best.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        before = state_dict_checksum(model)
        extracted = extract_representations(
            model,
            self.loader,
            device=torch.device("cpu"),
            layers=LAYERS,
        )
        if not torch.equal(
            extracted["sample_id"],
            torch.as_tensor(
                self.sample_ids,
                dtype=extracted["sample_id"].dtype,
            ),
        ):
            raise RuntimeError("Stage 1B sample ID order mismatch")
        after = state_dict_checksum(model)
        if before != after:
            raise RuntimeError("Stage 1B health extraction mutated the model")

        layers: dict[str, Any] = {}
        for layer in LAYERS:
            metrics, _, _ = compute_layer_health(
                extracted[layer],
                winner_fraction=float(config["hebbian"]["winner_fraction"]),
                activation_epsilon=float(
                    self.health_config["activation_epsilon"]
                ),
                variance_epsilon=float(
                    self.health_config["variance_epsilon"]
                ),
            )
            layers[layer] = {
                "metrics": metrics,
                **assess_layer_health(
                    metrics,
                    self.health_config["thresholds"],
                ),
            }
        payload = {
            "schema_version": "stage1b-health-v1",
            "trial_id": trial_id,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "subset_manifest": str(self.subset_path.resolve()),
            "subset_manifest_sha256": file_sha256(self.subset_path),
            "test_samples_accessed": 0,
            "layers": layers,
            "health_pass": all(value["gate_pass"] for value in layers.values()),
        }
        _atomic_json(self.health_dir / f"{trial_id}.json", payload)
        return bool(payload["health_pass"]), payload

    def execute(self, candidate: dict[str, Any]) -> Stage1BResult:
        trial_id = candidate["id"]
        saved = self.state["trials"].get(trial_id)
        if saved and saved.get("status") == "completed":
            return Stage1BResult(**saved["result"])
        config = self.candidate_config(candidate)
        fingerprint = config_fingerprint(config)
        _atomic_yaml(self.configs_dir / f"{trial_id}.yaml", config)
        try:
            run_dir = (
                Path(saved["run_dir"])
                if saved and saved.get("run_dir")
                else self._discover(fingerprint)
            )
            if run_dir is None:
                run_dir = train_config(
                    config,
                    run_root=self.runs_dir,
                    on_run_created=lambda path: self._register(
                        trial_id,
                        path,
                        fingerprint,
                    ),
                )
            else:
                status = json.loads(
                    (run_dir / "run_status.json").read_text(encoding="utf-8")
                )
                if status["status"] != "completed":
                    run_dir = train_config(config, resume_run_dir=run_dir)
            metric = _validation_metric(run_dir)
            if metric is None:
                train_linear_probe_config(
                    config,
                    run_dir,
                    validation_only=True,
                )
                metric = _validation_metric(run_dir)
            if metric is None:
                raise RuntimeError("Stage 1B probe produced no validation metric")
            health_pass, health = self._health(trial_id, run_dir, config)
            accuracy_pass = metric["accuracy"] >= self.minimum_accuracy
            layer_metrics = {
                layer: health["layers"][layer]["metrics"] for layer in LAYERS
            }
            result = Stage1BResult(
                trial_id=trial_id,
                config_sha256=fingerprint,
                run_dir=str(run_dir.resolve()),
                competition_mode=candidate["competition_mode"],
                competition_power=float(candidate["competition_power"]),
                winner_fraction=float(candidate["winner_fraction"]),
                validation_accuracy=metric["accuracy"],
                validation_macro_f1=metric["macro_f1"],
                validation_ce=metric["classification_ce"],
                health_pass=health_pass,
                accuracy_pass=accuracy_pass,
                eligible=health_pass and accuracy_pass,
                h1_effective_rank=float(layer_metrics["h1"]["effective_rank"]),
                h2_effective_rank=float(layer_metrics["h2"]["effective_rank"]),
                z_effective_rank=float(layer_metrics["z"]["effective_rank"]),
                h1_winner_coverage=float(
                    layer_metrics["h1"]["winner_coverage_ratio"]
                ),
                h2_winner_coverage=float(
                    layer_metrics["h2"]["winner_coverage_ratio"]
                ),
                z_winner_coverage=float(
                    layer_metrics["z"]["winner_coverage_ratio"]
                ),
                status="completed",
                error=None,
                update_centering=candidate.get("update_centering", "none"),
            )
        except Exception as error:
            result = Stage1BResult(
                trial_id=trial_id,
                config_sha256=fingerprint,
                run_dir="" if saved is None else saved.get("run_dir", ""),
                competition_mode=candidate["competition_mode"],
                competition_power=float(candidate["competition_power"]),
                winner_fraction=float(candidate["winner_fraction"]),
                validation_accuracy=None,
                validation_macro_f1=None,
                validation_ce=None,
                health_pass=False,
                accuracy_pass=False,
                eligible=False,
                h1_effective_rank=None,
                h2_effective_rank=None,
                z_effective_rank=None,
                h1_winner_coverage=None,
                h2_winner_coverage=None,
                z_winner_coverage=None,
                status="failed",
                error=f"{type(error).__name__}: {error}",
                update_centering=candidate.get("update_centering", "none"),
            )
        self.state["trials"][trial_id] = {
            "status": result.status,
            "run_dir": result.run_dir,
            "config_sha256": fingerprint,
            "result": asdict(result),
        }
        self._save()
        results = [
            Stage1BResult(**value["result"])
            for value in self.state["trials"].values()
            if "result" in value
        ]
        _write_table(self.output_dir / "trial_table.csv", results)
        print(
            f"trial={trial_id} status={result.status} "
            f"val_acc={result.validation_accuracy} "
            f"health_pass={result.health_pass} eligible={result.eligible}",
            flush=True,
        )
        return result

    def run(self) -> Path:
        results = [
            self.execute(candidate)
            for candidate in self.manifest["candidates"]
        ]
        selected = select_eligible(results)
        decision = {
            "schema_version": "stage1b-selection-v1",
            "completed_at_utc": utc_now(),
            "decision": "PASS" if selected is not None else "FAIL",
            "minimum_validation_accuracy": self.minimum_accuracy,
            "selection_rule": (
                "health PASS on h1/h2/z and validation accuracy above "
                "noninferiority floor; then max accuracy, min CE, lexical ID"
            ),
            "test_samples_accessed": 0,
            "selected": None if selected is None else asdict(selected),
            "trials": [asdict(result) for result in results],
        }
        if selected is not None:
            selected_config = yaml.safe_load(
                (self.configs_dir / f"{selected.trial_id}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            selected_config["training"].pop("encoder_only_tuning", None)
            _atomic_yaml(
                self.output_dir / "selected_config_validation.yaml",
                selected_config,
            )
        _atomic_json(self.output_dir / "selection_decision.json", decision)
        self.state["status"] = (
            "completed_pass" if selected is not None else "completed_fail"
        )
        self.state["selected_trial_id"] = (
            None if selected is None else selected.trial_id
        )
        self._save()
        return self.output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/tuning/stage1b_homeostasis_v1.yaml",
    )
    args = parser.parse_args()
    print(Stage1BRunner(args.manifest).run().resolve())


if __name__ == "__main__":
    main()
