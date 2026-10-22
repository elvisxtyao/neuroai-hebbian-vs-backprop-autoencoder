"""Run and gate the five-seed RBB/RRB Stage 3 matched controls."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from schemas import load_config, validate_config
from training.run_stage3_formal_core import (
    _csv_values_finite,
    _numbers_finite,
    _required_artifacts,
)
from training.train_hybrid import train_hybrid_config
from training.train_linear_probe import train_linear_probe_config
from training.train_standardized_decoder import train_standardized_decoder_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance
from utils.results import read_run_status, write_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("random_hbb", "random_rrb")
RULES = {
    "random_hbb": {"enc1": "frozen", "enc2": "bp", "enc3": "bp"},
    "random_rrb": {"enc1": "frozen", "enc2": "frozen", "enc3": "bp"},
}
CORE_REFERENCE = {
    "random_hbb": "hybrid_hbb",
    "random_rrb": "hybrid_hhb",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("version") != "stage3-matched-controls-v1":
        raise ValueError("Unsupported matched-control protocol")
    if tuple(protocol.get("seeds", ())) != SEEDS:
        raise ValueError("Matched-control seeds must be [0,1,2,3,4]")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("Matched-control methods changed")
    if protocol.get("pre_freeze_test_access_policy") != "validation_only":
        raise ValueError("Matched controls must remain validation-only before freeze")
    if protocol.get("post_freeze_test_access_policy") != (
        "single_evaluation_per_checkpoint"
    ):
        raise ValueError("Matched-control test policy changed")
    if set(protocol.get("method_configs", {})) != set(METHODS):
        raise ValueError("Matched-control config map is incomplete")


def resolved_config(
    protocol: dict[str, Any], *, method: str, seed: int
) -> dict[str, Any]:
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown matched-control method or seed")
    config = deepcopy(load_config(_resolve(protocol["method_configs"][method])))
    config["training"]["seed"] = seed
    validate_config(config)
    if config["hybrid"]["method_id"] != method:
        raise RuntimeError("Matched-control method/config mismatch")
    if config["hybrid"]["encoder_layer_rules"] != RULES[method]:
        raise RuntimeError("Matched-control layer allocation changed")
    if float(config["backprop"]["lr"]) != 0.003:
        raise RuntimeError("Matched-control BP learning rate changed")
    return config


def _run_record(run_dir: Path) -> dict[str, Any]:
    missing = [
        str(path.relative_to(run_dir))
        for path in _required_artifacts(run_dir)
        if not path.exists()
    ]
    metadata = _json(run_dir / "metadata.json")
    training = _json(run_dir / "hybrid_training_summary.json")
    probe = _json(run_dir / "linear_probe_summary.json")
    standard = _json(
        run_dir / "standardized_decoder" / "standardized_decoder_summary.json"
    )
    statuses = (
        read_run_status(run_dir),
        read_run_status(run_dir / "standardized_decoder"),
    )
    test_access = max(
        int(metadata["test_samples_accessed"]),
        int(training["test_samples_accessed"]),
        int(probe["test_samples_accessed"]),
        int(standard["test_samples_accessed"]),
        *(int(status["test_samples_accessed"]) for status in statuses),
    )
    finite = (
        _numbers_finite(training)
        and _numbers_finite(probe)
        and _numbers_finite(standard)
        and _csv_values_finite(run_dir / "metrics.csv")
        and _csv_values_finite(run_dir / "standardized_decoder" / "metrics.csv")
    )
    return {
        "run_dir": str(run_dir),
        "git_commit": metadata["git_commit"],
        "git_worktree_dirty": metadata["git_worktree_dirty"],
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "initial_state_hash": metadata["initial_state_hash"],
        "initial_decoder_hash": metadata["initial_decoder_hash"],
        "classifier_initial_hash": probe["classifier_initial_hash"],
        "standardized_decoder_initial_hash": standard["decoder_initial_hash"],
        "frozen_layers_unchanged": training["frozen_layers_unchanged"],
        "standardized_encoder_unchanged": standard["encoder_unchanged"],
        "artifact_complete": not missing,
        "missing_artifacts": missing,
        "statuses_complete": all(status["status"] == "completed" for status in statuses),
        "metrics_finite": finite,
        "test_samples_accessed": test_access,
        "source_checkpoint_sha256": file_sha256(run_dir / "model_best.pt"),
    }


def run(
    protocol_path: str | Path,
    *,
    selected_seed: int | None = None,
    selected_method: str | None = None,
) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if selected_seed is not None and selected_seed not in SEEDS:
        raise ValueError("Selected seed must be 0..4")
    if selected_method is not None and selected_method not in METHODS:
        raise ValueError("Selected matched-control method is invalid")
    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Matched controls require a clean source worktree")
    core_dir = _resolve(protocol["core_results_dir"])
    core_gate = _json(core_dir / "freeze_gate.json")
    core_manifest = _json(core_dir / "run_manifest.json")
    if core_gate.get("decision") != "PASS":
        raise RuntimeError("Core technical freeze gate is not PASS")
    output_dir = _resolve(protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = (selected_seed,) if selected_seed is not None else SEEDS
    methods = (selected_method,) if selected_method is not None else METHODS
    for seed in seeds:
        for method in methods:
            config = resolved_config(protocol, method=method, seed=seed)
            run_dir = output_dir / "runs" / f"seed_{seed}" / method
            train_hybrid_config(config, run_dir=run_dir)
            if read_run_status(run_dir)["status"] != "completed":
                raise RuntimeError(f"Incomplete matched control: seed={seed} {method}")
            if not (run_dir / "linear_probe_summary.json").exists():
                train_linear_probe_config(config, run_dir, validation_only=True)
            train_standardized_decoder_config(config, run_dir=run_dir)
            print(f"completed seed={seed} method={method} run_dir={run_dir}", flush=True)

    all_dirs = {
        (seed, method): output_dir / "runs" / f"seed_{seed}" / method
        for seed in SEEDS
        for method in METHODS
    }
    if not all(
        (path / "hybrid_training_summary.json").exists()
        and (path / "linear_probe_summary.json").exists()
        and (path / "standardized_decoder" / "standardized_decoder_summary.json").exists()
        for path in all_dirs.values()
    ):
        return output_dir

    records: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        seed_rows: dict[str, Any] = {}
        for method in METHODS:
            row = _run_record(all_dirs[(seed, method)])
            core = core_manifest["records"][
                f"seed_{seed}/{CORE_REFERENCE[method]}"
            ]
            row["paired_core_method"] = CORE_REFERENCE[method]
            row["paired_initial_state"] = (
                row["initial_state_hash"] == core["initial_state_hash"]
            )
            row["paired_system_decoder"] = (
                row["initial_decoder_hash"] == core["initial_decoder_hash"]
            )
            row["paired_probe"] = (
                row["classifier_initial_hash"] == core["classifier_initial_hash"]
            )
            row["paired_standardized_decoder"] = (
                row["standardized_decoder_initial_hash"]
                == core["standardized_decoder_initial_hash"]
            )
            row["paired_split"] = (
                row["split_manifest_sha256"] == core["split_manifest_sha256"]
            )
            seed_rows[method] = row
            records[f"seed_{seed}/{method}"] = row
        checks = {
            "two_methods_complete": len(seed_rows) == 2,
            "paired_core_initialization": all(
                row["paired_initial_state"] for row in seed_rows.values()
            ),
            "paired_system_decoder": all(
                row["paired_system_decoder"] for row in seed_rows.values()
            ),
            "paired_probe": all(row["paired_probe"] for row in seed_rows.values()),
            "paired_standardized_decoder": all(
                row["paired_standardized_decoder"] for row in seed_rows.values()
            ),
            "paired_split": all(row["paired_split"] for row in seed_rows.values()),
            "frozen_prefixes_unchanged": all(
                row["frozen_layers_unchanged"] for row in seed_rows.values()
            ),
            "standardized_encoders_unchanged": all(
                row["standardized_encoder_unchanged"] for row in seed_rows.values()
            ),
            "artifacts_complete": all(
                row["artifact_complete"] and row["statuses_complete"]
                for row in seed_rows.values()
            ),
            "finite_metrics": all(row["metrics_finite"] for row in seed_rows.values()),
            "zero_test_access": all(
                row["test_samples_accessed"] == 0 for row in seed_rows.values()
            ),
        }
        per_seed[str(seed)] = {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        }

    global_checks = {
        "all_five_seeds_pass": all(
            per_seed[str(seed)]["decision"] == "PASS" for seed in SEEDS
        ),
        "all_10_artifacts_complete": all(
            row["artifact_complete"] and row["statuses_complete"]
            for row in records.values()
        ),
        "all_metrics_finite": all(row["metrics_finite"] for row in records.values()),
        "same_clean_source_commit": len(
            {row["git_commit"] for row in records.values()}
        )
        == 1
        and not any(row["git_worktree_dirty"] for row in records.values()),
        "protocol_source_matches_runs": {
            row["git_commit"] for row in records.values()
        }
        == {provenance["git_commit"]},
        "zero_test_access": all(
            row["test_samples_accessed"] == 0 for row in records.values()
        ),
    }
    gate = {
        "schema_version": "stage3-matched-controls-freeze-gate-v1",
        "completed_at_utc": utc_now(),
        "decision": "PASS" if all(global_checks.values()) else "FAIL",
        "performance_gate_applied": False,
        "global_checks": global_checks,
        "per_seed": per_seed,
        "test_samples_accessed": 0,
    }
    write_json(output_dir, "freeze_gate.json", gate, overwrite=True)
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-matched-controls-runs-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "core_manifest_sha256": file_sha256(core_dir / "run_manifest.json"),
            "records": records,
            "freeze_gate": gate["decision"],
            "test_samples_accessed": 0,
            **provenance,
        },
        overwrite=True,
    )
    if gate["decision"] != "PASS":
        raise RuntimeError("Matched-control technical freeze gate failed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/stage3_matched_controls_v1.yaml",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    args = parser.parse_args()
    print(
        run(
            args.config,
            selected_seed=args.seed,
            selected_method=args.method,
        ).resolve()
    )


if __name__ == "__main__":
    main()
