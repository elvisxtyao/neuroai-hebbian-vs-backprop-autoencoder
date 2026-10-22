"""Run the frozen Stage 3 validation-only five-method, five-seed core matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from schemas import load_config, validate_config
from training.train_hybrid import train_hybrid_config
from training.train_linear_probe import train_linear_probe_config
from training.train_standardized_decoder import train_standardized_decoder_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum
from utils.results import read_run_status, write_json


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_METHODS = (
    "full_bp",
    "full_hebbian",
    "hybrid_hhb",
    "hybrid_hbb",
    "full_random",
)
EXPECTED_RULES = {
    "full_bp": {"enc1": "bp", "enc2": "bp", "enc3": "bp"},
    "full_hebbian": {
        "enc1": "hebbian",
        "enc2": "hebbian",
        "enc3": "hebbian",
    },
    "hybrid_hhb": {"enc1": "hebbian", "enc2": "hebbian", "enc3": "bp"},
    "hybrid_hbb": {"enc1": "hebbian", "enc2": "bp", "enc3": "bp"},
    "full_random": {"enc1": "frozen", "enc2": "frozen", "enc3": "frozen"},
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numbers_finite(item) for item in value)
    return True


def _csv_values_finite(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value in {None, ""}:
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    return False
    return True


def validate_stage3_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("version") != "stage3-formal-core-v1":
        raise ValueError("Unsupported Stage 3 formal protocol")
    if tuple(protocol.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Stage 3 formal seeds are frozen to [0,1,2,3,4]")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("Stage 3 core methods changed")
    if protocol.get("pre_freeze_test_access_policy") != "validation_only":
        raise ValueError("Stage 3 pre-freeze work must remain validation-only")
    if (
        protocol.get("post_freeze_test_access_policy")
        != "single_evaluation_per_checkpoint"
    ):
        raise ValueError("Stage 3 post-freeze test policy changed")
    if set(protocol.get("method_configs", {})) != set(EXPECTED_METHODS):
        raise ValueError("Stage 3 method config map is incomplete")
    semantics = protocol.get("candidate_semantics", {})
    if (
        semantics.get("hybrid_hhb")
        != "confirmed_rank_repair_candidate_with_unresolved_reconstruction_stability"
    ):
        raise ValueError("HHB candidate wording changed")
    if semantics.get("standardized_reconstruction_role") != (
        "formal_outcome_not_entry_gate"
    ):
        raise ValueError("Standardized reconstruction must remain an outcome")
    if semantics.get("stage2d_history_preserved") is not True:
        raise ValueError("Stage 2D failure history must be preserved")


def resolved_method_config(
    protocol: dict[str, Any],
    *,
    method: str,
    seed: int,
) -> dict[str, Any]:
    if method not in EXPECTED_METHODS:
        raise ValueError(f"Unknown Stage 3 method: {method}")
    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"Unknown Stage 3 seed: {seed}")
    config = deepcopy(load_config(_resolve(protocol["method_configs"][method])))
    config["training"]["seed"] = seed
    validate_config(config)
    if config["hybrid"]["method_id"] != method:
        raise RuntimeError("Protocol/config method mismatch")
    if config["hybrid"]["encoder_layer_rules"] != EXPECTED_RULES[method]:
        raise RuntimeError("Stage 3 layer allocation changed")
    if config["hybrid"].get("confirmation_stage") != "stage3_core":
        raise RuntimeError("Stage 3 config marker missing")
    if float(config["backprop"]["lr"]) != 0.003:
        raise RuntimeError("Stage 3 BP learning rate changed")
    return config


def _required_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "config_resolved.yaml",
        run_dir / "metadata.json",
        run_dir / "run_status.json",
        run_dir / "metrics.csv",
        run_dir / "model_best.pt",
        run_dir / "model_last.pt",
        run_dir / "resume_checkpoint.pt",
        run_dir / "hybrid_training_summary.json",
        run_dir / "linear_probe.pt",
        run_dir / "linear_probe_summary.json",
        run_dir / "trainable_frozen_parameter_manifest.json",
        run_dir / "standardized_decoder" / "run_status.json",
        run_dir / "standardized_decoder" / "metrics.csv",
        run_dir / "standardized_decoder" / "decoder_best.pt",
        run_dir / "standardized_decoder" / "decoder_last.pt",
        run_dir / "standardized_decoder" / "resume_checkpoint.pt",
        run_dir
        / "standardized_decoder"
        / "standardized_decoder_summary.json",
    ]


def _encoder_stage_hash(run_dir: Path, stage: str) -> str:
    state = torch.load(
        run_dir / f"encoder_{stage}_end.pt",
        map_location="cpu",
        weights_only=True,
    )
    return state_dict_checksum(state)


def run_stage3_core(
    protocol_path: str | Path,
    *,
    selected_seed: int | None = None,
    selected_method: str | None = None,
) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_stage3_protocol(protocol)
    if selected_seed is not None and selected_seed not in EXPECTED_SEEDS:
        raise ValueError("Selected seed must be one of 0,1,2,3,4")
    if selected_method is not None and selected_method not in EXPECTED_METHODS:
        raise ValueError("Unknown Stage 3 method")

    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Stage 3 requires a clean implementation worktree")
    output_dir = _resolve(protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [selected_seed] if selected_seed is not None else list(EXPECTED_SEEDS)
    methods = (
        [selected_method] if selected_method is not None else list(EXPECTED_METHODS)
    )
    for seed in seeds:
        for method in methods:
            config = resolved_method_config(protocol, method=method, seed=seed)
            run_dir = output_dir / "runs" / f"seed_{seed}" / method
            train_hybrid_config(config, run_dir=run_dir)
            if read_run_status(run_dir)["status"] != "completed":
                raise RuntimeError(f"Incomplete Stage 3 run: seed={seed} {method}")
            if not (run_dir / "linear_probe_summary.json").exists():
                train_linear_probe_config(config, run_dir, validation_only=True)
            train_standardized_decoder_config(config, run_dir=run_dir)
            print(
                f"completed seed={seed} method={method} run_dir={run_dir}",
                flush=True,
            )

    all_run_dirs = {
        (seed, method): output_dir / "runs" / f"seed_{seed}" / method
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
    }
    if not all(
        (path / "hybrid_training_summary.json").exists()
        and (path / "linear_probe_summary.json").exists()
        and (
            path
            / "standardized_decoder"
            / "standardized_decoder_summary.json"
        ).exists()
        for path in all_run_dirs.values()
    ):
        return output_dir

    records: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        seed_records: dict[str, Any] = {}
        for method in EXPECTED_METHODS:
            run_dir = all_run_dirs[(seed, method)]
            missing = [
                str(path.relative_to(run_dir))
                for path in _required_artifacts(run_dir)
                if not path.exists()
            ]
            metadata = _json(run_dir / "metadata.json")
            training = _json(run_dir / "hybrid_training_summary.json")
            probe = _json(run_dir / "linear_probe_summary.json")
            standard = _json(
                run_dir
                / "standardized_decoder"
                / "standardized_decoder_summary.json"
            )
            statuses = [
                read_run_status(run_dir),
                read_run_status(run_dir / "standardized_decoder"),
            ]
            numerical_metrics_finite = (
                _numbers_finite(training)
                and _numbers_finite(probe)
                and _numbers_finite(standard)
                and _csv_values_finite(run_dir / "metrics.csv")
                and _csv_values_finite(
                    run_dir / "standardized_decoder" / "metrics.csv"
                )
            )
            test_access = max(
                int(metadata["test_samples_accessed"]),
                int(training["test_samples_accessed"]),
                int(probe["test_samples_accessed"]),
                int(standard["test_samples_accessed"]),
                *(int(status["test_samples_accessed"]) for status in statuses),
            )
            row = {
                "seed": seed,
                "method_id": method,
                "run_dir": str(run_dir),
                "config_template": protocol["method_configs"][method],
                "resolved_config_sha256": file_sha256(
                    run_dir / "config_resolved.yaml"
                ),
                "git_commit": metadata["git_commit"],
                "git_worktree_dirty": metadata["git_worktree_dirty"],
                "split_manifest_sha256": metadata["split_manifest_sha256"],
                "initial_state_hash": metadata["initial_state_hash"],
                "initial_decoder_hash": metadata["initial_decoder_hash"],
                "classifier_initial_hash": probe["classifier_initial_hash"],
                "frozen_layers_unchanged": training["frozen_layers_unchanged"],
                "standardized_decoder_initial_hash": standard[
                    "decoder_initial_hash"
                ],
                "standardized_encoder_unchanged": standard["encoder_unchanged"],
                "source_checkpoint_sha256": file_sha256(
                    run_dir / "model_best.pt"
                ),
                "artifact_complete": not missing,
                "missing_artifacts": missing,
                "statuses_complete": all(
                    status["status"] == "completed" for status in statuses
                ),
                "numerical_metrics_finite": numerical_metrics_finite,
                "test_samples_accessed": test_access,
            }
            seed_records[method] = row
            records[f"seed_{seed}/{method}"] = row

        hhh = all_run_dirs[(seed, "full_hebbian")]
        hhb = all_run_dirs[(seed, "hybrid_hhb")]
        hbb = all_run_dirs[(seed, "hybrid_hbb")]
        prefix_hashes = {
            "hhh_enc1": _encoder_stage_hash(hhh, "enc1"),
            "hhb_enc1": _encoder_stage_hash(hhb, "enc1"),
            "hbb_enc1": _encoder_stage_hash(hbb, "enc1"),
            "hhh_enc2": _encoder_stage_hash(hhh, "enc2"),
            "hhb_enc2": _encoder_stage_hash(hhb, "enc2"),
        }
        checks = {
            "five_methods_complete": len(seed_records) == 5,
            "same_initial_model": len(
                {row["initial_state_hash"] for row in seed_records.values()}
            )
            == 1,
            "same_system_decoder_initialization": len(
                {row["initial_decoder_hash"] for row in seed_records.values()}
            )
            == 1,
            "same_standardized_decoder_initialization": len(
                {
                    row["standardized_decoder_initial_hash"]
                    for row in seed_records.values()
                }
            )
            == 1,
            "same_probe_initialization": len(
                {row["classifier_initial_hash"] for row in seed_records.values()}
            )
            == 1,
            "same_split": len(
                {row["split_manifest_sha256"] for row in seed_records.values()}
            )
            == 1,
            "same_clean_source_commit": len(
                {row["git_commit"] for row in seed_records.values()}
            )
            == 1
            and not any(
                row["git_worktree_dirty"] for row in seed_records.values()
            ),
            "hebbian_enc1_prefix_paired": len(
                {
                    prefix_hashes["hhh_enc1"],
                    prefix_hashes["hhb_enc1"],
                    prefix_hashes["hbb_enc1"],
                }
            )
            == 1,
            "hebbian_enc2_prefix_paired": (
                prefix_hashes["hhh_enc2"] == prefix_hashes["hhb_enc2"]
            ),
            "frozen_layers_unchanged": all(
                row["frozen_layers_unchanged"] for row in seed_records.values()
            ),
            "standardized_encoders_unchanged": all(
                row["standardized_encoder_unchanged"]
                for row in seed_records.values()
            ),
            "artifacts_complete": all(
                row["artifact_complete"] and row["statuses_complete"]
                for row in seed_records.values()
            ),
            "finite_metrics": all(
                row["numerical_metrics_finite"]
                for row in seed_records.values()
            ),
            "zero_test_access": all(
                row["test_samples_accessed"] == 0
                for row in seed_records.values()
            ),
        }
        per_seed[str(seed)] = {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "prefix_hashes": prefix_hashes,
        }

    global_checks = {
        "all_five_seeds_pass": all(
            per_seed[str(seed)]["decision"] == "PASS"
            for seed in EXPECTED_SEEDS
        ),
        "same_source_commit_across_matrix": len(
            {row["git_commit"] for row in records.values()}
        )
        == 1,
        "protocol_source_matches_runs": {
            row["git_commit"] for row in records.values()
        }
        == {provenance["git_commit"]},
        "all_25_artifacts_complete": all(
            row["artifact_complete"] and row["statuses_complete"]
            for row in records.values()
        ),
        "all_metrics_finite": all(
            row["numerical_metrics_finite"] for row in records.values()
        ),
        "zero_test_access": all(
            row["test_samples_accessed"] == 0 for row in records.values()
        ),
    }
    gate = {
        "schema_version": "stage3-core-freeze-gate-v1",
        "completed_at_utc": utc_now(),
        "decision": "PASS" if all(global_checks.values()) else "FAIL",
        "performance_gate_applied": False,
        "standardized_reconstruction_is_outcome": True,
        "per_seed": per_seed,
        "global_checks": global_checks,
        "test_samples_accessed": 0,
    }
    write_json(output_dir, "freeze_gate.json", gate, overwrite=True)
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-core-runs-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "protocol_document": protocol["protocol_document"],
            "protocol_document_sha256": file_sha256(
                _resolve(protocol["protocol_document"])
            ),
            "records": records,
            "freeze_gate": gate["decision"],
            "candidate_semantics": protocol["candidate_semantics"],
            "test_samples_accessed": 0,
            **provenance,
        },
        overwrite=True,
    )
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 3 technical freeze gate failed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/stage3_formal_core_v1.yaml",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    args = parser.parse_args()
    print(
        run_stage3_core(
            args.config,
            selected_seed=args.seed,
            selected_method=args.method,
        ).resolve()
    )


if __name__ == "__main__":
    main()
