"""Run the frozen Stage 3 Q5/Q6 dimension and width sweeps."""

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

from models import autoencoder_from_config
from schemas import load_config, validate_config
from training.train_hybrid import train_hybrid_config
from training.train_linear_probe import train_linear_probe_config
from training.train_standardized_decoder import train_standardized_decoder_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum
from utils.results import read_run_status, write_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("full_bp", "full_hebbian", "hybrid_hhb", "hybrid_hbb")
RULES = {
    "full_bp": {"enc1": "bp", "enc2": "bp", "enc3": "bp"},
    "full_hebbian": {
        "enc1": "hebbian",
        "enc2": "hebbian",
        "enc3": "hebbian",
    },
    "hybrid_hhb": {"enc1": "hebbian", "enc2": "hebbian", "enc3": "bp"},
    "hybrid_hbb": {"enc1": "hebbian", "enc2": "bp", "enc3": "bp"},
}
SWEEPS = ("dimension", "architecture")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _csv_finite(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value in {"", None}:
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    return False
    return True


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("version") != "stage3-q5q6-sweeps-v1":
        raise ValueError("Unsupported Q5/Q6 sweep protocol")
    if tuple(protocol.get("seeds", ())) != SEEDS:
        raise ValueError("Q5/Q6 seeds are frozen to [0,1,2,3,4]")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("Q5/Q6 methods changed")
    if float(protocol.get("bp_learning_rate", -1)) != 0.003:
        raise ValueError("Q5/Q6 BP learning rate changed")
    if protocol.get("pre_freeze_test_access_policy") != "validation_only":
        raise ValueError("Q5/Q6 training must be validation-only")
    dimensions = protocol.get("dimension_cases", {})
    if {
        key: int(value["latent_dim"]) for key, value in dimensions.items()
    } != {"L16": 16, "L32": 32, "L64": 64, "L128": 128}:
        raise ValueError("Frozen latent-dimension matrix changed")
    architectures = protocol.get("architecture_cases", {})
    actual_widths = {
        key: list(value["encoder_channels"])
        for key, value in architectures.items()
    }
    if actual_widths != {
        "early_heavy": [64, 28],
        "balanced": [16, 32],
        "late_heavy": [4, 33],
    }:
        raise ValueError("Frozen architecture matrix changed")
    counts = [
        int(value["expected_encoder_parameters"])
        for value in architectures.values()
    ]
    if (max(counts) - min(counts)) / (sum(counts) / len(counts)) >= 0.01:
        raise ValueError("Encoder parameter range/mean must remain below 1%")
    if dimensions["L64"]["source"] != "reuse_stage3_core":
        raise ValueError("L64 must reuse the accepted formal core")
    if architectures["balanced"]["source"] != "reuse_stage3_core":
        raise ValueError("Balanced architecture must reuse the formal core")


def _case_map(protocol: dict[str, Any], sweep: str) -> dict[str, Any]:
    if sweep not in SWEEPS:
        raise ValueError(f"Unknown sweep: {sweep}")
    return protocol[f"{sweep}_cases"]


def resolved_sweep_config(
    protocol: dict[str, Any],
    *,
    sweep: str,
    case: str,
    method: str,
    seed: int,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if seed not in SEEDS:
        raise ValueError("Unknown formal seed")
    if method not in METHODS:
        raise ValueError("Unknown formal method")
    cases = _case_map(protocol, sweep)
    if case not in cases:
        raise ValueError(f"Unknown {sweep} case: {case}")
    if cases[case]["source"] != "new_formal_run":
        raise ValueError(f"{sweep}/{case} must be reused, not retrained")
    config = deepcopy(load_config(_resolve(protocol["method_configs"][method])))
    config["training"]["seed"] = seed
    config["model"]["latent_dim"] = int(cases[case]["latent_dim"])
    config["model"]["encoder_channels"] = list(cases[case]["encoder_channels"])
    config["hybrid"]["confirmation_stage"] = "stage3_sweep"
    config["results"]["root"] = (
        f"results/formal/phase0_v1_1/stage3_q5q6_sweeps/"
        f"{sweep}/{case}/runs"
    )
    validate_config(config)
    if config["hybrid"]["encoder_layer_rules"] != RULES[method]:
        raise RuntimeError("Q5/Q6 method rule allocation changed")
    if float(config["backprop"]["lr"]) != 0.003:
        raise RuntimeError("Q5/Q6 BP learning rate changed")
    return config


def _run_dir(output_dir: Path, sweep: str, case: str, seed: int, method: str) -> Path:
    return output_dir / sweep / case / "runs" / f"seed_{seed}" / method


def _required(run_dir: Path) -> list[Path]:
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


def _stage_hash(run_dir: Path, layer: str) -> str:
    state = torch.load(
        run_dir / f"encoder_{layer}_end.pt",
        map_location="cpu",
        weights_only=True,
    )
    return state_dict_checksum(state)


def _validate_parameter_count(config: dict[str, Any], case_spec: dict[str, Any]) -> dict[str, int]:
    metadata = autoencoder_from_config(config, seed=0).architecture_metadata()
    counts = {
        "encoder": int(metadata["encoder_parameter_count"]),
        "decoder": int(metadata["decoder_parameter_count"]),
        "total": int(metadata["parameter_count"]),
    }
    expected = case_spec.get("expected_encoder_parameters")
    if expected is not None and counts["encoder"] != int(expected):
        raise RuntimeError(
            f"Encoder parameter count {counts['encoder']} != expected {expected}"
        )
    return counts


def _write_case_gate(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    output_dir: Path,
    sweep: str,
    case: str,
    provenance: dict[str, Any],
) -> Path | None:
    cases = _case_map(protocol, sweep)
    case_spec = cases[case]
    if case_spec["source"] != "new_formal_run":
        return None
    run_dirs = {
        (seed, method): _run_dir(output_dir, sweep, case, seed, method)
        for seed in SEEDS
        for method in METHODS
    }
    if not all(
        (run_dir / "hybrid_training_summary.json").exists()
        and (run_dir / "linear_probe_summary.json").exists()
        and (
            run_dir
            / "standardized_decoder"
            / "standardized_decoder_summary.json"
        ).exists()
        for run_dir in run_dirs.values()
    ):
        return None

    records: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    parameter_counts: dict[str, int] | None = None
    for seed in SEEDS:
        seed_rows: dict[str, Any] = {}
        for method in METHODS:
            run_dir = run_dirs[(seed, method)]
            config = load_config(run_dir / "config_resolved.yaml")
            counts = _validate_parameter_count(config, case_spec)
            if parameter_counts is None:
                parameter_counts = counts
            elif parameter_counts != counts:
                raise RuntimeError("Parameter counts differ within a sweep case")
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
            missing = [
                str(path.relative_to(run_dir))
                for path in _required(run_dir)
                if not path.exists()
            ]
            row = {
                "seed": seed,
                "method_id": method,
                "run_dir": str(run_dir),
                "git_commit": metadata["git_commit"],
                "git_worktree_dirty": metadata["git_worktree_dirty"],
                "split_manifest_sha256": metadata["split_manifest_sha256"],
                "initial_state_hash": metadata["initial_state_hash"],
                "initial_decoder_hash": metadata["initial_decoder_hash"],
                "probe_initial_hash": probe["classifier_initial_hash"],
                "standardized_decoder_initial_hash": standard["decoder_initial_hash"],
                "frozen_layers_unchanged": training["frozen_layers_unchanged"],
                "standardized_encoder_unchanged": standard["encoder_unchanged"],
                "artifact_complete": not missing,
                "statuses_complete": all(
                    status["status"] == "completed" for status in statuses
                ),
                "metrics_finite": (
                    _finite(training)
                    and _finite(probe)
                    and _finite(standard)
                    and _csv_finite(run_dir / "metrics.csv")
                    and _csv_finite(
                        run_dir / "standardized_decoder" / "metrics.csv"
                    )
                ),
                "test_samples_accessed": max(
                    int(metadata["test_samples_accessed"]),
                    int(training["test_samples_accessed"]),
                    int(probe["test_samples_accessed"]),
                    int(standard["test_samples_accessed"]),
                    *(int(status["test_samples_accessed"]) for status in statuses),
                ),
                "missing_artifacts": missing,
            }
            records[f"seed_{seed}/{method}"] = row
            seed_rows[method] = row

        hhh = run_dirs[(seed, "full_hebbian")]
        hhb = run_dirs[(seed, "hybrid_hhb")]
        hbb = run_dirs[(seed, "hybrid_hbb")]
        checks = {
            "four_methods_complete": len(seed_rows) == 4,
            "same_initial_model": len(
                {row["initial_state_hash"] for row in seed_rows.values()}
            ) == 1,
            "same_system_decoder_initialization": len(
                {row["initial_decoder_hash"] for row in seed_rows.values()}
            ) == 1,
            "same_probe_initialization": len(
                {row["probe_initial_hash"] for row in seed_rows.values()}
            ) == 1,
            "same_standardized_decoder_initialization": len(
                {
                    row["standardized_decoder_initial_hash"]
                    for row in seed_rows.values()
                }
            ) == 1,
            "same_split": len(
                {row["split_manifest_sha256"] for row in seed_rows.values()}
            ) == 1,
            "same_clean_source_commit": len(
                {row["git_commit"] for row in seed_rows.values()}
            ) == 1
            and not any(row["git_worktree_dirty"] for row in seed_rows.values()),
            "hebbian_enc1_prefix_paired": len(
                {
                    _stage_hash(hhh, "enc1"),
                    _stage_hash(hhb, "enc1"),
                    _stage_hash(hbb, "enc1"),
                }
            ) == 1,
            "hebbian_enc2_prefix_paired": (
                _stage_hash(hhh, "enc2") == _stage_hash(hhb, "enc2")
            ),
            "frozen_layers_unchanged": all(
                row["frozen_layers_unchanged"] for row in seed_rows.values()
            ),
            "standardized_encoders_unchanged": all(
                row["standardized_encoder_unchanged"]
                for row in seed_rows.values()
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
        "five_complete_paired_seeds": len(per_seed) == 5
        and all(row["decision"] == "PASS" for row in per_seed.values()),
        "source_snapshot_unique_and_clean": len(
            {row["git_commit"] for row in records.values()}
        ) == 1
        and not any(row["git_worktree_dirty"] for row in records.values()),
        "all_metrics_finite": all(row["metrics_finite"] for row in records.values()),
        "all_artifacts_complete": all(
            row["artifact_complete"] and row["statuses_complete"]
            for row in records.values()
        ),
        "zero_test_access": all(
            row["test_samples_accessed"] == 0 for row in records.values()
        ),
    }
    gate = {
        "schema_version": "stage3-q5q6-case-freeze-gate-v1",
        "decision": "PASS" if all(global_checks.values()) else "FAIL",
        "sweep": sweep,
        "case": case,
        "case_spec": case_spec,
        "parameter_counts": parameter_counts,
        "source_git_commit": next(iter(records.values()))["git_commit"],
        "per_seed": per_seed,
        "global_checks": global_checks,
        "test_samples_accessed": 0,
    }
    case_dir = output_dir / sweep / case
    write_json(case_dir, "freeze_gate.json", gate, overwrite=True)
    write_json(
        case_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-q5q6-case-runs-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "records": records,
            "freeze_gate": gate["decision"],
            "test_samples_accessed": 0,
            **provenance,
        },
        overwrite=True,
    )
    if gate["decision"] != "PASS":
        raise RuntimeError(f"Freeze gate failed for {sweep}/{case}")
    return case_dir / "freeze_gate.json"


def run(
    protocol_path: str | Path,
    *,
    sweep: str,
    selected_case: str | None = None,
    selected_seed: int | None = None,
    selected_method: str | None = None,
) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    cases = _case_map(protocol, sweep)
    if selected_case is not None and selected_case not in cases:
        raise ValueError(f"Unknown {sweep} case")
    if selected_seed is not None and selected_seed not in SEEDS:
        raise ValueError("Unknown formal seed")
    if selected_method is not None and selected_method not in METHODS:
        raise ValueError("Unknown formal method")
    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Q5/Q6 formal sweeps require a clean worktree")
    output_dir = _resolve(protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_cases = [selected_case] if selected_case else list(cases)
    seeds = [selected_seed] if selected_seed is not None else list(SEEDS)
    methods = [selected_method] if selected_method else list(METHODS)

    for case in requested_cases:
        spec = cases[case]
        if spec["source"] == "reuse_stage3_core":
            continue
        for seed in seeds:
            for method in methods:
                config = resolved_sweep_config(
                    protocol,
                    sweep=sweep,
                    case=case,
                    method=method,
                    seed=seed,
                )
                run_dir = _run_dir(output_dir, sweep, case, seed, method)
                train_hybrid_config(config, run_dir=run_dir)
                if read_run_status(run_dir)["status"] != "completed":
                    raise RuntimeError(f"Incomplete run: {sweep}/{case}/{seed}/{method}")
                if not (run_dir / "linear_probe_summary.json").exists():
                    train_linear_probe_config(config, run_dir, validation_only=True)
                train_standardized_decoder_config(config, run_dir=run_dir)
                print(
                    f"completed sweep={sweep} case={case} "
                    f"seed={seed} method={method}",
                    flush=True,
                )
        _write_case_gate(
            protocol=protocol,
            protocol_path=protocol_path,
            output_dir=output_dir,
            sweep=sweep,
            case=case,
            provenance=provenance,
        )
    return output_dir / sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/stage3_q5q6_sweeps_v1.yaml",
    )
    parser.add_argument("--sweep", choices=SWEEPS, required=True)
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    args = parser.parse_args()
    print(
        run(
            args.config,
            sweep=args.sweep,
            selected_case=args.case,
            selected_seed=args.seed,
            selected_method=args.method,
        ).resolve()
    )


if __name__ == "__main__":
    main()
