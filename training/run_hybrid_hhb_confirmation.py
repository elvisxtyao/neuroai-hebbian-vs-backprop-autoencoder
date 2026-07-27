"""Run the preregistered Stage 2D validation-only HHB confirmation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from schemas import load_config
from training.train_hybrid import train_hybrid_config
from training.train_linear_probe import train_linear_probe_config
from training.train_standardized_decoder import train_standardized_decoder_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance
from utils.results import read_run_status, write_json


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (43, 44)
EXPECTED_METHODS = ("full_bp", "full_hebbian", "hybrid_hhb")
EXPECTED_THRESHOLDS = {
    "validation_accuracy_floor": 0.8863,
    "standardized_decoder_mse_ratio_to_bp_max": 1.25,
    "z_effective_rank_min": 2.0,
    "z_effective_rank_ratio_to_full_hebbian_min": 2.0,
    "z_to_h2_effective_rank_ratio_min": 2.0,
    "epsilon": 1.0e-12,
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_confirmation_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("version") != "hybrid-hhb-confirmation-v1":
        raise ValueError("Unsupported Stage 2D protocol")
    if tuple(protocol.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Stage 2D seeds are frozen to [43, 44]")
    if tuple(protocol.get("methods", ())) != EXPECTED_METHODS:
        raise ValueError("Stage 2D methods are frozen to BBB/HHH/HHB")
    if protocol.get("test_access_policy") != "validation_only":
        raise ValueError("Stage 2D must be validation-only")
    configs = protocol.get("configs", {})
    if set(configs) != {"43", "44"}:
        raise ValueError("Stage 2D config map must contain exactly seeds 43 and 44")
    for seed in EXPECTED_SEEDS:
        if set(configs[str(seed)]) != set(EXPECTED_METHODS):
            raise ValueError(f"Stage 2D seed {seed} config map is incomplete")
    if protocol.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("Stage 2D thresholds changed")


def _required_artifacts(run_dir: Path) -> list[Path]:
    return [
        run_dir / "config_resolved.yaml",
        run_dir / "metadata.json",
        run_dir / "run_status.json",
        run_dir / "metrics.csv",
        run_dir / "model_best.pt",
        run_dir / "model_last.pt",
        run_dir / "hybrid_training_summary.json",
        run_dir / "linear_probe.pt",
        run_dir / "linear_probe_summary.json",
        run_dir / "trainable_frozen_parameter_manifest.json",
        run_dir / "standardized_decoder" / "run_status.json",
        run_dir / "standardized_decoder" / "metrics.csv",
        run_dir / "standardized_decoder" / "decoder_best.pt",
        run_dir
        / "standardized_decoder"
        / "standardized_decoder_summary.json",
    ]


def run_confirmation(
    protocol_path: str | Path,
    *,
    selected_seed: int | None = None,
    selected_method: str | None = None,
) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_confirmation_protocol(protocol)
    if selected_seed is not None and selected_seed not in EXPECTED_SEEDS:
        raise ValueError("Selected seed must be 43 or 44")
    if selected_method is not None and selected_method not in EXPECTED_METHODS:
        raise ValueError("Unknown Stage 2D method")

    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Stage 2D requires a clean implementation worktree")
    output_dir = _resolve(protocol["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [selected_seed] if selected_seed is not None else list(EXPECTED_SEEDS)
    methods = (
        [selected_method] if selected_method is not None else list(EXPECTED_METHODS)
    )

    for seed in seeds:
        for method in methods:
            config_path = _resolve(protocol["configs"][str(seed)][method])
            config = load_config(config_path)
            if int(config["training"]["seed"]) != seed:
                raise RuntimeError("Protocol/config seed mismatch")
            if config["hybrid"]["method_id"] != method:
                raise RuntimeError("Protocol/config method mismatch")
            run_dir = output_dir / "runs" / f"seed_{seed}" / method
            train_hybrid_config(config, run_dir=run_dir)
            if read_run_status(run_dir)["status"] != "completed":
                raise RuntimeError(f"Incomplete representation run: seed={seed} {method}")
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
    if not all(path.exists() for path in all_run_dirs.values()):
        return output_dir

    records: dict[str, Any] = {}
    per_seed_checks: dict[str, Any] = {}
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
            standard_status = read_run_status(run_dir / "standardized_decoder")
            test_access = max(
                int(metadata["test_samples_accessed"]),
                int(training["test_samples_accessed"]),
                int(probe["test_samples_accessed"]),
                int(standard["test_samples_accessed"]),
                int(standard_status["test_samples_accessed"]),
            )
            row = {
                "seed": seed,
                "method_id": method,
                "run_dir": str(run_dir),
                "config": protocol["configs"][str(seed)][method],
                "config_sha256": file_sha256(
                    _resolve(protocol["configs"][str(seed)][method])
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
                "test_samples_accessed": test_access,
            }
            seed_records[method] = row
            records[f"seed_{seed}/{method}"] = row

        checks = {
            "three_methods_complete": len(seed_records) == 3,
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
            "frozen_layers_unchanged": all(
                row["frozen_layers_unchanged"] for row in seed_records.values()
            ),
            "standardized_encoders_unchanged": all(
                row["standardized_encoder_unchanged"]
                for row in seed_records.values()
            ),
            "artifacts_complete": all(
                row["artifact_complete"] for row in seed_records.values()
            ),
            "zero_test_access": all(
                row["test_samples_accessed"] == 0
                for row in seed_records.values()
            ),
        }
        per_seed_checks[str(seed)] = {
            "decision": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        }

    global_checks = {
        "seed_43_pairing_pass": per_seed_checks["43"]["decision"] == "PASS",
        "seed_44_pairing_pass": per_seed_checks["44"]["decision"] == "PASS",
        "same_source_commit_across_seeds": len(
            {row["git_commit"] for row in records.values()}
        )
        == 1,
        "protocol_source_matches_runs": {
            row["git_commit"] for row in records.values()
        }
        == {provenance["git_commit"]},
        "all_artifacts_complete": all(
            row["artifact_complete"] for row in records.values()
        ),
        "zero_test_access": all(
            row["test_samples_accessed"] == 0 for row in records.values()
        ),
    }
    gate = {
        "schema_version": "hybrid-hhb-confirmation-pairing-v1",
        "completed_at_utc": utc_now(),
        "decision": "PASS" if all(global_checks.values()) else "FAIL",
        "per_seed": per_seed_checks,
        "global_checks": global_checks,
        "test_samples_accessed": 0,
    }
    write_json(output_dir, "pairing_gate.json", gate, overwrite=True)
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "hybrid-hhb-confirmation-runs-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "protocol_document": protocol["protocol_document"],
            "protocol_document_sha256": file_sha256(
                _resolve(protocol["protocol_document"])
            ),
            "records": records,
            "pairing_gate": gate["decision"],
            "test_samples_accessed": 0,
            **provenance,
        },
        overwrite=True,
    )
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 2D pairing/integrity gate failed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/hybrid_hhb_confirmation_v1.yaml",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    args = parser.parse_args()
    print(
        run_confirmation(
            args.config,
            selected_seed=args.seed,
            selected_method=args.method,
        ).resolve()
    )


if __name__ == "__main__":
    main()
