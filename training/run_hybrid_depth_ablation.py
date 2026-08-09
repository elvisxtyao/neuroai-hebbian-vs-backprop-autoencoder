"""Run the four preregistered seed-42 hybrid depth-ablation methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from schemas import load_config
from training.train_hybrid import train_hybrid_config
from training.train_linear_probe import train_linear_probe_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance
from utils.results import read_run_status, write_json


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(
    protocol_path: str | Path,
    *,
    selected_method: str | None = None,
) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["version"] != "hybrid-depth-ablation-v1":
        raise ValueError("Unsupported hybrid-depth protocol")
    if int(protocol["seed"]) != 42:
        raise ValueError("Hybrid depth ablation is frozen to seed 42")
    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Hybrid runs require a clean implementation worktree")

    output_dir = _resolve(protocol["output_dir"])
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    methods = protocol["methods"]
    if selected_method is not None:
        methods = [item for item in methods if item["id"] == selected_method]
        if len(methods) != 1:
            raise ValueError(f"Unknown method: {selected_method}")

    for method in methods:
        config_path = _resolve(method["config"])
        config = load_config(config_path)
        if config["hybrid"]["method_id"] != method["id"]:
            raise RuntimeError("Protocol/config method ID mismatch")
        run_dir = runs_dir / f"{method['id']}_seed42"
        train_hybrid_config(config, run_dir=run_dir)
        if read_run_status(run_dir)["status"] != "completed":
            raise RuntimeError(f"Training did not complete: {method['id']}")
        train_linear_probe_config(
            config,
            run_dir,
            validation_only=True,
        )
        print(f"completed={method['id']} run_dir={run_dir}", flush=True)

    all_run_dirs = {
        method["id"]: runs_dir / f"{method['id']}_seed42"
        for method in protocol["methods"]
    }
    if not all(path.exists() for path in all_run_dirs.values()):
        return output_dir

    records: dict[str, Any] = {}
    for method in protocol["methods"]:
        method_id = method["id"]
        run_dir = all_run_dirs[method_id]
        metadata = _json(run_dir / "metadata.json")
        training = _json(run_dir / "hybrid_training_summary.json")
        probe = _json(run_dir / "linear_probe_summary.json")
        records[method_id] = {
            "role": method["role"],
            "run_dir": str(run_dir),
            "config": method["config"],
            "config_file_sha256": file_sha256(_resolve(method["config"])),
            "config_fingerprint": metadata["config_sha256"],
            "git_commit": metadata["git_commit"],
            "git_worktree_dirty": metadata["git_worktree_dirty"],
            "split_manifest_sha256": metadata["split_manifest_sha256"],
            "initial_state_hash": metadata["initial_state_hash"],
            "initial_encoder_hash": metadata["initial_encoder_hash"],
            "initial_decoder_hash": metadata["initial_decoder_hash"],
            "initial_layer_hashes": metadata["initial_layer_hashes"],
            "classifier_initial_hash": probe["classifier_initial_hash"],
            "frozen_layers_unchanged": training["frozen_layers_unchanged"],
            "test_samples_accessed": max(
                int(metadata["test_samples_accessed"]),
                int(training["test_samples_accessed"]),
                int(probe["test_samples_accessed"]),
            ),
        }

    initial_state_hashes = {row["initial_state_hash"] for row in records.values()}
    decoder_hashes = {row["initial_decoder_hash"] for row in records.values()}
    split_hashes = {row["split_manifest_sha256"] for row in records.values()}
    probe_hashes = {row["classifier_initial_hash"] for row in records.values()}
    commits = {row["git_commit"] for row in records.values()}
    checks = {
        "four_methods_complete": len(records) == 4,
        "same_full_model_initialization": len(initial_state_hashes) == 1,
        "same_decoder_initialization": len(decoder_hashes) == 1,
        "same_split": len(split_hashes) == 1,
        "same_probe_classifier_initialization": len(probe_hashes) == 1,
        "same_clean_source_commit": len(commits) == 1
        and not any(row["git_worktree_dirty"] for row in records.values()),
        "all_frozen_layers_unchanged": all(
            row["frozen_layers_unchanged"] for row in records.values()
        ),
        "zero_test_access": all(
            row["test_samples_accessed"] == 0 for row in records.values()
        ),
    }
    gate = {
        "schema_version": "hybrid-depth-pairing-gate-v1",
        "completed_at_utc": utc_now(),
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "test_samples_accessed": 0,
    }
    write_json(output_dir, "pairing_gate.json", gate, overwrite=True)
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "hybrid-depth-run-manifest-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "seed": 42,
            "methods": records,
            "pairing_gate": gate["decision"],
            "test_samples_accessed": 0,
            **provenance,
        },
        overwrite=True,
    )
    if gate["decision"] != "PASS":
        raise RuntimeError("Hybrid pairing/integrity gate failed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/hybrid_depth_ablation_v1.yaml",
    )
    parser.add_argument("--method")
    args = parser.parse_args()
    print(run_experiment(args.config, selected_method=args.method).resolve())


if __name__ == "__main__":
    main()
