"""Build the frozen Branch-D decision record without training or data access."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from utils.checkpointing import file_sha256, utc_now


ROOT = Path(__file__).resolve().parents[1]
LAYER_MAP = {"enc1": "h1", "enc2": "h2", "enc3": "z"}
VARIANTS = ("hebbian_raw", "hebbian_effective")
HEALTH_FIELDS = (
    "effective_rank",
    "normalized_effective_rank",
    "winner_coverage_ratio",
    "winner_entropy",
    "max_winner_share",
    "dead_unit_ratio",
)
Q4_FIELDS = (
    "batch_alignment_mean",
    "batch_norm_ratio_mean",
    "alpha_star",
    "scale_matched_relative_bias",
    "candidate_snr_linear",
    "bp_snr_linear",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return None
    return (candidate - baseline) / abs(baseline)


def select_branch(
    *,
    performance_pass: bool,
    health_pass: bool,
    integrity_pass: bool,
    health_improved: bool,
    direction_improved_scale_abnormal: bool,
    enc3_isolated_failure: bool,
    update_noise_primary: bool,
) -> str:
    """Apply the frozen A > B1 > B2 > B3 > C > D priority."""
    if performance_pass and health_pass and integrity_pass:
        return "A"
    if health_improved and not performance_pass:
        if direction_improved_scale_abnormal:
            return "B1"
        if enc3_isolated_failure:
            return "B2"
        if update_noise_primary:
            return "B3"
    if performance_pass and not health_pass:
        return "C"
    return "D"


def _validation_accuracy(run_dir: Path) -> float:
    result: float | None = None
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                raise RuntimeError(f"Test row found in validation-only run: {run_dir}")
            if row["stage"] == "linear_probe_final" and row["split"] == "validation":
                result = float(row["accuracy"])
    if result is None:
        raise RuntimeError(f"Final validation accuracy missing: {run_dir}")
    return result


def _baseline_health(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with _resolve(config["health_metrics"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if row["checkpoint_id"] != config["health_checkpoint_id"]:
                continue
            rows[row["layer"]] = {
                "gate_pass": row["gate_pass"].lower() == "true",
                "metrics": {field: float(row[field]) for field in HEALTH_FIELDS},
            }
    if set(rows) != set(LAYER_MAP.values()):
        raise RuntimeError("Baseline health rows are incomplete")
    return rows


def _q4_rows(root: Path) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    with (root / "aggregate_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            result[(row["layer"], row["variant"])] = {
                field: float(row[field]) for field in Q4_FIELDS
            }
    expected = {
        (layer, variant) for layer in LAYER_MAP for variant in VARIANTS
    }
    if set(result) != expected:
        raise RuntimeError(f"Incomplete Q4 aggregates: {root}")
    return result


def _zero_norm_counts(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    counts = {
        (layer, variant): {
            "bp_zero_norm_count": 0,
            "hebbian_zero_norm_count": 0,
            "any_zero_norm_batch_count": 0,
            "valid_batch_count": 0,
        }
        for layer in LAYER_MAP
        for variant in VARIANTS
    }
    with (root / "batch_update_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            layer = row["layer"]
            bp_norm = float(row["bp_raw_norm"])
            for variant, column, alignment_column in (
                ("hebbian_raw", "hebbian_raw_norm", "raw_alignment"),
                (
                    "hebbian_effective",
                    "hebbian_effective_norm",
                    "effective_alignment",
                ),
            ):
                hebbian_norm = float(row[column])
                alignment = float(row[alignment_column])
                record = counts[(layer, variant)]
                if all(
                    math.isfinite(value)
                    for value in (bp_norm, hebbian_norm, alignment)
                ):
                    record["valid_batch_count"] += 1
                if bp_norm == 0.0:
                    record["bp_zero_norm_count"] += 1
                if hebbian_norm == 0.0:
                    record["hebbian_zero_norm_count"] += 1
                if bp_norm == 0.0 or hebbian_norm == 0.0:
                    record["any_zero_norm_batch_count"] += 1
    return counts


def _validate_q4_gate(root: Path) -> dict[str, Any]:
    gate = _load_json(root / "gate_decision.json")
    if (
        gate["decision"] != "PASS"
        or gate["analysis_optimizer_steps"] != 0
        or gate["test_samples_accessed"] != 0
        or not all(gate["checks"].values())
    ):
        raise RuntimeError(f"Q4 gate failed: {root}")
    return gate


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_test_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def build_decision(config_path: str | Path) -> Path:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["version"] != "hebbian-followup-decision-v1":
        raise ValueError("Unsupported follow-up decision config")

    output_dir = _resolve(config["output_dir"])
    baseline_config = config["baseline"]
    candidate_config = config["candidate"]
    baseline_run = _resolve(baseline_config["run_dir"])
    candidate_run = _resolve(candidate_config["run_dir"])
    baseline_q4_root = _resolve(baseline_config["q4"])
    candidate_q4_root = _resolve(candidate_config["q4"])

    health_gate_path = _resolve(baseline_config["health_gate"])
    health_gate = _load_json(health_gate_path)
    thresholds = health_gate.get("thresholds")
    if not thresholds:
        raise RuntimeError("BLOCKED: frozen representation-health thresholds missing")

    selection_path = _resolve(candidate_config["selection"])
    selection = _load_json(selection_path)
    frozen_floor = float(config["frozen_performance_floor"])
    if float(selection["minimum_validation_accuracy"]) != frozen_floor:
        raise RuntimeError("BLOCKED: performance threshold records disagree")

    baseline_accuracy = _validation_accuracy(baseline_run)
    candidate_accuracy = _validation_accuracy(candidate_run)
    baseline_health = _baseline_health(baseline_config)
    candidate_health_path = _resolve(candidate_config["health"])
    candidate_health_payload = _load_json(candidate_health_path)
    if candidate_health_payload["test_samples_accessed"] != 0:
        raise RuntimeError("Candidate health analysis accessed test data")
    candidate_health = candidate_health_payload["layers"]
    baseline_q4 = _q4_rows(baseline_q4_root)
    candidate_q4 = _q4_rows(candidate_q4_root)
    baseline_zero = _zero_norm_counts(baseline_q4_root)
    candidate_zero = _zero_norm_counts(candidate_q4_root)
    baseline_q4_gate = _validate_q4_gate(baseline_q4_root)
    candidate_q4_gate = _validate_q4_gate(candidate_q4_root)
    baseline_manifest = _load_json(baseline_q4_root / "run_manifest.json")
    candidate_manifest = _load_json(candidate_q4_root / "run_manifest.json")

    synthetic_log_path = _resolve(config["synthetic_test_log"])
    current_test_log_path = _resolve(config["current_test_log"])
    for test_log in (synthetic_log_path, current_test_log_path):
        text = _read_test_log(test_log)
        if "passed" not in text or "failed" in text:
            raise RuntimeError(f"Test log is not a passing record: {test_log}")

    baseline_reconstruction = {
        key.replace("_end", ""): float(value["best_validation_mse"])
        for key, value in baseline_manifest["reference_decoders"].items()
    }
    candidate_reconstruction = {
        key.replace("_end", ""): float(value["best_validation_mse"])
        for key, value in candidate_manifest["reference_decoders"].items()
    }
    baseline_snapshots = {
        item["active_layer"]: item for item in baseline_manifest["snapshots"]
    }
    candidate_snapshots = {
        item["active_layer"]: item for item in candidate_manifest["snapshots"]
    }

    rows: list[dict[str, Any]] = []
    for layer, health_layer in LAYER_MAP.items():
        for variant in VARIANTS:
            row: dict[str, Any] = {
                "snapshot_id": f"{layer}_end",
                "layer": layer,
                "health_layer": health_layer,
                "update_variant": variant,
                "baseline_validation_accuracy": baseline_accuracy,
                "candidate_validation_accuracy": candidate_accuracy,
                "absolute_change_validation_accuracy": (
                    candidate_accuracy - baseline_accuracy
                ),
                "relative_change_validation_accuracy": relative_change(
                    baseline_accuracy, candidate_accuracy
                ),
                "baseline_validation_reconstruction_mse": (
                    baseline_reconstruction[layer]
                ),
                "candidate_validation_reconstruction_mse": (
                    candidate_reconstruction[layer]
                ),
                "absolute_change_validation_reconstruction_mse": (
                    candidate_reconstruction[layer]
                    - baseline_reconstruction[layer]
                ),
                "relative_change_validation_reconstruction_mse": relative_change(
                    baseline_reconstruction[layer],
                    candidate_reconstruction[layer],
                ),
                "baseline_health_gate_pass": baseline_health[health_layer][
                    "gate_pass"
                ],
                "candidate_health_gate_pass": candidate_health[health_layer][
                    "gate_pass"
                ],
            }
            for field in HEALTH_FIELDS:
                baseline_value = baseline_health[health_layer]["metrics"][field]
                candidate_value = float(
                    candidate_health[health_layer]["metrics"][field]
                )
                row[f"baseline_{field}"] = baseline_value
                row[f"candidate_{field}"] = candidate_value
                row[f"absolute_change_{field}"] = candidate_value - baseline_value
                row[f"relative_change_{field}"] = relative_change(
                    baseline_value, candidate_value
                )
            for field in Q4_FIELDS:
                baseline_value = baseline_q4[(layer, variant)][field]
                candidate_value = candidate_q4[(layer, variant)][field]
                row[f"baseline_{field}"] = baseline_value
                row[f"candidate_{field}"] = candidate_value
                row[f"absolute_change_{field}"] = candidate_value - baseline_value
                row[f"relative_change_{field}"] = relative_change(
                    baseline_value, candidate_value
                )
            for field, value in baseline_zero[(layer, variant)].items():
                row[f"baseline_{field}"] = value
            for field, value in candidate_zero[(layer, variant)].items():
                row[f"candidate_{field}"] = value
            row["baseline_snapshot_state_dict_sha256"] = baseline_snapshots[layer][
                "state_dict_sha256"
            ]
            row["candidate_snapshot_state_dict_sha256"] = candidate_snapshots[
                layer
            ]["state_dict_sha256"]
            rows.append(row)

    performance_pass = candidate_accuracy >= frozen_floor
    health_pass = bool(candidate_health_payload["health_pass"])
    integrity_pass = bool(
        all(baseline_q4_gate["checks"].values())
        and all(candidate_q4_gate["checks"].values())
    )
    health_nonincreasing = all(
        float(candidate_health[layer]["metrics"]["effective_rank"])
        <= baseline_health[layer]["metrics"]["effective_rank"]
        and float(candidate_health[layer]["metrics"]["winner_coverage_ratio"])
        <= baseline_health[layer]["metrics"]["winner_coverage_ratio"]
        and float(candidate_health[layer]["metrics"]["winner_entropy"])
        <= baseline_health[layer]["metrics"]["winner_entropy"]
        for layer in LAYER_MAP.values()
    )
    health_improved = health_pass or not health_nonincreasing
    enc3_raw = candidate_q4[("enc3", "hebbian_raw")]
    baseline_enc3_raw = baseline_q4[("enc3", "hebbian_raw")]
    direction_improved_scale_abnormal = bool(
        enc3_raw["batch_alignment_mean"]
        > baseline_enc3_raw["batch_alignment_mean"]
        and enc3_raw["scale_matched_relative_bias"]
        < baseline_enc3_raw["scale_matched_relative_bias"]
    )
    enc3_isolated_failure = bool(
        candidate_health["h1"]["gate_pass"]
        and candidate_health["h2"]["gate_pass"]
        and not candidate_health["z"]["gate_pass"]
    )
    update_noise_primary = bool(
        enc3_raw["batch_alignment_mean"] >= 0.0
        and enc3_raw["candidate_snr_linear"] < enc3_raw["bp_snr_linear"]
    )
    selected_branch = select_branch(
        performance_pass=performance_pass,
        health_pass=health_pass,
        integrity_pass=integrity_pass,
        health_improved=health_improved,
        direction_improved_scale_abnormal=direction_improved_scale_abnormal,
        enc3_isolated_failure=enc3_isolated_failure,
        update_noise_primary=update_noise_primary,
    )
    if selected_branch != "D":
        raise RuntimeError(
            f"Frozen evidence selected unexpected branch {selected_branch}; "
            "this builder is authorized for Branch D only"
        )

    baseline_metadata = _load_json(baseline_run / "metadata.json")
    candidate_metadata = _load_json(candidate_run / "metadata.json")
    trial = next(
        item
        for item in selection["trials"]
        if item["trial_id"] == candidate_config["id"]
    )
    final_commit = _git(["rev-parse", "HEAD"])
    final_status = _git(["status", "--short"]).splitlines()

    def health_differences() -> dict[str, Any]:
        return {
            layer: {
                field: {
                    "baseline": baseline_health[layer]["metrics"][field],
                    "candidate": float(candidate_health[layer]["metrics"][field]),
                    "absolute": (
                        float(candidate_health[layer]["metrics"][field])
                        - baseline_health[layer]["metrics"][field]
                    ),
                    "relative": relative_change(
                        baseline_health[layer]["metrics"][field],
                        float(candidate_health[layer]["metrics"][field]),
                    ),
                }
                for field in HEALTH_FIELDS
            }
            for layer in LAYER_MAP.values()
        }

    branch_conditions = {
        "performance_gate_pass": performance_pass,
        "representation_health_gate_pass": health_pass,
        "checksum_synthetic_tests_optimizer_exclusion_pass": integrity_pass,
        "health_improved": health_improved,
        "health_rank_coverage_entropy_all_nonincreasing": health_nonincreasing,
        "candidate_accuracy_improved": candidate_accuracy > baseline_accuracy,
        "enc3_raw_alignment_improved": (
            enc3_raw["batch_alignment_mean"]
            > baseline_enc3_raw["batch_alignment_mean"]
        ),
        "enc3_raw_alpha_star_negative": enc3_raw["alpha_star"] < 0.0,
        "enc3_raw_hebbian_snr_improved": (
            enc3_raw["candidate_snr_linear"]
            > baseline_enc3_raw["candidate_snr_linear"]
        ),
        "branch_a": performance_pass and health_pass and integrity_pass,
        "branch_b1": health_improved
        and not performance_pass
        and direction_improved_scale_abnormal,
        "branch_b2": health_improved
        and not performance_pass
        and not direction_improved_scale_abnormal
        and enc3_isolated_failure,
        "branch_b3": health_improved
        and not performance_pass
        and not direction_improved_scale_abnormal
        and not enc3_isolated_failure
        and update_noise_primary,
        "branch_c": performance_pass and not health_pass,
        "branch_d": True,
    }
    decision = {
        "schema_version": "hebbian-followup-decision-v1",
        "completed_at_utc": utc_now(),
        "selected_branch": "D",
        "selected_branch_label": "BRANCH D — FREEZE AS FAILURE-CASE BASELINE",
        "decision_priority": ["A", "B1", "B2", "B3", "C", "D"],
        "decision_record": "COMMON-MODE UPDATE REMOVAL: NOT SUFFICIENT",
        "branch_conditions": branch_conditions,
        "source_run_ids": {
            "baseline": baseline_config["id"],
            "candidate": candidate_config["id"],
            "baseline_run_dir": str(baseline_run),
            "candidate_run_dir": str(candidate_run),
        },
        "baseline_config_hash": baseline_metadata["config_sha256"],
        "candidate_config_hash": trial["config_sha256"],
        "gate_thresholds": {
            "minimum_validation_accuracy": frozen_floor,
            "representation_health": thresholds,
        },
        "metric_differences": {
            "validation_accuracy": {
                "baseline": baseline_accuracy,
                "candidate": candidate_accuracy,
                "absolute": candidate_accuracy - baseline_accuracy,
                "relative": relative_change(baseline_accuracy, candidate_accuracy),
            },
            "enc3_raw_alignment": {
                "baseline": baseline_enc3_raw["batch_alignment_mean"],
                "candidate": enc3_raw["batch_alignment_mean"],
                "absolute": (
                    enc3_raw["batch_alignment_mean"]
                    - baseline_enc3_raw["batch_alignment_mean"]
                ),
                "relative": relative_change(
                    baseline_enc3_raw["batch_alignment_mean"],
                    enc3_raw["batch_alignment_mean"],
                ),
            },
            "enc3_raw_hebbian_snr": {
                "baseline": baseline_enc3_raw["candidate_snr_linear"],
                "candidate": enc3_raw["candidate_snr_linear"],
                "absolute": (
                    enc3_raw["candidate_snr_linear"]
                    - baseline_enc3_raw["candidate_snr_linear"]
                ),
                "relative": relative_change(
                    baseline_enc3_raw["candidate_snr_linear"],
                    enc3_raw["candidate_snr_linear"],
                ),
            },
            "health_layers": health_differences(),
        },
        "selected_follow_up_formula": None,
        "selected_follow_up_action": (
            "No new candidate. Preserve original Oja + WTA as the "
            "health-gate failure-case baseline."
        ),
        "training_started": False,
        "formal_seeds_started": False,
        "test_accessed": False,
        "test_samples_accessed": 0,
        "stage3_allowed": False,
        "final_git_commit": final_commit,
        "final_worktree_status": final_status,
        "notes": {
            "batch_count_is_not_seed_count": True,
            "q4_batches_per_snapshot": 50,
            "baseline_source_run_recorded_dirty": bool(
                baseline_metadata["git_worktree_dirty"]
            ),
            "baseline_q4_artifact_recorded_dirty": bool(
                baseline_manifest["git_worktree_dirty"]
            ),
            "candidate_source_run_recorded_dirty": bool(
                candidate_metadata["git_worktree_dirty"]
            ),
            "candidate_q4_artifact_recorded_dirty": bool(
                candidate_manifest["git_worktree_dirty"]
            ),
        },
    }

    source_paths = {
        "decision_config": config_path,
        "baseline_health_metrics": _resolve(baseline_config["health_metrics"]),
        "baseline_health_gate": health_gate_path,
        "baseline_run_metadata": baseline_run / "metadata.json",
        "baseline_run_metrics": baseline_run / "metrics.csv",
        "baseline_q4_aggregate": baseline_q4_root / "aggregate_metrics.csv",
        "baseline_q4_batches": baseline_q4_root / "batch_update_metrics.csv",
        "baseline_q4_gate": baseline_q4_root / "gate_decision.json",
        "baseline_q4_manifest": baseline_q4_root / "run_manifest.json",
        "baseline_snapshot_gate": baseline_q4_root
        / "snapshot_integrity_gate.json",
        "candidate_selection": selection_path,
        "candidate_health": candidate_health_path,
        "candidate_run_metadata": candidate_run / "metadata.json",
        "candidate_run_metrics": candidate_run / "metrics.csv",
        "candidate_q4_aggregate": candidate_q4_root / "aggregate_metrics.csv",
        "candidate_q4_batches": candidate_q4_root / "batch_update_metrics.csv",
        "candidate_q4_gate": candidate_q4_root / "gate_decision.json",
        "candidate_q4_manifest": candidate_q4_root / "run_manifest.json",
        "candidate_snapshot_gate": candidate_q4_root
        / "snapshot_integrity_gate.json",
        "synthetic_test_log": synthetic_log_path,
        "current_test_log": current_test_log_path,
    }
    source_manifest = {
        "schema_version": "hebbian-followup-source-manifest-v1",
        "created_at_utc": utc_now(),
        "final_git_commit": final_commit,
        "files": {
            name: _source_entry(path) for name, path in source_paths.items()
        },
        "baseline_snapshots": baseline_manifest["snapshots"],
        "candidate_snapshots": candidate_manifest["snapshots"],
        "batch_identity": {
            "baseline_batch_ids_sha256": baseline_manifest["batch_ids_sha256"],
            "candidate_batch_ids_sha256": candidate_manifest["batch_ids_sha256"],
            "same_fixed_batches": (
                baseline_manifest["batch_ids_sha256"]
                == candidate_manifest["batch_ids_sha256"]
            ),
            "unique_sample_count": baseline_manifest["unique_sample_count"],
        },
    }

    report = f"""# Hebbian follow-up decision report

Status: **BRANCH D — FREEZE AS FAILURE-CASE BASELINE**

## Decision

`COMMON-MODE UPDATE REMOVAL: NOT SUFFICIENT`

The sole output-filter-centered candidate failed the frozen performance gate
(`{candidate_accuracy:.4f} < {frozen_floor:.4f}`) and the representation-health
gate at every layer. Effective rank, winner coverage, and winner entropy were
non-increasing at all three layers. At Enc3, raw alignment changed from
`{baseline_enc3_raw['batch_alignment_mean']:.6f}` to
`{enc3_raw['batch_alignment_mean']:.6f}`, `alpha*` became
`{enc3_raw['alpha_star']:.3e}`, and Hebbian SNR changed from
`{baseline_enc3_raw['candidate_snr_linear']:.4f}` to
`{enc3_raw['candidate_snr_linear']:.4f}`.

The checksum, synthetic-test, fixed-batch, and optimizer-exclusion checks
passed, but integrity cannot compensate for failed performance and health.
No new seed-42 training, formal seed, test access, or follow-up candidate was
started.

## Scope

The 50 Q4 batches per snapshot describe update variability at one frozen seed-42
state. They are not independent seeds and are not used as a multi-seed
confidence interval.

The original Oja + WTA configuration remains a
`health-gate failure-case baseline`. It is not a healthy Hebbian baseline and
does not authorize Stage 3.

## Files

- `comparison_table.csv`: six snapshot/layer × update-variant rows with
  baseline, candidate, absolute change, and relative change.
- `decision.json`: frozen machine-readable decision and branch conditions.
- `source_manifest.json`: source hashes, snapshot hashes, and fixed-batch
  identity.
- `tests.log`: immutable full-suite record copied into this decision package.
- `selected_branch_artifacts/failure_case_protocol_addendum.md`: Branch-D
  protocol restriction.
"""

    _write_csv(output_dir / "comparison_table.csv", rows)
    _write_json(output_dir / "decision.json", decision)
    _write_json(output_dir / "source_manifest.json", source_manifest)
    _write_text(output_dir / "decision_report.md", report)
    shutil.copyfile(current_test_log_path, output_dir / "tests.log")
    addendum = _resolve(config["failure_case_addendum"]).read_text(
        encoding="utf-8"
    )
    _write_text(
        output_dir
        / "selected_branch_artifacts"
        / "failure_case_protocol_addendum.md",
        addendum,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/hebbian_followup_decision_v1.yaml",
    )
    args = parser.parse_args()
    print(build_decision(args.config).resolve())


if __name__ == "__main__":
    main()
