"""Build the compact, artifact-only evidence bundle for v1.0-final.

The builder reads only accepted CSV/JSON result summaries and resolved YAML
configuration records. It never imports training/evaluation entry points,
loads a dataset or checkpoint, or writes inside ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release" / "v1.0-final"
FORMAL_ROOT = Path("results/formal/phase0_v1_1")

# Keep the v1.0 manifest's historical logical paths stable while allowing the
# public repository to present governance records under docs/.
FROZEN_GOVERNANCE_PATHS = {
    "HEBBIAN_PROJECT_PLAN.md": "docs/history/HEBBIAN_PROJECT_PLAN.md",
    "PROJECT_STATUS.md": "docs/history/PROJECT_STATUS.md",
    "PHASE0_STANDARD_V1.md": "docs/protocols/PHASE0_STANDARD_V1.md",
    "PHASE0_STANDARD_V1_1_ADDENDUM.md": (
        "docs/protocols/PHASE0_STANDARD_V1_1_ADDENDUM.md"
    ),
    "docs/hybrid_hhb_confirmation_protocol.md": (
        "docs/confirmation/hybrid_hhb_confirmation_protocol.md"
    ),
    "docs/hybrid_hhb_confirmation_results.md": (
        "docs/confirmation/hybrid_hhb_confirmation_results.md"
    ),
    "docs/stage3_formal_protocol_v1.md": (
        "docs/protocols/stage3_formal_protocol_v1.md"
    ),
    "docs/final_statistical_protocol_audit.md": (
        "docs/audits/final_statistical_protocol_audit.md"
    ),
}

FORBIDDEN_SOURCE_PARTS = {
    "_recovery",
    "recovery",
    "checkpoints",
    "representations",
    "embeddings",
    "tuning",
    "q1_clean_v1",
    "reconstruction_sanity",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".log", ".png", ".pdf"}


SOURCE_FILES: dict[str, str] = {
    "data/governance/stage2d_confirmation_decision.json": (
        "results/hybrid_hhb_confirmation/confirmation_decision.json"
    ),
    "data/governance/stage3_core_freeze_gate.json": (
        "results/formal/phase0_v1_1/stage3_core/freeze_gate.json"
    ),
    "data/governance/stage3_core_test_summary.json": (
        "results/formal/phase0_v1_1/stage3_core/test_evaluation/summary.json"
    ),
    "data/governance/stage3_matched_controls_freeze_gate.json": (
        "results/formal/phase0_v1_1/stage3_matched_controls/freeze_gate.json"
    ),
    "data/governance/stage3_matched_controls_test_summary.json": (
        "results/formal/phase0_v1_1/stage3_matched_controls/test_evaluation/summary.json"
    ),
    "data/q1/per_seed_complete.csv": (
        "results/formal/phase0_v1_1/stage3_q1_complete/per_seed_complete.csv"
    ),
    "data/q1/method_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q1_complete/method_summary.csv"
    ),
    "data/q1/paired_contrasts.csv": (
        "results/formal/phase0_v1_1/stage3_q1_complete/paired_contrasts.csv"
    ),
    "data/q1/summary.json": (
        "results/formal/phase0_v1_1/stage3_q1_complete/summary.json"
    ),
    "data/q1/provenance.json": (
        "results/formal/phase0_v1_1/stage3_q1_complete/provenance.json"
    ),
    "data/q2/per_seed_layer_metrics.csv": (
        "results/formal/phase0_v1_1/stage3_q2_representation/per_seed_layer_metrics.csv"
    ),
    "data/q2/method_layer_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q2_representation/method_layer_summary.csv"
    ),
    "data/q2/compensation_metrics.csv": (
        "results/formal/phase0_v1_1/stage3_q2_representation/compensation_metrics.csv"
    ),
    "data/q2/layerwise_cka.csv": (
        "results/formal/phase0_v1_1/stage3_q2_representation/layerwise_cka.csv"
    ),
    "data/q3/per_seed_condition_metrics.csv": (
        "results/formal/phase0_v1_1/stage3_q3_noise/per_seed_condition_metrics.csv"
    ),
    "data/q3/condition_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q3_noise/condition_summary.csv"
    ),
    "data/q3/paired_degradation_contrasts.csv": (
        "results/formal/phase0_v1_1/stage3_q3_noise/paired_degradation_contrasts.csv"
    ),
    "data/q4/per_seed_layer_update_metrics.csv": (
        "results/formal/phase0_v1_1/stage3_q4_updates/per_seed_layer_update_metrics.csv"
    ),
    "data/q4/method_layer_update_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q4_updates/method_layer_update_summary.csv"
    ),
    "data/q4/cross_metric_join.csv": (
        "results/formal/phase0_v1_1/stage3_q4_updates/cross_metric_join.csv"
    ),
    "data/q4/exploratory_correlations.csv": (
        "results/formal/phase0_v1_1/stage3_q4_updates/exploratory_correlations.csv"
    ),
    "data/q5q6/architecture_cross_case_cka.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/architecture_cross_case_cka.csv"
    ),
    "data/q5q6/architecture_update_per_seed.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/architecture_update_per_seed.csv"
    ),
    "data/q5q6/architecture_update_representation_join.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/architecture_update_representation_join.csv"
    ),
    "data/q5q6/architecture_update_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/architecture_update_summary.csv"
    ),
    "data/q5q6/compensation_metrics.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/compensation_metrics.csv"
    ),
    "data/q5q6/interaction_tests.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/interaction_tests.csv"
    ),
    "data/q5q6/noise_severity_0_4_per_seed.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/noise_severity_0_4_per_seed.csv"
    ),
    "data/q5q6/noise_severity_0_4_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/noise_severity_0_4_summary.csv"
    ),
    "data/q5q6/paired_performance_contrasts.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/paired_performance_contrasts.csv"
    ),
    "data/q5q6/performance_per_seed.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/performance_per_seed.csv"
    ),
    "data/q5q6/performance_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/performance_summary.csv"
    ),
    "data/q5q6/relative_to_baseline.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/relative_to_baseline.csv"
    ),
    "data/q5q6/representation_per_seed_layer.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/representation_per_seed_layer.csv"
    ),
    "data/q5q6/representation_summary.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/representation_summary.csv"
    ),
    "data/q5q6/run_manifest.json": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/run_manifest.json"
    ),
    "data/q5q6/sensitivity.csv": (
        "results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/sensitivity.csv"
    ),
    "data/audit/q1_samples_seen_curve_per_seed.csv": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/q1_samples_seen_curve_per_seed.csv"
    ),
    "data/audit/q1_samples_seen_curve_summary.csv": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/q1_samples_seen_curve_summary.csv"
    ),
    "data/audit/q3_curve_auc_per_seed.csv": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/q3_curve_auc_per_seed.csv"
    ),
    "data/audit/q3_curve_auc_summary.csv": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/q3_curve_auc_summary.csv"
    ),
    "data/audit/q3_curve_auc_paired_contrasts.csv": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/q3_curve_auc_paired_contrasts.csv"
    ),
    "data/audit/protocol_audit.json": (
        "results/formal/phase0_v1_1/stage3_final_audit_supplement/protocol_audit.json"
    ),
}

SANITIZED_JSON_FIELDS = {
    "data/q1/provenance.json": {
        "controls_root": "results/formal/phase0_v1_1/stage3_matched_controls",
        "core_root": "results/formal/phase0_v1_1/stage3_core",
    },
    "data/q5q6/run_manifest.json": {
        "protocol": "configs/experiments/stage3_q5q6_sweeps_v1.yaml",
    },
}

ROW_EXPECTATIONS = {
    "data/q1/per_seed_complete.csv": 35,
    "data/q2/per_seed_layer_metrics.csv": 90,
    "data/q3/per_seed_condition_metrics.csv": 260,
    "data/q4/per_seed_layer_update_metrics.csv": 90,
    "data/q5q6/performance_per_seed.csv": 140,
    "data/q5q6/representation_per_seed_layer.csv": 420,
    "data/q5q6/noise_severity_0_4_per_seed.csv": 420,
    "data/q5q6/architecture_update_per_seed.csv": 270,
    "data/audit/q1_samples_seen_curve_per_seed.csv": 350,
    "data/audit/q3_curve_auc_per_seed.csv": 60,
    "data/audit/q3_curve_auc_summary.csv": 12,
    "data/audit/q3_curve_auc_paired_contrasts.csv": 63,
}

EVIDENCE_ROOTS = (
    ("final_formal", "stage3_core", "results/formal/phase0_v1_1/stage3_core", True),
    ("final_formal", "stage3_matched_controls", "results/formal/phase0_v1_1/stage3_matched_controls", True),
    ("final_formal", "stage3_q1_complete", "results/formal/phase0_v1_1/stage3_q1_complete", True),
    ("final_formal", "stage3_q2_representation", "results/formal/phase0_v1_1/stage3_q2_representation", True),
    ("final_formal", "stage3_q3_noise", "results/formal/phase0_v1_1/stage3_q3_noise", True),
    ("final_formal", "stage3_q4_updates", "results/formal/phase0_v1_1/stage3_q4_updates", True),
    ("final_formal", "stage3_q5q6_sweeps", "results/formal/phase0_v1_1/stage3_q5q6_sweeps", True),
    ("final_formal", "stage3_final_audit_supplement", "results/formal/phase0_v1_1/stage3_final_audit_supplement", True),
    ("formal_governance", "stage1_representation_health", "results/formal/phase0_v1_1/stage1_representation_health", False),
    ("formal_governance", "stage1c_effective_rank_audit", "results/formal/phase0_v1_1/stage1c_effective_rank_audit", False),
    ("formal_governance", "stage2_q4_tooling", "results/formal/phase0_v1_1/stage2_q4_tooling", False),
    ("confirmation", "hybrid_hhb_confirmation", "results/hybrid_hhb_confirmation", True),
    ("diagnostic", "hybrid_depth_ablation", "results/hybrid_depth_ablation", False),
    ("diagnostic", "hebbian_followup_decision", "results/hebbian_followup_decision", False),
    ("exploratory", "historical_bp_seed0", "results/20260720T044850Z_bp_seed0", False),
    ("exploratory", "aborted_hebbian_seed0", "results/20260720T065348Z_hebbian_seed0", False),
    ("exploratory", "historical_hebbian_seed0", "results/20260720T065919Z_hebbian_seed0", False),
    ("exploratory", "random_decoder_seed0", "results/20260722T080715Z_random_decoder_seed0", False),
    ("exploratory", "q1_clean_v1", "results/q1_clean_v1", False),
    ("exploratory", "tuning", "results/tuning", False),
    ("exploratory", "reconstruction_sanity", "results/reconstruction_sanity", False),
    ("excluded", "results_recovery", "results/recovery", False),
    ("excluded", "q5q6_recovery", "results/formal/phase0_v1_1/stage3_q5q6_sweeps/_recovery", False),
    ("exploratory", "presentation_stage1b", "figures/presentation_stage1b", False),
)

# Captured immediately before Phase 1 with a read-only PowerShell metadata
# inventory. Keeping the snapshot explicit avoids repeatedly traversing and
# stat-ing thousands of multi-gigabyte checkpoint files in Python on Windows.
FROZEN_INVENTORY = {
    "results/formal/phase0_v1_1/stage3_core": (1395, 1489138515),
    "results/formal/phase0_v1_1/stage3_matched_controls": (428, 520713571),
    "results/formal/phase0_v1_1/stage3_q1_complete": (11, 431518),
    "results/formal/phase0_v1_1/stage3_q2_representation": (189, 211199986),
    "results/formal/phase0_v1_1/stage3_q3_noise": (14, 1093518),
    "results/formal/phase0_v1_1/stage3_q4_updates": (210, 355061198),
    "results/formal/phase0_v1_1/stage3_q5q6_sweeps": (6779, 6853814605),
    "results/formal/phase0_v1_1/stage3_final_audit_supplement": (9, 207890),
    "results/formal/phase0_v1_1/stage1_representation_health": (4, 51968),
    "results/formal/phase0_v1_1/stage1c_effective_rank_audit": (20, 782942),
    "results/formal/phase0_v1_1/stage2_q4_tooling": (26, 21074674),
    "results/hybrid_hhb_confirmation": (369, 386596092),
    "results/hybrid_depth_ablation": (169, 170940981),
    "results/hebbian_followup_decision": (6, 28855),
    "results/20260720T044850Z_bp_seed0": (9, 2018822),
    "results/20260720T065348Z_hebbian_seed0": (5, 427899),
    "results/20260720T065919Z_hebbian_seed0": (15, 3963358),
    "results/20260722T080715Z_random_decoder_seed0": (21, 22003674),
    "results/q1_clean_v1": (207, 205593227),
    "results/tuning": (1004, 835773065),
    "results/reconstruction_sanity": (5, 76413),
    "results/recovery": (19, 30446),
    "results/formal/phase0_v1_1/stage3_q5q6_sweeps/_recovery": (65, 11252662),
    "figures/presentation_stage1b": (12, 1253000),
    "results": (11200, 11083515220),
    "results/formal/phase0_v1_1": (9085, 9453570385),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_source_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ":" in relative:
        raise ValueError(f"Source path must be repository-relative: {relative}")
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_SOURCE_PARTS:
        raise ValueError(f"Forbidden recovery/exploratory/raw source: {relative}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden binary/generated source type: {relative}")


def _require_seed_set(rows: list[dict[str, str]], name: str) -> None:
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(range(5)):
        raise ValueError(f"{name}: expected paired seeds 0..4, found {sorted(seeds)}")


def validate_source_tables(repo_root: Path = ROOT) -> dict[str, int]:
    counts: dict[str, int] = {}
    for destination, expected in ROW_EXPECTATIONS.items():
        source = repo_root / SOURCE_FILES[destination]
        rows = csv_rows(source)
        counts[destination] = len(rows)
        if len(rows) != expected:
            raise ValueError(f"{source}: expected {expected} rows, found {len(rows)}")
        if "seed" in rows[0]:
            _require_seed_set(rows, destination)

    method_expectations = {
        "data/q1/per_seed_complete.csv": {"BBB", "HHH", "HHB", "HBB", "Random", "RBB", "RRB"},
        "data/q2/per_seed_layer_metrics.csv": {"BBB", "HHH", "HHB", "HBB", "RBB", "RRB"},
        "data/q3/per_seed_condition_metrics.csv": {"BBB", "HHH", "HHB", "HBB"},
        "data/q5q6/performance_per_seed.csv": {"BBB", "HHH", "HHB", "HBB"},
    }
    for destination, expected in method_expectations.items():
        methods = {row["method"] for row in csv_rows(repo_root / SOURCE_FILES[destination])}
        if methods != expected:
            raise ValueError(f"{destination}: expected methods {sorted(expected)}, found {sorted(methods)}")

    q2 = csv_rows(repo_root / SOURCE_FILES["data/q2/per_seed_layer_metrics.csv"])
    if {row["layer"] for row in q2} != {"h1", "h2", "z"}:
        raise ValueError("Q2 must contain exactly h1/h2/z")

    stage2d = json.loads(
        (repo_root / SOURCE_FILES["data/governance/stage2d_confirmation_decision.json"]).read_text(
            encoding="utf-8"
        )
    )
    if stage2d["decision"] != "FAIL" or stage2d["test_samples_accessed"] != 0:
        raise ValueError("Stage 2D historical FAIL/zero-test boundary changed")

    audit = json.loads(
        (repo_root / SOURCE_FILES["data/audit/protocol_audit.json"]).read_text(encoding="utf-8")
    )
    if audit["overall_decision"] != "PASS":
        raise ValueError("Final protocol audit is not PASS")
    if audit["stage2d_to_stage3"]["amendment_is_preregistered_success"]:
        raise ValueError("Stage 3 amendment must not be recorded as preregistered success")
    if audit["standardized_decoder_fairness"]["config_audit"]["config_count"] != 135:
        raise ValueError("Expected 135 standardized-decoder configs")

    late_rows = [
        row
        for row in csv_rows(repo_root / SOURCE_FILES["data/q5q6/performance_per_seed.csv"])
        if row["sweep"] == "architecture" and row["case"] == "late_heavy" and row["seed"] == "4"
    ]
    if len(late_rows) != 4 or {row["method"] for row in late_rows} != {"BBB", "HHH", "HHB", "HBB"}:
        raise ValueError("Late-heavy seed 4 is incomplete")
    return counts


def standardized_decoder_config_rows(repo_root: Path = ROOT) -> list[dict[str, Any]]:
    formal = repo_root / FORMAL_ROOT
    roots = [formal / "stage3_core/runs", formal / "stage3_matched_controls/runs"]
    roots.extend(
        formal / "stage3_q5q6_sweeps" / sweep / case / "runs"
        for sweep, cases in (
            ("dimension", ("L16", "L32", "L128")),
            ("architecture", ("early_heavy", "late_heavy")),
        )
        for case in cases
    )
    paths = sorted(path for root in roots for path in root.glob("seed_*/*/config_resolved.yaml"))
    expected = {
        "optimizer": "adam",
        "lr": 0.003,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "epochs": 10,
        "loss": "mse_pixel_mean",
        "validation_selection": "min_reconstruction_mse",
    }
    rows = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        matches = (
            config["standardized_decoder"] == expected
            and config["data"]["split_manifest"] == "data/splits/mnist_split_v1.npz"
            and int(config["data"]["batch_size"]) == 128
            and int(config["training"]["decoder_epochs"]) == 10
            and config["training"]["reconstruction_loss"] == "mse_pixel_mean"
            and config["protocol"]["test_access_policy"]
            == "validation_select_then_single_test_evaluation"
        )
        rows.append(
            {
                "logical_source": path.relative_to(repo_root).as_posix(),
                "sha256": sha256(path),
                "contract_matches": str(bool(matches)).lower(),
            }
        )
    if len(rows) != 135 or not all(row["contract_matches"] == "true" for row in rows):
        raise ValueError("Standardized-decoder config audit did not pass 135/135")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def build(output: Path = DEFAULT_OUTPUT, repo_root: Path = ROOT) -> Path:
    output = output if output.is_absolute() else repo_root / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite release output: {output}")
    if output.resolve().is_relative_to((repo_root / "results").resolve()):
        raise ValueError("Release output must not be inside results/")

    for source in SOURCE_FILES.values():
        validate_source_path(source)
        if not (repo_root / source).is_file():
            raise FileNotFoundError(repo_root / source)
    counts = validate_source_tables(repo_root)
    config_rows = standardized_decoder_config_rows(repo_root)

    output.mkdir(parents=True)
    bundle_records = []
    for destination, source in SOURCE_FILES.items():
        source_path = repo_root / source
        destination_path = output / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_digest = sha256(source_path)
        if destination in SANITIZED_JSON_FIELDS:
            value = json.loads(source_path.read_text(encoding="utf-8"))
            value.update(SANITIZED_JSON_FIELDS[destination])
            destination_path.write_text(
                json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source_path, destination_path)
        bundle_records.append(
            {
                "bundle_path": destination,
                "source_path": source,
                "source_sha256": source_digest,
                "sha256": sha256(destination_path),
                "bytes": destination_path.stat().st_size,
                "rows": counts.get(destination),
            }
        )

    config_audit_path = output / "data/audit/standardized_decoder_configs.csv"
    _write_csv(config_audit_path, config_rows)
    bundle_records.append(
        {
                "bundle_path": "data/audit/standardized_decoder_configs.csv",
                "source_path": "135 accepted config_resolved.yaml records (hash/index only)",
                "source_sha256": None,
                "sha256": sha256(config_audit_path),
            "bytes": config_audit_path.stat().st_size,
            "rows": 135,
        }
    )

    registry_rows = []
    for evidence_class, artifact_id, relative, bundled in EVIDENCE_ROOTS:
        if relative not in FROZEN_INVENTORY:
            raise ValueError(f"Missing frozen inventory record: {relative}")
        file_count, total_bytes = FROZEN_INVENTORY[relative]
        registry_rows.append(
            {
                "evidence_class": evidence_class,
                "artifact_id": artifact_id,
                "source_root": relative,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "compact_evidence_bundled": str(bundled).lower(),
                "disposition": (
                    "accepted aggregate evidence only"
                    if bundled
                    else "preserved locally; excluded from final compact claims"
                ),
            }
        )
    _write_csv(output / "artifact_registry.csv", registry_rows)

    protocol_audit = json.loads(
        (repo_root / SOURCE_FILES["data/audit/protocol_audit.json"]).read_text(encoding="utf-8")
    )
    raw_results_files, raw_results_bytes = FROZEN_INVENTORY["results"]
    formal_files, formal_bytes = FROZEN_INVENTORY["results/formal/phase0_v1_1"]
    manifest = {
        "schema_version": "v1.0-final-compact-evidence-v1",
        "release_id": "v1.0-final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audited_source_commit": _git_head(repo_root),
        "input_mode": "accepted aggregate CSV/JSON and resolved-config metadata only",
        "scientific_actions": {
            "datasets_loaded": False,
            "checkpoints_loaded": False,
            "training_performed": False,
            "model_evaluation_performed": False,
            "test_access_increment": 0,
        },
        "formal_seeds": [0, 1, 2, 3, 4],
        "statistics": {
            "paired_unit": "seed",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 2026,
            "confidence_level": 0.95,
            "primary_contrasts": ["HHB-HHH", "HBB-HHB", "BBB-HHB"],
            "protocol_mandated_secondary": ["HBB-RBB", "HHB-RRB"],
            "exploratory": ["Q4 cross-outcome correlations", "Q3 whole-curve AUC supplement"],
        },
        "governance": {
            "stage2d_decision": "FAIL",
            "stage2d_test_samples_accessed": 0,
            "stage3_amendment_approved_date": "2026-07-28",
            "stage3_is_post_confirmation_rescoping": True,
            "amendment_is_preregistered_success": False,
            "late_heavy_seed4_disposition": "retained as formal outcome",
        },
        "test_access_audit": protocol_audit["test_usage"],
        "row_expectations": ROW_EXPECTATIONS,
        "standardized_decoder_config_count": 135,
        "local_raw_inventory_at_freeze": {
            "capture_method": "independent read-only filesystem metadata audit before bundle creation",
            "results_files": raw_results_files,
            "results_bytes": raw_results_bytes,
            "formal_files": formal_files,
            "formal_bytes": formal_bytes,
            "raw_artifacts_bundled": False,
        },
        "bundle_files": sorted(bundle_records, key=lambda item: item["bundle_path"]),
        "excluded_content": [
            "checkpoints and model weights",
            "representation/embedding archives",
            "update tensors",
            "raw logs",
            "recovery and partial outputs",
            "exploratory/tuning results",
            "preliminary Stage 1B presentation figures",
        ],
        "governance_document_hashes": {
            logical: sha256(repo_root / current)
            for logical, current in FROZEN_GOVERNANCE_PATHS.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output).resolve())


if __name__ == "__main__":
    main()
