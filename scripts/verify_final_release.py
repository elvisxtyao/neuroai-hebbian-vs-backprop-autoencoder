"""Verify a compact v1.0-final evidence bundle without local raw results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "release" / "v1.0-final"
EXPECTED_SCHEMA_VERSION = "v1.0-final-compact-evidence-v1"
FORMAL_SEEDS = {0, 1, 2, 3, 4}
FORBIDDEN_PARTS = {"_recovery", "recovery", "checkpoints", "representations", "embeddings"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".log"}
RELEASE_METADATA_FILES = {"RELEASE_NOTES.md"}
REQUIRED_MANIFEST_KEYS = {
    "audited_source_commit",
    "bundle_files",
    "formal_seeds",
    "governance",
    "release_id",
    "row_expectations",
    "schema_version",
    "scientific_actions",
    "statistics",
    "test_access_audit",
}
REQUIRED_TABLE_COLUMNS = {
    "data/q1/per_seed_complete.csv": {
        "seed",
        "method",
        "accuracy",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    },
    "data/q2/per_seed_layer_metrics.csv": {
        "seed",
        "method",
        "layer",
        "effective_rank",
        "linear_probe_cv_accuracy",
    },
    "data/q3/per_seed_condition_metrics.csv": {
        "seed",
        "method",
        "noise_type",
        "severity",
        "accuracy",
    },
    "data/q4/per_seed_layer_update_metrics.csv": {
        "seed",
        "method",
        "layer",
        "rule",
        "alignment",
        "scale_matched_bias",
        "update_snr_linear",
    },
    "data/q5q6/performance_per_seed.csv": {
        "sweep",
        "case",
        "seed",
        "method",
        "accuracy",
    },
    "data/q5q6/representation_per_seed_layer.csv": {
        "sweep",
        "case",
        "seed",
        "method",
        "layer",
        "effective_rank",
    },
    "data/q5q6/noise_severity_0_4_per_seed.csv": {
        "sweep",
        "case",
        "seed",
        "method",
        "noise_type",
        "accuracy",
    },
    "data/q5q6/architecture_update_per_seed.csv": {
        "case",
        "seed",
        "method",
        "layer",
        "rule",
        "alignment",
    },
    "data/audit/q1_samples_seen_curve_per_seed.csv": {
        "seed",
        "method",
        "samples_seen",
        "validation_reconstruction_mse",
    },
    "data/audit/standardized_decoder_configs.csv": {
        "logical_source",
        "sha256",
        "contract_matches",
    },
}
EXPECTED_FIGURES = {
    "fig1_performance",
    "fig2_prefix_value_training_cost",
    "fig3_layerwise_representation",
    "fig4_update_mechanism",
    "fig5_robustness",
    "fig6_dimension_architecture",
    "method_design",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_keys(record: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(record)
    if missing:
        raise ValueError(f"Missing {label} fields: {sorted(missing)}")


def _validate_relative(path: str) -> None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ":" in path or ".." in pure.parts:
        raise ValueError(f"Non-portable bundle path: {path}")
    if {part.lower() for part in pure.parts} & FORBIDDEN_PARTS:
        raise ValueError(f"Forbidden raw/recovery bundle path: {path}")
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden raw binary in compact bundle: {path}")


def _validate_manifest_schema(manifest: dict[str, Any]) -> None:
    _require_keys(manifest, REQUIRED_MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise ValueError("Unexpected release manifest schema")
    if manifest["release_id"] != "v1.0-final":
        raise ValueError("Unexpected release id")
    if manifest["formal_seeds"] != sorted(FORMAL_SEEDS):
        raise ValueError("Unexpected formal seed set")
    if not isinstance(manifest["bundle_files"], list) or not manifest["bundle_files"]:
        raise ValueError("Manifest bundle_files must be a non-empty list")
    if not isinstance(manifest["row_expectations"], dict):
        raise ValueError("Manifest row_expectations must be a mapping")
    for required in REQUIRED_TABLE_COLUMNS:
        if required not in {
            record.get("bundle_path") for record in manifest["bundle_files"]
        }:
            raise ValueError(f"Required release table is absent: {required}")


def _verify_figure_sources(
    repository_root: Path,
    release_root: Path,
    release_hashes: dict[str, str],
) -> dict[str, int]:
    figures_root = repository_root / "figures" / "final"
    manifest_path = figures_root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Final figure source manifest is missing")
    figure_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_keys(
        figure_manifest,
        {
            "builder",
            "figures",
            "formal_seeds",
            "generated_file_hashes",
            "release_id",
            "scientific_actions",
        },
        "figure manifest",
    )
    if figure_manifest["release_id"] != "v1.0-final":
        raise ValueError("Unexpected figure release id")
    if figure_manifest["formal_seeds"] != sorted(FORMAL_SEEDS):
        raise ValueError("Figure manifest formal seeds changed")
    if set(figure_manifest["figures"]) != EXPECTED_FIGURES:
        raise ValueError("Final figure set is incomplete")
    actions = figure_manifest["scientific_actions"]
    if any(value for key, value in actions.items() if key != "test_access_increment"):
        raise ValueError("Figure manifest records a forbidden scientific action")
    if actions["test_access_increment"] != 0:
        raise ValueError("Figure generation added test access")

    builder = figure_manifest["builder"]
    _require_keys(builder, {"path", "sha256"}, "figure builder")
    _validate_relative(builder["path"])
    builder_path = repository_root / builder["path"]
    if not builder_path.is_file() or sha256(builder_path) != builder["sha256"]:
        raise ValueError("Figure builder hash mismatch")

    source_count = 0
    for name, record in figure_manifest["figures"].items():
        _require_keys(record, {"sources", "source_hashes"}, f"figure {name}")
        if set(record["sources"]) != set(record["source_hashes"]):
            raise ValueError(f"Figure source/hash mismatch: {name}")
        for relative in record["sources"]:
            _validate_relative(relative)
            if not relative.startswith("data/"):
                raise ValueError(f"Figure source is outside compact data: {relative}")
            expected = record["source_hashes"][relative]
            if release_hashes.get(relative) != expected:
                raise ValueError(f"Figure source is not pinned by release manifest: {relative}")
            if sha256(release_root / relative) != expected:
                raise ValueError(f"Figure source hash mismatch: {relative}")
            source_count += 1

    generated = figure_manifest["generated_file_hashes"]
    plotted_count = 0
    for relative, digest in generated.items():
        _validate_relative(relative)
        path = figures_root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Committed figure artifact hash mismatch: {relative}")
        if relative.startswith("plotted_values/"):
            plotted_count += 1
    if plotted_count == 0:
        raise ValueError("No plotted-value tables were verified")
    return {"figure_sources": source_count, "plotted_value_tables": plotted_count}


def verify(
    release_root: Path = DEFAULT_RELEASE,
    *,
    repository_root: Path | None = None,
    verify_figure_sources: bool = True,
) -> dict[str, Any]:
    release_root = release_root.resolve()
    repository_root = (repository_root or ROOT).resolve()
    manifest_path = release_root / "manifest.json"
    checksums_path = release_root / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError("Release manifest/checksums are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_schema(manifest)
    actions = manifest["scientific_actions"]
    if any(
        actions[key]
        for key in (
            "datasets_loaded",
            "checkpoints_loaded",
            "training_performed",
            "model_evaluation_performed",
        )
    ) or actions["test_access_increment"] != 0:
        raise ValueError("Compact release records a forbidden scientific action")

    expected_checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  .+", line):
            raise ValueError(f"Malformed checksum line: {line!r}")
        digest, relative = line.split("  ", 1)
        _validate_relative(relative)
        if relative in expected_checksums:
            raise ValueError(f"Duplicate checksum path: {relative}")
        expected_checksums[relative] = digest
    actual_files = {
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and path.relative_to(release_root).as_posix() not in RELEASE_METADATA_FILES
    }
    if actual_files != set(expected_checksums):
        raise ValueError("Bundle file set does not match checksums.sha256")
    for relative, digest in expected_checksums.items():
        path = release_root / relative
        if sha256(path) != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        if path.suffix.lower() in {".csv", ".json"} and re.search(
            r"[A-Za-z]:\\\\", path.read_text(encoding="utf-8")
        ):
            raise ValueError(f"Absolute local path leaked into compact bundle: {relative}")

    listed: dict[str, dict[str, Any]] = {}
    for record in manifest["bundle_files"]:
        _require_keys(
            record,
            {"bundle_path", "bytes", "rows", "sha256", "source_path", "source_sha256"},
            "bundle record",
        )
        relative = record["bundle_path"]
        if relative in listed:
            raise ValueError(f"Duplicate manifest bundle path: {relative}")
        listed[relative] = record
    bundled_data_files = {relative for relative in actual_files if relative.startswith("data/")}
    if set(listed) != bundled_data_files:
        raise ValueError("Manifest bundle records do not match compact data files")
    for relative, record in listed.items():
        _validate_relative(relative)
        _validate_relative(record["source_path"])
        path = release_root / relative
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Manifest mismatch: {relative}")
        if expected_checksums.get(relative) != record["sha256"]:
            raise ValueError(f"Manifest/checksum disagreement: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Byte-count mismatch: {relative}")
        if record["rows"] is not None and len(_rows(path)) != int(record["rows"]):
            raise ValueError(f"Row-count mismatch: {relative}")

    row_expectations = manifest["row_expectations"]
    for relative, expected in row_expectations.items():
        rows = _rows(release_root / relative)
        if len(rows) != int(expected):
            raise ValueError(f"Expected {expected} rows in {relative}, found {len(rows)}")
        if rows and "seed" in rows[0]:
            seeds = {int(row["seed"]) for row in rows}
            if seeds != FORMAL_SEEDS:
                raise ValueError(f"Incomplete paired seeds in {relative}")

    for relative, required_columns in REQUIRED_TABLE_COLUMNS.items():
        rows = _rows(release_root / relative)
        if not rows:
            raise ValueError(f"Required release table is empty: {relative}")
        missing = required_columns - set(rows[0])
        if missing:
            raise ValueError(f"Missing columns in {relative}: {sorted(missing)}")

    q1_rows = _rows(release_root / "data/q1/per_seed_complete.csv")
    if {row["method"] for row in q1_rows} != {
        "BBB",
        "HBB",
        "HHB",
        "HHH",
        "RBB",
        "RRB",
        "Random",
    }:
        raise ValueError("Q1 method matrix is incomplete")

    stage2d = json.loads(
        (release_root / "data/governance/stage2d_confirmation_decision.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (release_root / "data/audit/protocol_audit.json").read_text(encoding="utf-8")
    )
    if stage2d["decision"] != "FAIL" or stage2d["test_samples_accessed"] != 0:
        raise ValueError("Stage 2D governance boundary changed")
    if audit["overall_decision"] != "PASS":
        raise ValueError("Protocol audit is not PASS")
    if audit["stage2d_to_stage3"]["amendment_is_preregistered_success"]:
        raise ValueError("Stage 3 re-scoping was misclassified")

    config_rows = _rows(release_root / "data/audit/standardized_decoder_configs.csv")
    if len(config_rows) != 135 or {row["contract_matches"] for row in config_rows} != {"true"}:
        raise ValueError("Standardized-decoder config evidence is incomplete")
    figure_counts = {"figure_sources": 0, "plotted_value_tables": 0}
    if verify_figure_sources:
        figure_counts = _verify_figure_sources(
            repository_root,
            release_root,
            {relative: record["sha256"] for relative, record in listed.items()},
        )
    return {
        "decision": "PASS",
        "release_id": manifest["release_id"],
        "bundle_files": len(actual_files),
        "formal_seeds": manifest["formal_seeds"],
        "test_access_increment": actions["test_access_increment"],
        **figure_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--skip-figure-sources", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.release_root,
                repository_root=args.repository_root,
                verify_figure_sources=not args.skip_figure_sources,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
