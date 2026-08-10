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
FORBIDDEN_PARTS = {"_recovery", "recovery", "checkpoints", "representations", "embeddings"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".log"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_relative(path: str) -> None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ":" in path or ".." in pure.parts:
        raise ValueError(f"Non-portable bundle path: {path}")
    if {part.lower() for part in pure.parts} & FORBIDDEN_PARTS:
        raise ValueError(f"Forbidden raw/recovery bundle path: {path}")
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden raw binary in compact bundle: {path}")


def verify(release_root: Path = DEFAULT_RELEASE) -> dict[str, Any]:
    manifest_path = release_root / "manifest.json"
    checksums_path = release_root / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError("Release manifest/checksums are missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["release_id"] != "v1.0-final":
        raise ValueError("Unexpected release id")
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
        digest, relative = line.split("  ", 1)
        _validate_relative(relative)
        expected_checksums[relative] = digest
    actual_files = {
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
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

    listed = {record["bundle_path"]: record for record in manifest["bundle_files"]}
    for relative, record in listed.items():
        _validate_relative(relative)
        path = release_root / relative
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Manifest mismatch: {relative}")
        if record["rows"] is not None and len(_rows(path)) != int(record["rows"]):
            raise ValueError(f"Row-count mismatch: {relative}")

    row_expectations = manifest["row_expectations"]
    for relative, expected in row_expectations.items():
        rows = _rows(release_root / relative)
        if len(rows) != int(expected):
            raise ValueError(f"Expected {expected} rows in {relative}, found {len(rows)}")
        if rows and "seed" in rows[0]:
            seeds = {int(row["seed"]) for row in rows}
            if seeds != set(range(5)):
                raise ValueError(f"Incomplete paired seeds in {relative}")

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
    return {
        "decision": "PASS",
        "release_id": manifest["release_id"],
        "bundle_files": len(actual_files),
        "formal_seeds": manifest["formal_seeds"],
        "test_access_increment": actions["test_access_increment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    print(json.dumps(verify(args.release_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
