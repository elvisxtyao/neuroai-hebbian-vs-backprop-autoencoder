"""Read-only verification for maintainers who possess the ignored full archive.

This is deliberately separate from public artifact verification. It hashes only
accepted aggregate source files, resolved YAML configurations, and governance
documents; it never imports training/evaluation code or loads datasets,
checkpoints, representation arrays, update tensors, or recovery contents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "release" / "v1.0-final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Local archive path is missing: {root}")
    files = 0
    bytes_seen = 0
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.name == ".git":
                    continue
                if entry.is_dir(follow_symlinks=False):
                    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                    if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                        continue
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    bytes_seen += entry.stat(follow_symlinks=False).st_size
    return {"files": files, "bytes": bytes_seen}


def verify_local_archive(
    repository_root: Path = ROOT,
    release_root: Path = DEFAULT_RELEASE,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    release_root = release_root.resolve()
    sys.path.insert(0, str(repository_root))
    from scripts.verify_final_release import verify

    public = verify(release_root, repository_root=repository_root)
    manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))

    with (release_root / "artifact_registry.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        registry = list(csv.DictReader(handle))
    inventory_roots_verified = 0
    for record in registry:
        if (
            record["evidence_class"] == "excluded"
            or "recovery" in record["source_root"].lower()
            or record["artifact_id"] == "stage3_q5q6_sweeps"
        ):
            continue
        source_root = repository_root / record["source_root"]
        observed = _inventory(source_root)
        expected = {
            "files": int(record["file_count"]),
            "bytes": int(record["total_bytes"]),
        }
        if observed != expected:
            raise ValueError(
                f"Archive inventory mismatch for {record['artifact_id']}: "
                f"{observed} != {expected}"
            )
        inventory_roots_verified += 1

    aggregate_hashes = 0
    for record in manifest["bundle_files"]:
        source_digest = record.get("source_sha256")
        if not source_digest:
            continue
        source = repository_root / record["source_path"]
        if not source.is_file() or sha256(source) != source_digest:
            raise ValueError(f"Frozen aggregate source mismatch: {record['source_path']}")
        aggregate_hashes += 1

    with (release_root / "data/audit/standardized_decoder_configs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        configs = list(csv.DictReader(handle))
    for record in configs:
        source = repository_root / record["logical_source"]
        if record["contract_matches"] != "true":
            raise ValueError(f"Config contract is not accepted: {record['logical_source']}")
        if not source.is_file() or sha256(source) != record["sha256"]:
            raise ValueError(f"Resolved config fingerprint mismatch: {record['logical_source']}")

    for relative, expected in manifest["governance_document_hashes"].items():
        path = repository_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Governance document fingerprint mismatch: {relative}")

    frozen_inventory = manifest["local_raw_inventory_at_freeze"]
    return {
        "decision": "PASS",
        "public_artifact_verification": public["decision"],
        "frozen_results_files_declared": int(frozen_inventory["results_files"]),
        "frozen_results_bytes_declared": int(frozen_inventory["results_bytes"]),
        "frozen_formal_files_declared": int(frozen_inventory["formal_files"]),
        "frozen_formal_bytes_declared": int(frozen_inventory["formal_bytes"]),
        "inventory_roots_verified": inventory_roots_verified,
        "excluded_or_recovery_roots_opened": 0,
        "aggregate_source_hashes": aggregate_hashes,
        "resolved_config_hashes": len(configs),
        "datasets_loaded": False,
        "checkpoints_loaded": False,
        "test_access_increment": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_local_archive(args.repository_root, args.release_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
