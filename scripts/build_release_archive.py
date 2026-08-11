"""Build the deterministic, public-safe v1.0-final evidence archive.

The archive is deliberately narrower than a source checkout. It contains the
compact frozen evidence, final figures and release-facing documents only. It
does not read ``results/`` or include training code, datasets, checkpoints,
raw representations, update tensors, recovery outputs or tutorial sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_final_release import DEFAULT_RELEASE, verify


DEFAULT_OUTPUT = ROOT / "dist" / "v1.0-final-evidence.zip"
ARCHIVE_ROOT = PurePosixPath("v1.0-final-evidence")

DOCUMENTS = (
    "FINAL_REPORT.md",
    "LICENSE",
    "RESULTS_SUMMARY.md",
    "REPRODUCIBILITY.md",
    "THIRD_PARTY_NOTICES.md",
    "project_demo.ipynb",
)
DIRECTORIES = (
    "release/v1.0-final",
    "figures/final",
)

FORBIDDEN_PARTS = {
    "_recovery",
    "recovery",
    "results",
    "checkpoints",
    "representations",
    "embeddings",
    "presentation_stage1b",
    "tutorial",
    "tutorials",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".log",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_inventory(repo_root: Path = ROOT) -> list[tuple[Path, PurePosixPath]]:
    """Return the fixed, sorted source-to-member archive inventory."""

    inventory: list[tuple[Path, PurePosixPath]] = []
    for relative in DIRECTORIES:
        directory = repo_root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for source in directory.rglob("*"):
            if source.is_file():
                member = ARCHIVE_ROOT / source.relative_to(repo_root).as_posix()
                inventory.append((source, member))
    for relative in DOCUMENTS:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        inventory.append((source, ARCHIVE_ROOT / relative))

    inventory.sort(key=lambda item: item[1].as_posix())
    members = [member.as_posix() for _, member in inventory]
    if len(members) != len(set(members)):
        raise ValueError("Archive inventory contains duplicate member paths")
    for source, member in inventory:
        lowered = {part.lower() for part in member.parts}
        if lowered & FORBIDDEN_PARTS:
            raise ValueError(f"Forbidden archive path: {member}")
        if source.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Forbidden archive file type: {member}")
    return inventory


def build_archive(
    output: Path = DEFAULT_OUTPUT,
    repo_root: Path = ROOT,
    *,
    write_checksum: bool = False,
) -> dict[str, object]:
    """Create one deterministic ZIP and optionally its SHA-256 sidecar."""

    output = output if output.is_absolute() else repo_root / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Release archive output must end in .zip")

    # Verify the frozen compact bundle before packaging it. This verification
    # is artifact-only and does not import models or access a dataset.
    release_root = repo_root / DEFAULT_RELEASE.relative_to(ROOT)
    verification = verify(release_root)
    if verification["decision"] != "PASS":
        raise ValueError("Compact release verification did not pass")

    inventory = archive_inventory(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source, member in inventory:
            info = ZipInfo(member.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)

    result: dict[str, object] = {
        "archive": output.as_posix(),
        "files": len(inventory),
        "bytes": output.stat().st_size,
        "checksum_written": False,
    }
    if write_checksum:
        checksum_path = output.with_suffix(output.suffix + ".sha256")
        if checksum_path.exists():
            raise FileExistsError(f"Refusing to overwrite checksum: {checksum_path}")
        digest = _sha256(output)
        checksum_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
        result.update(
            {
                "sha256": digest,
                "checksum": checksum_path.as_posix(),
                "checksum_written": True,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="Write .zip.sha256 only after the candidate contents are final.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_archive(args.output, write_checksum=args.write_checksum),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
