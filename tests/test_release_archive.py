from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_release_archive import (
    ARCHIVE_ROOT,
    FORBIDDEN_PARTS,
    archive_inventory,
    build_archive,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_archive_inventory_is_fixed_and_public_safe():
    inventory = archive_inventory(ROOT)
    members = [member.as_posix() for _, member in inventory]
    assert len(members) == 84
    assert members == sorted(members)
    assert all(member.startswith(f"{ARCHIVE_ROOT.as_posix()}/") for member in members)
    assert f"{ARCHIVE_ROOT}/project_demo.ipynb" in members
    assert f"{ARCHIVE_ROOT}/release/v1.0-final/RELEASE_NOTES.md" in members
    assert f"{ARCHIVE_ROOT}/figures/final/source_manifest.json" in members
    for member in members:
        assert not ({part.lower() for part in Path(member).parts} & FORBIDDEN_PARTS)


def test_release_archive_is_byte_deterministic(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_result = build_archive(first, ROOT)
    second_result = build_archive(second, ROOT)
    assert first.read_bytes() == second.read_bytes()
    assert first_result["files"] == second_result["files"] == 84
    assert first_result["checksum_written"] is False
    with ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_release_archive_refuses_overwrite(tmp_path):
    output = tmp_path / "candidate.zip"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_archive(output, ROOT)
    assert output.read_bytes() == b"preserve"
