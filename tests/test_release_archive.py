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
    assert len(members) == 86
    assert members == sorted(members)
    assert all(member.startswith(f"{ARCHIVE_ROOT.as_posix()}/") for member in members)
    assert f"{ARCHIVE_ROOT}/project_demo.ipynb" in members
    assert f"{ARCHIVE_ROOT}/LICENSE" in members
    assert f"{ARCHIVE_ROOT}/THIRD_PARTY_NOTICES.md" in members
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
    assert first_result["files"] == second_result["files"] == 86
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


def test_repository_license_and_third_party_notice_are_complete():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    audit = (ROOT / "docs/final/FINAL_RELEASE_AUDIT.md").read_text(encoding="utf-8")
    migration = (ROOT / "docs/tutorial_migration.md").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Xiaotian Yao" in license_text
    for required in (
        "Copyright 2020 Neuromatch Academy",
        "Redistribution and use in source and binary forms",
        "Neither the name of the copyright holder",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"',
        "f8cdef10d7463ff626b8c6555a29a0fd918b9fd4",
        "do not endorse this project",
    ):
        assert required in notice
    assert "License status: **PASS**" in audit
    assert "were written entirely by the owner" in audit
    assert "Neuromatch software-provenance item is resolved" in migration
    for stale in (
        "cannot yet be safely declared MIT",
        "Phase 0 BP ownership boundary are unresolved",
        "upstream URL/version and license or sharing terms remain unknown",
    ):
        assert stale not in audit
        assert stale not in migration
