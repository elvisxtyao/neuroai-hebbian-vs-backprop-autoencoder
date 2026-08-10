from __future__ import annotations

import json
import shutil

import pytest

from scripts.build_final_release import build, validate_source_path, validate_source_tables
from scripts.verify_final_release import DEFAULT_RELEASE, verify


def test_committed_compact_release_bundle_passes():
    result = verify(DEFAULT_RELEASE)
    assert result["decision"] == "PASS"
    assert result["formal_seeds"] == [0, 1, 2, 3, 4]
    assert result["test_access_increment"] == 0


def test_release_notes_are_metadata_outside_frozen_evidence_checksums():
    notes = DEFAULT_RELEASE / "RELEASE_NOTES.md"
    if notes.exists():
        checked = (DEFAULT_RELEASE / "checksums.sha256").read_text(encoding="utf-8")
        assert "RELEASE_NOTES.md" not in checked
        assert verify(DEFAULT_RELEASE)["decision"] == "PASS"


def test_compact_bundle_contains_no_windows_absolute_paths():
    for path in DEFAULT_RELEASE.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            text = path.read_text(encoding="utf-8")
            assert ":\\\\" not in text


def test_formal_source_tables_have_frozen_counts_methods_and_seeds():
    counts = validate_source_tables()
    assert counts["data/q1/per_seed_complete.csv"] == 35
    assert counts["data/q5q6/architecture_update_per_seed.csv"] == 270


def test_builder_refuses_to_overwrite_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        build(output)


@pytest.mark.parametrize(
    "relative",
    [
        "results/recovery/partial.csv",
        "results/formal/phase0_v1_1/stage/_recovery/table.csv",
        "results/q1_clean_v1/summary.csv",
        "results/formal/phase0_v1_1/stage3_core/model_best.pt",
        "results/formal/phase0_v1_1/stage3_q2_representation/representations/data.csv",
    ],
)
def test_builder_rejects_recovery_exploratory_and_raw_inputs(relative):
    with pytest.raises(ValueError, match="Forbidden"):
        validate_source_path(relative)


def test_verifier_rejects_hash_mismatch(tmp_path):
    copied = tmp_path / "release"
    shutil.copytree(DEFAULT_RELEASE, copied)
    path = copied / "data/q1/per_seed_complete.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(copied)


def test_verifier_rejects_governance_reclassification(tmp_path):
    copied = tmp_path / "release"
    shutil.copytree(DEFAULT_RELEASE, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_actions"]["training_performed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden scientific action"):
        verify(copied)
