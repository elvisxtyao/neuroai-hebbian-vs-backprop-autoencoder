from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> tuple[Path, dict[str, object], str]:
    path = WORKFLOWS / name
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return path, parsed, text


def test_ci_runs_the_complete_artifact_only_boundary():
    _, workflow, text = _workflow("ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request" in workflow["on"]
    assert "push" in workflow["on"]
    assert 'python-version: "3.11"' in text
    assert "requirements-release.txt" in text
    assert "python scripts/verify_final_release.py" in text
    assert "python scripts/qa_clean_export.py" in text
    assert "persist-credentials: false" in text
    for test in (
        "test_final_release_bundle.py",
        "test_final_figures.py",
        "test_release_narrative.py",
        "test_phase4_reproducibility.py",
        "test_release_archive.py",
        "test_ci_workflows.py",
    ):
        assert test in text
    for forbidden in (" -r requirements.txt", "torchvision", "MNIST", "results/"):
        assert forbidden not in text


def test_release_workflow_verifies_determinism_without_publishing():
    _, workflow, text = _workflow("release-verification.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" in workflow["on"]
    assert "tags" in workflow["on"]["push"]
    assert text.count("python scripts/build_release_archive.py") == 2
    assert "cmp dist/v1.0-final-evidence.zip" in text
    assert "sha256sum --check" in text
    assert "actions/upload-artifact@v6" in text
    assert "verified-evidence-candidate-${{ github.sha }}" in text
    assert "persist-credentials: false" in text
    assert "gh release" not in text
    assert "contents: write" not in text


def test_readme_exposes_ci_and_accurate_repository_structure():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "actions/workflows/release-verification.yml/badge.svg" in readme
    structure = readme.split("## Repository Structure", 1)[1].split(
        "## Documentation", 1
    )[0]
    assert "learning_rules/, models/" in structure
    assert "ae/" not in structure
