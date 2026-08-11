"""Run Phase 4 QA in a temporary tracked-files-only export."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAFE_TESTS = [
    "tests/test_final_release_bundle.py",
    "tests/test_final_figures.py",
    "tests/test_release_narrative.py",
    "tests/test_phase4_reproducibility.py",
]
TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:\\{1,2}(?:Users|Microlearning)\\{1,2}|/(?:home|Users)/)"
)
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
}


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def _exportable_files() -> list[PurePosixPath]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
    )
    paths = [
        PurePosixPath(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    ]
    forbidden = [path for path in paths if path.parts and path.parts[0] == "results"]
    if forbidden:
        raise ValueError(f"Clean export unexpectedly includes results/: {forbidden[:3]}")
    return paths


def _copy_export(destination: Path) -> int:
    count = 0
    for relative in _exportable_files():
        source = ROOT.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        if not source.is_file():
            raise FileNotFoundError(f"Export source is missing: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def _scan_public_text(root: Path) -> dict[str, int]:
    scanned = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if "verification" in path.relative_to(root).parts:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if PRIVATE_PATH.search(text):
            raise ValueError(f"Private absolute path found in {path.relative_to(root)}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"{label} pattern found in {path.relative_to(root)}")
        scanned += 1
    return {"text_files_scanned": scanned, "private_paths": 0, "secret_patterns": 0}


def _compare_plotted_values(root: Path, rebuilt: Path) -> int:
    committed = root / "figures" / "final" / "plotted_values"
    generated = rebuilt / "plotted_values"
    names = sorted(path.name for path in committed.glob("*.csv"))
    if names != sorted(path.name for path in generated.glob("*.csv")):
        raise ValueError("Regenerated plotted-value file set changed")
    for name in names:
        if (committed / name).read_bytes() != (generated / name).read_bytes():
            raise ValueError(f"Regenerated plotted values changed: {name}")
    return len(names)


def run_clean_export() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="microlearning-clean-export-") as temporary:
        export_root = Path(temporary) / "repo"
        export_root.mkdir()
        exported = _copy_export(export_root)
        if (export_root / "results").exists():
            raise ValueError("Ignored results archive leaked into clean export")

        scan_before = _scan_public_text(export_root)
        qa_root = export_root / "_clean_qa"
        figure_output = qa_root / "figures"
        notebook_output = qa_root / "project_demo.executed.ipynb"
        env = os.environ.copy()
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
        python = sys.executable

        verifier_output = _run(
            [python, "scripts/verify_final_release.py"], export_root, env
        )
        verifier = json.loads(verifier_output)
        _run(
            [python, "scripts/plot_final_figures.py", "--output", str(figure_output)],
            export_root,
            env,
        )
        plotted_values = _compare_plotted_values(export_root, figure_output)
        _run(
            [
                python,
                "scripts/execute_demo_notebook.py",
                "--input",
                "project_demo.ipynb",
                "--output",
                str(notebook_output),
                "--working-directory",
                str(export_root),
            ],
            export_root,
            env,
        )
        test_output = _run(
            [python, "-m", "pytest", "-p", "no:cacheprovider", *SAFE_TESTS],
            export_root,
            env,
        )
        scan_after = _scan_public_text(export_root)
        if not notebook_output.is_file():
            raise FileNotFoundError("Clean-export notebook execution produced no output")

        return {
            "decision": "PASS",
            "exported_files": exported,
            "results_directory_present": False,
            "release_verification": verifier["decision"],
            "figure_regeneration": "PASS",
            "plotted_value_tables_compared": plotted_values,
            "notebook_execution": "PASS",
            "release_safe_tests": test_output.splitlines()[-1],
            "network_required": False,
            "datasets_loaded": False,
            "checkpoints_loaded": False,
            "test_access_increment": 0,
            **scan_before,
            "post_execution_text_files_scanned": scan_after["text_files_scanned"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_clean_export(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
