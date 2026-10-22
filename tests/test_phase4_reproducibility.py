from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import nbformat

from scripts.plot_final_figures import build
from scripts.verify_final_release import DEFAULT_RELEASE, verify


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "project_demo.ipynb"
SAFE_IMPORT_DENYLIST = {"data", "evaluation", "models", "torch", "torchvision", "training"}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def test_public_verifier_and_figure_builder_have_release_safe_import_boundaries():
    for relative in ("scripts/verify_final_release.py", "scripts/plot_final_figures.py"):
        assert not (_top_level_imports(ROOT / relative) & SAFE_IMPORT_DENYLIST)


def test_public_verifier_checks_schema_figures_and_plotted_values():
    result = verify(DEFAULT_RELEASE)
    assert result == {
        "bundle_files": 46,
        "decision": "PASS",
        "figure_sources": 16,
        "formal_seeds": [0, 1, 2, 3, 4],
        "plotted_value_tables": 17,
        "release_id": "v1.0-final",
        "test_access_increment": 0,
    }


def test_artifact_only_figure_rebuild_preserves_all_plotted_values(tmp_path):
    output = tmp_path / "rebuilt"
    build(output)
    expected_values = ROOT / "figures/final/plotted_values"
    rebuilt_values = output / "plotted_values"
    expected_names = sorted(path.name for path in expected_values.glob("*.csv"))
    assert expected_names == sorted(path.name for path in rebuilt_values.glob("*.csv"))
    for name in expected_names:
        assert (rebuilt_values / name).read_bytes() == (expected_values / name).read_bytes()
    committed = json.loads((ROOT / "figures/final/source_manifest.json").read_text(encoding="utf-8"))
    rebuilt = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
    assert rebuilt["figures"] == committed["figures"]
    assert rebuilt["scientific_actions"] == committed["scientific_actions"]


def test_demo_notebook_is_valid_executed_and_artifact_only():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(cell.outputs for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.outputs
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    code_source = "\n".join(cell.source for cell in code_cells)
    lowered = code_source.lower()
    for forbidden in (
        "results/",
        "_recovery",
        "checkpoint",
        "torchvision",
        "requests.",
        "urllib.",
        "http://",
        "https://",
    ):
        assert forbidden not in lowered
    assert "release/v1.0-final" in source
    assert "read_csv('data/q1/method_summary.csv')" in source
    assert "read_csv('data/q2/method_layer_summary.csv')" in source
    assert "read_csv('data/q4/method_layer_update_summary.csv')" in source
    plotted_cells = [cell for cell in code_cells if cell.id.startswith("plot-")]
    assert all(
        any(output.output_type == "display_data" for output in cell.outputs)
        for cell in plotted_cells
    )


def test_demo_ends_with_exactly_three_takeaways():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    final = notebook.cells[-1]
    assert final.cell_type == "markdown"
    takeaways = re.findall(r"^[1-3]\. .+$", final.source, flags=re.MULTILINE)
    assert takeaways == [
        "1. Shallow Hebbian prefixes retain useful classification information.",
        "2. Fully local deep stacking produces severe low-dimensional compression under this tested rule.",
        "3. BP restores representation dimensionality at the layer where it is introduced, with earlier intervention improving broader system behavior.",
    ]


def test_release_dependencies_exclude_training_stack():
    requirements = (ROOT / "requirements-release.txt").read_text(encoding="utf-8").lower()
    assert "torch" not in requirements
    assert "torchvision" not in requirements
    assert "scikit-learn" not in requirements


def test_reproducibility_guide_has_three_tiers_and_exact_public_commands():
    guide = (ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    for heading in (
        "## Tier 1 — Artifact-only reproduction",
        "## Tier 2 — Existing full local archive verification",
        "## Tier 3 — Expensive historical full reproduction",
    ):
        assert heading in guide
    for command in (
        "python scripts/verify_final_release.py",
        "python scripts/plot_final_figures.py --output build/v1.0-final-figures",
        "python scripts/execute_demo_notebook.py",
        "python scripts/verify_local_archive.py",
        "python scripts/qa_clean_export.py",
    ):
        assert command in guide
    assert "does **not** promise bitwise-identical training" in guide


def test_readme_phase4_cleanup_is_complete():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "five paired seeds `[0, 1, 2, 3, 4]`" in readme
    assert "seed is the statistical unit" in readme
    assert "frozen 95% bootstrap confidence intervals" in readme
    assert "Ignored local full archive" in readme
    assert "[reproducibility guide](REPRODUCIBILITY.md)" in readme
    assert "[artifact-only project demo](project_demo.ipynb)" in readme
    assert "reserved for the reproducibility phase" not in readme


def test_public_docs_and_notebook_do_not_expose_private_absolute_paths():
    paths = [ROOT / "REPRODUCIBILITY.md", ROOT / "README.md", NOTEBOOK]
    private = re.compile(
        r"(?:[A-Za-z]:\\{1,2}(?:Users|Microlearning)\\{1,2}|/(?:home|Users)/)"
    )
    for path in paths:
        assert not private.search(path.read_text(encoding="utf-8")), path
