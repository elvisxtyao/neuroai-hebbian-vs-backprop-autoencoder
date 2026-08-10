from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NARRATIVE_FILES = [
    ROOT / "FINAL_REPORT.md",
    ROOT / "RESULTS_SUMMARY.md",
    ROOT / "README.md",
    ROOT / "release/v1.0-final/RELEASE_NOTES.md",
    ROOT / "docs/final/FIGURE_CAPTIONS.md",
]


def _read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _local_markdown_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        targets.append((path.parent / target.split("#", 1)[0]).resolve())
    return targets


def test_required_narrative_files_and_local_links_exist():
    for path in NARRATIVE_FILES:
        assert path.is_file(), path
        for target in _local_markdown_targets(path):
            assert target.exists(), f"Broken local link in {path}: {target}"


def test_headline_numbers_match_compact_frozen_tables():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    q1 = _read_csv("release/v1.0-final/data/q1/method_summary.csv")
    q2 = _read_csv("release/v1.0-final/data/q2/method_layer_summary.csv")

    by_method = {row["method"]: row for row in q1}
    by_method_layer = {(row["method"], row["layer"]): row for row in q2}
    for method in ("BBB", "HBB", "HHB", "HHH"):
        rank = float(by_method_layer[(method, "z")]["effective_rank_mean"])
        assert f"{rank:.3f}" in readme
    assert f"{float(by_method['HHH']['accuracy_mean']):.4f}" in readme
    assert f"{float(by_method['HHB']['accuracy_mean']):.4f}" in readme
    assert f"{float(by_method['HHH']['standardized_reconstruction_mse_mean']):.6f}" in report


def test_results_summary_tables_match_frozen_q1_to_q6_values():
    summary = (ROOT / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")

    q1 = {row["method"]: row for row in _read_csv("release/v1.0-final/data/q1/method_summary.csv")}
    for method in ("BBB", "HBB", "HHB", "HHH", "Random"):
        row = q1[method]
        expected = (
            f"{float(row['accuracy_mean']):.5f} ± {float(row['accuracy_sd']):.5f} | "
            f"{float(row['system_reconstruction_mse_mean']):.6f} | "
            f"{float(row['standardized_reconstruction_mse_mean']):.6f}"
        )
        assert expected in summary

    q2 = {
        (row["method"], row["layer"]): row
        for row in _read_csv("release/v1.0-final/data/q2/method_layer_summary.csv")
    }
    for method in ("BBB", "HBB", "HHB", "HHH"):
        cells = []
        for layer in ("h1", "h2", "z"):
            row = q2[(method, layer)]
            cells.append(
                f"{float(row['effective_rank_mean']):.3f} / "
                f"{float(row['linear_probe_cv_accuracy_mean']):.4f}"
            )
        assert " | ".join(cells) in summary

    q3 = {
        (row["method"], row["noise_type"]): row
        for row in _read_csv("release/v1.0-final/data/q3/condition_summary.csv")
        if float(row["severity"]) == 0.4
    }
    for method in ("BBB", "HBB", "HHB", "HHH"):
        expected = " | ".join(
            f"{float(q3[(method, noise)]['accuracy_mean']):.5f}"
            for noise in ("gaussian", "salt_pepper", "pixel_masking")
        )
        assert expected in summary

    q4 = {
        row["layer"]: row
        for row in _read_csv("release/v1.0-final/data/q4/method_layer_update_summary.csv")
        if row["method"] == "HHH" and row["rule"] == "hebbian_effective"
    }
    for layer, digits in (("enc1", 4), ("enc2", 5), ("enc3", 7)):
        assert f"{float(q4[layer]['alignment_mean']):.{digits}f}" in summary

    performance = _read_csv("release/v1.0-final/data/q5q6/performance_summary.csv")
    representation = _read_csv("release/v1.0-final/data/q5q6/representation_summary.csv")
    perf = {(row["sweep"], row["case"], row["method"]): row for row in performance}
    rank = {
        (row["sweep"], row["case"], row["method"]): row
        for row in representation
        if row["layer"] == "z"
    }
    for sweep, cases in (
        ("dimension", ("L16", "L32", "L64", "L128")),
        ("architecture", ("early_heavy", "balanced", "late_heavy")),
    ):
        for case in cases:
            for method in ("BBB", "HBB", "HHB", "HHH"):
                expected = (
                    f"{float(perf[(sweep, case, method)]['accuracy_mean']):.4f} / "
                    f"{float(rank[(sweep, case, method)]['effective_rank_mean']):.2f}"
                )
                assert expected in summary


def test_governance_and_claim_boundaries_are_preserved():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in NARRATIVE_FILES)
    assert "CONFIRMATION FAILED" in combined
    assert "post-confirmation re-scoping" in combined
    assert "Stage 2D passed" not in combined
    assert "Stage 2D: PASS" not in combined
    assert "HHB is a Hebbian model" not in combined
    assert "HHB is a pure Hebbian" not in combined
    assert "HHB is a complete reconstruction repair" not in combined
    assert "HHB provides a complete reconstruction repair" not in combined
    assert "rank equals semantic information" not in combined
    assert "significant method×architecture" not in combined


def test_terminology_and_figure_disclosure_are_present():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in NARRATIVE_FILES)
    for term in (
        "standardized decoder",
        "system reconstruction",
        "effective rank",
        "frozen linear probe",
        "matched BP reconstruction direction",
    ):
        assert term.lower() in combined.lower()
    captions = (ROOT / "docs/final/FIGURE_CAPTIONS.md").read_text(encoding="utf-8")
    assert "zoomed accuracy axis" in captions
    schematic_source = (ROOT / "scripts/plot_final_figures.py").read_text(encoding="utf-8")
    assert '"Frozen linear\\nprobe"' in schematic_source


def test_readme_uses_selected_figures_only():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "method_design.png" in readme
    for figure in ("fig3_layerwise_representation.png", "fig1_performance.png", "fig4_update_mechanism.png"):
        assert figure in readme
    assert "fig6_dimension_architecture.png" not in readme
