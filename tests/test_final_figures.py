from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "final"
RELEASE = ROOT / "release" / "v1.0-final"
HERO_STEMS = [
    "fig1_performance",
    "fig2_prefix_value_training_cost",
    "fig3_layerwise_representation",
    "fig4_update_mechanism",
    "fig5_robustness",
    "fig6_dimension_architecture",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_final_figure_manifest_is_artifact_only_and_hash_complete() -> None:
    manifest = json.loads((FIGURES / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["builder"]["path"] == "scripts/plot_final_figures.py"
    assert manifest["builder"]["sha256"] == _sha256(ROOT / manifest["builder"]["path"])
    assert manifest["formal_seeds"] == [0, 1, 2, 3, 4]
    assert manifest["statistical_unit"] == "paired seed"
    assert manifest["bootstrap"] == {"confidence_level": 0.95, "resamples": 10000, "seed": 2026}
    assert set(manifest["figures"]) == set(HERO_STEMS) | {"method_design"}
    assert sum(bool(record.get("formal_hero_figure", True)) for record in manifest["figures"].values()) == 6
    assert not any(
        value
        for key, value in manifest["scientific_actions"].items()
        if key != "test_access_increment"
    )
    assert manifest["scientific_actions"]["test_access_increment"] == 0

    release_manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    release_hashes = {record["bundle_path"]: record["sha256"] for record in release_manifest["bundle_files"]}
    for figure in manifest["figures"].values():
        for source, digest in figure["source_hashes"].items():
            assert source.startswith("data/")
            assert "results/" not in source and "recovery" not in source.lower()
            assert digest == release_hashes[source]
            assert _sha256(RELEASE / source) == digest
    for relative, digest in manifest["generated_file_hashes"].items():
        assert _sha256(FIGURES / relative) == digest


def test_final_hero_outputs_are_high_resolution_png_and_pdf_pairs() -> None:
    for stem in HERO_STEMS + ["method_design"]:
        png = FIGURES / f"{stem}.png"
        pdf = FIGURES / f"{stem}.pdf"
        assert png.is_file() and pdf.is_file()
        with Image.open(png) as image:
            assert image.width >= 2400
            assert image.height >= 1000
            dpi = image.info.get("dpi")
            assert dpi is not None and min(dpi) >= 299
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert pdf.stat().st_size > 10_000


def test_plotted_values_preserve_formal_seeds_and_late_heavy_seed4() -> None:
    values = FIGURES / "plotted_values"
    for path in values.glob("*per_seed*.csv"):
        rows = _csv(path)
        assert rows
        if "seed" in rows[0]:
            assert {int(row["seed"]) for row in rows} == {0, 1, 2, 3, 4}
    trajectory = _csv(values / "fig2_samples_seen_per_seed.csv")
    assert len(trajectory) == 200
    assert {
        (row["method"], int(row["seed"])) for row in trajectory
    } == {(method, seed) for method in ("BBB", "HBB", "HHB", "HHH") for seed in range(5)}
    performance = _csv(values / "fig6_per_seed_performance.csv")
    late_seed4 = [
        row
        for row in performance
        if row["sweep"] == "architecture" and row["case"] == "late_heavy" and row["seed"] == "4"
    ]
    assert {row["method"] for row in late_seed4} == {"BBB", "HBB", "HHB", "HHH"}


def test_plotted_value_rows_trace_to_compact_tables() -> None:
    values = FIGURES / "plotted_values"

    def rows(relative: str) -> list[dict[str, str]]:
        return _csv(RELEASE / "data" / relative)

    def frozen_set(relative: str) -> set[tuple[tuple[str, str], ...]]:
        return {tuple(sorted(row.items())) for row in rows(relative)}

    direct_tables = {
        "fig1_per_seed.csv": "q1/per_seed_complete.csv",
        "fig1_summary.csv": "q1/method_summary.csv",
        "fig2_paired_summary.csv": "q1/paired_contrasts.csv",
        "fig2_samples_seen_per_seed.csv": "audit/q1_samples_seen_curve_per_seed.csv",
        "fig2_samples_seen_summary.csv": "audit/q1_samples_seen_curve_summary.csv",
        "fig2_training_cost_per_seed.csv": "q1/per_seed_complete.csv",
        "fig3_per_seed.csv": "q2/per_seed_layer_metrics.csv",
        "fig3_summary.csv": "q2/method_layer_summary.csv",
        "fig4_per_seed.csv": "q4/per_seed_layer_update_metrics.csv",
        "fig4_summary.csv": "q4/method_layer_update_summary.csv",
        "fig6_per_seed_performance.csv": "q5q6/performance_per_seed.csv",
        "fig6_summary_performance.csv": "q5q6/performance_summary.csv",
        "fig6_per_seed_z_rank.csv": "q5q6/representation_per_seed_layer.csv",
        "fig6_summary_z_rank.csv": "q5q6/representation_summary.csv",
    }
    for plotted_name, source_relative in direct_tables.items():
        accepted = frozen_set(source_relative)
        assert all(tuple(sorted(row.items())) in accepted for row in _csv(values / plotted_name))

    q1 = {(row["method"], int(row["seed"])): row for row in rows("q1/per_seed_complete.csv")}
    for row in _csv(values / "fig2_paired_seed_effects.csv"):
        expected = float(q1[(row["left"], int(row["seed"]))]["accuracy"]) - float(
            q1[(row["right"], int(row["seed"]))]["accuracy"]
        )
        assert float(row["accuracy_difference"]) == expected

    for plotted_name, source_relative in (
        ("fig5_per_seed_curves.csv", "q3/per_seed_condition_metrics.csv"),
        ("fig5_summary_curves.csv", "q3/condition_summary.csv"),
    ):
        accepted = frozen_set(source_relative)
        for plotted in _csv(values / plotted_name):
            restored = dict(plotted)
            source_condition = restored.pop("source_condition")
            if source_condition == "clean":
                restored["noise_type"] = "clean"
            assert tuple(sorted(restored.items())) in accepted
