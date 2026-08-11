from __future__ import annotations

import pytest

from evaluation.analyze_stage3_final_audit import (
    normalized_trapezoid_auc,
    q3_curve_auc_rows,
    standardized_decoder_config_audit,
    summarize_q3_auc,
)


def _curve_records():
    rows = []
    clean_accuracy = {"BBB": 0.9, "HHH": 0.8, "HHB": 0.85, "HBB": 0.88}
    for seed in range(5):
        for method, clean in clean_accuracy.items():
            common = {
                "seed": seed,
                "method": method,
                "macro_f1": clean,
                "system_reconstruction_mse_increase": 0.0,
                "standardized_reconstruction_mse_increase": 0.0,
                "accuracy_absolute_degradation": 0.0,
                "representation_cosine": 1.0,
                "prediction_js_divergence": 0.0,
            }
            rows.append(
                {
                    **common,
                    "noise_type": "clean",
                    "severity": 0.0,
                    "accuracy": clean,
                }
            )
            for noise_type in ("gaussian", "salt_pepper", "pixel_masking"):
                for severity in (0.1, 0.2, 0.3, 0.4):
                    degradation = severity * (1.0 if method != "HBB" else 0.5)
                    rows.append(
                        {
                            **common,
                            "noise_type": noise_type,
                            "severity": severity,
                            "accuracy": clean - degradation,
                            "macro_f1": clean - degradation,
                            "accuracy_absolute_degradation": degradation,
                            "system_reconstruction_mse_increase": degradation,
                            "standardized_reconstruction_mse_increase": degradation,
                            "representation_cosine": 1.0 - degradation,
                            "prediction_js_divergence": degradation,
                        }
                    )
    return rows


def test_normalized_trapezoid_auc_has_known_linear_result():
    assert normalized_trapezoid_auc([0.0, 0.2, 0.4], [1.0, 0.8, 0.6]) == pytest.approx(
        0.8
    )


def test_q3_curve_auc_uses_all_severities_and_keeps_paired_seeds():
    rows = q3_curve_auc_rows(_curve_records())
    assert len(rows) == 60
    indexed = {
        (row["seed"], row["method"], row["noise_type"]): row for row in rows
    }
    assert indexed[(0, "BBB", "gaussian")]["accuracy_degradation_auc"] == pytest.approx(
        0.2
    )
    assert indexed[(0, "HBB", "gaussian")]["accuracy_degradation_auc"] == pytest.approx(
        0.1
    )


def test_q3_auc_contrasts_bootstrap_seed_level_paired_differences():
    rows = q3_curve_auc_rows(_curve_records())
    summary, contrasts = summarize_q3_auc(rows)
    assert len(summary) == 12
    assert len(contrasts) == 63
    selected = next(
        row
        for row in contrasts
        if row["contrast"] == "HBB_minus_HHB"
        and row["noise_type"] == "gaussian"
        and row["metric"] == "accuracy_degradation_auc"
    )
    assert selected["mean_paired_difference"] == pytest.approx(-0.1)
    assert selected["ci_low"] == pytest.approx(-0.1)
    assert selected["ci_high"] == pytest.approx(-0.1)


def test_formal_standardized_decoder_configs_are_identical():
    audit = standardized_decoder_config_audit()
    assert audit["config_count"] == 135
    assert audit["all_configs_match"] is True
