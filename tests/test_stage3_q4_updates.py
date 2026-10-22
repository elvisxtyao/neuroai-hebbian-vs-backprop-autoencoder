from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from evaluation.run_stage3_q4_updates import (
    analyze_bp_final,
    summarize_rows,
)
from evaluation.run_q4_tooling import analysis_model_from_run_config
from models import ConvAutoencoder


ROOT = Path(__file__).resolve().parents[1]


def test_q4_protocol_freezes_shared_snapshots_and_bp_boundaries() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/experiments/stage3_q4_updates_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["data"]["batch_count"] == 50
    assert protocol["analysis"]["optimizer_steps"] == 0
    assert protocol["shared_hebbian_snapshots"]["enc1"]["shared_methods"] == [
        "full_hebbian",
        "hybrid_hhb",
        "hybrid_hbb",
    ]
    assert protocol["bp_final_layers"] == {
        "full_bp": ["enc1", "enc2", "enc3"],
        "hybrid_hhb": ["enc3"],
        "hybrid_hbb": ["enc2", "enc3"],
    }


def test_final_bp_analysis_is_nonmutating(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = ConvAutoencoder(latent_dim=4, seed=3)
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    ids = torch.arange(4)
    loader = DataLoader(TensorDataset(images, labels, ids), batch_size=2)
    batch_ids = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    summary = analyze_bp_final(
        model=model,
        method="full_bp",
        layer="enc3",
        seed=3,
        batch_ids=batch_ids,
        loader=loader,
        output_dir=tmp_path,
        epsilon=1e-12,
    )
    assert summary["batch_count"] == 2
    assert summary["analysis_optimizer_steps"] == 0
    assert summary["model_hash_before"] == summary["model_hash_after"]
    assert all(
        torch.equal(before[key], model.state_dict()[key]) for key in before
    )


def test_final_bp_analysis_supports_asymmetric_encoder_widths(
    tmp_path: Path,
) -> None:
    model = ConvAutoencoder(
        latent_dim=4,
        encoder_channels=[4, 7],
        seed=3,
    )
    images = torch.rand(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    ids = torch.arange(4)
    loader = DataLoader(TensorDataset(images, labels, ids), batch_size=2)
    summary = analyze_bp_final(
        model=model,
        method="hybrid_hbb",
        layer="enc2",
        seed=3,
        batch_ids=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        loader=loader,
        output_dir=tmp_path,
        epsilon=1e-12,
    )
    assert summary["batch_count"] == 2
    assert summary["analysis_optimizer_steps"] == 0


def test_pair_analysis_factory_preserves_asymmetric_source_widths() -> None:
    run_config = {
        "model": {
            "latent_dim": 11,
            "encoder_channels": [4, 7],
        }
    }
    model = analysis_model_from_run_config(run_config, seed=3)
    features = model.encode(
        torch.zeros(2, 1, 28, 28), return_all_layers=True
    )
    assert tuple(features["h1"].shape[1:]) == (4, 14, 14)
    assert tuple(features["h2"].shape[1:]) == (7, 7, 7)
    assert tuple(features["z"].shape[1:]) == (11, 1, 1)


def test_q6_update_protocols_freeze_case_paths_and_disable_core_correlations():
    expected = {
        "stage3_q6_update_early_heavy_v1.yaml": "early_heavy",
        "stage3_q6_update_late_heavy_v1.yaml": "late_heavy",
    }
    for filename, case in expected.items():
        protocol = yaml.safe_load(
            (
                ROOT / "configs" / "experiments" / filename
            ).read_text(encoding="utf-8")
        )
        assert f"/architecture/{case}" in protocol[
            "source_results_root"
        ].replace("\\", "/")
        assert protocol["seeds"] == [0, 1, 2, 3, 4]
        assert protocol["data"]["batch_count"] == 50
        assert protocol["analysis"]["exploratory_correlations"] is False


def test_update_summary_uses_five_seed_dispersion() -> None:
    rows = []
    for seed in range(5):
        rows.append(
            {
                "seed": seed,
                "method": "HHH",
                "layer": "enc1",
                "rule": "hebbian_effective",
                "alignment": 0.1 + seed * 0.01,
                "norm_ratio": 1.0,
                "alpha_star": 2.0,
                "scale_matched_bias": 0.9,
                "update_snr_linear": 0.2,
                "update_snr_db": -7.0,
                "matched_bp_snr_linear": 0.3,
            }
        )
    summary = summarize_rows(rows)
    assert len(summary) == 1
    assert summary[0]["alignment_mean"] == pytest.approx(0.12)
    assert summary[0]["alignment_sd"] > 0
