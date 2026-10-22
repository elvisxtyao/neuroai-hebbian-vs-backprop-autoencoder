import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.run_q4_tooling import _source_run_dir
from evaluation.update_analysis import (
    bp_raw_negative_gradient,
    cosine_alignment,
    effective_hebbian_delta,
    norm_ratio,
    prepare_fixed_batch_manifest,
    raw_relative_difference,
    scale_matched_metrics,
    snapshot_integrity_gate,
    update_snr,
)
from learning_rules.hebbian import CompetitiveOjaConv2d
from models import ConvAutoencoder
from utils.reproducibility import state_dict_checksum


EPSILON = 1e-12


def test_cosine_alignment_known_directions_and_zero_vector():
    first = torch.tensor([1.0, 0.0])
    assert cosine_alignment(first, first, epsilon=EPSILON) == pytest.approx(1.0)
    assert cosine_alignment(first, -first, epsilon=EPSILON) == pytest.approx(-1.0)
    assert cosine_alignment(
        first, torch.tensor([0.0, 1.0]), epsilon=EPSILON
    ) == pytest.approx(0.0)
    assert cosine_alignment(first, torch.zeros(2), epsilon=EPSILON) == 0.0


def test_norm_ratio_and_raw_relative_difference():
    reference = torch.tensor([1.0, 2.0])
    candidate = 2 * reference
    assert norm_ratio(candidate, reference, epsilon=EPSILON) == pytest.approx(2.0)
    assert raw_relative_difference(
        candidate, reference, epsilon=EPSILON
    ) == pytest.approx(1.0)


def test_scale_matching_removes_pure_scale_difference():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    candidate = 2 * reference
    metrics = scale_matched_metrics(candidate, reference, epsilon=EPSILON)
    assert metrics["alpha_star"] == pytest.approx(0.5)
    assert metrics["scale_matched_relative_bias"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mean_update_alignment"] == pytest.approx(1.0)


def test_snr_constant_and_zero_updates_have_controlled_boundaries():
    constant = torch.ones(5, 3)
    constant_metrics = update_snr(constant, epsilon=EPSILON)
    assert constant_metrics["noise_power"] == 0.0
    assert constant_metrics["snr_linear"] == pytest.approx(3 / EPSILON)

    zeros = torch.zeros(5, 3)
    zero_metrics = update_snr(zeros, epsilon=EPSILON)
    assert zero_metrics["signal_power"] == 0.0
    assert zero_metrics["noise_power"] == 0.0
    assert zero_metrics["snr_linear"] == 0.0
    assert zero_metrics["snr_db"] == float("-inf")


def test_effective_delta_matches_apply_on_clone():
    layer = torch.nn.Conv2d(1, 3, kernel_size=2, bias=False)
    generator = torch.Generator().manual_seed(5)
    raw = torch.randn(layer.weight.shape, generator=generator)
    rule = CompetitiveOjaConv2d(
        learning_rate=0.01,
        winner_fraction=0.5,
        normalization_epsilon=1e-8,
    )
    before = layer.weight.detach().clone()
    expected = effective_hebbian_delta(
        before,
        raw,
        learning_rate=rule.learning_rate,
        normalization_epsilon=rule.normalization_epsilon,
    )
    rule.apply_local_update(layer, raw)
    assert torch.allclose(layer.weight - before, expected, atol=1e-7, rtol=1e-6)


def test_bp_raw_direction_does_not_mutate_model_or_populate_gradients():
    model = ConvAutoencoder(latent_dim=4, seed=3)
    images = torch.rand(2, 1, 28, 28)
    checksum_before = state_dict_checksum(model)
    expected_loss = torch.nn.functional.mse_loss(model(images), images)
    (expected_gradient,) = torch.autograd.grad(
        expected_loss, model.encoder.enc2.weight
    )
    for parameter in model.parameters():
        parameter.grad = None
    direction, loss = bp_raw_negative_gradient(model, images, layer_name="enc2")

    assert direction.shape == model.encoder.enc2.weight.shape
    assert torch.isfinite(direction).all()
    assert loss == pytest.approx(float(expected_loss.detach().item()))
    assert torch.equal(direction, -expected_gradient)
    assert state_dict_checksum(model) == checksum_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fixed_batch_manifest_is_deterministic_unique_and_verified(tmp_path: Path):
    split_path = tmp_path / "split.npz"
    output_path = tmp_path / "batches.npz"
    np.savez_compressed(split_path, train_indices=np.arange(100, dtype=np.int64))

    prepare_fixed_batch_manifest(
        split_manifest=split_path,
        output_path=output_path,
        batch_count=5,
        batch_size=10,
        seed=31415,
        version="test-v1",
    )
    first = np.load(output_path, allow_pickle=False)["batch_ids"].copy()
    prepare_fixed_batch_manifest(
        split_manifest=split_path,
        output_path=output_path,
        batch_count=5,
        batch_size=10,
        seed=31415,
        version="test-v1",
    )
    second = np.load(output_path, allow_pickle=False)["batch_ids"]

    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 50


def test_snapshot_integrity_gate_distinguishes_greedy_freezing():
    model = ConvAutoencoder(latent_dim=4, seed=7)
    initial = {
        key: value.detach().clone() for key, value in model.encoder.state_dict().items()
    }
    enc1_end = {key: value.clone() for key, value in initial.items()}
    enc1_end["enc1.weight"] += 1
    enc2_end = {key: value.clone() for key, value in enc1_end.items()}
    enc2_end["enc2.weight"] += 2
    enc3_end = {key: value.clone() for key, value in enc2_end.items()}
    enc3_end["enc3.weight"] += 3

    gate = snapshot_integrity_gate(
        {
            "enc1_end": enc1_end,
            "enc2_end": enc2_end,
            "enc3_end": enc3_end,
        },
        initial,
    )
    assert gate["gate_pass"]

    broken = {key: value.clone() for key, value in enc3_end.items()}
    broken["enc1.weight"] += 1
    failed = snapshot_integrity_gate(
        {
            "enc1_end": enc1_end,
            "enc2_end": enc2_end,
            "enc3_end": broken,
        },
        initial,
    )
    assert not failed["gate_pass"]


def test_q4_source_run_can_be_resolved_from_frozen_candidate_decision(tmp_path):
    run_dir = tmp_path / "candidate_run"
    decision_path = tmp_path / "selection_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "trial_id": "frozen_candidate",
                        "run_dir": str(run_dir),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = _source_run_dir(
        {
            "selection_decision": str(decision_path),
            "trial_id": "frozen_candidate",
        }
    )

    assert resolved == run_dir.resolve()
