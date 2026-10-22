import numpy as np
import torch

from evaluation.representation_health import (
    assess_layer_health,
    compute_layer_health,
    select_balanced_validation_ids,
)


THRESHOLDS = {
    "winner_density_abs_tolerance": 0.01,
    "winner_density_relative_tolerance": 0.10,
    "min_active_unit_ratio": 0.50,
    "min_nonzero_variance_ratio": 0.50,
    "min_effective_rank": 2.0,
    "min_normalized_effective_rank": 0.10,
    "min_winner_coverage_ratio": 0.50,
    "min_winner_entropy": 0.50,
    "max_winner_share": 0.25,
}


def test_balanced_validation_manifest_is_deterministic_and_class_balanced():
    labels = np.repeat(np.arange(10), 30)
    validation_ids = np.arange(labels.size)
    ids_a, labels_a = select_balanced_validation_ids(
        validation_ids, labels, samples_per_class=10, seed=17
    )
    ids_b, labels_b = select_balanced_validation_ids(
        validation_ids, labels, samples_per_class=10, seed=17
    )

    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(labels_a, labels_b)
    assert np.unique(ids_a).size == 100
    np.testing.assert_array_equal(
        np.bincount(labels_a, minlength=10),
        np.full(10, 10),
    )


def test_expected_topk_sparsity_is_not_mistaken_for_winner_collapse():
    observations = 128
    channels = 8
    activations = torch.zeros(observations, channels, 1, 1)
    for index in range(observations):
        activations[index, index % channels, 0, 0] = 2.0
        activations[index, (index + 1) % channels, 0, 0] = 1.0

    metrics, _, _ = compute_layer_health(
        activations,
        winner_fraction=0.25,
    )
    assessment = assess_layer_health(metrics, THRESHOLDS)

    assert metrics["expected_winner_density"] == 0.25
    assert metrics["observed_winner_density"] == 0.25
    assert metrics["winner_coverage_ratio"] == 1.0
    assert assessment["expected_wta_sparsity_consistent"]
    assert not assessment["pathological_winner_concentration"]
    assert assessment["gate_pass"]


def test_fixed_winners_fail_coverage_entropy_and_rank_despite_correct_density():
    observations = 128
    channels = 8
    activations = torch.zeros(observations, channels, 1, 1)
    signal = torch.linspace(1.0, 2.0, observations)
    activations[:, 0, 0, 0] = signal
    activations[:, 1, 0, 0] = signal * 0.5

    metrics, _, _ = compute_layer_health(
        activations,
        winner_fraction=0.25,
    )
    assessment = assess_layer_health(metrics, THRESHOLDS)

    assert metrics["observed_winner_density"] == 0.25
    assert metrics["winner_coverage_ratio"] == 0.25
    assert assessment["expected_wta_sparsity_consistent"]
    assert assessment["pathological_winner_concentration"]
    assert assessment["representation_degeneracy"]
    assert assessment["pathological_collapse"]
    assert not assessment["gate_pass"]


def test_health_metrics_are_bitwise_deterministic_and_do_not_mutate_input():
    generator = torch.Generator().manual_seed(91)
    activations = torch.rand(32, 16, 3, 3, generator=generator)
    before = activations.clone()

    first, first_counts, first_shares = compute_layer_health(
        activations, winner_fraction=0.10
    )
    second, second_counts, second_shares = compute_layer_health(
        activations, winner_fraction=0.10
    )

    assert first == second
    assert torch.equal(first_counts, second_counts)
    assert torch.equal(first_shares, second_shares)
    assert torch.equal(activations, before)
