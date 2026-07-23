import pytest
import torch

from evaluation.effective_rank_audit import (
    apply_topk_wta,
    class_center,
    epsilon_sensitivity,
    l2_normalize_samples,
    representation_matrix,
    spectrum_metrics,
)


def test_axis_views_have_explicit_expected_shapes_and_order():
    activations = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    channel, channel_metadata = representation_matrix(
        activations, view="channel_health"
    )
    sample, sample_metadata = representation_matrix(
        activations, view="sample_flat"
    )

    assert channel.shape == (8, 3)
    assert sample.shape == (2, 12)
    assert channel_metadata["observation_axis"] == "sample_x_spatial_location"
    assert channel_metadata["feature_axis"] == "channel"
    assert sample_metadata["observation_axis"] == "sample"
    assert sample_metadata["feature_axis"] == "channel_x_height_x_width"
    assert torch.equal(channel[0], activations[0, :, 0, 0])
    assert torch.equal(sample[0], activations[0].flatten())


def test_z_channel_and_sample_views_are_identical_at_one_by_one():
    generator = torch.Generator().manual_seed(7)
    z = torch.rand(11, 5, 1, 1, generator=generator)
    channel, _ = representation_matrix(z, view="channel_health")
    sample, _ = representation_matrix(z, view="sample_flat")
    assert torch.equal(channel, sample)


def test_covariance_rank_uses_feature_axis_and_dataset_centering():
    signal = torch.arange(1, 9, dtype=torch.float64)
    matrix = torch.stack((signal, 2 * signal, -3 * signal), dim=1)
    result = spectrum_metrics(matrix)

    assert result.conceptual_covariance_shape == (3, 3)
    assert result.dataset_centered
    assert result.participation_ratio == pytest.approx(1.0)
    assert result.stable_rank == pytest.approx(1.0)
    assert result.rank_ratio == pytest.approx(1 / 3)
    assert result.numerical_rank == 1


def test_dual_gram_spectrum_matches_feature_covariance_nonzero_spectrum():
    generator = torch.Generator().manual_seed(19)
    wide = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    result = spectrum_metrics(wide)
    centered = wide - wide.mean(dim=0, keepdim=True)
    direct = torch.linalg.eigvalsh(centered.T @ centered / 5).clamp_min(0)

    assert result.backend == "dual_gram_equivalent_nonzero_spectrum"
    assert result.conceptual_covariance_shape == (9, 9)
    assert torch.allclose(
        result.eigenvalues.flip(0),
        direct,
        atol=1e-12,
        rtol=1e-10,
    )


def test_wta_centering_l2_and_class_centering_are_non_mutating():
    matrix = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float64,
    )
    before = matrix.clone()
    post, mask = apply_topk_wta(matrix, winner_fraction=0.5)
    normalized = l2_normalize_samples(post)
    centered = class_center(post, torch.tensor([0, 0]))

    assert mask.sum(dim=1).tolist() == [2, 2]
    assert torch.equal(post, torch.tensor([[4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]], dtype=torch.float64))
    assert torch.allclose(normalized.norm(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(centered.mean(dim=0), torch.zeros(4, dtype=torch.float64))
    assert torch.equal(matrix, before)


def test_epsilon_audit_detects_non_dominant_and_dominant_cases():
    eigenvalues = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64)
    rows, valid = epsilon_sensitivity(
        eigenvalues,
        relative_cutoffs=[0.0, 1e-12, 1e-6],
        formula_epsilons=[0.0, 1e-12],
    )
    assert valid
    assert not any(row["epsilon_dominates"] for row in rows)

    _, valid_with_huge_epsilon = epsilon_sensitivity(
        eigenvalues,
        relative_cutoffs=[0.0],
        formula_epsilons=[100.0],
    )
    assert not valid_with_huge_epsilon


def test_constant_l2_directions_have_numerical_zero_centered_variance():
    scales = torch.arange(1, 9, dtype=torch.float64).unsqueeze(1)
    direction = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
    normalized = l2_normalize_samples(scales * direction)
    result = spectrum_metrics(normalized)

    assert result.trace < 1e-12
    assert result.squared_trace < 1e-12
