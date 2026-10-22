from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.deterministic_noise import apply_deterministic_noise
from evaluation.run_stage3_q3_noise import (
    _cosine_per_sample,
    _js_divergence,
    degradation_rows,
)


@pytest.mark.parametrize(
    "noise_type", ["gaussian", "salt_pepper", "pixel_masking"]
)
def test_same_noise_key_is_bitwise_deterministic_and_bounded(
    noise_type: str,
) -> None:
    images = torch.linspace(0, 1, 2 * 28 * 28).reshape(2, 1, 28, 28)
    ids = torch.tensor([7, 99])
    first = apply_deterministic_noise(
        images, ids, noise_type=noise_type, severity=0.3
    )
    second = apply_deterministic_noise(
        images, ids, noise_type=noise_type, severity=0.3
    )
    assert torch.equal(first, second)
    assert float(first.min()) >= 0
    assert float(first.max()) <= 1


def test_noise_is_sample_id_keyed_and_order_invariant() -> None:
    images = torch.stack([torch.full((1, 4, 4), 0.2), torch.full((1, 4, 4), 0.8)])
    ids = torch.tensor([10, 20])
    forward = apply_deterministic_noise(
        images, ids, noise_type="gaussian", severity=0.2
    )
    reverse = apply_deterministic_noise(
        images.flip(0), ids.flip(0), noise_type="gaussian", severity=0.2
    )
    assert torch.equal(forward, reverse.flip(0))


def test_zero_severity_is_exact_and_nonmutating() -> None:
    images = torch.rand(3, 1, 4, 4)
    before = images.clone()
    output = apply_deterministic_noise(
        images,
        torch.arange(3),
        noise_type="salt_pepper",
        severity=0.0,
    )
    assert torch.equal(output, images)
    assert torch.equal(images, before)
    assert output.data_ptr() != images.data_ptr()


def test_noise_types_have_expected_extreme_behavior() -> None:
    images = torch.full((2, 1, 4, 4), 0.5)
    ids = torch.tensor([1, 2])
    masked = apply_deterministic_noise(
        images, ids, noise_type="pixel_masking", severity=1.0
    )
    salt_pepper = apply_deterministic_noise(
        images, ids, noise_type="salt_pepper", severity=1.0
    )
    assert torch.count_nonzero(masked) == 0
    assert set(torch.unique(salt_pepper).tolist()).issubset({0.0, 1.0})


def test_representation_cosine_and_js_have_known_boundaries() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.equal(_cosine_per_sample(values, values), torch.ones(2))
    probabilities = torch.tensor([[0.8, 0.2], [0.5, 0.5]])
    assert torch.allclose(
        _js_divergence(probabilities, probabilities), torch.zeros(2)
    )
    swapped = probabilities.flip(1)
    assert torch.all(_js_divergence(probabilities, swapped) >= 0)


def test_degradation_uses_within_seed_method_clean_baseline() -> None:
    common = {
        "seed": 0,
        "method": "BBB",
        "macro_f1": 0.8,
        "system_reconstruction_mse": 0.1,
        "standardized_reconstruction_mse": 0.1,
    }
    rows = [
        {
            **common,
            "noise_type": "clean",
            "severity": 0.0,
            "accuracy": 0.9,
        },
        {
            **common,
            "noise_type": "gaussian",
            "severity": 0.2,
            "accuracy": 0.7,
            "macro_f1": 0.6,
            "system_reconstruction_mse": 0.2,
            "standardized_reconstruction_mse": 0.3,
        },
    ]
    result = degradation_rows(rows)
    assert result[1]["accuracy_absolute_degradation"] == pytest.approx(0.2)
    assert result[1]["accuracy_relative_degradation"] == pytest.approx(2 / 9)
    assert result[1]["system_reconstruction_mse_increase"] == pytest.approx(0.1)
    assert result[1]["standardized_reconstruction_mse_increase"] == pytest.approx(
        0.2
    )
