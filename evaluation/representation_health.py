"""Deterministic representation-health metrics for the Stage 1 collapse gate."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def select_balanced_validation_ids(
    validation_ids: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and deterministically order an equal number of IDs per class."""

    validation_ids = np.asarray(validation_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if validation_ids.ndim != 1:
        raise ValueError("validation_ids must be one-dimensional")
    if labels.ndim != 1 or validation_ids.max(initial=-1) >= labels.size:
        raise ValueError("labels must cover every validation sample ID")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(10):
        candidates = validation_ids[labels[validation_ids] == class_id]
        if candidates.size < samples_per_class:
            raise ValueError(
                f"class {class_id} has {candidates.size} validation samples; "
                f"need {samples_per_class}"
            )
        selected.append(
            rng.choice(candidates, size=samples_per_class, replace=False)
        )
    sample_ids = np.concatenate(selected)
    rng.shuffle(sample_ids)
    return sample_ids, labels[sample_ids]


def _effective_rank(eigenvalues: torch.Tensor, epsilon: float) -> tuple[float, float]:
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    squared = eigenvalues.square().sum()
    if total <= epsilon or squared <= epsilon:
        return 0.0, 0.0
    rank = float((total.square() / squared).item())
    return rank, rank / eigenvalues.numel()


def compute_layer_health(
    activations: torch.Tensor,
    *,
    winner_fraction: float,
    activation_epsilon: float = 1e-12,
    variance_epsilon: float = 1e-12,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    """Measure activation diversity and WTA winner concentration.

    The WTA winner density ``k / C`` is a per-location quantity. Winner
    coverage is instead the fraction of channels that win at least once over
    all sample/spatial locations. A healthy large subset has expected coverage
    near one even when the per-location winner density is small.
    """

    if activations.ndim != 4:
        raise ValueError("activations must have shape N x C x H x W")
    if not 0 < winner_fraction <= 1:
        raise ValueError("winner_fraction must be in (0,1]")
    if activation_epsilon < 0 or variance_epsilon < 0:
        raise ValueError("epsilons must be non-negative")

    values = (
        activations.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .reshape(-1, activations.shape[1])
        .to(torch.float64)
    )
    if values.shape[0] == 0:
        raise ValueError("activations contain no observations")
    observations, channels = values.shape
    finite = bool(torch.isfinite(values).all().item())

    positive = values > activation_epsilon
    active_unit_ratio = float(positive.any(dim=0).to(torch.float64).mean().item())
    activation_density = float(positive.to(torch.float64).mean().item())
    per_unit_variance = values.var(dim=0, unbiased=False)
    nonzero_variance_ratio = float(
        (per_unit_variance > variance_epsilon).to(torch.float64).mean().item()
    )

    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(observations)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    effective_rank, normalized_effective_rank = _effective_rank(
        eigenvalues, variance_epsilon
    )

    winner_count = max(1, math.ceil(winner_fraction * channels))
    winner_values, winner_indices = torch.topk(values, k=winner_count, dim=1)
    positive_winners = winner_values > activation_epsilon
    winner_counts = torch.zeros(channels, dtype=torch.float64)
    winner_counts.scatter_add_(
        0,
        winner_indices.reshape(-1),
        positive_winners.reshape(-1).to(torch.float64),
    )
    total_positive_winners = winner_counts.sum()
    winner_shares = torch.zeros_like(winner_counts)
    if total_positive_winners > 0:
        winner_shares = winner_counts / total_positive_winners
        nonzero = winner_shares > 0
        entropy = -(
            winner_shares[nonzero] * winner_shares[nonzero].log()
        ).sum()
        normalized_entropy = float(
            (entropy / math.log(channels)).item() if channels > 1 else 1.0
        )
        max_winner_share = float(winner_shares.max().item())
    else:
        normalized_entropy = 0.0
        max_winner_share = 0.0

    expected_winner_density = winner_count / channels
    observed_winner_density = float(
        (total_positive_winners / (observations * channels)).item()
    )
    if winner_count == channels:
        expected_winner_coverage = 1.0
    else:
        expected_winner_coverage = float(
            -math.expm1(
                observations * math.log1p(-expected_winner_density)
            )
        )
    winner_coverage_ratio = float(
        (winner_counts > 0).to(torch.float64).mean().item()
    )

    metrics: dict[str, float | int] = {
        "num_samples": int(activations.shape[0]),
        "num_observations": int(observations),
        "channels": int(channels),
        "spatial_height": int(activations.shape[2]),
        "spatial_width": int(activations.shape[3]),
        "finite": int(finite),
        "activation_density": activation_density,
        "activation_sparsity": 1.0 - activation_density,
        "active_unit_ratio": active_unit_ratio,
        "dead_unit_ratio": 1.0 - active_unit_ratio,
        "mean_unit_variance": float(per_unit_variance.mean().item()),
        "median_unit_variance": float(per_unit_variance.median().item()),
        "nonzero_variance_ratio": nonzero_variance_ratio,
        "effective_rank": effective_rank,
        "normalized_effective_rank": normalized_effective_rank,
        "winner_count_per_location": int(winner_count),
        "expected_winner_density": expected_winner_density,
        "observed_winner_density": observed_winner_density,
        "expected_wta_sparsity": 1.0 - expected_winner_density,
        "observed_wta_sparsity": 1.0 - observed_winner_density,
        "expected_winner_coverage": expected_winner_coverage,
        "winner_coverage_ratio": winner_coverage_ratio,
        "winner_dead_unit_ratio": 1.0 - winner_coverage_ratio,
        "winner_entropy": normalized_entropy,
        "max_winner_share": max_winner_share,
    }
    return metrics, winner_counts, winner_shares


def assess_layer_health(
    metrics: dict[str, float | int],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Apply preregistered Stage 1 checks and classify collapse mechanisms."""

    density_tolerance = max(
        thresholds["winner_density_abs_tolerance"],
        thresholds["winner_density_relative_tolerance"]
        * float(metrics["expected_winner_density"]),
    )
    checks = {
        "finite": bool(metrics["finite"]),
        "winner_density_matches_topk": abs(
            float(metrics["observed_winner_density"])
            - float(metrics["expected_winner_density"])
        )
        <= density_tolerance,
        "active_unit_ratio": float(metrics["active_unit_ratio"])
        >= thresholds["min_active_unit_ratio"],
        "nonzero_variance_ratio": float(metrics["nonzero_variance_ratio"])
        >= thresholds["min_nonzero_variance_ratio"],
        "effective_rank": float(metrics["effective_rank"])
        >= thresholds["min_effective_rank"],
        "normalized_effective_rank": float(metrics["normalized_effective_rank"])
        >= thresholds["min_normalized_effective_rank"],
        "winner_coverage_ratio": float(metrics["winner_coverage_ratio"])
        >= thresholds["min_winner_coverage_ratio"],
        "winner_entropy": float(metrics["winner_entropy"])
        >= thresholds["min_winner_entropy"],
        "max_winner_share": float(metrics["max_winner_share"])
        <= thresholds["max_winner_share"],
    }
    winner_concentration = not all(
        checks[key]
        for key in (
            "winner_coverage_ratio",
            "winner_entropy",
            "max_winner_share",
        )
    )
    representation_degeneracy = not all(
        checks[key]
        for key in (
            "active_unit_ratio",
            "nonzero_variance_ratio",
            "effective_rank",
            "normalized_effective_rank",
        )
    )
    return {
        "checks": checks,
        "expected_wta_sparsity_consistent": checks[
            "winner_density_matches_topk"
        ],
        "pathological_winner_concentration": winner_concentration,
        "representation_degeneracy": representation_degeneracy,
        "pathological_collapse": winner_concentration
        and representation_degeneracy,
        "gate_pass": all(checks.values()),
        "failed_checks": [key for key, passed in checks.items() if not passed],
    }
