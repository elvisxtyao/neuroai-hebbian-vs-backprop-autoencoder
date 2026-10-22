"""Numerically explicit rank metrics for the Stage 1C representation audit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectrumResult:
    eigenvalues: torch.Tensor
    singular_values: torch.Tensor
    observation_count: int
    feature_count: int
    conceptual_covariance_shape: tuple[int, int]
    backend: str
    dataset_centered: bool
    participation_ratio: float
    stable_rank: float
    rank_ratio: float
    max_rank_ratio: float
    numerical_rank: int
    numerical_rank_tolerance: float
    trace: float
    squared_trace: float


def representation_matrix(
    activations: torch.Tensor,
    *,
    view: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return a float64 observations-by-features matrix with explicit axes."""

    if activations.ndim != 4:
        raise ValueError("activations must have shape N x C x H x W")
    if view not in {"channel_health", "sample_flat"}:
        raise ValueError("view must be channel_health or sample_flat")
    values = activations.detach().cpu().to(torch.float64)
    samples, channels, height, width = values.shape
    if samples == 0:
        raise ValueError("activations contain no samples")
    if view == "channel_health":
        matrix = values.permute(0, 2, 3, 1).reshape(-1, channels)
        observation_axis = "sample_x_spatial_location"
        feature_axis = "channel"
    else:
        matrix = values.reshape(samples, -1)
        observation_axis = "sample"
        feature_axis = "channel_x_height_x_width"
    metadata = {
        "source_shape": [samples, channels, height, width],
        "matrix_shape": list(matrix.shape),
        "view": view,
        "observation_axis": observation_axis,
        "feature_axis": feature_axis,
        "spatial_locations_per_sample": height * width,
    }
    return matrix, metadata


def apply_topk_wta(
    matrix: torch.Tensor,
    *,
    winner_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply deterministic per-sample top-k WTA to an N x D matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must have shape observations x features")
    if not 0 < winner_fraction <= 1:
        raise ValueError("winner_fraction must be in (0,1]")
    feature_count = matrix.shape[1]
    winner_count = max(1, math.ceil(winner_fraction * feature_count))
    indices = torch.topk(matrix, k=winner_count, dim=1).indices
    mask = torch.zeros_like(matrix, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return matrix * mask, mask


def dataset_center(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return matrix - matrix.mean(dim=0, keepdim=True)


def l2_normalize_samples(
    matrix: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    norms = matrix.norm(dim=1, keepdim=True)
    return matrix / norms.clamp_min(epsilon)


def class_center(matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    labels = torch.as_tensor(labels).detach().cpu().long()
    if labels.ndim != 1 or labels.numel() != matrix.shape[0]:
        raise ValueError("labels must have one entry per observation")
    centered = matrix.clone()
    for class_id in torch.unique(labels, sorted=True):
        selected = labels == class_id
        centered[selected] -= matrix[selected].mean(dim=0, keepdim=True)
    return centered


def _covariance_eigenvalues(
    centered: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    observations, features = centered.shape
    denominator = float(observations)
    if features <= observations:
        covariance = centered.T @ centered / denominator
        eigenvalues = torch.linalg.eigvalsh(covariance)
        backend = "feature_covariance"
    else:
        dual_gram = centered @ centered.T / denominator
        nonzero = torch.linalg.eigvalsh(dual_gram)
        eigenvalues = torch.cat(
            [
                torch.zeros(
                    features - observations,
                    dtype=nonzero.dtype,
                    device=nonzero.device,
                ),
                nonzero,
            ]
        )
        backend = "dual_gram_equivalent_nonzero_spectrum"
    return eigenvalues.clamp_min(0).cpu(), backend


def spectrum_metrics(
    matrix: torch.Tensor,
    *,
    center: bool = True,
) -> SpectrumResult:
    """Compute the exact centered covariance spectrum and rank metrics."""

    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("matrix must be a non-empty observations x features array")
    values = matrix.detach().cpu().to(torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("matrix contains NaN or Inf")
    centered = dataset_center(values) if center else values
    observations, features = centered.shape
    eigenvalues, backend = _covariance_eigenvalues(centered)
    total = eigenvalues.sum()
    squared = eigenvalues.square().sum()
    largest = eigenvalues.max()
    if total <= 0 or squared <= 0 or largest <= 0:
        participation_ratio = 0.0
        stable_rank = 0.0
    else:
        participation_ratio = float((total.square() / squared).item())
        stable_rank = float((total / largest).item())
    maximum_rank = min(max(observations - int(center), 0), features)
    machine_tolerance = (
        float(largest.item())
        * max(observations, features)
        * torch.finfo(torch.float64).eps
    )
    numerical_rank = int((eigenvalues > machine_tolerance).sum().item())
    singular_values = torch.sqrt(eigenvalues * float(observations))
    singular_values = singular_values[-min(observations, features) :]
    return SpectrumResult(
        eigenvalues=eigenvalues.flip(0),
        singular_values=singular_values.flip(0),
        observation_count=observations,
        feature_count=features,
        conceptual_covariance_shape=(features, features),
        backend=backend,
        dataset_centered=center,
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        rank_ratio=participation_ratio / features,
        max_rank_ratio=(
            0.0 if maximum_rank == 0 else participation_ratio / maximum_rank
        ),
        numerical_rank=numerical_rank,
        numerical_rank_tolerance=machine_tolerance,
        trace=float(total.item()),
        squared_trace=float(squared.item()),
    )


def participation_rank_without_spectrum(
    matrix: torch.Tensor,
    *,
    center: bool = True,
) -> float:
    """Compute covariance participation rank without an eigendecomposition."""

    values = matrix.detach().cpu().to(torch.float64)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("matrix must be non-empty and two-dimensional")
    values = dataset_center(values) if center else values
    observations, features = values.shape
    if features <= observations:
        second_moment = values.T @ values / float(observations)
    else:
        second_moment = values @ values.T / float(observations)
    total = values.square().sum() / float(observations)
    squared = second_moment.square().sum()
    if total <= 0 or squared <= 0:
        return 0.0
    return float((total.square() / squared).item())


def class_rank_metrics(
    matrix: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    """Return per-class, pooled-within and between-class covariance ranks."""

    labels = torch.as_tensor(labels).detach().cpu().long()
    values = matrix.detach().cpu().to(torch.float64)
    if labels.ndim != 1 or labels.numel() != values.shape[0]:
        raise ValueError("labels must align with matrix observations")
    class_rows: list[dict[str, float | int]] = []
    residuals = torch.empty_like(values)
    centroids: list[torch.Tensor] = []
    class_ids = torch.unique(labels, sorted=True)
    for class_id in class_ids:
        selected = labels == class_id
        class_values = values[selected]
        centroid = class_values.mean(dim=0)
        centroids.append(centroid)
        residuals[selected] = class_values - centroid
        class_rows.append(
            {
                "class_id": int(class_id.item()),
                "sample_count": int(selected.sum().item()),
                "participation_ratio": participation_rank_without_spectrum(
                    class_values
                ),
            }
        )
    within_rank = participation_rank_without_spectrum(residuals, center=False)
    centroid_matrix = torch.stack(centroids)
    between_rank = participation_rank_without_spectrum(centroid_matrix)
    return class_rows, {
        "within_class_covariance_participation_rank": within_rank,
        "between_class_covariance_participation_rank": between_rank,
    }


def epsilon_sensitivity(
    eigenvalues: torch.Tensor,
    *,
    relative_cutoffs: list[float],
    formula_epsilons: list[float],
) -> tuple[list[dict[str, float | bool | str]], bool]:
    """Audit cutoff sensitivity and whether Stage 1's epsilon gates dominate."""

    values = eigenvalues.detach().cpu().to(torch.float64).clamp_min(0)
    largest = float(values.max().item())
    rows: list[dict[str, float | bool | str]] = []
    cutoff_ranks: list[float] = []
    for cutoff in relative_cutoffs:
        if cutoff < 0:
            raise ValueError("relative cutoffs must be non-negative")
        retained = values.clone()
        retained[retained <= largest * cutoff] = 0
        total = retained.sum()
        squared = retained.square().sum()
        rank = (
            0.0
            if total <= 0 or squared <= 0
            else float((total.square() / squared).item())
        )
        cutoff_ranks.append(rank)
        rows.append(
            {
                "kind": "relative_eigenvalue_cutoff",
                "value": float(cutoff),
                "participation_ratio": rank,
                "epsilon_dominates": False,
            }
        )
    total = values.sum()
    squared = values.square().sum()
    formula_dominance = []
    for epsilon in formula_epsilons:
        if epsilon < 0:
            raise ValueError("formula epsilons must be non-negative")
        dominates = bool(total <= epsilon or squared <= epsilon)
        formula_dominance.append(dominates)
        rank = (
            0.0
            if dominates
            else float((total.square() / squared).item())
        )
        rows.append(
            {
                "kind": "stage1_denominator_epsilon",
                "value": float(epsilon),
                "participation_ratio": rank,
                "epsilon_dominates": dominates,
            }
        )
    positive = [rank for rank in cutoff_ranks if rank > 0]
    cutoff_stable = not positive or max(positive) - min(positive) <= max(
        1e-6, 0.01 * max(positive)
    )
    return rows, cutoff_stable and not any(formula_dominance)
