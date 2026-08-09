from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.run_stage3_q2_representation import (
    compensation_metrics,
    cross_validated_predictions,
    linear_cka,
    representation_metrics,
    select_balanced_indices,
    validate_subset,
)
from evaluation.deterministic_umap import (
    DeterministicUMAP,
    fuzzy_graph,
    smooth_knn_membership,
)


def test_balanced_subset_is_deterministic_and_balanced() -> None:
    labels = np.repeat(np.arange(10), 30)
    first = select_balanced_indices(labels, samples_per_class=5, seed=17)
    second = select_balanced_indices(labels, samples_per_class=5, seed=17)
    assert np.array_equal(first, second)
    validate_subset(first, labels[first], samples_per_class=5)
    assert np.array_equal(np.bincount(labels[first]), np.full(10, 5))


def test_subset_validation_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="not unique"):
        validate_subset(
            np.asarray([1, 1]),
            np.asarray([0, 0]),
            samples_per_class=0,
        )


def test_linear_cka_identity_and_orthogonal_invariance() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(40, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert linear_cka(values, values) == pytest.approx(1.0)
    assert linear_cka(values, values @ q) == pytest.approx(1.0)


def test_cross_validated_predictions_cover_each_sample_once() -> None:
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(2), 20)
    features = np.column_stack(
        [labels + rng.normal(scale=0.05, size=40), rng.normal(size=40)]
    ).astype(np.float32)
    linear, knn, predictions = cross_validated_predictions(
        features, labels, folds=4, seed=17, neighbors=3
    )
    assert linear > 0.9
    assert knn > 0.9
    assert predictions.shape == labels.shape


def test_representation_metrics_use_explicit_channel_and_sample_axes() -> None:
    torch.manual_seed(0)
    activations = torch.rand(40, 4, 2, 2)
    labels = np.repeat(np.arange(10), 4)
    metrics, embedding, predictions = representation_metrics(
        activations,
        labels,
        winner_fraction=0.25,
        epsilon=1e-12,
        pca_components=3,
        pca_seed=17,
        cv_folds=2,
        cv_seed=17,
        knn_neighbors=3,
    )
    assert metrics["channel_observation_shape"] == [160, 4]
    assert metrics["sample_geometry_shape"] == [40, 16]
    assert metrics["channel_observation_axis"] == "sample_x_spatial_location"
    assert metrics["channel_feature_axis"] == "channel"
    assert 0 <= metrics["effective_rank"] <= 4
    assert embedding.shape == (40, 3)
    assert predictions.shape == labels.shape


def test_compensation_is_computed_within_seed_and_method() -> None:
    rows = [
        {
            "seed": 0,
            "method": "HHB",
            "layer": "h2",
            "effective_rank": 2.0,
            "linear_probe_cv_accuracy": 0.8,
            "between_within_scatter_ratio": 0.5,
        },
        {
            "seed": 0,
            "method": "HHB",
            "layer": "z",
            "effective_rank": 8.0,
            "linear_probe_cv_accuracy": 0.9,
            "between_within_scatter_ratio": 1.0,
        },
    ]
    result = compensation_metrics(rows)
    assert result[0]["effective_rank_z_over_h2"] == 4.0
    assert result[0]["linear_probe_z_minus_h2"] == pytest.approx(0.1)
    assert result[0]["separability_z_over_h2"] == 2.0


def test_smooth_knn_and_fuzzy_graph_are_finite_and_symmetric() -> None:
    distances = np.asarray([[0.1, 0.3, 0.8], [0.2, 0.5, 1.0]])
    rho, sigma = smooth_knn_membership(distances)
    assert np.all(np.isfinite(rho))
    assert np.all(sigma > 0)
    rng = np.random.default_rng(4)
    graph = fuzzy_graph(
        rng.normal(size=(30, 4)), n_neighbors=5, metric="euclidean"
    )
    assert graph.shape == (30, 30)
    assert np.max(np.abs((graph - graph.T).data), initial=0.0) < 1e-12


def test_internal_umap_is_deterministic() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(40, 5))
    reducer = DeterministicUMAP(
        n_neighbors=5,
        min_dist=0.1,
        random_state=17,
        epochs=5,
        negative_samples=2,
    )
    first = reducer.fit_transform(features)
    second = reducer.fit_transform(features)
    assert first.shape == (40, 2)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
