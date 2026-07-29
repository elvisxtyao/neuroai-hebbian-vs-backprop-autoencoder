"""Small deterministic exact-neighbor UMAP for fixed research visualizations.

The project analyzes only 2,000 points at a time, so an exact k-nearest-neighbor
graph is preferable to adding an approximate-neighbor runtime dependency.  The
implementation follows the UMAP construction stages: smooth-kNN distances,
fuzzy simplicial-set union, spectral initialization, and attractive/repulsive
low-dimensional cross-entropy optimization.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


def _find_ab(spread: float, min_dist: float) -> tuple[float, float]:
    if spread <= 0 or min_dist < 0 or min_dist > spread:
        raise ValueError("require spread > 0 and 0 <= min_dist <= spread")
    x = np.linspace(0, spread * 3, 300)
    target = np.where(x <= min_dist, 1.0, np.exp(-(x - min_dist) / spread))

    def curve(values, a, b):
        return 1.0 / (1.0 + a * values ** (2 * b))

    parameters, _ = curve_fit(
        curve, x, target, p0=(1.0, 1.0), bounds=(1e-6, np.inf), maxfev=10_000
    )
    return float(parameters[0]), float(parameters[1])


def smooth_knn_membership(
    distances: np.ndarray,
    *,
    local_connectivity: float = 1.0,
    iterations: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row rho and sigma for non-self neighbor distances."""

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("distances must be N by at least two neighbors")
    if local_connectivity != 1.0:
        raise ValueError("this fixed protocol supports local_connectivity=1")
    target = np.log2(values.shape[1])
    rho = values[:, 0].copy()
    sigma = np.empty(values.shape[0], dtype=np.float64)
    for row_index, row in enumerate(values):
        lower = 0.0
        upper = np.inf
        current = 1.0
        shifted = np.maximum(row - rho[row_index], 0.0)
        for _ in range(iterations):
            membership_sum = np.exp(-shifted / max(current, 1e-12)).sum()
            if abs(membership_sum - target) < 1e-5:
                break
            if membership_sum > target:
                upper = current
                current = (lower + upper) / 2.0
            else:
                lower = current
                current = current * 2.0 if np.isinf(upper) else (lower + upper) / 2.0
        sigma[row_index] = max(current, 1e-6)
    return rho, sigma


def fuzzy_graph(
    features: np.ndarray,
    *,
    n_neighbors: int,
    metric: str,
) -> sparse.csr_matrix:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] <= n_neighbors:
        raise ValueError("features must be 2D with more rows than n_neighbors")
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors + 1, metric=metric, algorithm="auto", n_jobs=1
    )
    distances, indices = neighbors.fit(values).kneighbors(values)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    rho, sigma = smooth_knn_membership(distances)
    memberships = np.exp(
        -np.maximum(distances - rho[:, None], 0.0) / sigma[:, None]
    )
    row_ids = np.repeat(np.arange(values.shape[0]), n_neighbors)
    directed = sparse.coo_matrix(
        (memberships.ravel(), (row_ids, indices.ravel())),
        shape=(values.shape[0], values.shape[0]),
    ).tocsr()
    product = directed.multiply(directed.T)
    union = directed + directed.T - product
    union.setdiag(0)
    union.eliminate_zeros()
    return union


def _spectral_initialization(
    graph: sparse.csr_matrix,
    *,
    seed: int,
) -> np.ndarray:
    spectral = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        random_state=seed,
        eigen_solver="arpack",
    )
    embedding = spectral.fit_transform(graph).astype(np.float64)
    scale = np.max(np.abs(embedding))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("spectral initialization is degenerate")
    rng = np.random.default_rng(seed)
    return embedding / scale * 10.0 + rng.normal(
        scale=1e-4, size=embedding.shape
    )


def optimize_layout(
    graph: sparse.csr_matrix,
    initial: np.ndarray,
    *,
    min_dist: float,
    spread: float,
    epochs: int,
    negative_samples: int,
    seed: int,
) -> np.ndarray:
    if epochs <= 0 or negative_samples <= 0:
        raise ValueError("epochs and negative_samples must be positive")
    coo = sparse.triu(graph, k=1).tocoo()
    heads = coo.row.astype(np.int64)
    tails = coo.col.astype(np.int64)
    weights = np.clip(coo.data.astype(np.float64), 0.0, 1.0)
    if weights.size == 0:
        raise ValueError("fuzzy graph contains no edges")
    a, b = _find_ab(spread, min_dist)
    rng = np.random.default_rng(seed)
    embedding = np.asarray(initial, dtype=np.float64).copy()
    vertex_count = embedding.shape[0]

    for epoch in range(epochs):
        alpha = 1.0 - epoch / epochs
        selected = rng.random(weights.size) < weights
        selected_heads = heads[selected]
        selected_tails = tails[selected]
        if selected_heads.size == 0:
            continue
        delta = np.zeros_like(embedding)
        counts = np.zeros((vertex_count, 1), dtype=np.float64)

        difference = embedding[selected_heads] - embedding[selected_tails]
        distance_squared = np.square(difference).sum(axis=1).clip(min=1e-12)
        coefficient = (
            -2.0
            * a
            * b
            * distance_squared ** (b - 1.0)
            / (1.0 + a * distance_squared**b)
        )
        gradient = np.clip(coefficient[:, None] * difference, -4.0, 4.0)
        np.add.at(delta, selected_heads, gradient)
        np.add.at(delta, selected_tails, -gradient)
        np.add.at(counts[:, 0], selected_heads, 1.0)
        np.add.at(counts[:, 0], selected_tails, 1.0)

        negative_heads = np.repeat(selected_heads, negative_samples)
        negative_tails = rng.integers(
            0, vertex_count, size=negative_heads.size
        )
        valid = negative_heads != negative_tails
        negative_heads = negative_heads[valid]
        negative_tails = negative_tails[valid]
        difference = embedding[negative_heads] - embedding[negative_tails]
        distance_squared = np.square(difference).sum(axis=1).clip(min=1e-12)
        coefficient = (
            2.0
            * b
            / ((0.001 + distance_squared) * (1.0 + a * distance_squared**b))
        )
        gradient = np.clip(coefficient[:, None] * difference, -4.0, 4.0)
        np.add.at(delta, negative_heads, gradient)
        np.add.at(delta, negative_tails, -gradient)
        np.add.at(counts[:, 0], negative_heads, 1.0)
        np.add.at(counts[:, 0], negative_tails, 1.0)

        embedding += alpha * delta / np.maximum(counts, 1.0)
        if not np.isfinite(embedding).all():
            raise FloatingPointError("UMAP layout optimization became non-finite")
    return embedding.astype(np.float32)


class DeterministicUMAP:
    """Fixed, exact-neighbor UMAP transformer for the formal Q2 figures."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 17,
        epochs: int = 200,
        negative_samples: int = 5,
        spread: float = 1.0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.spread = spread

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        graph = fuzzy_graph(
            features, n_neighbors=self.n_neighbors, metric=self.metric
        )
        initial = _spectral_initialization(graph, seed=self.random_state)
        return optimize_layout(
            graph,
            initial,
            min_dist=self.min_dist,
            spread=self.spread,
            epochs=self.epochs,
            negative_samples=self.negative_samples,
            seed=self.random_state,
        )
