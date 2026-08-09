"""Formal five-seed layerwise representation analysis for Stage 3 / Q2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.deterministic_umap import DeterministicUMAP
from evaluation.effective_rank_audit import (
    class_rank_metrics,
    representation_matrix,
    spectrum_metrics,
)
from evaluation.representation_health import compute_layer_health
from evaluation.representations import extract_representations
from models import ConvAutoencoder
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]
LAYER_TO_ENCODER = {"h1": "enc1", "h2": "enc2", "z": "enc3"}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def select_balanced_indices(
    labels: np.ndarray,
    *,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or samples_per_class <= 0:
        raise ValueError("labels must be 1D and samples_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(10):
        candidates = np.flatnonzero(labels == class_id)
        if candidates.size < samples_per_class:
            raise ValueError(f"class {class_id} has insufficient samples")
        selected.append(
            np.sort(
                rng.choice(candidates, size=samples_per_class, replace=False)
            )
        )
    values = np.concatenate(selected)
    return values[rng.permutation(values.size)]


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def create_subset(
    test_dataset,
    *,
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_all = np.asarray(test_dataset.targets, dtype=np.int64)
    sample_ids = select_balanced_indices(
        labels_all, samples_per_class=samples_per_class, seed=seed
    )
    labels = labels_all[sample_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sample_ids=sample_ids,
        labels=labels,
        subset_seed=np.asarray(seed, dtype=np.int64),
        samples_per_class=np.asarray(samples_per_class, dtype=np.int64),
    )
    return sample_ids, labels


def validate_subset(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int,
) -> None:
    if sample_ids.ndim != 1 or labels.shape != sample_ids.shape:
        raise ValueError("subset IDs and labels must be aligned 1D arrays")
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("subset sample IDs are not unique")
    counts = np.bincount(labels, minlength=10)
    if not np.array_equal(counts, np.full(10, samples_per_class)):
        raise ValueError(f"subset is not class balanced: {counts.tolist()}")


def layer_rule(resolved: dict, layer: str) -> str:
    return resolved["hybrid"]["encoder_layer_rules"][LAYER_TO_ENCODER[layer]]


def geometry_embedding(
    features: np.ndarray,
    *,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    component_count = min(components, values.shape[0] - 1, values.shape[1])
    if component_count < 2:
        raise ValueError("at least two PCA components are required")
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=seed,
    )
    embedding = pca.fit_transform(values).astype(np.float32, copy=False)
    return embedding, pca.explained_variance_ratio_.astype(
        np.float64, copy=False
    )


def cross_validated_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    neighbors: int,
) -> tuple[float, float, np.ndarray]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    linear_predictions = np.empty_like(labels)
    linear_scores: list[float] = []
    knn_scores: list[float] = []
    for train_index, validation_index in splitter.split(features, labels):
        scaler = StandardScaler(copy=True)
        train = scaler.fit_transform(features[train_index]).astype(
            np.float32, copy=False
        )
        validation = scaler.transform(features[validation_index]).astype(
            np.float32, copy=False
        )
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
            shuffle=True,
        )
        classifier.fit(train, labels[train_index])
        predictions = classifier.predict(validation)
        linear_predictions[validation_index] = predictions
        linear_scores.append(
            accuracy_score(labels[validation_index], predictions)
        )
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(train, labels[train_index])
        knn_scores.append(
            accuracy_score(labels[validation_index], knn.predict(validation))
        )
    return (
        float(np.mean(linear_scores)),
        float(np.mean(knn_scores)),
        linear_predictions,
    )


def class_geometry(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    values = StandardScaler().fit_transform(embedding).astype(
        np.float64, copy=False
    )
    global_mean = values.mean(axis=0)
    within_scatter = 0.0
    centroids = []
    within_distances = []
    for class_id in range(10):
        class_values = values[labels == class_id]
        centroid = class_values.mean(axis=0)
        centroids.append(centroid)
        residual = class_values - centroid
        within_scatter += float(np.square(residual).sum())
        within_distances.extend(np.linalg.norm(residual, axis=1).tolist())
    centroid_array = np.stack(centroids)
    between_scatter = sum(
        int((labels == class_id).sum())
        * float(np.square(centroid_array[class_id] - global_mean).sum())
        for class_id in range(10)
    )
    centroid_distances = np.linalg.norm(
        centroid_array[:, None, :] - centroid_array[None, :, :], axis=2
    )
    off_diagonal = ~np.eye(10, dtype=bool)
    pair_index = np.argmin(
        np.where(off_diagonal, centroid_distances, np.inf)
    )
    confused_pair = np.unravel_index(pair_index, centroid_distances.shape)
    tensor = torch.from_numpy(values)
    per_class, covariance_ranks = class_rank_metrics(
        tensor, torch.from_numpy(labels)
    )
    return {
        "within_class_mean_distance": float(np.mean(within_distances)),
        "between_class_mean_centroid_distance": float(
            centroid_distances[off_diagonal].mean()
        ),
        "between_within_scatter_ratio": between_scatter
        / max(within_scatter, 1e-12),
        "silhouette_score": float(silhouette_score(values, labels)),
        "closest_centroid_class_a": int(confused_pair[0]),
        "closest_centroid_class_b": int(confused_pair[1]),
        "closest_centroid_distance": float(
            centroid_distances[confused_pair]
        ),
        "per_class_effective_rank": {
            str(row["class_id"]): float(row["participation_ratio"])
            for row in per_class
        },
        **covariance_ranks,
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("CKA inputs must be sample-by-feature matrices")
    if left_values.shape[0] != right_values.shape[0]:
        raise ValueError("CKA inputs must have the same sample count")
    left_values -= left_values.mean(axis=0, keepdims=True)
    right_values -= right_values.mean(axis=0, keepdims=True)
    cross = left_values.T @ right_values
    left_cov = left_values.T @ left_values
    right_cov = right_values.T @ right_values
    denominator = np.linalg.norm(left_cov) * np.linalg.norm(right_cov)
    return (
        0.0
        if denominator <= np.finfo(np.float64).eps
        else float(np.square(cross).sum() / denominator)
    )


def representation_metrics(
    activations: torch.Tensor,
    labels: np.ndarray,
    *,
    winner_fraction: float,
    epsilon: float,
    pca_components: int,
    pca_seed: int,
    cv_folds: int,
    cv_seed: int,
    knn_neighbors: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if activations.ndim != 4 or activations.shape[0] != labels.size:
        raise ValueError("activations must be NCHW and align with labels")
    if not torch.isfinite(activations).all():
        raise ValueError("activations contain NaN or Inf")
    channel_matrix, channel_axes = representation_matrix(
        activations, view="channel_health"
    )
    spectrum = spectrum_metrics(channel_matrix)
    health, _, _ = compute_layer_health(
        activations,
        winner_fraction=winner_fraction,
        activation_epsilon=epsilon,
        variance_epsilon=epsilon,
    )
    raw_features = (
        activations.flatten(start_dim=1).numpy().astype(np.float32, copy=False)
    )
    embedding, explained = geometry_embedding(
        raw_features, components=pca_components, seed=pca_seed
    )
    probe_accuracy, knn_accuracy, predictions = cross_validated_predictions(
        raw_features,
        labels,
        folds=cv_folds,
        seed=cv_seed,
        neighbors=knn_neighbors,
    )
    geometry = class_geometry(embedding, labels)
    metrics = {
        "source_shape": list(activations.shape),
        "sample_geometry_shape": list(raw_features.shape),
        "channel_observation_shape": channel_axes["matrix_shape"],
        "channel_observation_axis": channel_axes["observation_axis"],
        "channel_feature_axis": channel_axes["feature_axis"],
        "activation_sparsity": float(
            (activations.abs() <= epsilon).to(torch.float64).mean().item()
        ),
        "active_neuron_ratio": float(health["active_unit_ratio"]),
        "dead_unit_ratio": float(health["dead_unit_ratio"]),
        "winner_coverage_ratio": float(health["winner_coverage_ratio"]),
        "winner_entropy": float(health["winner_entropy"]),
        "max_winner_share": float(health["max_winner_share"]),
        "effective_rank": spectrum.participation_ratio,
        "stable_rank": spectrum.stable_rank,
        "rank_ratio": spectrum.rank_ratio,
        "numerical_rank": spectrum.numerical_rank,
        "spectrum": spectrum.eigenvalues.numpy().tolist(),
        "pca_components": int(embedding.shape[1]),
        "pca_explained_variance_ratio_sum": float(explained.sum()),
        "linear_probe_cv_accuracy": probe_accuracy,
        "knn_cv_accuracy": knn_accuracy,
        **geometry,
    }
    return metrics, embedding, predictions


def compensation_metrics(rows: list[dict]) -> list[dict]:
    by_key = {(row["seed"], row["method"], row["layer"]): row for row in rows}
    output = []
    methods = sorted({row["method"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    for method in methods:
        for seed in seeds:
            h2 = by_key[(seed, method, "h2")]
            z = by_key[(seed, method, "z")]
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "effective_rank_z_over_h2": z["effective_rank"]
                    / max(h2["effective_rank"], 1e-12),
                    "effective_rank_z_minus_h2": z["effective_rank"]
                    - h2["effective_rank"],
                    "linear_probe_z_minus_h2": z[
                        "linear_probe_cv_accuracy"
                    ]
                    - h2["linear_probe_cv_accuracy"],
                    "separability_z_over_h2": z[
                        "between_within_scatter_ratio"
                    ]
                    / max(h2["between_within_scatter_ratio"], 1e-12),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> list[dict]:
    output = []
    numeric = (
        "activation_sparsity",
        "active_neuron_ratio",
        "winner_coverage_ratio",
        "winner_entropy",
        "max_winner_share",
        "effective_rank",
        "stable_rank",
        "rank_ratio",
        "linear_probe_cv_accuracy",
        "knn_cv_accuracy",
        "within_class_mean_distance",
        "between_class_mean_centroid_distance",
        "between_within_scatter_ratio",
        "silhouette_score",
        "within_class_covariance_participation_rank",
        "between_class_covariance_participation_rank",
    )
    for method in sorted({row["method"] for row in rows}):
        for layer in ("h1", "h2", "z"):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["layer"] == layer
            ]
            summary: dict[str, Any] = {"method": method, "layer": layer}
            for field in numeric:
                values = np.asarray([row[field] for row in selected])
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_sd"] = float(values.std(ddof=1))
            output.append(summary)
    return output


def _method_order() -> tuple[str, ...]:
    return ("BBB", "HHH", "HHB", "HBB", "RBB", "RRB")


def plot_analysis(
    output_dir: Path,
    rows: list[dict],
    embeddings: dict[tuple[int, str, str], np.ndarray],
    umap_embeddings: dict[tuple[str, str], np.ndarray],
    labels: np.ndarray,
    confusion: dict[tuple[str, str], np.ndarray],
    cka: dict[str, np.ndarray],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir()
    summary = summarize_rows(rows)
    methods = _method_order()
    layers = ("h1", "h2", "z")
    colors = plt.get_cmap("tab10")

    for metric, ylabel, filename in (
        ("effective_rank", "Channel covariance effective rank", "effective_rank"),
        (
            "linear_probe_cv_accuracy",
            "Layerwise CV linear-probe accuracy",
            "linear_probe",
        ),
        (
            "between_within_scatter_ratio",
            "Between/within scatter ratio",
            "separability",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(layers))
        for method in methods:
            selected = {
                row["layer"]: row for row in summary if row["method"] == method
            }
            ax.errorbar(
                x,
                [selected[layer][f"{metric}_mean"] for layer in layers],
                yerr=[selected[layer][f"{metric}_sd"] for layer in layers],
                marker="o",
                capsize=3,
                label=method,
            )
        ax.set_xticks(x, layers)
        ax.set_ylabel(ylabel)
        ax.legend(ncol=3)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f"{filename}.png", dpi=160)
        fig.savefig(figures / f"{filename}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(len(methods), len(layers), figsize=(12, 18))
    for row_index, method in enumerate(methods):
        for column_index, layer in enumerate(layers):
            ax = axes[row_index, column_index]
            embedding = embeddings[(0, method, layer)]
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="tab10",
                s=3,
                alpha=0.65,
            )
            ax.set_title(f"{method} {layer}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Seed-0 PCA (fixed representative visualization)")
    fig.tight_layout()
    fig.savefig(figures / "pca_seed0_grid.png", dpi=160)
    fig.savefig(figures / "pca_seed0_grid.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(len(methods), len(layers), figsize=(12, 18))
    for row_index, method in enumerate(methods):
        for column_index, layer in enumerate(layers):
            ax = axes[row_index, column_index]
            embedding = umap_embeddings[(method, layer)]
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="tab10",
                s=3,
                alpha=0.65,
            )
            ax.set_title(f"{method} {layer}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Seed-0 UMAP (n_neighbors=15, min_dist=0.1, seed=17)")
    fig.tight_layout()
    fig.savefig(figures / "umap_seed0_grid.png", dpi=160)
    fig.savefig(figures / "umap_seed0_grid.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, method in zip(axes.flat, methods):
        matrix = confusion[(method, "z")]
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{method} z")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle("Five-seed aggregated out-of-fold confusion matrices")
    fig.savefig(figures / "z_confusion_matrices.png", dpi=160)
    fig.savefig(figures / "z_confusion_matrices.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, layer in zip(axes, layers):
        image = ax.imshow(cka[layer], vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(methods)), methods, rotation=45)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(layer)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Paired-seed mean PCA-50 linear CKA")
    fig.savefig(figures / "layerwise_cka.png", dpi=160)
    fig.savefig(figures / "layerwise_cka.pdf")
    plt.close(fig)


def run(config_path: Path) -> Path:
    protocol = yaml.safe_load(resolve(config_path).read_text(encoding="utf-8"))
    if protocol["version"] != "stage3-q2-representation-v1":
        raise ValueError("unsupported Q2 protocol")
    output_dir = resolve(protocol["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"immutable Q2 output already exists: {output_dir}")

    for method in protocol["methods"].values():
        root = resolve(method["results_root"])
        if read_json(root / "freeze_gate.json")["decision"] != "PASS":
            raise RuntimeError(f"freeze gate did not pass: {root}")
        if not read_json(root / "test_evaluation" / "summary.json")[
            "records_complete"
        ]:
            raise RuntimeError(f"formal test evaluation incomplete: {root}")

    output_dir.mkdir(parents=True)
    representation_dir = output_dir / "representations"
    representation_dir.mkdir()
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir()

    data_spec = protocol["data"]
    test_dataset = datasets.MNIST(
        root=str(resolve(data_spec["root"])),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    manifest_path = output_dir / "mnist_test_representation_seed17_v1.npz"
    sample_ids, expected_labels = create_subset(
        test_dataset,
        output_path=manifest_path,
        samples_per_class=int(data_spec["samples_per_class"]),
        seed=int(data_spec["subset_seed"]),
    )
    validate_subset(
        sample_ids,
        expected_labels,
        samples_per_class=int(data_spec["samples_per_class"]),
    )
    loader = DataLoader(
        IndexedDataset(test_dataset, sample_ids),
        batch_size=int(data_spec["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    analysis = protocol["analysis"]
    rows: list[dict[str, Any]] = []
    embeddings: dict[tuple[int, str, str], np.ndarray] = {}
    confusion: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((10, 10), dtype=np.int64)
    )
    integrity: dict[str, Any] = {}
    input_cache: np.ndarray | None = None
    labels_cache: np.ndarray | None = None

    for seed in protocol["seeds"]:
        for method_id, method_spec in protocol["methods"].items():
            method = method_spec["label"]
            run_dir = (
                resolve(method_spec["results_root"])
                / "runs"
                / f"seed_{seed}"
                / method_id
            )
            resolved = yaml.safe_load(
                (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
            )
            model = ConvAutoencoder(
                latent_dim=int(resolved["model"]["latent_dim"]), seed=int(seed)
            )
            checkpoint = run_dir / "model_best.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            checksum_before = state_dict_checksum(model)

            extracted = extract_representations(
                model,
                loader,
                device=torch.device("cpu"),
                layers=analysis["layers"],
            )
            checksum_after = state_dict_checksum(model)
            labels = extracted["label"].numpy().astype(np.int64, copy=False)
            ids = extracted["sample_id"].numpy().astype(np.int64, copy=False)
            if not np.array_equal(ids, sample_ids) or not np.array_equal(
                labels, expected_labels
            ):
                raise RuntimeError("representation sample IDs/order do not match")
            if checksum_before != checksum_after:
                raise RuntimeError("representation extraction mutated encoder")

            if input_cache is None:
                input_batches = []
                for images, _, _ in loader:
                    input_batches.append(images.numpy())
                input_cache = np.concatenate(input_batches).astype(
                    np.float16, copy=False
                )
                labels_cache = labels

            archive = representation_dir / f"seed_{seed}_{method}.npz"
            np.savez_compressed(
                archive,
                input=input_cache,
                h1=extracted["h1"].numpy().astype(np.float16),
                h2=extracted["h2"].numpy().astype(np.float16),
                z=extracted["z"].numpy().astype(np.float16),
                labels=labels,
                sample_ids=ids,
            )
            archive_metadata = {
                "schema_version": "stage3-q2-representation-archive-v1",
                "seed": int(seed),
                "method_id": method_id,
                "method": method,
                "encoder_layer_rules": resolved["hybrid"][
                    "encoder_layer_rules"
                ],
                "architecture_id": resolved["model"]["architecture"],
                "latent_dim": int(resolved["model"]["latent_dim"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_state_before": checksum_before,
                "checkpoint_state_after": checksum_after,
                "checkpoint_unchanged": checksum_before == checksum_after,
                "sample_ids_sha256": sha256_array(ids),
                "labels_sha256": sha256_array(labels),
                "archive_sha256": file_sha256(archive),
                "shapes": {
                    key: list(extracted[key].shape)
                    for key in analysis["layers"]
                },
                "all_finite": all(
                    bool(torch.isfinite(extracted[key]).all())
                    for key in analysis["layers"]
                ),
                "test_samples_accessed": int(labels.size),
            }
            write_json(archive.with_suffix(".json"), archive_metadata)
            integrity[f"seed_{seed}/{method}"] = archive_metadata

            for layer in analysis["layers"]:
                metric, embedding, predictions = representation_metrics(
                    extracted[layer],
                    labels,
                    winner_fraction=float(analysis["winner_fraction"]),
                    epsilon=float(analysis["activation_epsilon"]),
                    pca_components=int(analysis["pca_components"]),
                    pca_seed=int(analysis["pca_seed"]),
                    cv_folds=int(analysis["cv_folds"]),
                    cv_seed=int(analysis["cv_seed"]),
                    knn_neighbors=int(analysis["knn_neighbors"]),
                )
                embedding_path = (
                    embedding_dir / f"seed_{seed}_{method}_{layer}_pca.npz"
                )
                np.savez_compressed(
                    embedding_path,
                    embedding=embedding,
                    labels=labels,
                    sample_ids=ids,
                    predictions=predictions,
                )
                embeddings[(int(seed), method, layer)] = embedding
                confusion[(method, layer)] += confusion_matrix(
                    labels, predictions, labels=np.arange(10)
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "method_id": method_id,
                        "method": method,
                        "layer": layer,
                        "layer_rule": layer_rule(resolved, layer),
                        **{
                            key: value
                            for key, value in metric.items()
                            if key
                            not in {
                                "spectrum",
                                "per_class_effective_rank",
                                "source_shape",
                                "sample_geometry_shape",
                                "channel_observation_shape",
                            }
                        },
                        "source_shape": json.dumps(metric["source_shape"]),
                        "sample_geometry_shape": json.dumps(
                            metric["sample_geometry_shape"]
                        ),
                        "channel_observation_shape": json.dumps(
                            metric["channel_observation_shape"]
                        ),
                        "per_class_effective_rank": json.dumps(
                            metric["per_class_effective_rank"], sort_keys=True
                        ),
                        "spectrum": json.dumps(metric["spectrum"]),
                    }
                )

    if labels_cache is None:
        raise RuntimeError("no representations extracted")
    write_csv(output_dir / "per_seed_layer_metrics.csv", rows)
    summary_rows = summarize_rows(rows)
    write_csv(output_dir / "method_layer_summary.csv", summary_rows)
    compensation = compensation_metrics(rows)
    write_csv(output_dir / "compensation_metrics.csv", compensation)

    methods = _method_order()
    cka: dict[str, np.ndarray] = {}
    cka_rows = []
    for layer in ("h1", "h2", "z"):
        matrix = np.zeros((len(methods), len(methods)), dtype=np.float64)
        for left_index, left in enumerate(methods):
            for right_index, right in enumerate(methods):
                values = [
                    linear_cka(
                        embeddings[(seed, left, layer)],
                        embeddings[(seed, right, layer)],
                    )
                    for seed in protocol["seeds"]
                ]
                matrix[left_index, right_index] = float(np.mean(values))
                if left_index <= right_index:
                    cka_rows.append(
                        {
                            "layer": layer,
                            "left_method": left,
                            "right_method": right,
                            "mean_pca50_linear_cka": float(np.mean(values)),
                            "sd_pca50_linear_cka": float(
                                np.std(values, ddof=1)
                            ),
                        }
                    )
        cka[layer] = matrix
    write_csv(output_dir / "layerwise_cka.csv", cka_rows)

    umap_embeddings: dict[tuple[str, str], np.ndarray] = {}
    umap_spec = analysis["umap"]
    for method in methods:
        for layer in ("h1", "h2", "z"):
            reducer = DeterministicUMAP(
                n_neighbors=int(umap_spec["n_neighbors"]),
                min_dist=float(umap_spec["min_dist"]),
                metric=umap_spec["metric"],
                random_state=int(umap_spec["random_state"]),
                epochs=int(umap_spec["epochs"]),
                negative_samples=int(umap_spec["negative_samples"]),
                spread=float(umap_spec["spread"]),
            )
            values = reducer.fit_transform(embeddings[(0, method, layer)])
            umap_embeddings[(method, layer)] = values
            np.savez_compressed(
                embedding_dir / f"seed_0_{method}_{layer}_umap.npz",
                embedding=values,
                labels=labels_cache,
                sample_ids=sample_ids,
            )

    plot_analysis(
        output_dir,
        rows,
        embeddings,
        umap_embeddings,
        labels_cache,
        confusion,
        cka,
    )
    write_json(
        output_dir / "integrity.json",
        {
            "schema_version": "stage3-q2-integrity-v1",
            "completed_at_utc": utc_now(),
            "records": integrity,
            "record_count": len(integrity),
            "expected_record_count": 30,
            "all_checkpoints_unchanged": all(
                record["checkpoint_unchanged"]
                for record in integrity.values()
            ),
            "all_arrays_finite": all(
                record["all_finite"] for record in integrity.values()
            ),
            "same_sample_ids": len(
                {record["sample_ids_sha256"] for record in integrity.values()}
            )
            == 1,
            "same_labels": len(
                {record["labels_sha256"] for record in integrity.values()}
            )
            == 1,
            "samples_per_class": np.bincount(
                labels_cache, minlength=10
            ).tolist(),
            "test_samples_per_checkpoint": int(labels_cache.size),
            "test_samples_total": int(labels_cache.size * len(integrity)),
            "test_used_for_selection": False,
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "stage3-q2-run-manifest-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(resolve(config_path)),
            "protocol_sha256": file_sha256(resolve(config_path)),
            "methods": list(methods),
            "seeds": list(protocol["seeds"]),
            "layers": list(analysis["layers"]),
            "subset_manifest": str(manifest_path),
            "subset_manifest_sha256": file_sha256(manifest_path),
            "metric_rows": len(rows),
            "expected_metric_rows": 90,
            "performance_gate_applied": False,
            "training_performed": False,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/stage3_q2_representation_v1.yaml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.config).resolve())


if __name__ == "__main__":
    main()
