"""Framework-independent deterministic stratified split logic."""

from __future__ import annotations

import hashlib

import numpy as np


def labels_checksum(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    return hashlib.sha256(labels.tobytes()).hexdigest()


def _allocate_validation_counts(counts: np.ndarray, validation_size: int) -> np.ndarray:
    expected = counts.astype(np.float64) * validation_size / counts.sum()
    allocation = np.floor(expected).astype(np.int64)
    remaining = validation_size - int(allocation.sum())
    if remaining:
        order = np.argsort(-(expected - allocation), kind="stable")
        allocation[order[:remaining]] += 1
    return allocation


def create_stratified_split(
    labels: np.ndarray,
    *,
    validation_size: int = 10_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices with proportional classes."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < validation_size < labels.size:
        raise ValueError("validation_size must be between 0 and dataset size")

    classes, counts = np.unique(labels, return_counts=True)
    validation_counts = _allocate_validation_counts(counts, validation_size)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_id, validation_count in zip(classes, validation_counts, strict=True):
        class_indices = np.flatnonzero(labels == class_id)
        class_indices = rng.permutation(class_indices)
        validation_parts.append(class_indices[:validation_count])
        train_parts.append(class_indices[validation_count:])

    train_indices = np.sort(np.concatenate(train_parts)).astype(np.int64)
    validation_indices = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    validate_split_indices(
        train_indices,
        validation_indices,
        dataset_size=labels.size,
        expected_validation_size=validation_size,
    )
    return train_indices, validation_indices


def validate_split_indices(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    dataset_size: int,
    expected_validation_size: int,
) -> None:
    train_indices = np.asarray(train_indices)
    validation_indices = np.asarray(validation_indices)
    if validation_indices.size != expected_validation_size:
        raise ValueError("Validation split has the wrong size")
    if train_indices.size + validation_indices.size != dataset_size:
        raise ValueError("Train and validation splits do not cover the dataset")
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train and validation splits overlap")
    combined = np.concatenate([train_indices, validation_indices])
    if combined.min(initial=0) < 0 or combined.max(initial=-1) >= dataset_size:
        raise ValueError("Split index is outside the dataset")
    if np.unique(combined).size != dataset_size:
        raise ValueError("Split contains duplicate or missing indices")

