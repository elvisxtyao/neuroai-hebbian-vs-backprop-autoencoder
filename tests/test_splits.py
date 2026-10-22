from copy import deepcopy
from pathlib import Path

import numpy as np

from data.mnist import build_mnist_dataloaders
from data.splits import (
    create_stratified_split,
    labels_checksum,
    validate_split_indices,
)
from schemas import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_stratified_split_is_reproducible_and_disjoint():
    labels = np.repeat(np.arange(10), 60)
    train_a, validation_a = create_stratified_split(
        labels, validation_size=100, seed=0
    )
    train_b, validation_b = create_stratified_split(
        labels, validation_size=100, seed=0
    )
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    validate_split_indices(
        train_a,
        validation_a,
        dataset_size=600,
        expected_validation_size=100,
    )
    validation_counts = np.bincount(labels[validation_a], minlength=10)
    np.testing.assert_array_equal(validation_counts, np.full(10, 10))


def test_train_validation_loaders_do_not_construct_test_dataset(
    tmp_path, monkeypatch
):
    config = deepcopy(load_config(ROOT / "configs" / "bp_main.yaml"))
    config["data"]["root"] = str(tmp_path / "raw")
    config["data"]["split_manifest"] = str(tmp_path / "split.npz")
    labels = np.arange(60_000, dtype=np.int64) % 10
    np.savez_compressed(
        config["data"]["split_manifest"],
        train_indices=np.arange(50_000, dtype=np.int64),
        validation_indices=np.arange(50_000, 60_000, dtype=np.int64),
        test_indices=np.arange(10_000, dtype=np.int64),
        split_seed=np.asarray(0, dtype=np.int64),
        train_labels_checksum=np.asarray(labels_checksum(labels)),
        test_size=np.asarray(10_000, dtype=np.int64),
    )
    constructed_splits = []

    class FakeMNIST:
        def __init__(self, *, root, train, download, transform):
            constructed_splits.append("train" if train else "test")
            if not train:
                raise AssertionError("include_test=False constructed MNIST test")
            self.targets = labels

        def __len__(self):
            return 60_000

    monkeypatch.setattr("data.mnist.datasets.MNIST", FakeMNIST)
    loaders = build_mnist_dataloaders(
        config,
        seed=0,
        download=False,
        include_test=False,
    )

    assert set(loaders) == {"train", "validation"}
    assert constructed_splits == ["train", "train"]
