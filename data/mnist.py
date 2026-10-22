"""MNIST manifest generation and indexed DataLoaders."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from data.splits import create_stratified_split, labels_checksum, validate_split_indices
from schemas import load_config


class IndexedDataset(Dataset):
    """Return ``(image, label, original_sample_id)`` for stable joins."""

    def __init__(self, dataset: Dataset, indices: np.ndarray | list[int] | None = None) -> None:
        self.dataset = dataset
        self.indices = (
            np.arange(len(dataset), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int):
        sample_id = int(self.indices[position])
        image, label = self.dataset[sample_id]
        return image, label, sample_id


def ensure_split_manifest(
    config: dict[str, Any],
    *,
    download: bool = True,
    verify_test: bool = True,
) -> Path:
    data_config = config["data"]
    manifest_path = Path(data_config["split_manifest"])
    transform = transforms.ToTensor()
    training = datasets.MNIST(
        root=data_config["root"], train=True, download=download, transform=transform
    )
    labels = np.asarray(training.targets, dtype=np.int64)
    checksum = labels_checksum(labels)

    if manifest_path.exists():
        manifest = np.load(manifest_path, allow_pickle=False)
        train_indices = manifest["train_indices"]
        validation_indices = manifest["validation_indices"]
        saved_checksum = str(manifest["train_labels_checksum"].item())
        validate_split_indices(
            train_indices,
            validation_indices,
            dataset_size=len(training),
            expected_validation_size=data_config["validation_size"],
        )
        if saved_checksum != checksum:
            raise RuntimeError("MNIST labels differ from the saved split manifest")
        if verify_test:
            test = datasets.MNIST(
                root=data_config["root"],
                train=False,
                download=download,
                transform=transform,
            )
            if int(manifest["test_size"].item()) != len(test):
                raise RuntimeError("MNIST test size differs from the split manifest")
        return manifest_path

    test = datasets.MNIST(
        root=data_config["root"], train=False, download=download, transform=transform
    )
    train_indices, validation_indices = create_stratified_split(
        labels,
        validation_size=data_config["validation_size"],
        seed=data_config["split_seed"],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        manifest_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=np.arange(len(test), dtype=np.int64),
        split_seed=np.asarray(data_config["split_seed"], dtype=np.int64),
        train_labels_checksum=np.asarray(checksum),
        test_size=np.asarray(len(test), dtype=np.int64),
    )
    return manifest_path


def build_mnist_dataloaders(
    config: dict[str, Any],
    *,
    seed: int,
    download: bool = True,
    include_test: bool = True,
) -> dict[str, DataLoader]:
    data_config = config["data"]
    manifest_path = ensure_split_manifest(
        config,
        download=download,
        verify_test=include_test,
    )
    manifest = np.load(manifest_path, allow_pickle=False)
    transform = transforms.ToTensor()
    training = datasets.MNIST(
        root=data_config["root"], train=True, download=download, transform=transform
    )
    datasets_by_split = {
        "train": IndexedDataset(training, manifest["train_indices"]),
        "validation": IndexedDataset(training, manifest["validation_indices"]),
    }
    if include_test:
        test = datasets.MNIST(
            root=data_config["root"],
            train=False,
            download=download,
            transform=transform,
        )
        datasets_by_split["test"] = IndexedDataset(test, manifest["test_indices"])
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets_by_split.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=data_config["batch_size"],
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=data_config["num_workers"],
            pin_memory=data_config["pin_memory"],
            drop_last=data_config["drop_last"],
        )
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--create-split", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.create_split:
        path = ensure_split_manifest(config)
        print(path.resolve())


if __name__ == "__main__":
    main()
