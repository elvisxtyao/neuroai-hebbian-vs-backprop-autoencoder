"""Train the frozen, standardized single-layer linear probe."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.mnist import build_mnist_dataloaders
from evaluation.metrics import classification_metrics
from evaluation.representations import extract_representations
from models import ConvAutoencoder, LinearProbe
from schemas import load_config
from utils.reproducibility import set_global_seed, state_dict_checksum
from utils.results import append_metric, remove_metric_stages


def _feature_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        drop_last=False,
    )


@torch.no_grad()
def _evaluate_probe(probe, loader, device: torch.device) -> dict[str, float]:
    probe.eval()
    labels: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for features, batch_labels in loader:
        logits = probe(features.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        labels.append(batch_labels.cpu())
    labels_np = torch.cat(labels).numpy()
    probabilities_np = torch.cat(probabilities).numpy()
    predictions_np = probabilities_np.argmax(axis=1)
    return classification_metrics(labels_np, predictions_np, probabilities_np)


def train_linear_probe(config_path: str | Path, run_dir: str | Path) -> Path:
    config = load_config(config_path)
    run_dir = Path(run_dir)
    remove_metric_stages(run_dir, {"linear_probe", "linear_probe_final"})
    checkpoint = run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing representation checkpoint: {checkpoint}")

    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_hash_before = state_dict_checksum(model.encoder)

    loaders = build_mnist_dataloaders(config, seed=seed)
    representations = {
        split: extract_representations(
            model, loader, device=device, layers=("z",)
        )
        for split, loader in loaders.items()
    }
    features = {
        split: values["z"].flatten(start_dim=1).float()
        for split, values in representations.items()
    }
    labels = {split: values["label"].long() for split, values in representations.items()}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        probe = LinearProbe(
            config["model"]["latent_dim"],
            epsilon=config["probe"]["standardization_epsilon"],
        )
    probe.standardizer.fit(features["train"])
    probe.to(device)
    optimizer = torch.optim.SGD(
        probe.classifier.parameters(),
        lr=config["probe"]["lr"],
        momentum=config["probe"]["momentum"],
        weight_decay=config["probe"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    feature_loaders = {
        split: _feature_loader(
            features[split],
            labels[split],
            batch_size=config["data"]["batch_size"],
            shuffle=split == "train",
            seed=seed,
        )
        for split in ("train", "validation", "test")
    }

    best_accuracy = float("-inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, config["probe"]["epochs"] + 1):
        probe.train()
        for batch_features, batch_labels in feature_loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
        validation_metrics = _evaluate_probe(probe, feature_loaders["validation"], device)
        append_metric(
            run_dir,
            {
                "stage": "linear_probe",
                "split": "validation",
                "epoch": epoch,
                **validation_metrics,
                "num_samples": labels["validation"].numel(),
            },
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_state = deepcopy(probe.state_dict())
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("Linear probe produced no checkpoint")
    probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), run_dir / "linear_probe.pt")
    for split in ("validation", "test"):
        metrics = _evaluate_probe(probe, feature_loaders[split], device)
        append_metric(
            run_dir,
            {
                "stage": "linear_probe_final",
                "split": split,
                "epoch": best_epoch,
                **metrics,
                "num_samples": labels[split].numel(),
            },
        )
        print(split, {"selected_epoch": best_epoch, **metrics})

    encoder_hash_after = state_dict_checksum(model.encoder)
    if encoder_hash_before != encoder_hash_after:
        raise RuntimeError("Frozen encoder changed during linear-probe training")
    return run_dir / "linear_probe.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(train_linear_probe(args.config, args.run_dir).resolve())


if __name__ == "__main__":
    main()
