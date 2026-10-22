"""Synthetic, download-free check of the shared Phase 0 skeleton."""

from __future__ import annotations

import argparse
import math

import torch

from learning_rules import build_trainer
from models import ConvAutoencoder, LinearProbe
from schemas import load_config
from utils.reproducibility import set_global_seed, state_dict_checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["training"]["learning_rule"] != "bp":
        raise ValueError("The initial smoke test requires the BP config")

    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = torch.device("cpu")
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
    twin = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
    if state_dict_checksum(model) != state_dict_checksum(twin):
        raise RuntimeError("Paired initialization is not reproducible")

    images = torch.rand(16, 1, 28, 28)
    features = model.encode(images, return_all_layers=True)
    expected = {
        "h1": (16, 16, 14, 14),
        "h2": (16, 32, 7, 7),
        "z": (16, config["model"]["latent_dim"], 1, 1),
    }
    for layer, shape in expected.items():
        if tuple(features[layer].shape) != shape:
            raise RuntimeError(f"Wrong {layer} shape: {tuple(features[layer].shape)}")
    reconstruction = model.reconstruct(images)
    if tuple(reconstruction.shape) != tuple(images.shape):
        raise RuntimeError("Reconstruction shape mismatch")
    if reconstruction.min() < 0 or reconstruction.max() > 1:
        raise RuntimeError("Sigmoid reconstruction left [0,1]")

    trainer = build_trainer(model, config, device)
    losses = [trainer.train_batch(images) for _ in range(3)]
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Non-finite BP smoke loss")

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    encoder_hash = state_dict_checksum(model.encoder)
    z = model.encode(images).detach().flatten(start_dim=1)
    probe = LinearProbe(config["model"]["latent_dim"])
    probe.standardizer.fit(z)
    optimizer = torch.optim.SGD(probe.classifier.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    probe(z).sum().backward()
    optimizer.step()
    if state_dict_checksum(model.encoder) != encoder_hash:
        raise RuntimeError("Frozen encoder changed during probe smoke step")

    print("phase0-v1 synthetic smoke test passed")
    print("shapes", expected)
    print("bp_losses", losses)
    print("model_hash", state_dict_checksum(twin))


if __name__ == "__main__":
    main()
