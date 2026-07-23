"""Evaluate reconstruction MSE and save an original/reconstruction grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from data.mnist import build_mnist_dataloaders
from models import ConvAutoencoder
from schemas import load_config, validate_config
from utils.results import append_metric


@torch.no_grad()
def evaluate_config(
    config: dict,
    run_dir: str | Path,
    *,
    num_images: int = 10,
    loaders=None,
) -> Path:
    validate_config(config)
    run_dir = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(config["model"]["latent_dim"])
    model.load_state_dict(
        torch.load(run_dir / "model_best.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    if loaders is None:
        loaders = build_mnist_dataloaders(
            config, seed=int(config["training"]["seed"]), download=False
        )
    loader = loaders["test"]
    criterion = nn.MSELoss(reduction="sum")
    squared_error = 0.0
    num_pixels = 0
    grid_items: list[torch.Tensor] = []
    for images, _, _ in loader:
        images = images.to(device)
        reconstructions = model.reconstruct(images)
        squared_error += float(criterion(reconstructions, images).item())
        num_pixels += images.numel()
        if not grid_items:
            for original, reconstructed in zip(
                images[:num_images].cpu(), reconstructions[:num_images].cpu(), strict=True
            ):
                grid_items.extend([original, reconstructed])

    test_mse = squared_error / num_pixels
    output_path = run_dir / "reconstructions_original_then_reconstructed.png"
    save_image(torch.stack(grid_items), output_path, nrow=2, padding=2, pad_value=1.0)
    append_metric(
        run_dir,
        {
            "stage": "reconstruction_final",
            "split": "test",
            "epoch": config["training"]["bp_epochs"],
            "reconstruction_loss": test_mse,
            "num_samples": len(loader.dataset),
        },
    )
    print(f"test_mse={test_mse:.8f}")
    print(output_path.resolve())
    return output_path


def evaluate(
    config_path: str | Path, run_dir: str | Path, *, num_images: int = 10
) -> Path:
    return evaluate_config(load_config(config_path), run_dir, num_images=num_images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-images", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.config, args.run_dir, num_images=args.num_images)


if __name__ == "__main__":
    main()
