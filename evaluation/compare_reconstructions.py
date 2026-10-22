"""Deterministic BP/Hebbian reconstruction sanity check.

This module performs inference only. It never updates a checkpoint or appends
to either source run's metrics file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.mnist import build_mnist_dataloaders
from models import ConvAutoencoder
from schemas import load_config
from utils.reproducibility import state_dict_checksum


MODEL_NAMES = ("bp", "hebbian", "untrained")


def validate_comparable_configs(bp_config: dict, hebbian_config: dict) -> None:
    """Reject comparisons that do not share the frozen data/model contract."""

    if bp_config["training"]["learning_rule"] != "bp":
        raise ValueError("BP run config does not use learning_rule=bp")
    if hebbian_config["training"]["learning_rule"] != "hebbian":
        raise ValueError("Hebbian run config does not use learning_rule=hebbian")
    if bp_config["data"] != hebbian_config["data"]:
        raise ValueError("BP and Hebbian runs do not share the same data config")
    if bp_config["model"] != hebbian_config["model"]:
        raise ValueError("BP and Hebbian runs do not share the same model config")
    if bp_config["training"]["seed"] != hebbian_config["training"]["seed"]:
        raise ValueError("BP and Hebbian runs do not use the same model seed")


def _as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def select_stratified_samples(
    dataset,
    *,
    samples_per_class: int,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select deterministic sample IDs, grouped by class for visualization."""

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    candidates: dict[int, list[int]] = defaultdict(list)
    for position in range(len(dataset)):
        _, label, _ = dataset[position]
        label_value = _as_int(label)
        if 0 <= label_value < num_classes:
            candidates[label_value].append(position)

    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    for label in range(num_classes):
        class_positions = candidates[label]
        if len(class_positions) < samples_per_class:
            raise ValueError(
                f"Class {label} has {len(class_positions)} samples; "
                f"need {samples_per_class}"
            )
        chosen = rng.choice(class_positions, size=samples_per_class, replace=False)
        selected_positions.extend(sorted(int(position) for position in chosen))

    images: list[torch.Tensor] = []
    labels: list[int] = []
    sample_ids: list[int] = []
    for position in selected_positions:
        image, label, sample_id = dataset[position]
        images.append(torch.as_tensor(image).float())
        labels.append(_as_int(label))
        sample_ids.append(_as_int(sample_id))
    return (
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sample_ids, dtype=torch.long),
    )


def load_trained_model(config: dict, run_dir: Path, device: torch.device):
    checkpoint = run_dir / "model_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model = ConvAutoencoder(config["model"]["latent_dim"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def evaluate_model(model, loader, *, device: torch.device) -> dict[str, Any]:
    """Evaluate reconstruction metrics and inference-only safety gates."""

    model.eval()
    checksum_before = state_dict_checksum(model)
    squared_error = 0.0
    absolute_error = 0.0
    num_pixels = 0
    num_samples = 0
    output_sum: torch.Tensor | None = None
    output_square_sum: torch.Tensor | None = None
    output_min = float("inf")
    output_max = float("-inf")
    all_finite = True
    shape_ok = True
    class_squared_error = np.zeros(10, dtype=np.float64)
    class_absolute_error = np.zeros(10, dtype=np.float64)
    class_pixels = np.zeros(10, dtype=np.int64)

    for images, labels, _ in loader:
        images = images.to(device)
        reconstructions = model.reconstruct(images)
        shape_ok = shape_ok and reconstructions.shape == images.shape
        all_finite = all_finite and bool(torch.isfinite(reconstructions).all().item())
        output_min = min(output_min, float(reconstructions.min().item()))
        output_max = max(output_max, float(reconstructions.max().item()))

        difference = reconstructions - images
        per_sample_squared = difference.square().flatten(start_dim=1).sum(dim=1)
        per_sample_absolute = difference.abs().flatten(start_dim=1).sum(dim=1)
        squared_error += float(per_sample_squared.sum().item())
        absolute_error += float(per_sample_absolute.sum().item())
        num_pixels += images.numel()
        num_samples += images.shape[0]

        batch_output = reconstructions.detach().cpu().to(torch.float64)
        batch_sum = batch_output.sum(dim=0)
        batch_square_sum = batch_output.square().sum(dim=0)
        output_sum = batch_sum if output_sum is None else output_sum + batch_sum
        output_square_sum = (
            batch_square_sum
            if output_square_sum is None
            else output_square_sum + batch_square_sum
        )

        for label in range(10):
            mask = labels == label
            count = int(mask.sum().item())
            if count:
                class_squared_error[label] += float(per_sample_squared[mask].sum().item())
                class_absolute_error[label] += float(per_sample_absolute[mask].sum().item())
                class_pixels[label] += count * images[0].numel()

    if num_samples == 0 or output_sum is None or output_square_sum is None:
        raise ValueError("Evaluation loader is empty")
    mean_output = output_sum / num_samples
    sample_variance = (output_square_sum / num_samples - mean_output.square()).clamp_min(0)
    checksum_after = state_dict_checksum(model)
    per_class = {
        str(label): {
            "mse": float(class_squared_error[label] / class_pixels[label]),
            "mae": float(class_absolute_error[label] / class_pixels[label]),
            "num_pixels": int(class_pixels[label]),
        }
        for label in range(10)
        if class_pixels[label] > 0
    }
    return {
        "mse": squared_error / num_pixels,
        "mae": absolute_error / num_pixels,
        "num_samples": num_samples,
        "num_pixels": num_pixels,
        "output_min": output_min,
        "output_max": output_max,
        "mean_pixel_variance_across_samples": float(sample_variance.mean().item()),
        "all_finite": all_finite,
        "shape_ok": shape_ok,
        "output_in_unit_interval": output_min >= -1e-7 and output_max <= 1 + 1e-7,
        "nonconstant_across_samples": float(sample_variance.mean().item()) > 1e-8,
        "checksum_before": checksum_before,
        "checksum_after": checksum_after,
        "checksum_unchanged": checksum_before == checksum_after,
        "per_class": per_class,
    }


@torch.no_grad()
def reconstruct_selected(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return model.reconstruct(images.to(device)).detach().cpu()


def _recorded_test_mse(run_dir: Path) -> float | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    matches: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("stage") == "reconstruction_final"
                and row.get("split") == "test"
                and row.get("reconstruction_loss")
            ):
                matches.append(float(row["reconstruction_loss"]))
    return matches[-1] if matches else None


def _tensor_to_pil(tensor: torch.Tensor, *, scale: float = 1.0, size: int = 112):
    array = tensor.detach().cpu().squeeze().clamp(0, 1).numpy()
    array = np.uint8(np.clip(array * scale, 0, 1) * 255)
    return Image.fromarray(array, mode="L").resize((size, size), Image.Resampling.NEAREST)


def save_comparison_grid(
    output_path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    bp_reconstructions: torch.Tensor,
    hebbian_reconstructions: torch.Tensor,
) -> None:
    cell = 112
    gap = 8
    left = 112
    header = 34
    columns = ("Original", "BP", "Hebbian", "4x BP error", "4x Hebb error")
    width = left + len(columns) * (cell + gap) + gap
    height = header + len(images) * (cell + gap) + gap
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, title in enumerate(columns):
        x = left + gap + column * (cell + gap)
        draw.text((x, 10), title, fill=0, font=font)

    for row in range(len(images)):
        y = header + row * (cell + gap)
        draw.text(
            (8, y + cell // 2 - 6),
            f"digit={int(labels[row])} id={int(sample_ids[row])}",
            fill=0,
            font=font,
        )
        bp_error = (images[row] - bp_reconstructions[row]).abs()
        hebbian_error = (images[row] - hebbian_reconstructions[row]).abs()
        panels = (
            _tensor_to_pil(images[row], size=cell),
            _tensor_to_pil(bp_reconstructions[row], size=cell),
            _tensor_to_pil(hebbian_reconstructions[row], size=cell),
            _tensor_to_pil(bp_error, scale=4.0, size=cell),
            _tensor_to_pil(hebbian_error, scale=4.0, size=cell),
        )
        for column, panel in enumerate(panels):
            x = left + gap + column * (cell + gap)
            canvas.paste(panel, (x, y))
    canvas.save(output_path)


def _create_output_directory(path: Path) -> Path:
    candidate = path
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_sanity_check(
    *,
    bp_run: str | Path,
    hebbian_run: str | Path,
    output_dir: str | Path,
    samples_per_class: int = 2,
    sample_seed: int = 23,
) -> Path:
    bp_run = Path(bp_run)
    hebbian_run = Path(hebbian_run)
    bp_config = load_config(bp_run / "config_resolved.yaml")
    hebbian_config = load_config(hebbian_run / "config_resolved.yaml")
    validate_comparable_configs(bp_config, hebbian_config)
    output_dir = _create_output_directory(Path(output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_mnist_dataloaders(
        bp_config,
        seed=int(bp_config["training"]["seed"]),
        download=False,
    )
    selected_images, selected_labels, selected_ids = select_stratified_samples(
        loaders["test"].dataset,
        samples_per_class=samples_per_class,
        seed=sample_seed,
    )
    models = {
        "bp": load_trained_model(bp_config, bp_run, device),
        "hebbian": load_trained_model(hebbian_config, hebbian_run, device),
        "untrained": ConvAutoencoder(
            bp_config["model"]["latent_dim"],
            seed=int(bp_config["training"]["seed"]),
        ).to(device).eval(),
    }

    full_metrics = {
        name: evaluate_model(model, loaders["test"], device=device)
        for name, model in models.items()
    }
    selected_reconstructions = {
        name: reconstruct_selected(model, selected_images, device)
        for name, model in models.items()
    }
    save_comparison_grid(
        output_dir / "reconstruction_grid.png",
        selected_images,
        selected_labels,
        selected_ids,
        selected_reconstructions["bp"],
        selected_reconstructions["hebbian"],
    )

    manifest = {
        "sample_seed": sample_seed,
        "samples_per_class": samples_per_class,
        "sample_ids": selected_ids.tolist(),
        "labels": selected_labels.tolist(),
        "row_layout": [
            "original",
            "bp_reconstruction",
            "hebbian_reconstruction",
            "4x_bp_absolute_error",
            "4x_hebbian_absolute_error",
        ],
    }
    with (output_dir / "sample_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    sample_rows: list[dict[str, Any]] = []
    for index, (label, sample_id) in enumerate(zip(selected_labels, selected_ids)):
        for name in MODEL_NAMES:
            difference = selected_reconstructions[name][index] - selected_images[index]
            sample_rows.append(
                {
                    "sample_id": int(sample_id),
                    "label": int(label),
                    "model": name,
                    "mse": float(difference.square().mean().item()),
                    "mae": float(difference.abs().mean().item()),
                }
            )
    with (output_dir / "sample_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "label", "model", "mse", "mae")
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    metric_rows: list[dict[str, Any]] = []
    for name, metrics in full_metrics.items():
        metric_rows.append(
            {
                "model": name,
                "scope": "overall",
                "label": "all",
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "num_samples": metrics["num_samples"],
            }
        )
        for label, class_metrics in metrics["per_class"].items():
            metric_rows.append(
                {
                    "model": name,
                    "scope": "class",
                    "label": label,
                    "mse": class_metrics["mse"],
                    "mae": class_metrics["mae"],
                    "num_samples": class_metrics["num_pixels"] // (28 * 28),
                }
            )
    with (output_dir / "reconstruction_metrics.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "scope", "label", "mse", "mae", "num_samples"),
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    recorded = {
        "bp": _recorded_test_mse(bp_run),
        "hebbian": _recorded_test_mse(hebbian_run),
    }
    recorded_match = {
        name: value is None or abs(full_metrics[name]["mse"] - value) <= 1e-7
        for name, value in recorded.items()
    }
    trained_better_than_untrained = {
        name: full_metrics[name]["mse"] < full_metrics["untrained"]["mse"]
        for name in ("bp", "hebbian")
    }
    safety_pass = {
        name: all(
            (
                metrics["all_finite"],
                metrics["shape_ok"],
                metrics["output_in_unit_interval"],
                metrics["nonconstant_across_samples"],
                metrics["checksum_unchanged"],
            )
        )
        for name, metrics in full_metrics.items()
    }
    bp_hebbian_mean_absolute_difference = float(
        (selected_reconstructions["bp"] - selected_reconstructions["hebbian"])
        .abs()
        .mean()
        .item()
    )
    summary = {
        "protocol": {
            "inference_only": True,
            "device": str(device),
            "bp_run": str(bp_run),
            "hebbian_run": str(hebbian_run),
            "checkpoint": "model_best.pt",
            "comparison_note": (
                "BP is a jointly trained autoencoder; Hebbian uses a locally trained "
                "encoder followed by a frozen-encoder BP-trained decoder."
            ),
        },
        "metrics": full_metrics,
        "recorded_test_mse": recorded,
        "recorded_mse_matches": recorded_match,
        "trained_better_than_untrained": trained_better_than_untrained,
        "safety_pass": safety_pass,
        "bp_hebbian_selected_mean_absolute_difference": (
            bp_hebbian_mean_absolute_difference
        ),
        "all_automatic_gates_pass": (
            all(safety_pass.values())
            and all(recorded_match.values())
            and all(trained_better_than_untrained.values())
            and bp_hebbian_mean_absolute_difference > 1e-8
        ),
        "qualitative_gate": (
            "Human inspection required: digits should remain recognizable and error "
            "patterns should not indicate constant or input-independent output."
        ),
    }
    with (output_dir / "reconstruction_summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"output_dir={output_dir.resolve()}")
    for name in MODEL_NAMES:
        metrics = full_metrics[name]
        print(
            f"{name}: mse={metrics['mse']:.8f} mae={metrics['mae']:.8f} "
            f"variance={metrics['mean_pixel_variance_across_samples']:.8f} "
            f"safety_pass={safety_pass[name]}"
        )
    print(f"all_automatic_gates_pass={summary['all_automatic_gates_pass']}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp-run", required=True)
    parser.add_argument("--hebbian-run", required=True)
    parser.add_argument("--output-dir", default="results/reconstruction_sanity/seed0")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=23)
    args = parser.parse_args()
    run_sanity_check(
        bp_run=args.bp_run,
        hebbian_run=args.hebbian_run,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
