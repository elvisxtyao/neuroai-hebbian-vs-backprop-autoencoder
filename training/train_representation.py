"""Train BP or greedy layer-wise Hebbian representations."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch
from torch import nn

from data.mnist import build_mnist_dataloaders
from learning_rules import build_trainer
from models import ConvAutoencoder
from schemas import load_config
from utils.reproducibility import set_global_seed, state_dict_checksum
from utils.results import (
    append_metric,
    create_run_directory,
    write_json,
    write_metadata,
    write_resolved_config,
)


def _train_bp(model, trainer, loaders, config: dict, run_dir: Path) -> None:
    best_validation_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, config["training"]["bp_epochs"] + 1):
        train_metrics = trainer.run_epoch(loaders["train"], training=True)
        validation_metrics = trainer.run_epoch(loaders["validation"], training=False)
        for split, metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            append_metric(
                run_dir,
                {
                    "stage": "representation",
                    "split": split,
                    "epoch": epoch,
                    "global_epoch": epoch,
                    "reconstruction_loss": metrics.loss,
                    "num_samples": metrics.num_samples,
                },
            )
        torch.save(model.state_dict(), run_dir / "model_last.pt")
        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        print(
            f"epoch={epoch:02d} train_mse={train_metrics.loss:.6f} "
            f"validation_mse={validation_metrics.loss:.6f}",
            flush=True,
        )
    write_json(
        run_dir,
        "training_summary.json",
        {"best_validation_mse": best_validation_loss, "best_epoch": best_epoch},
    )


@torch.no_grad()
def _decoder_validation(model, loader, criterion, device: torch.device) -> tuple[float, int]:
    model.encoder.eval()
    model.decoder.eval()
    total_loss = 0.0
    total_samples = 0
    for images, _, _ in loader:
        images = images.to(device)
        latent = model.encode(images)
        reconstruction = model.decode(latent)
        loss = criterion(reconstruction, images)
        total_loss += float(loss.item()) * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / total_samples, total_samples


def _train_frozen_decoder(model, loaders, config: dict, run_dir: Path, device) -> dict:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)
    encoder_hash_before = state_dict_checksum(model.encoder)
    bp = config["backprop"]
    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=bp["lr"],
        betas=tuple(bp["betas"]),
        weight_decay=bp["weight_decay"],
    )
    criterion = nn.MSELoss(reduction="mean")
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs = config["training"]["decoder_epochs"]
    for epoch in range(1, epochs + 1):
        model.encoder.eval()
        model.decoder.train()
        total_loss = 0.0
        total_samples = 0
        for images, _, _ in loaders["train"]:
            images = images.to(device)
            with torch.no_grad():
                latent = model.encode(images)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model.decode(latent)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * images.shape[0]
            total_samples += images.shape[0]
        train_loss = total_loss / total_samples
        validation_loss, validation_samples = _decoder_validation(
            model, loaders["validation"], criterion, device
        )
        for split, loss_value, sample_count in (
            ("train", train_loss, total_samples),
            ("validation", validation_loss, validation_samples),
        ):
            append_metric(
                run_dir,
                {
                    "stage": "decoder",
                    "split": split,
                    "epoch": epoch,
                    "global_epoch": 3 * config["training"]["hebbian_epochs_per_layer"]
                    + epoch,
                    "reconstruction_loss": loss_value,
                    "num_samples": sample_count,
                },
            )
        torch.save(model.state_dict(), run_dir / "model_last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        print(
            f"decoder epoch={epoch:02d} train_mse={train_loss:.6f} "
            f"validation_mse={validation_loss:.6f}",
            flush=True,
        )

    encoder_hash_after = state_dict_checksum(model.encoder)
    if encoder_hash_after != encoder_hash_before:
        raise RuntimeError("Hebbian encoder changed during frozen decoder training")
    return {
        "best_decoder_epoch": best_epoch,
        "best_decoder_validation_mse": best_validation_loss,
        "encoder_hash_before_decoder": encoder_hash_before,
        "encoder_hash_after_decoder": encoder_hash_after,
        "encoder_unchanged_during_decoder_training": True,
    }


def _train_hebbian(model, trainer, loaders, config: dict, run_dir: Path, device) -> None:
    epochs_per_layer = config["training"]["hebbian_epochs_per_layer"]
    global_epoch = 0
    layer_summaries: dict[str, dict] = {}
    for layer_name in trainer.layer_names:
        for epoch in range(1, epochs_per_layer + 1):
            global_epoch += 1
            diagnostics = trainer.train_layer_epoch(loaders["train"], layer_name)
            append_metric(
                run_dir,
                {
                    "stage": "hebbian_encoder",
                    "split": "train",
                    "layer": layer_name,
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "update_norm": diagnostics.update_norm,
                    "weight_norm_mean": diagnostics.weight_norm_mean,
                    "weight_norm_std": diagnostics.weight_norm_std,
                    "activation_mean": diagnostics.activation_mean,
                    "activation_sparsity": diagnostics.activation_sparsity,
                    "active_neuron_ratio": diagnostics.active_neuron_ratio,
                    "winner_entropy": diagnostics.winner_entropy,
                    "num_samples": diagnostics.num_samples,
                },
            )
            print(
                f"layer={layer_name} epoch={epoch:02d} "
                f"update_norm={diagnostics.update_norm:.6f} "
                f"activation={diagnostics.activation_mean:.6f} "
                f"sparsity={diagnostics.activation_sparsity:.4f} "
                f"active={diagnostics.active_neuron_ratio:.4f} "
                f"winner_entropy={diagnostics.winner_entropy:.4f}",
                flush=True,
            )
        checkpoint_path = run_dir / f"encoder_{layer_name}_end.pt"
        torch.save(model.encoder.state_dict(), checkpoint_path)
        layer_summaries[layer_name] = {
            "checkpoint": checkpoint_path.name,
            "encoder_hash": state_dict_checksum(model.encoder),
            "last_epoch": {
                "update_norm": diagnostics.update_norm,
                "activation_mean": diagnostics.activation_mean,
                "activation_sparsity": diagnostics.activation_sparsity,
                "active_neuron_ratio": diagnostics.active_neuron_ratio,
                "winner_entropy": diagnostics.winner_entropy,
            },
        }

    torch.save(model.encoder.state_dict(), run_dir / "encoder_hebbian.pt")
    encoder_hash = state_dict_checksum(model.encoder)
    decoder_summary = _train_frozen_decoder(model, loaders, config, run_dir, device)
    write_json(
        run_dir,
        "hebbian_training_summary.json",
        {
            "rule": "competitive_oja_topk",
            "winner_fraction": config["hebbian"]["winner_fraction"],
            "learning_rate": config["hebbian"]["lr"],
            "layer_lrs": config["hebbian"].get("layer_lrs", {}),
            "epochs_per_layer": epochs_per_layer,
            "layer_order": list(trainer.layer_names),
            "final_encoder_hash": encoder_hash,
            "layers": layer_summaries,
            **decoder_summary,
        },
    )


def train(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed = int(config["training"]["seed"])
    rule = config["training"]["learning_rule"]
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_mnist_dataloaders(config, seed=seed)
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
    initial_hash = state_dict_checksum(model)
    trainer = build_trainer(model, config, device)

    run_dir = create_run_directory(config["results"]["root"], rule=rule, seed=seed)
    write_resolved_config(run_dir, config)
    write_metadata(
        run_dir,
        {
            "version": config["version"],
            "learning_rule": rule,
            "seed": seed,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "initial_state_hash": initial_hash,
            **model.architecture_metadata(),
        },
    )

    if rule == "bp":
        _train_bp(model, trainer, loaders, config, run_dir)
    elif rule == "hebbian":
        _train_hebbian(model, trainer, loaders, config, run_dir, device)
    else:
        raise ValueError(f"Unsupported learning rule: {rule}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train(args.config).resolve())


if __name__ == "__main__":
    main()
