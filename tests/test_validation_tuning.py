import csv
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from schemas import load_config
import training.train_linear_probe as probe_training
from training.run_validation_tuning import ValidationTuner
from training.train_linear_probe import train_linear_probe_config
from training.train_representation import train_config
from utils.checkpointing import config_fingerprint


ROOT = Path(__file__).resolve().parents[1]


class FailIfRead(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        raise AssertionError("validation-only tuning accessed the test dataset")


class FailUntilValidation(Dataset):
    def __init__(self, ready, images, labels, sample_ids):
        self.ready = ready
        self.samples = TensorDataset(images, labels, sample_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not self.ready["value"]:
            raise AssertionError("formal probe accessed test before validation selection")
        return self.samples[index]


def _synthetic_loaders():
    generator = torch.Generator().manual_seed(73)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8)
    sample_ids = torch.arange(8)
    dataset = TensorDataset(images, labels, sample_ids)
    return {
        "train": DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        ),
        "validation": DataLoader(dataset, batch_size=4),
        "test": DataLoader(FailIfRead(), batch_size=4),
    }


def test_formal_probe_defers_test_until_after_validation_selection(
    tmp_path, monkeypatch
):
    config = deepcopy(load_config(ROOT / "configs" / "bp_main.yaml"))
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["training"]["bp_epochs"] = 1
    config["training"]["seed"] = 42
    config["model"]["latent_dim"] = 4
    config["probe"]["epochs"] = 2

    generator = torch.Generator().manual_seed(73)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8)
    sample_ids = torch.arange(8)
    train_validation = TensorDataset(images, labels, sample_ids)
    validation_complete = {"value": False}
    loaders = {
        "train": DataLoader(train_validation, batch_size=4),
        "validation": DataLoader(train_validation, batch_size=4),
        "test": DataLoader(
            FailUntilValidation(
                validation_complete,
                images,
                labels,
                sample_ids,
            ),
            batch_size=4,
        ),
    }
    run_dir = train_config(config, loaders=loaders, run_root=tmp_path)

    original_evaluate = probe_training._evaluate_probe

    def record_validation(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        validation_complete["value"] = True
        return metrics

    monkeypatch.setattr(probe_training, "_evaluate_probe", record_validation)
    train_linear_probe_config(config, run_dir, validation_only=False, loaders=loaders)

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(
        row["stage"] == "linear_probe_final" and row["split"] == "test"
        for row in rows
    )


def test_validation_only_probe_never_reads_or_records_test_split(tmp_path):
    config = deepcopy(load_config(ROOT / "configs" / "bp_main.yaml"))
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["training"]["bp_epochs"] = 1
    config["training"]["seed"] = 42
    config["model"]["latent_dim"] = 4
    config["probe"]["epochs"] = 2
    loaders = _synthetic_loaders()
    run_dir = train_config(config, loaders=loaders, run_root=tmp_path)

    train_linear_probe_config(
        config,
        run_dir,
        validation_only=True,
        loaders=loaders,
    )

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert not any(row["split"] == "test" for row in rows)
    assert any(
        row["stage"] == "linear_probe_final" and row["split"] == "validation"
        for row in rows
    )


def test_tuning_manifest_has_balanced_eight_trial_budget_and_shared_main_shape(
    tmp_path,
):
    manifest = ROOT / "configs" / "tuning" / "validation_tuning_v1.yaml"
    tuner = ValidationTuner(manifest, tmp_path / "tuning")
    hebbian = tuner.manifest["hebbian"]
    lr_configs = [
        tuner.hebbian_config(
            lr=lr,
            winner=hebbian["fixed_winner_fraction_for_lr"],
            latent_dim=hebbian["fixed_latent_dim"],
        )
        for lr in hebbian["learning_rates"]
    ]
    winner_configs = [
        tuner.hebbian_config(lr=0.0005, winner=winner, latent_dim=64)
        for winner in hebbian["winner_fractions"]
    ]
    latent_configs = [
        tuner.hebbian_config(lr=0.0005, winner=0.10, latent_dim=latent)
        for latent in hebbian["coarse_latent_dims"]
    ]
    hebbian_unique = {
        config_fingerprint(config)
        for config in (*lr_configs, *winner_configs, *latent_configs)
    }
    bp_unique = {
        config_fingerprint(tuner.bp_config(lr=lr, weight_decay=weight_decay))
        for lr in tuner.manifest["backprop"]["learning_rates"]
        for weight_decay in tuner.manifest["backprop"]["weight_decays"]
    }

    assert len(hebbian_unique) == 8
    assert len(bp_unique) == 8
    selected_hebbian = load_config(
        ROOT / "configs" / "selected" / "hebbian_validation_selected.yaml"
    )
    selected_bp = load_config(
        ROOT / "configs" / "selected" / "bp_validation_selected.yaml"
    )
    assert selected_hebbian["model"] == selected_bp["model"]
