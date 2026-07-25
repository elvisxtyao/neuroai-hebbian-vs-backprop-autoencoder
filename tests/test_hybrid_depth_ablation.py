import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import ConvAutoencoder
from schemas import load_config
from training.train_hybrid import (
    _parameter_manifest,
    train_hybrid_config,
)
from utils.reproducibility import state_dict_checksum
from utils.results import read_run_status


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "hybrid_depth"


def _config(name: str = "hybrid_hhb_seed42.yaml") -> dict:
    config = deepcopy(load_config(CONFIG_ROOT / name))
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["model"]["latent_dim"] = 4
    config["training"]["hebbian_epochs_per_layer"] = 1
    config["training"]["bp_epochs"] = 2
    return config


def _loaders(*, include_test: bool = False, seed: int = 0):
    generator = torch.Generator().manual_seed(123)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8) % 10
    sample_ids = torch.arange(8)
    dataset = TensorDataset(images, labels, sample_ids)
    loaders = {
        "train": DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        "validation": DataLoader(dataset, batch_size=4, shuffle=False),
    }
    if include_test:
        loaders["test"] = DataLoader(dataset, batch_size=4, shuffle=False)
    return loaders


def _checkpoint_hash(path: Path) -> str:
    return state_dict_checksum(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def test_four_preregistered_methods_share_initialization_and_frozen_values():
    names = (
        "full_bp_seed42.yaml",
        "full_hebbian_seed42.yaml",
        "hybrid_hhb_seed42.yaml",
        "hybrid_hbb_seed42.yaml",
    )
    configs = [load_config(CONFIG_ROOT / name) for name in names]
    assert [config["hybrid"]["method_id"] for config in configs] == [
        "full_bp",
        "full_hebbian",
        "hybrid_hhb",
        "hybrid_hbb",
    ]
    assert {config["training"]["seed"] for config in configs} == {42}
    assert {config["backprop"]["lr"] for config in configs} == {0.003}
    assert {config["hebbian"]["lr"] for config in configs} == {0.0005}
    assert {config["hebbian"]["winner_fraction"] for config in configs} == {0.1}
    hashes = {
        state_dict_checksum(ConvAutoencoder(64, seed=42)) for _ in configs
    }
    assert len(hashes) == 1


def test_hhb_bp_parameter_group_contains_only_enc3_and_decoder():
    config = _config()
    model = ConvAutoencoder(4, seed=42)
    manifest, parameters = _parameter_manifest(
        model, config["hybrid"]["encoder_layer_rules"]
    )
    names = manifest["bp_trainable_parameter_names"]
    assert all(name.startswith(("encoder.enc3.", "decoder.")) for name in names)
    assert all(
        name.startswith(("encoder.enc1.", "encoder.enc2."))
        for name in manifest["bp_frozen_parameter_names"]
    )
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_hybrid_rejects_any_test_loader(tmp_path):
    with pytest.raises(RuntimeError, match="test loader"):
        train_hybrid_config(
            _config(),
            run_dir=tmp_path / "forbidden",
            loaders=_loaders(include_test=True),
        )


def test_hhb_resume_is_exact_and_frozen_layers_do_not_change(tmp_path):
    config = _config()
    full = train_hybrid_config(
        config,
        run_dir=tmp_path / "full",
        loaders=_loaders(),
    )
    interrupted = train_hybrid_config(
        config,
        run_dir=tmp_path / "resumed",
        loaders=_loaders(),
        stop_after_global_epoch=3,
    )
    assert read_run_status(interrupted)["status"] == "paused"
    resumed = train_hybrid_config(
        config,
        run_dir=interrupted,
        loaders=_loaders(),
    )

    assert read_run_status(resumed)["status"] == "completed"
    assert read_run_status(resumed)["resume_count"] == 1
    assert _checkpoint_hash(full / "model_last.pt") == _checkpoint_hash(
        resumed / "model_last.pt"
    )
    summary = json.loads(
        (resumed / "hybrid_training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["frozen_layers_unchanged"]
    assert summary["frozen_layer_hashes_before_bp"] == summary[
        "frozen_layer_hashes_after_bp"
    ]
    assert summary["test_samples_accessed"] == 0
