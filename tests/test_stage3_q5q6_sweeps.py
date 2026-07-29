from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from models import ConvAutoencoder, autoencoder_from_config
from evaluation.run_stage3_q5q6_test import _summarize
from schemas import ConfigError, validate_config
from training.run_stage3_q5q6_sweeps import (
    METHODS,
    SEEDS,
    resolved_sweep_config,
    validate_protocol,
)
from training.train_hybrid import train_hybrid_config
from utils.reproducibility import state_dict_checksum
from utils.results import read_run_status


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "configs" / "experiments" / "stage3_q5q6_sweeps_v1.yaml"
)


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _loaders(seed: int = 0):
    generator = torch.Generator().manual_seed(777)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8) % 10
    sample_ids = torch.arange(8)
    dataset = TensorDataset(images, labels, sample_ids)
    return {
        "train": DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        "validation": DataLoader(dataset, batch_size=4, shuffle=False),
    }


def _tiny_config(case: str = "early_heavy", method: str = "hybrid_hhb") -> dict:
    config = deepcopy(
        resolved_sweep_config(
            _protocol(),
            sweep="architecture",
            case=case,
            method=method,
            seed=0,
        )
    )
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["model"]["latent_dim"] = 4
    config["training"]["hebbian_epochs_per_layer"] = 1
    config["training"]["bp_epochs"] = 1
    return config


def test_protocol_freezes_cases_methods_seeds_and_parameter_matching():
    protocol = _protocol()
    validate_protocol(protocol)
    assert tuple(protocol["seeds"]) == SEEDS
    assert tuple(protocol["methods"]) == METHODS
    assert {
        key: value["latent_dim"]
        for key, value in protocol["dimension_cases"].items()
    } == {"L16": 16, "L32": 32, "L64": 64, "L128": 128}
    assert {
        key: value["encoder_channels"]
        for key, value in protocol["architecture_cases"].items()
    } == {
        "early_heavy": [64, 28],
        "balanced": [16, 32],
        "late_heavy": [4, 33],
    }
    counts = [
        value["expected_encoder_parameters"]
        for value in protocol["architecture_cases"].values()
    ]
    assert (max(counts) - min(counts)) / (sum(counts) / len(counts)) < 0.01


def test_reused_balanced_cases_cannot_be_resolved_for_retraining():
    protocol = _protocol()
    with pytest.raises(ValueError, match="reused"):
        resolved_sweep_config(
            protocol,
            sweep="dimension",
            case="L64",
            method="full_bp",
            seed=0,
        )
    with pytest.raises(ValueError, match="reused"):
        resolved_sweep_config(
            protocol,
            sweep="architecture",
            case="balanced",
            method="full_bp",
            seed=0,
        )


def test_sweep_configs_preserve_rules_learning_rates_and_validation_only_marker():
    protocol = _protocol()
    configs = [
        resolved_sweep_config(
            protocol,
            sweep="architecture",
            case="early_heavy",
            method=method,
            seed=seed,
        )
        for seed in SEEDS
        for method in METHODS
    ]
    assert {config["training"]["seed"] for config in configs} == set(SEEDS)
    assert {config["backprop"]["lr"] for config in configs} == {0.003}
    assert {config["hebbian"]["lr"] for config in configs} == {0.0005}
    assert {config["hebbian"]["winner_fraction"] for config in configs} == {
        0.10
    }
    assert {
        config["hybrid"]["confirmation_stage"] for config in configs
    } == {"stage3_sweep"}
    assert {tuple(config["model"]["encoder_channels"]) for config in configs} == {
        (64, 28)
    }


def test_nonbalanced_widths_require_formal_sweep_marker():
    config = resolved_sweep_config(
        _protocol(),
        sweep="architecture",
        case="late_heavy",
        method="hybrid_hbb",
        seed=0,
    )
    changed = deepcopy(config)
    changed["hybrid"]["confirmation_stage"] = "stage3_core"
    with pytest.raises(ConfigError, match="reserved"):
        validate_config(changed)


@pytest.mark.parametrize(
    ("channels", "expected_encoder", "expected_total"),
    [
        ([64, 28], 104_512, 222_109),
        ([16, 32], 105_104, 213_953),
        ([4, 33], 104_712, 210_414),
    ],
)
def test_configurable_width_shapes_and_parameter_counts(
    channels, expected_encoder, expected_total
):
    model = ConvAutoencoder(
        latent_dim=64,
        encoder_channels=channels,
        seed=0,
    )
    features = model.encode(
        torch.zeros(2, 1, 28, 28), return_all_layers=True
    )
    assert tuple(features["h1"].shape[1:]) == (channels[0], 14, 14)
    assert tuple(features["h2"].shape[1:]) == (channels[1], 7, 7)
    assert tuple(features["z"].shape[1:]) == (64, 1, 1)
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 1, 28, 28)
    metadata = model.architecture_metadata()
    assert metadata["encoder_parameter_count"] == expected_encoder
    assert metadata["parameter_count"] == expected_total


def test_factory_preserves_default_checkpoint_compatibility():
    config = resolved_sweep_config(
        _protocol(),
        sweep="dimension",
        case="L32",
        method="full_bp",
        seed=0,
    )
    first = autoencoder_from_config(config, seed=0)
    second = ConvAutoencoder(32, encoder_channels=[16, 32], seed=0)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert state_dict_checksum(first) == state_dict_checksum(second)


def test_nonbalanced_hybrid_training_resume_is_exact(tmp_path):
    config = _tiny_config()
    full = train_hybrid_config(
        config,
        run_dir=tmp_path / "full",
        loaders=_loaders(),
    )
    interrupted = train_hybrid_config(
        config,
        run_dir=tmp_path / "resumed",
        loaders=_loaders(),
        stop_after_global_epoch=2,
    )
    assert read_run_status(interrupted)["status"] == "paused"
    resumed = train_hybrid_config(
        config,
        run_dir=interrupted,
        loaders=_loaders(),
    )
    assert read_run_status(resumed)["status"] == "completed"
    full_state = torch.load(
        full / "model_best.pt", map_location="cpu", weights_only=True
    )
    resumed_state = torch.load(
        resumed / "model_best.pt", map_location="cpu", weights_only=True
    )
    assert state_dict_checksum(full_state) == state_dict_checksum(resumed_state)


def test_sweep_test_summary_uses_paired_seed_differences():
    rows = []
    for seed in SEEDS:
        for method_index, method in enumerate(METHODS):
            value = float(seed + method_index)
            rows.append(
                {
                    "seed": seed,
                    "method_id": method,
                    "accuracy": value,
                    "macro_f1": value,
                    "classification_ce": value,
                    "system_reconstruction_mse": value,
                    "standardized_reconstruction_mse": value,
                }
            )
    summary = _summarize(rows)
    assert (
        summary["paired_contrasts"]["HHB_minus_HHH"]["accuracy"][
            "paired_differences"
        ]
        == [1.0] * 5
    )
