import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from evaluation.run_hybrid_hhb_confirmation_analysis import (
    evaluate_seed_confirmation,
)
from models import ConvAutoencoder
from schemas import load_config
from training.run_hybrid_hhb_confirmation import (
    validate_confirmation_protocol,
)
from training.train_hybrid import train_hybrid_config
from training.train_standardized_decoder import (
    train_standardized_decoder_config,
)
from utils.reproducibility import state_dict_checksum
from utils.results import read_run_status


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "experiments" / "hybrid_confirmation"
PROTOCOL_PATH = (
    ROOT / "configs" / "experiments" / "hybrid_hhb_confirmation_v1.yaml"
)


def _config(seed: int = 43, method: str = "hybrid_hhb") -> dict:
    config = deepcopy(load_config(CONFIG_ROOT / f"{method}_seed{seed}.yaml"))
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["model"]["latent_dim"] = 4
    config["training"]["hebbian_epochs_per_layer"] = 1
    config["training"]["bp_epochs"] = 1
    return config


def _loaders(*, include_test: bool = False, seed: int = 7):
    generator = torch.Generator().manual_seed(987)
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


def _prepare_source_run(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    model = ConvAutoencoder(config["model"]["latent_dim"], seed=43)
    torch.save(model.state_dict(), path / "model_best.pt")


def test_protocol_is_exactly_two_seeds_and_three_methods():
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_confirmation_protocol(protocol)
    assert protocol["seeds"] == [43, 44]
    assert protocol["methods"] == ["full_bp", "full_hebbian", "hybrid_hhb"]

    invalid = deepcopy(protocol)
    invalid["seeds"].append(45)
    with pytest.raises(ValueError, match="frozen"):
        validate_confirmation_protocol(invalid)

    invalid = deepcopy(protocol)
    invalid["thresholds"]["validation_accuracy_floor"] = 0.80
    with pytest.raises(ValueError, match="thresholds changed"):
        validate_confirmation_protocol(invalid)


def test_confirmation_configs_are_frozen_and_paired():
    for seed in (43, 44):
        configs = [
            load_config(CONFIG_ROOT / f"{method}_seed{seed}.yaml")
            for method in ("full_bp", "full_hebbian", "hybrid_hhb")
        ]
        assert {config["training"]["seed"] for config in configs} == {seed}
        assert {config["backprop"]["lr"] for config in configs} == {0.003}
        assert {config["hebbian"]["lr"] for config in configs} == {0.0005}
        assert {config["hebbian"]["winner_fraction"] for config in configs} == {
            0.1
        }
        assert {
            config["standardized_decoder"]["lr"] for config in configs
        } == {0.003}
        initial_hashes = {
            state_dict_checksum(ConvAutoencoder(64, seed=seed))
            for _ in configs
        }
        assert len(initial_hashes) == 1


def test_stage2d_hybrid_training_accepts_confirmation_seed(tmp_path):
    run_dir = train_hybrid_config(
        _config(),
        run_dir=tmp_path / "hybrid",
        loaders=_loaders(),
    )
    assert read_run_status(run_dir)["status"] == "completed"
    summary = json.loads(
        (run_dir / "hybrid_training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["test_samples_accessed"] == 0
    assert summary["frozen_layers_unchanged"]


def test_standardized_decoder_rejects_test_loader(tmp_path):
    config = _config()
    run_dir = tmp_path / "forbidden"
    _prepare_source_run(run_dir, config)
    with pytest.raises(RuntimeError, match="test loader"):
        train_standardized_decoder_config(
            config,
            run_dir=run_dir,
            loaders=_loaders(include_test=True),
        )


def test_standardized_decoder_resume_is_exact_and_encoder_stays_frozen(tmp_path):
    config = _config()
    full_run = tmp_path / "full"
    resumed_run = tmp_path / "resumed"
    _prepare_source_run(full_run, config)
    _prepare_source_run(resumed_run, config)

    full = train_standardized_decoder_config(
        config,
        run_dir=full_run,
        loaders=_loaders(),
    )
    interrupted = train_standardized_decoder_config(
        config,
        run_dir=resumed_run,
        loaders=_loaders(),
        stop_after_epoch=1,
    )
    assert read_run_status(interrupted)["status"] == "paused"
    resumed = train_standardized_decoder_config(
        config,
        run_dir=resumed_run,
        loaders=_loaders(),
    )

    full_state = torch.load(
        full / "decoder_best.pt", map_location="cpu", weights_only=True
    )
    resumed_state = torch.load(
        resumed / "decoder_best.pt", map_location="cpu", weights_only=True
    )
    assert state_dict_checksum(full_state) == state_dict_checksum(resumed_state)
    summary = json.loads(
        (resumed / "standardized_decoder_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["encoder_unchanged"]
    assert summary["encoder_hash_before"] == summary["encoder_hash_after"]
    assert summary["test_samples_accessed"] == 0
    assert read_run_status(resumed)["resume_count"] == 1


def test_confirmation_gate_passes_and_fails_only_frozen_checks():
    thresholds = {
        "validation_accuracy_floor": 0.8863,
        "standardized_decoder_mse_ratio_to_bp_max": 1.25,
        "z_effective_rank_min": 2.0,
        "z_effective_rank_ratio_to_full_hebbian_min": 2.0,
        "z_to_h2_effective_rank_ratio_min": 2.0,
        "epsilon": 1.0e-12,
    }
    performance = {
        "full_bp": {
            "validation_accuracy": 0.92,
            "system_validation_reconstruction_mse": 0.006,
            "standardized_decoder_validation_mse": 0.006,
        },
        "full_hebbian": {
            "validation_accuracy": 0.90,
            "system_validation_reconstruction_mse": 0.018,
            "standardized_decoder_validation_mse": 0.018,
        },
        "hybrid_hhb": {
            "validation_accuracy": 0.91,
            "system_validation_reconstruction_mse": 0.0062,
            "standardized_decoder_validation_mse": 0.0065,
        },
    }
    ranks = {
        "full_bp": {"h1": 2.0, "h2": 6.0, "z": 12.0},
        "full_hebbian": {"h1": 1.1, "h2": 1.05, "z": 1.02},
        "hybrid_hhb": {"h1": 1.1, "h2": 1.05, "z": 10.0},
    }
    result = evaluate_seed_confirmation(
        seed=43,
        performance=performance,
        ranks=ranks,
        thresholds=thresholds,
        pairing_pass=True,
        numerical_integrity_pass=True,
        zero_test_access=True,
    )
    assert result["decision"] == "PASS"
    assert all(result["checks"].values())

    failed = deepcopy(performance)
    failed["hybrid_hhb"]["validation_accuracy"] = 0.88
    result = evaluate_seed_confirmation(
        seed=43,
        performance=failed,
        ranks=ranks,
        thresholds=thresholds,
        pairing_pass=True,
        numerical_integrity_pass=True,
        zero_test_access=True,
    )
    assert result["decision"] == "FAIL"
    assert not result["checks"]["validation_accuracy_floor"]
