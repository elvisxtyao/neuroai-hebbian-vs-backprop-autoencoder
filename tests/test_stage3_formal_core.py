from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from schemas import ConfigError
from training.run_stage3_formal_core import (
    EXPECTED_METHODS,
    EXPECTED_SEEDS,
    _csv_values_finite,
    resolved_method_config,
    validate_stage3_protocol,
)
from training.train_hybrid import train_hybrid_config
from utils.reproducibility import state_dict_checksum
from utils.results import read_run_status


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "stage3_formal_core_v1.yaml"


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _loaders(seed: int = 0, *, include_test: bool = False):
    generator = torch.Generator().manual_seed(1234)
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


def _tiny_config(method: str, seed: int = 0) -> dict:
    config = deepcopy(
        resolved_method_config(_protocol(), method=method, seed=seed)
    )
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["model"]["latent_dim"] = 4
    config["training"]["hebbian_epochs_per_layer"] = 1
    config["training"]["bp_epochs"] = 1
    return config


def test_stage3_protocol_freezes_matrix_and_candidate_semantics():
    protocol = _protocol()
    validate_stage3_protocol(protocol)
    assert tuple(protocol["seeds"]) == EXPECTED_SEEDS
    assert tuple(protocol["methods"]) == EXPECTED_METHODS
    assert "thresholds" not in protocol
    assert (
        protocol["candidate_semantics"]["hybrid_hhb"]
        == "confirmed_rank_repair_candidate_with_unresolved_reconstruction_stability"
    )
    assert (
        protocol["candidate_semantics"]["standardized_reconstruction_role"]
        == "formal_outcome_not_entry_gate"
    )

    changed = deepcopy(protocol)
    changed["seeds"].append(5)
    with pytest.raises(ValueError, match="frozen"):
        validate_stage3_protocol(changed)

    changed = deepcopy(protocol)
    changed["candidate_semantics"]["standardized_reconstruction_role"] = (
        "entry_gate"
    )
    with pytest.raises(ValueError, match="outcome"):
        validate_stage3_protocol(changed)


def test_stage3_resolved_configs_are_formal_paired_and_rule_exact():
    protocol = _protocol()
    for seed in EXPECTED_SEEDS:
        configs = {
            method: resolved_method_config(
                protocol,
                method=method,
                seed=seed,
            )
            for method in EXPECTED_METHODS
        }
        assert {config["training"]["seed"] for config in configs.values()} == {
            seed
        }
        assert {config["version"] for config in configs.values()} == {
            "phase0-v1.1"
        }
        assert {config["backprop"]["lr"] for config in configs.values()} == {
            0.003
        }
        assert {
            config["standardized_decoder"]["lr"]
            for config in configs.values()
        } == {0.003}
        assert all(
            config["results"]["root"].startswith(
                "results/formal/phase0_v1_1/"
            )
            for config in configs.values()
        )


def test_stage3_seed_and_bp_lr_cannot_drift():
    protocol = _protocol()
    with pytest.raises(ValueError, match="Unknown Stage 3 seed"):
        resolved_method_config(protocol, method="full_bp", seed=5)

    config = resolved_method_config(protocol, method="hybrid_hhb", seed=0)
    config["backprop"]["lr"] = 0.001
    from schemas import validate_config

    with pytest.raises(ConfigError, match="0.003"):
        validate_config(config)


def test_stage3_hybrid_training_accepts_formal_seed_and_rejects_test_loader(
    tmp_path,
):
    config = _tiny_config("hybrid_hhb")
    run_dir = train_hybrid_config(
        config,
        run_dir=tmp_path / "hhb",
        loaders=_loaders(),
    )
    assert read_run_status(run_dir)["status"] == "completed"

    with pytest.raises(RuntimeError, match="test loader"):
        train_hybrid_config(
            _tiny_config("full_bp"),
            run_dir=tmp_path / "forbidden",
            loaders=_loaders(include_test=True),
        )


def test_full_random_uses_unmodified_encoder_and_trained_decoder(tmp_path):
    config = _tiny_config("full_random")
    run_dir = train_hybrid_config(
        config,
        run_dir=tmp_path / "random",
        loaders=_loaders(),
    )
    initial = torch.load(
        run_dir / "checkpoints" / "initial_state.pt",
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    final = torch.load(
        run_dir / "model_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    initial_encoder = {
        key: value
        for key, value in initial.items()
        if key.startswith("encoder.")
    }
    final_encoder = {
        key: value
        for key, value in final.items()
        if key.startswith("encoder.")
    }
    initial_decoder = {
        key: value
        for key, value in initial.items()
        if key.startswith("decoder.")
    }
    final_decoder = {
        key: value
        for key, value in final.items()
        if key.startswith("decoder.")
    }
    assert state_dict_checksum(initial_encoder) == state_dict_checksum(
        final_encoder
    )
    assert state_dict_checksum(initial_decoder) != state_dict_checksum(
        final_decoder
    )


def test_stage3_freeze_gate_rejects_nonfinite_csv_metrics(tmp_path):
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "stage,reconstruction_loss\nvalidation,0.01\n",
        encoding="utf-8",
    )
    assert _csv_values_finite(valid)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "stage,reconstruction_loss\nvalidation,nan\n",
        encoding="utf-8",
    )
    assert not _csv_values_finite(invalid)
