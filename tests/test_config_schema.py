from pathlib import Path

import pytest

from schemas import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_bp_config_resolves_common_settings():
    config = load_config(ROOT / "configs" / "bp_main.yaml")
    assert config["training"]["learning_rule"] == "bp"
    assert config["data"]["batch_size"] == 128
    assert config["model"]["latent_dim"] == 64
    assert config["model"]["target_clamping"] is False


def test_hebbian_config_uses_same_forward_settings():
    bp = load_config(ROOT / "configs" / "bp_main.yaml")
    hebbian = load_config(ROOT / "configs" / "hebbian_main.yaml")
    assert bp["model"] == hebbian["model"]
    assert bp["data"] == hebbian["data"]
    assert hebbian["training"]["learning_rule"] == "hebbian"


def test_phase0_v1_1_formal_configs_are_paired_and_frozen():
    bp = load_config(ROOT / "configs" / "formal" / "bp_phase0_v1_1.yaml")
    hebbian = load_config(
        ROOT / "configs" / "formal" / "hebbian_phase0_v1_1.yaml"
    )

    assert bp["version"] == hebbian["version"] == "phase0-v1.1"
    assert bp["backprop"]["lr"] == 0.003
    assert bp["training"]["seed"] == hebbian["training"]["seed"] == 0
    assert bp["training"]["paired_seeds"] == hebbian["training"]["paired_seeds"]
    assert bp["data"] == hebbian["data"]
    assert bp["model"] == hebbian["model"]
    assert bp["probe"] == hebbian["probe"]
    assert bp["protocol"] == hebbian["protocol"]
    assert bp["results"] == hebbian["results"]


def test_phase0_v1_1_rejects_wrong_bp_learning_rate():
    config = load_config(ROOT / "configs" / "formal" / "bp_phase0_v1_1.yaml")
    config["backprop"]["lr"] = 0.001
    with pytest.raises(ConfigError, match="0.003"):
        validate_config(config)
