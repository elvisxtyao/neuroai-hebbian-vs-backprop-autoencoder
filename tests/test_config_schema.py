from pathlib import Path

from schemas import load_config


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
