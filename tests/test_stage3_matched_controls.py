from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from training.run_stage3_matched_controls import (
    METHODS,
    RULES,
    SEEDS,
    resolved_config,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "stage3_matched_controls_v1.yaml"


def _protocol():
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_matched_protocol_freezes_methods_seeds_and_test_boundary():
    protocol = _protocol()
    validate_protocol(protocol)
    assert tuple(protocol["seeds"]) == SEEDS
    assert tuple(protocol["methods"]) == METHODS
    assert protocol["pre_freeze_test_access_policy"] == "validation_only"
    assert protocol["post_freeze_test_access_policy"] == (
        "single_evaluation_per_checkpoint"
    )
    changed = deepcopy(protocol)
    changed["seeds"].append(5)
    with pytest.raises(ValueError, match="seeds"):
        validate_protocol(changed)


def test_matched_configs_have_exact_allocations_and_frozen_bp_lr():
    protocol = _protocol()
    for seed in SEEDS:
        for method in METHODS:
            config = resolved_config(protocol, method=method, seed=seed)
            assert config["training"]["seed"] == seed
            assert config["hybrid"]["encoder_layer_rules"] == RULES[method]
            assert config["backprop"]["lr"] == 0.003
            assert config["standardized_decoder"]["lr"] == 0.003
            assert "stage3_matched_controls" in config["results"]["root"]


def test_matched_controls_are_random_prefix_not_full_random():
    assert RULES["random_hbb"] == {
        "enc1": "frozen",
        "enc2": "bp",
        "enc3": "bp",
    }
    assert RULES["random_rrb"] == {
        "enc1": "frozen",
        "enc2": "frozen",
        "enc3": "bp",
    }
