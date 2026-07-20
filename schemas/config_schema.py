"""Load, merge and validate Phase 0 experiment YAML files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config violates the shared contract."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Circular config inheritance detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"Top-level config must be a mapping: {path}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_load_yaml(parent_path, seen), config)


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load a YAML config, resolving a single or nested ``extends`` chain."""

    config = _load_yaml(Path(path), set())
    if validate:
        validate_config(config)
    return config


def _require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"Missing required config field: {dotted_key}")
        value = value[part]
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate invariants that make BP and Hebbian runs comparable."""

    required = [
        "version",
        "data.dataset",
        "data.split_manifest",
        "data.split_seed",
        "data.train_size",
        "data.validation_size",
        "data.test_size",
        "data.normalization",
        "data.batch_size",
        "model.architecture",
        "model.encoder_channels",
        "model.latent_dim",
        "model.target_clamping",
        "training.learning_rule",
        "training.paired_seeds",
        "training.batch_size",
        "probe.type",
        "probe.freeze_encoder",
    ]
    for key in required:
        _require(config, key)

    if config["version"] != "phase0-v1":
        raise ConfigError("This skeleton implements version 'phase0-v1' only")
    if config["data"]["dataset"] != "MNIST":
        raise ConfigError("phase0-v1 requires MNIST")
    if config["data"]["normalization"] != "none":
        raise ConfigError("phase0-v1 requires [0,1] inputs without z-score")
    if config["data"]["input_range"] != [0.0, 1.0]:
        raise ConfigError("phase0-v1 input_range must be [0.0, 1.0]")
    if (
        config["data"]["train_size"],
        config["data"]["validation_size"],
        config["data"]["test_size"],
    ) != (50_000, 10_000, 10_000):
        raise ConfigError("phase0-v1 split sizes must be 50000/10000/10000")
    if config["model"]["architecture"] != "conv3_ae_v1":
        raise ConfigError("phase0-v1 architecture must be conv3_ae_v1")
    if config["model"]["encoder_channels"] != [16, 32]:
        raise ConfigError("phase0-v1 hidden encoder channels must be [16, 32]")
    if config["model"]["target_clamping"] is not False:
        raise ConfigError("Target clamping is forbidden in the main experiment")
    if config["training"]["learning_rule"] not in {"bp", "hebbian"}:
        raise ConfigError("training.learning_rule must be 'bp' or 'hebbian'")
    if config["training"]["paired_seeds"] != [0, 1, 2, 3, 4]:
        raise ConfigError("phase0-v1 paired seeds must be [0,1,2,3,4]")
    if config["data"]["batch_size"] != config["training"]["batch_size"]:
        raise ConfigError("data and training batch sizes must match")
    if config["probe"]["type"] != "linear" or not config["probe"]["freeze_encoder"]:
        raise ConfigError("phase0-v1 requires a frozen single-layer linear probe")

