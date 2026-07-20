"""Learning-rule trainers that operate on the shared forward model."""

from __future__ import annotations

from typing import Any

from .backprop import BackpropTrainer


def build_trainer(model, config: dict[str, Any], device):
    rule = config["training"]["learning_rule"]
    if rule == "bp":
        return BackpropTrainer(model, config, device)
    if rule == "hebbian":
        from .hebbian import HebbianTrainer

        return HebbianTrainer(model, config, device)
    raise ValueError(f"Unknown learning rule: {rule}")


__all__ = ["BackpropTrainer", "build_trainer"]

