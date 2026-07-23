"""Frozen-snapshot candidate-update metrics for Q4."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from learning_rules.hebbian import CompetitiveOjaConv2d
from utils.checkpointing import file_sha256


def cosine_alignment(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float,
) -> float:
    if first.shape != second.shape:
        raise ValueError("update tensors must have the same shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first_vector = first.detach().to(torch.float64).reshape(-1)
    second_vector = second.detach().to(torch.float64).reshape(-1)
    denominator = first_vector.norm() * second_vector.norm() + epsilon
    return float((torch.dot(first_vector, second_vector) / denominator).item())


def norm_ratio(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float,
) -> float:
    if candidate.shape != reference.shape:
        raise ValueError("update tensors must have the same shape")
    return float(
        (
            candidate.detach().to(torch.float64).norm()
            / (reference.detach().to(torch.float64).norm() + epsilon)
        ).item()
    )


def raw_relative_difference(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    epsilon: float,
) -> float:
    if candidate.shape != reference.shape:
        raise ValueError("update tensors must have the same shape")
    candidate = candidate.detach().to(torch.float64)
    reference = reference.detach().to(torch.float64)
    return float(
        ((candidate - reference).norm() / (reference.norm() + epsilon)).item()
    )


def scale_matched_metrics(
    candidate_updates: torch.Tensor,
    reference_updates: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, float]:
    """Compare cross-batch mean candidate and reference directions."""

    if candidate_updates.shape != reference_updates.shape:
        raise ValueError("candidate and reference stacks must have the same shape")
    if candidate_updates.ndim < 2:
        raise ValueError("update stacks must have batch as their first axis")
    candidate_mean = candidate_updates.to(torch.float64).mean(dim=0)
    reference_mean = reference_updates.to(torch.float64).mean(dim=0)
    alpha = float(
        (
            torch.dot(candidate_mean.reshape(-1), reference_mean.reshape(-1))
            / (candidate_mean.square().sum() + epsilon)
        ).item()
    )
    matched = candidate_mean * alpha
    bias = float(
        ((matched - reference_mean).norm() / (reference_mean.norm() + epsilon)).item()
    )
    return {
        "alpha_star": alpha,
        "scale_matched_relative_bias": bias,
        "mean_update_alignment": cosine_alignment(
            candidate_mean, reference_mean, epsilon=epsilon
        ),
        "mean_update_norm_ratio": norm_ratio(
            candidate_mean, reference_mean, epsilon=epsilon
        ),
        "raw_relative_difference": raw_relative_difference(
            candidate_mean, reference_mean, epsilon=epsilon
        ),
    }


def update_snr(
    updates: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, float]:
    """Return linear and dB SNR across frozen-state mini-batch candidates."""

    if updates.ndim < 2 or updates.shape[0] < 2:
        raise ValueError("updates must contain at least two mini-batch candidates")
    values = updates.detach().to(torch.float64)
    mean = values.mean(dim=0)
    signal_power = mean.square().sum()
    residuals = values - mean
    noise_power = residuals.flatten(start_dim=1).square().sum(dim=1).mean()
    snr = signal_power / (noise_power + epsilon)
    linear = float(snr.item())
    return {
        "signal_power": float(signal_power.item()),
        "noise_power": float(noise_power.item()),
        "snr_linear": linear,
        "snr_db": (
            float("-inf") if linear <= 0 else 10.0 * math.log10(linear)
        ),
    }


def effective_hebbian_delta(
    weight: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    learning_rate: float,
    normalization_epsilon: float,
) -> torch.Tensor:
    """Return the exact apply-plus-filter-normalization weight displacement."""

    if weight.shape != raw_delta.shape:
        raise ValueError("weight and raw_delta shapes must match")
    if learning_rate < 0 or normalization_epsilon <= 0:
        raise ValueError("invalid learning rate or normalization epsilon")
    if learning_rate == 0:
        return torch.zeros_like(weight)
    proposed = weight.detach() + learning_rate * raw_delta.detach()
    flat = proposed.flatten(start_dim=1)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(normalization_epsilon)
    normalized = proposed / norms.view(-1, 1, 1, 1)
    return normalized - weight.detach()


def layer_inputs(model, images: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Return the exact presynaptic tensor for an encoder layer."""

    if layer_name == "enc1":
        return images
    h1 = model.encoder.activation(model.encoder.enc1(images))
    if layer_name == "enc2":
        return h1
    h2 = model.encoder.activation(model.encoder.enc2(h1))
    if layer_name == "enc3":
        return h2
    raise ValueError(f"Unknown encoder layer: {layer_name}")


@torch.no_grad()
def hebbian_candidate_deltas(
    model,
    images: torch.Tensor,
    *,
    layer_name: str,
    rule: CompetitiveOjaConv2d,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute raw/effective Hebbian deltas without changing the model."""

    layer = getattr(model.encoder, layer_name)
    inputs = layer_inputs(model, images, layer_name)
    preactivation = layer(inputs)
    postactivity = model.encoder.activation(preactivation)
    raw_delta, diagnostics = rule.compute_local_update(
        layer,
        preactivation,
        postactivity,
        inputs=inputs,
    )
    effective_delta = effective_hebbian_delta(
        layer.weight,
        raw_delta,
        learning_rate=rule.learning_rate,
        normalization_epsilon=rule.normalization_epsilon,
    )
    return raw_delta, effective_delta, {
        "hebbian_preactivation_mean": diagnostics.preactivation_mean,
        "hebbian_preactivation_std": diagnostics.preactivation_std,
        "hebbian_activation_mean": diagnostics.activation_mean,
        "hebbian_activation_variance": diagnostics.activation_variance,
        "hebbian_activation_sparsity": diagnostics.activation_sparsity,
    }


def bp_raw_negative_gradient(
    model,
    images: torch.Tensor,
    *,
    layer_name: str,
) -> tuple[torch.Tensor, float]:
    """Compute raw reconstruction negative gradient with no optimizer state."""

    layer = getattr(model.encoder, layer_name)
    original_flags = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        layer.weight.requires_grad_(True)
        model.eval()
        reconstruction = model(images)
        loss = F.mse_loss(reconstruction, images, reduction="mean")
        (gradient,) = torch.autograd.grad(
            loss,
            layer.weight,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        if layer.weight.grad is not None:
            raise RuntimeError("autograd.grad unexpectedly populated .grad")
        return -gradient.detach(), float(loss.detach().item())
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])
            parameter.grad = None


def prepare_fixed_batch_manifest(
    *,
    split_manifest: str | Path,
    output_path: str | Path,
    batch_count: int,
    batch_size: int,
    seed: int,
    version: str,
) -> Path:
    """Create or verify fixed original-sample IDs for Q4 update analysis."""

    split_manifest = Path(split_manifest)
    output_path = Path(output_path)
    if batch_count <= 0 or batch_size <= 0:
        raise ValueError("batch count and size must be positive")
    split = np.load(split_manifest, allow_pickle=False)
    training_ids = np.asarray(split["train_indices"], dtype=np.int64)
    required = batch_count * batch_size
    if training_ids.size < required:
        raise ValueError("training split is too small for unique fixed batches")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(training_ids.size, generator=generator).numpy()
    batch_ids = training_ids[order[:required]].reshape(batch_count, batch_size)
    split_hash = file_sha256(split_manifest)
    ids_hash = hashlib.sha256(batch_ids.tobytes()).hexdigest()
    if output_path.exists():
        existing = np.load(output_path, allow_pickle=False)
        required_fields = {
            "batch_ids",
            "batch_count",
            "batch_size",
            "seed",
            "source_split_sha256",
            "batch_ids_sha256",
            "version",
        }
        if not required_fields.issubset(existing.files):
            raise RuntimeError("Existing Q4 batch manifest is incomplete")
        if (
            not np.array_equal(existing["batch_ids"], batch_ids)
            or int(existing["batch_count"].item()) != batch_count
            or int(existing["batch_size"].item()) != batch_size
            or int(existing["seed"].item()) != seed
            or str(existing["source_split_sha256"].item()) != split_hash
            or str(existing["batch_ids_sha256"].item()) != ids_hash
            or str(existing["version"].item()) != version
        ):
            raise RuntimeError("Existing Q4 batch manifest does not reproduce")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        batch_ids=batch_ids,
        batch_count=np.asarray(batch_count, dtype=np.int64),
        batch_size=np.asarray(batch_size, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        source_split_sha256=np.asarray(split_hash),
        batch_ids_sha256=np.asarray(ids_hash),
        version=np.asarray(version),
    )
    return output_path


def snapshot_integrity_gate(
    snapshot_states: dict[str, dict[str, torch.Tensor]],
    initial_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Verify greedy snapshots and frozen earlier-layer weights."""

    required = {"enc1_end", "enc2_end", "enc3_end"}
    if set(snapshot_states) != required:
        raise ValueError(f"snapshot states must be {sorted(required)}")
    expected_keys = set(initial_state)
    finite = {}
    for snapshot_id, state in snapshot_states.items():
        if set(state) != expected_keys:
            raise RuntimeError(f"{snapshot_id} encoder keys differ from initialization")
        finite[snapshot_id] = all(torch.isfinite(value).all() for value in state.values())

    checks = {
        "all_tensors_finite": all(finite.values()),
        "enc1_frozen_after_enc1_end": torch.equal(
            snapshot_states["enc1_end"]["enc1.weight"],
            snapshot_states["enc2_end"]["enc1.weight"],
        )
        and torch.equal(
            snapshot_states["enc1_end"]["enc1.weight"],
            snapshot_states["enc3_end"]["enc1.weight"],
        ),
        "enc2_initial_at_enc1_end": torch.equal(
            snapshot_states["enc1_end"]["enc2.weight"],
            initial_state["enc2.weight"],
        ),
        "enc2_frozen_after_enc2_end": torch.equal(
            snapshot_states["enc2_end"]["enc2.weight"],
            snapshot_states["enc3_end"]["enc2.weight"],
        ),
        "enc3_initial_at_enc1_end": torch.equal(
            snapshot_states["enc1_end"]["enc3.weight"],
            initial_state["enc3.weight"],
        ),
        "enc3_initial_at_enc2_end": torch.equal(
            snapshot_states["enc2_end"]["enc3.weight"],
            initial_state["enc3.weight"],
        ),
    }
    checks["gate_pass"] = all(checks.values())
    return checks
