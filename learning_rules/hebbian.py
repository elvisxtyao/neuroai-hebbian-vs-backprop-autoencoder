"""Explicit convolutional WTA/Oja learning with no custom autograd."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .base import RepresentationTrainer, images_from_batch


def center_output_filter_updates(update: torch.Tensor) -> torch.Tensor:
    """Remove the update direction shared across output filters.

    Conv2d weights use ``[out_channels, in_channels, kernel_h, kernel_w]``.
    Centering dimension 0 therefore reproduces the notebook's linear-layer
    ``grad_weight - grad_weight.mean(axis=0)`` operation at filter level.
    """

    if update.ndim != 4:
        raise ValueError("Conv2d update must be a four-dimensional tensor")
    return update - update.mean(dim=0, keepdim=True)


@dataclass(frozen=True)
class HebbianBatchDiagnostics:
    update_norm: float
    preactivation_mean: float
    preactivation_std: float
    activation_mean: float
    activation_variance: float
    activation_sparsity: float
    winner_counts: torch.Tensor


@dataclass(frozen=True)
class HebbianEpochDiagnostics:
    layer: str
    update_norm: float
    weight_norm_mean: float
    weight_norm_std: float
    preactivation_mean: float
    preactivation_std: float
    activation_mean: float
    activation_variance: float
    activation_sparsity: float
    active_neuron_ratio: float
    winner_entropy: float
    max_winner_share: float
    collapse_detected: bool
    num_samples: int


def assess_competition(
    winner_counts: torch.Tensor,
    *,
    min_active_ratio: float,
    max_winner_share: float,
) -> tuple[float, float, bool]:
    """Return active ratio, maximum winner share and a collapse flag."""

    if winner_counts.ndim != 1:
        raise ValueError("winner_counts must be one-dimensional")
    if not 0 <= min_active_ratio <= 1 or not 0 <= max_winner_share <= 1:
        raise ValueError("collapse thresholds must be in [0,1]")
    counts = winner_counts.to(torch.float64)
    active_ratio = float((counts > 0).float().mean().item())
    total = counts.sum()
    largest_share = 0.0 if total <= 0 else float((counts.max() / total).item())
    collapsed = active_ratio < min_active_ratio or largest_share > max_winner_share
    return active_ratio, largest_share, collapsed


class CompetitiveOjaConv2d:
    """Compute and apply a local competitive Oja update for one Conv2d layer."""

    def __init__(
        self,
        *,
        learning_rate: float,
        winner_fraction: float,
        normalization_epsilon: float = 1e-8,
        competition_mode: str = "raw",
        competition_power: float = 1.0,
        competition_epsilon: float = 1e-6,
        center_inputs: bool = False,
        update_centering: str = "none",
    ) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 < winner_fraction <= 1:
            raise ValueError("winner_fraction must be in (0,1]")
        if competition_mode not in {"raw", "channel_rms", "channel_standardized"}:
            raise ValueError(
                "competition_mode must be raw, channel_rms or "
                "channel_standardized"
            )
        if competition_power <= 0:
            raise ValueError("competition_power must be positive")
        if competition_epsilon <= 0:
            raise ValueError("competition_epsilon must be positive")
        if update_centering not in {"none", "output_filters"}:
            raise ValueError(
                "update_centering must be none or output_filters"
            )
        self.learning_rate = learning_rate
        self.winner_fraction = winner_fraction
        self.normalization_epsilon = normalization_epsilon
        self.competition_mode = competition_mode
        self.competition_power = competition_power
        self.competition_epsilon = competition_epsilon
        self.center_inputs = center_inputs
        self.update_centering = update_centering

    def _competition_scores(self, post_activity: torch.Tensor) -> torch.Tensor:
        if self.competition_mode == "raw":
            return post_activity
        dimensions = (0, 2, 3)
        if self.competition_mode == "channel_rms":
            rms = post_activity.square().mean(
                dim=dimensions, keepdim=True
            ).sqrt()
            denominator = (rms + self.competition_epsilon).pow(
                self.competition_power
            )
            return post_activity / denominator
        mean = post_activity.mean(dim=dimensions, keepdim=True)
        std = post_activity.std(dim=dimensions, unbiased=False, keepdim=True)
        return (post_activity - mean) / (std + self.competition_epsilon)

    @torch.no_grad()
    def compute_local_update(
        self,
        layer: nn.Conv2d,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor | None = None,
        *,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, HebbianBatchDiagnostics]:
        """Return an Oja candidate update without mutating ``layer``.

        Competition is local to each sample and spatial output position. The
        top ``ceil(winner_fraction * out_channels)`` channels are retained.
        """

        if not isinstance(layer, nn.Conv2d):
            raise TypeError("CompetitiveOjaConv2d only supports nn.Conv2d")
        if pre_activity.ndim != 4 or pre_activity.shape[1] != layer.out_channels:
            raise ValueError("pre_activity must be B x out_channels x H x W")
        if post_activity is None:
            post_activity = F.relu(pre_activity)
        if post_activity.shape != pre_activity.shape:
            raise ValueError("post_activity and pre_activity shapes must match")

        batch_size, out_channels, out_height, out_width = post_activity.shape
        num_locations = out_height * out_width
        top_k = max(1, math.ceil(self.winner_fraction * out_channels))
        competition_scores = self._competition_scores(post_activity)
        winner_indices = torch.topk(competition_scores, k=top_k, dim=1).indices
        winner_mask = torch.zeros_like(post_activity, dtype=torch.bool)
        winner_mask.scatter_(1, winner_indices, True)
        winning_activity = post_activity * winner_mask

        patches = F.unfold(
            inputs,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )
        if self.center_inputs:
            patches = patches - patches.mean(dim=(0, 2), keepdim=True)
        winning_flat = winning_activity.flatten(start_dim=2)
        if patches.shape[-1] != num_locations:
            raise RuntimeError("Unfolded patch count does not match convolution output")

        denominator = float(batch_size * num_locations)
        hebbian_term = torch.einsum("bol,bil->oi", winning_flat, patches) / denominator
        hebbian_term = hebbian_term.reshape_as(layer.weight)
        oja_coefficient = winning_flat.square().sum(dim=(0, 2)) / denominator
        oja_term = oja_coefficient.view(-1, 1, 1, 1) * layer.weight
        delta_weight = hebbian_term - oja_term
        if delta_weight.shape[0] != layer.out_channels:
            raise RuntimeError("Conv2d output-filter dimension is not axis 0")
        if self.update_centering == "output_filters":
            # Center the complete raw Oja candidate before learning-rate
            # scaling. ``apply_local_update`` remains responsible for scaling
            # and the unchanged per-filter L2 normalization.
            delta_weight = center_output_filter_updates(delta_weight)

        positive_winners = winner_mask & (post_activity > 0)
        winner_counts = positive_winners.sum(dim=(0, 2, 3)).detach().cpu()
        diagnostics = HebbianBatchDiagnostics(
            update_norm=float(delta_weight.norm().item()),
            preactivation_mean=float(pre_activity.mean().item()),
            preactivation_std=float(pre_activity.std(unbiased=False).item()),
            activation_mean=float(post_activity.mean().item()),
            activation_variance=float(post_activity.var(unbiased=False).item()),
            activation_sparsity=float((post_activity <= 0).float().mean().item()),
            winner_counts=winner_counts,
        )
        return delta_weight, diagnostics

    @torch.no_grad()
    def normalize_weights(self, layer: nn.Conv2d) -> None:
        flat = layer.weight.flatten(start_dim=1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(self.normalization_epsilon)
        layer.weight.div_(norms.view(-1, 1, 1, 1))

    @torch.no_grad()
    def apply_local_update(self, layer: nn.Conv2d, delta_weight: torch.Tensor) -> None:
        if delta_weight.shape != layer.weight.shape:
            raise ValueError("delta_weight and layer.weight shapes must match")
        if self.learning_rate == 0:
            return
        layer.weight.add_(delta_weight, alpha=self.learning_rate)
        self.normalize_weights(layer)


class HebbianTrainer(RepresentationTrainer):
    """Greedy layer-wise trainer for Conv1, Conv2 and Conv3."""

    layer_names = ("enc1", "enc2", "enc3")

    def __init__(self, model, config: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        hebbian = config["hebbian"]
        layer_lrs = hebbian.get("layer_lrs", {})
        self.rules = {
            layer_name: CompetitiveOjaConv2d(
                learning_rate=layer_lrs.get(layer_name, hebbian["lr"]),
                winner_fraction=hebbian["winner_fraction"],
                normalization_epsilon=hebbian["normalization_epsilon"],
                competition_mode=hebbian.get("competition_mode", "raw"),
                competition_power=float(hebbian.get("competition_power", 1.0)),
                competition_epsilon=float(
                    hebbian.get("competition_epsilon", 1e-6)
                ),
                center_inputs=bool(hebbian.get("center_inputs", False)),
                update_centering=hebbian.get("update_centering", "none"),
            )
            for layer_name in self.layer_names
        }
        self.active_layer: str | None = None
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad_(False)

    def _layer(self, layer_name: str) -> nn.Conv2d:
        if layer_name not in self.layer_names:
            raise ValueError(f"Unknown encoder layer: {layer_name}")
        return getattr(self.model.encoder, layer_name)

    @torch.no_grad()
    def _inputs_for_layer(self, images: torch.Tensor, layer_name: str) -> torch.Tensor:
        images = images.to(self.device)
        if layer_name == "enc1":
            return images
        h1 = self.model.encoder.activation(self.model.encoder.enc1(images))
        if layer_name == "enc2":
            return h1
        h2 = self.model.encoder.activation(self.model.encoder.enc2(h1))
        if layer_name == "enc3":
            return h2
        raise ValueError(f"Unknown encoder layer: {layer_name}")

    def set_active_layer(self, layer_name: str) -> None:
        self._layer(layer_name)
        self.active_layer = layer_name

    @torch.no_grad()
    def train_batch(self, images: torch.Tensor) -> HebbianBatchDiagnostics:
        if self.active_layer is None:
            raise RuntimeError("Call set_active_layer before train_batch")
        layer = self._layer(self.active_layer)
        rule = self.rules[self.active_layer]
        layer_inputs = self._inputs_for_layer(images, self.active_layer)
        pre_activity = layer(layer_inputs)
        post_activity = self.model.encoder.activation(pre_activity)
        delta, diagnostics = rule.compute_local_update(
            layer,
            pre_activity,
            post_activity,
            inputs=layer_inputs,
        )
        rule.apply_local_update(layer, delta)
        return diagnostics

    @torch.no_grad()
    def train_layer_epoch(self, loader, layer_name: str) -> HebbianEpochDiagnostics:
        self.set_active_layer(layer_name)
        layer = self._layer(layer_name)
        self.model.encoder.eval()
        total_samples = 0
        weighted_update_norm = 0.0
        weighted_preactivation_mean = 0.0
        weighted_preactivation_std = 0.0
        weighted_activation_mean = 0.0
        weighted_activation_variance = 0.0
        weighted_activation_sparsity = 0.0
        winner_counts = torch.zeros(layer.out_channels, dtype=torch.float64)
        for batch in loader:
            images = images_from_batch(batch)
            diagnostics = self.train_batch(images)
            batch_size = images.shape[0]
            total_samples += batch_size
            weighted_update_norm += diagnostics.update_norm * batch_size
            weighted_preactivation_mean += diagnostics.preactivation_mean * batch_size
            weighted_preactivation_std += diagnostics.preactivation_std * batch_size
            weighted_activation_mean += diagnostics.activation_mean * batch_size
            weighted_activation_variance += diagnostics.activation_variance * batch_size
            weighted_activation_sparsity += diagnostics.activation_sparsity * batch_size
            winner_counts += diagnostics.winner_counts.to(torch.float64)

        weight_norms = layer.weight.flatten(start_dim=1).norm(dim=1).detach().cpu()
        total_winners = winner_counts.sum()
        if total_winners > 0:
            probabilities = winner_counts / total_winners
            nonzero = probabilities > 0
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            entropy /= math.log(layer.out_channels) if layer.out_channels > 1 else 1.0
            winner_entropy = float(entropy.item())
        else:
            winner_entropy = 0.0
        active_neuron_ratio, max_winner_share, collapse_detected = assess_competition(
            winner_counts,
            min_active_ratio=float(
                self.config["hebbian"].get("collapse_min_active_ratio", 0.25)
            ),
            max_winner_share=float(
                self.config["hebbian"].get("collapse_max_winner_share", 0.50)
            ),
        )
        return HebbianEpochDiagnostics(
            layer=layer_name,
            update_norm=weighted_update_norm / total_samples,
            weight_norm_mean=float(weight_norms.mean().item()),
            weight_norm_std=float(weight_norms.std(unbiased=False).item()),
            preactivation_mean=weighted_preactivation_mean / total_samples,
            preactivation_std=weighted_preactivation_std / total_samples,
            activation_mean=weighted_activation_mean / total_samples,
            activation_variance=weighted_activation_variance / total_samples,
            activation_sparsity=weighted_activation_sparsity / total_samples,
            active_neuron_ratio=active_neuron_ratio,
            winner_entropy=winner_entropy,
            max_winner_share=max_winner_share,
            collapse_detected=collapse_detected,
            num_samples=total_samples,
        )

    def run_epoch(self, loader, *, training: bool):
        raise RuntimeError(
            "Hebbian training is layer-wise; use train_layer_epoch(loader, layer_name)"
        )
