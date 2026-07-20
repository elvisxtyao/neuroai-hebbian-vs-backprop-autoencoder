"""Frozen-encoder linear evaluation components."""

from __future__ import annotations

import torch
from torch import nn


class FeatureStandardizer(nn.Module):
    """Apply statistics fitted only on frozen training representations."""

    def __init__(self, latent_dim: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("features must have shape N x latent_dim")
        self.mean.copy_(features.mean(dim=0))
        self.std.copy_(features.std(dim=0, unbiased=False))
        self.is_fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("FeatureStandardizer must be fitted on train features first")
        return (features - self.mean) / (self.std + self.epsilon)


class LinearProbe(nn.Module):
    """Exactly one affine classification layer after fixed standardization."""

    def __init__(self, latent_dim: int, num_classes: int = 10, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.standardizer = FeatureStandardizer(latent_dim, epsilon)
        self.classifier = nn.Linear(latent_dim, num_classes, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return self.classifier(self.standardizer(features))

