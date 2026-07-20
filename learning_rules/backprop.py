"""Backpropagation autoencoder baseline using the shared model."""

from __future__ import annotations

import torch
from torch import nn

from .base import EpochMetrics, RepresentationTrainer, images_from_batch


class BackpropTrainer(RepresentationTrainer):
    def __init__(self, model: nn.Module, config: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        bp = config["backprop"]
        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=bp["lr"],
            betas=tuple(bp["betas"]),
            weight_decay=bp["weight_decay"],
        )

    def train_batch(self, images: torch.Tensor) -> float:
        self.model.train()
        images = images.to(self.device)
        if images.min().item() < 0.0 or images.max().item() > 1.0:
            raise ValueError("phase0-v1 reconstruction inputs must stay in [0,1]")
        self.optimizer.zero_grad(set_to_none=True)
        reconstruction = self.model(images)
        loss = self.criterion(reconstruction, images)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())

    def run_epoch(self, loader, *, training: bool) -> EpochMetrics:
        self.model.train(training)
        total_loss = 0.0
        num_samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = images_from_batch(batch).to(self.device)
                if training:
                    loss = self.train_batch(images)
                else:
                    reconstruction = self.model(images)
                    loss = float(self.criterion(reconstruction, images).item())
                batch_size = images.shape[0]
                total_loss += loss * batch_size
                num_samples += batch_size
        return EpochMetrics(loss=total_loss / num_samples, num_samples=num_samples)

