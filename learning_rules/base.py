"""Shared trainer types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    num_samples: int


def images_from_batch(batch) -> torch.Tensor:
    """Return images while intentionally ignoring labels and sample IDs."""

    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


class RepresentationTrainer(ABC):
    @abstractmethod
    def train_batch(self, images: torch.Tensor) -> float:
        raise NotImplementedError

    @abstractmethod
    def run_epoch(self, loader, *, training: bool) -> EpochMetrics:
        raise NotImplementedError

