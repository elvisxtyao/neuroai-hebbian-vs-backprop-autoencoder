"""Classification metrics shared by both learning rules."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_ce": float(log_loss(labels, probabilities, labels=list(range(10)))),
    }

