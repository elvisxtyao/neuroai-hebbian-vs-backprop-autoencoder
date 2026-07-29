from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.run_stage3_test_evaluation import (
    evaluate_frozen_components,
    require_freeze_gate,
)
from models import ConvAutoencoder, LinearProbe


def test_test_evaluator_refuses_missing_or_failed_gate(tmp_path: Path):
    with pytest.raises(RuntimeError, match="missing"):
        require_freeze_gate(tmp_path)
    (tmp_path / "freeze_gate.json").write_text(
        '{"decision":"FAIL","global_checks":{},"test_samples_accessed":0}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        require_freeze_gate(tmp_path)


def test_frozen_test_evaluation_is_finite_and_does_not_mutate_components():
    model = ConvAutoencoder(4, seed=7)
    standardized = ConvAutoencoder(4, seed=7)
    probe = LinearProbe(4)
    probe.standardizer.fit(torch.randn(12, 4))
    images = torch.rand(12, 1, 28, 28)
    labels = torch.arange(12) % 10
    sample_ids = torch.arange(12)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=5)

    result = evaluate_frozen_components(
        model,
        standardized,
        probe,
        loader,
        device=torch.device("cpu"),
    )

    assert result["num_samples"] == 12
    assert result["sample_ids_unique"]
    assert result["frozen_components_unchanged"]
    assert result["checksums_before"] == result["checksums_after"]
    assert torch.isfinite(
        torch.tensor(
            [
                result["accuracy"],
                result["classification_ce"],
                result["system_reconstruction_mse"],
                result["standardized_reconstruction_mse"],
            ]
        )
    ).all()


def test_frozen_test_evaluation_rejects_duplicate_sample_ids():
    model = ConvAutoencoder(4, seed=3)
    standardized = ConvAutoencoder(4, seed=3)
    probe = LinearProbe(4)
    probe.standardizer.fit(torch.randn(4, 4))
    loader = DataLoader(
        TensorDataset(
            torch.rand(4, 1, 28, 28),
            torch.arange(4),
            torch.tensor([0, 1, 1, 3]),
        ),
        batch_size=2,
    )
    with pytest.raises(RuntimeError, match="not unique"):
        evaluate_frozen_components(
            model,
            standardized,
            probe,
            loader,
            device=torch.device("cpu"),
        )
