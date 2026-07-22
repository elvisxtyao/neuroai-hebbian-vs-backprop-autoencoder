import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from evaluation.compare_reconstructions import (
    evaluate_model,
    select_stratified_samples,
    validate_comparable_configs,
)


class IndexedToyDataset(Dataset):
    def __init__(self):
        self.images = torch.arange(60, dtype=torch.float32).view(60, 1, 1, 1) / 60
        self.labels = torch.arange(60) % 10

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], 1000 + index


class IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def reconstruct(self, images):
        return images * self.scale


class ConstantAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()), requires_grad=False)

    def reconstruct(self, images):
        return torch.zeros_like(images) + self.value


def test_stratified_selection_is_deterministic_and_balanced():
    dataset = IndexedToyDataset()
    first = select_stratified_samples(dataset, samples_per_class=2, seed=23)
    second = select_stratified_samples(dataset, samples_per_class=2, seed=23)
    for first_tensor, second_tensor in zip(first, second, strict=True):
        torch.testing.assert_close(first_tensor, second_tensor)
    _, labels, sample_ids = first
    assert labels.bincount(minlength=10).tolist() == [2] * 10
    assert sample_ids.unique().numel() == 20


def test_evaluation_is_inference_only_and_detects_nonconstant_output():
    images = torch.rand(12, 1, 4, 4)
    labels = torch.arange(12) % 10
    sample_ids = torch.arange(12)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=5)
    model = IdentityAutoencoder()

    metrics = evaluate_model(model, loader, device=torch.device("cpu"))

    assert metrics["mse"] == 0
    assert metrics["mae"] == 0
    assert metrics["shape_ok"]
    assert metrics["all_finite"]
    assert metrics["output_in_unit_interval"]
    assert metrics["nonconstant_across_samples"]
    assert metrics["checksum_unchanged"]


def test_evaluation_flags_constant_reconstruction():
    images = torch.rand(12, 1, 4, 4)
    labels = torch.arange(12) % 10
    sample_ids = torch.arange(12)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=6)

    metrics = evaluate_model(
        ConstantAutoencoder(), loader, device=torch.device("cpu")
    )

    assert not metrics["nonconstant_across_samples"]
    assert metrics["checksum_unchanged"]


def test_config_comparison_rejects_model_mismatch():
    common = {
        "data": {"dataset": "MNIST"},
        "model": {"latent_dim": 64},
        "training": {"learning_rule": "bp", "seed": 0},
    }
    hebbian = {
        "data": {"dataset": "MNIST"},
        "model": {"latent_dim": 32},
        "training": {"learning_rule": "hebbian", "seed": 0},
    }
    try:
        validate_comparable_configs(common, hebbian)
    except ValueError as error:
        assert "model config" in str(error)
    else:
        raise AssertionError("Expected a model-config mismatch")
