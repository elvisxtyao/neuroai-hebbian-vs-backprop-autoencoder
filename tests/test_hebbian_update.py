import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learning_rules.hebbian import CompetitiveOjaConv2d, HebbianTrainer
from models import ConvAutoencoder
from schemas import load_config
from utils.reproducibility import state_dict_checksum


def test_competitive_oja_candidate_matches_hand_calculation_and_is_explicit():
    layer = nn.Conv2d(1, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[[0.5]]], [[[-0.5]]]]))
    inputs = torch.tensor([[[[2.0]]]])
    pre = layer(inputs)
    post = torch.relu(pre)
    rule = CompetitiveOjaConv2d(learning_rate=0.1, winner_fraction=0.5)
    before = layer.weight.detach().clone()

    delta, diagnostics = rule.compute_local_update(
        layer, pre, post, inputs=inputs
    )

    torch.testing.assert_close(layer.weight, before)
    torch.testing.assert_close(delta[0], torch.tensor([[[1.5]]]))
    torch.testing.assert_close(delta[1], torch.zeros_like(delta[1]))
    assert diagnostics.winner_counts.tolist() == [1, 0]
    assert diagnostics.update_norm == 1.5


def test_apply_local_update_normalizes_each_filter():
    layer = nn.Conv2d(2, 3, kernel_size=3, bias=False)
    rule = CompetitiveOjaConv2d(learning_rate=0.01, winner_fraction=0.5)
    delta = torch.randn_like(layer.weight)
    rule.apply_local_update(layer, delta)
    norms = layer.weight.flatten(start_dim=1).norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)


def test_greedy_epoch_changes_only_active_layer():
    config = load_config("configs/hebbian_main.yaml")
    model = ConvAutoencoder(latent_dim=64, seed=0)
    trainer = HebbianTrainer(model, config, torch.device("cpu"))
    images = torch.rand(8, 1, 28, 28)
    labels = torch.zeros(8, dtype=torch.long)
    sample_ids = torch.arange(8)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=4)
    before = {
        name: state_dict_checksum(getattr(model.encoder, name))
        for name in trainer.layer_names
    }

    diagnostics = trainer.train_layer_epoch(loader, "enc1")

    after = {
        name: state_dict_checksum(getattr(model.encoder, name))
        for name in trainer.layer_names
    }
    assert before["enc1"] != after["enc1"]
    assert before["enc2"] == after["enc2"]
    assert before["enc3"] == after["enc3"]
    assert diagnostics.layer == "enc1"
    assert diagnostics.num_samples == 8
    assert 0 <= diagnostics.winner_entropy <= 1
    assert 0 <= diagnostics.active_neuron_ratio <= 1
