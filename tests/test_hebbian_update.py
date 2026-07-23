import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learning_rules.hebbian import (
    CompetitiveOjaConv2d,
    HebbianTrainer,
    assess_competition,
)
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


def test_zero_learning_rate_leaves_weights_bitwise_unchanged():
    layer = nn.Conv2d(2, 3, kernel_size=3, bias=False)
    before = layer.weight.detach().clone()
    rule = CompetitiveOjaConv2d(learning_rate=0.0, winner_fraction=0.5)

    rule.apply_local_update(layer, torch.randn_like(layer.weight))

    assert torch.equal(layer.weight, before)


def test_500_local_updates_remain_finite_and_normalized():
    generator = torch.Generator().manual_seed(123)
    layer = nn.Conv2d(1, 4, kernel_size=1, bias=False)
    rule = CompetitiveOjaConv2d(learning_rate=0.001, winner_fraction=0.5)
    for _ in range(500):
        inputs = torch.rand(2, 1, 2, 2, generator=generator)
        pre = layer(inputs)
        delta, _ = rule.compute_local_update(layer, pre, inputs=inputs)
        rule.apply_local_update(layer, delta)

    assert torch.isfinite(layer.weight).all()
    norms = layer.weight.flatten(start_dim=1).norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)


def test_multistep_local_updates_are_reproducible():
    first = nn.Conv2d(1, 4, kernel_size=1, bias=False)
    second = nn.Conv2d(1, 4, kernel_size=1, bias=False)
    second.load_state_dict(first.state_dict())
    first_rule = CompetitiveOjaConv2d(learning_rate=0.001, winner_fraction=0.5)
    second_rule = CompetitiveOjaConv2d(learning_rate=0.001, winner_fraction=0.5)
    generator = torch.Generator().manual_seed(456)
    batches = [torch.rand(2, 1, 2, 2, generator=generator) for _ in range(20)]

    for inputs in batches:
        for layer, rule in ((first, first_rule), (second, second_rule)):
            pre = layer(inputs)
            delta, _ = rule.compute_local_update(layer, pre, inputs=inputs)
            rule.apply_local_update(layer, delta)

    assert torch.equal(first.weight, second.weight)


def test_channel_rms_competition_breaks_scale_monopoly():
    layer = nn.Conv2d(1, 2, kernel_size=1, bias=False)
    inputs = torch.ones(2, 1, 1, 1)
    post = torch.tensor([[[[10.0]], [[1.0]]], [[[9.0]], [[0.1]]]])
    raw = CompetitiveOjaConv2d(
        learning_rate=0.1,
        winner_fraction=0.5,
        competition_mode="raw",
    )
    homeostatic = CompetitiveOjaConv2d(
        learning_rate=0.1,
        winner_fraction=0.5,
        competition_mode="channel_rms",
        competition_power=1.0,
    )

    _, raw_diagnostics = raw.compute_local_update(
        layer, post, post, inputs=inputs
    )
    _, homeostatic_diagnostics = homeostatic.compute_local_update(
        layer, post, post, inputs=inputs
    )

    assert raw_diagnostics.winner_counts.tolist() == [2, 0]
    assert homeostatic_diagnostics.winner_counts.tolist() == [1, 1]


def test_channel_standardized_competition_is_deterministic_and_non_mutating():
    generator = torch.Generator().manual_seed(303)
    layer = nn.Conv2d(2, 4, kernel_size=1, bias=False)
    inputs = torch.rand(8, 2, 3, 3, generator=generator)
    post = torch.rand(8, 4, 3, 3, generator=generator)
    before = layer.weight.detach().clone()
    rule = CompetitiveOjaConv2d(
        learning_rate=0.1,
        winner_fraction=0.25,
        competition_mode="channel_standardized",
    )

    first, first_diagnostics = rule.compute_local_update(
        layer, post, post, inputs=inputs
    )
    second, second_diagnostics = rule.compute_local_update(
        layer, post, post, inputs=inputs
    )

    assert torch.equal(first, second)
    assert torch.equal(
        first_diagnostics.winner_counts,
        second_diagnostics.winner_counts,
    )
    assert torch.equal(layer.weight, before)


def test_competition_collapse_detector_covers_balanced_and_collapsed_cases():
    balanced = assess_competition(
        torch.tensor([10, 10, 10, 10]),
        min_active_ratio=0.5,
        max_winner_share=0.6,
    )
    collapsed = assess_competition(
        torch.tensor([100, 0, 0, 0]),
        min_active_ratio=0.5,
        max_winner_share=0.6,
    )

    assert balanced == (1.0, 0.25, False)
    assert collapsed == (0.25, 1.0, True)


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
