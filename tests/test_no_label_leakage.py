import inspect

import torch

from learning_rules.backprop import BackpropTrainer
from learning_rules.base import images_from_batch
from schemas import load_config


def test_representation_train_batch_accepts_images_only():
    parameters = list(inspect.signature(BackpropTrainer.train_batch).parameters)
    assert parameters == ["self", "images"]


def test_labels_are_not_used_by_representation_batch_adapter():
    images = torch.rand(4, 1, 28, 28)
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([9, 9, 9, 9])
    sample_ids = torch.arange(4)
    assert images_from_batch((images, labels_a, sample_ids)) is images
    assert images_from_batch((images, labels_b, sample_ids)) is images


def test_target_clamping_is_disabled_for_both_rules():
    for path in ("configs/bp_main.yaml", "configs/hebbian_main.yaml"):
        assert load_config(path)["model"]["target_clamping"] is False
