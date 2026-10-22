import torch

from models import ConvAutoencoder


def test_phase0_shapes_and_ranges():
    model = ConvAutoencoder(latent_dim=64, seed=0)
    images = torch.rand(4, 1, 28, 28)
    features = model.encode(images, return_all_layers=True)
    assert tuple(features) == ("h1", "h2", "z")
    assert features["h1"].shape == (4, 16, 14, 14)
    assert features["h2"].shape == (4, 32, 7, 7)
    assert features["z"].shape == (4, 64, 1, 1)
    reconstruction = model.reconstruct(images)
    assert reconstruction.shape == images.shape
    assert reconstruction.min() >= 0
    assert reconstruction.max() <= 1


def test_parameter_counts_match_standard():
    metadata = ConvAutoencoder(latent_dim=64, seed=0).architecture_metadata()
    assert metadata["encoder_parameter_count"] == 105_104
    assert metadata["decoder_parameter_count"] == 108_849
    assert metadata["parameter_count"] == 213_953
