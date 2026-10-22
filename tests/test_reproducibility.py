from models import ConvAutoencoder
from utils.reproducibility import state_dict_checksum


def test_paired_seed_produces_identical_initial_state():
    first = ConvAutoencoder(latent_dim=64, seed=3)
    second = ConvAutoencoder(latent_dim=64, seed=3)
    assert state_dict_checksum(first) == state_dict_checksum(second)


def test_different_seeds_produce_different_initial_state():
    first = ConvAutoencoder(latent_dim=64, seed=3)
    second = ConvAutoencoder(latent_dim=64, seed=4)
    assert state_dict_checksum(first) != state_dict_checksum(second)
