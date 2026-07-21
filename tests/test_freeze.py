import torch

from models import ConvAutoencoder, LinearProbe
from utils.reproducibility import state_dict_checksum


def test_linear_probe_step_does_not_change_encoder():
    model = ConvAutoencoder(latent_dim=64, seed=0)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    before = state_dict_checksum(model.encoder)

    images = torch.rand(8, 1, 28, 28)
    with torch.no_grad():
        features = model.encode(images).flatten(start_dim=1)
    probe = LinearProbe(64)
    probe.standardizer.fit(features)
    optimizer = torch.optim.SGD(probe.classifier.parameters(), lr=0.1)
    labels = torch.arange(8) % 10
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(probe(features), labels)
    loss.backward()
    optimizer.step()

    assert state_dict_checksum(model.encoder) == before
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
