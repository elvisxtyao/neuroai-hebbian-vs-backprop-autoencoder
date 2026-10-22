"""The shared Phase 0-v1 three-layer convolutional autoencoder."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def _validate_encoder_channels(
    encoder_channels: tuple[int, int] | list[int],
) -> tuple[int, int]:
    if len(encoder_channels) != 2:
        raise ValueError("encoder_channels must contain exactly two values")
    channels = tuple(int(value) for value in encoder_channels)
    if any(value <= 0 for value in channels):
        raise ValueError("encoder_channels must be positive")
    return channels


class ConvEncoder(nn.Module):
    """Three learnable convolutions mapping MNIST to ``B x L x 1 x 1``."""

    def __init__(
        self,
        latent_dim: int = 64,
        *,
        encoder_channels: tuple[int, int] | list[int] = (16, 32),
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        hidden1, hidden2 = _validate_encoder_channels(encoder_channels)
        self.latent_dim = latent_dim
        self.encoder_channels = (hidden1, hidden2)
        self.enc1 = nn.Conv2d(
            1, hidden1, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.enc2 = nn.Conv2d(
            hidden1, hidden2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.enc3 = nn.Conv2d(
            hidden2, latent_dim, kernel_size=7, stride=1, padding=0, bias=False
        )
        self.activation = nn.ReLU()

    def forward_features(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError(f"Expected Bx1x28x28 input, received {tuple(x.shape)}")
        h1 = self.activation(self.enc1(x))
        h2 = self.activation(self.enc2(h1))
        z = self.activation(self.enc3(h2))
        return OrderedDict(h1=h1, h2=h2, z=z)

    def forward(
        self, x: torch.Tensor, *, return_all_layers: bool = False
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        features = self.forward_features(x)
        return features if return_all_layers else features["z"]


class ConvDecoder(nn.Module):
    """Three transposed convolutions reconstructing ``B x 1 x 28 x 28``."""

    def __init__(
        self,
        latent_dim: int = 64,
        *,
        encoder_channels: tuple[int, int] | list[int] = (16, 32),
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        hidden1, hidden2 = _validate_encoder_channels(encoder_channels)
        self.encoder_channels = (hidden1, hidden2)
        self.dec1 = nn.ConvTranspose2d(
            latent_dim, hidden2, kernel_size=7, stride=1, padding=0, bias=True
        )
        self.dec2 = nn.ConvTranspose2d(
            hidden2, hidden1, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.dec3 = nn.ConvTranspose2d(
            hidden1, 1, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        expected = (self.dec1.in_channels, 1, 1)
        if z.ndim != 4 or tuple(z.shape[1:]) != expected:
            raise ValueError(f"Expected Bx{expected[0]}x1x1 latent, received {tuple(z.shape)}")
        h = self.activation(self.dec1(z))
        h = self.activation(self.dec2(h))
        return self.output_activation(self.dec3(h))


class ConvAutoencoder(nn.Module):
    """Shared forward model; learning rules live outside this module."""

    def __init__(
        self,
        latent_dim: int = 64,
        *,
        encoder_channels: tuple[int, int] | list[int] = (16, 32),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_channels = _validate_encoder_channels(encoder_channels)
        self.encoder = ConvEncoder(
            latent_dim, encoder_channels=self.encoder_channels
        )
        self.decoder = ConvDecoder(
            latent_dim, encoder_channels=self.encoder_channels
        )
        if seed is not None:
            self.initialize(seed)

    def initialize(self, seed: int) -> None:
        """Apply the frozen paired initialization without leaking RNG state."""

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    if module is self.decoder.dec3:
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        nn.init.kaiming_uniform_(
                            module.weight, mode="fan_in", nonlinearity="relu"
                        )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def encode(
        self, x: torch.Tensor, *, return_all_layers: bool = False
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        return self.encoder(x, return_all_layers=return_all_layers)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(x)

    def architecture_metadata(self) -> dict[str, object]:
        hidden1, hidden2 = self.encoder_channels
        return {
            "architecture": "conv3_ae_v1",
            "encoder_channels": [hidden1, hidden2],
            "latent_dim": self.latent_dim,
            "encoder_shapes": {
                "h1": [hidden1, 14, 14],
                "h2": [hidden2, 7, 7],
                "z": [self.latent_dim, 1, 1],
            },
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "encoder_parameter_count": sum(
                parameter.numel() for parameter in self.encoder.parameters()
            ),
            "decoder_parameter_count": sum(
                parameter.numel() for parameter in self.decoder.parameters()
            ),
        }


def autoencoder_from_config(
    config: dict[str, object], *, seed: int | None = None
) -> ConvAutoencoder:
    """Construct the shared model from the frozen model-config fields."""

    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise TypeError("config.model must be a mapping")
    return ConvAutoencoder(
        int(model_config["latent_dim"]),
        encoder_channels=model_config["encoder_channels"],
        seed=seed,
    )
