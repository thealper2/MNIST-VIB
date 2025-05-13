from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network that produces mean and variance for the latent representation."""

    def __init__(
        self, input_dim: int = 784, hidden_dim: int = 1024, latent_dim: int = 256
    ):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mean(h), F.softplus(self.fc_std(h) - 5, beta=1)


class Decoder(nn.Module):
    """Decoder network that predicts classes from latent representation."""

    def __init__(
        self, latent_dim: int = 256, hidden_dim: int = 1024, output_dim: int = 10
    ):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.fc_out(h)


class VIB(nn.Module):
    """Variational Information Bottleneck model combining encoder and decoder."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 1024,
        latent_dim: int = 256,
        output_dim: int = 10,
    ):
        super(VIB, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * std)
        eps = torch.rand_like(std)
        return mean + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encoder(x)
        z = self.reparameterize(mean, std)
        return self.decoder(z), mean, std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tot latent space without sampling (deterministic)."""
        mean, _ = self.encoder(x)
        return mean
