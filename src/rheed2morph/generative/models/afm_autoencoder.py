"""Spatial-latent AFM autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return min(8, channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ResBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res = ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.res(self.conv(x))


class AFMAutoencoder(nn.Module):
    def __init__(self, image_size: int = 128, latent_channels: int = 8) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.latent_size = self.image_size // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            ResBlock(32),
            DownBlock(32, 64),
            DownBlock(64, 128),
            DownBlock(128, 128),
            ResBlock(128),
            nn.GroupNorm(_groups(128), 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            ResBlock(128),
            UpBlock(128, 128),
            UpBlock(128, 64),
            UpBlock(64, 32),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def build_afm_autoencoder(image_size: int = 128, latent_channels: int = 8) -> AFMAutoencoder:
    return AFMAutoencoder(image_size=image_size, latent_channels=latent_channels)
