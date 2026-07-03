"""Stronger spatial-latent AFM autoencoder for prior v2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return math.gcd(channels, min(8, channels)) or 1


class V2ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class V2DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            V2ResBlock(out_channels, dropout),
            V2ResBlock(out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V2UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res1 = V2ResBlock(out_channels, dropout)
        self.res2 = V2ResBlock(out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return self.res2(self.res1(x))


class AFMAutoencoderV2(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        latent_channels: int = 16,
        latent_size: int = 16,
        base_channels: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if image_size % latent_size != 0:
            raise ValueError("image_size must be divisible by latent_size.")
        down_factor = image_size // latent_size
        if down_factor < 2 or down_factor & (down_factor - 1):
            raise ValueError("image_size / latent_size must be a power of two and at least 2.")
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.base_channels = int(base_channels)
        self.dropout = float(dropout)
        levels = int(math.log2(down_factor))
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        encoder: list[nn.Module] = [
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            V2ResBlock(base_channels, dropout),
            V2ResBlock(base_channels, dropout),
        ]
        current = base_channels
        down_channels: list[int] = []
        for level in range(levels):
            out_channels = channels[min(level, len(channels) - 1)]
            encoder.append(V2DownBlock(current, out_channels, dropout))
            current = out_channels
            down_channels.append(current)
        encoder.extend(
            [
                V2ResBlock(current, dropout),
                nn.GroupNorm(_groups(current), current),
                nn.SiLU(),
                nn.Conv2d(current, latent_channels, kernel_size=3, padding=1),
            ]
        )
        decoder: list[nn.Module] = [
            nn.Conv2d(latent_channels, current, kernel_size=3, padding=1),
            V2ResBlock(current, dropout),
        ]
        for out_channels in reversed([base_channels] + down_channels[:-1]):
            decoder.append(V2UpBlock(current, out_channels, dropout))
            current = out_channels
        decoder.extend(
            [
                nn.GroupNorm(_groups(current), current),
                nn.SiLU(),
                nn.Conv2d(current, 1, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        if latent.shape[-2:] != (self.latent_size, self.latent_size):
            latent = F.interpolate(latent, size=(self.latent_size, self.latent_size), mode="bilinear", align_corners=False)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.decoder(latent)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return image

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def build_afm_autoencoder_v2(
    image_size: int = 128,
    latent_channels: int = 16,
    latent_size: int = 16,
    base_channels: int = 32,
    dropout: float = 0.05,
) -> AFMAutoencoderV2:
    return AFMAutoencoderV2(
        image_size=image_size,
        latent_channels=latent_channels,
        latent_size=latent_size,
        base_channels=base_channels,
        dropout=dropout,
    )
