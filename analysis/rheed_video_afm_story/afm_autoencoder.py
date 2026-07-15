from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rq_disentanglement import project_unit_rq_torch


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(4, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SmallResidualAFMAutoencoder(nn.Module):
    has_unet_skip_connections = False

    def __init__(self, latent_dim: int, resolution: int = 128, base_channels: int = 12) -> None:
        super().__init__()
        if latent_dim not in (16, 32, 64):
            raise ValueError("latent_dim must be one of 16, 32, 64")
        self.latent_dim = int(latent_dim)
        self.resolution = int(resolution)
        self.base_channels = int(base_channels)
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            ResidualBlock(c),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            ResidualBlock(c * 4),
            nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1),
            ResidualBlock(c * 4),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.enc_flat = c * 4 * 4 * 4
        self.to_latent = nn.Linear(self.enc_flat, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.enc_flat)
        self.decoder_blocks = nn.ModuleList(
            [
                ResidualBlock(c * 4),
                ResidualBlock(c * 4),
                ResidualBlock(c * 2),
                ResidualBlock(c),
            ]
        )
        self.up_convs = nn.ModuleList(
            [
                nn.Conv2d(c * 4, c * 4, 3, padding=1),
                nn.Conv2d(c * 4, c * 2, 3, padding=1),
                nn.Conv2d(c * 2, c, 3, padding=1),
            ]
        )
        self.final = nn.Conv2d(c, 1, 3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return self.to_latent(h)

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        c = self.base_channels
        h = self.from_latent(z).view(z.shape[0], c * 4, 4, 4)
        h = self.decoder_blocks[0](h)
        for block, conv in zip(self.decoder_blocks[1:], self.up_convs):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = conv(h)
            h = block(h)
        h = F.interpolate(h, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return self.final(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return project_unit_rq_torch(self.decode_raw(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def architecture_summary(model: nn.Module) -> dict[str, object]:
    return {
        "class": model.__class__.__name__,
        "latent_dim": getattr(model, "latent_dim", None),
        "resolution": getattr(model, "resolution", None),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "normalization": "GroupNorm",
        "decoder_upsampling": "resize_convolution",
        "has_unet_skip_connections": bool(getattr(model, "has_unet_skip_connections", False)),
        "final_activation": "linear_then_unit_rq_projection",
    }
