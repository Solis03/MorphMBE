"""Conditioned latent denoiser for AFM spatial latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(channels: int) -> int:
    return min(8, channels)


class CondResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb = nn.Linear(emb_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class LatentUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 8,
        condition_dim: int = 24,
        base_channels: int = 64,
        emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.condition_dim = int(condition_dim)
        self.base_channels = int(base_channels)
        self.emb_dim = int(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.in_proj = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.enc1 = CondResBlock(base_channels, base_channels, emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = CondResBlock(base_channels * 2, base_channels * 2, emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.mid = CondResBlock(base_channels * 4, base_channels * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.merge2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.dec2 = CondResBlock(base_channels * 2, base_channels * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.merge1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.dec1 = CondResBlock(base_channels, base_channels, emb_dim)
        self.out_norm = nn.GroupNorm(_groups(base_channels), base_channels)
        self.out = nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1)

    def _embedding(self, timesteps: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.emb_dim))
        if condition is None:
            c_emb = torch.zeros_like(t_emb)
        else:
            c_emb = self.cond_mlp(condition)
        return t_emb + c_emb

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        emb = self._embedding(timesteps, condition)
        x0 = self.in_proj(x)
        e1 = self.enc1(x0, emb)
        e2 = self.enc2(self.down1(e1), emb)
        mid = self.mid(self.down2(e2), emb)
        d2 = self.up2(mid)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(self.merge2(torch.cat([d2, e2], dim=1)), emb)
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(self.merge1(torch.cat([d1, e1], dim=1)), emb)
        return self.out(F.silu(self.out_norm(d1)))
