from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rheed2morph.generative.diffusion_v2 import (
    cosine_beta_schedule,
    extract,
)


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequency[None]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return F.pad(embedding, (0, dimension - embedding.shape[1]))


def _groups(channels: int) -> int:
    return math.gcd(channels, min(channels, 8)) or 1


class GuidedBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, emb: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(input_channels), input_channels)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.embedding = nn.Linear(emb, 2 * output_channels)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.embedding(embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = (
            hidden * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class StructureGuidedResidualUNet(nn.Module):
    """Image-space residual DDPM that keeps a supplied island structure map."""

    def __init__(self, base_channels: int = 32, embedding_dim: int = 128):
        super().__init__()
        base = int(base_channels)
        emb = int(embedding_dim)
        self.base_channels = base
        self.embedding_dim = emb
        self.time_mlp = nn.Sequential(
            nn.Linear(emb, emb),
            nn.SiLU(),
            nn.Linear(emb, emb),
        )
        self.input = nn.Conv2d(2, base, kernel_size=3, padding=1)
        self.enc1 = GuidedBlock(base, base, emb)
        self.down1 = nn.Conv2d(
            base, 2 * base, kernel_size=3, stride=2, padding=1
        )
        self.enc2 = GuidedBlock(2 * base, 2 * base, emb)
        self.down2 = nn.Conv2d(
            2 * base, 4 * base, kernel_size=3, stride=2, padding=1
        )
        self.middle = GuidedBlock(4 * base, 4 * base, emb)
        self.up2 = nn.ConvTranspose2d(
            4 * base, 2 * base, kernel_size=4, stride=2, padding=1
        )
        self.merge2 = nn.Conv2d(4 * base, 2 * base, kernel_size=1)
        self.dec2 = GuidedBlock(2 * base, 2 * base, emb)
        self.up1 = nn.ConvTranspose2d(
            2 * base, base, kernel_size=4, stride=2, padding=1
        )
        self.merge1 = nn.Conv2d(2 * base, base, kernel_size=1)
        self.dec1 = GuidedBlock(base, base, emb)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(),
            nn.Conv2d(base, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.time_mlp(
            timestep_embedding(timesteps, self.embedding_dim)
        )
        x0 = self.input(torch.cat([noisy_residual, guide], dim=1))
        e1 = self.enc1(x0, embedding)
        e2 = self.enc2(self.down1(e1), embedding)
        middle = self.middle(self.down2(e2), embedding)
        d2 = self.up2(middle)
        d2 = self.dec2(self.merge2(torch.cat([d2, e2], dim=1)), embedding)
        d1 = self.up1(d2)
        d1 = self.dec1(self.merge1(torch.cat([d1, e1], dim=1)), embedding)
        return self.output(d1)


@dataclass
class StructureGuidedDiffusion:
    timesteps: int = 200
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        betas = cosine_beta_schedule(int(self.timesteps)).to(device)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def q_sample(
        self,
        residual: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_alpha_bars, timesteps, residual.shape) * residual
            + extract(
                self.sqrt_one_minus_alpha_bars,
                timesteps,
                residual.shape,
            )
            * noise
        )

    def training_loss(
        self,
        model: StructureGuidedResidualUNet,
        residual: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        batch = residual.shape[0]
        timesteps = torch.randint(
            0,
            int(self.timesteps),
            (batch,),
            device=residual.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(residual)
        noisy = self.q_sample(residual, timesteps, noise)
        prediction = model(noisy, timesteps, guide)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(
        self,
        model: StructureGuidedResidualUNet,
        guide: torch.Tensor,
        *,
        steps: int,
        seed: int,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Sample a residual; strength < 1 performs conservative SDEdit-like refinement."""

        if not 0.0 < float(strength) <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        start = max(
            1,
            int(round((int(self.timesteps) - 1) * float(strength))),
        )
        sample = torch.randn(
            guide.shape,
            generator=generator,
            device="cpu",
        ).to(guide.device)
        sample = sample * torch.sqrt(1.0 - self.alpha_bars[start]).to(
            guide.device
        )
        indices = np.linspace(
            start, 0, min(int(steps), start + 1), dtype=np.int64
        ).tolist()
        for index, value in enumerate(indices):
            timestep = torch.full(
                (len(guide),),
                int(value),
                device=guide.device,
                dtype=torch.long,
            )
            epsilon = model(sample, timestep, guide)
            alpha = self.alpha_bars[int(value)].to(guide.device)
            previous = (
                torch.tensor(1.0, device=guide.device)
                if index == len(indices) - 1
                else self.alpha_bars[int(indices[index + 1])].to(guide.device)
            )
            predicted_clean = (
                sample - torch.sqrt(1.0 - alpha) * epsilon
            ) / torch.sqrt(alpha)
            predicted_clean = torch.clamp(predicted_clean, -5.0, 5.0)
            sample = (
                torch.sqrt(previous) * predicted_clean
                + torch.sqrt(1.0 - previous) * epsilon
            )
        return sample
