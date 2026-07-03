"""Small DDPM/DDIM utilities for AFM latent diffusion."""

from __future__ import annotations

import numpy as np
import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    gathered = values.to(timesteps.device).gather(0, timesteps)
    return gathered.view(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, device: torch.device | str = "cpu") -> None:
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        betas = linear_beta_schedule(self.timesteps).to(self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alpha_bars, timesteps, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape
        ) * noise

    def training_loss(self, model: torch.nn.Module, x_start: torch.Tensor, condition: torch.Tensor, cond_dropout: float = 0.1) -> torch.Tensor:
        batch = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        model_condition = condition
        if cond_dropout > 0.0:
            keep = (torch.rand(batch, device=x_start.device) >= cond_dropout).float().view(batch, 1)
            model_condition = condition * keep
        predicted = model(noisy, timesteps, model_condition)
        return torch.mean((predicted - noise) ** 2)

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        condition: torch.Tensor,
        steps: int = 50,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        device = condition.device
        x = torch.randn(shape, device=device)
        indices = np.linspace(self.timesteps - 1, 0, int(steps), dtype=np.int64).tolist()
        for step_index, timestep_value in enumerate(indices):
            t = torch.full((shape[0],), int(timestep_value), device=device, dtype=torch.long)
            eps_cond = model(x, t, condition)
            if guidance_scale != 1.0:
                eps_uncond = model(x, t, torch.zeros_like(condition))
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            alpha_bar = self.alpha_bars[int(timestep_value)].to(device)
            if step_index == len(indices) - 1:
                alpha_prev = torch.tensor(1.0, device=device)
            else:
                alpha_prev = self.alpha_bars[int(indices[step_index + 1])].to(device)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps
        return x
