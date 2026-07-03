"""Diffusion utilities for AFM prior v2."""

from __future__ import annotations

import math

import numpy as np
import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clip(betas, 1e-5, 0.999).float()


def extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    gathered = values.to(timesteps.device).gather(0, timesteps)
    return gathered.view(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class GaussianDiffusionV2:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_target: str = "epsilon",
        device: torch.device | str = "cpu",
    ) -> None:
        if prediction_target not in {"epsilon", "v"}:
            raise ValueError("prediction_target must be 'epsilon' or 'v'.")
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.prediction_target = prediction_target
        self.device = torch.device(device)
        betas = cosine_beta_schedule(self.timesteps) if beta_schedule == "cosine" else linear_beta_schedule(self.timesteps)
        betas = betas.to(self.device)
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

    def _target(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.prediction_target == "epsilon":
            return noise
        sqrt_alpha = extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape)
        return sqrt_alpha * noise - sqrt_one_minus * x_start

    def predict_epsilon(self, x_t: torch.Tensor, timesteps: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        if self.prediction_target == "epsilon":
            return model_output
        sqrt_alpha = extract(self.sqrt_alpha_bars, timesteps, x_t.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t.shape)
        return sqrt_alpha * model_output + sqrt_one_minus * x_t

    def training_loss(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        cond_dropout: float = 0.15,
    ) -> torch.Tensor:
        batch = x_start.shape[0]
        timesteps = torch.randint(0, self.timesteps, (batch,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, timesteps, noise)
        model_condition = condition
        if cond_dropout > 0:
            keep = (torch.rand(batch, device=x_start.device) >= cond_dropout).float().view(batch, 1)
            model_condition = condition * keep
        target = self._target(x_start, timesteps, noise)
        predicted = model(noisy, timesteps, model_condition)
        return torch.mean((predicted - target) ** 2)

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
            pred_cond = model(x, t, condition)
            if guidance_scale != 1.0:
                pred_uncond = model(x, t, torch.zeros_like(condition))
                model_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                model_output = pred_cond
            eps = self.predict_epsilon(x, t, model_output)
            alpha_bar = self.alpha_bars[int(timestep_value)].to(device)
            if step_index == len(indices) - 1:
                alpha_prev = torch.tensor(1.0, device=device)
            else:
                alpha_prev = self.alpha_bars[int(indices[step_index + 1])].to(device)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps
        return x
