"""Descriptor-guided DDIM sampling and candidate reranking."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import compute_afm_descriptors_v2
from rheed2morph.generative.condition_control_v3_utils import descriptor_error_score, finite_float
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2


def _condition_target(condition: torch.Tensor, descriptor_dim: int) -> torch.Tensor:
    return condition[:, :descriptor_dim]


def _predict_model_output_with_cfg(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    pred_cond = model(x, t, condition)
    if guidance_scale == 1.0:
        return pred_cond
    pred_uncond = model(x, t, torch.zeros_like(condition))
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


def ddim_sample_with_descriptor_guidance(
    diffusion_model: torch.nn.Module,
    diffusion: GaussianDiffusionV2,
    shape: tuple[int, int, int, int],
    condition: torch.Tensor,
    descriptor_regressor: torch.nn.Module | None = None,
    descriptor_dim: int = 0,
    steps: int = 100,
    guidance_scale: float = 2.0,
    descriptor_guidance_weight: float = 0.0,
    grad_clip: float = 1.0,
) -> torch.Tensor:
    device = condition.device
    x = torch.randn(shape, device=device)
    indices = np.linspace(diffusion.timesteps - 1, 0, int(steps), dtype=np.int64).tolist()
    target = _condition_target(condition, descriptor_dim)
    for step_index, timestep_value in enumerate(indices):
        t = torch.full((shape[0],), int(timestep_value), device=device, dtype=torch.long)
        x = x.detach()
        if descriptor_regressor is not None and descriptor_guidance_weight > 0.0 and descriptor_dim > 0:
            x = x.requires_grad_(True)
            model_output_for_guidance = _predict_model_output_with_cfg(diffusion_model, x, t, condition, guidance_scale)
            eps_for_guidance = diffusion.predict_epsilon(x, t, model_output_for_guidance)
            alpha_bar = diffusion.alpha_bars[int(timestep_value)].to(device)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps_for_guidance) / torch.sqrt(alpha_bar)
            pred_desc, _proto = descriptor_regressor(pred_x0)
            desc_loss = torch.mean((pred_desc - target) ** 2)
            grad = torch.autograd.grad(desc_loss, x, retain_graph=False, create_graph=False)[0]
            grad_norm = grad.flatten(1).norm(dim=1).clamp(min=1e-6).view(-1, 1, 1, 1)
            grad = grad / grad_norm * grad_norm.clamp(max=grad_clip)
            x = (x - float(descriptor_guidance_weight) * grad).detach()
        with torch.no_grad():
            model_output = _predict_model_output_with_cfg(diffusion_model, x, t, condition, guidance_scale)
            eps = diffusion.predict_epsilon(x, t, model_output)
            alpha_bar = diffusion.alpha_bars[int(timestep_value)].to(device)
            alpha_prev = torch.tensor(1.0, device=device) if step_index == len(indices) - 1 else diffusion.alpha_bars[int(indices[step_index + 1])].to(device)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * eps
    return x.detach()


@torch.no_grad()
def decode_latents(autoencoder: torch.nn.Module, standardized_latents: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor) -> np.ndarray:
    raw = standardized_latents * latent_std + latent_mean
    decoded = autoencoder.decode(raw).detach().cpu().numpy()
    return decoded[:, 0]


def rerank_decoded_candidates(
    images: np.ndarray,
    target_row: dict[str, str],
    schema: dict[str, Any],
    keep_top_k: int = 4,
    duplicate_penalty_weight: float = 0.05,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    flattened = images.reshape(images.shape[0], -1)
    for index, image in enumerate(images):
        descriptors = compute_afm_descriptors_v2(image)
        descriptor_error = descriptor_error_score(descriptors, target_row, schema)
        std = float(np.std(image))
        realness_penalty = 0.0
        if std < 1e-4:
            realness_penalty += 100.0
        if std > 1.5:
            realness_penalty += float(std - 1.5)
        duplicate_penalty = 0.0
        if index > 0:
            distances = np.mean((flattened[:index] - flattened[index][None]) ** 2, axis=1)
            duplicate_penalty = float(np.mean(distances < 1e-7)) * duplicate_penalty_weight
        score = descriptor_error + realness_penalty + duplicate_penalty
        row: dict[str, Any] = {
            "candidate_index": index,
            "score": score,
            "descriptor_error": descriptor_error,
            "realness_penalty": realness_penalty,
            "duplicate_penalty": duplicate_penalty,
            "generated_std": std,
            "generated_min": float(np.min(image)),
            "generated_max": float(np.max(image)),
        }
        for name in schema["descriptor_columns"]:
            if name in descriptors:
                row[f"generated_{name}"] = float(descriptors[name])
            if target_row.get(name, "") != "":
                row[f"requested_{name}"] = finite_float(target_row[name])
        metrics.append(row)
    order = np.argsort([row["score"] for row in metrics])[: int(keep_top_k)]
    for rank, candidate_index in enumerate(order):
        metrics[int(candidate_index)]["rank"] = rank + 1
    return images[order], metrics
