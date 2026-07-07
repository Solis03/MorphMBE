"""Exposure-invariance helpers for MVP-9 shape-bag training."""

from __future__ import annotations

from typing import Any

import torch

from rheed2morph.rheed.models.shape_bag_morphology_predictor import exposure_consistency_loss


def photometric_perturb_shape_batch(
    frames: torch.Tensor,
    consensus_maps: torch.Tensor,
    *,
    brightness: float = 0.08,
    contrast: float = 0.15,
    noise_std: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply nuisance-only photometric perturbations without changing geometry."""
    batch = frames.shape[0]
    scale = torch.empty(batch, 1, 1, 1, 1, device=frames.device, dtype=frames.dtype).uniform_(1.0 - contrast, 1.0 + contrast)
    offset = torch.empty(batch, 1, 1, 1, 1, device=frames.device, dtype=frames.dtype).uniform_(-brightness, brightness)
    perturbed_frames = frames * scale + offset
    if noise_std > 0:
        perturbed_frames = perturbed_frames + torch.randn_like(perturbed_frames) * float(noise_std)
    map_scale = scale.squeeze(1)
    map_offset = offset.squeeze(1)
    perturbed_maps = consensus_maps * map_scale + map_offset
    if noise_std > 0:
        perturbed_maps = perturbed_maps + torch.randn_like(perturbed_maps) * float(noise_std)
    return perturbed_frames.clamp(-1.5, 1.5), perturbed_maps.clamp(-1.5, 1.5)


def perturb_batch_photometric(batch: dict[str, Any]) -> dict[str, Any]:
    out = dict(batch)
    out["frames"], out["consensus_maps"] = photometric_perturb_shape_batch(batch["frames"], batch["consensus_maps"])
    return out


def compute_exposure_invariance_loss(model: torch.nn.Module, batch: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    tensor_batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    perturbed = perturb_batch_photometric(tensor_batch)
    out_a = model(
        frames=tensor_batch["frames"],
        frame_mask=tensor_batch["frame_mask"],
        frame_weights=tensor_batch["frame_weights"],
        consensus_maps=tensor_batch["consensus_maps"],
        stable_shape_features=tensor_batch["shape_features"],
        metadata=tensor_batch["metadata"] if "metadata" in tensor_batch and tensor_batch["metadata"].numel() else None,
    )
    out_b = model(
        frames=perturbed["frames"],
        frame_mask=perturbed["frame_mask"],
        frame_weights=perturbed["frame_weights"],
        consensus_maps=perturbed["consensus_maps"],
        stable_shape_features=perturbed["shape_features"],
        metadata=perturbed["metadata"] if "metadata" in perturbed and perturbed["metadata"].numel() else None,
    )
    return exposure_consistency_loss(out_a, out_b)
