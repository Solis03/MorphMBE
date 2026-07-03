"""Morphology-preserving losses for AFM reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def image_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


def gradient_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rx, ry = image_gradients(recon)
    tx, ty = image_gradients(target)
    return 0.5 * (F.l1_loss(rx, tx) + F.l1_loss(ry, ty))


def log_psd_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    recon = recon.float()
    target = target.float()
    recon_centered = recon - recon.mean(dim=(-2, -1), keepdim=True)
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    recon_power = torch.abs(torch.fft.rfft2(recon_centered, norm="ortho")) ** 2
    target_power = torch.abs(torch.fft.rfft2(target_centered, norm="ortho")) ** 2
    return F.l1_loss(torch.log1p(recon_power), torch.log1p(target_power))


def roughness_consistency(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    recon_rq = torch.std(recon, dim=(-2, -1), unbiased=False)
    target_rq = torch.std(target, dim=(-2, -1), unbiased=False)
    recon_ra = torch.mean(torch.abs(recon - recon.mean(dim=(-2, -1), keepdim=True)), dim=(-2, -1))
    target_ra = torch.mean(torch.abs(target - target.mean(dim=(-2, -1), keepdim=True)), dim=(-2, -1))
    return F.l1_loss(recon_rq, target_rq) + F.l1_loss(recon_ra, target_ra)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(recon, target)
    grad = gradient_l1(recon, target)
    psd = log_psd_l1(recon, target)
    roughness = roughness_consistency(recon, target)
    total = l1 + 0.25 * grad + 0.05 * psd + 0.10 * roughness
    return total, {
        "l1": l1.detach(),
        "gradient_l1": grad.detach(),
        "psd_l1": psd.detach(),
        "roughness_error": roughness.detach(),
        "loss": total.detach(),
    }
