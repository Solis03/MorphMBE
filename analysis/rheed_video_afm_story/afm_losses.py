from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx = torch.abs((x[..., :, 1:] - x[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])).mean()
    dy = torch.abs((x[..., 1:, :] - x[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).mean()
    return 0.5 * (dx + dy)


def multiscale_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(x, y)
    for scale in (2, 4):
        loss = loss + F.l1_loss(F.avg_pool2d(x, scale), F.avg_pool2d(y, scale))
    return loss / 3.0


def radial_profile_torch(power: torch.Tensor, bins: int = 16) -> torch.Tensor:
    _, _, h, w = power.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=power.device), torch.arange(w, device=power.device), indexing="ij")
    rr = torch.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    edges = torch.linspace(1, rr.max(), bins + 1, device=power.device)
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rr >= lo) & (rr < hi)
        vals.append(power[:, :, mask].mean(dim=-1))
    return torch.stack(vals, dim=-1)


def log_radial_psd_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    fx = torch.fft.fftshift(torch.fft.fft2(x.squeeze(1)), dim=(-2, -1))
    fy = torch.fft.fftshift(torch.fft.fft2(y.squeeze(1)), dim=(-2, -1))
    px = (fx.real.square() + fx.imag.square()).unsqueeze(1)
    py = (fy.real.square() + fy.imag.square()).unsqueeze(1)
    rx = radial_profile_torch(px)
    ry = radial_profile_torch(py)
    return F.l1_loss(torch.log1p(rx), torch.log1p(ry))


def height_quantile_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    qs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=x.device, dtype=x.dtype)
    xq = torch.quantile(x.flatten(1), qs, dim=1).transpose(0, 1)
    yq = torch.quantile(y.flatten(1), qs, dim=1).transpose(0, 1)
    return F.l1_loss(xq, yq)


def afm_loss(pred: torch.Tensor, target: torch.Tensor, preset: str) -> tuple[torch.Tensor, dict[str, float]]:
    pixel = F.l1_loss(pred, target)
    grad = gradient_l1(pred, target)
    multi = multiscale_l1(pred, target)
    total = pixel + 0.5 * grad + 0.5 * multi
    parts = {"pixel_l1": float(pixel.detach().cpu()), "gradient_l1": float(grad.detach().cpu()), "multiscale_l1": float(multi.detach().cpu())}
    if preset == "physics_shape":
        psd = log_radial_psd_loss(pred, target)
        quant = height_quantile_loss(pred, target)
        total = total + 0.25 * psd + 0.10 * quant
        parts["log_radial_psd"] = float(psd.detach().cpu())
        parts["height_quantile"] = float(quant.detach().cpu())
    elif preset != "pixel_gradient":
        raise ValueError(f"Unknown loss preset: {preset}")
    parts["total"] = float(total.detach().cpu())
    return total, parts
