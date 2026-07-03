"""RHEED-safe augmentations for frame and clip representation learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _finite_clip(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def random_patch_mask(x: torch.Tensor, patch_size: int = 16, mask_ratio: float = 0.6) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask square patches in a BCHW tensor without changing orientation."""

    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    b, _c, h, w = x.shape
    patch = max(1, int(patch_size))
    gh = max(1, h // patch)
    gw = max(1, w // patch)
    mask_small = torch.rand((b, 1, gh, gw), device=x.device, dtype=x.dtype) < float(mask_ratio)
    mask = F.interpolate(mask_small.float(), size=(h, w), mode="nearest").to(dtype=x.dtype)
    return x * (1.0 - mask), mask


@dataclass
class RheedAugmentationConfig:
    crop_scale: float = 0.92
    translate_fraction: float = 0.04
    brightness: float = 0.08
    contrast: float = 0.12
    gamma: float = 0.12
    noise_std: float = 0.015
    blur_probability: float = 0.15
    patch_mask_ratio: float = 0.0
    patch_size: int = 16
    allow_flip: bool = False
    allow_rotation: bool = False


class RheedSafeAugment:
    """Mild RHEED augmentation that preserves streak direction by default."""

    def __init__(self, config: RheedAugmentationConfig | None = None) -> None:
        self.config = config or RheedAugmentationConfig()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_ndim = x.ndim
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(x.shape)}")
        out = _finite_clip(x.float())
        cfg = self.config
        b, _c, h, w = out.shape
        if cfg.crop_scale < 1.0:
            crop_h = max(2, int(round(h * cfg.crop_scale)))
            crop_w = max(2, int(round(w * cfg.crop_scale)))
            max_y = h - crop_h
            max_x = w - crop_w
            cropped = []
            for item in out:
                y = int(torch.randint(0, max_y + 1, (1,), device=out.device).item()) if max_y > 0 else 0
                x0 = int(torch.randint(0, max_x + 1, (1,), device=out.device).item()) if max_x > 0 else 0
                crop = item[:, y : y + crop_h, x0 : x0 + crop_w].unsqueeze(0)
                cropped.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)[0])
            out = torch.stack(cropped, dim=0)
        if cfg.translate_fraction > 0:
            max_shift_y = int(round(h * cfg.translate_fraction))
            max_shift_x = int(round(w * cfg.translate_fraction))
            shifted = []
            for item in out:
                dy = int(torch.randint(-max_shift_y, max_shift_y + 1, (1,), device=out.device).item()) if max_shift_y else 0
                dx = int(torch.randint(-max_shift_x, max_shift_x + 1, (1,), device=out.device).item()) if max_shift_x else 0
                shifted.append(torch.roll(item, shifts=(dy, dx), dims=(-2, -1)))
            out = torch.stack(shifted, dim=0)
        if cfg.brightness > 0:
            out = out + (torch.rand((b, 1, 1, 1), device=out.device) * 2.0 - 1.0) * cfg.brightness
        if cfg.contrast > 0:
            mean = out.mean(dim=(-2, -1), keepdim=True)
            scale = 1.0 + (torch.rand((b, 1, 1, 1), device=out.device) * 2.0 - 1.0) * cfg.contrast
            out = (out - mean) * scale + mean
        if cfg.gamma > 0:
            gamma = 1.0 + (torch.rand((b, 1, 1, 1), device=out.device) * 2.0 - 1.0) * cfg.gamma
            out = _finite_clip(out).pow(gamma)
        if cfg.noise_std > 0:
            out = out + torch.randn_like(out) * cfg.noise_std
        if cfg.blur_probability > 0 and torch.rand((), device=out.device).item() < cfg.blur_probability:
            out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        if cfg.allow_flip and torch.rand((), device=out.device).item() < 0.5:
            out = torch.flip(out, dims=(-1,))
        if cfg.allow_rotation and torch.rand((), device=out.device).item() < 0.25:
            out = torch.rot90(out, k=2, dims=(-2, -1))
        out = _finite_clip(out)
        if cfg.patch_mask_ratio > 0:
            out, _mask = random_patch_mask(out, cfg.patch_size, cfg.patch_mask_ratio)
        return out[0] if original_ndim == 3 else out


def augment_clip(clip: torch.Tensor, augmenter: RheedSafeAugment | None = None, time_jitter: bool = True) -> torch.Tensor:
    """Apply frame-safe augmentation to a TCHW or BTCHW clip."""

    aug = augmenter or RheedSafeAugment()
    if clip.ndim == 4:
        frames = clip
        if time_jitter and frames.shape[0] > 2:
            order = torch.arange(frames.shape[0], device=frames.device)
            shift = int(torch.randint(0, frames.shape[0], (1,), device=frames.device).item())
            frames = frames[torch.roll(order, shifts=shift)]
        return aug(frames)
    if clip.ndim == 5:
        return torch.stack([augment_clip(item, aug, time_jitter=time_jitter) for item in clip], dim=0)
    raise ValueError(f"Expected TCHW or BTCHW clip, got {tuple(clip.shape)}")
