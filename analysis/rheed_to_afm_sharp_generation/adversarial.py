from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_torch

from analysis.rheed_to_afm_generation.data import ConditionScaler


class CircularConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        spectral: bool = False,
    ) -> None:
        super().__init__()
        self.padding = dilation * (kernel_size // 2)
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.convolution = spectral_norm(convolution) if spectral else convolution

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = F.pad(
            tensor,
            (self.padding, self.padding, self.padding, self.padding),
            mode="circular",
        )
        return self.convolution(tensor)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.normalization1 = nn.GroupNorm(groups, channels)
        self.normalization2 = nn.GroupNorm(groups, channels)
        self.convolution1 = CircularConv2d(
            channels, channels, 3, dilation=dilation
        )
        self.convolution2 = CircularConv2d(channels, channels, 3)
        self.modulation = nn.Linear(condition_dim, channels * 4)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self, tensor: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.modulation(condition).chunk(4, dim=1)
        hidden = self.normalization1(tensor)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale1[:, :, None, None]))
        hidden = hidden + 0.25 * shift1[:, :, None, None]
        hidden = self.convolution1(F.silu(hidden))
        hidden = self.normalization2(hidden)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale2[:, :, None, None]))
        hidden = hidden + 0.25 * shift2[:, :, None, None]
        hidden = self.convolution2(F.silu(hidden))
        return tensor + hidden / math.sqrt(2.0)


class PhysicsSeededAFMRefiner(nn.Module):
    """Circular residual generator that sharpens a learned random field.

    It has no upsampling path, which removes the dominant border/resize
    artifact observed in the earlier CVAE.
    """

    def __init__(
        self,
        condition_dim: int,
        channels: int = 32,
        residual_scale: float = 0.75,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.channels = int(channels)
        self.residual_scale = float(residual_scale)
        self.input = CircularConv2d(2, channels, 5)
        self.blocks = nn.ModuleList(
            [
                ConditionalResidualBlock(channels, condition_dim, dilation)
                for dilation in (1, 2, 4, 8, 4, 2)
            ]
        )
        self.output = CircularConv2d(channels, 1, 3)
        nn.init.zeros_(self.output.convolution.weight)
        nn.init.zeros_(self.output.convolution.bias)

    def forward(
        self, spectral_seed: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        highpass = spectral_seed - F.avg_pool2d(
            F.pad(spectral_seed, (2, 2, 2, 2), mode="circular"),
            kernel_size=5,
            stride=1,
        )
        hidden = self.input(torch.cat([spectral_seed, highpass], dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        residual = self.output(F.silu(hidden))
        return project_unit_rq_torch(
            spectral_seed + self.residual_scale * residual
        )


class ProjectionDiscriminator(nn.Module):
    def __init__(self, condition_dim: int, base_channels: int = 24) -> None:
        super().__init__()
        channels = [
            (1, base_channels),
            (base_channels, base_channels * 2),
            (base_channels * 2, base_channels * 4),
            (base_channels * 4, base_channels * 8),
        ]
        blocks: list[nn.Module] = []
        for in_channels, out_channels in channels:
            blocks.extend(
                [
                    CircularConv2d(
                        in_channels,
                        out_channels,
                        3,
                        stride=2,
                        spectral=True,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.features = nn.Sequential(*blocks)
        feature_dim = base_channels * 8
        self.unconditional = spectral_norm(nn.Linear(feature_dim, 1))
        self.condition_projection = spectral_norm(
            nn.Linear(condition_dim, feature_dim, bias=False)
        )

    def forward(
        self,
        image: torch.Tensor,
        condition: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(image)
        features = feature_map.mean(dim=(2, 3))
        score = self.unconditional(features).squeeze(1)
        projection = torch.sum(
            self.condition_projection(condition) * features, dim=1
        ) / math.sqrt(features.shape[1])
        score = score + projection
        return (score, features) if return_features else score


def initialize_orthogonal(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            weight = getattr(child, "weight_orig", child.weight)
            nn.init.orthogonal_(weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def diff_augment(
    images: torch.Tensor,
    *,
    translation_fraction: float = 0.125,
    cutout_fraction: float = 0.35,
) -> torch.Tensor:
    """AFM-safe differentiable translation, symmetry, and cutout augmentation."""

    output = images
    if bool(torch.rand((), device=images.device) < 0.5):
        output = torch.flip(output, dims=(-1,))
    if bool(torch.rand((), device=images.device) < 0.5):
        output = torch.flip(output, dims=(-2,))
    rotations = int(torch.randint(0, 4, (), device=images.device).item())
    output = torch.rot90(output, rotations, dims=(-2, -1))
    maximum = max(int(round(images.shape[-1] * translation_fraction)), 1)
    shift_y = int(
        torch.randint(-maximum, maximum + 1, (), device=images.device).item()
    )
    shift_x = int(
        torch.randint(-maximum, maximum + 1, (), device=images.device).item()
    )
    output = torch.roll(output, shifts=(shift_y, shift_x), dims=(-2, -1))
    if cutout_fraction > 0 and bool(torch.rand((), device=images.device) < 0.5):
        height, width = output.shape[-2:]
        cut_height = max(int(round(height * cutout_fraction)), 1)
        cut_width = max(int(round(width * cutout_fraction)), 1)
        center_y = int(torch.randint(0, height, (), device=images.device).item())
        center_x = int(torch.randint(0, width, (), device=images.device).item())
        yy = torch.arange(height, device=images.device)
        xx = torch.arange(width, device=images.device)
        mask_y = (yy - center_y).abs() > cut_height // 2
        mask_x = (xx - center_x).abs() > cut_width // 2
        mask = (mask_y[:, None] | mask_x[None, :]).to(output.dtype)
        output = output * mask[None, None]
    return output


class MorphologyConditionLoss(nn.Module):
    """Differentiable AFM descriptor consistency for the generated unit shape."""

    def __init__(self, condition_scaler: ConditionScaler, resolution: int) -> None:
        super().__init__()
        self.columns = list(condition_scaler.columns)
        self.register_buffer(
            "mean", torch.as_tensor(condition_scaler.mean, dtype=torch.float32)
        )
        self.register_buffer(
            "scale", torch.as_tensor(condition_scaler.scale, dtype=torch.float32)
        )
        yy, xx = np.indices((resolution, resolution))
        radius = np.hypot(yy - resolution / 2, xx - resolution / 2)
        edges = np.linspace(1.0, float(radius.max()), 25)
        masks = [
            ((radius >= low) & (radius < high)).astype(np.float32)
            for low, high in zip(edges[:-1], edges[1:])
        ]
        self.register_buffer("radial_masks", torch.from_numpy(np.stack(masks)))
        self.register_buffer(
            "log_frequency",
            torch.log(torch.from_numpy(0.5 * (edges[:-1] + edges[1:])).float()),
        )
        self.selected_columns = [
            "unit_ra",
            "unit_psd_mid_fraction",
            "unit_psd_high_fraction",
            "unit_psd_slope",
            "log_unit_anisotropy_ratio",
            "unit_skewness",
            "unit_kurtosis",
        ]
        self.selected_indices = [
            self.columns.index(column) for column in self.selected_columns
        ]

    def descriptors(self, images: torch.Tensor) -> torch.Tensor:
        centered = images - images.mean(dim=(2, 3), keepdim=True)
        variance = centered.square().mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        unit = centered / variance.sqrt()
        ra = unit.abs().mean(dim=(1, 2, 3))
        skew = unit.pow(3).mean(dim=(1, 2, 3))
        kurtosis = unit.pow(4).mean(dim=(1, 2, 3))

        power = torch.fft.fftshift(
            torch.fft.fft2(unit[:, 0]), dim=(-2, -1)
        ).abs().square()
        mask = self.radial_masks
        radial = (
            power[:, None] * mask[None]
        ).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1.0)[None]
        fractions = radial / radial.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mid = fractions[:, 8:16].sum(dim=1)
        high = fractions[:, 16:].sum(dim=1)
        log_power = torch.log(radial.clamp_min(1e-8))
        frequency = self.log_frequency
        centered_frequency = frequency - frequency.mean()
        slope = torch.sum(
            (log_power - log_power.mean(dim=1, keepdim=True))
            * centered_frequency[None],
            dim=1,
        ) / centered_frequency.square().sum().clamp_min(1e-8)

        frequency_axis = torch.fft.fftshift(
            torch.fft.fftfreq(images.shape[-1], device=images.device)
        )
        fx = frequency_axis[None, None, :]
        fy = frequency_axis[None, :, None]
        total = power.sum(dim=(-2, -1)).clamp_min(1e-8)
        moment_x = (power * fx.square()).sum(dim=(-2, -1)) / total
        moment_y = (power * fy.square()).sum(dim=(-2, -1)) / total
        anisotropy = torch.maximum(moment_x, moment_y) / torch.minimum(
            moment_x, moment_y
        ).clamp_min(1e-8)
        log_anisotropy = 0.5 * torch.log(anisotropy.clamp_min(1.0))

        values = {
            "unit_ra": ra,
            "unit_psd_mid_fraction": mid,
            "unit_psd_high_fraction": high,
            "unit_psd_slope": slope,
            "log_unit_anisotropy_ratio": log_anisotropy,
            "unit_skewness": skew,
            "unit_kurtosis": kurtosis,
        }
        return torch.stack([values[column] for column in self.selected_columns], dim=1)

    def forward(
        self, images: torch.Tensor, standardized_condition: torch.Tensor
    ) -> torch.Tensor:
        raw = self.descriptors(images)
        indices = torch.as_tensor(
            self.selected_indices, dtype=torch.long, device=images.device
        )
        predicted_z = (raw - self.mean[indices]) / self.scale[indices]
        target_z = standardized_condition[:, indices]
        return F.smooth_l1_loss(predicted_z, target_z)


def gradient_statistics(images: torch.Tensor) -> torch.Tensor:
    dx = images[..., :, 1:] - images[..., :, :-1]
    dy = images[..., 1:, :] - images[..., :-1, :]
    return torch.stack(
        [
            dx.abs().mean(dim=(1, 2, 3)),
            dy.abs().mean(dim=(1, 2, 3)),
            dx.square().mean(dim=(1, 2, 3)).sqrt(),
            dy.square().mean(dim=(1, 2, 3)).sqrt(),
        ],
        dim=1,
    )


def calibrate_random_fields(
    fields: np.ndarray,
    standardized_condition: np.ndarray,
    *,
    condition_scaler: ConditionScaler,
    device: torch.device,
    steps: int,
    learning_rate: float,
    content_weight: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Optimize scientific descriptors while preserving the stochastic seed.

    This is not retrieval or pixel matching to a measured AFM. The only target
    is the RHEED-predicted morphology condition.
    """

    initial = torch.as_tensor(
        np.asarray(fields)[:, None], dtype=torch.float32, device=device
    )
    parameter = nn.Parameter(initial.clone())
    condition = torch.as_tensor(
        np.repeat(
            np.asarray(standardized_condition, dtype=np.float32)[None],
            len(initial),
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    objective = MorphologyConditionLoss(
        condition_scaler, int(initial.shape[-1])
    ).to(device)
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    history: list[dict[str, float]] = []
    for step in range(1, max(int(steps), 1) + 1):
        optimizer.zero_grad(set_to_none=True)
        projected = project_unit_rq_torch(parameter)
        condition_loss = objective(projected, condition)
        content_loss = F.smooth_l1_loss(projected, initial)
        total = condition_loss + float(content_weight) * content_loss
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"non-finite descriptor calibration loss at step {step}"
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 5.0)
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(-8.0, 8.0)
        if step == 1 or step == steps or step % 10 == 0:
            history.append(
                {
                    "step": float(step),
                    "condition_loss": float(condition_loss.detach().cpu()),
                    "content_loss": float(content_loss.detach().cpu()),
                    "total_loss": float(total.detach().cpu()),
                }
            )
    with torch.no_grad():
        calibrated = project_unit_rq_torch(parameter).cpu().numpy()[:, 0]
    return calibrated.astype(np.float32), history
