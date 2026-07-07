"""Exposure-invariant preprocessing for RHEED shape-bag inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.rheed.frame_quality import finite_float, frame_to_gray_float32, resize_image


try:  # pragma: no cover - availability depends on the local environment.
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None

try:  # pragma: no cover - availability depends on the local environment.
    import imageio.v3 as iio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    iio = None


EPS = 1e-8
DEFAULT_CHANNEL_NAMES = [
    "pclip_norm",
    "log_bgsub",
    "local_zscore",
    "dog_response",
    "ridge_or_edge_response",
    "soft_spot_streak_mask",
]


@dataclass
class ShapePreprocessResult:
    raw_gray: np.ndarray
    channels: dict[str, np.ndarray]
    artifact_mask: np.ndarray
    audit_features: dict[str, float]


def _finite_image(image: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def robust_rescale(
    image: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    values = _finite_image(image)
    low = finite_float(np.percentile(values, low_percentile), 0.0)
    high = finite_float(np.percentile(values, high_percentile), 1.0)
    if high <= low + EPS:
        low = finite_float(values.min(), 0.0)
        high = finite_float(values.max(), 1.0)
    scaled = (values - low) / max(high - low, EPS)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32, copy=False)


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    values = _finite_image(image)
    if ndimage is not None:
        return ndimage.gaussian_filter(values, sigma=sigma).astype(np.float32, copy=False)
    radius = max(1, int(round(sigma * 2)))
    padded = np.pad(values, radius, mode="reflect")
    output = np.zeros_like(values, dtype=np.float32)
    kernel_size = 2 * radius + 1
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            output[y, x] = float(np.mean(padded[y : y + kernel_size, x : x + kernel_size]))
    return output


def local_mean_std(image: np.ndarray, window_size: int = 31) -> tuple[np.ndarray, np.ndarray]:
    values = _finite_image(image)
    if ndimage is not None:
        mean = ndimage.uniform_filter(values, size=window_size)
        mean_sq = ndimage.uniform_filter(values * values, size=window_size)
    else:
        mean = gaussian_filter(values, sigma=max(1.0, window_size / 6.0))
        mean_sq = gaussian_filter(values * values, sigma=max(1.0, window_size / 6.0))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return mean.astype(np.float32, copy=False), np.sqrt(variance + EPS).astype(np.float32, copy=False)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    values = _finite_image(image)
    if ndimage is not None:
        gx = ndimage.sobel(values, axis=1)
        gy = ndimage.sobel(values, axis=0)
    else:
        gy, gx = np.gradient(values)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def multi_sigma_dog(image: np.ndarray, sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)) -> np.ndarray:
    values = _finite_image(image)
    responses = []
    for sigma in sigmas:
        narrow = gaussian_filter(values, sigma=sigma)
        wide = gaussian_filter(values, sigma=sigma * 1.8)
        responses.append(narrow - wide)
    stack = np.stack(responses, axis=0)
    index = np.argmax(np.abs(stack), axis=0)
    combined = np.take_along_axis(stack, index[None, :, :], axis=0)[0]
    return combined.astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)


def adaptive_soft_mask(
    log_bgsub: np.ndarray,
    dog_response: np.ndarray,
    local_zscore: np.ndarray,
    artifact_mask: np.ndarray,
) -> np.ndarray:
    score = (
        0.45 * robust_rescale(np.maximum(log_bgsub, 0.0))
        + 0.35 * robust_rescale(np.abs(dog_response))
        + 0.20 * robust_rescale(np.maximum(local_zscore, 0.0))
    )
    valid = 1.0 - np.clip(artifact_mask, 0.0, 1.0)
    center = finite_float(np.percentile(score[valid > 0.5], 82) if np.any(valid > 0.5) else np.percentile(score, 82), 0.5)
    spread = max(finite_float(np.percentile(score, 95) - np.percentile(score, 50), 0.1), 0.05)
    soft = _sigmoid((score - center) / spread * 3.0) * valid
    if ndimage is not None:
        soft = ndimage.gaussian_filter(soft, sigma=0.7)
    return np.clip(soft, 0.0, 1.0).astype(np.float32, copy=False)


def artifact_mask_from_raw(gray: np.ndarray, pclip_norm: np.ndarray) -> np.ndarray:
    raw = frame_to_gray_float32(gray)
    height, width = raw.shape
    dark = raw < 0.025
    saturated = raw > 0.985
    edge_w = max(2, int(width * 0.08))
    column_mean = raw.mean(axis=0)
    center_mean = finite_float(raw[:, width // 4 : max(width // 4 + 1, width * 3 // 4)].mean(), finite_float(raw.mean(), 0.0))
    shadow_columns = column_mean < max(0.04, center_mean * 0.45)
    shadow = np.zeros_like(raw, dtype=np.float32)
    if np.any(shadow_columns[:edge_w]) or np.any(shadow_columns[-edge_w:]):
        shadow[:, shadow_columns] = 1.0
    mask = np.maximum.reduce([dark.astype(np.float32), saturated.astype(np.float32), shadow])
    if ndimage is not None:
        mask = ndimage.binary_dilation(mask > 0, iterations=1).astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def preprocess_frame_for_shape(
    frame: np.ndarray,
    *,
    image_size: int = 256,
    local_window: int = 31,
    log_alpha: float = 12.0,
) -> ShapePreprocessResult:
    raw = resize_image(frame_to_gray_float32(frame), size=image_size)
    pclip = robust_rescale(raw, low_percentile=1, high_percentile=99, out_min=0, out_max=1)
    logged = np.log1p(log_alpha * pclip).astype(np.float32)
    background = gaussian_filter(logged, sigma=max(8.0, image_size / 18.0))
    log_bgsub = robust_rescale(logged - background, low_percentile=2, high_percentile=98, out_min=-1, out_max=1)

    mean, std = local_mean_std(pclip, window_size=local_window)
    local_z = np.clip((pclip - mean) / np.maximum(std, 0.02), -4.0, 4.0)
    local_zscore = robust_rescale(local_z, low_percentile=1, high_percentile=99, out_min=-1, out_max=1)

    dog_raw = multi_sigma_dog(pclip)
    dog_response = robust_rescale(np.abs(dog_raw), low_percentile=1, high_percentile=99, out_min=0, out_max=1)
    edge_response = robust_rescale(gradient_magnitude(log_bgsub), low_percentile=1, high_percentile=99, out_min=0, out_max=1)
    artifact_mask = artifact_mask_from_raw(raw, pclip)
    soft_mask = adaptive_soft_mask(log_bgsub, dog_response, local_zscore, artifact_mask)

    channels = {
        "pclip_norm": pclip,
        "log_bgsub": log_bgsub,
        "local_zscore": local_zscore,
        "dog_response": dog_response,
        "ridge_or_edge_response": edge_response,
        "soft_spot_streak_mask": soft_mask,
    }
    audit = {
        "raw_mean": finite_float(raw.mean()),
        "raw_std": finite_float(raw.std()),
        "raw_p01": finite_float(np.percentile(raw, 1)),
        "raw_p99": finite_float(np.percentile(raw, 99)),
        "artifact_fraction": finite_float(artifact_mask.mean()),
        "mask_confidence": finite_float(np.clip(soft_mask.mean() * 8.0 + np.percentile(soft_mask, 98), 0.0, 1.0)),
        "snr_score": finite_float(np.clip(np.std(log_bgsub) * 2.5, 0.0, 1.0)),
    }
    for key, value in list(channels.items()):
        channels[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return ShapePreprocessResult(
        raw_gray=np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        channels=channels,
        artifact_mask=artifact_mask,
        audit_features=audit,
    )


def channels_to_tensor(channels: dict[str, np.ndarray], names: list[str] | None = None) -> np.ndarray:
    ordered = names or DEFAULT_CHANNEL_NAMES
    return np.stack([channels[name].astype(np.float32, copy=False) for name in ordered], axis=0)


def read_grayscale_image(path: Path, *, image_size: int = 256) -> np.ndarray:
    if iio is None:
        raise RuntimeError("imageio is required to read candidate PNG frames.")
    image = iio.imread(path)
    return resize_image(frame_to_gray_float32(image), size=image_size).astype(np.float32, copy=False)


def photometric_perturbations(frame: np.ndarray) -> dict[str, np.ndarray]:
    raw = frame_to_gray_float32(frame)
    height, width = raw.shape
    yy, xx = np.indices((height, width))
    gradient = (xx / max(width - 1, 1) - 0.5) * 0.25 + (yy / max(height - 1, 1) - 0.5) * 0.15
    rng = np.random.default_rng(123)
    perturbations: dict[str, np.ndarray] = {"original": raw}
    for scale in (0.5, 0.75, 1.25, 1.75):
        perturbations[f"brightness_{scale:g}"] = np.clip(raw * scale, 0.0, 1.0)
    for scale in (0.5, 1.5):
        perturbations[f"contrast_{scale:g}"] = np.clip((raw - 0.5) * scale + 0.5, 0.0, 1.0)
    for gamma in (0.6, 1.8):
        perturbations[f"gamma_{gamma:g}"] = np.clip(raw, 0.0, 1.0) ** gamma
    perturbations["lowfreq_gradient"] = np.clip(raw + gradient, 0.0, 1.0)
    perturbations["mild_noise"] = np.clip(raw + rng.normal(0.0, 0.025, size=raw.shape), 0.0, 1.0)
    perturbations["mild_blur"] = gaussian_filter(raw, sigma=1.0)
    return {key: value.astype(np.float32, copy=False) for key, value in perturbations.items()}


def save_gray_png(path: Path, image: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(image, dtype=np.float32)
    if vmin is None:
        vmin = finite_float(values.min(), 0.0)
    if vmax is None:
        vmax = finite_float(values.max(), 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, values, cmap="gray", vmin=vmin, vmax=vmax)


def make_rgb_overlay(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.15, 0.05)) -> np.ndarray:
    gray = robust_rescale(base)
    soft = np.clip(mask, 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    color_arr = np.asarray(color, dtype=np.float32)
    return np.clip(rgb * (1.0 - 0.45 * soft[..., None]) + color_arr * (0.55 * soft[..., None]), 0.0, 1.0)

