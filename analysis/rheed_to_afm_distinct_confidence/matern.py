from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0))))


def _standardized_moments(values: np.ndarray) -> tuple[float, float, float]:
    centered = values - float(np.mean(values))
    centered /= max(float(np.sqrt(np.mean(centered**2))), 1e-8)
    return (
        float(np.mean(np.abs(centered))),
        float(stats.skew(centered)),
        float(stats.kurtosis(centered, fisher=False)),
    )


def _height_distribution(
    *,
    count: int,
    target_ra: float,
    target_skewness: float,
    target_kurtosis: float,
) -> np.ndarray:
    """Parametric non-Gaussian height distribution fit to AFM descriptors."""

    probability = (np.arange(count, dtype=float) + 0.5) / count
    normal = stats.norm.ppf(probability)
    h2 = normal**2 - 1.0
    h3 = normal**3 - 3.0 * normal
    h4 = normal**4 - 6.0 * normal**2 + 3.0
    target = np.asarray(
        [
            np.clip(target_ra, 0.55, 0.90),
            np.clip(target_skewness, -2.0, 2.0),
            np.clip(target_kurtosis, 1.8, 8.0),
        ]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = normal + parameters[0] * h2 + parameters[1] * h3 + parameters[2] * h4
        measured = np.asarray(_standardized_moments(values))
        return np.asarray(
            [
                4.0 * (measured[0] - target[0]),
                measured[1] - target[1],
                0.35 * (measured[2] - target[2]),
            ]
        )

    result = optimize.least_squares(
        residual,
        np.zeros(3),
        bounds=([-0.55, -0.25, -0.08], [0.55, 0.25, 0.08]),
        max_nfev=80,
    )
    values = normal + result.x[0] * h2 + result.x[1] * h3 + result.x[2] * h4
    values -= float(np.mean(values))
    values /= max(float(np.sqrt(np.mean(values**2))), 1e-8)
    return np.sort(values.astype(np.float32))


def _rank_map(field: np.ndarray, sorted_values: np.ndarray) -> np.ndarray:
    order = np.argsort(field.ravel(), kind="mergesort")
    result = np.empty(field.size, dtype=np.float32)
    result[order] = sorted_values
    return result.reshape(field.shape)


@dataclass(frozen=True)
class DescriptorMaternGenerator:
    """Stochastic, non-retrieval AFM generator with explicit morphology knobs.

    A RHEED-predicted condition sets Rq, correlation scale, PSD slope,
    anisotropy, the coarse/fine spectral mixture, and height-distribution
    moments. Random phase and orientation supply stochasticity without loading
    or retrieving any training AFM at inference.
    """

    condition_scaler: ConditionScaler
    resolution: int = 128
    scan_size_nm: float = 1000.0
    correlation_scale: float = 1.00
    slope_scale: float = 1.08

    def raw_condition(self, condition_z: np.ndarray) -> dict[str, float]:
        raw = self.condition_scaler.inverse_transform(
            np.asarray(condition_z, dtype=float)[None]
        )[0]
        return {
            column: float(value)
            for column, value in zip(self.condition_scaler.columns, raw, strict=False)
        }

    def generate(
        self,
        condition_z: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        raw = self.raw_condition(condition_z)
        rng = np.random.default_rng(int(seed))
        n = int(self.resolution)
        pixel_nm = self.scan_size_nm / max(n - 1, 1)
        corr_nm = float(np.exp(raw["log_unit_autocorr_length_nm"]))
        correlation_px = np.clip(
            self.correlation_scale * corr_nm / pixel_nm, 0.8, n / 3
        )
        beta = np.clip(-raw["unit_psd_slope"] * self.slope_scale, 1.4, 5.5)
        anisotropy = np.clip(np.exp(raw["log_unit_anisotropy_ratio"]), 1.0, 4.0)
        axis_scale = np.sqrt(anisotropy)
        angle = rng.uniform(0.0, np.pi)
        fy = np.fft.fftfreq(n)
        fx = np.fft.fftfreq(n)
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        rotated_x = np.cos(angle) * xx + np.sin(angle) * yy
        rotated_y = -np.sin(angle) * xx + np.cos(angle) * yy
        q2 = (rotated_x * axis_scale) ** 2 + (rotated_y / axis_scale) ** 2
        coarse_power = (1.0 + (2.0 * np.pi * correlation_px) ** 2 * q2) ** (-beta / 2.0)
        fine_scale = max(correlation_px / 3.5, 0.55)
        fine_power = (1.0 + (2.0 * np.pi * fine_scale) ** 2 * q2) ** (
            -max(beta - 0.45, 1.2) / 2.0
        )
        columns = self.condition_scaler.columns
        mid_z = float(condition_z[columns.index("unit_psd_mid_fraction")])
        high_z = float(condition_z[columns.index("unit_psd_high_fraction")])
        # AFM radial PSDs in this cohort place only ~1e-4 of their binned
        # power in the highest third. A visually tempting 5--30% fine-scale
        # mixture therefore creates nonphysical salt-and-pepper sharpness.
        # Keep a small, condition-responsive component instead.
        fine_weight = 0.0002 + 0.0028 * _sigmoid(0.7 * high_z + 0.3 * mid_z)
        power = (1.0 - fine_weight) * coarse_power + fine_weight * fine_power
        power[0, 0] = 0.0
        white = rng.normal(size=(n, n))
        spectrum = np.fft.fft2(white) * np.sqrt(np.maximum(power, 0.0))
        field = np.fft.ifft2(spectrum).real
        sorted_heights = _height_distribution(
            count=n * n,
            target_ra=raw["unit_ra"],
            target_skewness=raw["unit_skewness"],
            target_kurtosis=raw["unit_kurtosis"],
        )
        field = _rank_map(field, sorted_heights)
        return project_unit_rq_np(field).astype(np.float32)

    def generate_ensemble(
        self,
        condition_z: np.ndarray,
        *,
        draws: int,
        seed: int,
    ) -> list[np.ndarray]:
        return [
            self.generate(condition_z, seed=int(seed) + draw)
            for draw in range(int(draws))
        ]
