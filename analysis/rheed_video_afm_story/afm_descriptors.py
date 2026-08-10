from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage, stats

from .common import repo_path
from .rq_disentanglement import ra_np, rq_np


def gradients(
    arr: np.ndarray, scan_size_um: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    pixel_um = scan_size_um / max(arr.shape[0] - 1, 1)
    gy, gx = np.gradient(arr.astype(float), pixel_um, pixel_um)
    return gx, gy


def radial_psd(arr: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    a = arr.astype(float) - float(np.mean(arr))
    fft = np.fft.fftshift(np.fft.fft2(a))
    power = np.abs(fft) ** 2
    h, w = a.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    edges = np.linspace(1, rr.max(), bins + 1)
    vals = []
    centers = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (rr >= lo) & (rr < hi)
        vals.append(float(np.mean(power[mask])) if mask.any() else 0.0)
        centers.append(float((lo + hi) / 2))
    return np.asarray(centers), np.asarray(vals)


def psd_band_descriptors(arr: np.ndarray) -> dict[str, float]:
    freq, p = radial_psd(arr)
    total = float(np.sum(p) + 1e-12)
    n = len(p)
    low = float(np.sum(p[: n // 3]) / total)
    mid = float(np.sum(p[n // 3 : 2 * n // 3]) / total)
    high = float(np.sum(p[2 * n // 3 :]) / total)
    finite = (p > 0) & (freq > 0)
    slope = (
        float(np.polyfit(np.log(freq[finite]), np.log(p[finite] + 1e-12), 1)[0])
        if finite.sum() >= 3
        else np.nan
    )
    return {
        "psd_low_fraction": low,
        "psd_mid_fraction": mid,
        "psd_high_fraction": high,
        "psd_slope": slope,
    }


def autocorr_lengths(
    arr: np.ndarray, scan_size_um: float = 1.0
) -> tuple[float, float, float, float]:
    a = arr.astype(float) - float(np.mean(arr))
    f = np.fft.fft2(a)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    ac /= float(ac.max() + 1e-12)
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    pixel_nm = scan_size_um * 1000.0 / max(h - 1, 1)

    def first_below(profile: np.ndarray) -> float:
        target = np.exp(-1)
        right = profile[len(profile) // 2 :]
        below = np.where(right < target)[0]
        return float((below[0] if len(below) else len(right) - 1) * pixel_nm)

    lx = first_below(ac[cy, :])
    ly = first_below(ac[:, cx])
    yy, xx = np.indices(ac.shape)
    labels = np.round(np.hypot(yy - cy, xx - cx)).astype(int)
    radial = ndimage.mean(ac, labels=labels, index=np.arange(0, min(h, w) // 2))
    below = np.where(radial < np.exp(-1))[0]
    lr = float((below[0] if len(below) else len(radial) - 1) * pixel_nm)
    anis = float(max(lx, ly) / max(min(lx, ly), 1e-6))
    return lr, lx, ly, anis


def orientation_entropy(gx: np.ndarray, gy: np.ndarray, bins: int = 18) -> float:
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(angle, bins=bins, range=(0, np.pi), weights=mag)
    prob = hist / (hist.sum() + 1e-12)
    ent = -np.sum(prob * np.log(prob + 1e-12)) / np.log(bins)
    return float(ent)


def describe_map(
    arr: np.ndarray, prefix: str, scan_size_um: float = 1.0
) -> dict[str, float]:
    a = np.asarray(arr, dtype=float)
    a = a - float(np.mean(a))
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    gx, gy = gradients(a, scan_size_um=scan_size_um)
    grad_mag = np.hypot(gx, gy)
    lr, lx, ly, anis = autocorr_lengths(a, scan_size_um=scan_size_um)
    desc = {
        f"{prefix}_rq": rq_np(a),
        f"{prefix}_ra": ra_np(a),
        f"{prefix}_robust_height_range": float(q99 - q01),
        f"{prefix}_p95_p5": float(q95 - q05),
        f"{prefix}_peak_to_valley": float(np.max(a) - np.min(a)),
        f"{prefix}_skewness": float(stats.skew(a.ravel())),
        f"{prefix}_kurtosis": float(stats.kurtosis(a.ravel(), fisher=False)),
        f"{prefix}_mean_abs_gradient": float(np.mean(np.abs(grad_mag))),
        f"{prefix}_rms_gradient": float(np.sqrt(np.mean(grad_mag**2))),
        f"{prefix}_slope_q05": float(np.percentile(grad_mag, 5)),
        f"{prefix}_slope_q50": float(np.percentile(grad_mag, 50)),
        f"{prefix}_slope_q95": float(np.percentile(grad_mag, 95)),
        f"{prefix}_autocorr_length_nm": lr,
        f"{prefix}_corr_length_x_nm": lx,
        f"{prefix}_corr_length_y_nm": ly,
        f"{prefix}_anisotropy_ratio": anis,
        f"{prefix}_gradient_orientation_entropy": orientation_entropy(gx, gy),
        f"{prefix}_q01": float(q01),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_q99": float(q99),
    }
    desc.update({f"{prefix}_{k}": v for k, v in psd_band_descriptors(a).items()})
    return desc


def descriptor_distance(
    true_shape: np.ndarray, pred_shape: np.ndarray
) -> dict[str, float]:
    t = describe_map(true_shape, "true_unit")
    p = describe_map(pred_shape, "pred_unit")
    freq_t, psd_t = radial_psd(true_shape)
    _, psd_p = radial_psd(pred_shape)
    log_psd = float(np.mean(np.abs(np.log1p(psd_t) - np.log1p(psd_p))))
    qt = np.percentile(true_shape, [1, 5, 25, 50, 75, 95, 99])
    qp = np.percentile(pred_shape, [1, 5, 25, 50, 75, 95, 99])
    return {
        "normalized_psd_log_distance": log_psd,
        "psd_low_band_error": abs(
            t["true_unit_psd_low_fraction"] - p["pred_unit_psd_low_fraction"]
        ),
        "psd_mid_band_error": abs(
            t["true_unit_psd_mid_fraction"] - p["pred_unit_psd_mid_fraction"]
        ),
        "psd_high_band_error": abs(
            t["true_unit_psd_high_fraction"] - p["pred_unit_psd_high_fraction"]
        ),
        "psd_slope_error": abs(t["true_unit_psd_slope"] - p["pred_unit_psd_slope"]),
        "correlation_length_abs_error_nm": abs(
            t["true_unit_autocorr_length_nm"] - p["pred_unit_autocorr_length_nm"]
        ),
        "correlation_length_relative_error": abs(
            t["true_unit_autocorr_length_nm"] - p["pred_unit_autocorr_length_nm"]
        )
        / max(t["true_unit_autocorr_length_nm"], 1e-6),
        "anisotropy_error": abs(
            t["true_unit_anisotropy_ratio"] - p["pred_unit_anisotropy_ratio"]
        ),
        "skewness_error": abs(t["true_unit_skewness"] - p["pred_unit_skewness"]),
        "kurtosis_error": abs(t["true_unit_kurtosis"] - p["pred_unit_kurtosis"]),
        "height_quantile_error": float(np.mean(np.abs(qt - qp))),
        "ra_rq_error": abs(t["true_unit_ra"] - p["pred_unit_ra"]),
    }


def write_descriptor_definitions(report_root: str | Path) -> None:
    text = """# AFM Descriptor Definitions

All descriptors are computed from plane-corrected physical height arrays in nm, never from rendered PNGs.

- Rq: root mean square of mean-centered height.
- Ra: mean absolute mean-centered height.
- robust height range: p99 - p01.
- p95-p5: 95th minus 5th height percentile.
- peak-to-valley: max - min, reported as an auxiliary outlier-sensitive descriptor.
- skewness/kurtosis: scipy moment descriptors on centered heights.
- gradient metrics: finite differences using scan-size-derived pixel spacing.
- radial PSD: FFT power averaged in radial frequency bins with DC excluded.
- PSD band fractions: low/mid/high thirds of radial PSD power normalized by total radial power.
- PSD slope: linear fit of log(power) vs log(radial frequency).
- autocorrelation length: first radial autocorrelation crossing below exp(-1), in nm.
- directional correlation lengths: same crossing along x and y center lines.
- anisotropy ratio: max directional length divided by min directional length.
- gradient orientation entropy: weighted entropy of gradient orientation modulo pi.
- Unit-shape descriptors: same morphology descriptors after mean-centering and unit-Rq normalization; amplitude descriptors become dimensionless.

No island/grain segmentation is used in Phase 3A because this repository does not yet contain a validated segmentation method.
"""
    path = repo_path(report_root) / "descriptor_definitions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
