from __future__ import annotations

import numpy as np


def spatial_features(frame: np.ndarray) -> dict[str, float]:
    x = frame.astype(np.float32) / 255.0
    values = x.ravel()
    hist, _ = np.histogram(values, bins=32, range=(0, 1), density=False)
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)) / np.log2(32))
    gy, gx = np.gradient(x)
    grad_mag = np.hypot(gx, gy)
    lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    power = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    rows, cols = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    low = np.mean(power[(radius > 0) & (radius <= 0.08)]) if np.any((radius > 0) & (radius <= 0.08)) else 0.0
    mid = np.mean(power[(radius > 0.08) & (radius <= 0.22)]) if np.any((radius > 0.08) & (radius <= 0.22)) else 0.0
    high = np.mean(power[radius > 0.22]) if np.any(radius > 0.22) else 0.0
    row_proj = x.mean(axis=1)
    col_proj = x.mean(axis=0)
    h_band = power[np.abs(fy) < 0.03, :].mean()
    v_band = power[:, np.abs(fx) < 0.03].mean()
    margin = max(1, min(x.shape) // 5)
    center = x[margin:-margin, margin:-margin] if x.shape[0] > 2 * margin and x.shape[1] > 2 * margin else x
    border_mask = np.ones_like(x, dtype=bool)
    border_mask[margin:-margin, margin:-margin] = False
    border = x[border_mask]
    features = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "entropy": entropy,
        "laplacian_var": float(lap.var()),
        "sobel_horizontal_energy": float(np.mean(gx**2)),
        "sobel_vertical_energy": float(np.mean(gy**2)),
        "gradient_anisotropy": float((np.mean(gx**2) - np.mean(gy**2)) / (np.mean(gx**2) + np.mean(gy**2) + 1e-12)),
        "edge_density": float(np.mean(grad_mag > np.percentile(grad_mag, 90))),
        "fft_low": float(low),
        "fft_mid": float(mid),
        "fft_high": float(high),
        "fft_high_low_ratio": float(high / (low + 1e-12)),
        "fft_horizontal_vertical_anisotropy": float((h_band - v_band) / (h_band + v_band + 1e-12)),
        "row_proj_mean": float(row_proj.mean()),
        "row_proj_std": float(row_proj.std()),
        "row_proj_range": float(row_proj.max() - row_proj.min()),
        "col_proj_mean": float(col_proj.mean()),
        "col_proj_std": float(col_proj.std()),
        "col_proj_range": float(col_proj.max() - col_proj.min()),
        "center_border_ratio": float(center.mean() / (border.mean() + 1e-12)) if border.size else 1.0,
    }
    for q in (1, 5, 25, 50, 75, 95, 99):
        features[f"p{q:02d}"] = float(np.percentile(values, q))
    return features


def keyframe_feature_vector(frames: np.ndarray, keyframe_offset: int) -> tuple[list[str], np.ndarray]:
    features = spatial_features(frames[keyframe_offset])
    names = sorted(features)
    return names, np.asarray([features[name] for name in names], dtype=float)


def temporal_feature_vector(frames: np.ndarray) -> tuple[list[str], np.ndarray]:
    per_frame = [spatial_features(frame) for frame in frames]
    base_names = sorted(per_frame[0])
    values = np.asarray([[features[name] for name in base_names] for features in per_frame], dtype=float)
    names: list[str] = []
    feats: list[float] = []
    t = np.arange(values.shape[0], dtype=float)
    for i, name in enumerate(base_names):
        series = values[:, i]
        slope = float(np.polyfit(t, series, 1)[0]) if len(series) > 1 else 0.0
        for agg_name, value in (
            ("mean", np.mean(series)),
            ("std", np.std(series)),
            ("min", np.min(series)),
            ("max", np.max(series)),
            ("slope", slope),
            ("first_last_diff", series[-1] - series[0]),
        ):
            names.append(f"{name}_{agg_name}")
            feats.append(float(value))
    arr = frames.astype(np.float32) / 255.0
    diffs = np.abs(np.diff(arr, axis=0))
    pixel_var = np.var(arr, axis=0)
    frame_mean = arr.mean(axis=(1, 2))
    sharpness = np.asarray([spatial_features(frame)["laplacian_var"] for frame in frames], dtype=float)
    anisotropy = np.asarray([spatial_features(frame)["gradient_anisotropy"] for frame in frames], dtype=float)
    extras = {
        "consecutive_diff_mean": float(diffs.mean()) if diffs.size else 0.0,
        "consecutive_diff_std": float(diffs.std()) if diffs.size else 0.0,
        "temporal_pixel_variance_mean": float(pixel_var.mean()),
        "temporal_pixel_variance_p95": float(np.percentile(pixel_var, 95)),
        "brightness_drift": float(frame_mean[-1] - frame_mean[0]),
        "sharpness_drift": float(sharpness[-1] - sharpness[0]),
        "gradient_anisotropy_drift": float(anisotropy[-1] - anisotropy[0]),
    }
    names.extend(sorted(extras))
    feats.extend([extras[name] for name in sorted(extras)])
    return names, np.asarray(feats, dtype=float)
