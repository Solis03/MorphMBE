from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity
from skimage.transform import resize

from .afm_descriptors import descriptor_distance, describe_map, gradients
from .rq_disentanglement import physical_from_q, project_unit_rq_np, rq_np


COMPOSITE_WEIGHTS = {
    "normalized_psd_log_distance": 0.30,
    "correlation_length_relative_error": 0.25,
    "gradient_mae": 0.20,
    "height_quantile_error": 0.15,
    "ssim_penalty": 0.10,
}


def metric_grid(arr: np.ndarray, max_size: int = 128) -> np.ndarray:
    if max(arr.shape) <= max_size:
        return arr
    return resize(arr, (max_size, max_size), order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)


def reconstruction_metrics(true_unit: np.ndarray, pred_unit: np.ndarray, q_true_nm: float) -> dict[str, float]:
    t = project_unit_rq_np(metric_grid(true_unit))
    p = project_unit_rq_np(metric_grid(pred_unit))
    diff = t - p
    gx_t, gy_t = gradients(t)
    gx_p, gy_p = gradients(p)
    grad_mae = float(np.mean(np.abs(gx_t - gx_p)) + np.mean(np.abs(gy_t - gy_p))) / 2.0
    data_range = float(max(np.percentile(t, 99) - np.percentile(t, 1), np.percentile(p, 99) - np.percentile(p, 1), 1e-6))
    try:
        ssim = float(structural_similarity(t, p, data_range=data_range))
    except Exception:
        ssim = np.nan
    desc = descriptor_distance(t, p)
    zt = float(q_true_nm) * t
    zp = physical_from_q(p, float(q_true_nm))
    phys_t = describe_map(zt, "true_physical")
    phys_p = describe_map(zp, "pred_physical")
    metrics = {
        "unit_l1": float(np.mean(np.abs(diff))),
        "unit_rmse": float(np.sqrt(np.mean(diff**2))),
        "ssim": ssim,
        "gradient_mae": grad_mae,
        "physical_ra_error_nm": abs(phys_t["true_physical_ra"] - phys_p["pred_physical_ra"]),
        "physical_robust_height_range_error_nm": abs(phys_t["true_physical_robust_height_range"] - phys_p["pred_physical_robust_height_range"]),
        "physical_histogram_wasserstein_nm": float(wasserstein_distance(zt.ravel(), zp.ravel())),
        "reconstructed_rq_nm": rq_np(zp),
        "rq_architectural_consistency_error_nm": abs(rq_np(zp) - float(q_true_nm)),
    }
    metrics.update(desc)
    metrics["physical_psd_distance"] = desc["normalized_psd_log_distance"]
    metrics["physical_correlation_length_error_nm"] = desc["correlation_length_abs_error_nm"]
    metrics["ssim_penalty"] = 1.0 - ssim if np.isfinite(ssim) else 1.0
    metrics["composite_score"] = float(sum(COMPOSITE_WEIGHTS[k] * metrics[k] for k in COMPOSITE_WEIGHTS))
    return metrics


def summarize_metrics(scan_metrics: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_cols = [
        "unit_l1",
        "unit_rmse",
        "ssim",
        "gradient_mae",
        "normalized_psd_log_distance",
        "correlation_length_relative_error",
        "height_quantile_error",
        "anisotropy_error",
        "skewness_error",
        "kurtosis_error",
        "physical_histogram_wasserstein_nm",
        "rq_architectural_consistency_error_nm",
        "composite_score",
    ]
    rows = []
    for group_key, g in scan_metrics.groupby(keys):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, Any] = dict(zip(keys, group_key))
        row["N_scans"] = len(g)
        row["N_groups"] = g["growth_run_id"].nunique() if "growth_run_id" in g.columns else np.nan
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_median"] = float(g[col].median())
                row[f"{col}_mean"] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("composite_score_median")


def group_level_metrics(scan_metrics: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    metric_cols = [c for c in scan_metrics.columns if c.endswith("_error") or c in {"unit_l1", "unit_rmse", "ssim", "gradient_mae", "normalized_psd_log_distance", "correlation_length_relative_error", "height_quantile_error", "composite_score"}]
    grouped = scan_metrics.groupby(model_cols + ["growth_run_id"])[metric_cols].median().reset_index()
    return grouped
