from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

from .afm_descriptors import descriptor_distance
from .common import repo_path, write_csv, write_json
from .rq_disentanglement import project_unit_rq_np, rq_np


def load_shape(path: str) -> np.ndarray:
    arr = np.load(repo_path(path)).astype(np.float32)
    if arr.shape != (256, 256):
        arr = resize(arr, (256, 256), order=1, preserve_range=True, anti_aliasing=True)
    return project_unit_rq_np(arr)


def compare_shapes(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ar = project_unit_rq_np(a)
    br = project_unit_rq_np(b)
    ssim = float(structural_similarity(ar, br, data_range=max(np.percentile(ar, 99) - np.percentile(ar, 1), 1e-6)))
    shift, _, _ = phase_cross_correlation(ar, br, upsample_factor=1)
    bal = np.roll(br, tuple(int(round(x)) for x in shift), axis=(0, 1))
    aligned = float(structural_similarity(ar, bal, data_range=max(np.percentile(ar, 99) - np.percentile(ar, 1), 1e-6)))
    desc = descriptor_distance(ar, br)
    return {
        "raw_ssim": ssim,
        "translation_aligned_ssim": aligned,
        "multiscale_ssim": float(np.mean([ssim, structural_similarity(resize(ar, (128, 128), anti_aliasing=True), resize(br, (128, 128), anti_aliasing=True), data_range=4)])),
        "lpips": np.nan,
        "height_histogram_wasserstein": float(wasserstein_distance(ar.ravel(), br.ravel())),
        "normalized_psd_distance": desc["normalized_psd_log_distance"],
        "correlation_length_relative_difference": desc["correlation_length_relative_error"],
        "anisotropy_difference": desc["anisotropy_error"],
        "rq_difference": abs(rq_np(ar) - rq_np(br)),
    }


def same_growth_similarity(decoder_manifest: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows = []
    for sid, g in decoder_manifest.groupby("sample_id"):
        if len(g) < 2:
            continue
        rep = g[g["representative_for_sample"].astype(bool)]
        ref_row = rep.iloc[0] if len(rep) else g.iloc[0]
        ref = load_shape(json.loads(ref_row["unit_shape_paths"])["256"])
        for _, row in g.iterrows():
            if row["afm_file_id"] == ref_row["afm_file_id"]:
                continue
            comp = load_shape(json.loads(row["unit_shape_paths"])["256"])
            rows.append({"sample_id": sid, "reference_afm_file_id": ref_row["afm_file_id"], "comparison_afm_file_id": row["afm_file_id"], **compare_shapes(ref, comp)})
    df = pd.DataFrame(rows)
    summary_rows = []
    for sid, g in df.groupby("sample_id"):
        summary_rows.append({"sample_id": sid, "same_sample_best_ssim": float(g["translation_aligned_ssim"].max()), "same_sample_median_ssim": float(g["translation_aligned_ssim"].median()), "same_sample_worst_ssim": float(g["translation_aligned_ssim"].min()), "median_psd_distance": float(g["normalized_psd_distance"].median())})
    summary_df = pd.DataFrame(summary_rows)
    summary = {
        "median_raw_ssim": float(df["raw_ssim"].median()),
        "median_translation_aligned_ssim": float(df["translation_aligned_ssim"].median()),
        "median_multiscale_ssim": float(df["multiscale_ssim"].median()),
        "median_descriptor_psd_distance": float(df["normalized_psd_distance"].median()),
        "lpips_available": False,
        "supports_90_percent_pixel_similarity": bool(df["translation_aligned_ssim"].median() >= 0.9),
    }
    out = repo_path(config["output_root"])
    write_csv(df, out / "same_growth_afm_similarity.csv")
    write_csv(summary_df, out / "same_growth_similarity_by_sample.csv")
    write_json(summary, out / "same_growth_similarity_summary.json")
    return df, summary, summary_df


def oracle_retrieval(manifest: pd.DataFrame, bank: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    desc_cols = ["psd_low_fraction", "psd_mid_fraction", "psd_high_fraction", "correlation_length_nm", "anisotropy", "height_skewness", "height_kurtosis"]
    for _, h in manifest.iterrows():
        sid = str(h["sample_id"])
        train = bank[bank["growth_run_id"].astype(str) != sid].copy()
        rq_dist = np.abs(np.log(train["rq_nm"].to_numpy(float)) - np.log(float(h["primary_rq_nm_median"])))
        top = train.iloc[int(np.argmin(rq_dist))]
        rows.append({"sample_id": sid, "oracle_type": "ORACLE_A_TRUE_RQ", "candidate_group_id": top["growth_run_id"], "distance": float(rq_dist.min()), "oracle_label": "ORACLE DEVELOPMENT UPPER BOUND USES HELD-OUT TARGET INFORMATION NOT A DEPLOYABLE MODEL"})
        hv = bank[bank["sample_id"].astype(str) == sid][desc_cols].iloc[0].to_numpy(float)
        tv = train[desc_cols].to_numpy(float)
        d = np.linalg.norm((tv - hv) / (np.nanstd(tv, axis=0) + 1e-6), axis=1)
        top = train.iloc[int(np.argmin(d))]
        rows.append({"sample_id": sid, "oracle_type": "ORACLE_B_TRUE_AFM_DESCRIPTOR", "candidate_group_id": top["growth_run_id"], "distance": float(d.min()), "oracle_label": "ORACLE DEVELOPMENT UPPER BOUND USES HELD-OUT TARGET INFORMATION NOT A DEPLOYABLE MODEL"})
    out = pd.DataFrame(rows)
    write_csv(out, repo_path(config["output_root"]) / "oracle_retrieval_ceiling.csv")
    return out, out.groupby("oracle_type")["distance"].median().reset_index(name="median_distance")
