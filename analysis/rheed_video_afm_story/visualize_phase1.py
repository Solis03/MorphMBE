from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import repo_path


def _save(fig: plt.Figure, path: Path, pdf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_key_images(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(repo_path(row["clip_cache_path"]))
    frames = data["frames_uint8"]
    key = int(data["keyframe_offset"])
    return frames[0], frames[key], frames[-1]


def paired_atlas(manifest: pd.DataFrame, path: Path, title: str, common_scale: bool = False) -> None:
    df = manifest[manifest["representative_afm_height_array"].astype(str) != ""].copy()
    if df.empty:
        return
    df = df.sort_values("primary_rq_nm_median" if "primary_rq_nm_median" in df else "best_available_rq_nm_median")
    n = len(df)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(8.5, max(2.0, n * 1.35)), squeeze=False)
    common_min = common_max = None
    if common_scale:
        vals = []
        for p in df["representative_afm_height_array"]:
            arr = np.load(repo_path(p))
            vals.extend(np.percentile(arr[np.isfinite(arr)], [2, 98]).tolist())
        common_min, common_max = float(min(vals)), float(max(vals))
    for r, (_, row) in enumerate(df.iterrows()):
        first, key, last = _load_key_images(row)
        for c, image in enumerate([first, key, last]):
            axes[r, c].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(["first", "keyframe", "last"][c], fontsize=8)
        afm = np.load(repo_path(row["representative_afm_height_array"]))
        finite = afm[np.isfinite(afm)]
        if common_scale:
            vmin, vmax = common_min, common_max
        else:
            vmin, vmax = np.percentile(finite, [2, 98])
        im = axes[r, 3].imshow(afm, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[r, 3].set_axis_off()
        if r == 0:
            axes[r, 3].set_title("representative AFM (nm)", fontsize=8)
        label = (
            f"{row['sample_id']}  Rq={row['primary_rq_nm_median']:.2f} nm  "
            f"n={int(row['primary_afm_scan_count'])} IQR={row['primary_rq_nm_iqr']:.2f}\n"
            f"{row.get('growth_stage', 'unknown')} / {row.get('video_stage', 'unknown')}"
        )
        axes[r, 0].text(-0.02, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(hspace=0.15, wspace=0.04, right=0.92)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.75])
    fig.colorbar(im, cax=cax, label="height (nm)")
    _save(fig, path)


def summary_figures(manifest: pd.DataFrame, quality: pd.DataFrame, scan_audit: pd.DataFrame, report_root: Path) -> None:
    fig_dir = report_root / "figures"
    primary = manifest[manifest["cohort_primary_1um"] & manifest["usable_for_modeling"]].copy()
    all_valid = manifest[manifest["cohort_exploratory_best_available"] & manifest["usable_for_modeling"]].copy()
    paired_atlas(primary, fig_dir / "paired_atlas_primary_1um_sorted_by_rq.png", "Primary 1 x 1 um paired atlas", common_scale=False)
    paired_atlas(primary, fig_dir / "paired_atlas_primary_1um_sorted_by_rq_common_scale.png", "Primary 1 x 1 um paired atlas, common height scale", common_scale=True)
    paired_atlas(all_valid, fig_dir / "paired_atlas_all_valid_sorted_by_rq.png", "All valid paired atlas", common_scale=False)
    paired_atlas(all_valid, fig_dir / "paired_atlas_all_valid_sorted_by_rq_common_scale.png", "All valid paired atlas, common height scale", common_scale=True)
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(primary["primary_rq_nm_median"], bins=min(12, max(4, len(primary) // 2)), color="#476b6b", edgecolor="white")
        ax.set_xlabel("median Rq (nm)")
        ax.set_ylabel("sample count")
        ax.set_title("Primary cohort Rq distribution")
        _save(fig, fig_dir / "rq_distribution.png", pdf=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for stage, group in primary.groupby("video_stage"):
            ax.scatter([stage] * len(group), group["primary_rq_nm_median"], label=str(stage), s=35)
        ax.set_ylabel("median Rq (nm)")
        ax.set_xlabel("video stage")
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Rq by stage/material")
        _save(fig, fig_dir / "rq_by_stage_material.png", pdf=False)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.scatter(primary["primary_rq_nm_median"], primary["primary_rq_nm_iqr"], s=35, color="#7a4c7a")
        for _, row in primary.iterrows():
            ax.text(row["primary_rq_nm_median"], row["primary_rq_nm_iqr"], str(row["sample_id"]), fontsize=6)
        ax.set_xlabel("median Rq (nm)")
        ax.set_ylabel("Rq IQR (nm)")
        ax.set_title("AFM target variability")
        _save(fig, fig_dir / "afm_target_variability.png", pdf=False)
    if not quality.empty:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].hist(quality["mean_intensity"], bins=12, color="#4c6f9f")
        axes[0].set_title("mean intensity")
        axes[1].hist(quality["frame_to_frame_absdiff_mean"], bins=12, color="#9f6f4c")
        axes[1].set_title("frame absdiff")
        axes[2].hist(quality["roi_area_fraction"], bins=12, color="#5a875a")
        axes[2].set_title("ROI area fraction")
        for ax in axes:
            ax.set_ylabel("count")
        fig.tight_layout()
        _save(fig, fig_dir / "rheed_quality_overview.png", pdf=False)


def baseline_figures(pred_df: pd.DataFrame, metrics_df: pd.DataFrame, neighbor_df: pd.DataFrame, manifest: pd.DataFrame, report_root: Path) -> None:
    fig_dir = report_root / "figures"
    if pred_df.empty:
        return
    best_model = metrics_df.sort_values("MAE_nm").iloc[0]["model_name"]
    best = pred_df[pred_df["model_name"] == best_model]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax in axes:
        ax.scatter(best["true_rq_nm"], best["pred_rq_nm"], s=35, color="#41646f")
        for _, row in best.iterrows():
            ax.text(row["true_rq_nm"], row["pred_rq_nm"], str(row["sample_id"]), fontsize=6)
        lo = min(best["true_rq_nm"].min(), best["pred_rq_nm"].min())
        hi = max(best["true_rq_nm"].max(), best["pred_rq_nm"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("true Rq (nm)")
        ax.set_ylabel("predicted Rq (nm)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title(best_model)
    axes[1].set_title("log scale")
    _save(fig, fig_dir / "baseline_true_vs_predicted_rq.png")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ordered = metrics_df.sort_values("MAE_nm")
    ax.barh(ordered["model_name"], ordered["MAE_nm"], color="#6f7f4c")
    ax.set_xlabel("OOF MAE (nm)")
    ax.set_title("Baseline model comparison")
    _save(fig, fig_dir / "baseline_model_comparison.png")
    sf = metrics_df[metrics_df["model_name"].str.startswith("B1")]
    tf = metrics_df[metrics_df["model_name"].str.startswith("B2")]
    if not sf.empty and not tf.empty:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(["single keyframe", "16-frame temporal"], [sf["MAE_nm"].min(), tf["MAE_nm"].min()], color=["#4f6b8a", "#8a5f4f"])
        ax.set_ylabel("best OOF MAE (nm)")
        ax.set_title("Single frame vs 16-frame")
        _save(fig, fig_dir / "single_frame_vs_16_frame.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    rank_true = best["true_rq_nm"].rank().to_numpy()
    rank_pred = best["pred_rq_nm"].rank().to_numpy()
    ax.scatter(rank_true, rank_pred, color="#7a4c7a")
    for _, row in best.iterrows():
        ax.text(row["true_rq_nm"], row["pred_rq_nm"], "", fontsize=1)
    ax.plot([1, len(best)], [1, len(best)], color="black", linewidth=1)
    ax.set_xlabel("true Rq rank")
    ax.set_ylabel("predicted Rq rank")
    ax.set_title("Rq rank comparison")
    _save(fig, fig_dir / "rq_rank_comparison.png")
    if not neighbor_df.empty:
        counts = neighbor_df.groupby("heldout_sample_id").size()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(counts.index.astype(str), counts.values, color="#4c7a70")
        ax.set_ylabel("audited neighbors")
        ax.set_xlabel("held-out sample")
        ax.tick_params(axis="x", rotation=90)
        ax.set_title("KNN neighbor audit")
        _save(fig, fig_dir / "knn_neighbor_examples.png")
