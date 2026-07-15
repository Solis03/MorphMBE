from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .common import repo_path
from .embedding_regression import load_embedding


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def simple_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, limit: int = 20) -> None:
    g = df.sort_values(y).head(limit)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(g[x].astype(str), g[y], color="#5b7480")
    ax.set_xlabel(y)
    ax.set_title(title)
    save(fig, path)


def phase2_figures(
    manifest: pd.DataFrame,
    timing: pd.DataFrame,
    variant_manifest: pd.DataFrame,
    embedding_registry: pd.DataFrame,
    reg_pred: pd.DataFrame,
    reg_metrics: pd.DataFrame,
    rank_pred: pd.DataFrame,
    rank_metrics: pd.DataFrame,
    extreme_pred: pd.DataFrame,
    extreme_metrics: pd.DataFrame,
    metadata_metrics: pd.DataFrame,
    confidence: pd.DataFrame,
    intervals: pd.DataFrame,
    coverage: pd.DataFrame,
    retrieval: pd.DataFrame,
    report_root: Path,
) -> None:
    fig_root = report_root / "figures"
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(timing["sample_id"], timing["original_clip_duration_seconds"])
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("duration (s)")
    ax.set_title("16-frame duration by sample")
    save(fig, fig_root / "clip_duration_by_sample.png")
    avail = variant_manifest.groupby("clip_variant")["available"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(avail["clip_variant"], avail["available"], color="#6b805b")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("available primary samples")
    ax.set_title("Frame count and timing ablation availability")
    save(fig, fig_root / "frame_count_and_timing_ablation.png")
    simple_bar(reg_metrics.assign(label=reg_metrics["encoder"] + "/" + reg_metrics["clip_variant"] + "/" + reg_metrics["head"]), "label", "MAE_nm", "Pretrained encoder model comparison", fig_root / "pretrained_encoder_model_comparison.png")
    best = reg_metrics.iloc[0]
    best_pred = reg_pred[(reg_pred["embedding_id"] == best["embedding_id"]) & (reg_pred["head"] == best["head"])]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(best_pred["true_rq_nm"], best_pred["pred_rq_nm"], color="#496f84")
    for _, r in best_pred.iterrows():
        ax.text(r["true_rq_nm"], r["pred_rq_nm"], str(r["sample_id"]), fontsize=6)
    lo = min(best_pred["true_rq_nm"].min(), best_pred["pred_rq_nm"].min())
    hi = max(best_pred["true_rq_nm"].max(), best_pred["pred_rq_nm"].max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("true Rq (nm)")
    ax.set_ylabel("predicted Rq (nm)")
    ax.set_title("OOF true vs predicted Rq")
    save(fig, fig_root / "true_vs_predicted_rq_all_samples.png")
    best_rank = rank_metrics.iloc[0]
    rp = rank_pred[rank_pred["embedding_id"] == best_rank["embedding_id"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pd.Series(rp["true_rq_nm"]).rank(), rp["predicted_rank_percentile"], color="#7a5b80")
    ax.set_xlabel("true Rq rank")
    ax.set_ylabel("predicted rank percentile")
    ax.set_title("Rq rank true vs predicted")
    save(fig, fig_root / "rq_rank_true_vs_predicted.png")
    simple_bar(rank_metrics, "embedding_id", "pairwise_concordance", "Pairwise concordance comparison", fig_root / "pairwise_concordance_comparison.png")
    if not extreme_metrics.empty:
        three = extreme_pred[extreme_pred["task"] == "three_class"]
        if not three.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(three["true_label"], three["pred_label"], labels=["low", "middle", "high"], ax=ax, colorbar=False)
            ax.set_title("Low/middle/high confusion")
            save(fig, fig_root / "low_middle_high_confusion.png")
        binary = extreme_pred[(extreme_pred["task"] == "extreme_binary") & (extreme_pred["true_label"].isin(["low", "high"]))]
        if not binary.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            RocCurveDisplay.from_predictions((binary["true_label"] == "high").astype(int), binary["prob_high"], ax=ax)
            ax.set_title("Extreme low vs high ROC")
            save(fig, fig_root / "extreme_low_vs_high_roc.png")
    if not embedding_registry.empty:
        emb = embedding_registry.iloc[0]
        ids, X = load_embedding(emb["path"])
        coords = PCA(n_components=2, random_state=17).fit_transform(X)
        rq_map = manifest.set_index("sample_id")["primary_rq_nm_median"].to_dict()
        stage_map = manifest.set_index("sample_id")["video_stage"].to_dict()
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=[rq_map[i] for i in ids], cmap="viridis")
        fig.colorbar(sc, ax=ax, label="Rq (nm)")
        ax.set_title("Embedding PCA colored by Rq")
        save(fig, fig_root / "embedding_umap_colored_by_rq.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        stages = sorted(set(stage_map[i] for i in ids))
        for st in stages:
            mask = [stage_map[i] == st for i in ids]
            ax.scatter(coords[mask, 0], coords[mask, 1], label=st)
        ax.legend(fontsize=7)
        ax.set_title("Embedding PCA marked by stage")
        save(fig, fig_root / "embedding_umap_marked_by_stage.png")
    raw_robust = reg_metrics.groupby("preprocessing")["MAE_nm"].min().reset_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(raw_robust["preprocessing"], raw_robust["MAE_nm"], color="#806f5b")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("best MAE (nm)")
    save(fig, fig_root / "raw_vs_robust_contrast_ablation.png")
    frame = reg_metrics.groupby("clip_variant")["MAE_nm"].min().reset_index()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(frame["clip_variant"], frame["MAE_nm"], color="#5b8068")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("best MAE (nm)")
    save(fig, fig_root / "single_frame_vs_4_vs_8_vs_16.png")
    if not metadata_metrics.empty:
        simple_bar(metadata_metrics.rename(columns={"control_model": "model"}), "model", "MAE_nm", "Metadata vs RHEED vs fusion", fig_root / "metadata_vs_rheed_vs_fusion.png")
    fig, ax = plt.subplots(figsize=(5, 3))
        # coverage is already support sorted.
    ax.plot(coverage["coverage_fraction"], coverage["MAE_nm"], marker="o")
    ax.invert_xaxis()
    ax.set_xlabel("support coverage")
    ax.set_ylabel("MAE (nm)")
    ax.set_title("Confidence coverage performance")
    save(fig, fig_root / "confidence_coverage_performance.png")
    fig, ax = plt.subplots(figsize=(5, 3))
    widths = intervals["pi90_high_nm"] - intervals["pi90_low_nm"]
    ax.scatter(widths, np.abs(intervals["true_rq_nm"] - intervals["pred_rq_nm"]))
    ax.set_xlabel("90% interval width (nm)")
    ax.set_ylabel("absolute error (nm)")
    ax.set_title("Interval coverage")
    save(fig, fig_root / "interval_coverage.png")
    counts = confidence["support_level"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.index, counts.values, color="#80605b")
    ax.set_title("Support levels")
    save(fig, fig_root / "retrieval_neighbor_examples.png")
    for name, subset in [
        ("all_sample_prediction_grid", best_pred),
        ("high_support_prediction_grid", best_pred[best_pred["sample_id"].isin(confidence[confidence["support_level"] == "high"]["sample_id"])]),
        ("failure_and_low_support_grid", best_pred[best_pred["sample_id"].isin(confidence[confidence["support_level"] == "low"]["sample_id"])]),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(subset["sample_id"].astype(str), subset["absolute_error_nm"], color="#746a80")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("|error| (nm)")
        ax.set_title(name)
        save(fig, fig_root / f"{name}.png")
