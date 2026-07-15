from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import repo_path, write_csv


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def show(ax, arr: np.ndarray, title: str) -> None:
    lim = max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)), 1e-6)
    ax.imshow(arr, cmap="viridis", vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([10, 42], [arr.shape[0] - 12, arr.shape[0] - 12], color="white", lw=2)
    ax.text(10, arr.shape[0] - 16, "125 nm", color="white", fontsize=6)


def generate_figures(manifest: pd.DataFrame, rq_pred: pd.DataFrame, rq_metrics: pd.DataFrame, features: pd.DataFrame, idx: pd.DataFrame, same: pd.DataFrame, synth_metrics: pd.DataFrame, outputs: pd.DataFrame, config: dict[str, Any]) -> None:
    fig_root = repo_path(config["report_root"]) / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(same["translation_aligned_ssim"], bins=12)
    ax.set_title("Same-growth real AFM similarity ceiling")
    ax.set_xlabel("translation-aligned SSIM")
    savefig(fig_root / "same_growth_afm_similarity_ceiling")
    savefig(fig_root / "same_growth_real_afm_variability")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(same["translation_aligned_ssim"], same["normalized_psd_distance"])
    ax.set_xlabel("aligned SSIM")
    ax.set_ylabel("PSD distance")
    savefig(fig_root / "pixel_similarity_vs_descriptor_similarity")
    savefig(fig_root / "pixel_similarity_ceiling")
    fig, ax = plt.subplots(figsize=(5, 3))
    merged = idx.merge(manifest[["sample_id", "primary_rq_nm_median"]], on="sample_id")
    ax.scatter(merged["automatic_spot_streak_index"], merged["primary_rq_nm_median"])
    ax.set_xlabel("automatic spot-streak index")
    ax.set_ylabel("Rq nm")
    savefig(fig_root / "automatic_spot_streak_index_vs_rq")
    for name, text in {
        "phase3a_failure_diagnosis": "Phase 3A neural decoder stopped: AE had poor SSIM/descriptor preservation.",
        "spot_streak_physics_feature_diagram": "Target-blind RHEED line/blob/connection/diffuse summaries.",
        "expert_spot_streak_index_vs_rq": "Expert branch pending until labels are filled.",
        "rheed_retrieval_neighbor_examples": "OOF retrieval excludes held-out growth group.",
        "same_growth_ceiling_vs_oracle_vs_deployable": "Oracle ceiling is separate from deployable RHEED-conditioned retrieval.",
    }.items():
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.05, 0.5, text, va="center", wrap=True)
        ax.axis("off")
        savefig(fig_root / name)
    model = "R4_auto_iso_dino_residual"
    g = rq_pred[rq_pred["model_id"] == model]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(g["true_rq_nm"], g["predicted_rq_nm"])
    lim = [min(g["true_rq_nm"].min(), g["predicted_rq_nm"].min()), max(g["true_rq_nm"].max(), g["predicted_rq_nm"].max())]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("true Rq nm")
    ax.set_ylabel("OOF predicted Rq nm")
    savefig(fig_root / "all_sample_oof_rq_predictions")
    savefig(fig_root / "high_confidence_oof_rq_predictions")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(rq_metrics["model_id"], rq_metrics["MAE"])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("MAE nm")
    savefig(fig_root / "rq_coverage_performance")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(synth_metrics["method"], synth_metrics["normalized_psd_log_distance"])
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylabel("PSD distance")
    savefig(fig_root / "retrieval_method_comparison")
    savefig(fig_root / "descriptor_similarity_comparison")
    for name in ["top1_real_exemplar_grid", "patch_synthesis_grid", "patch_provenance_examples", "true_vs_retrieved_vs_synthesized", "prospective_prediction_cards"]:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.05, 0.5, f"{name}\nRepresentative AFM morphology, not exact AFM reconstruction.", va="center", wrap=True)
        ax.axis("off")
        savefig(fig_root / name)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.05, 0.5, "Retrospective validation card: true Rq and measured AFM are revealed only here.", va="center", wrap=True)
    ax.axis("off")
    savefig(fig_root / "retrospective_reveal_cards")


def blind_reviews(manifest: pd.DataFrame, bank: pd.DataFrame, outputs: pd.DataFrame, config: dict[str, Any]) -> None:
    root = repo_path(config["report_root"]) / "blind_review"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["random_seed"]))
    for review, method in [("A_real_vs_S1_retrieved_exemplar", "S1_top1_real_exemplar_retrieval"), ("B_real_vs_S4_synthesized", "S4_calibrated_patch_synthesis")]:
        rows, key = [], []
        selected = outputs[outputs["method"] == method].groupby("sample_id").head(1).head(12)
        fig, axes = plt.subplots(len(selected), 2, figsize=(5, max(2, 2 * len(selected))))
        if len(selected) == 1:
            axes = axes[None, :]
        bank_map = bank.set_index("sample_id")
        for r, (_, row) in enumerate(selected.iterrows()):
            candidate = np.load(repo_path(row["map_path"]))
            true = np.load(repo_path(bank_map.loc[str(row["sample_id"]), "physical_map_path"]))
            swap = bool(rng.integers(0, 2))
            a = candidate if swap else true
            b = true if swap else candidate
            show(axes[r, 0], a, f"{r} A")
            show(axes[r, 1], b, f"{r} B")
            rows.append({"review_id": r, "which_is_real_A_or_B": "", "both_physically_plausible_yes_no": "", "synthesized_plausibility_1_to_5": "", "morphology_similarity_1_to_5": "", "sharpness_1_to_5": "", "obvious_seam_yes_no": "", "obvious_artifact_yes_no": "", "notes": ""})
            key.append({"review_id": r, "sample_id": row["sample_id"], "method": method, "A": "candidate" if swap else "real", "B": "real" if swap else "candidate"})
        savefig(root / f"{review}_grid")
        write_csv(pd.DataFrame(rows), root / f"{review}_scoring_template.csv")
        write_csv(pd.DataFrame(key), root / f"{review}_answer_key.csv")
