#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalize_image(arr: np.ndarray) -> np.ndarray:
    data = np.squeeze(np.asarray(arr, dtype=float))
    if data.ndim == 3:
        data = data[0] if data.shape[0] <= 32 else data[..., 0]
    lo, hi = np.nanpercentile(data, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return (np.clip((data - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_keyframe(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    key = next(
        (k for k in ["image", "frame", "keyframe", "clip", "frames", "arr_0"] if k in z.files),
        z.files[0],
    )
    arr = np.squeeze(z[key])
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] <= 32:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.mean(axis=-1)
    return arr


def add_heightbar(ax, arr: np.ndarray) -> None:
    lo, hi = np.nanpercentile(arr, [1, 99])
    cax = ax.inset_axes([1.02, 0.08, 0.045, 0.84])
    cax.imshow(np.linspace(hi, lo, 128)[:, None], cmap="viridis", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 127])
    cax.set_yticklabels([f"{hi:.1f}", f"{lo:.1f}"], fontsize=5)
    cax.set_title("nm", fontsize=5, pad=1)


def save_all(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def draw_scatter() -> None:
    pred = pd.read_csv(ROOT / "results/strict_oof/predictions.csv").sort_values("sample_id")
    metrics = json.loads((ROOT / "results/strict_oof/metrics.json").read_text())
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(
        pred.true_target_nm,
        pred.predicted_target_nm,
        s=32,
        c="#235c6b",
        edgecolors="white",
        linewidth=0.6,
    )
    lo = min(pred.true_target_nm.min(), pred.predicted_target_nm.min()) - 0.4
    hi = max(pred.true_target_nm.max(), pred.predicted_target_nm.max()) + 0.4
    ax.plot([lo, hi], [lo, hi], c="0.25", lw=1, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Rq (nm)")
    ax.set_ylabel("Predicted Rq q50 (nm)")
    ax.set_title("Strict OOF single-frame RHEED-to-Rq")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.text(
        0.02,
        0.98,
        f"N=23\nMAE={metrics['MAE']:.2f} nm\nSpearman={metrics['Spearman']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=.25", fc="white", ec=".8", alpha=.9),
    )
    save_all(fig, ROOT / "figures/main/Figure2_strict_oof_rq_scatter")
    plt.close(fig)


def draw_atlas() -> None:
    ret = pd.read_csv(ROOT / "results/strict_oof/retrieval_results.csv").sort_values("sample_id")
    groups_per_row = 3
    nrows = int(np.ceil(len(ret) / groups_per_row))
    fig = plt.figure(figsize=(11.2, nrows * 1.55))
    grid = fig.add_gridspec(nrows, groups_per_row * 3, wspace=0.10, hspace=0.42)
    for idx, row in ret.iterrows():
        sid = int(row.sample_id)
        rr = idx // groups_per_row
        cc = (idx % groups_per_row) * 3
        rheed = load_keyframe(ROOT / f"data_snapshot/selected_rheed_keyframes/{sid}_keyframe_1_raw_luminance.npz")
        gt = np.load(ROOT / f"data_snapshot/representative_afm/{sid}_ground_truth_second_order.npy")
        got = np.load(ROOT / f"results/strict_oof/retrieved_maps_q50/{sid}_A3_q50_retrieved.npy")
        axes = [fig.add_subplot(grid[rr, cc + i]) for i in range(3)]
        axes[0].imshow(normalize_image(rheed), cmap="gray")
        axes[0].set_title(f"{sid} RHEED", fontsize=7, pad=2)
        axes[1].imshow(gt, cmap="viridis")
        axes[1].set_title(f"GT AFM\nRq {row.true_rq_nm:.2f}", fontsize=7, pad=2)
        add_heightbar(axes[1], gt)
        axes[2].imshow(got, cmap="viridis")
        axes[2].set_title(f"A3 q50\nRq {row.output_q50_measured_rq_nm:.2f}", fontsize=7, pad=2)
        add_heightbar(axes[2], got)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Strict OOF A3 Retrieval: RHEED, Ground Truth AFM, Retrieved AFM q50",
        fontsize=11,
        y=0.995,
    )
    save_all(fig, ROOT / "figures/main/Figure3_strict_a3_q50_atlas_all23")
    plt.close(fig)


def draw_q_bands() -> None:
    err = pd.read_csv(ROOT / "results/strict_oof/per_sample_errors.csv")
    ret = pd.read_csv(ROOT / "results/strict_oof/retrieval_results.csv")
    sid = int(err.sort_values("absolute_error_nm", ascending=False).iloc[0].sample_id)
    row = ret[ret.sample_id == sid].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.2))
    for ax, label, rq in zip(
        axes,
        ["q10", "q50", "q90"],
        [row.predicted_rq_q10, row.predicted_rq_q50, row.predicted_rq_q90],
    ):
        subdir = "retrieved_maps_q50" if label == "q50" else "retrieved_maps_q10_q90"
        path = ROOT / f"results/strict_oof/{subdir}/{sid}_A3_{label}_retrieved.npy"
        arr = np.load(path)
        ax.imshow(arr, cmap="viridis")
        ax.set_title(f"{sid} {label}\nRq {rq:.2f} nm", fontsize=8)
        add_heightbar(ax, arr)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Same retrieved morphology, amplitude-only q10/q50/q90 rescaling", fontsize=10)
    save_all(fig, ROOT / "figures/main/Figure4_q10_q50_q90_amplitude_example")
    plt.close(fig)


def main() -> None:
    draw_scatter()
    draw_atlas()
    draw_q_bands()
    print("Regenerated main figures from frozen data.")


if __name__ == "__main__":
    main()
