#!/usr/bin/env python3
"""Build reproducible, publication-resolution visual assets for the drawing prompt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[1]
EXPERIMENT = PACKAGE.parent
REPO = next(
    parent
    for parent in EXPERIMENT.parents
    if (parent / "publication_freeze/rheed_afm_single_frame_v1_2026-07-18").is_dir()
)


def rq_nm(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    values = values[np.isfinite(values)]
    values -= float(np.mean(values))
    return float(np.sqrt(np.mean(values**2)))


def add_height_bar(ax: plt.Axes, array: np.ndarray) -> None:
    low, high = np.nanpercentile(array, [1, 99])
    bar = ax.inset_axes([1.025, 0.08, 0.045, 0.84])
    bar.imshow(np.linspace(high, low, 256)[:, None], cmap="viridis", aspect="auto")
    bar.set_xticks([])
    bar.set_yticks([0, 255])
    bar.set_yticklabels([f"{high:.1f}", f"{low:.1f}"], fontsize=8)
    bar.set_title("nm", fontsize=8, pad=2)


def save_afm(array: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    low, high = np.nanpercentile(array, [1, 99])
    ax.imshow(array, cmap="viridis", vmin=low, vmax=high)
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    add_height_bar(ax, array)
    fig.savefig(path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_model_ready_assets() -> None:
    output = PACKAGE / "01_rheed_inputs/model_ready"
    for npz_path in sorted(output.glob("N*_keyframe_1_raw_luminance.npz")):
        archive = np.load(npz_path, allow_pickle=False)
        sample_id = str(archive["sample_id"])
        frame = np.asarray(archive["frames_uint8"][0], dtype=np.uint8)
        Image.fromarray(frame, mode="L").save(
            output / f"{sample_id}_model_ready_224x224_luminance.png"
        )
        if sample_id != "N6390":
            continue
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for coordinate in range(0, 225, 14):
            ax.axvline(coordinate - 0.5, color="#20C4D9", linewidth=0.55, alpha=0.8)
            ax.axhline(coordinate - 0.5, color="#20C4D9", linewidth=0.55, alpha=0.8)
        ax.set_title("N6390 model-ready RHEED\n16 × 16 non-overlapping 14 × 14 patches")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(
            output / "N6390_model_ready_patch14_grid.png",
            dpi=500,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def build_candidate_bank_assets() -> None:
    bank = pd.read_csv(
        EXPERIMENT / "models/visual_model/expanded_afm_bank_manifest.csv",
        dtype={"sample_id": str},
    )
    retrieval = pd.read_csv(
        EXPERIMENT / "predictions/retrained_23/retrieval/retrieval_results.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    result = retrieval.loc["N6390"]
    ranked_ids = json.loads(result["top5_source_afm_file_ids"])
    bank_index = bank.set_index("afm_file_id")
    output = PACKAGE / "04_afm_candidate_bank"
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    for rank, file_id in enumerate(ranked_ids, start=1):
        row = bank_index.loc[file_id]
        array = np.load(REPO / str(row["second_order_afm_path"]), allow_pickle=False)
        sample_id = str(row["sample_id"])
        selected = rank == 1
        status = "SELECTED" if selected else "candidate"
        title = (
            f"Rank {rank} · {status}\n"
            f"{sample_id} / {file_id}\n"
            f"Rq = {rq_nm(array):.2f} nm · 1 × 1 µm"
        )
        save_afm(
            array,
            output
            / f"rank{rank}_{sample_id}_{file_id}_{'selected' if selected else 'candidate'}_heightbar_Rq.png",
            title,
        )
        arrays.append(array)
        labels.append(title)

    fig, axes = plt.subplots(1, 5, figsize=(18.5, 4.0))
    for ax, array, label in zip(axes, arrays, labels, strict=True):
        low, high = np.nanpercentile(array, [1, 99])
        ax.imshow(array, cmap="viridis", vmin=low, vmax=high)
        ax.set_title(label, fontsize=8, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        add_height_bar(ax, array)
    fig.suptitle(
        "N6390 A3 top-five source candidates (rank 1 is the selected morphology)",
        fontsize=13,
        y=1.02,
    )
    fig.savefig(
        output / "N6390_A3_top5_candidate_bank_montage.png",
        dpi=500,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    build_model_ready_assets()
    build_candidate_bank_assets()


if __name__ == "__main__":
    main()
