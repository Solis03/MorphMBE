"""Diagnostic figures for Stage 1 synthetic peak-saddle validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rheed_peak_saddle.pair_features import PairMasks
from analysis.rheed_peak_saddle.synthetic import SyntheticRheed


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bridge_strength_sweep(pair_rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    splits = sorted({str(row.get("split", "")) for row in pair_rows if str(row.get("split", ""))})
    for split, color in zip(splits, colors * 4):
        rows = []
        for row in pair_rows:
            if row.get("split") != split or not int(row.get("valid", 0)):
                continue
            try:
                float(row["true_bridge_strength"])
                float(row["estimated_adhesion"])
            except (TypeError, ValueError):
                continue
            rows.append(row)
        if rows:
            ax.scatter(
                [float(row["true_bridge_strength"]) for row in rows],
                [float(row["estimated_adhesion"]) for row in rows],
                s=12,
                alpha=0.45,
                label=split,
                color=color,
            )
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="ideal")
    ax.set_xlabel("True bridge strength")
    ax.set_ylabel("Estimated peak-saddle adhesion")
    ax.set_title("Continuous grayscale bridge-strength sweep")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_nuisance_invariance(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    labels = []
    values = []
    for split in sorted({str(row.get("split", "")) for row in rows if str(row.get("split", ""))}):
        split_rows = [row for row in rows if row.get("split") == split]
        if split_rows:
            labels.append(split)
            values.append([float(row["abs_delta"]) for row in split_rows])
    ax.boxplot(values, labels=labels, showfliers=True)
    ax.axhline(0.05, color="crimson", linestyle="--", linewidth=1, label="threshold")
    ax.set_ylabel("Absolute adhesion change")
    ax.set_title("Exposure, offset, and display-gamma nuisance invariance")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_example_grid(
    examples: Sequence[SyntheticRheed],
    diagnostics: dict[str, dict[str, Any]],
    path: Path,
    *,
    title: str,
    max_examples: int = 6,
) -> None:
    chosen = list(examples[:max_examples])
    cols = min(3, max(1, len(chosen)))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, example in zip(axes.ravel(), chosen):
        ax.imshow(example.display_image, cmap="magma", origin="upper")
        diag = diagnostics.get(example.image_id, {})
        for spot in diag.get("spots", []):
            ax.plot(float(spot["center_x"]), float(spot["center_y"]), "o", ms=3.5, mfc="none", mec="cyan", mew=0.8)
            ax.text(float(spot["center_x"]) + 2, float(spot["center_y"]) - 2, str(spot.get("row_label", "")), color="white", fontsize=6)
        for pair in diag.get("pairs", [])[:8]:
            x0, y0 = pair["left_center"]
            x1, y1 = pair["right_center"]
            ax.plot([x0, x1], [y0, y1], color="lime", linewidth=0.9, alpha=0.8)
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            ax.text(mx, my, f"{pair.get('estimated_adhesion', np.nan):.2f}", color="white", fontsize=6)
        first_pair = next(iter(example.pairs), None)
        strength = first_pair.true_bridge_strength if first_pair is not None else float("nan")
        nuisance = example.nuisance
        ax.set_title(
            f"{example.image_id}\ntrue={strength:.2f}; rot={float(nuisance.get('rotation_degrees', 0.0)):.1f}; "
            f"halo={float(nuisance.get('halo_strength', 0.0)):.2f}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    _save(fig, path)


def plot_merge_tree_example(example: SyntheticRheed, diag: dict[str, Any], masks: PairMasks | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    axes[0].imshow(example.display_image, cmap="magma", origin="upper")
    axes[0].set_title("Image and detected pair")
    for pair in diag.get("pairs", [])[:1]:
        x0, y0 = pair["left_center"]
        x1, y1 = pair["right_center"]
        axes[0].plot([x0, x1], [y0, y1], color="lime", linewidth=1.2)
        axes[0].text((x0 + x1) / 2.0, (y0 + y1) / 2.0, f"adh={pair.get('estimated_adhesion', np.nan):.2f}", color="white", fontsize=8)
    axes[1].imshow(example.bridge_map, cmap="viridis", origin="upper")
    axes[1].set_title("True bridge map")
    if masks is not None:
        overlay = np.zeros((*example.image.shape, 4), dtype=float)
        overlay[masks.corridor_mask] = (0.1, 1.0, 0.1, 0.28)
        overlay[masks.seed_i_mask | masks.seed_j_mask] = (0.0, 0.8, 1.0, 0.45)
        overlay[masks.background_mask] = (1.0, 1.0, 0.0, 0.22)
        axes[2].imshow(example.display_image, cmap="gray", origin="upper")
        axes[2].imshow(overlay, origin="upper")
    else:
        axes[2].imshow(example.display_image, cmap="gray", origin="upper")
    axes[2].set_title("Corridor, seeds, background")
    for ax in axes:
        ax.axis("off")
    _save(fig, path)


def plot_old_vs_new(pair_rows: Sequence[dict[str, Any]], path: Path) -> None:
    rows = [row for row in pair_rows if row.get("split") == "holdout" and int(row.get("valid", 0))]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, key, title in (
        (axes[0], "estimated_adhesion", "Peak-saddle adhesion"),
        (axes[1], "old_binary_connectivity", "Historical binary/closing baseline"),
    ):
        ax.scatter([float(row["true_bridge_strength"]) for row in rows], [float(row[key]) for row in rows], s=13, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel("True bridge strength")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated connectivity")
    _save(fig, path)


def plot_failure_cases(failure_rows: Sequence[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axis("off")
    if not failure_rows:
        ax.text(0.5, 0.5, "No mandatory Stage 1 failure cases recorded.", ha="center", va="center", fontsize=12)
    else:
        lines = ["Dominant failure cases:"]
        for row in failure_rows[:12]:
            lines.append(f"{row.get('split')} {row.get('criterion')}: {row.get('detail')}")
        ax.text(0.02, 0.95, "\n".join(lines), ha="left", va="top", fontsize=9)
    _save(fig, path)
