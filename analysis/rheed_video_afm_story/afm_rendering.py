from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .rq_disentanglement import ra_np, rq_np


def load_physical_map(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def display_limits(arrays: list[np.ndarray], mode: str) -> list[tuple[float, float]]:
    centered = [a - float(np.mean(a)) for a in arrays]
    if mode == "row_shared":
        vals = np.concatenate([a.ravel() for a in centered])
        lo, hi = np.percentile(vals, [1, 99])
        lim = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return [(-lim, lim)] * len(arrays)
    if mode == "per_image":
        out = []
        for a in centered:
            lo, hi = np.percentile(a, [1, 99])
            lim = max(abs(float(lo)), abs(float(hi)), 1e-6)
            out.append((-lim, lim))
        return out
    raise ValueError(mode)


def scale_bar_pixels(scan_size_nm: float, image_pixels: int, bar_nm: float = 125.0) -> int:
    return int(round(image_pixels * bar_nm / scan_size_nm))


def render_afm(ax, arr: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "viridis", scan_size_nm: float = 1000.0, bar_nm: float = 125.0):
    z = arr.astype(float) - float(np.mean(arr))
    im = ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    bar_px = scale_bar_pixels(scan_size_nm, z.shape[1], bar_nm)
    x0 = int(z.shape[1] * 0.08)
    y0 = int(z.shape[0] * 0.92)
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=2)
    ax.text(x0, y0 - 7, f"{int(bar_nm)} nm", color="white", fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Height relative to mean (nm)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    return {"colorbar": True, "scale_bar_nm": float(bar_nm), "scale_bar_pixels": int(bar_px), "vmin": float(vmin), "vmax": float(vmax), "rq_nm": rq_np(arr), "ra_nm": ra_np(arr)}


def render_rheed(ax, frame: np.ndarray, title: str = "RHEED keyframe"):
    ax.imshow(frame, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
