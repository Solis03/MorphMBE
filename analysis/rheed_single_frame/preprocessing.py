"""Aspect-preserving preprocessing for manually selected RHEED screenshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from PIL import Image

from analysis.rheed_roughness.run import display_path, safe_float
from analysis.rheed_single_frame.data import ExperimentPaths, write_csv_rows
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples
from rheed2morph.rheed.frame_quality import extract_frame_quality_features, frame_to_gray_float32
from rheed2morph.rheed.shape_preprocessing import robust_rescale


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREPROCESSING_AUDIT_FIELDS = [
    "sample_id",
    "manual_rheed_path",
    "original_height",
    "original_width",
    "roi_rule",
    "roi_x0",
    "roi_y0",
    "roi_x1",
    "roi_y1",
    "gray_rule",
    "output_height",
    "output_width",
    "valid_roi_fraction",
    "padding_top",
    "padding_bottom",
    "padding_left",
    "padding_right",
    "minimal_gray_path",
    "normalized_image_path",
    "padding_mask_path",
    "contact_sheet_path",
    "mean_intensity",
    "median_intensity",
    "intensity_std",
    "dynamic_range",
    "saturation_fraction",
    "underexposure_fraction",
    "background_gradient",
    "sharpness",
]


@dataclass(frozen=True)
class PreprocessedImage:
    sample_id: str
    manual_rheed_path: Path
    original_rgb: np.ndarray
    cropped_gray: np.ndarray
    gray_padded: np.ndarray
    normalized: np.ndarray
    valid_mask: np.ndarray
    audit_row: dict[str, Any]


def load_manual_image(path: Path) -> np.ndarray:
    image = Image.open(path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    raise ValueError(f"Unsupported manual RHEED image shape for {path}: {arr.shape}")


def resize_with_padding(gray: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Resize without stretching and return padded image, mask, and padding."""
    values = np.asarray(gray, dtype=np.float32)
    height, width = values.shape
    if height <= 0 or width <= 0:
        raise ValueError("Cannot pad an empty image.")
    scale = min(float(size) / float(height), float(size) / float(width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    pil = Image.fromarray(np.asarray(np.clip(values * 255.0, 0, 255), dtype=np.uint8), mode="L")
    resized = np.asarray(pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    out = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    mask[top : top + new_h, left : left + new_w] = True
    bottom = size - top - new_h
    right = size - left - new_w
    return out, mask, (top, bottom, left, right)


def _save_gray(path: Path, image: np.ndarray, *, mask: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is not None:
        values = np.where(mask, image, 0.0)
    else:
        values = image
    arr = np.asarray(np.clip(values * 255.0, 0, 255), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask, dtype=np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def _contact_sheet(path: Path, original: np.ndarray, cropped_gray: np.ndarray, gray_padded: np.ndarray, normalized: np.ndarray, mask: np.ndarray, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2), dpi=180)
    axes[0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("original", fontsize=7)
    axes[1].imshow(cropped_gray, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("cropped ROI", fontsize=7)
    axes[2].imshow(gray_padded, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("grayscale", fontsize=7)
    axes[3].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("normalized", fontsize=7)
    axes[4].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("padding mask", fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Sample {sample_id}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def preprocess_one(sample_id: str, image_path: Path, paths: ExperimentPaths, image_size: int) -> PreprocessedImage:
    original = load_manual_image(image_path)
    gray = frame_to_gray_float32(original)
    roi = gray
    padded, mask, padding = resize_with_padding(roi, image_size)
    valid_values = padded[mask]
    if valid_values.size:
        lo, hi = np.percentile(valid_values, [1, 99])
        norm = np.zeros_like(padded)
        norm[mask] = np.clip((valid_values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)
    else:
        norm = robust_rescale(padded)
    out_dir = paths.outputs_dir / "preprocessed"
    report_dir = paths.reports_dir / "preprocessing_audit"
    minimal_path = out_dir / f"{sample_id}_gray_padded.png"
    norm_path = out_dir / f"{sample_id}_normalized.png"
    mask_path = out_dir / f"{sample_id}_padding_mask.png"
    contact_path = report_dir / f"{sample_id}_preprocessing_contact.png"
    _save_gray(minimal_path, padded)
    _save_gray(norm_path, norm, mask=mask)
    _save_mask(mask_path, mask)
    _contact_sheet(contact_path, original, roi, padded, norm, mask, sample_id)
    quality = extract_frame_quality_features(padded)
    background_gradient = abs(safe_float(quality.get("left_edge_mean")) - safe_float(quality.get("right_edge_mean"))) + abs(
        safe_float(quality.get("top_edge_mean")) - safe_float(quality.get("bottom_edge_mean"))
    )
    top, bottom, left, right = padding
    audit_row = {
        "sample_id": sample_id,
        "manual_rheed_path": display_path(image_path, paths.repo_root),
        "original_height": int(gray.shape[0]),
        "original_width": int(gray.shape[1]),
        "roi_rule": "no_crop_original_manual_selection_aspect_preserving_padding",
        "roi_x0": 0,
        "roi_y0": 0,
        "roi_x1": int(gray.shape[1]),
        "roi_y1": int(gray.shape[0]),
        "gray_rule": "channel average via frame_to_gray_float32",
        "output_height": image_size,
        "output_width": image_size,
        "valid_roi_fraction": float(mask.mean()),
        "padding_top": top,
        "padding_bottom": bottom,
        "padding_left": left,
        "padding_right": right,
        "minimal_gray_path": display_path(minimal_path, paths.repo_root),
        "normalized_image_path": display_path(norm_path, paths.repo_root),
        "padding_mask_path": display_path(mask_path, paths.repo_root),
        "contact_sheet_path": display_path(contact_path, paths.repo_root),
        "mean_intensity": safe_float(quality.get("mean_intensity")),
        "median_intensity": safe_float(quality.get("p50")),
        "intensity_std": safe_float(quality.get("std_intensity")),
        "dynamic_range": safe_float(quality.get("dynamic_range_p99_p01")),
        "saturation_fraction": safe_float(quality.get("saturated_pixel_fraction")),
        "underexposure_fraction": safe_float(quality.get("dark_pixel_fraction")),
        "background_gradient": safe_float(background_gradient),
        "sharpness": safe_float(quality.get("laplacian_variance")),
    }
    return PreprocessedImage(
        sample_id=sample_id,
        manual_rheed_path=image_path,
        original_rgb=original,
        cropped_gray=roi,
        gray_padded=padded,
        normalized=norm,
        valid_mask=mask,
        audit_row=audit_row,
    )


def render_preprocessing_contact_sheet(images: Sequence[PreprocessedImage], paths: ExperimentPaths) -> None:
    if not images:
        return
    cols = 5
    rows = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, max(2.2, rows * 1.85)), dpi=180, squeeze=False)
    for r, item in enumerate(images):
        panels = [item.original_rgb, item.cropped_gray, item.gray_padded, item.normalized, item.valid_mask.astype(float)]
        titles = ["original", "cropped ROI", "grayscale", "normalized", "padding mask"]
        for c, (panel, title) in enumerate(zip(panels, titles)):
            axes[r, c].imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0 if panel.ndim == 2 else None, vmax=1 if panel.ndim == 2 else None)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=7)
            if c == 0:
                axes[r, c].set_ylabel(item.sample_id, fontsize=7)
    fig.tight_layout()
    png = paths.figures_dir / "preprocessing_audit_contact_sheet.png"
    pdf = paths.figures_dir / "preprocessing_audit_contact_sheet.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def preprocess_pairs(
    pairs: Sequence[Any],
    paths: ExperimentPaths,
    config: dict[str, Any],
    removelist: RemovelistAudit,
) -> list[PreprocessedImage]:
    sample_ids = [pair.sample_id for pair in pairs]
    assert_no_removed_samples(sample_ids, removelist.sample_ids, context="RHEED preprocessing")
    images = [
        preprocess_one(pair.sample_id, pair.manual_rheed_path, paths, int(config.get("rheed", {}).get("image_size", 256)))
        for pair in pairs
    ]
    write_csv_rows(paths.outputs_dir / "image_preprocessing_audit.csv", [item.audit_row for item in images], PREPROCESSING_AUDIT_FIELDS)
    render_preprocessing_contact_sheet(images, paths)
    return images

