#!/usr/bin/env python3
"""Evaluate automatic RHEED ROI and rotation-phase keyframe selection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage, stats
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rheed2morph.rheed.automatic_roi_keyframe import (  # noqa: E402
    KEYFRAME_METHODS,
    ROI_METHODS,
    Rect,
    extract_multi_roi_trajectories,
    iter_png_frames,
    load_frame,
    predict_roi,
    sample_frames,
    select_keyframes,
)


COLORS = {
    "manual": "#35d07f",
    "aperture_inscribed": "#ee7733",
    "activity_safe": "#0077bb",
    "calibrated_safe": "#cc3311",
    "quality_only": "#999999",
    "vertex_clarity": "#33bbee",
    "physics_vertex": "#cc3311",
    "front_visibility": "#aa3377",
    "compact_physics": "#ee3377",
    "compact_visibility": "#228833",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_auto_roi_keyframe_v1.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only: process the first N annotated videos.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse per-sample trajectory JSON files when present.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rect_iou(first: Rect, second: Rect) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x2, second.x2)
    y1 = min(first.y2, second.y2)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    return float(intersection / max(union, 1))


def manual_overlap(auto: Rect, manual: Rect) -> float:
    x0 = max(auto.x, manual.x)
    y0 = max(auto.y, manual.y)
    x1 = min(auto.x2, manual.x2)
    y1 = min(auto.y2, manual.y2)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    return float(intersection / max(manual.width * manual.height, 1))


def rect_center_error(auto: Rect, manual: Rect) -> float:
    dx = (auto.x + 0.5 * auto.width) - (manual.x + 0.5 * manual.width)
    dy = (auto.y + 0.5 * auto.height) - (
        manual.y + 0.5 * manual.height
    )
    return float(
        math.hypot(dx / manual.source_width, dy / manual.source_height)
    )


def crop_for_compare(frame: np.ndarray, rect: Rect) -> np.ndarray:
    ys, xs = rect.as_slices()
    rgb = np.asarray(frame)[ys, xs, :3]
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    image = np.asarray(
        Image.fromarray(gray).resize(
            (128, 192), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    low, high = np.percentile(image, [3.0, 99.7])
    normalized = np.clip(
        (image - low) / max(float(high - low), 1e-5), 0.0, 1.0
    )
    background = ndimage.gaussian_filter(normalized, 12.0)
    response = normalized - background
    scale = float(np.percentile(np.abs(response), 99.0))
    return np.clip(0.5 + response / max(2.0 * scale, 1e-5), 0.0, 1.0)


def pattern_similarity(
    first_frame: np.ndarray,
    second_frame: np.ndarray,
    rect: Rect,
) -> tuple[float, float, float]:
    first = crop_for_compare(first_frame, rect)
    second = crop_for_compare(second_frame, rect)
    first_flat = first.ravel()
    second_flat = second.ravel()
    if np.std(first_flat) < 1e-8 or np.std(second_flat) < 1e-8:
        ncc = 0.0
    else:
        ncc = float(np.corrcoef(first_flat, second_flat)[0, 1])
    ssim = float(
        structural_similarity(first, second, data_range=1.0)
    )
    gradient_first = np.hypot(*np.gradient(first))
    gradient_second = np.hypot(*np.gradient(second))
    if (
        np.std(gradient_first) < 1e-8
        or np.std(gradient_second) < 1e-8
    ):
        gradient_ncc = 0.0
    else:
        gradient_ncc = float(
            np.corrcoef(
                gradient_first.ravel(), gradient_second.ravel()
            )[0, 1]
        )
    return ncc, ssim, gradient_ncc


def draw_rect(
    image: Image.Image,
    rect: Rect,
    *,
    color: str,
    width: int,
    label: str,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (rect.x, rect.y, rect.x2 - 1, rect.y2 - 1),
        outline=color,
        width=width,
    )
    draw.text(
        (rect.x + 3, max(2, rect.y - 22)),
        label,
        fill=color,
        stroke_width=2,
        stroke_fill="black",
    )


def crop_display(frame: np.ndarray, rect: Rect) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    crop = image.crop((rect.x, rect.y, rect.x2, rect.y2))
    crop.thumbnail((360, 420), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(crop).enhance(1.8)


def frame_safe_fraction(analysis: Any, rect: Rect) -> float:
    scale = analysis.scale
    x0 = int(round(rect.x * scale))
    y0 = int(round(rect.y * scale))
    x1 = int(round(rect.x2 * scale))
    y1 = int(round(rect.y2 * scale))
    sub = analysis.safe_mask[
        max(0, y0) : min(analysis.safe_mask.shape[0], y1),
        max(0, x0) : min(analysis.safe_mask.shape[1], x1),
    ]
    return float(sub.mean()) if sub.size else 0.0


def _serialize_trajectory(
    trajectories: dict[str, list[dict[str, float | int]]],
    path: Path,
) -> None:
    payload = {
        name: [
            {key: float(value) if key != "frame_index" else int(value)
             for key, value in row.items()}
            for row in records
        ]
        for name, records in trajectories.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )


def _load_trajectory(
    path: Path,
) -> dict[str, list[dict[str, float | int]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_sample(
    row: pd.Series,
    *,
    config: dict[str, Any],
    output_root: Path,
    reuse: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sample_id = str(row["sample_id"])
    frames_dir = ROOT / str(row["frames_dir"])
    source_video = ROOT / str(row["source_video"])
    manual = Rect(
        x=int(row["roi_x"]),
        y=int(row["roi_y"]),
        width=int(row["roi_width"]),
        height=int(row["roi_height"]),
        source_width=int(row["source_width"]),
        source_height=int(row["source_height"]),
    )
    sampled, frame_count = sample_frames(
        frames_dir, maximum=int(config["roi_sample_count"])
    )
    roi_predictions = {}
    analysis = None
    roi_rows: list[dict[str, Any]] = []
    for method in config["roi_methods"]:
        prediction, analysis = predict_roi(
            sampled,
            method=method,
            aspect_ratio=float(config["roi_aspect_ratio"]),
            calibrated_scale=float(config["roi_calibrated_scale"]),
            analysis=analysis,
        )
        roi_predictions[method] = prediction
        roi_rows.append(
            {
                "sample_id": sample_id,
                "video_id": row["video_id"],
                "method": method,
                "iou_vs_manual": rect_iou(prediction.rect, manual),
                "manual_area_coverage": manual_overlap(
                    prediction.rect, manual
                ),
                "center_error_normalized": rect_center_error(
                    prediction.rect, manual
                ),
                "safe_pixel_fraction": prediction.safe_pixel_fraction,
                "border_contamination_fraction": (
                    1.0 - prediction.safe_pixel_fraction
                ),
                "activity_coverage": prediction.activity_coverage,
                "confidence": prediction.confidence,
                **{
                    f"auto_{key}": value
                    for key, value in asdict(prediction.rect).items()
                    if not key.startswith("source_")
                },
                "manual_x": manual.x,
                "manual_y": manual.y,
                "manual_width": manual.width,
                "manual_height": manual.height,
                "manual_safe_pixel_fraction": frame_safe_fraction(
                    analysis, manual
                ),
            }
        )

    final_roi = roi_predictions[config["final_roi_method"]].rect
    trajectory_path = (
        output_root / "trajectories" / f"{sample_id}.json"
    )
    if reuse and trajectory_path.exists():
        trajectories = _load_trajectory(trajectory_path)
    else:
        trajectories = extract_multi_roi_trajectories(
            iter_png_frames(frames_dir),
            {"automatic": final_roi, "manual": manual},
        )
        _serialize_trajectory(trajectories, trajectory_path)

    predictions, candidates = select_keyframes(trajectories["automatic"])
    compact_trajectory = [
        {
            **item,
            "spot_x": item.get("compact_spot_x", item["spot_x"]),
            "spot_y": item.get("compact_spot_y", item["spot_y"]),
        }
        for item in trajectories["automatic"]
    ]
    _, compact_candidates = select_keyframes(compact_trajectory)
    manual_predictions, manual_candidates = select_keyframes(
        trajectories["manual"]
    )
    manual_index = int(row["keyframe_index"])
    manual_frame = load_frame(frames_dir, manual_index)
    frame_cache: dict[int, np.ndarray] = {manual_index: manual_frame}
    nearest_candidate_error = min(
        abs(int(candidate["frame_index"]) - manual_index)
        for candidate in candidates
    )
    compact_nearest_candidate_error = min(
        abs(int(candidate["frame_index"]) - manual_index)
        for candidate in compact_candidates
    )
    nearest_manual_roi_candidate_error = min(
        abs(int(candidate["frame_index"]) - manual_index)
        for candidate in manual_candidates
    )
    keyframe_rows: list[dict[str, Any]] = []
    for method in config["keyframe_methods"]:
        prediction = predictions[method]
        model_nearest_candidate_error = (
            compact_nearest_candidate_error
            if method.startswith("compact_")
            else nearest_candidate_error
        )
        selected_index = prediction.frame_index
        if selected_index not in frame_cache:
            frame_cache[selected_index] = load_frame(
                frames_dir, selected_index
            )
        ncc, ssim, gradient_ncc = pattern_similarity(
            manual_frame, frame_cache[selected_index], final_roi
        )
        exact_error = abs(selected_index - manual_index)
        period = prediction.estimated_period_frames
        if period and period > 0:
            cycle_offset = int(round((selected_index - manual_index) / period))
            phase_error = abs(
                selected_index - manual_index - cycle_offset * period
            )
            phase_error_fraction = phase_error / period
        else:
            cycle_offset = 0
            phase_error = float(exact_error)
            phase_error_fraction = float("nan")
        manual_trajectory_row = min(
            trajectories["automatic"],
            key=lambda item: abs(int(item["frame_index"]) - manual_index),
        )
        keyframe_rows.append(
            {
                "sample_id": sample_id,
                "video_id": row["video_id"],
                "method": method,
                "manual_frame_index": manual_index,
                "selected_frame_index": selected_index,
                "absolute_frame_error": exact_error,
                "estimated_period_frames": period,
                "cycle_offset": cycle_offset,
                "periodic_phase_error_frames": phase_error,
                "periodic_phase_error_fraction": phase_error_fraction,
                "nearest_vertex_to_manual_error_frames": (
                    model_nearest_candidate_error
                ),
                "manual_roi_nearest_vertex_error_frames": (
                    nearest_manual_roi_candidate_error
                ),
                "pattern_ncc": ncc,
                "pattern_ssim": ssim,
                "gradient_ncc": gradient_ncc,
                "clarity": prediction.clarity,
                "manual_frame_clarity": float(
                    manual_trajectory_row["clarity"]
                ),
                "clarity_ratio_vs_manual": prediction.clarity
                / max(float(manual_trajectory_row["clarity"]), 1e-8),
                "score": prediction.score,
                "confidence": prediction.confidence,
                "candidate_count": prediction.candidate_count,
                "direction_consistent": prediction.direction_consistent,
            }
        )

    selection_payload = {
        "sample_id": sample_id,
        "video_id": str(row["video_id"]),
        "source_video": str(row["source_video"]),
        "source_video_size_bytes": source_video.stat().st_size,
        "source_video_mtime_ns": source_video.stat().st_mtime_ns,
        "frames_dir": str(row["frames_dir"]),
        "frame_count": frame_count,
        "manual": {
            "frame_index": manual_index,
            "roi": asdict(manual),
        },
        "roi_predictions": {
            method: asdict(prediction)
            for method, prediction in roi_predictions.items()
        },
        "keyframe_predictions": {
            method: asdict(prediction)
            for method, prediction in predictions.items()
        },
        "candidate_count": len(candidates),
    }
    selection_path = output_root / "selections" / f"{sample_id}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return roi_rows, keyframe_rows, selection_payload


def summarize_metrics(
    roi: pd.DataFrame, keyframe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_summary = (
        roi.groupby("method", sort=False)
        .agg(
            n=("sample_id", "size"),
            median_iou=("iou_vs_manual", "median"),
            median_manual_coverage=("manual_area_coverage", "median"),
            median_center_error=("center_error_normalized", "median"),
            median_safe_fraction=("safe_pixel_fraction", "median"),
            minimum_safe_fraction=("safe_pixel_fraction", "min"),
            median_activity_coverage=("activity_coverage", "median"),
        )
        .reset_index()
    )
    key_summary = (
        keyframe.groupby("method", sort=False)
        .agg(
            n=("sample_id", "size"),
            median_absolute_frame_error=("absolute_frame_error", "median"),
            median_periodic_phase_error=(
                "periodic_phase_error_frames",
                "median",
            ),
            median_nearest_vertex_error=(
                "nearest_vertex_to_manual_error_frames",
                "median",
            ),
            p90_nearest_vertex_error=(
                "nearest_vertex_to_manual_error_frames",
                lambda values: np.percentile(values, 90),
            ),
            median_pattern_ncc=("pattern_ncc", "median"),
            median_pattern_ssim=("pattern_ssim", "median"),
            median_gradient_ncc=("gradient_ncc", "median"),
            median_clarity_ratio=("clarity_ratio_vs_manual", "median"),
            direction_consistency_rate=(
                "direction_consistent", "mean"
            ),
        )
        .reset_index()
    )
    return roi_summary, key_summary


def save_method_summary_figure(
    roi: pd.DataFrame,
    keyframe: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    roi_order = list(dict.fromkeys(roi["method"]))
    key_order = list(dict.fromkeys(keyframe["method"]))
    roi_metrics = [
        ("safe_pixel_fraction", "Safe ROI fraction", (0.85, 1.005)),
        ("activity_coverage", "Diffraction activity coverage", (0.0, 1.02)),
        ("iou_vs_manual", "IoU with human ROI", (0.0, 1.02)),
    ]
    key_metrics = [
        (
            "nearest_vertex_to_manual_error_frames",
            "Human frame → nearest detected vertex (frames)",
            None,
        ),
        ("pattern_ncc", "Human–machine pattern NCC", (-0.2, 1.02)),
        (
            "clarity_ratio_vs_manual",
            "Machine / human clarity",
            (0.0, None),
        ),
    ]
    for axis, (metric, title, ylim) in zip(axes[0], roi_metrics):
        data = [
            roi.loc[roi["method"] == method, metric].to_numpy()
            for method in roi_order
        ]
        boxes = axis.boxplot(data, patch_artist=True, showfliers=True)
        for patch, method in zip(boxes["boxes"], roi_order):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)
        axis.set_xticks(range(1, len(roi_order) + 1))
        axis.set_xticklabels(
            [item.replace("_", "\n") for item in roi_order], fontsize=8
        )
        axis.set_title(title)
        if ylim:
            bottom, top = ylim
            axis.set_ylim(bottom=bottom, top=top)
        axis.grid(axis="y", alpha=0.25)
    for axis, (metric, title, ylim) in zip(axes[1], key_metrics):
        data = [
            keyframe.loc[keyframe["method"] == method, metric]
            .dropna()
            .to_numpy()
            for method in key_order
        ]
        boxes = axis.boxplot(data, patch_artist=True, showfliers=True)
        for patch, method in zip(boxes["boxes"], key_order):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)
        axis.set_xticks(range(1, len(key_order) + 1))
        axis.set_xticklabels(
            [item.replace("_", "\n") for item in key_order], fontsize=8
        )
        axis.set_title(title)
        if ylim:
            bottom, top = ylim
            axis.set_ylim(bottom=bottom, top=top)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Automatic RHEED ROI and rotation-phase keyframe benchmark",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_roi_atlases(
    manifest: pd.DataFrame,
    roi: pd.DataFrame,
    report_root: Path,
) -> None:
    for method in roi["method"].drop_duplicates():
        method_rows = roi.loc[roi["method"] == method].set_index(
            "sample_id"
        )
        pages = math.ceil(len(manifest) / 6)
        pdf_path = report_root / "roi_models" / f"{method}_all_samples.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for page in range(pages):
                subset = manifest.iloc[page * 6 : (page + 1) * 6]
                fig, axes = plt.subplots(
                    len(subset), 2, figsize=(9.0, 3.05 * len(subset))
                )
                axes = np.atleast_2d(axes)
                for row_index, (_, record) in enumerate(subset.iterrows()):
                    sample = str(record["sample_id"])
                    frame = load_frame(
                        ROOT / str(record["frames_dir"]),
                        int(record["keyframe_index"]),
                    )
                    manual = Rect(
                        int(record["roi_x"]),
                        int(record["roi_y"]),
                        int(record["roi_width"]),
                        int(record["roi_height"]),
                        int(record["source_width"]),
                        int(record["source_height"]),
                    )
                    metric = method_rows.loc[sample]
                    automatic = Rect(
                        int(metric["auto_x"]),
                        int(metric["auto_y"]),
                        int(metric["auto_width"]),
                        int(metric["auto_height"]),
                        manual.source_width,
                        manual.source_height,
                    )
                    overlay = Image.fromarray(frame).convert("RGB")
                    width = max(3, round(min(overlay.size) / 180))
                    draw_rect(
                        overlay,
                        manual,
                        color=COLORS["manual"],
                        width=width,
                        label="Human ROI",
                    )
                    draw_rect(
                        overlay,
                        automatic,
                        color=COLORS[method],
                        width=width,
                        label=f"{method}",
                    )
                    axes[row_index, 0].imshow(overlay)
                    axes[row_index, 0].axis("off")
                    axes[row_index, 0].set_title(
                        f"{sample} | IoU {metric['iou_vs_manual']:.2f} | "
                        f"safe {100*metric['safe_pixel_fraction']:.1f}%",
                        fontsize=9,
                    )
                    axes[row_index, 1].imshow(
                        crop_display(frame, automatic)
                    )
                    axes[row_index, 1].axis("off")
                    axes[row_index, 1].set_title(
                        "Automatic ROI crop (contrast enhanced)",
                        fontsize=9,
                    )
                fig.suptitle(
                    f"ROI model: {method} — human vs automatic",
                    fontsize=13,
                    fontweight="bold",
                )
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                pdf.savefig(fig, bbox_inches="tight")
                png_path = (
                    report_root
                    / "roi_models"
                    / f"{method}_page_{page + 1:02d}.png"
                )
                fig.savefig(png_path, dpi=190, bbox_inches="tight")
                plt.close(fig)


def save_keyframe_atlases(
    manifest: pd.DataFrame,
    keyframe: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
    report_root: Path,
    final_roi_method: str,
) -> None:
    for method in keyframe["method"].drop_duplicates():
        method_rows = keyframe.loc[
            keyframe["method"] == method
        ].set_index("sample_id")
        pages = math.ceil(len(manifest) / 5)
        pdf_path = (
            report_root / "keyframe_models" / f"{method}_all_samples.pdf"
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for page in range(pages):
                subset = manifest.iloc[page * 5 : (page + 1) * 5]
                fig, axes = plt.subplots(
                    len(subset), 3, figsize=(11.8, 3.2 * len(subset))
                )
                axes = np.atleast_2d(axes)
                for row_index, (_, record) in enumerate(subset.iterrows()):
                    sample = str(record["sample_id"])
                    metric = method_rows.loc[sample]
                    frames_dir = ROOT / str(record["frames_dir"])
                    human_frame = load_frame(
                        frames_dir, int(record["keyframe_index"])
                    )
                    machine_frame = load_frame(
                        frames_dir, int(metric["selected_frame_index"])
                    )
                    roi_payload = selections[sample]["roi_predictions"][
                        final_roi_method
                    ]["rect"]
                    automatic = Rect(**roi_payload)
                    manual = Rect(
                        int(record["roi_x"]),
                        int(record["roi_y"]),
                        int(record["roi_width"]),
                        int(record["roi_height"]),
                        int(record["source_width"]),
                        int(record["source_height"]),
                    )
                    axes[row_index, 0].imshow(
                        crop_display(human_frame, manual)
                    )
                    axes[row_index, 0].axis("off")
                    axes[row_index, 0].set_title(
                        f"{sample} human frame {int(record['keyframe_index'])}",
                        fontsize=9,
                    )
                    axes[row_index, 1].imshow(
                        crop_display(machine_frame, automatic)
                    )
                    axes[row_index, 1].axis("off")
                    axes[row_index, 1].set_title(
                        f"machine frame {int(metric['selected_frame_index'])}",
                        fontsize=9,
                    )
                    trajectory = _load_trajectory(
                        ROOT
                        / "outputs"
                        / "rheed_auto_roi_keyframe"
                        / selections[sample]["experiment_id"]
                        / "trajectories"
                        / f"{sample}.json"
                    )["automatic"]
                    frame_indices = np.asarray(
                        [item["frame_index"] for item in trajectory]
                    )
                    spot_x = np.asarray(
                        [item["spot_x"] for item in trajectory]
                    )
                    spot_x = ndimage.gaussian_filter1d(
                        ndimage.median_filter(spot_x, size=3), 1.5
                    )
                    axes[row_index, 2].plot(
                        frame_indices,
                        spot_x,
                        color="#555555",
                        linewidth=0.8,
                    )
                    axes[row_index, 2].axvline(
                        int(record["keyframe_index"]),
                        color=COLORS["manual"],
                        label="human",
                    )
                    axes[row_index, 2].axvline(
                        int(metric["selected_frame_index"]),
                        color=COLORS[method],
                        linestyle="--",
                        label="machine",
                    )
                    axes[row_index, 2].set_ylabel("tracked spot x (px)")
                    axes[row_index, 2].set_xlabel("frame")
                    axes[row_index, 2].set_title(
                        f"NCC {metric['pattern_ncc']:.2f} | "
                        f"clarity ×{metric['clarity_ratio_vs_manual']:.2f}",
                        fontsize=9,
                    )
                    axes[row_index, 2].grid(alpha=0.2)
                fig.suptitle(
                    f"Keyframe model: {method} — all predictions, no cherry-picking",
                    fontsize=13,
                    fontweight="bold",
                )
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                pdf.savefig(fig, bbox_inches="tight")
                fig.savefig(
                    report_root
                    / "keyframe_models"
                    / f"{method}_page_{page + 1:02d}.png",
                    dpi=190,
                    bbox_inches="tight",
                )
                plt.close(fig)


def save_failure_figure(
    manifest: pd.DataFrame,
    keyframe: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
    report_root: Path,
    *,
    method: str,
    final_roi_method: str,
) -> None:
    selected = (
        keyframe.loc[keyframe["method"] == method]
        .sort_values(["pattern_ncc", "clarity_ratio_vs_manual"])
        .head(6)
    )
    manifest_lookup = manifest.set_index("sample_id")
    fig, axes = plt.subplots(6, 2, figsize=(8.5, 17.0))
    for axis_row, (_, metric) in zip(axes, selected.iterrows()):
        sample = str(metric["sample_id"])
        record = manifest_lookup.loc[sample]
        frames_dir = ROOT / str(record["frames_dir"])
        human_frame = load_frame(frames_dir, int(record["keyframe_index"]))
        machine_frame = load_frame(
            frames_dir, int(metric["selected_frame_index"])
        )
        manual = Rect(
            int(record["roi_x"]),
            int(record["roi_y"]),
            int(record["roi_width"]),
            int(record["roi_height"]),
            int(record["source_width"]),
            int(record["source_height"]),
        )
        automatic = Rect(
            **selections[sample]["roi_predictions"][final_roi_method]["rect"]
        )
        axis_row[0].imshow(crop_display(human_frame, manual))
        axis_row[1].imshow(crop_display(machine_frame, automatic))
        for axis in axis_row:
            axis.axis("off")
        axis_row[0].set_title(f"{sample} human", fontsize=10)
        axis_row[1].set_title(
            f"machine | NCC {metric['pattern_ncc']:.2f} | "
            f"phase err {metric['periodic_phase_error_frames']:.1f} f",
            fontsize=10,
        )
    fig.suptitle(
        f"Lowest-similarity cases: {method} (systematic failure review)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = report_root / "failure_cases"
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / "lowest_similarity_cases.png", dpi=220)
    fig.savefig(path / "lowest_similarity_cases.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    output_root = (
        ROOT / "outputs" / "rheed_auto_roi_keyframe" / experiment_id
    )
    report_root = (
        ROOT / "reports" / "rheed_auto_roi_keyframe" / experiment_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / config["manifest"], dtype={"sample_id": str})
    manifest = manifest.drop_duplicates(
        subset=["sample_id", "source_video"], keep="first"
    ).reset_index(drop=True)
    if args.limit:
        manifest = manifest.iloc[: args.limit].copy()
    required = {
        "sample_id",
        "source_video",
        "frames_dir",
        "keyframe_index",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "source_width",
        "source_height",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Every evaluation row must have a complete annotation")

    frozen_config = {
        **config,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(ROOT / config["manifest"]),
        "evaluated_rows": len(manifest),
        "generated_unix_time": time.time(),
    }
    (output_root / "config_snapshot.json").write_text(
        json.dumps(frozen_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    roi_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    raw_audit_before = {}
    start = time.perf_counter()
    for index, row in manifest.iterrows():
        source_path = ROOT / str(row["source_video"])
        raw_audit_before[str(row["sample_id"])] = (
            source_path.stat().st_size,
            source_path.stat().st_mtime_ns,
        )
        print(
            f"[{index + 1:02d}/{len(manifest):02d}] {row['sample_id']} "
            f"{row['video_id']}",
            flush=True,
        )
        sample_roi, sample_keyframe, selection = evaluate_sample(
            row,
            config=config,
            output_root=output_root,
            reuse=args.reuse,
        )
        selection["experiment_id"] = experiment_id
        roi_rows.extend(sample_roi)
        keyframe_rows.extend(sample_keyframe)
        selections[str(row["sample_id"])] = selection

    roi = pd.DataFrame(roi_rows)
    keyframe = pd.DataFrame(keyframe_rows)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    roi.to_csv(metrics_dir / "roi_predictions.csv", index=False)
    keyframe.to_csv(metrics_dir / "keyframe_predictions.csv", index=False)
    roi_summary, key_summary = summarize_metrics(roi, keyframe)
    roi_summary.to_csv(metrics_dir / "roi_summary.csv", index=False)
    key_summary.to_csv(metrics_dir / "keyframe_summary.csv", index=False)

    raw_audit_after = {
        str(row["sample_id"]): (
            (ROOT / str(row["source_video"])).stat().st_size,
            (ROOT / str(row["source_video"])).stat().st_mtime_ns,
        )
        for _, row in manifest.iterrows()
    }
    raw_ok = raw_audit_before == raw_audit_after
    audit = {
        "raw_source_size_and_mtime_unchanged": raw_ok,
        "source_count": len(raw_audit_before),
        "elapsed_seconds": time.perf_counter() - start,
        "roi_methods": list(roi["method"].drop_duplicates()),
        "keyframe_methods": list(keyframe["method"].drop_duplicates()),
    }
    (output_root / "run_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not raw_ok:
        raise RuntimeError("Raw source size/mtime changed during evaluation")

    save_method_summary_figure(
        roi, keyframe, report_root / "method_benchmark"
    )
    save_roi_atlases(manifest, roi, report_root)
    save_keyframe_atlases(
        manifest,
        keyframe,
        selections,
        report_root,
        config["final_roi_method"],
    )
    save_failure_figure(
        manifest,
        keyframe,
        selections,
        report_root,
        method=config["final_keyframe_method"],
        final_roi_method=config["final_roi_method"],
    )
    print("\nROI summary")
    print(roi_summary.to_string(index=False))
    print("\nKeyframe summary")
    print(key_summary.to_string(index=False))
    print(f"\nElapsed: {audit['elapsed_seconds']:.1f} s")


if __name__ == "__main__":
    main()
