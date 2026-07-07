"""Build exposure-invariant multi-frame RHEED shape-bag inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.frame_quality import finite_float
from rheed2morph.rheed.manual_frame_selection import (
    match_selection,
    parse_manual_selection_line,
)
from rheed2morph.rheed.shape_preprocessing import (
    DEFAULT_CHANNEL_NAMES,
    channels_to_tensor,
    make_rgb_overlay,
    photometric_perturbations,
    preprocess_frame_for_shape,
    read_grayscale_image,
    robust_rescale,
    save_gray_png,
)
from rheed2morph.rheed.spot_streak_geometry import (
    COMPONENT_TYPES,
    FRAME_SHAPE_FEATURE_NAMES,
    colorize_component_overlay,
    component_rows_for_csv,
    extract_components_and_frame_features,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data" / "rheed_roi_shadow_right_v2_main_raw_crop_videos_256"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "rheed_shape_bag_input_mvp"
CONSENSUS_MAP_NAMES = [
    "weighted_mean_log_bgsub",
    "weighted_median_log_bgsub",
    "mask_vote",
    "persistent_mask",
    "dog_max_response",
    "uncertainty_map",
]


@dataclass
class FrameRecord:
    candidate_row: dict[str, str]
    frame_idx: int
    timestamp_sec: float
    rank: int
    png_path: Path
    source_type: str
    raw_gray: np.ndarray
    channels: dict[str, np.ndarray]
    artifact_mask: np.ndarray
    audit: dict[str, float]
    components: list[dict[str, Any]]
    features: dict[str, float]
    frame_weight: float = 0.0
    raw_weight: float = 0.0


@dataclass
class SampleResult:
    sample_id: str
    source_type: str
    sample_folder: Path
    frame_selection_folder: Path
    shape_input_folder: Path
    num_frames_available: int
    num_frames_used: int
    candidate_csv: Path
    manual_selection_file: Path
    shape_bag_npz: Path
    sample_feature_json: Path
    sample_feature_csv: Path
    preview_grid: Path
    exposure_audit_json: Path
    low_confidence_flag: bool
    failure: str = ""


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def git_status_short() -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    except OSError as exc:
        return f"<git status unavailable: {exc}>"
    return result.stdout.strip() or "<clean>"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, bool):
                    encoded[key] = int(value)
                elif isinstance(value, float):
                    encoded[key] = f"{value:.8g}"
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(child) for child in value]
    return value


def candidate_png_path(row: dict[str, str]) -> Path:
    raw = row.get("candidate_png_path", "")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def discover_candidate_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("frame_selection/candidate_frames.csv"))


def parse_manual_rows(manual_file: Path, candidate_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    if not manual_file.is_file():
        return []
    selected: list[dict[str, str]] = []
    for line in manual_file.read_text(encoding="utf-8").splitlines():
        parsed = parse_manual_selection_line(line)
        if parsed is None or parsed.get("kind") == "invalid":
            continue
        matched = match_selection(parsed, candidate_rows)
        if matched is not None:
            selected.append(matched)
    return selected


def select_frame_rows(
    *,
    candidate_csv: Path,
    mode: str,
    candidate_count: int,
    max_frames_per_sample: int,
    min_quality_score: float,
) -> tuple[list[dict[str, str]], str, int]:
    candidate_rows = read_csv(candidate_csv)
    manual_file = candidate_csv.parent / "manual_selected_frames.txt"
    manual_rows = parse_manual_rows(manual_file, candidate_rows)
    rows: list[dict[str, str]]
    source_type = "auto_candidate_fallback"
    if mode in {"manual_only", "manual_or_candidates"} and manual_rows:
        rows = manual_rows
        source_type = "manual"
    elif mode == "manual_only":
        rows = []
        source_type = "manual"
    else:
        rows = sorted(candidate_rows, key=lambda row: int(float(row.get("candidate_rank", "999") or 999)))

    filtered = [
        row
        for row in rows
        if finite_float(row.get("quality_score", 0.0), 0.0) >= min_quality_score
    ]
    limit = min(candidate_count, max_frames_per_sample)
    return filtered[:limit], source_type, len(rows)


def clear_generated_frame_images(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("frame*_*.png"):
        path.unlink()


def save_frame_debug_images(frames_dir: Path, frame: FrameRecord) -> Path:
    frame_idx = frame.frame_idx
    for name, image in frame.channels.items():
        vmin, vmax = (-1.0, 1.0) if image.min() < -0.01 else (0.0, 1.0)
        if name == "soft_spot_streak_mask":
            save_gray_png(frames_dir / f"frame{frame_idx:06d}_soft_mask.png", image, vmin=0.0, vmax=1.0)
        else:
            save_gray_png(frames_dir / f"frame{frame_idx:06d}_{name}.png", image, vmin=vmin, vmax=vmax)
    overlay = make_rgb_overlay(frame.channels["pclip_norm"], frame.channels["soft_spot_streak_mask"])
    overlay_path = frames_dir / f"frame{frame_idx:06d}_overlay.png"
    plt.imsave(overlay_path, overlay)
    return overlay_path


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if np.sum(weights) <= 0:
        return finite_float(np.median(values))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * sorted_weights.sum()
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return finite_float(sorted_values[min(index, sorted_values.size - 1)])


def trimmed_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        return 0.0
    trim = int(values.size * trim_fraction)
    if trim > 0 and values.size > 2 * trim:
        values = values[trim:-trim]
    return finite_float(values.mean())


def compute_frame_weights(frames: list[FrameRecord]) -> None:
    raw_weights = []
    for frame in frames:
        quality = finite_float(frame.candidate_row.get("quality_score", 1.0), 1.0)
        mask_conf = finite_float(frame.features.get("mask_confidence", frame.audit.get("mask_confidence", 0.5)), 0.5)
        non_artifact = 1.0 - finite_float(frame.features.get("artifact_fraction", frame.audit.get("artifact_fraction", 0.0)), 0.0)
        snr = finite_float(frame.features.get("snr_score", frame.audit.get("snr_score", 0.5)), 0.5)
        low_conf = finite_float(frame.candidate_row.get("low_confidence_candidate", 0.0), 0.0)
        penalty = 0.6 if low_conf >= 1 else 1.0
        raw = max(0.0, quality) * max(0.0, mask_conf) * max(0.0, non_artifact) * max(0.0, snr) * penalty
        frame.raw_weight = finite_float(raw)
        raw_weights.append(raw)
    total = float(np.sum(raw_weights))
    if total <= 1e-8 and frames:
        for frame in frames:
            frame.frame_weight = 1.0 / len(frames)
    else:
        for frame in frames:
            frame.frame_weight = finite_float(frame.raw_weight / total)


def aggregate_sample_features(frames: Sequence[FrameRecord]) -> tuple[np.ndarray, list[str], dict[str, float]]:
    names: list[str] = []
    values: list[float] = []
    summary: dict[str, float] = {}
    if not frames:
        return np.zeros(0, dtype=np.float32), names, summary
    weights = np.asarray([frame.frame_weight for frame in frames], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.full(len(frames), 1.0 / len(frames), dtype=np.float64)
    else:
        weights = weights / weights.sum()
    for feature_name in FRAME_SHAPE_FEATURE_NAMES:
        arr = np.asarray([finite_float(frame.features.get(feature_name, 0.0)) for frame in frames], dtype=np.float64)
        weighted_mean = finite_float(np.sum(arr * weights))
        median = weighted_median(arr, weights)
        iqr = finite_float(np.percentile(arr, 75) - np.percentile(arr, 25))
        std = finite_float(np.sqrt(np.sum(weights * (arr - weighted_mean) ** 2)))
        tmean = trimmed_mean(arr)
        for suffix, value in [
            ("weighted_mean", weighted_mean),
            ("weighted_median", median),
            ("trimmed_mean", tmean),
            ("std", std),
            ("iqr", iqr),
        ]:
            key = f"{suffix}_{feature_name}"
            names.append(key)
            values.append(value)
            summary[key] = value
    return np.asarray(values, dtype=np.float32), names, summary


def build_consensus_maps(frames_array: np.ndarray, frame_mask: np.ndarray, frame_weights: np.ndarray) -> np.ndarray:
    valid = frame_mask > 0.5
    height, width = frames_array.shape[-2:]
    if not np.any(valid):
        return np.zeros((len(CONSENSUS_MAP_NAMES), height, width), dtype=np.float32)
    frames = frames_array[valid]
    weights = frame_weights[valid].astype(np.float32)
    if weights.sum() <= 0:
        weights = np.full(frames.shape[0], 1.0 / frames.shape[0], dtype=np.float32)
    else:
        weights = weights / weights.sum()
    log_channel = frames[:, DEFAULT_CHANNEL_NAMES.index("log_bgsub")]
    dog_channel = frames[:, DEFAULT_CHANNEL_NAMES.index("dog_response")]
    mask_channel = frames[:, DEFAULT_CHANNEL_NAMES.index("soft_spot_streak_mask")]
    weighted_mean = np.sum(log_channel * weights[:, None, None], axis=0)
    median = np.median(log_channel, axis=0)
    mask_vote = np.sum(mask_channel * weights[:, None, None], axis=0)
    persistent = (mask_vote >= 0.40).astype(np.float32)
    dog_max = np.max(dog_channel, axis=0)
    variance = np.sum(weights[:, None, None] * (log_channel - weighted_mean[None, :, :]) ** 2, axis=0)
    maps = np.stack([weighted_mean, median, mask_vote, persistent, dog_max, np.sqrt(variance)], axis=0)
    return np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def write_shape_npz(
    path: Path,
    *,
    frames: Sequence[FrameRecord],
    image_size: int,
    max_frames: int,
    sample_feature_vector: np.ndarray,
    sample_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_count = len(DEFAULT_CHANNEL_NAMES)
    frames_array = np.zeros((max_frames, channel_count, image_size, image_size), dtype=np.float32)
    frame_mask = np.zeros(max_frames, dtype=np.float32)
    frame_weights = np.zeros(max_frames, dtype=np.float32)
    frame_indices = np.full(max_frames, -1, dtype=np.int32)
    timestamps = np.full(max_frames, np.nan, dtype=np.float32)
    for index, frame in enumerate(frames[:max_frames]):
        frames_array[index] = channels_to_tensor(frame.channels, DEFAULT_CHANNEL_NAMES)
        frame_mask[index] = 1.0
        frame_weights[index] = frame.frame_weight
        frame_indices[index] = frame.frame_idx
        timestamps[index] = frame.timestamp_sec
    consensus_maps = build_consensus_maps(frames_array, frame_mask, frame_weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames_array,
        frame_mask=frame_mask,
        frame_weights=frame_weights,
        consensus_maps=consensus_maps,
        consensus_map_names=np.asarray(CONSENSUS_MAP_NAMES, dtype="U"),
        sample_feature_vector=sample_feature_vector.astype(np.float32, copy=False),
        sample_feature_names=np.asarray(sample_feature_names, dtype="U"),
        channel_names=np.asarray(DEFAULT_CHANNEL_NAMES, dtype="U"),
        frame_indices=frame_indices,
        timestamps_sec=timestamps,
    )
    return frames_array, frame_mask, frame_weights, consensus_maps, frame_indices, timestamps


def write_consensus_maps(path: Path, consensus_maps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, maps=consensus_maps, names=np.asarray(CONSENSUS_MAP_NAMES, dtype="U"))


def write_overview_grid(path: Path, frames: Sequence[FrameRecord], max_rows: int = 8) -> None:
    rows = min(max_rows, len(frames))
    if rows <= 0:
        return
    columns = ["original", "pclip_norm", "log_bgsub", "local_zscore", "dog_response", "soft mask", "components"]
    fig, axes = plt.subplots(rows, len(columns), figsize=(len(columns) * 2.0, rows * 2.1), squeeze=False)
    for row_index, frame in enumerate(frames[:rows]):
        images = [
            frame.raw_gray,
            frame.channels["pclip_norm"],
            frame.channels["log_bgsub"],
            frame.channels["local_zscore"],
            frame.channels["dog_response"],
            frame.channels["soft_spot_streak_mask"],
            colorize_component_overlay(frame.channels["pclip_norm"], frame.components, frame.channels["soft_spot_streak_mask"]),
        ]
        for col_index, (image, title) in enumerate(zip(images, columns)):
            axis = axes[row_index, col_index]
            if image.ndim == 3:
                axis.imshow(image)
            else:
                vmin, vmax = (-1.0, 1.0) if title in {"log_bgsub", "local_zscore"} else (0.0, 1.0)
                axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title, fontsize=8)
            if col_index == 0:
                axis.set_ylabel(f"rank{frame.rank:02d}\nframe {frame.frame_idx}", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_consensus_grid(path: Path, consensus_maps: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(CONSENSUS_MAP_NAMES), figsize=(len(CONSENSUS_MAP_NAMES) * 2.4, 2.5), squeeze=False)
    for axis, image, name in zip(axes.ravel(), consensus_maps, CONSENSUS_MAP_NAMES):
        vmin, vmax = (-1.0, 1.0) if "log_bgsub" in name else (0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(name.replace("_", "\n"), fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_component_grid(path: Path, frames: Sequence[FrameRecord], max_rows: int = 16) -> None:
    count = min(max_rows, len(frames))
    if count <= 0:
        return
    cols = 4
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.7), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, frame in zip(axes.ravel(), frames[:count]):
        overlay = colorize_component_overlay(frame.channels["pclip_norm"], frame.components, frame.channels["soft_spot_streak_mask"])
        axis.imshow(overlay)
        axis.set_title(
            f"rank{frame.rank:02d} frame {frame.frame_idx}\nbar={frame.features.get('bar_like_score', 0.0):.2f}",
            fontsize=8,
        )
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_frame_weight_plot(path: Path, frames: Sequence[FrameRecord]) -> None:
    if not frames:
        return
    x = np.arange(len(frames))
    quality = np.asarray([finite_float(frame.candidate_row.get("quality_score", 0.0)) for frame in frames])
    mask_conf = np.asarray([finite_float(frame.features.get("mask_confidence", 0.0)) for frame in frames])
    artifact_score = np.asarray([1.0 - finite_float(frame.features.get("artifact_fraction", 0.0)) for frame in frames])
    weights = np.asarray([frame.frame_weight for frame in frames])
    fig, axis = plt.subplots(figsize=(9, 4))
    width = 0.20
    axis.bar(x - 1.5 * width, quality, width=width, label="quality")
    axis.bar(x - 0.5 * width, mask_conf, width=width, label="mask confidence")
    axis.bar(x + 0.5 * width, artifact_score, width=width, label="non-artifact")
    axis.bar(x + 1.5 * width, weights, width=width, label="final weight")
    axis.set_xticks(x)
    axis.set_xticklabels([f"r{frame.rank:02d}" for frame in frames], rotation=45, fontsize=7)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_feature_summary(path: Path, sample_feature_summary: dict[str, float]) -> None:
    keys = [
        "weighted_mean_round_spot_count",
        "weighted_mean_elongated_spot_count",
        "weighted_mean_horizontal_bar_count",
        "weighted_mean_vertical_streak_count",
        "weighted_mean_bar_like_score",
        "weighted_mean_spot_to_streak_ratio",
        "weighted_mean_mean_aspect_ratio",
        "weighted_mean_orientation_entropy",
        "weighted_mean_horizontal_spacing_peak",
        "weighted_mean_fft_anisotropy",
    ]
    lines = ["# RHEED Shape Feature Summary", "", "| feature | value |", "| --- | ---: |"]
    for key in keys:
        lines.append(f"| `{key}` | {sample_feature_summary.get(key, 0.0):.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-8:
        return 0.0
    return finite_float(np.dot(a, b) / denom)


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a) > 0.4
    bb = np.asarray(b) > 0.4
    denom = aa.sum() + bb.sum()
    if denom <= 0:
        return 1.0
    return finite_float(2.0 * np.logical_and(aa, bb).sum() / denom)


def run_exposure_audit(sample_dir: Path, frames: Sequence[FrameRecord], image_size: int) -> tuple[dict[str, Any], Path]:
    audit_path = sample_dir / "exposure_invariance_audit.json"
    plot_path = sample_dir / "exposure_invariance_audit.png"
    if not frames:
        audit_path.write_text(json.dumps({"status": "no_frames"}, indent=2), encoding="utf-8")
        return {"status": "no_frames"}, audit_path
    raw_means = []
    shape_vectors = []
    mask_scores = []
    perturbation_names = []
    reference_mask = None
    for frame in frames[: min(2, len(frames))]:
        for name, perturbed in photometric_perturbations(frame.raw_gray).items():
            processed = preprocess_frame_for_shape(perturbed, image_size=image_size)
            components, features = extract_components_and_frame_features(
                soft_mask=processed.channels["soft_spot_streak_mask"],
                enhanced_image=processed.channels["log_bgsub"],
                artifact_mask=processed.artifact_mask,
            )
            _ = components
            raw_means.append(processed.audit_features["raw_mean"])
            shape_vectors.append(np.asarray([features[name] for name in FRAME_SHAPE_FEATURE_NAMES], dtype=np.float32))
            if reference_mask is None:
                reference_mask = processed.channels["soft_spot_streak_mask"]
            mask_scores.append(mask_dice(reference_mask, processed.channels["soft_spot_streak_mask"]))
            perturbation_names.append(name)
    shape_stack = np.stack(shape_vectors, axis=0) if shape_vectors else np.zeros((0, len(FRAME_SHAPE_FEATURE_NAMES)))
    raw_arr = np.asarray(raw_means, dtype=np.float64)
    raw_cv = finite_float(raw_arr.std() / max(abs(raw_arr.mean()), 1e-8)) if raw_arr.size else 0.0
    feature_means = np.mean(np.abs(shape_stack), axis=0) if shape_stack.size else np.ones(len(FRAME_SHAPE_FEATURE_NAMES))
    feature_stds = np.std(shape_stack, axis=0) if shape_stack.size else np.zeros(len(FRAME_SHAPE_FEATURE_NAMES))
    feature_cv = feature_stds / np.maximum(feature_means, 1e-6)
    reference_vector = shape_stack[0] if shape_stack.size else np.zeros(len(FRAME_SHAPE_FEATURE_NAMES), dtype=np.float32)
    cosines = [cosine_similarity(reference_vector, vector) for vector in shape_stack[1:]] if shape_stack.shape[0] > 1 else [1.0]
    unstable = [name for name, cv in zip(FRAME_SHAPE_FEATURE_NAMES, feature_cv) if finite_float(cv) > 0.75]
    audit = {
        "status": "ok",
        "perturbations": perturbation_names,
        "raw_brightness_cv": raw_cv,
        "shape_feature_cv_median": finite_float(np.median(feature_cv)) if feature_cv.size else 0.0,
        "shape_feature_cv_mean": finite_float(np.mean(feature_cv)) if feature_cv.size else 0.0,
        "sample_feature_cosine_mean": finite_float(np.mean(cosines)),
        "mask_dice_mean": finite_float(np.mean(mask_scores) if mask_scores else 0.0),
        "unstable_features": unstable,
        "default_training_feature_names": [name for name in FRAME_SHAPE_FEATURE_NAMES if name not in unstable],
        "interpretation": "shape features are more stable than raw brightness"
        if (finite_float(np.median(feature_cv)) < raw_cv)
        else "some shape features remain exposure-sensitive; inspect unstable_features",
    }
    audit_path.write_text(json.dumps(json_ready(audit), indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.bar(["raw brightness CV", "median shape CV", "mean mask Dice", "mean cosine"], [
        audit["raw_brightness_cv"],
        audit["shape_feature_cv_median"],
        audit["mask_dice_mean"],
        audit["sample_feature_cosine_mean"],
    ])
    axis.set_ylim(0.0, max(1.0, audit["raw_brightness_cv"] * 1.2))
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return audit, audit_path


def process_sample(candidate_csv: Path, args: argparse.Namespace) -> tuple[SampleResult, dict[str, Any] | None]:
    frame_selection_dir = candidate_csv.parent
    sample_folder = frame_selection_dir.parent
    sample_id = sample_folder.name
    shape_dir = sample_folder / "rheed_shape_input"
    frames_dir = shape_dir / "frames"
    shape_dir.mkdir(parents=True, exist_ok=True)
    if args.write_debug_images:
        clear_generated_frame_images(frames_dir)

    selected_rows, source_type, available_count = select_frame_rows(
        candidate_csv=candidate_csv,
        mode=args.mode,
        candidate_count=args.candidate_count,
        max_frames_per_sample=args.max_frames_per_sample,
        min_quality_score=args.min_quality_score,
    )
    frame_records: list[FrameRecord] = []
    component_rows: list[dict[str, Any]] = []
    frame_feature_rows: list[dict[str, Any]] = []
    missing_frames: list[str] = []
    for row in selected_rows:
        png_path = candidate_png_path(row)
        if not png_path.is_file():
            missing_frames.append(display_path(png_path))
            continue
        frame_idx = int(float(row.get("frame_idx", "-1") or -1))
        timestamp = finite_float(row.get("timestamp_sec", 0.0), 0.0)
        rank = int(float(row.get("candidate_rank", len(frame_records) + 1) or len(frame_records) + 1))
        raw = read_grayscale_image(png_path, image_size=args.image_size)
        processed = preprocess_frame_for_shape(raw, image_size=args.image_size)
        components, features = extract_components_and_frame_features(
            soft_mask=processed.channels["soft_spot_streak_mask"],
            enhanced_image=processed.channels["log_bgsub"],
            artifact_mask=processed.artifact_mask,
        )
        record = FrameRecord(
            candidate_row=row,
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            rank=rank,
            png_path=png_path,
            source_type=source_type,
            raw_gray=processed.raw_gray,
            channels=processed.channels,
            artifact_mask=processed.artifact_mask,
            audit=processed.audit_features,
            components=components,
            features=features,
        )
        frame_records.append(record)
        component_rows.extend(component_rows_for_csv(sample_id, frame_idx, components))

    compute_frame_weights(frame_records)
    for record in frame_records:
        if args.write_debug_images:
            save_frame_debug_images(frames_dir, record)
        row = {
            "sample_id": sample_id,
            "frame_idx": record.frame_idx,
            "candidate_rank": record.rank,
            "timestamp_sec": record.timestamp_sec,
            "candidate_png_path": display_path(record.png_path),
            "quality_score": finite_float(record.candidate_row.get("quality_score", 0.0)),
            "frame_weight": record.frame_weight,
            "raw_frame_weight": record.raw_weight,
        }
        row.update(record.audit)
        row.update(record.features)
        frame_feature_rows.append(row)

    sample_vector, sample_feature_names, sample_summary = aggregate_sample_features(frame_records)
    shape_bag_npz = shape_dir / "shape_bag.npz"
    frames_array, frame_mask, frame_weights, consensus_maps, frame_indices, timestamps = write_shape_npz(
        shape_bag_npz,
        frames=frame_records,
        image_size=args.image_size,
        max_frames=args.max_frames_per_sample,
        sample_feature_vector=sample_vector,
        sample_feature_names=sample_feature_names,
    )
    _ = frames_array, frame_mask, frame_weights, frame_indices, timestamps
    write_consensus_maps(shape_dir / "consensus_maps.npz", consensus_maps)

    component_csv = shape_dir / "frame_geometry_components.csv"
    component_fieldnames = [
        "sample_id",
        "frame_idx",
        "component_id",
        "component_type",
        "centroid_x",
        "centroid_y",
        "area",
        "bbox_width",
        "bbox_height",
        "aspect_ratio",
        "eccentricity",
        "orientation",
        "major_axis_length",
        "minor_axis_length",
        "solidity",
        "mean_enhanced_intensity",
        "local_background",
        "relative_intensity",
        "fwhm_major_proxy",
        "fwhm_minor_proxy",
        "artifact_fraction",
        "near_border",
    ]
    write_csv(component_csv, component_rows, component_fieldnames)
    frame_features_csv = shape_dir / "frame_shape_features.csv"
    write_csv(
        frame_features_csv,
        frame_feature_rows,
        [
            "sample_id",
            "frame_idx",
            "candidate_rank",
            "timestamp_sec",
            "candidate_png_path",
            "quality_score",
            "frame_weight",
            "raw_frame_weight",
            "raw_mean",
            "raw_std",
            *FRAME_SHAPE_FEATURE_NAMES,
        ],
    )
    (shape_dir / "component_geometry.json").write_text(
        json.dumps(json_ready({"sample_id": sample_id, "frames": frame_feature_rows, "components": component_rows}), indent=2),
        encoding="utf-8",
    )
    sample_csv = shape_dir / "sample_shape_features.csv"
    sample_json = shape_dir / "sample_shape_features.json"
    sample_row = {"sample_id": sample_id, **sample_summary}
    write_csv(sample_csv, [sample_row], ["sample_id", *sample_feature_names])
    sample_json.write_text(
        json.dumps(
            json_ready(
                {
                    "sample_id": sample_id,
                    "source_type": source_type,
                    "sample_feature_names": sample_feature_names,
                    "sample_feature_vector": sample_vector,
                    "sample_features": sample_summary,
                    "missing_frames": missing_frames,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    overview = shape_dir / "shape_input_overview.png"
    consensus_png = shape_dir / "consensus_shape_maps.png"
    component_png = shape_dir / "component_geometry_grid.png"
    overlay_grid = shape_dir / "component_overlay_grid.png"
    weight_png = shape_dir / "frame_weight_diagnostics.png"
    write_overview_grid(overview, frame_records)
    write_consensus_grid(consensus_png, consensus_maps)
    write_component_grid(component_png, frame_records)
    write_component_grid(overlay_grid, frame_records)
    write_frame_weight_plot(weight_png, frame_records)
    write_feature_summary(shape_dir / "feature_summary.md", sample_summary)

    audit: dict[str, Any] | None = None
    audit_path = shape_dir / "exposure_invariance_audit.json"
    if args.exposure_audit:
        audit, audit_path = run_exposure_audit(shape_dir, frame_records, args.image_size)
    else:
        audit_path.write_text(json.dumps({"status": "skipped"}, indent=2), encoding="utf-8")
        audit = {"status": "skipped"}

    low_confidence = (
        len(frame_records) < args.max_frames_per_sample
        or any(finite_float(frame.candidate_row.get("low_confidence_candidate", 0.0), 0.0) >= 1 for frame in frame_records)
        or finite_float(np.sum([frame.frame_weight > 0.01 for frame in frame_records]), 0.0) < max(2, len(frame_records) // 2)
    )
    result = SampleResult(
        sample_id=sample_id,
        source_type=source_type,
        sample_folder=sample_folder,
        frame_selection_folder=frame_selection_dir,
        shape_input_folder=shape_dir,
        num_frames_available=available_count,
        num_frames_used=len(frame_records),
        candidate_csv=candidate_csv,
        manual_selection_file=frame_selection_dir / "manual_selected_frames.txt",
        shape_bag_npz=shape_bag_npz,
        sample_feature_json=sample_json,
        sample_feature_csv=sample_csv,
        preview_grid=overview,
        exposure_audit_json=audit_path,
        low_confidence_flag=low_confidence,
    )
    return result, audit


def manifest_row(result: SampleResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "source_type": result.source_type,
        "sample_folder": display_path(result.sample_folder),
        "frame_selection_folder": display_path(result.frame_selection_folder),
        "shape_input_folder": display_path(result.shape_input_folder),
        "num_frames_available": result.num_frames_available,
        "num_frames_used": result.num_frames_used,
        "candidate_csv": display_path(result.candidate_csv),
        "manual_selection_file": display_path(result.manual_selection_file),
        "shape_bag_npz": display_path(result.shape_bag_npz),
        "sample_feature_json": display_path(result.sample_feature_json),
        "sample_feature_csv": display_path(result.sample_feature_csv),
        "preview_grid": display_path(result.preview_grid),
        "exposure_audit_json": display_path(result.exposure_audit_json),
        "low_confidence_flag": int(result.low_confidence_flag),
    }


MANIFEST_FIELDS = [
    "sample_id",
    "source_type",
    "sample_folder",
    "frame_selection_folder",
    "shape_input_folder",
    "num_frames_available",
    "num_frames_used",
    "candidate_csv",
    "manual_selection_file",
    "shape_bag_npz",
    "sample_feature_json",
    "sample_feature_csv",
    "preview_grid",
    "exposure_audit_json",
    "low_confidence_flag",
]


def collect_environment() -> dict[str, str]:
    env = {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "numpy": np.__version__}
    for package in ("scipy", "skimage", "cv2", "torch"):
        try:
            module = __import__(package)
            env[package] = getattr(module, "__version__", "available")
        except ModuleNotFoundError:
            env[package] = "not available"
    return env


def write_global_feature_table(report_dir: Path, results: Sequence[SampleResult]) -> Path:
    rows = []
    fieldnames: list[str] = []
    for result in results:
        if result.failure or not result.sample_feature_csv.is_file():
            continue
        row = read_csv(result.sample_feature_csv)[0]
        rows.append(row)
        if not fieldnames:
            fieldnames = list(row.keys())
    path = report_dir / "global_sample_shape_features.csv"
    write_csv(path, rows, fieldnames or ["sample_id"])
    return path


def write_global_exposure_summary(report_dir: Path, audits: dict[str, dict[str, Any]]) -> tuple[Path, Path, Path]:
    rows = []
    for sample_id, audit in audits.items():
        rows.append(
            {
                "sample_id": sample_id,
                "status": audit.get("status", ""),
                "raw_brightness_cv": audit.get("raw_brightness_cv", ""),
                "shape_feature_cv_median": audit.get("shape_feature_cv_median", ""),
                "shape_feature_cv_mean": audit.get("shape_feature_cv_mean", ""),
                "sample_feature_cosine_mean": audit.get("sample_feature_cosine_mean", ""),
                "mask_dice_mean": audit.get("mask_dice_mean", ""),
                "unstable_feature_count": len(audit.get("unstable_features", [])),
            }
        )
    csv_path = report_dir / "global_exposure_invariance_summary.csv"
    write_csv(
        csv_path,
        rows,
        [
            "sample_id",
            "status",
            "raw_brightness_cv",
            "shape_feature_cv_median",
            "shape_feature_cv_mean",
            "sample_feature_cosine_mean",
            "mask_dice_mean",
            "unstable_feature_count",
        ],
    )
    ok_rows = [row for row in rows if row["status"] == "ok"]
    raw_cv = np.asarray([finite_float(row["raw_brightness_cv"]) for row in ok_rows], dtype=float)
    shape_cv = np.asarray([finite_float(row["shape_feature_cv_median"]) for row in ok_rows], dtype=float)
    unstable_counts = {name: 0 for name in FRAME_SHAPE_FEATURE_NAMES}
    for audit in audits.values():
        for feature_name in audit.get("unstable_features", []):
            if feature_name in unstable_counts:
                unstable_counts[feature_name] += 1
    max_unstable = max(1, int(math.ceil(max(len(ok_rows), 1) * 0.25)))
    default_features = [name for name in FRAME_SHAPE_FEATURE_NAMES if unstable_counts.get(name, 0) <= max_unstable]
    default_path = report_dir / "default_training_feature_names.txt"
    default_path.write_text("\n".join(default_features) + "\n", encoding="utf-8")
    median_raw = finite_float(np.median(raw_cv)) if raw_cv.size else 0.0
    median_shape = finite_float(np.median(shape_cv)) if shape_cv.size else 0.0
    if raw_cv.size and shape_cv.size and median_shape < median_raw:
        interpretation = "Shape features were more stable than raw brightness by the median CV metric."
        recommendation = "Proceed with the default stable feature list and keep the audit in the next experiment."
    else:
        interpretation = "Some shape features were more exposure-sensitive than raw brightness by the median CV metric."
        recommendation = (
            "Use `default_training_feature_names.txt`, prioritize continuous geometry and consensus-map inputs, "
            "and tune mask thresholds before relying on raw component-count features."
        )
    report_path = report_dir / "global_exposure_invariance_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Global Exposure Invariance Summary",
                "",
                f"Samples audited: {len(ok_rows)}",
                f"Median raw brightness CV: {median_raw:.6g}" if raw_cv.size else "Median raw brightness CV: n/a",
                f"Median shape feature CV: {median_shape:.6g}" if shape_cv.size else "Median shape feature CV: n/a",
                f"Default stable feature count: {len(default_features)} / {len(FRAME_SHAPE_FEATURE_NAMES)}",
                f"Default stable feature list: `{display_path(default_path)}`",
                "",
                interpretation,
                "",
                f"Recommendation: {recommendation}",
                "",
                "Inspect sample-level audit JSON files for unstable feature names and perturbation-level behavior.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, report_path, default_path


def _load_preview(path: Path) -> np.ndarray | None:
    try:
        import imageio.v3 as iio

        return iio.imread(path)
    except Exception:
        return None


def write_global_overview(report_dir: Path, results: Sequence[SampleResult]) -> Path:
    path = report_dir / "global_shape_bag_overview.png"
    usable = [result for result in results if not result.failure and result.preview_grid.is_file()]
    if not usable:
        return path
    sample = usable[: min(12, len(usable))]
    cols = 3
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.2), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, result in zip(axes.ravel(), sample):
        image = _load_preview(result.preview_grid)
        if image is not None:
            axis.imshow(image)
        axis.set_title(f"{result.sample_id} | {result.source_type}", fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_component_type_summary(report_dir: Path, feature_table: Path) -> Path:
    path = report_dir / "global_component_type_summary.png"
    rows = read_csv(feature_table)
    if not rows:
        return path
    values = []
    labels = ["round_spot", "elongated_spot", "horizontal_bar", "vertical_streak", "diffuse_blob", "artifact_candidate"]
    for label in labels:
        key = f"weighted_mean_{label}_count"
        values.append(sum(finite_float(row.get(key, 0.0)) for row in rows))
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values)
    axis.tick_params(axis="x", labelrotation=30)
    axis.set_ylabel("quality-weighted count")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_example_grid(report_dir: Path, results: Sequence[SampleResult], feature_table: Path, *, key: str, output_name: str, reverse: bool = True) -> Path:
    path = report_dir / output_name
    rows = read_csv(feature_table)
    by_id = {result.sample_id: result for result in results if not result.failure}
    rows = [row for row in rows if row.get("sample_id") in by_id]
    rows.sort(key=lambda row: finite_float(row.get(key, 0.0)), reverse=reverse)
    selected = rows[: min(8, len(rows))]
    if not selected:
        return path
    cols = 4
    grid_rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.2, grid_rows * 2.6), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), selected):
        result = by_id[row["sample_id"]]
        image = _load_preview(result.preview_grid)
        if image is not None:
            axis.imshow(image)
        axis.set_title(f"{row['sample_id']} {key}={finite_float(row.get(key, 0.0)):.2f}", fontsize=8)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_codex_report(
    report_dir: Path,
    *,
    args: argparse.Namespace,
    git_before: str,
    git_after: str,
    env: dict[str, str],
    results: Sequence[SampleResult],
    failures: Sequence[SampleResult],
    manifest_path: Path,
    feature_table_path: Path,
    exposure_summary_path: Path,
    global_overview_path: Path,
    component_summary_path: Path,
    bar_examples_path: Path,
    low_quality_examples_path: Path,
    default_training_features_path: Path,
    commands: Sequence[str],
) -> Path:
    successful = [result for result in results if not result.failure]
    manual_count = sum(1 for result in successful if result.source_type == "manual")
    fallback_count = sum(1 for result in successful if result.source_type == "auto_candidate_fallback")
    low_count = sum(1 for result in successful if result.low_confidence_flag)
    examples = [display_path(result.preview_grid) for result in successful[:5]]
    report = report_dir / "codex_report.md"
    text = f"""# RHEED shape-bag input MVP report

Generated: {datetime.now(UTC).isoformat(timespec="seconds")}

This task builds exposure-invariant multi-frame RHEED shape-bag inputs. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status

Before:

```text
{git_before}
```

After:

```text
{git_after}
```

## Commands

```bash
{chr(10).join(commands)}
```

## Environment

| package | version |
| --- | --- |
| python | {env.get("python", "")} |
| platform | {env.get("platform", "")} |
| numpy | {env.get("numpy", "")} |
| scipy | {env.get("scipy", "")} |
| skimage | {env.get("skimage", "")} |
| cv2 | {env.get("cv2", "")} |
| torch | {env.get("torch", "")} |

## Input Inventory

- Sample folders found: {len(discover_candidate_csvs(resolve_path(args.root)))}
- Candidate CSVs found: {len(discover_candidate_csvs(resolve_path(args.root)))}
- Manual selections used: {manual_count}
- Auto-candidate fallbacks: {fallback_count}
- Samples processed: {len(successful)}
- Failures: {len(failures)}

## Representation Design

Preprocessing channels: `{", ".join(DEFAULT_CHANNEL_NAMES)}`. Background subtraction uses `log1p(alpha * pclip_norm)` followed by Gaussian low-frequency background subtraction and robust rescaling. Local normalization uses local mean/std z-scoring with a 31-pixel default window. Mask extraction combines positive background-subtracted signal, DoG response, local z-score, and artifact masking. Components are classified with deterministic aspect-ratio, eccentricity, orientation, area, border, and artifact rules.

## Multi-Frame Aggregation

Frame weights use `candidate_quality_score * mask_confidence * non_artifact_score * SNR_score`, with low-confidence candidates penalized. Scalar shape features save weighted mean, weighted median, trimmed mean, weighted std, and IQR. Consensus maps include weighted mean log-bgsub, median log-bgsub, mask vote, persistent mask, DoG max response, and uncertainty. Translation alignment is currently not applied; all selected crop frames are assumed pre-cropped and comparable.

## Exposure Invariance

Perturbations tested: brightness scales, contrast scales, gamma shifts, low-frequency gradient, mild noise, and mild blur. Raw brightness stability and shape-feature stability are summarized in `{display_path(exposure_summary_path)}`. Exposure-sensitive feature names are recorded in each sample's `exposure_invariance_audit.json`. A global default stable feature list is written to `{display_path(default_training_features_path)}`.

## Geometry Feature Summary

- Frame feature count: {len(FRAME_SHAPE_FEATURE_NAMES)}
- Low-confidence samples: {low_count}
- Component type summary: `{display_path(component_summary_path)}`
- High `bar_like_score` examples: `{display_path(bar_examples_path)}`
- Low-quality examples: `{display_path(low_quality_examples_path)}`

## Output Locations

- Global manifest: `{display_path(manifest_path)}`
- Global feature table: `{display_path(feature_table_path)}`
- Global overview grid: `{display_path(global_overview_path)}`
- Example `shape_input_overview.png` paths:
{chr(10).join(f"  - `{path}`" for path in examples)}

## Dataset And Encoder Interface

`shape_bag.npz` contains `frames [K,C,H,W]`, `frame_mask [K]`, `frame_weights [K]`, `consensus_maps [6,H,W]`, `sample_feature_vector [F]`, `sample_feature_names`, `frame_indices`, and `timestamps_sec`. `RHEEDShapeBagDataset` returns those arrays as tensors plus `sample_id` and `source_type`. `RHEEDShapeBagEncoder` accepts variable `K` with `frame_mask` and `frame_weights` for multi-instance pooling and emits `sample_embedding`, `attention_weights`, and `frame_embeddings`.

## Known Limitations

- Component rules are transparent heuristics, not a trained RHEED detector.
- Alignment is interface-ready but not enabled by default in this MVP.
- Exposure-invariance audit is diagnostic and may still flag threshold-sensitive count features.
- Manual visual verification remains required before using the representation in a supervised RHEED-to-AFM experiment.

## Recommended Next Command

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_input.py
```
"""
    report.write_text(text, encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--mode", choices=["manual_only", "candidates_only", "manual_or_candidates"], default="manual_or_candidates")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--min-quality-score", type=float, default=0.0)
    parser.add_argument("--max-frames-per-sample", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--align-frames", type=str_to_bool, default=False)
    parser.add_argument("--write-debug-images", type=str_to_bool, default=True)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--strict", type=str_to_bool, default=False)
    parser.add_argument("--exposure-audit", type=str_to_bool, default=True)
    parser.add_argument("--make-global-report", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-limit", type=int, default=3)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    root = resolve_path(args.root)
    report_root = resolve_path(args.report_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_dir = report_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    git_before = git_status_short()
    command_args = sys.argv[1:] if argv is None else list(argv)
    current_command = "PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_inputs " + shlex.join(command_args)
    candidate_csvs = discover_candidate_csvs(root)
    if args.debug:
        candidate_csvs = candidate_csvs[: max(1, args.debug_sample_limit)]
        print(f"[debug] Processing first {len(candidate_csvs)} samples.")
    print(f"Found {len(candidate_csvs)} candidate CSVs.")

    results: list[SampleResult] = []
    audits: dict[str, dict[str, Any]] = {}
    for index, candidate_csv in enumerate(candidate_csvs, start=1):
        sample_id = candidate_csv.parent.parent.name
        print(f"[{index}/{len(candidate_csvs)}] {sample_id}")
        try:
            result, audit = process_sample(candidate_csv, args)
            if audit is not None:
                audits[result.sample_id] = audit
        except Exception as exc:
            if args.strict:
                raise
            result = SampleResult(
                sample_id=sample_id,
                source_type="failed",
                sample_folder=candidate_csv.parent.parent,
                frame_selection_folder=candidate_csv.parent,
                shape_input_folder=candidate_csv.parent.parent / "rheed_shape_input",
                num_frames_available=0,
                num_frames_used=0,
                candidate_csv=candidate_csv,
                manual_selection_file=candidate_csv.parent / "manual_selected_frames.txt",
                shape_bag_npz=candidate_csv.parent.parent / "rheed_shape_input" / "shape_bag.npz",
                sample_feature_json=candidate_csv.parent.parent / "rheed_shape_input" / "sample_shape_features.json",
                sample_feature_csv=candidate_csv.parent.parent / "rheed_shape_input" / "sample_shape_features.csv",
                preview_grid=candidate_csv.parent.parent / "rheed_shape_input" / "shape_input_overview.png",
                exposure_audit_json=candidate_csv.parent.parent / "rheed_shape_input" / "exposure_invariance_audit.json",
                low_confidence_flag=True,
                failure=f"{type(exc).__name__}: {exc}",
            )
            print(f"  failed: {result.failure}")
        results.append(result)

    successful = [result for result in results if not result.failure]
    failures = [result for result in results if result.failure]
    manifest_path = report_dir / "rheed_shape_bag_manifest.csv"
    write_csv(manifest_path, [manifest_row(result) for result in successful], MANIFEST_FIELDS)
    write_csv(
        report_dir / "failed_samples.csv",
        [{"sample_id": result.sample_id, "candidate_csv": display_path(result.candidate_csv), "failure": result.failure} for result in failures],
        ["sample_id", "candidate_csv", "failure"],
    )
    feature_table_path = write_global_feature_table(report_dir, successful)
    exposure_summary_path, _, default_training_features_path = write_global_exposure_summary(report_dir, audits)
    global_overview = write_global_overview(report_dir, successful)
    component_summary = write_component_type_summary(report_dir, feature_table_path)
    bar_examples = write_example_grid(
        report_dir,
        successful,
        feature_table_path,
        key="weighted_mean_bar_like_score",
        output_name="global_bar_like_score_examples.png",
        reverse=True,
    )
    low_quality_examples = write_example_grid(
        report_dir,
        successful,
        feature_table_path,
        key="weighted_mean_mask_confidence",
        output_name="global_low_quality_examples.png",
        reverse=False,
    )

    env = collect_environment()
    git_after = git_status_short()
    report_path = write_codex_report(
        report_dir,
        args=args,
        git_before=git_before,
        git_after=git_after,
        env=env,
        results=successful,
        failures=failures,
        manifest_path=manifest_path,
        feature_table_path=feature_table_path,
        exposure_summary_path=exposure_summary_path,
        global_overview_path=global_overview,
        component_summary_path=component_summary,
        bar_examples_path=bar_examples,
        low_quality_examples_path=low_quality_examples,
        default_training_features_path=default_training_features_path,
        commands=[current_command],
    )
    print(f"Wrote manifest: {display_path(manifest_path)}")
    print(f"Wrote feature table: {display_path(feature_table_path)}")
    print(f"Wrote report: {display_path(report_path)}")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
