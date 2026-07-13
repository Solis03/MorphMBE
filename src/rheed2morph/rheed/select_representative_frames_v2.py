"""Select RHEED frames with v2 hard artifact rejection and variable-K output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.frame_quality import enhance_for_display, finite_float, normalize_for_display, resize_image
from rheed2morph.rheed.frame_quality_v2 import (
    FEATURE_KEYS_V2,
    FLAG_KEYS_V2,
    SCORE_KEYS_V2,
    active_reject_flags_v2,
    add_temporal_consistency_scores_v2,
    assign_sample_status_v2,
    critical_reject_flags_v2,
    extract_frame_quality_features_v2,
    passes_hard_reject_v2,
)
from rheed2morph.rheed.select_representative_frames import (
    DEFAULT_VIDEO_ROOT,
    VideoSource,
    discover_videos,
    display_path,
    git_status_short,
    iter_video_frames,
    resolve_path,
    str_to_bool,
    write_csv,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "rheed_frame_selection_v2_mvp"
SPECIFIC_AUDIT_IDS = ["N6041", "N6047", "N6027", "N6043"]


@dataclass
class ProcessedVideoResultV2:
    sample_id: str
    video_path: Path
    sample_dir: Path
    frame_selection_dir: Path
    scanned_frame_count: int
    sampled_frame_count: int
    valid_frame_count: int
    accepted_count: int
    rejected_count: int
    status: str
    status_reason: str
    low_confidence_flag: bool
    mean_validity_score: float
    mean_information_score: float
    mean_temporal_consistency_score: float
    accepted_grid_path: Path | None = None
    rejected_grid_path: Path | None = None
    accepted_overview_frame_path: Path | None = None
    rejected_overview_frame_path: Path | None = None
    failure: str = ""


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(image)
    if values.dtype == np.uint8:
        values = values.astype(np.float32) / 255.0
    plt.imsave(path, np.clip(values, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)


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


def _as_display_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.dtype == np.uint8:
        return values.astype(np.float32) / 255.0
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _grid_shape(count: int, max_cols: int = 4) -> tuple[int, int]:
    cols = min(max_cols, max(1, math.ceil(math.sqrt(max(count, 1)))))
    rows = max(1, math.ceil(max(count, 1) / cols))
    return rows, cols


def _flag_text(row: dict[str, Any], max_flags: int = 3) -> str:
    flags = str(row.get("flags", ""))
    if not flags:
        flags = ";".join(active_reject_flags_v2(row))
    pieces = [piece for piece in flags.split(";") if piece]
    return ";".join(pieces[:max_flags])


def candidate_filename_v2(row: dict[str, Any]) -> str:
    rank = int(row["candidate_rank"])
    frame_idx = int(row["frame_idx"])
    timestamp = finite_float(row.get("timestamp_sec", 0.0))
    score = finite_float(row.get("final_score", 0.0))
    return f"rank{rank:02d}_frame{frame_idx:06d}_t{timestamp:07.2f}s_score{score:.3f}.png"


def rejected_filename_v2(row: dict[str, Any], order: int) -> str:
    frame_idx = int(row["frame_idx"])
    reason = _flag_text(row, max_flags=1) or str(row.get("rejection_reason", "not_selected"))
    reason = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in reason)[:48]
    return f"rejected{order:02d}_frame{frame_idx:06d}_{reason}.png"


def write_manual_template_v2(
    path: Path,
    *,
    sample_id: str,
    source_video: Path,
    overwrite_manual: bool = False,
) -> bool:
    if path.exists() and not overwrite_manual:
        return False
    content = f"""# Manual RHEED frame selection v2 for sample: {sample_id}
# Source video: {display_path(source_video)}
# Review `accepted_candidate_frames_grid.png` and `rejected_bad_frames_grid.png`.
# One selected accepted frame per line. Accepted forms:
#   rank01
#   frame_idx=123
#   rank01_frame000123_t004.10s_score0.873.png
#
# Do not add rejected artifact frames unless you have manually verified the frame is real RHEED signal.
#
# selected frames below:
# rank01
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def write_sample_readme_v2(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# RHEED frame selection v2

This folder contains MVP-12 frame selection outputs. The v2 selector first applies hard artifact rejection and then ranks only valid RHEED-like frames. Accepted frame count is variable; it is not forced to 16.

## Files

- `frame_quality_scores_v2.csv`: every sampled frame with v2 features, flags, and scores.
- `accepted_candidate_frames.csv`: accepted frames in rank order.
- `rejected_frames.csv`: all non-accepted frames with rejection reasons.
- `sample_frame_status.json`: sample-level GOOD / USABLE / LOW_CONFIDENCE / EXCLUDE status.
- `accepted_candidate_frames_grid.png`: accepted frames only.
- `rejected_bad_frames_grid.png`: representative rejected frames.
- `accepted_raw_and_enhanced_grid.png`: accepted raw-normalized and enhanced views.
- `frame_quality_timeseries_v2.png`: scores and artifact markers over time.
- `candidates_accepted/`: accepted frame PNGs.
- `candidates_rejected/`: representative rejected frame PNGs.

This step does not use AFM labels, KNN retrieval, or any RHEED-to-AFM model training.
""",
        encoding="utf-8",
    )


def _accepted_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    output = dict(row)
    output["candidate_rank"] = int(rank)
    output["flags"] = ";".join(active_reject_flags_v2(output))
    output["critical_flags"] = ";".join(critical_reject_flags_v2(output))
    output["selection_note"] = "v2_valid_quality_ranked"
    return output


def select_accepted_rows_v2(
    rows: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    min_frame_gap: int,
    hard_reject: bool,
    validity_threshold: float = 0.45,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_pool = []
    for row in rows:
        hard_ok = passes_hard_reject_v2(row) if hard_reject else True
        if hard_ok and finite_float(row.get("validity_score", 0.0)) >= validity_threshold:
            valid_pool.append(dict(row))
    ordered = sorted(valid_pool, key=lambda row: finite_float(row.get("final_score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    def add_with_gap(enforce_gap: bool) -> None:
        for row in ordered:
            if len(selected) >= max_candidates:
                return
            frame_idx = int(row.get("frame_idx", -1))
            if frame_idx in selected_indices:
                continue
            if enforce_gap and any(abs(frame_idx - int(existing.get("frame_idx", -1))) < min_frame_gap for existing in selected):
                continue
            selected.append(row)
            selected_indices.add(frame_idx)

    add_with_gap(True)
    add_with_gap(False)
    accepted = [_accepted_row(row, rank) for rank, row in enumerate(selected[:max_candidates], start=1)]
    accepted_indices = {int(row["frame_idx"]) for row in accepted}
    rejected = []
    for row in rows:
        output = dict(row)
        frame_idx = int(output.get("frame_idx", -1))
        if frame_idx in accepted_indices:
            continue
        flags = active_reject_flags_v2(output)
        critical = critical_reject_flags_v2(output)
        if critical:
            reason = ";".join(critical)
        elif finite_float(output.get("validity_score", 0.0)) < validity_threshold:
            reason = "validity_score_below_threshold"
        else:
            reason = "not_selected_lower_rank"
        output["flags"] = ";".join(flags)
        output["critical_flags"] = ";".join(critical)
        output["rejection_reason"] = reason
        rejected.append(output)
    return accepted, rejected


def write_accepted_grid_v2(
    path: Path,
    *,
    sample_id: str,
    source_video: Path,
    total_scanned: int,
    status: str,
    accepted_rows: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray],
) -> Path:
    if not accepted_rows:
        return path
    rows, cols = _grid_shape(len(accepted_rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), accepted_rows):
        frame_idx = int(row["frame_idx"])
        axis.imshow(_as_display_image(frame_images[frame_idx]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            f"rank{int(row['candidate_rank']):02d} frame {frame_idx}\n"
            f"t={finite_float(row.get('timestamp_sec', 0.0)):.2f}s final={finite_float(row.get('final_score', 0.0)):.3f}\n"
            f"valid={finite_float(row.get('validity_score', 0.0)):.3f}",
            fontsize=8,
        )
    fig.suptitle(
        f"{sample_id} | scanned {total_scanned} | accepted {len(accepted_rows)} | {status} | {source_video.name}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_rejected_grid_v2(
    path: Path,
    *,
    sample_id: str,
    rejected_rows: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray],
    max_frames: int = 24,
) -> Path:
    representative = list(rejected_rows)[:max_frames]
    if not representative:
        return path
    rows, cols = _grid_shape(len(representative), max_cols=6)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.6), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), representative):
        frame_idx = int(row["frame_idx"])
        axis.imshow(_as_display_image(frame_images[frame_idx]), cmap="gray", vmin=0.0, vmax=1.0)
        flags = _flag_text(row) or str(row.get("rejection_reason", "not_selected"))
        axis.set_title(
            f"frame {frame_idx}\n"
            f"valid={finite_float(row.get('validity_score', 0.0)):.2f} final={finite_float(row.get('final_score', 0.0)):.2f}\n"
            f"{flags}",
            fontsize=7,
        )
    fig.suptitle(f"{sample_id} representative rejected frames", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_raw_enhanced_grid_v2(
    path: Path,
    *,
    sample_id: str,
    accepted_rows: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray],
) -> Path:
    if not accepted_rows:
        return path
    count = min(len(accepted_rows), 16)
    fig, axes = plt.subplots(count, 2, figsize=(7.0, max(2.6, count * 2.3)), squeeze=False)
    for row_index, row in enumerate(accepted_rows[:count]):
        frame_idx = int(row["frame_idx"])
        raw = _as_display_image(frame_images[frame_idx])
        enhanced = enhance_for_display(raw)
        for col, image, title in [(0, raw, "raw-normalized"), (1, enhanced, "enhanced/bgsub proxy")]:
            axis = axes[row_index, col]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            axis.set_title(
                f"rank{int(row['candidate_rank']):02d} frame {frame_idx} {title}\n"
                f"final={finite_float(row.get('final_score', 0.0)):.3f}",
                fontsize=8,
            )
    fig.suptitle(f"{sample_id} accepted raw and enhanced comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_timeseries_v2(
    path: Path,
    *,
    scored_rows: Sequence[dict[str, Any]],
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
) -> Path:
    if not scored_rows:
        return path
    ordered = sorted(scored_rows, key=lambda row: int(row.get("frame_idx", 0)))
    x = np.asarray([int(row["frame_idx"]) for row in ordered], dtype=float)
    fig, axis = plt.subplots(figsize=(12, 5))
    for key, label in [
        ("validity_score", "validity"),
        ("information_score", "information"),
        ("temporal_consistency_score", "temporal"),
        ("final_score", "final"),
    ]:
        axis.plot(x, [finite_float(row.get(key, 0.0)) for row in ordered], label=label, linewidth=1.2)
    accepted_x = [int(row["frame_idx"]) for row in accepted_rows]
    accepted_y = [finite_float(row.get("final_score", 0.0)) for row in accepted_rows]
    axis.scatter(accepted_x, accepted_y, marker="o", s=48, color="black", label="accepted", zorder=5)
    artifact_rows = [row for row in rejected_rows if critical_reject_flags_v2(row)]
    artifact_x = [int(row["frame_idx"]) for row in artifact_rows]
    artifact_y = [finite_float(row.get("validity_score", 0.0)) for row in artifact_rows]
    if artifact_x:
        axis.scatter(artifact_x, artifact_y, marker="x", s=38, color="#d62728", label="hard rejected", zorder=4)
    axis.set_xlabel("frame index")
    axis.set_ylabel("score")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def representative_rejected_rows(rows: Sequence[dict[str, Any]], *, limit: int = 32) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float, int]:
        flags = set(active_reject_flags_v2(row))
        artifact_priority = int(bool(flags & {"binary_artifact", "blocky_artifact", "almost_black", "almost_white"}))
        return (
            artifact_priority,
            -finite_float(row.get("validity_score", 0.0)),
            -int(row.get("frame_idx", 0)),
        )

    ordered = sorted((dict(row) for row in rows), key=key, reverse=True)
    return ordered[:limit]


def write_status_json(
    path: Path,
    *,
    sample_id: str,
    source: VideoSource,
    scored_rows: Sequence[dict[str, Any]],
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    status: str,
    status_reason: str,
    low_confidence: bool,
) -> dict[str, Any]:
    reject_counts = Counter()
    for row in rejected_rows:
        flags = active_reject_flags_v2(row)
        if not flags:
            reject_counts[str(row.get("rejection_reason", "not_selected"))] += 1
        for flag in flags:
            reject_counts[flag] += 1
    valid_scores = [finite_float(row.get("validity_score", 0.0)) for row in accepted_rows]
    info_scores = [finite_float(row.get("information_score", 0.0)) for row in accepted_rows]
    temporal_scores = [finite_float(row.get("temporal_consistency_score", 0.0)) for row in accepted_rows]
    status_doc = {
        "sample_id": sample_id,
        "source_video": display_path(source.path),
        "total_frames_scanned": len(scored_rows),
        "num_valid_frames": sum(1 for row in scored_rows if passes_hard_reject_v2(row)),
        "num_accepted_frames": len(accepted_rows),
        "num_rejected_frames": len(rejected_rows),
        "status": status,
        "status_reason": status_reason,
        "low_confidence_flag": bool(low_confidence),
        "reject_flag_counts": dict(sorted(reject_counts.items())),
        "mean_validity_score": finite_float(np.mean(valid_scores) if valid_scores else 0.0),
        "mean_information_score": finite_float(np.mean(info_scores) if info_scores else 0.0),
        "mean_temporal_consistency_score": finite_float(np.mean(temporal_scores) if temporal_scores else 0.0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(status_doc), indent=2), encoding="utf-8")
    return status_doc


def process_video_v2(source: VideoSource, args: argparse.Namespace) -> ProcessedVideoResultV2:
    frame_selection_dir = source.sample_dir / "frame_selection_v2"
    accepted_dir = frame_selection_dir / "candidates_accepted"
    rejected_dir = frame_selection_dir / "candidates_rejected"
    source.sample_dir.mkdir(parents=True, exist_ok=True)
    frame_selection_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(accepted_dir.glob("rank*_frame*.png")) + list(rejected_dir.glob("rejected*_frame*.png")):
        stale.unlink()

    scored_input: list[dict[str, Any]] = []
    frame_images: dict[int, np.ndarray] = {}
    scanned_frame_count = 0
    for sample_order, (frame_idx, timestamp_sec, gray) in enumerate(
        iter_video_frames(
            source.path,
            sample_every_n_frames=args.sample_every_n_frames,
            max_frames_per_video=args.max_frames_per_video,
        )
    ):
        scanned_frame_count += 1
        features = extract_frame_quality_features_v2(gray)
        features.update(
            {
                "sample_id": source.sample_id,
                "video_path": display_path(source.path),
                "frame_idx": int(frame_idx),
                "sample_order": int(sample_order),
                "timestamp_sec": finite_float(timestamp_sec),
            }
        )
        scored_input.append(features)
        display = resize_image(normalize_for_display(gray), size=args.display_size)
        frame_images[int(frame_idx)] = np.asarray(np.clip(display * 255.0, 0.0, 255.0), dtype=np.uint8)
    if not scored_input:
        raise RuntimeError(f"No frames were decoded from {source.path}")

    scored_rows = add_temporal_consistency_scores_v2(
        scored_input,
        frame_images,
        enabled=args.temporal_consistency,
    )
    accepted_rows, rejected_rows = select_accepted_rows_v2(
        scored_rows,
        max_candidates=args.max_candidates,
        min_frame_gap=args.min_frame_gap,
        hard_reject=args.hard_reject,
    )
    status, status_reason, low_confidence = assign_sample_status_v2(
        accepted_rows=accepted_rows,
        min_accepted_for_good=args.min_accepted_for_good,
        min_accepted_for_usable=args.min_accepted_for_usable,
        min_accepted_for_low_confidence=args.min_accepted_for_low_confidence,
    )
    for row in accepted_rows:
        frame_idx = int(row["frame_idx"])
        png_path = accepted_dir / candidate_filename_v2(row)
        save_image(png_path, frame_images[frame_idx])
        row["candidate_png_path"] = display_path(png_path)
    rep_rejected = representative_rejected_rows(rejected_rows, limit=max(24, args.max_candidates))
    for order, row in enumerate(rep_rejected, start=1):
        frame_idx = int(row["frame_idx"])
        png_path = rejected_dir / rejected_filename_v2(row, order)
        save_image(png_path, frame_images[frame_idx])
        row["rejected_png_path"] = display_path(png_path)
        for rejected in rejected_rows:
            if int(rejected["frame_idx"]) == frame_idx:
                rejected["rejected_png_path"] = display_path(png_path)
                rejected["representative_rejected"] = 1

    (frame_selection_dir / "source_video.txt").write_text(f"{display_path(source.path)}\n", encoding="utf-8")
    quality_fields = [
        "sample_id",
        "video_path",
        "frame_idx",
        "sample_order",
        "timestamp_sec",
        *FEATURE_KEYS_V2,
        *SCORE_KEYS_V2,
        *FLAG_KEYS_V2,
    ]
    write_csv(frame_selection_dir / "frame_quality_scores_v2.csv", scored_rows, quality_fields)
    accepted_fields = [
        "sample_id",
        "video_path",
        "candidate_rank",
        "frame_idx",
        "sample_order",
        "timestamp_sec",
        *SCORE_KEYS_V2,
        "pattern_visibility_score",
        "plausible_spot_streak_score",
        "projection_peak_score",
        "local_contrast_after_bgsub",
        "artifact_penalty",
        "flags",
        "critical_flags",
        "selection_note",
        "candidate_png_path",
    ]
    write_csv(frame_selection_dir / "accepted_candidate_frames.csv", accepted_rows, accepted_fields)
    rejected_fields = [
        "sample_id",
        "video_path",
        "frame_idx",
        "sample_order",
        "timestamp_sec",
        *SCORE_KEYS_V2,
        "artifact_penalty",
        "flags",
        "critical_flags",
        "rejection_reason",
        "representative_rejected",
        "rejected_png_path",
    ]
    write_csv(frame_selection_dir / "rejected_frames.csv", rejected_rows, rejected_fields)

    status_doc = write_status_json(
        frame_selection_dir / "sample_frame_status.json",
        sample_id=source.sample_id,
        source=source,
        scored_rows=scored_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        status=status,
        status_reason=status_reason,
        low_confidence=low_confidence,
    )
    accepted_grid = write_accepted_grid_v2(
        frame_selection_dir / "accepted_candidate_frames_grid.png",
        sample_id=source.sample_id,
        source_video=source.path,
        total_scanned=len(scored_rows),
        status=status,
        accepted_rows=accepted_rows,
        frame_images=frame_images,
    )
    rejected_grid = write_rejected_grid_v2(
        frame_selection_dir / "rejected_bad_frames_grid.png",
        sample_id=source.sample_id,
        rejected_rows=rep_rejected,
        frame_images=frame_images,
    )
    write_raw_enhanced_grid_v2(
        frame_selection_dir / "accepted_raw_and_enhanced_grid.png",
        sample_id=source.sample_id,
        accepted_rows=accepted_rows,
        frame_images=frame_images,
    )
    write_timeseries_v2(
        frame_selection_dir / "frame_quality_timeseries_v2.png",
        scored_rows=scored_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
    )
    write_manual_template_v2(
        frame_selection_dir / "manual_selected_frames.txt",
        sample_id=source.sample_id,
        source_video=source.path,
        overwrite_manual=args.overwrite_manual,
    )
    write_sample_readme_v2(frame_selection_dir / "README_frame_selection_v2.md")

    overview = Path(accepted_rows[0]["candidate_png_path"]) if accepted_rows else None
    if overview is not None and not overview.is_absolute():
        overview = REPO_ROOT / overview
    rejected_overview = None
    if rep_rejected and rep_rejected[0].get("rejected_png_path"):
        rejected_overview = Path(str(rep_rejected[0]["rejected_png_path"]))
        if not rejected_overview.is_absolute():
            rejected_overview = REPO_ROOT / rejected_overview

    return ProcessedVideoResultV2(
        sample_id=source.sample_id,
        video_path=source.path,
        sample_dir=source.sample_dir,
        frame_selection_dir=frame_selection_dir,
        scanned_frame_count=scanned_frame_count,
        sampled_frame_count=len(scored_rows),
        valid_frame_count=int(status_doc["num_valid_frames"]),
        accepted_count=len(accepted_rows),
        rejected_count=len(rejected_rows),
        status=status,
        status_reason=status_reason,
        low_confidence_flag=low_confidence,
        mean_validity_score=finite_float(status_doc["mean_validity_score"]),
        mean_information_score=finite_float(status_doc["mean_information_score"]),
        mean_temporal_consistency_score=finite_float(status_doc["mean_temporal_consistency_score"]),
        accepted_grid_path=accepted_grid,
        rejected_grid_path=rejected_grid,
        accepted_overview_frame_path=overview,
        rejected_overview_frame_path=rejected_overview,
    )


def result_summary_row(result: ProcessedVideoResultV2) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "video_path": display_path(result.video_path),
        "sample_dir": display_path(result.sample_dir),
        "frame_selection_dir": display_path(result.frame_selection_dir),
        "scanned_frame_count": result.scanned_frame_count,
        "sampled_frame_count": result.sampled_frame_count,
        "valid_frame_count": result.valid_frame_count,
        "accepted_count": result.accepted_count,
        "rejected_count": result.rejected_count,
        "status": result.status,
        "status_reason": result.status_reason,
        "low_confidence_flag": int(result.low_confidence_flag),
        "mean_validity_score": result.mean_validity_score,
        "mean_information_score": result.mean_information_score,
        "mean_temporal_consistency_score": result.mean_temporal_consistency_score,
        "accepted_grid_path": display_path(result.accepted_grid_path) if result.accepted_grid_path else "",
        "rejected_grid_path": display_path(result.rejected_grid_path) if result.rejected_grid_path else "",
        "failure": result.failure,
    }


SUMMARY_FIELDS_V2 = [
    "sample_id",
    "video_path",
    "sample_dir",
    "frame_selection_dir",
    "scanned_frame_count",
    "sampled_frame_count",
    "valid_frame_count",
    "accepted_count",
    "rejected_count",
    "status",
    "status_reason",
    "low_confidence_flag",
    "mean_validity_score",
    "mean_information_score",
    "mean_temporal_consistency_score",
    "accepted_grid_path",
    "rejected_grid_path",
    "failure",
]


def _load_preview(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        import imageio.v3 as iio

        return iio.imread(path)
    except Exception:
        return None


def write_global_image_grid(
    path: Path,
    results: Sequence[ProcessedVideoResultV2],
    *,
    title: str,
    image_attr: str,
    status_filter: set[str] | None = None,
    limit: int = 40,
) -> Path:
    selected = []
    for result in results:
        if result.failure:
            continue
        if status_filter is not None and result.status not in status_filter:
            continue
        image = _load_preview(getattr(result, image_attr))
        if image is not None:
            selected.append((result, image))
    selected = selected[:limit]
    if not selected:
        return path
    cols = min(8, max(1, math.ceil(math.sqrt(len(selected)))))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (result, image) in zip(axes.ravel(), selected):
        axis.imshow(image, cmap="gray")
        axis.set_title(f"{result.sample_id}\n{result.status} K={result.accepted_count}", fontsize=7)
        axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def collect_environment_v2() -> dict[str, str]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for package in ("scipy", "skimage", "cv2", "torch"):
        try:
            module = __import__(package)
            env[package] = getattr(module, "__version__", "available")
        except ModuleNotFoundError:
            env[package] = "not available"
    return env


def global_reject_counts(results: Sequence[ProcessedVideoResultV2]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        status_path = result.frame_selection_dir / "sample_frame_status.json"
        if not status_path.is_file():
            continue
        try:
            doc = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        counts.update(doc.get("reject_flag_counts", {}))
    return counts


def write_global_outputs(
    report_dir: Path,
    *,
    inventory_rows: Sequence[dict[str, Any]],
    results: Sequence[ProcessedVideoResultV2],
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    inventory_path = report_dir / "video_inventory_v2.csv"
    write_csv(
        inventory_path,
        inventory_rows,
        ["video_path", "sample_id", "sample_dir", "matched_video_glob", "selected_for_processing", "discovery_reason"],
    )
    summary_path = report_dir / "frame_selection_summary_v2.csv"
    write_csv(summary_path, [result_summary_row(result) for result in results], SUMMARY_FIELDS_V2)
    failed_path = report_dir / "failed_videos_v2.csv"
    write_csv(
        failed_path,
        [
            {"sample_id": result.sample_id, "video_path": display_path(result.video_path), "failure": result.failure}
            for result in results
            if result.failure
        ],
        ["sample_id", "video_path", "failure"],
    )
    status_counts = Counter(result.status for result in results if not result.failure)
    successful = [result for result in results if not result.failure]
    accepted_counts = [result.accepted_count for result in successful]
    reject_counts = global_reject_counts(successful)
    status_doc = {
        "status_counts": dict(status_counts),
        "processed_successfully": len(successful),
        "failures": sum(1 for result in results if result.failure),
        "average_accepted_frames_per_sample": finite_float(np.mean(accepted_counts) if accepted_counts else 0.0),
        "min_accepted_frames": int(min(accepted_counts)) if accepted_counts else 0,
        "max_accepted_frames": int(max(accepted_counts)) if accepted_counts else 0,
        "common_rejection_reasons": dict(reject_counts.most_common(20)),
        "binary_or_block_artifacts_rejected": int(reject_counts.get("binary_artifact", 0) + reject_counts.get("blocky_artifact", 0)),
    }
    status_path = report_dir / "global_status_summary.json"
    status_path.write_text(json.dumps(json_ready(status_doc), indent=2), encoding="utf-8")
    accepted_grid = write_global_image_grid(
        report_dir / "global_accepted_overview_grid.png",
        successful,
        title="Accepted v2 rank-1 frames",
        image_attr="accepted_overview_frame_path",
    )
    rejected_grid = write_global_image_grid(
        report_dir / "global_rejected_artifact_examples.png",
        successful,
        title="Representative rejected artifact examples",
        image_attr="rejected_overview_frame_path",
    )
    low_grid = write_global_image_grid(
        report_dir / "global_low_confidence_samples.png",
        successful,
        title="LOW_CONFIDENCE and EXCLUDE samples",
        image_attr="accepted_overview_frame_path",
        status_filter={"LOW_CONFIDENCE", "EXCLUDE"},
    )
    return inventory_path, summary_path, failed_path, status_path, accepted_grid, rejected_grid, low_grid


def _sample_matches(sample_id: str, requested: str) -> bool:
    return sample_id == requested or sample_id.startswith(requested) or requested in sample_id


def write_specific_sample_audit(report_dir: Path, results: Sequence[ProcessedVideoResultV2]) -> Path:
    path = report_dir / "specific_sample_audit.md"
    lines = ["# Specific Sample Audit", ""]
    for requested in SPECIFIC_AUDIT_IDS:
        matches = [result for result in results if _sample_matches(result.sample_id, requested)]
        if not matches:
            lines.extend([f"## {requested}", "", "Not present in processed inventory.", ""])
            continue
        for result in matches:
            lines.extend(
                [
                    f"## {result.sample_id}",
                    "",
                    f"- Status: `{result.status}`",
                    f"- Accepted frames: {result.accepted_count}",
                    f"- Rejected frames: {result.rejected_count}",
                    f"- Mean validity: {result.mean_validity_score:.4g}",
                    f"- Mean information: {result.mean_information_score:.4g}",
                    f"- Accepted grid: `{display_path(result.accepted_grid_path) if result.accepted_grid_path else ''}`",
                    f"- Rejected grid: `{display_path(result.rejected_grid_path) if result.rejected_grid_path else ''}`",
                ]
            )
            status_path = result.frame_selection_dir / "sample_frame_status.json"
            if status_path.is_file():
                doc = json.loads(status_path.read_text(encoding="utf-8"))
                counts = doc.get("reject_flag_counts", {})
                important = {key: counts.get(key, 0) for key in ["binary_artifact", "blocky_artifact", "almost_black", "almost_white", "no_plausible_rheed_pattern"]}
                lines.append(f"- Key reject counts: `{important}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_codex_report_v2(
    report_dir: Path,
    *,
    args: argparse.Namespace,
    git_before: str,
    git_after: str,
    env: dict[str, str],
    inventory_rows: Sequence[dict[str, Any]],
    results: Sequence[ProcessedVideoResultV2],
    paths: tuple[Path, Path, Path, Path, Path, Path, Path],
    command: str,
    used_fallback: bool,
    specific_audit_path: Path,
) -> Path:
    inventory_path, summary_path, failed_path, status_path, accepted_grid, rejected_grid, low_grid = paths
    successful = [result for result in results if not result.failure]
    failures = [result for result in results if result.failure]
    status_counts = Counter(result.status for result in successful)
    accepted_counts = [result.accepted_count for result in successful]
    reject_counts = global_reject_counts(successful)
    files_changed = [
        "src/rheed2morph/rheed/frame_quality_v2.py",
        "src/rheed2morph/rheed/select_representative_frames_v2.py",
        "src/rheed2morph/rheed/build_shape_bag_inputs_v2.py",
        "src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py",
        "tests/test_rheed_frame_selection_v2.py",
    ]
    report = report_dir / "codex_report.md"
    text = f"""# MVP-12 RHEED frame selection v2 report

Generated: {datetime.now(UTC).isoformat(timespec="seconds")}

MVP-12 improves RHEED frame selection and variable-K shape-bag inputs. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status Before

```text
{git_before}
```

## Git Status After Selection

```text
{git_after}
```

## Files Created Or Modified

{chr(10).join(f"- `{path}`" for path in files_changed)}
- Per-sample `frame_selection_v2/` folders under `{display_path(resolve_path(args.out_root))}`
- `{display_path(inventory_path)}`
- `{display_path(summary_path)}`
- `{display_path(failed_path)}`
- `{display_path(status_path)}`
- `{display_path(accepted_grid)}`
- `{display_path(rejected_grid)}`
- `{display_path(low_grid)}`
- `{display_path(specific_audit_path)}`

## Exact Commands Run

```bash
{command}
```

Additional test and shape-bag commands are appended after the shape-bag v2 run.

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

- Video root: `{display_path(resolve_path(args.video_root))}`
- MP4 files found: {len(inventory_rows)}
- Number processed: {len(successful)}
- Failures: {len(failures)}
- Used all-MP4 fallback: {used_fallback}

## Difference From V1

- No forced 16 accepted frames; accepted bags are variable-K up to `{args.max_candidates}`.
- Hard artifact rejection separates validity from information quality.
- Temporal consistency penalizes isolated quality spikes when enabled.
- Accepted and rejected frames are written to separate CSVs, grids, and folders.
- Shape-bag v2 uses accepted rows and preserves valid frames with `frame_mask`.

## Frame Selection Summary

- GOOD: {status_counts.get("GOOD", 0)}
- USABLE: {status_counts.get("USABLE", 0)}
- LOW_CONFIDENCE: {status_counts.get("LOW_CONFIDENCE", 0)}
- EXCLUDE: {status_counts.get("EXCLUDE", 0)}
- Average accepted frames per sample: {finite_float(np.mean(accepted_counts) if accepted_counts else 0.0):.4g}
- Min/max accepted frames: {(min(accepted_counts) if accepted_counts else 0)} / {(max(accepted_counts) if accepted_counts else 0)}
- Common rejection reasons: `{dict(reject_counts.most_common(10))}`
- Binary/block artifacts rejected: {int(reject_counts.get("binary_artifact", 0) + reject_counts.get("blocky_artifact", 0))}

## Specific Sample Audit

See `{display_path(specific_audit_path)}` for N6041, N6047, N6027, and N6043 if present.

## Manual Review Workflow

Review LOW_CONFIDENCE and EXCLUDE samples first, then inspect accepted/rejected grids for samples with many binary or block artifacts.

- Accepted grids: `<sample>/frame_selection_v2/accepted_candidate_frames_grid.png`
- Rejected grids: `<sample>/frame_selection_v2/rejected_bad_frames_grid.png`
- Summary table: `{display_path(summary_path)}`

## Known Limitations

- The detector is a transparent heuristic gate, not a trained RHEED physics model.
- Some unusual but real patterns may need manual review if they resemble saturated or block-like artifacts.
- Temporal consistency uses local image similarity; abrupt real changes can be down-weighted.

## Recommended Next Command For Future Model Run

After manual review, run the future supervised experiment against `rheed_shape_input_v2/shape_bag_v2.npz` files listed in `rheed_shape_bag_manifest_v2.csv`.
"""
    report.write_text(text, encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--video-glob", default="*raw_crop*.mp4")
    parser.add_argument("--include-all-mp4", type=str_to_bool, default=False)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--sample-every-n-frames", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=1200)
    parser.add_argument("--display-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--min-accepted-for-good", type=int, default=8)
    parser.add_argument("--min-accepted-for-usable", type=int, default=3)
    parser.add_argument("--min-accepted-for-low-confidence", type=int, default=1)
    parser.add_argument("--hard-reject", type=str_to_bool, default=True)
    parser.add_argument("--temporal-consistency", type=str_to_bool, default=True)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--overwrite-manual", type=str_to_bool, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-video-limit", type=int, default=4)
    parser.add_argument("--strict", type=str_to_bool, default=False)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    video_root = resolve_path(args.video_root)
    out_root = resolve_path(args.out_root)
    report_root = resolve_path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_dir = report_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    git_before = git_status_short()
    env = collect_environment_v2()
    command_args = sys.argv[1:] if argv is None else list(argv)
    command = "PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames_v2 " + shlex.join(command_args)

    sources, inventory_rows, used_fallback = discover_videos(
        video_root,
        out_root,
        video_glob=args.video_glob,
        include_all_mp4=args.include_all_mp4,
    )
    if args.debug:
        sources = sources[: max(1, args.debug_video_limit)]
        print(f"[debug] Processing first {len(sources)} videos.")
    print(f"Discovered {len(inventory_rows)} MP4 files; processing {len(sources)} videos.")
    if used_fallback:
        print(f"No files matched {args.video_glob!r}; falling back to all MP4 files.")

    results: list[ProcessedVideoResultV2] = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.sample_id}: {display_path(source.path)}")
        try:
            result = process_video_v2(source, args)
            print(f"  status={result.status} accepted={result.accepted_count} rejected={result.rejected_count}")
        except Exception as exc:
            if args.strict:
                raise
            result = ProcessedVideoResultV2(
                sample_id=source.sample_id,
                video_path=source.path,
                sample_dir=source.sample_dir,
                frame_selection_dir=source.sample_dir / "frame_selection_v2",
                scanned_frame_count=0,
                sampled_frame_count=0,
                valid_frame_count=0,
                accepted_count=0,
                rejected_count=0,
                status="FAILED",
                status_reason=f"{type(exc).__name__}: {exc}",
                low_confidence_flag=True,
                mean_validity_score=0.0,
                mean_information_score=0.0,
                mean_temporal_consistency_score=0.0,
                failure=f"{type(exc).__name__}: {exc}",
            )
            print(f"  failed: {result.failure}")
        results.append(result)

    paths = write_global_outputs(report_dir, inventory_rows=inventory_rows, results=results)
    specific_audit_path = write_specific_sample_audit(report_dir, results)
    git_after = git_status_short()
    report_path = write_codex_report_v2(
        report_dir,
        args=args,
        git_before=git_before,
        git_after=git_after,
        env=env,
        inventory_rows=inventory_rows,
        results=results,
        paths=paths,
        command=command,
        used_fallback=used_fallback,
        specific_audit_path=specific_audit_path,
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Git Status After Report Write\n\n```text\n")
        handle.write(git_status_short())
        handle.write("\n```\n")
    print(f"Wrote report: {display_path(report_path)}")
    failures = sum(1 for result in results if result.failure)
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
