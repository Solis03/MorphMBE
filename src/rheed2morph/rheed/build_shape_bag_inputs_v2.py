"""Build variable-K RHEED shape-bag inputs from frame_selection_v2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shlex
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.build_shape_bag_inputs import (
    CONSENSUS_MAP_NAMES,
    FrameRecord,
    aggregate_sample_features,
    build_consensus_maps,
    compute_frame_weights,
    display_path,
    git_status_short,
    json_ready,
    parse_manual_rows,
    read_csv,
    resolve_path,
    run_exposure_audit,
    save_frame_debug_images,
    str_to_bool,
    write_component_grid,
    write_consensus_grid,
    write_csv,
    write_frame_weight_plot,
    write_overview_grid,
)
from rheed2morph.rheed.frame_quality import finite_float
from rheed2morph.rheed.frame_quality_v2 import status_rank_v2
from rheed2morph.rheed.shape_preprocessing import (
    DEFAULT_CHANNEL_NAMES,
    channels_to_tensor,
    preprocess_frame_for_shape,
    read_grayscale_image,
)
from rheed2morph.rheed.spot_streak_geometry import (
    FRAME_SHAPE_FEATURE_NAMES,
    component_rows_for_csv,
    extract_components_and_frame_features,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data" / "rheed_roi_shadow_right_v2_main_raw_crop_videos_256"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "rheed_frame_selection_v2_mvp"


@dataclass
class SampleResultV2:
    sample_id: str
    source_type: str
    sample_status: str
    sample_quality: float
    sample_folder: Path
    frame_selection_folder: Path
    shape_input_folder: Path
    num_frames_available: int
    num_frames_used: int
    rejected_frame_count: int
    accepted_csv: Path
    manual_selection_file: Path
    shape_bag_npz: Path
    sample_feature_json: Path
    sample_feature_csv: Path
    preview_grid: Path
    exposure_audit_json: Path
    failure: str = ""


def candidate_png_path_v2(row: dict[str, str]) -> Path:
    raw = row.get("candidate_png_path", "")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def discover_status_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("frame_selection_v2/sample_frame_status.json"))


def latest_report_dir(report_root: Path) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        path
        for path in sorted(report_root.iterdir())
        if path.is_dir() and (path / "frame_selection_summary_v2.csv").is_file()
    ]
    if candidates:
        return candidates[-1]
    path = report_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_allowed(status: str, args: argparse.Namespace) -> bool:
    if status == "EXCLUDE" and not args.include_exclude:
        return False
    if status == "LOW_CONFIDENCE" and not args.include_low_confidence:
        return False
    return status_rank_v2(status) >= status_rank_v2(args.min_status)


def select_frame_rows_v2(
    *,
    accepted_csv: Path,
    mode: str,
    max_frames_per_sample: int,
) -> tuple[list[dict[str, str]], str, int]:
    accepted_rows = read_csv(accepted_csv)
    accepted_rows = sorted(accepted_rows, key=lambda row: int(float(row.get("candidate_rank", "999") or 999)))
    manual_file = accepted_csv.parent / "manual_selected_frames.txt"
    manual_rows = parse_manual_rows(manual_file, accepted_rows)
    if mode in {"manual_or_v2_accepted", "manual_only"} and manual_rows:
        return manual_rows[:max_frames_per_sample], "manual", len(manual_rows)
    if mode == "manual_only":
        return [], "manual", len(manual_rows)
    return accepted_rows[:max_frames_per_sample], "accepted_v2_fallback", len(accepted_rows)


def clear_debug_images(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for path in frames_dir.glob("frame*_*.png"):
        path.unlink()


def write_shape_npz_v2(
    path: Path,
    *,
    frames: Sequence[FrameRecord],
    image_size: int,
    max_frames: int,
    pad_to_max: bool,
    sample_feature_vector: np.ndarray,
    sample_feature_names: list[str],
    sample_quality: float,
    sample_status: str,
    rejected_frame_count: int,
    source_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_count = len(DEFAULT_CHANNEL_NAMES)
    output_frames = max_frames if pad_to_max else min(max_frames, len(frames))
    output_frames = max(1, output_frames)
    frames_array = np.zeros((output_frames, channel_count, image_size, image_size), dtype=np.float32)
    frame_mask = np.zeros(output_frames, dtype=np.float32)
    frame_weights = np.zeros(output_frames, dtype=np.float32)
    frame_indices = np.full(output_frames, -1, dtype=np.int32)
    timestamps = np.full(output_frames, np.nan, dtype=np.float32)
    used = list(frames[:output_frames])
    for index, frame in enumerate(used):
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
        num_valid_frames=np.asarray(len(used), dtype=np.int32),
        sample_quality=np.asarray(float(sample_quality), dtype=np.float32),
        sample_status=np.asarray(sample_status, dtype="U"),
        rejected_frame_count=np.asarray(int(rejected_frame_count), dtype=np.int32),
        source_type=np.asarray(source_type, dtype="U"),
    )
    return frames_array, frame_mask, frame_weights, consensus_maps, frame_indices, timestamps


def write_consensus_maps_v2(path: Path, consensus_maps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, maps=consensus_maps, names=np.asarray(CONSENSUS_MAP_NAMES, dtype="U"))


def write_feature_summary_v2(path: Path, sample_feature_summary: dict[str, float]) -> None:
    keys = [
        "weighted_mean_round_spot_count",
        "weighted_mean_elongated_spot_count",
        "weighted_mean_horizontal_bar_count",
        "weighted_mean_vertical_streak_count",
        "weighted_mean_bar_like_score",
        "weighted_mean_mask_confidence",
    ]
    lines = ["# RHEED Shape Feature Summary v2", "", "| feature | value |", "| --- | ---: |"]
    for key in keys:
        lines.append(f"| `{key}` | {sample_feature_summary.get(key, 0.0):.6g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_sample_v2(status_json: Path, args: argparse.Namespace) -> tuple[SampleResultV2, dict[str, Any] | None]:
    frame_selection_dir = status_json.parent
    sample_folder = frame_selection_dir.parent
    sample_id = sample_folder.name
    status_doc = json.loads(status_json.read_text(encoding="utf-8"))
    sample_status = str(status_doc.get("status", "EXCLUDE"))
    accepted_csv = frame_selection_dir / "accepted_candidate_frames.csv"
    rejected_csv = frame_selection_dir / "rejected_frames.csv"
    rejected_count = len(read_csv(rejected_csv)) if rejected_csv.is_file() else int(status_doc.get("num_rejected_frames", 0))
    accepted_rows_all = read_csv(accepted_csv)
    sample_quality = finite_float(
        np.mean([finite_float(row.get("final_score", 0.0)) for row in accepted_rows_all]) if accepted_rows_all else 0.0
    )
    if not status_allowed(sample_status, args):
        raise RuntimeError(f"sample status {sample_status} excluded by current filters")
    if not accepted_csv.is_file():
        raise RuntimeError(f"missing accepted candidate CSV: {accepted_csv}")

    shape_dir = sample_folder / "rheed_shape_input_v2"
    frames_dir = shape_dir / "frames"
    shape_dir.mkdir(parents=True, exist_ok=True)
    if args.write_debug_images:
        clear_debug_images(frames_dir)

    selected_rows, source_type, available_count = select_frame_rows_v2(
        accepted_csv=accepted_csv,
        mode=args.mode,
        max_frames_per_sample=args.max_frames_per_sample,
    )
    if not selected_rows:
        raise RuntimeError(f"no usable {source_type} rows for {sample_id}")

    frame_records: list[FrameRecord] = []
    component_rows: list[dict[str, Any]] = []
    frame_feature_rows: list[dict[str, Any]] = []
    missing_frames: list[str] = []
    for row in selected_rows:
        row = dict(row)
        row.setdefault("quality_score", row.get("final_score", "1.0"))
        row.setdefault("low_confidence_candidate", "0")
        png_path = candidate_png_path_v2(row)
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

    if not frame_records:
        raise RuntimeError(f"all selected frame PNGs were missing for {sample_id}: {missing_frames[:3]}")

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
            "final_score": finite_float(record.candidate_row.get("final_score", 0.0)),
            "validity_score": finite_float(record.candidate_row.get("validity_score", 0.0)),
            "information_score": finite_float(record.candidate_row.get("information_score", 0.0)),
            "frame_weight": record.frame_weight,
            "raw_frame_weight": record.raw_weight,
        }
        row.update(record.audit)
        row.update(record.features)
        frame_feature_rows.append(row)

    sample_vector, sample_feature_names, sample_summary = aggregate_sample_features(frame_records)
    shape_bag_npz = shape_dir / "shape_bag_v2.npz"
    frames_array, frame_mask, frame_weights, consensus_maps, _frame_indices, _timestamps = write_shape_npz_v2(
        shape_bag_npz,
        frames=frame_records,
        image_size=args.image_size,
        max_frames=args.max_frames_per_sample,
        pad_to_max=args.pad_to_max,
        sample_feature_vector=sample_vector,
        sample_feature_names=sample_feature_names,
        sample_quality=sample_quality,
        sample_status=sample_status,
        rejected_frame_count=rejected_count,
        source_type=source_type,
    )
    _ = frames_array, frame_mask, frame_weights
    write_consensus_maps_v2(shape_dir / "consensus_maps_v2.npz", consensus_maps)

    write_csv(
        shape_dir / "frame_geometry_components_v2.csv",
        component_rows,
        [
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
        ],
    )
    write_csv(
        shape_dir / "frame_shape_features_v2.csv",
        frame_feature_rows,
        [
            "sample_id",
            "frame_idx",
            "candidate_rank",
            "timestamp_sec",
            "candidate_png_path",
            "final_score",
            "validity_score",
            "information_score",
            "frame_weight",
            "raw_frame_weight",
            "raw_mean",
            "raw_std",
            *FRAME_SHAPE_FEATURE_NAMES,
        ],
    )

    sample_csv = shape_dir / "sample_shape_features_v2.csv"
    sample_json = shape_dir / "sample_shape_features_v2.json"
    sample_row = {
        "sample_id": sample_id,
        "sample_status": sample_status,
        "sample_quality": sample_quality,
        "num_valid_frames": len(frame_records),
        "rejected_frame_count": rejected_count,
        "source_type": source_type,
        **sample_summary,
    }
    write_csv(sample_csv, [sample_row], list(sample_row.keys()))
    sample_json.write_text(
        json.dumps(
            json_ready(
                {
                    "sample_id": sample_id,
                    "source_type": source_type,
                    "sample_status": sample_status,
                    "sample_quality": sample_quality,
                    "num_valid_frames": len(frame_records),
                    "rejected_frame_count": rejected_count,
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

    overview = shape_dir / "shape_input_overview_v2.png"
    consensus_png = shape_dir / "consensus_shape_maps_v2.png"
    component_png = shape_dir / "component_geometry_grid_v2.png"
    weight_png = shape_dir / "frame_weight_diagnostics_v2.png"
    write_overview_grid(overview, frame_records)
    write_consensus_grid(consensus_png, consensus_maps)
    write_component_grid(component_png, frame_records)
    write_frame_weight_plot(weight_png, frame_records)
    write_feature_summary_v2(shape_dir / "feature_summary_v2.md", sample_summary)

    audit: dict[str, Any] | None = None
    audit_path = shape_dir / "exposure_invariance_audit_v2.json"
    if args.exposure_audit:
        audit, raw_audit_path = run_exposure_audit(shape_dir, frame_records, args.image_size)
        if raw_audit_path.is_file():
            audit_path.write_text(raw_audit_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        audit = {"status": "skipped"}
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    return (
        SampleResultV2(
            sample_id=sample_id,
            source_type=source_type,
            sample_status=sample_status,
            sample_quality=sample_quality,
            sample_folder=sample_folder,
            frame_selection_folder=frame_selection_dir,
            shape_input_folder=shape_dir,
            num_frames_available=available_count,
            num_frames_used=len(frame_records),
            rejected_frame_count=rejected_count,
            accepted_csv=accepted_csv,
            manual_selection_file=frame_selection_dir / "manual_selected_frames.txt",
            shape_bag_npz=shape_bag_npz,
            sample_feature_json=sample_json,
            sample_feature_csv=sample_csv,
            preview_grid=overview,
            exposure_audit_json=audit_path,
        ),
        audit,
    )


def manifest_row_v2(result: SampleResultV2) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "source_type": result.source_type,
        "sample_status": result.sample_status,
        "sample_quality": result.sample_quality,
        "sample_folder": display_path(result.sample_folder),
        "frame_selection_folder": display_path(result.frame_selection_folder),
        "shape_input_folder": display_path(result.shape_input_folder),
        "num_frames_available": result.num_frames_available,
        "num_frames_used": result.num_frames_used,
        "rejected_frame_count": result.rejected_frame_count,
        "accepted_csv": display_path(result.accepted_csv),
        "manual_selection_file": display_path(result.manual_selection_file),
        "shape_bag_npz": display_path(result.shape_bag_npz),
        "sample_feature_json": display_path(result.sample_feature_json),
        "sample_feature_csv": display_path(result.sample_feature_csv),
        "preview_grid": display_path(result.preview_grid),
        "exposure_audit_json": display_path(result.exposure_audit_json),
    }


MANIFEST_FIELDS_V2 = [
    "sample_id",
    "source_type",
    "sample_status",
    "sample_quality",
    "sample_folder",
    "frame_selection_folder",
    "shape_input_folder",
    "num_frames_available",
    "num_frames_used",
    "rejected_frame_count",
    "accepted_csv",
    "manual_selection_file",
    "shape_bag_npz",
    "sample_feature_json",
    "sample_feature_csv",
    "preview_grid",
    "exposure_audit_json",
]


def write_global_feature_table_v2(report_dir: Path, results: Sequence[SampleResultV2]) -> Path:
    rows = []
    fieldnames: list[str] = []
    for result in results:
        if result.failure or not result.sample_feature_csv.is_file():
            continue
        sample_rows = read_csv(result.sample_feature_csv)
        if not sample_rows:
            continue
        row = sample_rows[0]
        rows.append(row)
        if not fieldnames:
            fieldnames = list(row.keys())
    path = report_dir / "global_sample_shape_features_v2.csv"
    write_csv(path, rows, fieldnames or ["sample_id"])
    return path


def write_default_feature_names_v2(report_dir: Path, results: Sequence[SampleResultV2]) -> Path:
    path = report_dir / "default_training_feature_names_v2.txt"
    names: list[str] = []
    for result in results:
        if not result.shape_bag_npz.is_file():
            continue
        with np.load(result.shape_bag_npz, allow_pickle=False) as data:
            names = [str(item) for item in data["sample_feature_names"].tolist()]
        if names:
            break
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    return path


def collect_environment_v2() -> dict[str, str]:
    env = {"python": sys.version.replace("\n", " "), "platform": platform.platform(), "numpy": np.__version__}
    for package in ("scipy", "skimage", "cv2", "torch"):
        try:
            module = __import__(package)
            env[package] = getattr(module, "__version__", "available")
        except ModuleNotFoundError:
            env[package] = "not available"
    return env


def append_shape_report(
    report_dir: Path,
    *,
    args: argparse.Namespace,
    git_before: str,
    git_after: str,
    env: dict[str, str],
    results: Sequence[SampleResultV2],
    failures: Sequence[SampleResultV2],
    manifest_path: Path,
    feature_table_path: Path,
    default_features_path: Path,
    command: str,
) -> Path:
    report = report_dir / "codex_report.md"
    successful = [result for result in results if not result.failure]
    manual_count = sum(1 for result in successful if result.source_type == "manual")
    fallback_count = sum(1 for result in successful if result.source_type == "accepted_v2_fallback")
    status_counts: dict[str, int] = {}
    for result in successful:
        status_counts[result.sample_status] = status_counts.get(result.sample_status, 0) + 1
    shapes = []
    for result in successful[:5]:
        with np.load(result.shape_bag_npz, allow_pickle=False) as data:
            shapes.append(
                f"{result.sample_id}: frames={tuple(data['frames'].shape)} mask={tuple(data['frame_mask'].shape)} "
                f"valid={int(data['num_valid_frames'])} status={str(data['sample_status'])}"
            )
    text = f"""

## Shape-Bag V2 Summary

Generated: {datetime.now(UTC).isoformat(timespec="seconds")}

### Shape Command

```bash
{command}
```

### Shape Environment

| package | version |
| --- | --- |
| python | {env.get("python", "")} |
| platform | {env.get("platform", "")} |
| numpy | {env.get("numpy", "")} |
| scipy | {env.get("scipy", "")} |
| skimage | {env.get("skimage", "")} |
| cv2 | {env.get("cv2", "")} |
| torch | {env.get("torch", "")} |

### Shape-Bag Outputs

- Manifest: `{display_path(manifest_path)}`
- Global feature table: `{display_path(feature_table_path)}`
- Default training feature names: `{display_path(default_features_path)}`
- Shape bags written: {len(successful)}
- Samples included by status: `{status_counts}`
- Samples excluded or failed: {len(failures)}
- Manual selections used: {manual_count}
- Accepted-v2 fallback used: {fallback_count}
- `pad_to_max`: {args.pad_to_max}
- Example tensor shapes: `{'; '.join(shapes) if shapes else 'n/a'}`

`shape_bag_v2.npz` preserves variable-K through `frame_mask`; padded all-zero frames have mask value 0 and zero frame weight.

## Git Status After Shape-Bag V2

```text
{git_after}
```
"""
    if report.is_file():
        with report.open("a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        report.write_text("# MVP-12 RHEED frame selection v2 report\n" + text, encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--frame-selection-version", choices=["v2"], default="v2")
    parser.add_argument("--max-frames-per-sample", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["manual_or_v2_accepted", "v2_accepted_only", "manual_only"], default="manual_or_v2_accepted")
    parser.add_argument("--min-status", choices=["GOOD", "USABLE", "LOW_CONFIDENCE", "EXCLUDE"], default="USABLE")
    parser.add_argument("--include-low-confidence", type=str_to_bool, default=False)
    parser.add_argument("--include-exclude", type=str_to_bool, default=False)
    parser.add_argument("--pad-to-max", type=str_to_bool, default=True)
    parser.add_argument("--write-debug-images", type=str_to_bool, default=True)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--strict", type=str_to_bool, default=False)
    parser.add_argument("--exposure-audit", type=str_to_bool, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-sample-limit", type=int, default=4)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    root = resolve_path(args.root)
    report_root = resolve_path(args.report_root)
    report_dir = resolve_path(args.report_dir) if args.report_dir else latest_report_dir(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    git_before = git_status_short()
    env = collect_environment_v2()
    command_args = sys.argv[1:] if argv is None else list(argv)
    command = "PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_inputs_v2 " + shlex.join(command_args)

    status_jsons = discover_status_jsons(root)
    if args.debug:
        status_jsons = status_jsons[: max(1, args.debug_sample_limit)]
        print(f"[debug] Processing first {len(status_jsons)} samples.")
    print(f"Found {len(status_jsons)} frame_selection_v2 status files.")

    results: list[SampleResultV2] = []
    audits: dict[str, dict[str, Any]] = {}
    for index, status_json in enumerate(status_jsons, start=1):
        sample_id = status_json.parent.parent.name
        print(f"[{index}/{len(status_jsons)}] {sample_id}")
        try:
            result, audit = process_sample_v2(status_json, args)
            if audit is not None:
                audits[result.sample_id] = audit
            print(f"  wrote {display_path(result.shape_bag_npz)} ({result.num_frames_used} frames)")
        except Exception as exc:
            if args.strict:
                raise
            result = SampleResultV2(
                sample_id=sample_id,
                source_type="failed",
                sample_status="FAILED",
                sample_quality=0.0,
                sample_folder=status_json.parent.parent,
                frame_selection_folder=status_json.parent,
                shape_input_folder=status_json.parent.parent / "rheed_shape_input_v2",
                num_frames_available=0,
                num_frames_used=0,
                rejected_frame_count=0,
                accepted_csv=status_json.parent / "accepted_candidate_frames.csv",
                manual_selection_file=status_json.parent / "manual_selected_frames.txt",
                shape_bag_npz=status_json.parent.parent / "rheed_shape_input_v2" / "shape_bag_v2.npz",
                sample_feature_json=status_json.parent.parent / "rheed_shape_input_v2" / "sample_shape_features_v2.json",
                sample_feature_csv=status_json.parent.parent / "rheed_shape_input_v2" / "sample_shape_features_v2.csv",
                preview_grid=status_json.parent.parent / "rheed_shape_input_v2" / "shape_input_overview_v2.png",
                exposure_audit_json=status_json.parent.parent / "rheed_shape_input_v2" / "exposure_invariance_audit_v2.json",
                failure=f"{type(exc).__name__}: {exc}",
            )
            print(f"  skipped/failed: {result.failure}")
        results.append(result)

    successful = [result for result in results if not result.failure]
    failures = [result for result in results if result.failure]
    manifest_path = report_dir / "rheed_shape_bag_manifest_v2.csv"
    write_csv(manifest_path, [manifest_row_v2(result) for result in successful], MANIFEST_FIELDS_V2)
    write_csv(
        report_dir / "failed_shape_bag_samples_v2.csv",
        [{"sample_id": result.sample_id, "accepted_csv": display_path(result.accepted_csv), "failure": result.failure} for result in failures],
        ["sample_id", "accepted_csv", "failure"],
    )
    feature_table_path = write_global_feature_table_v2(report_dir, successful)
    default_features_path = write_default_feature_names_v2(report_dir, successful)
    git_after = git_status_short()
    report_path = append_shape_report(
        report_dir,
        args=args,
        git_before=git_before,
        git_after=git_after,
        env=env,
        results=successful,
        failures=failures,
        manifest_path=manifest_path,
        feature_table_path=feature_table_path,
        default_features_path=default_features_path,
        command=command,
    )
    print(f"Wrote manifest: {display_path(manifest_path)}")
    print(f"Wrote report: {display_path(report_path)}")
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
