from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analysis.rheed_manual_vs_auto_selection.dataset import (
    _decode_selected16,
    extract_embeddings,
)
from analysis.rheed_video_afm_story.common import (
    display_path,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)
from rheed2morph.rheed.automatic_roi_keyframe import Rect, _source_factory
from rheed2morph.rheed.orientation import rotation_for_sample
from rheed2morph.realtime.clips import build_model_clip, live_physics_row
from rheed2morph.realtime.selector import analyze_replay


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _hash_object(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_clip_variants(
    *,
    output_root: Path,
    sample_id: str,
    video_id: str,
    clip: np.ndarray,
    frame_indices: list[int],
) -> list[dict[str, Any]]:
    selection_source = "automatic_v5_v8_frozen_transfer"
    variants = {
        "keyframe_1": (clip[7:8], frame_indices[7:8]),
        "causal_8": (clip[:8], frame_indices[:8]),
        "selected_16": (clip, frame_indices),
    }
    records: list[dict[str, Any]] = []
    for variant, (frames, indices) in variants.items():
        destination = (
            output_root / "clip_variants" / variant / f"{sample_id}.npz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            frames_uint8=np.asarray(frames, dtype=np.uint8),
            frame_indices=np.asarray(indices, dtype=np.int32),
            sample_id=np.asarray(sample_id),
            growth_run_id=np.asarray(sample_id),
            video_id=np.asarray(video_id),
            clip_variant=np.asarray(variant),
            selection_source=np.asarray(selection_source),
        )
        records.append(
            {
                "sample_id": sample_id,
                "growth_run_id": sample_id,
                "video_id": video_id,
                "clip_variant": variant,
                "available": True,
                "frame_indices": json.dumps(indices),
                "frame_count": len(indices),
                "cache_path": display_path(destination),
                "preprocessing": "raw_luminance_resize_pad_224",
                "selection_source": selection_source,
            }
        )
    return records


def _video_inventory(config: dict[str, Any]) -> pd.DataFrame:
    root = repo_path(config["raw_rheed_root"])
    selected = {
        str(sample): repo_path(path).resolve()
        for sample, path in config["selected_videos"].items()
    }
    records: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        sample_id = str(sample_dir.name)
        for source in sorted(
            path
            for path in sample_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ):
            is_selected = (
                sample_id in selected and source.resolve() == selected[sample_id]
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "raw_rheed_video": display_path(source),
                    "raw_rheed_sha256": (
                        sha256_file(source) if is_selected else ""
                    ),
                    "size_bytes": int(source.stat().st_size),
                    "mtime_ns": int(source.stat().st_mtime_ns),
                    "selected_for_modeling": is_selected,
                    "selection_reason": (
                        "lossless final rampdown recording selected"
                        if is_selected and source.suffix.lower() == ".avi"
                        else (
                            "only directly decodable final rampdown recording"
                            if is_selected
                            else "retained raw alternative; not used"
                        )
                    ),
                    "integrity_scope": (
                        "full_sha256"
                        if is_selected
                        else "inventory_only_size_and_mtime"
                    ),
                    "raw_data_modified": False,
                }
            )
    table = pd.DataFrame(records).sort_values(
        ["sample_id", "raw_rheed_video"]
    )
    selected_rows = table.loc[table["selected_for_modeling"]]
    expected = set(map(str, config["included_samples"]))
    if (
        set(selected_rows["sample_id"].astype(str)) != expected
        or len(selected_rows) != len(expected)
    ):
        raise RuntimeError("selected RHEED video inventory is not one-per-growth")
    if "N6324" in set(table["sample_id"].loc[table["selected_for_modeling"]]):
        raise RuntimeError("N6324 entered the selected RHEED inventory")
    return table


def _selection_record(
    *,
    sample_id: str,
    source: Path,
    source_sha256: str,
    keyframe: int,
    selection: Any,
    frame_rotation_clockwise_degrees: int,
) -> dict[str, Any]:
    event = selection.events[0]
    machine = selection.model_input_roi.rect
    tracking = selection.tracking_roi.rect
    return {
        "sample_id": sample_id,
        "growth_run_id": sample_id,
        "video_id": source.stem,
        "source_video": display_path(source),
        "source_sha256": str(source_sha256),
        "source_size_bytes": int(source.stat().st_size),
        "source_mtime_ns": int(source.stat().st_mtime_ns),
        "human_keyframe_index": np.nan,
        "machine_keyframe_index": int(keyframe),
        "signed_keyframe_delta": np.nan,
        "cycle_phase_residual_frames": np.nan,
        "estimated_period_frames": float(selection.estimated_period_frames),
        "machine_keyframe_quality": float(event.keyframe_quality),
        "machine_visibility_rank": float(event.visibility_rank),
        "machine_model_input_visibility": float(
            event.model_input_visibility
        ),
        "machine_selector_score": float(event.selector_score),
        "machine_refined_from_frame_index": event.refined_from_frame_index,
        "selector_tracker": str(event.tracker),
        "human_roi_x": np.nan,
        "human_roi_y": np.nan,
        "human_roi_width": np.nan,
        "human_roi_height": np.nan,
        "machine_roi_x": int(machine.x),
        "machine_roi_y": int(machine.y),
        "machine_roi_width": int(machine.width),
        "machine_roi_height": int(machine.height),
        "tracking_roi_x": int(tracking.x),
        "tracking_roi_y": int(tracking.y),
        "tracking_roi_width": int(tracking.width),
        "tracking_roi_height": int(tracking.height),
        "roi_iou": np.nan,
        "human_roi_coverage": np.nan,
        "machine_roi_coverage_by_human": np.nan,
        "machine_to_human_area_ratio": np.nan,
        "selection_source": "automatic_v5_v8_frozen_transfer",
        "frame_rotation_clockwise_degrees": int(
            frame_rotation_clockwise_degrees
        ),
        "orientation_correction_stage": (
            "before_keyframe_roi_physics_and_embedding"
        ),
        "raw_data_modified": False,
    }


def _load_cached_selection(
    path: Path,
    *,
    source: Path,
    source_sha256: str,
    frame_rotation_clockwise_degrees: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        str(record.get("source_sha256", "")) != str(source_sha256)
        or str(record.get("source_video", "")) != display_path(source)
        or int(record.get("frame_rotation_clockwise_degrees", 0))
        != int(frame_rotation_clockwise_degrees)
    ):
        return None
    return record


def _clip_from_record(
    record: dict[str, Any],
    *,
    source: Path,
    output_size: int,
) -> tuple[np.ndarray, list[int], Rect]:
    keyframe = int(record["machine_keyframe_index"])
    indices = list(range(keyframe - 7, keyframe + 9))
    rotation = int(record.get("frame_rotation_clockwise_degrees", 0))
    decoded = _decode_selected16(
        source,
        keyframe,
        rotation_clockwise_degrees=rotation,
    )
    roi = Rect(
        x=int(record["machine_roi_x"]),
        y=int(record["machine_roi_y"]),
        width=int(record["machine_roi_width"]),
        height=int(record["machine_roi_height"]),
        source_width=int(np.asarray(decoded[0]).shape[1]),
        source_height=int(np.asarray(decoded[0]).shape[0]),
    )
    roi = _trim_portrait_aperture_edge(decoded, roi, record=record)
    clip = build_model_clip(decoded, roi, output_size=int(output_size))
    return clip, indices, roi


def _trim_portrait_aperture_edge(
    decoded: list[np.ndarray],
    roi: Rect,
    *,
    record: dict[str, Any],
) -> Rect:
    """Remove a detected lower eyepiece arc from portrait transfer videos.

    The frozen V8 calibration was fitted mostly on landscape recordings.
    Two extra-five recordings are portrait-oriented and its conservative
    rectangle extends into the lower circular edge.  This target-blind guard
    operates only when the ROI nearly touches the bottom of a portrait frame.
    It retains the largest upper prefix for which at least 97% of each row is
    inside the illuminated aperture in the temporal-median frame.
    """

    if (
        roi.source_height <= roi.source_width
        or roi.y2 / max(roi.source_height, 1) < 0.90
    ):
        record["aperture_edge_trim_applied"] = False
        return roi
    frames = np.asarray(decoded, dtype=np.float32)
    brightness = np.median(frames.max(axis=3) / 255.0, axis=0)
    values = brightness[roi.as_slices()]
    row_inside_fraction = (values > 0.05).mean(axis=1)
    safe = np.where(row_inside_fraction >= 0.97)[0]
    if not len(safe):
        record["aperture_edge_trim_applied"] = False
        return roi
    new_height = int(safe[-1] + 1)
    if new_height >= roi.height or new_height < int(0.65 * roi.height):
        record["aperture_edge_trim_applied"] = False
        return roi
    trimmed = Rect(
        x=roi.x,
        y=roi.y,
        width=roi.width,
        height=new_height,
        source_width=roi.source_width,
        source_height=roi.source_height,
    ).clipped()
    record["untrimmed_machine_roi_x"] = int(roi.x)
    record["untrimmed_machine_roi_y"] = int(roi.y)
    record["untrimmed_machine_roi_width"] = int(roi.width)
    record["untrimmed_machine_roi_height"] = int(roi.height)
    record["machine_roi_x"] = int(trimmed.x)
    record["machine_roi_y"] = int(trimmed.y)
    record["machine_roi_width"] = int(trimmed.width)
    record["machine_roi_height"] = int(trimmed.height)
    record["aperture_edge_trim_applied"] = True
    record["aperture_edge_trim_rule"] = (
        "portrait ROI y2>=90% frame; retain rows with >=97% temporal-median "
        "brightness above 0.05"
    )
    record["aperture_edge_removed_pixels"] = int(roi.height - trimmed.height)
    return trimmed


def _extra_manifest_row(
    *,
    columns: list[str],
    selection: dict[str, Any],
    target: pd.Series,
    clip_path: Path,
    frame_indices: list[int],
    roi: Rect,
    config_hash: str,
    removelist_hash: str,
) -> dict[str, Any]:
    sample_id = str(selection["sample_id"])
    row = {column: np.nan for column in columns}
    row.update(
        {
            "growth_run_id": sample_id,
            "sample_id": sample_id,
            "video_id": str(selection["video_id"]),
            "source_video": str(selection["source_video"]),
            "metadata_path": "",
            "frames_dir": "",
            "keyframe_index": int(selection["machine_keyframe_index"]),
            "clip_frame_count": 16,
            "roi_x": int(roi.x),
            "roi_y": int(roi.y),
            "roi_width": int(roi.width),
            "roi_height": int(roi.height),
            "source_width": int(roi.source_width),
            "source_height": int(roi.source_height),
            "clip_start_index": int(frame_indices[0]),
            "clip_end_index": int(frame_indices[-1]),
            "actual_clip_frame_count": 16,
            "sample_group_id": sample_id,
            "keyframe_offset_in_clip_x": 7,
            "growth_stage": "post_growth_rampdown",
            "video_stage": "final_rampdown",
            "material": "GaSb/AlGaSb",
            "substrate": "unknown",
            "camera_or_tool_id": "legacy_RHEED_camera",
            "fps": 10.0,
            "primary_afm_available": True,
            "primary_afm_scan_size_um": 1.0,
            "primary_afm_scan_count": int(target["primary_afm_scan_count"]),
            "primary_rq_nm_median": float(target["sample_median_sq_nm"]),
            "primary_rq_nm_iqr": float(target["sample_sq_iqr_nm"]),
            "primary_rq_nm_mad": np.nan,
            "primary_rq_nm_min": float(target["sample_sq_min_nm"]),
            "primary_rq_nm_max": float(target["sample_sq_max_nm"]),
            "representative_afm_path": str(
                target["representative_afm_height_array"]
            ),
            "representative_afm_height_array": str(
                target["representative_afm_height_array"]
            ),
            "representative_afm_scan_id": str(
                target["representative_afm_scan_id"]
            ),
            # The raw parent is 2 × 2 µm, but the model-visible AFM records
            # are the non-overlapping 1 × 1 µm line-3 subfields.
            "best_available_afm_scan_size_um": 1.0,
            "best_available_rq_nm_median": float(
                target["sample_median_sq_nm"]
            ),
            "best_available_scan_count": int(
                target["primary_afm_scan_count"]
            ),
            "cohort_primary_1um": True,
            "cohort_exploratory_best_available": True,
            "afm_scan_count_all": int(
                target["primary_afm_scan_count"]
            ),
            "clip_cache_path": display_path(clip_path),
            "clip_preview_path": "",
            "clip_frame_paths": json.dumps([]),
            "clip_frame_indices": json.dumps(frame_indices),
            "keyframe_offset_in_clip_y": 7,
            "roi_area_fraction": float(
                roi.width * roi.height
                / max(roi.source_width * roi.source_height, 1)
            ),
            "crop_verified": True,
            "resize_scale": np.nan,
            "padding": "",
            "rheed_quality_flags": "",
            "excluded_by_removelist": False,
            "exclusion_reason": "",
            "usable_for_modeling": True,
            "rheed_quality_pass": True,
            "manifest_source_hash": str(selection["source_sha256"]),
            "removelist_hash": removelist_hash,
            "config_hash": config_hash,
            "selection_source": "automatic_v5_v8_frozen_transfer",
            "human_keyframe_index": np.nan,
            "afm_target_variant": (
                "crop_2um_to_nonoverlap_1um_then_line3_scanline_flatten_v1"
            ),
            "roughness_nomenclature": "Sq_areal_RMS_height_nm",
        }
    )
    return row


def _combine_embeddings(
    *,
    base_registry_path: Path,
    extra_registry_path: Path,
    destination_root: Path,
    expected_groups: list[str],
) -> pd.DataFrame:
    base_registry = pd.read_csv(base_registry_path).set_index("embedding_id")
    extra_registry = pd.read_csv(extra_registry_path).set_index("embedding_id")
    if set(base_registry.index) != set(extra_registry.index):
        raise RuntimeError("base and extra embedding families do not match")
    records: list[dict[str, Any]] = []
    embedding_root = destination_root / "embeddings"
    embedding_root.mkdir(parents=True, exist_ok=True)
    for embedding_id in base_registry.index:
        base_path = repo_path(base_registry.loc[embedding_id, "path"])
        extra_path = repo_path(extra_registry.loc[embedding_id, "path"])
        base = np.load(base_path, allow_pickle=False)
        extra = np.load(extra_path, allow_pickle=False)
        base_groups = [str(x) for x in base["growth_run_ids"].tolist()]
        extra_groups = [str(x) for x in extra["growth_run_ids"].tolist()]
        groups = base_groups + extra_groups
        if groups != expected_groups:
            raise RuntimeError(
                f"{embedding_id}: combined embedding order mismatch"
            )
        if str(base["weight_identifier"]) != str(extra["weight_identifier"]):
            raise RuntimeError(f"{embedding_id}: frozen encoder weights differ")
        matrix = np.vstack([base["embeddings"], extra["embeddings"]]).astype(
            np.float32
        )
        output = embedding_root / f"{embedding_id}.npz"
        np.savez_compressed(
            output,
            sample_ids=np.asarray(groups),
            growth_run_ids=np.asarray(groups),
            embeddings=matrix,
            encoder_name=np.asarray(str(base["encoder_name"])),
            weight_identifier=np.asarray(str(base["weight_identifier"])),
            clip_variant=np.asarray(str(base["clip_variant"])),
            preprocessing=np.asarray(str(base["preprocessing"])),
            embedding_dim=np.asarray(matrix.shape[1]),
            selection_source=np.asarray(
                "automatic_v5_v8_base23_plus_frozen_transfer_extra5"
            ),
        )
        records.append(
            {
                "embedding_id": embedding_id,
                "encoder": str(base["encoder_name"]),
                "clip_variant": str(base["clip_variant"]),
                "preprocessing": str(base["preprocessing"]),
                "path": display_path(output),
                "N": len(groups),
                "embedding_dim": int(matrix.shape[1]),
                "weight_identifier": str(base["weight_identifier"]),
                "selection_source": (
                    "automatic_v5_v8_base23_plus_frozen_transfer_extra5"
                ),
            }
        )
    return pd.DataFrame(records)


def _keyframe(
    source: Path,
    frame_index: int,
    *,
    frame_rotation_clockwise_degrees: int = 0,
) -> np.ndarray:
    factory, _, _ = _source_factory(
        source,
        rotation_clockwise_degrees=frame_rotation_clockwise_degrees,
    )
    for index, frame in factory():
        if index == frame_index:
            return np.asarray(frame)
        if index > frame_index:
            break
    raise IndexError(f"{source}: frame {frame_index} is unavailable")


def _plot_extra_inputs(
    extra_selection: pd.DataFrame,
    report_root: Path,
) -> None:
    figure, axes = plt.subplots(
        len(extra_selection),
        2,
        figsize=(8.0, 3.0 * len(extra_selection)),
        constrained_layout=True,
    )
    for row_index, row in enumerate(extra_selection.itertuples(index=False)):
        source = repo_path(row.source_video)
        rotation = int(
            getattr(row, "frame_rotation_clockwise_degrees", 0)
        )
        frame = _keyframe(
            source,
            int(row.machine_keyframe_index),
            frame_rotation_clockwise_degrees=rotation,
        )
        roi = (
            int(row.machine_roi_x),
            int(row.machine_roi_y),
            int(row.machine_roi_width),
            int(row.machine_roi_height),
        )
        axes[row_index, 0].imshow(frame)
        axes[row_index, 0].add_patch(
            Rectangle(
                (roi[0], roi[1]),
                roi[2],
                roi[3],
                fill=False,
                edgecolor="#00D6B4",
                linewidth=1.6,
            )
        )
        axes[row_index, 0].set_title(
            f"{row.sample_id}: frame {row.machine_keyframe_index}, "
            f"quality {row.machine_keyframe_quality:.2f}; CW {rotation}°"
        )
        axes[row_index, 0].axis("off")
        crop = frame[
            roi[1] : roi[1] + roi[3],
            roi[0] : roi[0] + roi[2],
        ]
        axes[row_index, 1].imshow(crop)
        axes[row_index, 1].set_title(
            f"model-input ROI; period {row.estimated_period_frames:.1f} frames"
        )
        axes[row_index, 1].axis("off")
    figure.suptitle(
        "Frozen automatic selector applied to the five second-batch videos",
        fontsize=12,
    )
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            figure_root / f"Fig2_extra_five_machine_inputs{suffix}",
            dpi=260 if suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def run(config_path: str | Path, *, device: str) -> dict[str, Any]:
    started = time.time()
    config = _load_config(config_path)
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    data_root = repo_path(config["canonical_data_root"])
    machine_root = output_root / "machine_dataset_full28"
    extra_root = output_root / "machine_dataset_extra5"
    for path in (machine_root, extra_root, report_root, data_root):
        path.mkdir(parents=True, exist_ok=True)
    inventory = _video_inventory(config)
    write_csv(inventory, data_root / "raw_rheed_source_inventory.csv")
    write_csv(inventory, report_root / "raw_rheed_source_inventory.csv")
    selected_hashes = (
        inventory.loc[inventory["selected_for_modeling"]]
        .set_index("sample_id")["raw_rheed_sha256"]
        .astype(str)
        .to_dict()
    )
    target_table = repo_path(
        config.get(
            "extra_five_sample_target_table",
            output_root / "extra_five_sample_sq_targets.csv",
        )
    )
    targets = pd.read_csv(
        target_table,
        dtype={"sample_id": str, "growth_run_id": str},
    ).set_index("sample_id")
    included = list(map(str, config["included_samples"]))
    if list(targets.index) != included:
        targets = targets.loc[included]

    selection_records: list[dict[str, Any]] = []
    physics_records: list[pd.DataFrame] = []
    clip_records: list[dict[str, Any]] = []
    clip_metadata: dict[str, tuple[list[int], Rect]] = {}
    for position, sample_id in enumerate(included, start=1):
        source = repo_path(config["selected_videos"][sample_id])
        rotation = rotation_for_sample(
            config.get("rheed_rotation_clockwise_degrees_by_sample"),
            sample_id,
        )
        cache = extra_root / "selections" / f"{sample_id}.json"
        record = _load_cached_selection(
            cache,
            source=source,
            source_sha256=selected_hashes[sample_id],
            frame_rotation_clockwise_degrees=rotation,
        )
        if record is None and config.get("selection_seed_root"):
            seed_path = (
                repo_path(config["selection_seed_root"])
                / f"{sample_id}.json"
            )
            record = _load_cached_selection(
                seed_path,
                source=source,
                source_sha256=selected_hashes[sample_id],
                frame_rotation_clockwise_degrees=rotation,
            )
        if record is None:
            print(
                f"[RHEED select {position:02d}/{len(included):02d}] "
                f"{sample_id}: {source.name}",
                flush=True,
            )
            selection = analyze_replay(
                source,
                deep_visibility_ranker_path=repo_path(
                    config["deep_visibility_ranker"]
                ),
                model_input_calibration_path=repo_path(
                    config["model_input_roi_calibration"]
                ),
                full_lattice_calibration_path=repo_path(
                    config["full_lattice_roi_calibration"]
                ),
                foundation_cache_dir=repo_path(config["foundation_cache_dir"]),
                device=device,
                roi_sample_count=int(config["roi_sample_count"]),
                minimum_event_quality=float(config["minimum_keyframe_quality"]),
                event_policy="best_visible_cycle",
                refinement_period_fraction=float(
                    config["keyframe_refinement_period_fraction"]
                ),
                frame_rotation_clockwise_degrees=rotation,
            )
            if len(selection.events) != 1:
                raise RuntimeError(
                    f"{sample_id}: expected one selected event, "
                    f"found {len(selection.events)}"
                )
            if (
                selection.estimated_period_frames is None
                or not np.isfinite(selection.estimated_period_frames)
            ):
                raise RuntimeError(f"{sample_id}: no finite rotation period")
            record = _selection_record(
                sample_id=sample_id,
                source=source,
                source_sha256=selected_hashes[sample_id],
                keyframe=int(selection.events[0].frame_index),
                selection=selection,
                frame_rotation_clockwise_degrees=rotation,
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            write_json(record, cache)
        override_samples = set(
            map(str, config.get("keyframe_override_samples", []))
        )
        if (
            sample_id in override_samples
            and config.get("keyframe_override_records_root")
        ):
            override_path = (
                repo_path(config["keyframe_override_records_root"])
                / f"{sample_id}.json"
            )
            override = json.loads(override_path.read_text(encoding="utf-8"))
            original_rotated_index = int(
                record.get(
                    "rotated_reanalysis_keyframe_index",
                    record["machine_keyframe_index"],
                )
            )
            for field in (
                "machine_keyframe_index",
                "machine_keyframe_quality",
                "machine_visibility_rank",
                "machine_model_input_visibility",
                "machine_selector_score",
                "machine_refined_from_frame_index",
                "selector_tracker",
                "estimated_period_frames",
            ):
                if field in override:
                    record[field] = override[field]
            record["rotated_reanalysis_keyframe_index"] = (
                original_rotated_index
            )
            record["keyframe_override_applied"] = True
            record["keyframe_override_source"] = display_path(override_path)
            record["temporal_selection_coordinate_system"] = (
                "frozen_target_blind_raw_acquisition_coordinates"
            )
            record["model_visible_coordinate_system"] = (
                f"clockwise_{rotation}_degrees"
            )
        else:
            record["keyframe_override_applied"] = False
        clip, indices, roi = _clip_from_record(
            record,
            source=source,
            output_size=int(config["model_image_size"]),
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json(record, cache)
        clip_records.extend(
            _save_clip_variants(
                output_root=extra_root,
                sample_id=sample_id,
                video_id=str(record["video_id"]),
                clip=clip,
                frame_indices=indices,
            )
        )
        physics = live_physics_row(clip, sample_id=sample_id).reset_index(
            drop=True
        )
        physics["growth_run_id"] = sample_id
        physics["video_stage"] = "machine_selected_v5_v8_frozen_transfer"
        physics_records.append(physics)
        selection_records.append(record)
        clip_metadata[sample_id] = (indices, roi)

    extra_selection = pd.DataFrame(selection_records)
    extra_physics = pd.concat(physics_records, ignore_index=True)
    write_csv(extra_selection, extra_root / "selection_comparison.csv")
    write_csv(extra_physics, extra_root / "rheed_physics_features.csv")
    write_csv(pd.DataFrame(clip_records), extra_root / "clip_variant_manifest.csv")
    extra_registry = extract_embeddings(
        groups=included,
        output_root=extra_root,
        device_name=device,
    )
    write_csv(extra_registry, extra_root / "embedding_registry.csv")

    base_root = repo_path(config["base_machine_dataset_root"])
    base_selection = pd.read_csv(
        base_root / "selection_comparison.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    base_physics = pd.read_csv(
        base_root / "rheed_physics_features.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    base_manifest = pd.read_csv(
        repo_path(config["base_automatic_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    if len(base_manifest) != 23:
        raise RuntimeError("the audited base automatic manifest is not Full23")
    config_hash = _hash_object(config)
    removelist_hash = sha256_file(repo_path("removelist.txt"))
    extra_manifest_records = []
    for record in selection_records:
        sample_id = str(record["sample_id"])
        indices, roi = clip_metadata[sample_id]
        extra_manifest_records.append(
            _extra_manifest_row(
                columns=list(base_manifest.columns),
                selection=record,
                target=targets.loc[sample_id],
                clip_path=(
                    extra_root
                    / "clip_variants"
                    / "selected_16"
                    / f"{sample_id}.npz"
                ),
                frame_indices=indices,
                roi=roi,
                config_hash=config_hash,
                removelist_hash=removelist_hash,
            )
        )
    extra_manifest = pd.DataFrame(
        extra_manifest_records, columns=base_manifest.columns
    )
    combined_manifest = pd.concat(
        [base_manifest, extra_manifest],
        ignore_index=True,
    )
    base_groups = base_manifest["growth_run_id"].astype(str).tolist()
    combined_groups = base_groups + included
    if (
        combined_manifest["growth_run_id"].astype(str).tolist()
        != combined_groups
    ):
        raise RuntimeError("combined manifest growth order is unstable")
    if combined_manifest["growth_run_id"].duplicated().any():
        raise RuntimeError("duplicate growth ID in combined RHEED manifest")
    if "N6324" in set(combined_groups):
        raise RuntimeError("N6324 entered the combined RHEED manifest")
    combined_selection = pd.concat(
        [base_selection, extra_selection],
        ignore_index=True,
        sort=False,
    )
    combined_selection = (
        combined_selection.set_index("growth_run_id")
        .loc[combined_groups]
        .reset_index()
    )
    combined_physics = pd.concat(
        [base_physics, extra_physics],
        ignore_index=True,
        sort=False,
    )
    combined_physics = (
        combined_physics.set_index("growth_run_id")
        .loc[combined_groups]
        .reset_index()
    )
    combined_registry = _combine_embeddings(
        base_registry_path=base_root / "embedding_registry.csv",
        extra_registry_path=extra_root / "embedding_registry.csv",
        destination_root=machine_root,
        expected_groups=combined_groups,
    )
    write_csv(combined_manifest, machine_root / "modeling_manifest.csv")
    write_csv(combined_selection, machine_root / "selection_comparison.csv")
    write_csv(combined_physics, machine_root / "rheed_physics_features.csv")
    write_csv(combined_registry, machine_root / "embedding_registry.csv")
    shutil.copy2(
        extra_root / "clip_variant_manifest.csv",
        machine_root / "extra_five_clip_variant_manifest.csv",
    )
    _plot_extra_inputs(extra_selection, report_root)
    manifest = {
        "experiment_id": config["experiment_id"],
        "base_growth_count": 23,
        "extra_growth_count": 5,
        "combined_growth_count": 28,
        "combined_growth_ids": combined_groups,
        "extra_growth_ids": included,
        "excluded_growth_ids": list(map(str, config["excluded_samples"])),
        "selected_video_hashes": selected_hashes,
        "selection_method": (
            "frozen V5 DINOv2-S best-visible-cycle selector + frozen V8 "
            "complete-lattice model-input ROI"
        ),
        "selection_model_refit_on_extra_five": False,
        "afm_targets_used_for_selection": False,
        "extra_five_sample_target_table": display_path(target_table),
        "rheed_rotation_clockwise_degrees_by_sample": {
            sample_id: rotation_for_sample(
                config.get("rheed_rotation_clockwise_degrees_by_sample"),
                sample_id,
            )
            for sample_id in included
        },
        "orientation_correction_applied_before_all_rheed_analysis": True,
        "embedding_families": combined_registry["embedding_id"].tolist(),
        "n6324_used": False,
        "raw_data_modified": False,
        "standalone_modified": False,
        "runtime_seconds": time.time() - started,
    }
    write_json(manifest, machine_root / "dataset_manifest.json")
    write_json(manifest, report_root / "rheed_integration_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/extra_five_line3_full28_v1.json",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run(args.config, device=str(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
