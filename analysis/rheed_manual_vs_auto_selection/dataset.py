from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis.rheed_video_afm_story.common import (
    display_path,
    repo_path,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.pretrained_embeddings import (
    load_dino,
    load_r3d18,
    preprocess_frames,
    temporal_aggregate,
)
from rheed2morph.rheed.automatic_roi_keyframe import Rect, _source_factory
from rheed2morph.realtime.clips import build_model_clip, live_physics_row
from rheed2morph.realtime.selector import ReplaySelection, analyze_replay


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rect_dict(rect: Rect, prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_x": int(rect.x),
        f"{prefix}_y": int(rect.y),
        f"{prefix}_width": int(rect.width),
        f"{prefix}_height": int(rect.height),
    }


def _overlap(first: Rect, second: Rect) -> dict[str, float]:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.width, second.x + second.width)
    y1 = min(first.y + first.height, second.y + second.height)
    intersection = max(x1 - x0, 0) * max(y1 - y0, 0)
    first_area = max(first.width * first.height, 1)
    second_area = max(second.width * second.height, 1)
    union = first_area + second_area - intersection
    return {
        "roi_iou": float(intersection / max(union, 1)),
        "human_roi_coverage": float(intersection / first_area),
        "machine_roi_coverage_by_human": float(intersection / second_area),
        "machine_to_human_area_ratio": float(second_area / first_area),
    }


def _decode_selected16(
    source: Path,
    keyframe_index: int,
    *,
    rotation_clockwise_degrees: int = 0,
) -> list[np.ndarray]:
    requested = set(range(int(keyframe_index) - 7, int(keyframe_index) + 9))
    if min(requested) < 0:
        raise IndexError(f"keyframe {keyframe_index} is too close to video start")
    factory, _, _ = _source_factory(
        source,
        rotation_clockwise_degrees=rotation_clockwise_degrees,
    )
    decoded: dict[int, np.ndarray] = {}
    for index, frame in factory():
        if index in requested:
            decoded[index] = np.asarray(frame)
        if index > max(requested):
            break
    missing = sorted(requested - set(decoded))
    if missing:
        raise IndexError(f"{source}: selected-16 frames missing: {missing}")
    return [decoded[index] for index in sorted(requested)]


def _save_clip_variants(
    *,
    output_root: Path,
    sample_id: str,
    clip: np.ndarray,
    frame_indices: list[int],
    video_id: str,
) -> list[dict[str, Any]]:
    variants = {
        "keyframe_1": (clip[7:8], frame_indices[7:8]),
        "causal_8": (clip[:8], frame_indices[:8]),
        "selected_16": (clip, frame_indices),
    }
    rows: list[dict[str, Any]] = []
    for variant, (frames, indices) in variants.items():
        path = output_root / "clip_variants" / variant / f"{sample_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frames_uint8=np.asarray(frames, dtype=np.uint8),
            frame_indices=np.asarray(indices, dtype=np.int32),
            sample_id=np.asarray(sample_id),
            growth_run_id=np.asarray(sample_id),
            video_id=np.asarray(video_id),
            clip_variant=np.asarray(variant),
            selection_source=np.asarray("automatic_v5_v8"),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "growth_run_id": sample_id,
                "video_id": video_id,
                "clip_variant": variant,
                "available": True,
                "frame_indices": json.dumps(indices),
                "frame_count": len(indices),
                "cache_path": display_path(path),
                "preprocessing": "raw_luminance_resize_pad_224",
                "selection_source": "automatic_v5_v8",
            }
        )
    return rows


def _selection_record(
    row: pd.Series,
    selection: ReplaySelection,
    *,
    source: Path,
) -> dict[str, Any]:
    if len(selection.events) != 1:
        raise RuntimeError(
            f"{row['sample_id']}: expected one best-cycle event, "
            f"found {len(selection.events)}"
        )
    event = selection.events[0]
    human_roi = Rect(
        int(row["roi_x"]),
        int(row["roi_y"]),
        int(row["roi_width"]),
        int(row["roi_height"]),
        int(row["source_width"]),
        int(row["source_height"]),
    )
    machine_roi = selection.model_input_roi.rect
    delta = int(event.frame_index) - int(row["keyframe_index"])
    period = selection.estimated_period_frames
    phase_residual = abs(delta)
    if period is not None and np.isfinite(period) and period > 0:
        phase_residual = abs(delta - round(delta / period) * period)
    return {
        "sample_id": str(row["sample_id"]),
        "growth_run_id": str(row["growth_run_id"]),
        "video_id": str(row["video_id"]),
        "source_video": display_path(source),
        "source_size_bytes": int(source.stat().st_size),
        "source_mtime_ns": int(source.stat().st_mtime_ns),
        "human_keyframe_index": int(row["keyframe_index"]),
        "machine_keyframe_index": int(event.frame_index),
        "signed_keyframe_delta": int(delta),
        "cycle_phase_residual_frames": float(phase_residual),
        "estimated_period_frames": (
            float(period) if period is not None else np.nan
        ),
        "machine_keyframe_quality": float(event.keyframe_quality),
        "machine_visibility_rank": float(event.visibility_rank),
        "machine_model_input_visibility": float(event.model_input_visibility),
        "machine_selector_score": float(event.selector_score),
        "machine_refined_from_frame_index": event.refined_from_frame_index,
        "selector_tracker": event.tracker,
        **_rect_dict(human_roi, "human_roi"),
        **_rect_dict(machine_roi, "machine_roi"),
        **_rect_dict(selection.tracking_roi.rect, "tracking_roi"),
        **_overlap(human_roi, machine_roi),
        "raw_data_modified": False,
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    result = torch.device(name)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


@torch.inference_mode()
def extract_embeddings(
    *,
    groups: list[str],
    output_root: Path,
    device_name: str,
) -> pd.DataFrame:
    registry_rows: list[dict[str, Any]] = []
    device = _device(device_name)
    jobs = [
        ("r3d_18", "causal_8", True),
        ("r3d_18", "selected_16", True),
        ("dino_vits14", "keyframe_1", False),
    ]
    loaded: dict[str, tuple[torch.nn.Module, str]] = {}
    for encoder, _, _ in jobs:
        if encoder in loaded:
            continue
        model, status = load_r3d18() if encoder == "r3d_18" else load_dino()
        if model is None or not status.loaded:
            raise RuntimeError(
                f"{encoder} could not load frozen weights: {status.reason}"
            )
        # R3D-18 is faster and numerically aligned with the frozen cache on CPU.
        model_device = torch.device("cpu") if encoder == "r3d_18" else device
        loaded[encoder] = (model.to(model_device).eval(), status.weight_identifier)
    embed_root = output_root / "embeddings"
    embed_root.mkdir(parents=True, exist_ok=True)
    for encoder, variant, video in jobs:
        model, weight_identifier = loaded[encoder]
        model_device = next(model.parameters()).device
        embeddings: list[np.ndarray] = []
        for group in groups:
            payload = np.load(
                output_root / "clip_variants" / variant / f"{group}.npz",
                allow_pickle=False,
            )
            frames = np.asarray(payload["frames_uint8"], dtype=np.uint8)
            tensor = preprocess_frames(
                frames, "raw_luminance", video=video
            ).to(model_device)
            values = model(tensor).detach().cpu().numpy().astype(np.float32)
            embedding = (
                values[0] if video else temporal_aggregate(values)
            )
            embeddings.append(np.asarray(embedding, dtype=np.float32))
        matrix = np.vstack(embeddings).astype(np.float32)
        embedding_id = f"{encoder}__{variant}__raw_luminance"
        path = embed_root / f"{embedding_id}.npz"
        np.savez_compressed(
            path,
            sample_ids=np.asarray(groups),
            growth_run_ids=np.asarray(groups),
            embeddings=matrix,
            encoder_name=np.asarray(encoder),
            weight_identifier=np.asarray(weight_identifier),
            clip_variant=np.asarray(variant),
            preprocessing=np.asarray("raw_luminance"),
            embedding_dim=np.asarray(matrix.shape[1]),
            selection_source=np.asarray("automatic_v5_v8"),
        )
        registry_rows.append(
            {
                "embedding_id": embedding_id,
                "encoder": encoder,
                "clip_variant": variant,
                "preprocessing": "raw_luminance",
                "path": display_path(path),
                "N": len(groups),
                "embedding_dim": matrix.shape[1],
                "weight_identifier": weight_identifier,
                "selection_source": "automatic_v5_v8",
            }
        )
    return pd.DataFrame(registry_rows)


def build_dataset(config: dict[str, Any], *, device_name: str) -> None:
    started = time.time()
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        repo_path(config["human_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    cohort = pd.read_csv(
        Path(config["standalone_cohort"]).expanduser(),
        dtype={"growth_run_id": str},
    )
    groups = cohort["growth_run_id"].astype(str).tolist()
    if len(groups) != 23 or len(set(groups)) != 23:
        raise RuntimeError("frozen M14i cohort is not exactly 23 unique growths")
    rows = (
        manifest.loc[manifest["growth_run_id"].isin(groups)]
        .set_index("growth_run_id")
        .loc[groups]
        .reset_index()
    )
    if len(rows) != 23:
        raise RuntimeError("human manifest does not cover frozen M14i cohort")

    selection_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    physics_records: list[pd.DataFrame] = []
    machine_manifest_rows: list[pd.Series] = []
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        group = str(row["growth_run_id"])
        source = repo_path(str(row["source_video"]))
        cache_path = output_root / "selections" / f"{group}.json"
        print(f"[{position:02d}/23] automatic selection {group}: {source.name}", flush=True)
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
            device=device_name,
            roi_sample_count=int(config["roi_sample_count"]),
            minimum_event_quality=float(config["minimum_keyframe_quality"]),
            event_policy="best_visible_cycle",
            refinement_period_fraction=float(
                config["keyframe_refinement_period_fraction"]
            ),
        )
        record = _selection_record(row, selection, source=source)
        selection_records.append(record)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(record, cache_path)
        keyframe = int(record["machine_keyframe_index"])
        indices = list(range(keyframe - 7, keyframe + 9))
        decoded = _decode_selected16(source, keyframe)
        clip = build_model_clip(
            decoded,
            selection.model_input_roi.rect,
            output_size=int(config["model_image_size"]),
        )
        variant_records.extend(
            _save_clip_variants(
                output_root=output_root,
                sample_id=group,
                clip=clip,
                frame_indices=indices,
                video_id=str(row["video_id"]),
            )
        )
        physics = live_physics_row(clip, sample_id=group).reset_index(drop=True)
        physics["growth_run_id"] = group
        physics["video_stage"] = "machine_selected_v5_v8"
        physics_records.append(physics)
        updated = row.copy()
        roi = selection.model_input_roi.rect
        updated["keyframe_index"] = keyframe
        updated["clip_start_index"] = keyframe - 7
        updated["clip_end_index"] = keyframe + 8
        updated["roi_x"] = roi.x
        updated["roi_y"] = roi.y
        updated["roi_width"] = roi.width
        updated["roi_height"] = roi.height
        updated["selection_source"] = "automatic_v5_v8"
        updated["human_keyframe_index"] = int(row["keyframe_index"])
        machine_frame_paths = [
            display_path(Path(str(row["frames_dir"])) / f"{index}.png")
            for index in indices
        ]
        if not all(repo_path(path).exists() for path in machine_frame_paths):
            raise FileNotFoundError(
                f"{group}: extracted machine-selected frame paths are incomplete"
            )
        updated["clip_frame_paths"] = json.dumps(machine_frame_paths)
        updated["clip_frame_indices"] = json.dumps(indices)
        updated["clip_frame_count"] = 16
        updated["actual_clip_frame_count"] = 16
        updated["keyframe_offset_in_clip_x"] = 7
        updated["keyframe_offset_in_clip_y"] = 7
        updated["clip_cache_path"] = display_path(
            output_root / "clip_variants" / "selected_16" / f"{group}.npz"
        )
        updated["clip_preview_path"] = ""
        machine_manifest_rows.append(updated)

    selection_table = pd.DataFrame(selection_records)
    variant_table = pd.DataFrame(variant_records)
    physics_table = pd.concat(physics_records, ignore_index=True)
    machine_manifest = pd.DataFrame(machine_manifest_rows)
    write_csv(selection_table, output_root / "selection_comparison.csv")
    write_csv(variant_table, output_root / "clip_variant_manifest.csv")
    write_csv(physics_table, output_root / "rheed_physics_features.csv")
    write_csv(machine_manifest, output_root / "modeling_manifest.csv")
    registry = extract_embeddings(
        groups=groups,
        output_root=output_root,
        device_name=device_name,
    )
    write_csv(registry, output_root / "embedding_registry.csv")
    summary = {
        "experiment_id": config["experiment_id"],
        "cohort": "frozen M14i Full23; 6043 and 6055 excluded",
        "growth_group_count": len(groups),
        "growth_run_ids": groups,
        "selection": "V5 DINOv2-S best visible cycle + V8 full model-input ROI",
        "temporal_semantics": "selected_16 = k-7..k+8; keyframe at index 7",
        "median_cycle_phase_residual_frames": float(
            selection_table["cycle_phase_residual_frames"].median()
        ),
        "median_roi_iou": float(selection_table["roi_iou"].median()),
        "median_human_roi_coverage": float(
            selection_table["human_roi_coverage"].median()
        ),
        "median_keyframe_quality": float(
            selection_table["machine_keyframe_quality"].median()
        ),
        "source_video_content_hashes_computed": False,
        "source_video_size_and_mtime_recorded": True,
        "raw_data_modified": False,
        "standalone_modified": False,
        "standalone_target_parameters_sha256": _sha256_file(
            Path(config["standalone_target_parameters"]).expanduser()
        ),
        "standalone_generator_parameters_sha256": _sha256_file(
            Path(config["standalone_generator_parameters"]).expanduser()
        ),
        "runtime_seconds": time.time() - started,
    }
    write_json(summary, output_root / "dataset_manifest.json")
    write_json(summary, report_root / "dataset_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_manual_vs_auto_selection.json",
    )
    parser.add_argument("--device", default="mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_dataset(load_config(args.config), device_name=str(args.device))


if __name__ == "__main__":
    main()
