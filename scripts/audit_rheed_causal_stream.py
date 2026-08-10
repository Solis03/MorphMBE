#!/usr/bin/env python3
"""Run the live clear-moment detector without replay sleeps or AFM inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd

from rheed2morph.realtime.selector import (
    CausalClearMomentDetector,
    initialize_causal_stream,
)
from rheed2morph.rheed.automatic_roi_keyframe import _rgb_uint8
from rheed2morph.rheed.orientation import (
    rotate_frame_clockwise,
    rotation_for_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--config", default="configs/morphmbe_m22_realtime.json")
    parser.add_argument(
        "--output-dir",
        default="artifacts/causal_stream_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    repository = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rotation = rotation_for_sample(
        config.get("rheed_rotation_clockwise_degrees_by_sample"),
        str(args.sample_id),
    )
    reader = imageio.get_reader(str(source), "ffmpeg")
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps") or 30.0)
    duration = float(metadata.get("duration") or 0.0)
    frame_count = int(round(duration * fps)) if duration > 0 else 0
    warmup_count = int(config["online_roi_warmup_frames"])
    iterator = reader.iter_data()
    warmup = [
        rotate_frame_clockwise(_rgb_uint8(next(iterator)), rotation)
        for _ in range(warmup_count)
    ]
    selection = initialize_causal_stream(
        source,
        warmup,
        frame_count=max(frame_count, warmup_count),
        model_input_calibration_path=(
            repository / config["model_input_roi_calibration"]
        ),
        full_lattice_calibration_path=(
            repository / config["full_lattice_roi_calibration"]
        ),
        physics_calibration_path=(
            repository / config["physics_roi_calibration"]
            if config.get("physics_roi_calibration")
            else None
        ),
        frame_rotation_clockwise_degrees=rotation,
    )
    detector = CausalClearMomentDetector(
        tracking_roi=selection.tracking_roi,
        fallback_roi=(
            selection.model_input_roi
            if config.get("online_full_lattice_fallback_enabled", True)
            else None
        ),
        bundle_path=repository / config["online_clear_moment_detector"],
        minimum_event_frame=warmup_count,
        minimum_score=float(config["online_minimum_clear_score"]),
        lookahead_frames=int(config["online_vertex_lookahead_frames"]),
        history_frames=int(config["online_detector_history_frames"]),
        minimum_vertex_separation_frames=int(
            config["online_minimum_vertex_separation_frames"]
        ),
        fallback_minimum_score=float(
            config.get("online_fallback_minimum_clear_score", 0.30)
        ),
        fallback_minimum_visibility_proxy=float(
            config.get(
                "online_fallback_minimum_visibility_proxy",
                1.30,
            )
        ),
        fallback_maximum_shadow_fraction=float(
            config.get(
                "online_fallback_maximum_shadow_fraction",
                0.20,
            )
        ),
        fallback_minimum_spot_peak_count=float(
            config.get(
                "online_fallback_minimum_spot_peak_count",
                8.0,
            )
        ),
        fallback_minimum_clarity=float(
            config.get("online_fallback_minimum_clarity", 8.0)
        ),
        fallback_confirmation_delay_frames=int(
            config.get(
                "online_fallback_confirmation_delay_frames",
                8,
            )
        ),
        fallback_minimum_separation_frames=max(
            8,
            int(
                round(
                    fps
                    * float(
                        config.get(
                            "online_fallback_minimum_separation_seconds",
                            3.0,
                        )
                    )
                )
            ),
        ),
    )
    for index, frame in enumerate(warmup):
        detector.observe(index, frame)
    rows = []
    last_index = warmup_count - 1
    try:
        for index, frame in enumerate(iterator, start=warmup_count):
            last_index = index
            rgb = rotate_frame_clockwise(_rgb_uint8(frame), rotation)
            event = detector.observe(index, rgb)
            if event is not None:
                rows.append(
                    {
                        **asdict(event),
                        "event_time_seconds": event.frame_index / fps,
                        "estimated_period_frames_at_detection": (
                            detector.estimated_period_frames
                        ),
                    }
                )
    finally:
        reader.close()
    pd.DataFrame(rows).to_csv(output / "accepted_online_events.csv", index=False)
    payload = {
        "source": str(source),
        "sample_id": str(args.sample_id),
        "source_read_only": True,
        "selection_mode": "causal_stream",
        "full_video_preanalysis": False,
        "warmup_frame_count": warmup_count,
        "vertex_lookahead_frames": int(config["online_vertex_lookahead_frames"]),
        "prediction_context_delay_frames": int(
            config["prediction_trigger_delay_frames"]
        ),
        "decoded_frame_count": last_index + 1,
        "accepted_event_count": len(rows),
        "accepted_event_frames": [int(row["frame_index"]) for row in rows],
        "accepted_event_trackers": [str(row["tracker"]) for row in rows],
        "strict_event_count": int(detector.strict_event_count),
        "fallback_event_count": int(
            sum("full_lattice_fallback" in str(row["tracker"]) for row in rows)
        ),
        "primary_geometric_vertex_count": len(detector.geometric_vertices),
        "fallback_geometric_vertex_count": len(detector.fallback_geometric_vertices),
        "model_input_roi": asdict(selection.model_input_roi.rect),
        "tracking_roi_not_model_input": asdict(selection.tracking_roi.rect),
        "frame_rotation_clockwise_degrees": rotation,
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
