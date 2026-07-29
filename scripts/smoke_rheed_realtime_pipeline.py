#!/usr/bin/env python3
"""Run one raw-video event through ROI, keyframe, M15b and M12a."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from rheed2morph.realtime.clips import (
    build_causal_perturbation_clips,
    build_model_clip,
)
from rheed2morph.realtime.model import RealtimeMorphologyPredictor
from rheed2morph.realtime.selector import analyze_replay
from rheed2morph.rheed.automatic_roi_keyframe import iter_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--config", default="configs/rheed_realtime_ui.json"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rheed_realtime_ui/headless_smoke",
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
    started = time.perf_counter()
    selection = analyze_replay(
        source,
        deep_visibility_ranker_path=(
            repository / config["deep_visibility_ranker"]
        ),
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
        foundation_cache_dir=(
            repository / config["foundation_cache_dir"]
        ),
        device=config.get("selector_device"),
        roi_sample_count=int(config["roi_sample_count"]),
        minimum_event_quality=float(config["minimum_keyframe_quality"]),
        event_policy=str(
            config.get("replay_event_policy", "best_visible_cycle")
        ),
        refinement_period_fraction=float(
            config.get("keyframe_refinement_period_fraction", 0.45)
        ),
        progress=lambda message: print(f"[selector] {message}", flush=True),
    )
    event = max(selection.events, key=lambda item: item.selector_score)
    # Match ReplayWorker exactly. At trigger k+8, its 18-frame ring contains
    # k-9..k+8; selected-16 is ring[2:] and the extra two frames provide the
    # causal windows ending at k-2 and k-1.
    first = event.frame_index - 9
    last = event.frame_index + 8
    frames: list[np.ndarray] = []
    keyframe: np.ndarray | None = None
    for index, frame in iter_video_frames(source):
        if first <= index <= last:
            frames.append(frame)
        if index == event.frame_index:
            keyframe = frame
        if index >= last:
            break
    if len(frames) != 18 or keyframe is None:
        raise RuntimeError("could not decode the selected temporal window")
    selected_frames = frames[2:]
    clip = build_model_clip(
        selected_frames,
        selection.model_input_roi.rect,
        output_size=int(config["model_image_size"]),
    )
    physics_clip = build_model_clip(
        selected_frames,
        selection.physics_roi.rect,
        output_size=int(config["model_image_size"]),
    )
    view_names, causal_views = build_causal_perturbation_clips(
        frames,
        selection.model_input_roi.rect,
        output_size=int(config["model_image_size"]),
    )
    predictor = RealtimeMorphologyPredictor.from_path(
        repository / config["deployment_bundle"],
        device=str(config.get("prediction_device", "auto")),
    )
    prediction = predictor.predict(
        clip,
        physics_selected_16=physics_clip,
        causal_view_names=view_names,
        causal_8_views=causal_views,
        estimated_period_frames=selection.estimated_period_frames,
        keyframe_quality=event.keyframe_quality,
        seed=int(args.sample_id) * 1_000_003 + event.frame_index * 97,
    )
    np.savez_compressed(
        output / "prediction.npz",
        selected_16=clip,
        physics_selected_16=physics_clip,
        causal_8_views=causal_views,
        causal_view_names=np.asarray(view_names),
        keyframe_rgb=keyframe,
        unit_shape=prediction.unit_shape,
        height_nm=prediction.height_nm,
        predicted_rq_nm=np.asarray(prediction.rq.value),
        predicted_fsmi_nm=np.asarray(prediction.fsmi.value),
        combined_confidence=np.asarray(prediction.combined_confidence),
        retrieval_at_inference=np.asarray(False),
    )
    payload = {
        "source": str(source),
        "sample_id": str(args.sample_id),
        "frame_count": selection.frame_count,
        "model_input_roi": asdict(selection.model_input_roi.rect),
        "physics_feature_roi_not_generator_input": asdict(
            selection.physics_roi.rect
        ),
        "internal_tracking_roi_not_model_input": asdict(
            selection.tracking_roi.rect
        ),
        "conservative_audit_roi_not_model_input": asdict(
            selection.audit_full_lattice_roi.rect
        ),
        "estimated_period_frames": selection.estimated_period_frames,
        "retained_event_count": len(selection.events),
        "events": [asdict(item) for item in selection.events],
        "selected_smoke_event": asdict(event),
        "prediction": {
            "model_id": prediction.model_id,
            "Rq_nm": asdict(prediction.rq),
            "FSMI_nm": asdict(prediction.fsmi),
            "model_confidence": prediction.model_confidence,
            "keyframe_quality": prediction.keyframe_quality,
            "combined_confidence": prediction.combined_confidence,
            "generated_rq_nm": prediction.generated_rq_nm,
            "inference_seconds": prediction.inference_seconds,
            "retrieval_at_inference": False,
        },
        "total_seconds": time.perf_counter() - started,
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.5), constrained_layout=True
    )
    axes[0].imshow(keyframe)
    model_input = selection.model_input_roi.rect
    axes[0].add_patch(
        Rectangle(
            (model_input.x, model_input.y),
            model_input.width,
            model_input.height,
            fill=False,
            color="#00d2a0",
            linewidth=2,
            label="model input / full lattice",
        )
    )
    physics_roi = selection.physics_roi.rect
    axes[0].add_patch(
        Rectangle(
            (physics_roi.x, physics_roi.y),
            physics_roi.width,
            physics_roi.height,
            fill=False,
            color="#CC79A7",
            linestyle="--",
            linewidth=1.6,
            label="physics feature diagnostic ROI",
        )
    )
    axes[0].set_title(
        f"RHEED keyframe {event.frame_index}\n"
        f"input quality {event.keyframe_quality:.2f}"
    )
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_axis_off()
    axes[1].imshow(clip[7], cmap="gray")
    axes[1].set_title("Model keyframe ROI\n(selected-16 position 8)")
    axes[1].set_axis_off()
    image = axes[2].imshow(
        prediction.height_nm,
        cmap="afmhot",
        extent=(0, 1, 1, 0),
    )
    axes[2].set(
        xlabel="x (µm)",
        ylabel="y (µm)",
        title=(
            f"Generated AFM · Rq {prediction.rq.value:.2f} nm\n"
            f"FSMI {prediction.fsmi.value:.2f} nm · "
            f"model confidence {prediction.model_confidence:.0%}"
        ),
    )
    figure.colorbar(image, ax=axes[2], label="Relative height (nm)")
    figure.savefig(output / "rheed_to_generated_afm_panel.png", dpi=210)
    figure.savefig(output / "rheed_to_generated_afm_panel.pdf")
    plt.close(figure)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
