"""Create the frozen model's keyframe and temporal clip inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_generation.data import PHYSICS_COLUMNS
from analysis.rheed_video_afm_story.clip_cache import (
    luminance_uint8,
    resize_and_pad,
)
from analysis.rheed_video_afm_story.rheed_physics_features import (
    aggregate_temporal,
    frame_physics_features,
    summarize_categories,
)
from rheed2morph.rheed.automatic_roi_keyframe import Rect, _rgb_uint8


def crop_model_frame(
    frame: np.ndarray,
    roi: Rect,
    *,
    output_size: int = 224,
) -> np.ndarray:
    rgb = _rgb_uint8(frame)
    gray = luminance_uint8_array(rgb)
    ys, xs = roi.as_slices()
    crop = gray[ys, xs]
    padded, _, _ = resize_and_pad(crop, int(output_size))
    return padded


def luminance_uint8_array(rgb: np.ndarray) -> np.ndarray:
    """Match the manually curated model input's PIL luminance conversion."""

    # Reuse the frozen helper to avoid a silent change in RGB coefficients.
    from PIL import Image

    return luminance_uint8(Image.fromarray(_rgb_uint8(rgb), mode="RGB"))


def build_model_clip(
    frames: Sequence[np.ndarray],
    roi: Rect,
    *,
    output_size: int = 224,
) -> np.ndarray:
    if len(frames) != 16:
        raise ValueError(f"selected_16 requires exactly 16 frames, got {len(frames)}")
    return np.stack(
        [crop_model_frame(frame, roi, output_size=output_size) for frame in frames],
        axis=0,
    ).astype(np.uint8)


def build_causal_perturbation_clips(
    frames: Sequence[np.ndarray],
    roi: Rect,
    *,
    output_size: int = 224,
) -> tuple[list[str], np.ndarray]:
    """Build target-blind causal-8 views around a confirmed keyframe.

    At the normal ``k+8`` trigger, an 18-frame ring contains ``k-9..k+8``.
    That is sufficient for causal windows ending at ``k-2..k+2`` without
    adding playback latency.
    """

    from analysis.rheed_auto_input_robustness.perturbation import (
        DEFAULT_VIEWS,
        perturb_rect,
    )

    source = list(frames)
    if len(source) != 18:
        raise ValueError(f"causal perturbations require 18 frames, got {len(source)}")
    clips = []
    names = []
    # Ring index 2 is k-7, the first frame of the base causal-8 view.
    for view in DEFAULT_VIEWS:
        start = 2 + int(view.frame_offset)
        selected = source[start : start + 8]
        if len(selected) != 8:
            raise IndexError(f"causal view {view.name} is incomplete at start {start}")
        view_roi = perturb_rect(roi, view)
        clips.append(
            np.stack(
                [
                    crop_model_frame(
                        frame,
                        view_roi,
                        output_size=output_size,
                    )
                    for frame in selected
                ],
                axis=0,
            )
        )
        names.append(view.name)
    return names, np.stack(clips).astype(np.uint8)


def live_physics_row(
    selected_16: np.ndarray,
    *,
    sample_id: str = "__live__",
) -> pd.DataFrame:
    """Extract the exact target-blind RHEED physics schema used by M14i."""

    frames = np.asarray(selected_16, dtype=np.uint8)
    if frames.shape[0] != 16:
        raise ValueError("selected_16 must contain 16 frames")
    variants = {
        "keyframe_1": frames[7:8],
        "causal_8": frames[:8],
        "selected_16": frames,
    }
    row: dict[str, float | str] = {
        "sample_id": str(sample_id),
        "growth_run_id": str(sample_id),
        "video_stage": "live_stream",
    }
    for variant, variant_frames in variants.items():
        extracted = [frame_physics_features(frame) for frame in variant_frames]
        for key, value in aggregate_temporal(extracted).items():
            row[f"{variant}__{key}"] = float(value)
        if variant == "selected_16":
            row["temporal_brightness_drift"] = (
                float(variant_frames[-1].mean() - variant_frames[0].mean()) / 255.0
            )
    row.update(summarize_categories(pd.Series(row)))
    frame = pd.DataFrame([row]).set_index("sample_id", drop=False)
    missing = [column for column in PHYSICS_COLUMNS if column not in frame]
    if missing:
        raise RuntimeError(f"live physics summary is incomplete: {missing}")
    return frame
