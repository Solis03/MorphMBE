"""Replay analysis with role-separated tracking and model-input ROIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import rankdata

from rheed2morph.rheed.automatic_roi_keyframe import (
    ApertureAnalysis,
    ROIPrediction,
    _source_factory,
    _supervised_candidate_rows,
    extract_spot_trajectory,
    predict_roi,
    sample_frames,
)
from rheed2morph.rheed.spot_visibility import (
    analyze_spot_visibility,
    score_deep_visibility_candidates,
)


@dataclass(frozen=True)
class ReplayEvent:
    frame_index: int
    keyframe_quality: float
    visibility_rank: float
    selector_score: float
    tracker: str
    model_input_visibility: float = float("nan")
    refined_from_frame_index: int | None = None


@dataclass(frozen=True)
class ReplaySelection:
    source: Path
    frame_count: int
    tracking_roi: ROIPrediction
    model_input_roi: ROIPrediction
    physics_roi: ROIPrediction
    audit_full_lattice_roi: ROIPrediction
    events: tuple[ReplayEvent, ...]
    estimated_period_frames: float | None
    aperture: ApertureAnalysis
    frame_rotation_clockwise_degrees: int = 0


def _event_rows(
    candidate_scores: list[dict[str, object]],
    *,
    visibility_gate: float,
    estimated_period: float | None,
) -> list[ReplayEvent]:
    eligible = [
        row for row in candidate_scores if bool(row.get("eligible", False))
    ]
    if not eligible and candidate_scores:
        eligible = [max(candidate_scores, key=lambda row: float(row["score"]))]
    if not eligible:
        return []
    ordered = sorted(eligible, key=lambda row: int(row["frame_index"]))
    merge_distance = 6
    if estimated_period is not None and np.isfinite(estimated_period):
        merge_distance = int(np.clip(round(0.18 * estimated_period), 3, 10))
    clusters: list[list[dict[str, object]]] = []
    for row in ordered:
        if (
            not clusters
            or int(row["frame_index"])
            - int(clusters[-1][-1]["frame_index"])
            > merge_distance
        ):
            clusters.append([row])
        else:
            clusters[-1].append(row)

    scores = np.asarray([float(row["score"]) for row in eligible], dtype=float)
    low = float(np.min(scores))
    high = float(np.max(scores))
    result: list[ReplayEvent] = []
    for cluster in clusters:
        selected = max(
            cluster,
            key=lambda row: (
                float(row["score"]),
                float(row["spot_visibility_rank"]),
            ),
        )
        score = float(selected["score"])
        score_rank = 1.0 if high <= low else (score - low) / (high - low)
        visibility = float(selected["spot_visibility_rank"])
        visibility_support = np.clip(
            (visibility - visibility_gate)
            / max(1.0 - visibility_gate, 1e-6),
            0.0,
            1.0,
        )
        quality = float(
            np.clip(0.35 + 0.40 * visibility_support + 0.25 * score_rank, 0, 1)
        )
        result.append(
            ReplayEvent(
                frame_index=int(selected["frame_index"]),
                keyframe_quality=quality,
                visibility_rank=visibility,
                selector_score=score,
                tracker=str(selected["tracker"]),
            )
        )
    return result


def _lattice_vertex_scores(
    feature_rows: list[dict[str, float]],
) -> np.ndarray:
    """Rank a local phase neighborhood by complete, column-aligned spots.

    A few isolated bright spots can score highly under a generic visibility
    proxy.  At the requested rotation vertex, however, the complete RHEED
    family is compact horizontally, column aligned, and rich in localized
    high-frequency energy.  Ranking only within one physical-cycle
    neighborhood makes the score exposure invariant.
    """

    if not feature_rows:
        return np.empty(0, dtype=float)

    def ascending(name: str) -> np.ndarray:
        values = np.asarray(
            [float(row[name]) for row in feature_rows],
            dtype=float,
        )
        return rankdata(values, method="average") / len(values)

    low_horizontal_spread = 1.0 - ascending("spot_horizontal_spread")
    haze_rejection = 1.0 - ascending("haze_dominance")
    return (
        0.25 * ascending("spot_peak_top8_mass")
        + 0.25 * ascending("spot_column_alignment")
        + 0.15 * low_horizontal_spread
        + 0.10 * ascending("raw_std")
        + 0.10 * ascending("spot_energy_concentration")
        + 0.10 * ascending("spot_peak_count")
        + 0.05 * haze_rejection
    )


def analyze_replay(
    source: str | Path,
    *,
    deep_visibility_ranker_path: str | Path,
    model_input_calibration_path: str | Path,
    full_lattice_calibration_path: str | Path,
    physics_calibration_path: str | Path | None = None,
    foundation_cache_dir: str | Path | None = None,
    device: str | None = None,
    roi_sample_count: int = 48,
    minimum_event_quality: float = 0.0,
    event_policy: str = "best_visible_cycle",
    refinement_period_fraction: float = 0.45,
    frame_rotation_clockwise_degrees: int = 0,
    progress: Callable[[str], None] | None = None,
) -> ReplaySelection:
    """Analyze a complete recording, then emit causal replay events.

    The analysis pass is intentionally separate from playback.  It provides a
    deterministic simulation of the future streaming pipeline while the
    replay itself only triggers a prediction after the eight post-keyframe
    frames required by the frozen selected-16 morphology view have arrived.
    """

    source_path = Path(source).resolve()
    log = progress or (lambda _: None)
    if int(frame_rotation_clockwise_degrees) % 360:
        log(
            "应用采集方向校正："
            f"顺时针 {int(frame_rotation_clockwise_degrees) % 360}°"
        )
    log("采样视频并估计目镜孔径")
    sampled, counted = sample_frames(
        source_path,
        maximum=int(roi_sample_count),
        rotation_clockwise_degrees=frame_rotation_clockwise_degrees,
    )
    tracking_roi, aperture = predict_roi(
        sampled,
        method="calibrated_safe",
        aspect_ratio=1.54,
        calibrated_scale=0.90,
    )
    # The V5 rectangle is deliberately compact because it was fitted for
    # trajectory extraction.  It must never be used as morphology input.
    model_input_roi, _ = predict_roi(
        sampled,
        method="full_lattice",
        analysis=aperture,
        lattice_calibration=model_input_calibration_path,
    )
    audit_full_lattice_roi, _ = predict_roi(
        sampled,
        method="full_lattice",
        analysis=aperture,
        lattice_calibration=full_lattice_calibration_path,
    )
    physics_roi = model_input_roi
    if physics_calibration_path is not None:
        physics_roi, _ = predict_roi(
            sampled,
            method="full_lattice",
            analysis=aperture,
            lattice_calibration=physics_calibration_path,
        )
    log("提取亮斑轨迹与旋转顶点候选")
    factory, known_count, _ = _source_factory(
        source_path,
        rotation_clockwise_degrees=frame_rotation_clockwise_degrees,
    )
    trajectory = extract_spot_trajectory(factory(), tracking_roi.rect)
    candidates, periods = _supervised_candidate_rows(trajectory)
    required = {int(candidate["frame_index"]) for candidate in candidates}
    frames: dict[int, np.ndarray] = {}
    for frame_index, frame in factory():
        if frame_index in required:
            frames[frame_index] = frame
        if len(frames) == len(required):
            break
    if len(frames) != len(required):
        missing = sorted(required - set(frames))
        raise IndexError(f"候选帧解码不完整，缺少 {len(missing)} 帧")
    log("使用冻结 DINOv2-S 可见度模型排除阴影和模糊候选")
    scored = score_deep_visibility_candidates(
        candidates,
        frames,
        tracking_roi.rect,
        deep_visibility_ranker_path,
        foundation_cache_dir=foundation_cache_dir,
        device=device,
    )
    finite_periods = [
        float(value)
        for value in periods.values()
        if value is not None and np.isfinite(value)
    ]
    estimated_period = (
        float(np.median(finite_periods)) if finite_periods else None
    )
    gate = float(scored.get("visibility_gate", 0.0))
    events = _event_rows(
        list(scored["candidate_scores"]),
        visibility_gate=gate,
        estimated_period=estimated_period,
    )
    retained = [
        event
        for event in events
        if event.keyframe_quality >= float(minimum_event_quality)
    ]
    if retained:
        events = retained
    elif events:
        events = [max(events, key=lambda event: event.keyframe_quality)]

    if event_policy not in {"best_visible_cycle", "all_eligible_cycles"}:
        raise ValueError(f"Unknown replay event policy: {event_policy}")
    if events and event_policy == "best_visible_cycle":
        # V5 chooses the best physical rotation-cycle candidate.  Its compact
        # tracking crop is excellent for motion, but not for judging whether
        # the complete diffraction family is at peak visibility.  Refine the
        # winning candidate locally in the actual model-input ROI.
        selected = max(
            events,
            key=lambda event: (
                event.selector_score,
                event.visibility_rank,
            ),
        )
        radius = 8
        if estimated_period is not None and np.isfinite(estimated_period):
            radius = int(
                np.clip(
                    round(
                        float(refinement_period_fraction)
                        * estimated_period
                    ),
                    4,
                    24,
                )
            )
        frame_count = int(known_count or counted or len(trajectory))
        lower = max(7, selected.frame_index - radius)
        upper = min(frame_count - 9, selected.frame_index + radius)
        neighborhood: dict[int, np.ndarray] = {}
        for frame_index, frame in factory():
            if lower <= frame_index <= upper:
                neighborhood[frame_index] = frame
            if frame_index >= upper:
                break
        if neighborhood:
            local_indices = sorted(neighborhood)
            local_features = [
                analyze_spot_visibility(
                    neighborhood[frame_index],
                    model_input_roi.rect,
                ).features
                for frame_index in local_indices
            ]
            lattice_scores = _lattice_vertex_scores(local_features)
            local_visibility = {
                frame_index: float(score)
                for frame_index, score in zip(
                    local_indices,
                    lattice_scores,
                )
            }
            refined_index = max(
                local_visibility,
                key=lambda frame_index: (
                    local_visibility[frame_index],
                    -abs(frame_index - selected.frame_index),
                ),
            )
            visibility = local_visibility[refined_index]
            selection_confidence = float(
                np.clip(scored.get("confidence", 0.0), 0.0, 1.0)
            )
            quality = float(
                np.sqrt(
                    max(selection_confidence, 0.0)
                    * np.clip(visibility, 0.0, 1.0)
                )
            )
            events = [
                ReplayEvent(
                    frame_index=int(refined_index),
                    keyframe_quality=quality,
                    visibility_rank=1.0,
                    selector_score=selected.selector_score,
                    tracker=selected.tracker,
                    model_input_visibility=visibility,
                    refined_from_frame_index=selected.frame_index,
                )
            ]
            log(
                "完整点阵局部细化："
                f"V5 候选 {selected.frame_index} → "
                f"模型关键帧 {refined_index}"
            )
    frame_count = int(known_count or counted or len(trajectory))
    log(
        f"分析完成：{frame_count} 帧，检测到 {len(events)} 个可用旋转顶点"
    )
    return ReplaySelection(
        source=source_path,
        frame_count=frame_count,
        tracking_roi=tracking_roi,
        model_input_roi=model_input_roi,
        physics_roi=physics_roi,
        audit_full_lattice_roi=audit_full_lattice_roi,
        events=tuple(events),
        estimated_period_frames=estimated_period,
        aperture=aperture,
        frame_rotation_clockwise_degrees=(
            int(frame_rotation_clockwise_degrees) % 360
        ),
    )
