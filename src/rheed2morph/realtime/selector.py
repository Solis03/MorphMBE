"""Replay analysis using the frozen V5 keyframe and V7 ROI models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

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
    score_deep_visibility_candidates,
)


@dataclass(frozen=True)
class ReplayEvent:
    frame_index: int
    keyframe_quality: float
    visibility_rank: float
    selector_score: float
    tracker: str


@dataclass(frozen=True)
class ReplaySelection:
    source: Path
    frame_count: int
    tracking_roi: ROIPrediction
    display_roi: ROIPrediction
    events: tuple[ReplayEvent, ...]
    estimated_period_frames: float | None
    aperture: ApertureAnalysis


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


def analyze_replay(
    source: str | Path,
    *,
    deep_visibility_ranker_path: str | Path,
    full_lattice_calibration_path: str | Path,
    foundation_cache_dir: str | Path | None = None,
    device: str | None = None,
    roi_sample_count: int = 48,
    minimum_event_quality: float = 0.0,
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
    log("采样视频并估计目镜孔径")
    sampled, counted = sample_frames(source_path, maximum=int(roi_sample_count))
    tracking_roi, aperture = predict_roi(
        sampled,
        method="calibrated_safe",
        aspect_ratio=1.54,
        calibrated_scale=0.90,
    )
    log("提取亮斑轨迹与旋转顶点候选")
    factory, known_count, _ = _source_factory(source_path)
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
    display_roi, _ = predict_roi(
        sampled,
        method="full_lattice",
        analysis=aperture,
        lattice_calibration=full_lattice_calibration_path,
    )
    frame_count = int(known_count or counted or len(trajectory))
    log(
        f"分析完成：{frame_count} 帧，检测到 {len(events)} 个可用旋转顶点"
    )
    return ReplaySelection(
        source=source_path,
        frame_count=frame_count,
        tracking_roi=tracking_roi,
        display_roi=display_roi,
        events=tuple(events),
        estimated_period_frames=estimated_period,
        aperture=aperture,
    )
