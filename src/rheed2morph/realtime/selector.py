"""Replay analysis with role-separated tracking and model-input ROIs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import rankdata

from rheed2morph.rheed.automatic_roi_keyframe import (
    ApertureAnalysis,
    ROIPrediction,
    _source_factory,
    _spot_features,
    _supervised_candidate_rows,
    extract_spot_trajectory,
    predict_roi,
    sample_frames,
)
from rheed2morph.rheed.spot_visibility import (
    SPOT_VISIBILITY_FEATURES,
    analyze_spot_visibility,
    score_deep_visibility_candidates,
    visibility_proxy,
)


ONLINE_CLEAR_MOMENT_GEOMETRY_FEATURES = (
    "spot_x",
    "spot_y",
    "clarity",
    "sharpness",
    "spot_energy",
    "mean_intensity",
    "absolute_contrast",
    "prominence",
    "pre_dx",
    "post_dx",
    "upward_dy",
    "direction_consistent",
    "tracker_front",
    "cross_tracker_distance",
    "cross_tracker_agreement",
    "cross_tracker_direction_support",
)
ONLINE_CLEAR_MOMENT_FEATURES = (
    *ONLINE_CLEAR_MOMENT_GEOMETRY_FEATURES,
    *SPOT_VISIBILITY_FEATURES,
    "visibility_proxy",
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
    selection_mode: str = "offline_precomputed"
    warmup_frame_count: int = 0


def _smoothed_coordinate(
    history: Sequence[dict[str, float | int]],
    name: str,
) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in history], dtype=float)
    return ndimage.gaussian_filter1d(
        ndimage.median_filter(values, size=3, mode="nearest"),
        sigma=1.5,
        mode="nearest",
    )


def causal_candidate_rows(
    history: Sequence[dict[str, float | int]],
    *,
    lookahead_frames: int = 4,
) -> list[dict[str, float | int | bool | str]]:
    """Return physical-vertex candidates confirmable at the current frame.

    A vertex at frame ``k`` is confirmed only after frames through ``k+4``
    have arrived. No descriptor or rank uses a later frame. This bounded
    latency preserves the original left-opening trajectory definition while
    remaining valid for a live stream.
    """

    lookahead = int(lookahead_frames)
    if lookahead < 2:
        raise ValueError("causal vertex confirmation needs at least 2 frames")
    if len(history) < 2 * lookahead + 9:
        return []
    position = len(history) - 1 - lookahead
    if position - lookahead < 0:
        return []

    coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("spot_x", "compact_spot_x"):
        x = _smoothed_coordinate(history, name)
        prominence_floor = max(0.8, float(np.std(x) * 0.10))
        peaks, _ = find_peaks(
            x,
            distance=max(6, 2 * lookahead),
            prominence=prominence_floor,
        )
        coordinates[name] = (x, peaks)

    center = dict(history[position])
    local = history[position - lookahead : position + lookahead + 1]
    result: list[dict[str, float | int | bool | str]] = []
    for tracker, x_name, y_name, other_name in (
        ("front", "spot_x", "spot_y", "compact_spot_x"),
        (
            "compact",
            "compact_spot_x",
            "compact_spot_y",
            "spot_x",
        ),
    ):
        x, peaks = coordinates[x_name]
        if position not in set(map(int, peaks)):
            continue
        y = _smoothed_coordinate(history, y_name)
        left = position - lookahead
        right = position + lookahead
        pre_dx = float(x[position] - x[left])
        post_dx = float(x[position] - x[right])
        upward_dy = float(y[left] - y[right])
        direction_consistent = (
            pre_dx > 0.0 and post_dx > 0.0 and upward_dy > 0.0
        )
        if not direction_consistent:
            continue

        other_peaks = coordinates[other_name][1]
        if len(other_peaks):
            nearest = int(
                other_peaks[
                    int(np.argmin(np.abs(other_peaks - position)))
                ]
            )
            cross_distance = abs(nearest - position)
        else:
            cross_distance = 60
        prominence = float(
            peak_prominences(x, np.asarray([position], dtype=int))[0][0]
        )
        result.append(
            {
                **center,
                "tracker": tracker,
                "spot_x": float(x[position]),
                "spot_y": float(y[position]),
                "clarity": float(
                    np.median([float(row["clarity"]) for row in local])
                ),
                "sharpness": float(
                    np.median([float(row["sharpness"]) for row in local])
                ),
                "spot_energy": float(
                    np.median([float(row["spot_energy"]) for row in local])
                ),
                "mean_intensity": float(
                    np.median(
                        [float(row["mean_intensity"]) for row in local]
                    )
                ),
                "absolute_contrast": float(
                    np.median(
                        [float(row["absolute_contrast"]) for row in local]
                    )
                ),
                "prominence": prominence,
                "pre_dx": pre_dx,
                "post_dx": post_dx,
                "upward_dy": upward_dy,
                "direction_consistent": True,
                "tracker_front": float(tracker == "front"),
                "cross_tracker_distance": float(min(cross_distance, 60)),
                "cross_tracker_agreement": float(
                    np.exp(-cross_distance / 3.0)
                ),
                "cross_tracker_direction_support": float(
                    cross_distance <= lookahead
                ),
            }
        )
    return result


def full_lattice_fallback_eligible(
    candidate: dict[str, float | int | bool | str],
    score: float,
    *,
    minimum_score: float,
    minimum_visibility_proxy: float,
    maximum_shadow_fraction: float,
    minimum_spot_peak_count: float,
    minimum_clarity: float,
) -> bool:
    """Gate a conservative full-lattice candidate without future frames."""

    return bool(
        float(score) >= float(minimum_score)
        and float(candidate["visibility_proxy"])
        >= float(minimum_visibility_proxy)
        and float(candidate["raw_shadow_fraction"])
        <= float(maximum_shadow_fraction)
        and float(candidate["spot_peak_count"])
        >= float(minimum_spot_peak_count)
        and float(candidate["clarity"]) >= float(minimum_clarity)
    )


class CausalClearMomentDetector:
    """Causal multi-cycle detector calibrated against human keyframes."""

    def __init__(
        self,
        *,
        tracking_roi: ROIPrediction,
        bundle_path: str | Path,
        minimum_event_frame: int = 0,
        minimum_score: float | None = None,
        lookahead_frames: int = 4,
        history_frames: int = 41,
        minimum_vertex_separation_frames: int = 8,
        fallback_roi: ROIPrediction | None = None,
        fallback_minimum_score: float = 0.30,
        fallback_minimum_visibility_proxy: float = 1.30,
        fallback_maximum_shadow_fraction: float = 0.20,
        fallback_minimum_spot_peak_count: float = 8.0,
        fallback_minimum_clarity: float = 8.0,
        fallback_confirmation_delay_frames: int = 8,
        fallback_minimum_separation_frames: int = 90,
    ) -> None:
        import joblib

        bundle = joblib.load(Path(bundle_path))
        if int(bundle.get("schema_version", -1)) != 1:
            raise ValueError("Expected an online clear-moment schema-v1 bundle")
        if tuple(bundle["feature_names"]) != ONLINE_CLEAR_MOMENT_FEATURES:
            raise ValueError("Online clear-moment feature schema mismatch")
        self.tracking_roi = tracking_roi
        self.model = bundle["model"]
        self.feature_names = tuple(bundle["feature_names"])
        self.minimum_score = float(
            bundle["minimum_score"]
            if minimum_score is None
            else minimum_score
        )
        self.minimum_visibility_proxy = float(
            bundle["minimum_visibility_proxy"]
        )
        self.maximum_shadow_fraction = float(
            bundle["maximum_shadow_fraction"]
        )
        self.minimum_spot_peak_count = float(
            bundle["minimum_spot_peak_count"]
        )
        self.score_reference = np.sort(
            np.asarray(bundle["score_reference"], dtype=float)
        )
        self.minimum_event_frame = int(minimum_event_frame)
        self.lookahead_frames = int(lookahead_frames)
        self.minimum_vertex_separation_frames = int(
            minimum_vertex_separation_frames
        )
        self.fallback_roi = fallback_roi
        self.fallback_minimum_score = float(fallback_minimum_score)
        self.fallback_minimum_visibility_proxy = float(
            fallback_minimum_visibility_proxy
        )
        self.fallback_maximum_shadow_fraction = float(
            fallback_maximum_shadow_fraction
        )
        self.fallback_minimum_spot_peak_count = float(
            fallback_minimum_spot_peak_count
        )
        self.fallback_minimum_clarity = float(fallback_minimum_clarity)
        self.fallback_confirmation_delay_frames = max(
            self.lookahead_frames,
            int(fallback_confirmation_delay_frames),
        )
        self.fallback_minimum_separation_frames = int(
            fallback_minimum_separation_frames
        )
        self.history: deque[dict[str, float | int]] = deque(
            maxlen=max(int(history_frames), 2 * self.lookahead_frames + 9)
        )
        self.fallback_history: deque[dict[str, float | int]] = deque(
            maxlen=max(int(history_frames), 2 * self.lookahead_frames + 9)
        )
        self.geometric_vertices: list[int] = []
        self.fallback_geometric_vertices: list[int] = []
        self.accepted_events: list[ReplayEvent] = []
        self.strict_event_count = 0
        self.pending_fallbacks: dict[
            int,
            tuple[dict[str, float | int | bool | str], float],
        ] = {}
        self.last_fallback_frame: int | None = None

    @property
    def estimated_period_frames(self) -> float | None:
        if len(self.geometric_vertices) < 3:
            return None
        differences = np.diff(
            np.asarray(self.geometric_vertices[-24:], dtype=float)
        )
        differences = differences[
            (differences >= 10.0) & (differences <= 120.0)
        ]
        if len(differences) < 2:
            return None
        center = float(np.median(differences))
        retained = differences[
            np.abs(differences - center) <= max(5.0, 0.35 * center)
        ]
        return float(np.median(retained if len(retained) else differences))

    def _score_percentile(self, score: float) -> float:
        if not len(self.score_reference):
            return 0.5
        return float(
            np.searchsorted(self.score_reference, score, side="right")
            / len(self.score_reference)
        )

    def _features(
        self,
        frame_index: int,
        frame: np.ndarray,
        roi: ROIPrediction,
    ) -> dict[str, float | int]:
        tracking = _spot_features(frame, roi.rect)
        visibility = analyze_spot_visibility(frame, roi.rect).features
        return {
            "frame_index": int(frame_index),
            **tracking,
            **visibility,
            "visibility_proxy": visibility_proxy(visibility),
        }

    def _best_candidate(
        self,
        history: Sequence[dict[str, float | int]],
    ) -> tuple[dict[str, float | int | bool | str], float] | None:
        import pandas as pd

        candidates = causal_candidate_rows(
            history,
            lookahead_frames=self.lookahead_frames,
        )
        if not candidates:
            return None
        table = pd.DataFrame(candidates)
        scores = np.asarray(
            self.model.predict(table[list(self.feature_names)]),
            dtype=float,
        )
        position = int(np.argmax(scores))
        return candidates[position], float(scores[position])

    def _fallback_event_due(
        self,
        current_frame: int,
    ) -> ReplayEvent | None:
        if self.strict_event_count:
            self.pending_fallbacks.clear()
            return None
        due = [
            frame
            for frame in self.pending_fallbacks
            if frame + self.fallback_confirmation_delay_frames
            <= current_frame
        ]
        if not due:
            return None
        eligible = []
        for frame in due:
            candidate, score = self.pending_fallbacks.pop(frame)
            if (
                self.last_fallback_frame is None
                or frame - self.last_fallback_frame
                >= self.fallback_minimum_separation_frames
            ):
                eligible.append((score, frame, candidate))
        if not eligible:
            return None
        score, candidate_frame, candidate = max(eligible, key=lambda item: item[0])
        self.last_fallback_frame = int(candidate_frame)
        percentile = self._score_percentile(float(score))
        event = ReplayEvent(
            frame_index=int(candidate_frame),
            keyframe_quality=float(0.25 + 0.50 * percentile),
            visibility_rank=percentile,
            selector_score=float(score),
            tracker=f"causal_full_lattice_fallback_{candidate['tracker']}",
            model_input_visibility=percentile,
            refined_from_frame_index=None,
        )
        self.accepted_events.append(event)
        return event

    def observe(
        self,
        frame_index: int,
        frame: np.ndarray,
    ) -> ReplayEvent | None:
        self.history.append(
            self._features(
                frame_index,
                frame,
                self.tracking_roi,
            )
        )
        primary = self._best_candidate(list(self.history))
        if primary is not None:
            candidate, score = primary
            candidate_frame = int(candidate["frame_index"])
            separated = (
                not self.geometric_vertices
                or candidate_frame - self.geometric_vertices[-1]
                >= self.minimum_vertex_separation_frames
            )
            if separated:
                self.geometric_vertices.append(candidate_frame)
                accepted = (
                    candidate_frame >= self.minimum_event_frame
                    and score >= self.minimum_score
                    and float(candidate["visibility_proxy"])
                    >= self.minimum_visibility_proxy
                    and float(candidate["raw_shadow_fraction"])
                    <= self.maximum_shadow_fraction
                    and float(candidate["spot_peak_count"])
                    >= self.minimum_spot_peak_count
                )
                if accepted:
                    self.strict_event_count += 1
                    self.pending_fallbacks.clear()
                    if (
                        self.last_fallback_frame is not None
                        and abs(candidate_frame - self.last_fallback_frame)
                        < self.fallback_minimum_separation_frames
                    ):
                        return None
                    percentile = self._score_percentile(score)
                    event = ReplayEvent(
                        frame_index=candidate_frame,
                        keyframe_quality=float(0.35 + 0.65 * percentile),
                        visibility_rank=percentile,
                        selector_score=score,
                        tracker=f"causal_online_{candidate['tracker']}",
                        model_input_visibility=percentile,
                        refined_from_frame_index=None,
                    )
                    self.accepted_events.append(event)
                    return event

        if self.fallback_roi is not None and not self.strict_event_count:
            self.fallback_history.append(
                self._features(
                    frame_index,
                    frame,
                    self.fallback_roi,
                )
            )
            fallback = self._best_candidate(list(self.fallback_history))
            if fallback is not None:
                candidate, score = fallback
                candidate_frame = int(candidate["frame_index"])
                separated = (
                    not self.fallback_geometric_vertices
                    or candidate_frame - self.fallback_geometric_vertices[-1]
                    >= self.minimum_vertex_separation_frames
                )
                if separated:
                    self.fallback_geometric_vertices.append(candidate_frame)
                    if (
                        candidate_frame >= self.minimum_event_frame
                        and full_lattice_fallback_eligible(
                            candidate,
                            score,
                            minimum_score=self.fallback_minimum_score,
                            minimum_visibility_proxy=(
                                self.fallback_minimum_visibility_proxy
                            ),
                            maximum_shadow_fraction=(
                                self.fallback_maximum_shadow_fraction
                            ),
                            minimum_spot_peak_count=(
                                self.fallback_minimum_spot_peak_count
                            ),
                            minimum_clarity=self.fallback_minimum_clarity,
                        )
                    ):
                        previous = self.pending_fallbacks.get(candidate_frame)
                        if previous is None or score > previous[1]:
                            self.pending_fallbacks[candidate_frame] = (
                                candidate,
                                score,
                            )
        return self._fallback_event_due(int(frame_index))


def initialize_causal_stream(
    source: str | Path,
    warmup_frames: Sequence[np.ndarray],
    *,
    frame_count: int,
    model_input_calibration_path: str | Path,
    full_lattice_calibration_path: str | Path,
    physics_calibration_path: str | Path | None = None,
    frame_rotation_clockwise_degrees: int = 0,
) -> ReplaySelection:
    """Initialize ROI geometry using only an already-arrived warm-up prefix."""

    if len(warmup_frames) < 8:
        raise ValueError("Causal stream ROI initialization needs 8 frames")
    tracking_roi, aperture = predict_roi(
        warmup_frames,
        method="calibrated_safe",
        aspect_ratio=1.54,
        calibrated_scale=0.90,
    )
    model_input_roi, _ = predict_roi(
        warmup_frames,
        method="full_lattice",
        analysis=aperture,
        lattice_calibration=model_input_calibration_path,
    )
    audit_full_lattice_roi, _ = predict_roi(
        warmup_frames,
        method="full_lattice",
        analysis=aperture,
        lattice_calibration=full_lattice_calibration_path,
    )
    physics_roi = model_input_roi
    if physics_calibration_path is not None:
        physics_roi, _ = predict_roi(
            warmup_frames,
            method="full_lattice",
            analysis=aperture,
            lattice_calibration=physics_calibration_path,
        )
    return ReplaySelection(
        source=Path(source).resolve(),
        frame_count=int(frame_count),
        tracking_roi=tracking_roi,
        model_input_roi=model_input_roi,
        physics_roi=physics_roi,
        audit_full_lattice_roi=audit_full_lattice_roi,
        events=(),
        estimated_period_frames=None,
        aperture=aperture,
        frame_rotation_clockwise_degrees=(
            int(frame_rotation_clockwise_degrees) % 360
        ),
        selection_mode="causal_stream",
        warmup_frame_count=len(warmup_frames),
    )


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
            "Applying acquisition-orientation correction: "
            f"CW {int(frame_rotation_clockwise_degrees) % 360}°"
        )
    log("Sampling video and estimating the viewport aperture")
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
    log("Extracting bright-spot trajectories and rotational-vertex candidates")
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
        raise IndexError(
            f"Incomplete candidate-frame decoding; "
            f"{len(missing)} frame(s) missing"
        )
    log(
        "Using the frozen DINOv2-S visibility model to reject shadowed "
        "and blurred candidates"
    )
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
                "Full-lattice local refinement: "
                f"V5 candidate {selected.frame_index} → "
                f"model keyframe {refined_index}"
            )
    frame_count = int(known_count or counted or len(trajectory))
    log(
        f"Analysis complete: {frame_count} frames; detected "
        f"{len(events)} eligible rotational vertex/vertices"
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
