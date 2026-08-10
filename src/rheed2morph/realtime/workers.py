"""Background replay and prediction workers for the Qt application."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PySide6.QtCore import QThread, Signal

from rheed2morph.rheed.automatic_roi_keyframe import _rgb_uint8
from rheed2morph.rheed.orientation import (
    rotate_frame_clockwise,
    rotation_for_sample,
    value_for_sample,
)

from .clips import build_causal_perturbation_clips, build_model_clip
from .model import MorphologyPrediction, RealtimeMorphologyPredictor
from .selector import (
    CausalClearMomentDetector,
    ReplayEvent,
    analyze_replay,
    initialize_causal_stream,
)


@dataclass(frozen=True)
class PredictionJob:
    sample_id: str
    source: Path
    event: ReplayEvent
    event_time_seconds: float
    selected_16: np.ndarray
    physics_selected_16: np.ndarray | None = None
    causal_view_names: tuple[str, ...] = ()
    causal_8_views: np.ndarray | None = None
    estimated_period_frames: float | None = None


@dataclass(frozen=True)
class PredictionResult:
    job: PredictionJob
    prediction: MorphologyPrediction


class PredictionWorker(QThread):
    ready = Signal(str)
    log = Signal(str)
    result = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        bundle_path: str | Path,
        device: str = "auto",
        queue_capacity: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.bundle_path = Path(bundle_path)
        self.device = str(device)
        self.jobs: queue.Queue[PredictionJob | None] = queue.Queue(
            maxsize=max(0, int(queue_capacity))
        )

    def submit(self, job: PredictionJob) -> bool:
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.jobs.put_nowait(None)

    def run(self) -> None:
        try:
            self.log.emit(
                "Loading the configured Sq/FSMI heads, non-retrieval AFM "
                "generator, and R3D-18"
            )
            predictor = RealtimeMorphologyPredictor.from_path(
                self.bundle_path,
                device=self.device,
            )
            self.ready.emit(predictor.bundle.model_id)
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                self.log.emit(
                    f"Frame {job.event.frame_index}: running causal-8 + "
                    "selected-16 inference"
                )
                seed = (
                    int(str(job.sample_id).lstrip("Nn")) * 1_000_003
                    + int(job.event.frame_index) * 97
                ) % (2**31 - 1)
                prediction = predictor.predict(
                    job.selected_16,
                    physics_selected_16=job.physics_selected_16,
                    causal_view_names=list(job.causal_view_names),
                    causal_8_views=job.causal_8_views,
                    estimated_period_frames=job.estimated_period_frames,
                    keyframe_quality=job.event.keyframe_quality,
                    seed=seed,
                )
                self.result.emit(PredictionResult(job=job, prediction=prediction))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ReplayWorker(QThread):
    log = Signal(str)
    prepared = Signal(object, float)
    frame = Signal(object, int, float, bool)
    prediction_job = Signal(object)
    stream_summary = Signal(int, int)
    completed = Signal()
    failed = Signal(str)

    def __init__(
        self,
        *,
        sample_id: str,
        source: str | Path,
        config: dict,
        playback_ratio: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.sample_id = str(sample_id)
        self.source = Path(source).resolve()
        self.config = dict(config)
        self.playback_ratio = float(max(playback_ratio, 0.25))
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def _wait_if_paused(self) -> bool:
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.03)
        return self._stop_event.is_set()

    @staticmethod
    def _video_metadata(reader) -> tuple[float, int]:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 30.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        frame_count = 0
        raw_count = metadata.get("nframes")
        if raw_count is not None:
            try:
                if np.isfinite(float(raw_count)):
                    frame_count = int(round(float(raw_count)))
            except (TypeError, ValueError):
                frame_count = 0
        if frame_count <= 0:
            duration = float(metadata.get("duration") or 0.0)
            if np.isfinite(duration) and duration > 0:
                frame_count = int(round(duration * fps))
        return fps, frame_count

    def _run_causal_stream(self) -> None:
        repository = Path(self.config["repository_root"])
        rotation = rotation_for_sample(
            self.config.get("rheed_rotation_clockwise_degrees_by_sample"),
            self.sample_id,
        )
        reader = imageio.get_reader(str(self.source), "ffmpeg")
        fps, metadata_frame_count = self._video_metadata(reader)
        warmup_count = max(
            8,
            int(self.config.get("online_roi_warmup_frames", 48)),
        )
        self.log.emit(
            f"Causal ROI warm-up: buffering the first {warmup_count} "
            "arrived frames only; no full-video pre-analysis"
        )
        warmup_frames: list[np.ndarray] = []
        iterator = reader.iter_data()
        try:
            for _ in range(warmup_count):
                warmup_frames.append(
                    rotate_frame_clockwise(
                        _rgb_uint8(next(iterator)),
                        rotation,
                    )
                )
        except StopIteration:
            reader.close()
            raise RuntimeError(
                "Video ended before causal ROI warm-up completed"
            ) from None
        selection = initialize_causal_stream(
            self.source,
            warmup_frames,
            frame_count=max(metadata_frame_count, warmup_count),
            model_input_calibration_path=(
                repository / self.config["model_input_roi_calibration"]
            ),
            full_lattice_calibration_path=(
                repository / self.config["full_lattice_roi_calibration"]
            ),
            physics_calibration_path=(
                repository / self.config["physics_roi_calibration"]
                if self.config.get("physics_roi_calibration")
                else None
            ),
            frame_rotation_clockwise_degrees=rotation,
        )
        detector = CausalClearMomentDetector(
            tracking_roi=selection.tracking_roi,
            fallback_roi=(
                selection.model_input_roi
                if self.config.get(
                    "online_full_lattice_fallback_enabled",
                    True,
                )
                else None
            ),
            bundle_path=(repository / self.config["online_clear_moment_detector"]),
            minimum_event_frame=warmup_count,
            minimum_score=(
                float(self.config["online_minimum_clear_score"])
                if self.config.get("online_minimum_clear_score") is not None
                else None
            ),
            lookahead_frames=int(self.config.get("online_vertex_lookahead_frames", 4)),
            history_frames=int(self.config.get("online_detector_history_frames", 41)),
            minimum_vertex_separation_frames=int(
                self.config.get(
                    "online_minimum_vertex_separation_frames",
                    8,
                )
            ),
            fallback_minimum_score=float(
                self.config.get(
                    "online_fallback_minimum_clear_score",
                    0.30,
                )
            ),
            fallback_minimum_visibility_proxy=float(
                self.config.get(
                    "online_fallback_minimum_visibility_proxy",
                    1.30,
                )
            ),
            fallback_maximum_shadow_fraction=float(
                self.config.get(
                    "online_fallback_maximum_shadow_fraction",
                    0.20,
                )
            ),
            fallback_minimum_spot_peak_count=float(
                self.config.get(
                    "online_fallback_minimum_spot_peak_count",
                    8.0,
                )
            ),
            fallback_minimum_clarity=float(
                self.config.get(
                    "online_fallback_minimum_clarity",
                    8.0,
                )
            ),
            fallback_confirmation_delay_frames=int(
                self.config.get(
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
                            self.config.get(
                                "online_fallback_minimum_separation_seconds",
                                3.0,
                            )
                        )
                    )
                ),
            ),
        )
        override = value_for_sample(
            self.config.get("replay_keyframe_override_by_sample"),
            self.sample_id,
        )
        if override is not None:
            self.log.emit(
                "Archived single-keyframe locks are disabled in causal-stream "
                "mode; acquisition-orientation correction remains active"
            )
        if rotation:
            self.log.emit(
                f"Applying acquisition-orientation correction online: CW {rotation}°"
            )
        for index, frame in enumerate(warmup_frames):
            detector.observe(index, frame)
        if self._stop_event.is_set():
            reader.close()
            return

        self.prepared.emit(selection, fps)
        playback_fps = fps / self.playback_ratio
        maximum_display = float(self.config.get("maximum_display_fps", 24.0))
        display_stride = max(1, int(np.ceil(playback_fps / maximum_display)))
        interval = self.playback_ratio / fps
        ring: deque[np.ndarray] = deque(warmup_frames[-18:], maxlen=18)
        pending: dict[int, tuple[ReplayEvent, float | None]] = {}
        detected_count = 0
        submitted_count = 0
        start = time.perf_counter()
        pause_started: float | None = None
        paused_total = 0.0
        self.log.emit(
            "Online detector active: every accepted clear rotational vertex "
            "will trigger one prediction after the required 8-frame context"
        )
        try:
            for index, frame in enumerate(iterator, start=warmup_count):
                if self._stop_event.is_set():
                    return
                if self._pause_event.is_set() and pause_started is None:
                    pause_started = time.perf_counter()
                if self._wait_if_paused():
                    return
                if pause_started is not None:
                    paused_total += time.perf_counter() - pause_started
                    pause_started = None
                rgb = rotate_frame_clockwise(_rgb_uint8(frame), rotation)
                ring.append(rgb.copy())
                event = detector.observe(index, rgb)
                if event is not None:
                    detected_count += 1
                    trigger = event.frame_index + int(
                        self.config.get("prediction_trigger_delay_frames", 8)
                    )
                    pending[trigger] = (
                        event,
                        detector.estimated_period_frames,
                    )
                    if "full_lattice_fallback" in event.tracker:
                        route = "full-lattice safety fallback"
                    else:
                        route = "strict tracking path"
                    self.log.emit(
                        f"Online clear moment #{detected_count}: keyframe "
                        f"{event.frame_index}, score={event.selector_score:.3f} "
                        f"via {route}; prediction scheduled at frame {trigger}"
                    )

                pending_item = pending.pop(index, None)
                is_event = pending_item is not None
                if index % display_stride == 0 or is_event:
                    self.frame.emit(
                        rgb,
                        index,
                        index / fps,
                        is_event,
                    )
                if pending_item is not None:
                    event, period_at_detection = pending_item
                    if len(ring) != 18:
                        self.log.emit(
                            f"Frame {event.frame_index}: temporal buffer "
                            "incomplete; event skipped"
                        )
                    else:
                        ring_frames = list(ring)
                        selected_frames = ring_frames[2:]
                        clip = build_model_clip(
                            selected_frames,
                            selection.model_input_roi.rect,
                            output_size=int(self.config.get("model_image_size", 224)),
                        )
                        physics_clip = build_model_clip(
                            selected_frames,
                            selection.physics_roi.rect,
                            output_size=int(self.config.get("model_image_size", 224)),
                        )
                        view_names, causal_views = build_causal_perturbation_clips(
                            ring_frames,
                            selection.model_input_roi.rect,
                            output_size=int(
                                self.config.get(
                                    "model_image_size",
                                    224,
                                )
                            ),
                        )
                        submitted_count += 1
                        self.log.emit(
                            f"Frame {event.frame_index}: full-lattice context "
                            f"ready; submitting prediction #{submitted_count}"
                        )
                        self.prediction_job.emit(
                            PredictionJob(
                                sample_id=self.sample_id,
                                source=self.source,
                                event=event,
                                event_time_seconds=event.frame_index / fps,
                                selected_16=clip,
                                physics_selected_16=physics_clip,
                                causal_view_names=tuple(view_names),
                                causal_8_views=causal_views,
                                estimated_period_frames=period_at_detection,
                            )
                        )

                playback_position = index - warmup_count + 1
                target = start + paused_total + playback_position * interval
                while not self._stop_event.is_set() and time.perf_counter() < target:
                    time.sleep(
                        min(
                            0.015,
                            max(target - time.perf_counter(), 0.0),
                        )
                    )
        finally:
            reader.close()
        self.log.emit(
            f"Causal replay complete: detected {detected_count} clear "
            f"moments and submitted {submitted_count} predictions"
        )
        self.stream_summary.emit(detected_count, submitted_count)
        self.completed.emit()

    def run(self) -> None:
        try:
            if (
                str(
                    self.config.get(
                        "replay_detection_mode",
                        "precomputed",
                    )
                )
                == "causal_stream"
            ):
                self._run_causal_stream()
                return
            repository = Path(self.config["repository_root"])
            rotation = rotation_for_sample(
                self.config.get("rheed_rotation_clockwise_degrees_by_sample"),
                self.sample_id,
            )
            selection = analyze_replay(
                self.source,
                deep_visibility_ranker_path=(
                    repository / self.config["deep_visibility_ranker"]
                ),
                model_input_calibration_path=(
                    repository / self.config["model_input_roi_calibration"]
                ),
                full_lattice_calibration_path=(
                    repository / self.config["full_lattice_roi_calibration"]
                ),
                physics_calibration_path=(
                    repository / self.config["physics_roi_calibration"]
                    if self.config.get("physics_roi_calibration")
                    else None
                ),
                foundation_cache_dir=(repository / self.config["foundation_cache_dir"]),
                device=self.config.get("selector_device"),
                roi_sample_count=int(self.config["roi_sample_count"]),
                minimum_event_quality=float(
                    self.config.get("minimum_keyframe_quality", 0.0)
                ),
                event_policy=str(
                    self.config.get(
                        "replay_event_policy",
                        "best_visible_cycle",
                    )
                ),
                refinement_period_fraction=float(
                    self.config.get(
                        "keyframe_refinement_period_fraction",
                        0.45,
                    )
                ),
                frame_rotation_clockwise_degrees=rotation,
                progress=self.log.emit,
            )
            override = value_for_sample(
                self.config.get("replay_keyframe_override_by_sample"),
                self.sample_id,
            )
            if (
                override is not None
                and override.get("source_name")
                and self.source.name != str(override["source_name"])
            ):
                self.log.emit(
                    "Skipped the archive-video-specific vertex lock: "
                    f"current video {self.source.name!r} does not match "
                    "the configured source"
                )
                override = None
            if override is not None:
                if not selection.events:
                    raise RuntimeError(
                        "cannot apply keyframe override without a selector event"
                    )
                source_event = selection.events[0]
                event = replace(
                    source_event,
                    frame_index=int(override["frame_index"]),
                    keyframe_quality=float(
                        override.get(
                            "keyframe_quality",
                            source_event.keyframe_quality,
                        )
                    ),
                    refined_from_frame_index=source_event.frame_index,
                    tracker="frozen_raw_coordinate_v5",
                )
                selection = replace(selection, events=(event,))
                self.log.emit(
                    "Orientation-correction ablation lock: "
                    f"the model reads CW {rotation}° frames while retaining "
                    f"the target-blind V5 temporal vertex at frame "
                    f"{event.frame_index}"
                )
            if self._stop_event.is_set():
                return
            reader = imageio.get_reader(str(self.source), "ffmpeg")
            metadata = reader.get_meta_data()
            fps = float(metadata.get("fps") or 30.0)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            self.prepared.emit(selection, fps)
            playback_fps = fps / self.playback_ratio
            maximum_display = float(self.config.get("maximum_display_fps", 24.0))
            display_stride = max(1, int(np.ceil(playback_fps / maximum_display)))
            interval = self.playback_ratio / fps
            events = {
                event.frame_index
                + int(self.config.get("prediction_trigger_delay_frames", 8)): event
                for event in selection.events
            }
            submitted_count = 0
            ring: deque[np.ndarray] = deque(maxlen=18)
            start = time.perf_counter()
            pause_started: float | None = None
            paused_total = 0.0
            try:
                for index, frame in enumerate(reader):
                    if self._stop_event.is_set():
                        return
                    if self._pause_event.is_set() and pause_started is None:
                        pause_started = time.perf_counter()
                    if self._wait_if_paused():
                        return
                    if pause_started is not None:
                        paused_total += time.perf_counter() - pause_started
                        pause_started = None
                    rgb = rotate_frame_clockwise(
                        _rgb_uint8(frame),
                        rotation,
                    )
                    ring.append(rgb.copy())
                    event = events.get(index)
                    is_event = event is not None
                    if index % display_stride == 0 or is_event:
                        self.frame.emit(
                            rgb,
                            index,
                            index / fps,
                            is_event,
                        )
                    if event is not None:
                        if len(ring) != 18:
                            self.log.emit(
                                f"Frame {event.frame_index}: temporal buffer "
                                "incomplete; event skipped"
                            )
                        else:
                            ring_frames = list(ring)
                            selected_frames = ring_frames[2:]
                            clip = build_model_clip(
                                selected_frames,
                                selection.model_input_roi.rect,
                                output_size=int(
                                    self.config.get("model_image_size", 224)
                                ),
                            )
                            physics_clip = build_model_clip(
                                selected_frames,
                                selection.physics_roi.rect,
                                output_size=int(
                                    self.config.get("model_image_size", 224)
                                ),
                            )
                            view_names, causal_views = build_causal_perturbation_clips(
                                ring_frames,
                                selection.model_input_roi.rect,
                                output_size=int(
                                    self.config.get("model_image_size", 224)
                                ),
                            )
                            self.log.emit(
                                f"Frame {event.frame_index}: full-lattice "
                                "vertex confirmed; collected 16 frames and "
                                "confidence perturbation views"
                            )
                            self.prediction_job.emit(
                                PredictionJob(
                                    sample_id=self.sample_id,
                                    source=self.source,
                                    event=event,
                                    event_time_seconds=(event.frame_index / fps),
                                    selected_16=clip,
                                    physics_selected_16=physics_clip,
                                    causal_view_names=tuple(view_names),
                                    causal_8_views=causal_views,
                                    estimated_period_frames=(
                                        selection.estimated_period_frames
                                    ),
                                )
                            )
                            submitted_count += 1
                    target = start + paused_total + (index + 1) * interval
                    while (
                        not self._stop_event.is_set() and time.perf_counter() < target
                    ):
                        time.sleep(min(0.015, max(target - time.perf_counter(), 0.0)))
            finally:
                reader.close()
            self.stream_summary.emit(len(selection.events), submitted_count)
            self.completed.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
