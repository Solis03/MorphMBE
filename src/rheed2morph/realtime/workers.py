"""Background replay and prediction workers for the Qt application."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PySide6.QtCore import QThread, Signal

from rheed2morph.rheed.automatic_roi_keyframe import _rgb_uint8

from .clips import build_model_clip
from .model import MorphologyPrediction, RealtimeMorphologyPredictor
from .selector import ReplayEvent, ReplaySelection, analyze_replay


@dataclass(frozen=True)
class PredictionJob:
    sample_id: str
    source: Path
    event: ReplayEvent
    event_time_seconds: float
    selected_16: np.ndarray


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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.bundle_path = Path(bundle_path)
        self.device = str(device)
        self.jobs: queue.Queue[PredictionJob | None] = queue.Queue(maxsize=1)

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
            self.log.emit("加载 M14i 标量头、M12a 生成器与 R3D-18")
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
                    f"帧 {job.event.frame_index}: 运行 causal-8 + selected-16 推理"
                )
                seed = (
                    int(job.sample_id) * 1_000_003
                    + int(job.event.frame_index) * 97
                ) % (2**31 - 1)
                prediction = predictor.predict(
                    job.selected_16,
                    keyframe_quality=job.event.keyframe_quality,
                    seed=seed,
                )
                self.result.emit(
                    PredictionResult(job=job, prediction=prediction)
                )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ReplayWorker(QThread):
    log = Signal(str)
    prepared = Signal(object, float)
    frame = Signal(object, int, float, bool)
    prediction_job = Signal(object)
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

    def run(self) -> None:
        try:
            repository = Path(self.config["repository_root"])
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
                foundation_cache_dir=(
                    repository / self.config["foundation_cache_dir"]
                ),
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
                progress=self.log.emit,
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
            maximum_display = float(
                self.config.get("maximum_display_fps", 24.0)
            )
            display_stride = max(
                1, int(np.ceil(playback_fps / maximum_display))
            )
            interval = self.playback_ratio / fps
            events = {
                event.frame_index
                + int(self.config.get("prediction_trigger_delay_frames", 8)): event
                for event in selection.events
            }
            ring: deque[np.ndarray] = deque(maxlen=16)
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
                    rgb = _rgb_uint8(frame)
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
                        if len(ring) != 16:
                            self.log.emit(
                                f"帧 {event.frame_index}: 时序缓存不足，跳过"
                            )
                        else:
                            clip = build_model_clip(
                                list(ring),
                                selection.model_input_roi.rect,
                                output_size=int(
                                    self.config.get("model_image_size", 224)
                                ),
                            )
                            self.log.emit(
                                f"帧 {event.frame_index}: 完整点阵顶点确认，"
                                "已用模型输入 ROI 收齐 16 帧"
                            )
                            self.prediction_job.emit(
                                PredictionJob(
                                    sample_id=self.sample_id,
                                    source=self.source,
                                    event=event,
                                    event_time_seconds=(
                                        event.frame_index / fps
                                    ),
                                    selected_16=clip,
                                )
                            )
                    target = start + paused_total + (index + 1) * interval
                    while (
                        not self._stop_event.is_set()
                        and time.perf_counter() < target
                    ):
                        time.sleep(
                            min(0.015, max(target - time.perf_counter(), 0.0))
                        )
            finally:
                reader.close()
            self.completed.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
