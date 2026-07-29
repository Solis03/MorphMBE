"""Persist derived replay predictions and provenance outside raw data."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from .workers import PredictionResult

if TYPE_CHECKING:
    from .selector import ReplaySelection


class SessionRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        sample_id: str,
        source: str | Path,
        playback_ratio: float,
    ) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_video = Path(source).stem.replace(" ", "_")[:48]
        candidate = (
            Path(root) / f"{stamp}_{sample_id}_{safe_video}"
        ).resolve()
        suffix = 1
        while candidate.exists():
            candidate = (
                Path(root)
                / f"{stamp}_{sample_id}_{safe_video}_{suffix:02d}"
            ).resolve()
            suffix += 1
        self.root = candidate
        self.root.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.root / "prediction_timeline.csv"
        (self.root / "generated_afm").mkdir()
        self.metadata = {
            "sample_id": str(sample_id),
            "source_video": str(Path(source).resolve()),
            "source_video_read_only": True,
            "playback_duration_ratio": float(playback_ratio),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.root / "session.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def record_selection(
        self,
        selection: "ReplaySelection",
        *,
        fps: float,
    ) -> None:
        """Persist explicit ROI roles so tracking can never mimic model input."""

        self.metadata["selector"] = {
            "fps": float(fps),
            "frame_count": int(selection.frame_count),
            "estimated_period_frames": selection.estimated_period_frames,
            "model_input_roi": asdict(selection.model_input_roi.rect),
            "internal_tracking_roi_not_model_input": asdict(
                selection.tracking_roi.rect
            ),
            "conservative_audit_roi_not_model_input": asdict(
                selection.audit_full_lattice_roi.rect
            ),
            "events": [asdict(event) for event in selection.events],
        }
        (self.root / "session.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def record(self, result: PredictionResult) -> Path:
        event = result.job.event
        prediction = result.prediction
        stem = f"event_{event.frame_index:06d}"
        target = self.root / "generated_afm" / f"{stem}.npz"
        np.savez_compressed(
            target,
            height_nm=prediction.height_nm,
            unit_shape=prediction.unit_shape,
            predicted_rq_nm=np.asarray(prediction.rq.value),
            predicted_fsmi_nm=np.asarray(prediction.fsmi.value),
            combined_confidence=np.asarray(prediction.combined_confidence),
            keyframe_quality=np.asarray(prediction.keyframe_quality),
            event_frame=np.asarray(event.frame_index),
            event_time_seconds=np.asarray(result.job.event_time_seconds),
            model_id=np.asarray(prediction.model_id),
            retrieval_at_inference=np.asarray(False),
        )
        row = {
            "event_frame": event.frame_index,
            "event_time_seconds": result.job.event_time_seconds,
            "predicted_rq_nm": prediction.rq.value,
            "unconstrained_rq_nm": prediction.rq.unconstrained_value,
            "rq_support_clipped": prediction.rq.support_clipped,
            "rq_expected_absolute_error_nm": (
                prediction.rq.expected_absolute_error
            ),
            "rq_interval_lower_nm": prediction.rq.interval_lower,
            "rq_interval_upper_nm": prediction.rq.interval_upper,
            "predicted_fsmi_nm": prediction.fsmi.value,
            "unconstrained_fsmi_nm": prediction.fsmi.unconstrained_value,
            "fsmi_support_clipped": prediction.fsmi.support_clipped,
            "fsmi_expected_absolute_error_nm": (
                prediction.fsmi.expected_absolute_error
            ),
            "fsmi_interval_lower_nm": prediction.fsmi.interval_lower,
            "fsmi_interval_upper_nm": prediction.fsmi.interval_upper,
            "model_confidence": prediction.model_confidence,
            "keyframe_quality": prediction.keyframe_quality,
            "combined_confidence": prediction.combined_confidence,
            "inference_seconds": prediction.inference_seconds,
            "generated_rq_nm": prediction.generated_rq_nm,
            "model_id": prediction.model_id,
            "generated_npz": str(target.relative_to(self.root)),
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return target
