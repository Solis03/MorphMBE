from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from rheed2morph.rheed.automatic_roi_keyframe import (
    Rect,
    SUPERVISED_PHASE_FEATURES,
    _supervised_candidate_rows,
    iter_video_frames,
    predict_roi,
    select_from_source,
    select_keyframes,
)


def synthetic_rotation_frames(
    *,
    frame_count: int = 72,
    width: int = 180,
    height: int = 140,
    period: int = 24,
) -> list[np.ndarray]:
    yy, xx = np.indices((height, width))
    center_x, center_y = 92.0, 70.0
    radius = 58.0
    aperture = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    frames = []
    for index in range(frame_count):
        phase = (index % period) / period
        # Right-most x at phase 0.5; y continues upward through the vertex.
        theta = 2.0 * np.pi * (phase - 0.5)
        spot_y = center_y - 34.0 * np.sin(theta)
        spot_x = center_x + 22.0 - 0.018 * (spot_y - center_y) ** 2
        base = np.zeros((height, width), dtype=np.float32)
        base[aperture] = 0.10
        spot = np.exp(
            -(
                (xx - spot_x) ** 2 / (2.0 * 2.0**2)
                + (yy - spot_y) ** 2 / (2.0 * 2.5**2)
            )
        )
        base += 0.75 * spot
        base += 0.20 * np.exp(
            -(
                (xx - (spot_x - 7.0)) ** 2 / (2.0 * 2.4**2)
                + (yy - (spot_y + 15.0)) ** 2 / (2.0 * 2.4**2)
            )
        )
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 2] = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(base * 100.0, 0, 255).astype(np.uint8)
        frames.append(rgb)
    return frames


class AutomaticRHEEDSelectionTest(unittest.TestCase):
    def test_aperture_roi_is_inside_source_and_contains_signal(self) -> None:
        frames = synthetic_rotation_frames()
        prediction, analysis = predict_roi(
            frames[::3],
            method="calibrated_safe",
        )

        rect = prediction.rect
        self.assertGreater(rect.width, 20)
        self.assertGreater(rect.height, rect.width)
        self.assertGreaterEqual(rect.x, 0)
        self.assertGreaterEqual(rect.y, 0)
        self.assertLessEqual(rect.x2, rect.source_width)
        self.assertLessEqual(rect.y2, rect.source_height)
        self.assertGreater(prediction.safe_pixel_fraction, 0.98)
        self.assertGreater(prediction.activity_coverage, 0.50)
        self.assertGreater(float(analysis.aperture_mask.mean()), 0.10)

    def test_physics_vertex_finds_a_periodic_rightmost_phase(self) -> None:
        period = 24
        frames = synthetic_rotation_frames(period=period)
        roi = Rect(48, 22, 88, 112, 180, 140)
        from rheed2morph.rheed.automatic_roi_keyframe import (
            extract_spot_trajectory,
        )

        trajectory = extract_spot_trajectory(enumerate(frames), roi)
        predictions, candidates = select_keyframes(trajectory)

        selected = predictions["physics_vertex"].frame_index
        phase_error = min(
            abs(selected - target)
            for target in range(period // 2, len(frames), period)
        )
        self.assertLessEqual(phase_error, 3)
        self.assertGreaterEqual(len(candidates), 2)
        front_selected = predictions["front_visibility"].frame_index
        front_phase_error = min(
            abs(front_selected - target)
            for target in range(period // 2, len(frames), period)
        )
        self.assertLessEqual(front_phase_error, 3)
        compact_selected = predictions["compact_visibility"].frame_index
        compact_phase_error = min(
            abs(compact_selected - target)
            for target in range(period // 2, len(frames), period)
        )
        self.assertLessEqual(compact_phase_error, 3)

    def test_complete_png_directory_selection_is_non_destructive(self) -> None:
        frames = synthetic_rotation_frames(frame_count=48)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "frames"
            frame_dir.mkdir()
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{index}.png")
            before = {
                path.name: path.stat().st_mtime_ns
                for path in frame_dir.glob("*.png")
            }

            selection, trajectory, _ = select_from_source(
                frame_dir,
                roi_sample_count=16,
            )

            self.assertEqual(selection.frame_count, len(frames))
            self.assertEqual(len(trajectory), len(frames))
            self.assertIn("physics_vertex", selection.keyframes)
            after = {
                path.name: path.stat().st_mtime_ns
                for path in frame_dir.glob("*.png")
            }
            self.assertEqual(before, after)

    def test_supervised_candidate_schema_has_dual_tracker_support(self) -> None:
        frames = synthetic_rotation_frames(frame_count=72)
        roi = Rect(48, 22, 88, 112, 180, 140)
        from rheed2morph.rheed.automatic_roi_keyframe import (
            extract_spot_trajectory,
        )

        trajectory = extract_spot_trajectory(enumerate(frames), roi)
        candidates, periods = _supervised_candidate_rows(trajectory)

        self.assertGreater(len(candidates), 2)
        self.assertEqual(set(periods), {"front", "compact"})
        self.assertEqual(
            {row["tracker"] for row in candidates},
            {"front", "compact"},
        )
        for feature in SUPERVISED_PHASE_FEATURES:
            self.assertIn(feature, candidates[0])

    def test_common_video_decoder_supports_avi(self) -> None:
        frames = synthetic_rotation_frames(frame_count=6, width=64, height=48)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.avi"
            writer = imageio.get_writer(
                str(path),
                fps=6,
                codec="mpeg4",
                macro_block_size=1,
            )
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

            decoded = list(iter_video_frames(path))

            self.assertEqual(len(decoded), len(frames))
            self.assertEqual(decoded[0][1].shape, frames[0].shape)


if __name__ == "__main__":
    unittest.main()
