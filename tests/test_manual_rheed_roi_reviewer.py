from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rheed2morph.rheed.manual_roi import (
    ROI,
    crop_roi_pixels,
    discover_video_records,
    fit_display_transform,
    frame_path_for_index,
    roi_from_selection,
    selection_from_record,
    sorted_frame_paths,
    update_video_selection,
    validate_roi,
)


def write_frame(path: Path, width: int = 8, height: int = 6) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    arr[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    arr[:, :, 2] = 17
    Image.fromarray(arr, mode="RGB").save(path)
    return arr


def make_review_tree(root: Path) -> dict[str, np.ndarray]:
    sample = root / "6022"
    frames_dir = sample / "videos" / "video_a" / "frames"
    arrays = {
        "1": write_frame(frames_dir / "1.png"),
        "10": write_frame(frames_dir / "10.png"),
        "2": write_frame(frames_dir / "2.png"),
    }
    metadata = {
        "schema_version": 1,
        "sample_id": "6022",
        "source_sample_dir": "data/pair/6022",
        "notes": "keep me",
        "videos": {
            "video_a": {
                "video_id": "video_a",
                "source_video": "data/pair/6022/RHEED/video_a.MOV",
                "frames_dir": "videos/video_a/frames",
                "video_info": {"width": 8, "height": 6, "extracted_frame_count": 11},
                "selection": {"keyframe_index": None, "clip_frame_count": None},
            },
            "video_b": {
                "video_id": "video_b",
                "source_video": "data/pair/6022/RHEED/video_b.MOV",
                "frames_dir": "videos/video_b/frames",
                "video_info": {"width": 8, "height": 6, "extracted_frame_count": 0},
                "selection": {"keyframe_index": 5, "clip_frame_count": 7},
            },
        },
    }
    (sample / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return arrays


class ManualRHEEDROIReviewerTest(unittest.TestCase):
    def test_display_to_source_coordinate_mapping(self) -> None:
        transform = fit_display_transform(source_width=200, source_height=100, box_width=500, box_height=500)
        self.assertEqual(transform.display_width, 500)
        self.assertEqual(transform.display_height, 250)
        self.assertEqual(transform.display_to_source_point(transform.display_x + 250, transform.display_y + 125), (100, 50))
        roi = ROI(20, 10, 50, 25, 200, 100, 3)
        self.assertEqual(transform.source_to_display_rect(roi), (50.0, 150.0, 175.0, 212.5))

    def test_roi_boundary_validation(self) -> None:
        validate_roi(ROI(0, 0, 8, 6, 8, 6, 1))
        with self.assertRaises(ValueError):
            validate_roi(ROI(7, 0, 2, 1, 8, 6, 1))
        with self.assertRaises(ValueError):
            validate_roi(ROI(0, 0, 0, 1, 8, 6, 1))

    def test_numeric_frame_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_review_tree(root)
            frames = sorted_frame_paths(root / "6022" / "videos" / "video_a" / "frames")
            self.assertEqual([path.name for path in frames], ["1.png", "2.png", "10.png"])

    def test_metadata_atomic_update_preserves_other_selection_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_review_tree(root)
            metadata_path = root / "6022" / "metadata.json"
            roi = ROI(1, 2, 3, 2, 8, 6, 2)

            update_video_selection(metadata_path, "video_a", 2, 5, roi)

            payload = json.loads(metadata_path.read_text())
            self.assertEqual(payload["notes"], "keep me")
            self.assertEqual(payload["videos"]["video_a"]["selection"]["keyframe_index"], 2)
            self.assertEqual(payload["videos"]["video_a"]["selection"]["clip_frame_count"], 5)
            self.assertEqual(payload["videos"]["video_a"]["selection"]["roi"]["coordinate_space"], "source_frame_pixels")
            self.assertEqual(payload["videos"]["video_b"]["selection"], {"keyframe_index": 5, "clip_frame_count": 7})

    def test_saved_roi_restores_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_review_tree(root)
            metadata_path = root / "6022" / "metadata.json"
            roi = ROI(1, 2, 3, 2, 8, 6, 2)
            update_video_selection(metadata_path, "video_a", 2, 5, roi)

            record = [item for item in discover_video_records(root) if item.video_id == "video_a"][0]
            restored = roi_from_selection(selection_from_record(record))

            self.assertEqual(restored, roi)

    def test_roi_crop_matches_source_array_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = make_review_tree(root)
            frames_dir = root / "6022" / "videos" / "video_a" / "frames"
            roi = ROI(1, 2, 4, 3, 8, 6, 2)

            crop = np.asarray(crop_roi_pixels(frame_path_for_index(frames_dir, 2), roi))

            self.assertTrue(np.array_equal(crop, arrays["2"][2:5, 1:5, :]))


if __name__ == "__main__":
    unittest.main()
