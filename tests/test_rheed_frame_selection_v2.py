from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from rheed2morph.rheed.build_shape_bag_inputs_v2 import write_shape_npz_v2
from rheed2morph.rheed.frame_quality_v2 import (
    add_temporal_consistency_scores_v2,
    assign_sample_status_v2,
    extract_frame_quality_features_v2,
    passes_hard_reject_v2,
)
from rheed2morph.rheed.rheed_shape_bag_dataset_v2 import RHEEDShapeBagDatasetV2
from rheed2morph.rheed.select_representative_frames_v2 import (
    select_accepted_rows_v2,
    write_accepted_grid_v2,
    write_rejected_grid_v2,
)
from rheed2morph.rheed.shape_preprocessing import DEFAULT_CHANNEL_NAMES, preprocess_frame_for_shape
from rheed2morph.rheed.spot_streak_geometry import extract_components_and_frame_features


def synthetic_spots(size: int = 64) -> np.ndarray:
    y, x = np.indices((size, size))
    frame = np.full((size, size), 0.16, dtype=np.float32)
    for cy, cx, radius, amp in [(20, 22, 4, 0.45), (42, 38, 5, 0.38), (26, 47, 3, 0.34)]:
        frame += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))
    frame += 0.04 * np.sin(x / 4.0) + 0.02 * np.sin(y / 6.0)
    return np.clip(frame, 0.0, 0.92).astype(np.float32)


def synthetic_bar(size: int = 64) -> np.ndarray:
    frame = np.full((size, size), 0.15, dtype=np.float32)
    frame[18:23, 10:54] = 0.78
    frame[38:43, 8:56] = 0.68
    frame += 0.03 * np.sin(np.indices((size, size))[1] / 5.0)
    return np.clip(frame, 0.0, 0.92).astype(np.float32)


def binary_block(size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size), dtype=np.float32)
    frame[:, size // 2 :] = 1.0
    frame[8:32, 8:32] = 1.0
    return frame


def scored_row(frame: np.ndarray, idx: int) -> dict[str, object]:
    row = extract_frame_quality_features_v2(frame)
    row.update({"sample_id": "NTEST", "video_path": "synthetic.mp4", "frame_idx": idx, "sample_order": idx, "timestamp_sec": float(idx)})
    return add_temporal_consistency_scores_v2([row], {idx: frame}, enabled=False)[0]


class RheedFrameSelectionV2Test(unittest.TestCase):
    def test_binary_block_artifact_is_hard_rejected_even_if_sharp(self) -> None:
        row = scored_row(binary_block(), 0)
        self.assertTrue(row["binary_artifact"] or row["blocky_artifact"])
        self.assertFalse(passes_hard_reject_v2(row))
        accepted, rejected = select_accepted_rows_v2([row], max_candidates=16, min_frame_gap=1, hard_reject=True)
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 1)

    def test_normal_spot_like_frame_is_accepted(self) -> None:
        row = scored_row(synthetic_spots(), 0)
        self.assertFalse(row["binary_artifact"])
        self.assertFalse(row["blocky_artifact"])
        accepted, _ = select_accepted_rows_v2([row], max_candidates=16, min_frame_gap=1, hard_reject=True)
        self.assertEqual(len(accepted), 1)

    def test_elongated_bar_like_frame_is_accepted(self) -> None:
        row = scored_row(synthetic_bar(), 0)
        self.assertFalse(row["binary_artifact"])
        self.assertFalse(row["blocky_artifact"])
        accepted, _ = select_accepted_rows_v2([row], max_candidates=16, min_frame_gap=1, hard_reject=True)
        self.assertEqual(len(accepted), 1)

    def test_almost_black_and_saturated_frames_are_rejected(self) -> None:
        black = scored_row(np.zeros((64, 64), dtype=np.float32), 0)
        white = scored_row(np.ones((64, 64), dtype=np.float32), 1)
        self.assertTrue(black["almost_black"])
        self.assertTrue(white["almost_white"])
        self.assertFalse(passes_hard_reject_v2(black))
        self.assertFalse(passes_hard_reject_v2(white))

    def test_temporal_consistency_penalizes_isolated_spike(self) -> None:
        rows = []
        images = {}
        for idx, frame, validity, info in [
            (0, synthetic_spots() * 0.5, 0.10, 0.10),
            (1, binary_block(), 0.90, 0.90),
            (2, synthetic_spots() * 0.5, 0.10, 0.10),
        ]:
            row = {
                "frame_idx": idx,
                "sample_order": idx,
                "validity_score": validity,
                "information_score": info,
                "pattern_visibility_score": info,
                "plausible_spot_streak_score": info,
                "local_contrast_after_bgsub": info,
                "projection_peak_score": info,
                "fft_mid_frequency_power": info,
                "fft_anisotropy": info,
                "non_shadow_score": 1.0,
                "artifact_penalty": 0.0,
            }
            rows.append(row)
            images[idx] = frame
        scored = add_temporal_consistency_scores_v2(rows, images, enabled=True, window=1)
        self.assertTrue(scored[1]["isolated_quality_spike"])
        self.assertLess(scored[1]["temporal_consistency_score"], 0.25)

    def test_selector_does_not_force_max_candidates(self) -> None:
        rows = [scored_row(synthetic_spots(), 0), scored_row(binary_block(), 1), scored_row(np.zeros((64, 64), dtype=np.float32), 2)]
        accepted, rejected = select_accepted_rows_v2(rows, max_candidates=16, min_frame_gap=1, hard_reject=True)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 2)

    def test_status_assignment(self) -> None:
        row = scored_row(synthetic_spots(), 0)
        good = [dict(row, final_score=0.5, validity_score=0.8) for _ in range(8)]
        usable = [dict(row, final_score=0.35, validity_score=0.7) for _ in range(3)]
        low = [dict(row, final_score=0.2, validity_score=0.5)]
        self.assertEqual(assign_sample_status_v2(accepted_rows=good, min_accepted_for_good=8, min_accepted_for_usable=3, min_accepted_for_low_confidence=1)[0], "GOOD")
        self.assertEqual(assign_sample_status_v2(accepted_rows=usable, min_accepted_for_good=8, min_accepted_for_usable=3, min_accepted_for_low_confidence=1)[0], "USABLE")
        self.assertEqual(assign_sample_status_v2(accepted_rows=low, min_accepted_for_good=8, min_accepted_for_usable=3, min_accepted_for_low_confidence=1)[0], "LOW_CONFIDENCE")
        self.assertEqual(assign_sample_status_v2(accepted_rows=[], min_accepted_for_good=8, min_accepted_for_usable=3, min_accepted_for_low_confidence=1)[0], "EXCLUDE")

    def test_accepted_and_rejected_grids_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            accepted = [dict(scored_row(synthetic_spots(), 4), candidate_rank=1)]
            rejected = [dict(scored_row(binary_block(), 5), rejection_reason="binary_artifact")]
            images = {4: synthetic_spots(), 5: binary_block()}
            accepted_grid = write_accepted_grid_v2(
                root / "accepted_candidate_frames_grid.png",
                sample_id="NTEST",
                source_video=Path("/tmp/NTEST_raw_crop.mp4"),
                total_scanned=2,
                status="LOW_CONFIDENCE",
                accepted_rows=accepted,
                frame_images=images,
            )
            rejected_grid = write_rejected_grid_v2(
                root / "rejected_bad_frames_grid.png",
                sample_id="NTEST",
                rejected_rows=rejected,
                frame_images=images,
            )
            self.assertTrue(accepted_grid.is_file())
            self.assertTrue(rejected_grid.is_file())

    def test_shape_bag_v2_npz_and_dataset_load_variable_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame = synthetic_spots()
            processed = preprocess_frame_for_shape(frame, image_size=32)
            _, features = extract_components_and_frame_features(
                soft_mask=processed.channels["soft_spot_streak_mask"],
                enhanced_image=processed.channels["log_bgsub"],
                artifact_mask=processed.artifact_mask,
                min_area=4,
            )
            records = [
                SimpleNamespace(
                    channels=processed.channels,
                    frame_weight=1.0,
                    frame_idx=7,
                    timestamp_sec=0.7,
                    features=features,
                )
            ]
            npz = root / "shape_bag_v2.npz"
            write_shape_npz_v2(
                npz,
                frames=records,
                image_size=32,
                max_frames=4,
                pad_to_max=True,
                sample_feature_vector=np.asarray([1.0, 2.0], dtype=np.float32),
                sample_feature_names=["a", "b"],
                sample_quality=0.75,
                sample_status="USABLE",
                rejected_frame_count=3,
                source_type="accepted_v2_fallback",
            )
            with np.load(npz, allow_pickle=False) as data:
                for key in ["frames", "frame_mask", "frame_weights", "num_valid_frames", "sample_status"]:
                    self.assertIn(key, data.files)
                self.assertEqual(data["frames"].shape, (4, len(DEFAULT_CHANNEL_NAMES), 32, 32))
                self.assertEqual(int(data["num_valid_frames"]), 1)
            manifest = root / "rheed_shape_bag_manifest_v2.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "shape_bag_npz", "sample_quality", "sample_status", "source_type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "NTEST",
                        "shape_bag_npz": str(npz),
                        "sample_quality": "0.75",
                        "sample_status": "USABLE",
                        "source_type": "accepted_v2_fallback",
                    }
                )
            item = RHEEDShapeBagDatasetV2(manifest)[0]
            self.assertEqual(item["frames"].shape, (4, len(DEFAULT_CHANNEL_NAMES), 32, 32))
            self.assertEqual(item["frame_mask"].tolist(), [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(item["num_valid_frames"], 1)
            self.assertEqual(item["sample_status"], "USABLE")

    def test_v2_sources_do_not_import_knn_or_nearest_neighbor_retrieval(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "src" / "rheed2morph" / "rheed" / "frame_quality_v2.py",
            root / "src" / "rheed2morph" / "rheed" / "select_representative_frames_v2.py",
            root / "src" / "rheed2morph" / "rheed" / "build_shape_bag_inputs_v2.py",
            root / "src" / "rheed2morph" / "rheed" / "rheed_shape_bag_dataset_v2.py",
        ]
        source_text = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertNotIn("KNeighbors", source_text)
        self.assertNotIn("sklearn.neighbors", source_text)
        self.assertNotIn("NearestNeighbors", source_text)


if __name__ == "__main__":
    unittest.main()
