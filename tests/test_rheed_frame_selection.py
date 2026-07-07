from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from rheed2morph.rheed.frame_quality import (
    extract_frame_quality_features,
    score_frame_quality_rows,
)
from rheed2morph.rheed.manual_frame_selection import parse_manual_selection_line
from rheed2morph.rheed.select_representative_frames import (
    select_candidate_rows,
    write_candidate_grids,
    write_manual_template,
)


def stripe_frame(size: int = 64) -> np.ndarray:
    y, x = np.indices((size, size))
    frame = 0.35 + 0.25 * ((x // 4) % 2) + 0.15 * np.sin(y / 3.0)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def blurred(frame: np.ndarray) -> np.ndarray:
    return (
        frame
        + np.roll(frame, 1, axis=0)
        + np.roll(frame, -1, axis=0)
        + np.roll(frame, 1, axis=1)
        + np.roll(frame, -1, axis=1)
    ) / 5.0


class RheedFrameSelectionTest(unittest.TestCase):
    def test_extract_frame_quality_features_returns_finite_values(self) -> None:
        frame = stripe_frame()
        features = extract_frame_quality_features(frame)
        numeric_values = [value for value in features.values() if isinstance(value, float)]
        self.assertTrue(numeric_values)
        self.assertTrue(all(np.isfinite(value) for value in numeric_values))

    def test_almost_black_frames_are_flagged(self) -> None:
        features = extract_frame_quality_features(np.zeros((64, 64), dtype=np.float32))
        self.assertTrue(features["almost_black"])
        self.assertTrue(features["very_low_dynamic_range"])

    def test_saturated_frames_are_flagged(self) -> None:
        frame = np.ones((64, 64), dtype=np.float32)
        features = extract_frame_quality_features(frame)
        self.assertTrue(features["almost_white"])
        self.assertTrue(features["over_saturated"])

    def test_sharp_synthetic_frame_scores_higher_than_blurred_frame(self) -> None:
        sharp = extract_frame_quality_features(stripe_frame())
        smooth = extract_frame_quality_features(blurred(blurred(stripe_frame())))
        sharp.update({"frame_idx": 0, "timestamp_sec": 0.0})
        smooth.update({"frame_idx": 1, "timestamp_sec": 1.0})
        scored = score_frame_quality_rows([sharp, smooth])
        self.assertGreater(scored[0]["sharpness_score"], scored[1]["sharpness_score"])
        self.assertGreater(scored[0]["quality_score"], scored[1]["quality_score"])

    def test_shadowed_frame_receives_higher_shadow_penalty(self) -> None:
        clean = stripe_frame()
        shadowed = clean.copy()
        shadowed[:, :24] *= 0.05
        clean_features = extract_frame_quality_features(clean)
        shadow_features = extract_frame_quality_features(shadowed)
        self.assertGreater(shadow_features["shadow_penalty"], clean_features["shadow_penalty"])

    def test_candidate_selection_enforces_min_frame_gap(self) -> None:
        rows = []
        images = {}
        for idx in [10, 12, 20, 30]:
            row = {
                "frame_idx": idx,
                "timestamp_sec": float(idx),
                "quality_score": 1.0 - idx / 100.0,
                "almost_black": False,
                "almost_white": False,
                "over_saturated": False,
                "very_low_dynamic_range": False,
                "strong_shadow": False,
            }
            rows.append(row)
            images[idx] = np.full((32, 32), idx / 30.0, dtype=np.float32)
        selected = select_candidate_rows(rows, images, num_candidates=3, min_frame_gap=5, min_ssim_distance=0.0)
        selected_indices = [int(row["frame_idx"]) for row in selected[:2]]
        self.assertEqual(selected_indices, [10, 20])

    def test_candidate_grid_image_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            candidates = []
            images = {}
            for rank, idx in enumerate([5, 15], start=1):
                candidates.append(
                    {
                        "candidate_rank": rank,
                        "frame_idx": idx,
                        "timestamp_sec": float(idx) / 10.0,
                        "quality_score": 0.8,
                        "flags": "",
                    }
                )
                images[idx] = stripe_frame()
            grid, pair_grid = write_candidate_grids(
                root,
                sample_id="N0001",
                source_video=Path("/tmp/N0001_raw_crop.mp4"),
                scored_count=20,
                candidates=candidates,
                frame_images=images,
                low_confidence=False,
            )
            self.assertTrue(grid.is_file())
            self.assertTrue(pair_grid.is_file())

    def test_manual_template_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "manual_selected_frames.txt"
            write_manual_template(path, sample_id="N0001", source_video=Path("/tmp/video.mp4"))
            path.write_text("rank02\n", encoding="utf-8")
            changed = write_manual_template(path, sample_id="N0001", source_video=Path("/tmp/video.mp4"))
            self.assertFalse(changed)
            self.assertEqual(path.read_text(encoding="utf-8"), "rank02\n")

    def test_manual_parser_accepts_rank_and_frame_idx_forms(self) -> None:
        rank = parse_manual_selection_line("rank01")
        frame = parse_manual_selection_line("frame_idx=123")
        filename = parse_manual_selection_line("rank02_frame000456_t004.10s_score0.873.png")
        self.assertEqual(rank["rank"], 1)
        self.assertEqual(frame["frame_idx"], 123)
        self.assertEqual(filename["rank"], 2)
        self.assertEqual(filename["frame_idx"], 456)

    def test_new_frame_selection_sources_do_not_import_retrieval_models(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source_text = "\n".join(
            [
                (root / "src" / "rheed2morph" / "rheed" / "frame_quality.py").read_text(encoding="utf-8"),
                (root / "src" / "rheed2morph" / "rheed" / "select_representative_frames.py").read_text(encoding="utf-8"),
                (root / "src" / "rheed2morph" / "rheed" / "manual_frame_selection.py").read_text(encoding="utf-8"),
            ]
        )
        self.assertNotIn("KNeighbors", source_text)
        self.assertNotIn("sklearn.neighbors", source_text)


if __name__ == "__main__":
    unittest.main()
