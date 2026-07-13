from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_roughness.visualize_pairs import (
    build_afm_candidate_table,
    build_rheed_candidate_table,
    calculate_common_afm_scale,
    filter_removed_samples,
    format_value,
    read_removelist,
    render_pair_grid,
    select_representative_afm_scan,
    select_representative_rheed_frame,
    write_csv_rows,
)


class PairVisualizationTest(unittest.TestCase):
    def test_rheed_selection_is_deterministic_and_ignores_afm(self) -> None:
        frame_df = pd.DataFrame(
            {
                "sample_id": ["1", "1", "1"],
                "frame_idx": [1, 2, 3],
                "frame_timestamp_sec": [1.0, 2.0, 3.0],
                "detector_confidence": [0.2, 0.9, 0.4],
                "sharpness_score": [0.2, 0.9, 0.4],
                "contrast_score": [0.2, 0.9, 0.4],
                "robust_dynamic_range": [0.2, 0.9, 0.4],
                "dynamic_range_score": [0.2, 0.9, 0.4],
                "pattern_visibility_score": [0.2, 0.9, 0.4],
                "roi_coverage": [1.0, 1.0, 1.0],
                "roi_clipping": [0.0, 0.0, 0.0],
                "saturation_fraction": [0.0, 0.0, 0.0],
                "underexposed_fraction": [0.0, 0.0, 0.0],
                "frame_to_frame_motion_magnitude": [0.0, 0.0, 0.0],
                "frame_to_reference_translation_x": [0.0, 0.0, 0.0],
                "frame_to_reference_translation_y": [0.0, 0.0, 0.0],
                "frame_qc_reasons": ["", "", ""],
                "frame_valid": [1, 1, 1],
                "AFM_Rq_nm": [100.0, -100.0, 50.0],
            }
        )
        ranked_a = build_rheed_candidate_table(frame_df, type("P", (), {})())
        selected_a, _ = select_representative_rheed_frame(ranked_a)
        frame_df["AFM_Rq_nm"] = [-1000.0, 1000.0, -500.0]
        ranked_b = build_rheed_candidate_table(frame_df, type("P", (), {})())
        selected_b, _ = select_representative_rheed_frame(ranked_b)
        self.assertEqual(int(selected_a["frame_idx"]), int(selected_b["frame_idx"]))
        self.assertEqual(int(selected_a["frame_idx"]), 2)

    def test_afm_selection_closest_to_primary_median(self) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["1", "1", "1"],
                "scan_size_um": [1.0, 1.0, 1.0],
                "Rq_nm": [1.0, 2.0, 100.0],
                "Ra_nm": [0.8, 1.5, 80.0],
                "afm_path": ["a.npy", "b.npy", "c.npy"],
                "afm_scan_id": ["a", "b", "c"],
                "target_status": ["ok", "ok", "ok"],
            }
        )
        candidates = build_afm_candidate_table(df)
        selected = select_representative_afm_scan(candidates, primary_scan_size_um=1.0, tolerance_um=0.1)
        self.assertEqual(selected["afm_scan_id"], "b")
        self.assertAlmostEqual(float(selected["sample_median_rq_nm"]), 2.0)

    def test_common_scale_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.npy"
            b = root / "b.npy"
            np.save(a, np.asarray([[-1.0, 0.0], [1.0, 2.0]]))
            np.save(b, np.asarray([[-2.0, 0.0], [2.0, 3.0]]))
            paths = type("P", (), {"repo_root": root})()
            rows = [
                {"selected_height_map_path": "a.npy", "scan_size_um": 1.0, "fallback_used": 0, "height_unit_exported": "nm"},
                {"selected_height_map_path": "b.npy", "scan_size_um": 1.0, "fallback_used": 0, "height_unit_exported": "nm"},
            ]
            lo, hi = calculate_common_afm_scale(rows, paths)
            self.assertLess(lo, 0)
            self.assertGreater(hi, 0)
            self.assertAlmostEqual(abs(lo), abs(hi))

    def test_missing_value_format(self) -> None:
        self.assertEqual(format_value(float("nan"), "nm"), "N/A")

    def test_removelist_parses_leading_numeric_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "removelist.txt"
            path.write_text("6018\n6061 -black holes not reflected from the rheed\n# ignored\n", encoding="utf-8")
            sample_ids, rows = read_removelist(path)
            self.assertEqual(sample_ids, {"6018", "6061"})
            self.assertEqual(rows[1]["sample_id"], "6061")
            self.assertEqual(rows[1]["note"], "black holes not reflected from the rheed")

    def test_filter_removed_samples_drops_from_every_input_table(self) -> None:
        frame_df = pd.DataFrame({"sample_id": ["6023", "7001"], "frame_idx": [1, 2]})
        sample_df = pd.DataFrame({"sample_id": ["6023", "7001"], "rq": [1.0, 2.0]})
        afm_df = pd.DataFrame({"sample_id": ["6023", "7001"], "afm_path": ["a", "b"]})
        pairing_df = pd.DataFrame({"sample_id": ["6023", "7001"], "rheed_video_path": ["v1", "v2"]})
        filtered = filter_removed_samples(frame_df, sample_df, afm_df, pairing_df, {"6023"})
        for table in filtered:
            self.assertEqual(table["sample_id"].tolist(), ["7001"])

    def test_grid_generation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            report.mkdir()
            raw = root / "rheed.png"
            afm = root / "afm.npy"
            from PIL import Image

            Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(raw)
            np.save(afm, np.zeros((16, 16), dtype=float))
            paths = type("P", (), {"repo_root": root, "report_dir": report})()
            rows = [
                {
                    "sample_id": "1",
                    "cached_processed_roi_path": "rheed.png",
                    "selected_height_map_path": "afm.npy",
                    "height_unit_exported": "nm",
                    "scan_size_um": 1.0,
                    "rq_nm": 0.0,
                    "quality_score": 0.5,
                    "morphology_index": 0.5,
                }
            ]
            render_pair_grid(rows, paths, output_stem="grid", title="grid", sort_key="rq_nm", cards_per_row=1)
            self.assertTrue((report / "grid.png").is_file())
            self.assertTrue((report / "grid.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
