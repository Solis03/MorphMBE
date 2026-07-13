from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.rheed_roughness.visualize_manual_pairs import (
    AFMCandidate,
    ManualVizPaths,
    common_scale_for_pairs,
    discover_manual_rheed_images,
    nice_scale_bar_um,
    recompute_height_stats,
    render_pair_grid,
    select_representative_afm_scan,
    valid_select_image,
)


def candidate(scan_id: str, size: float, rq: float, path: Path) -> AFMCandidate:
    return AFMCandidate(
        sample_id="6001",
        sample_group_id="6001",
        growth_run_id="6001",
        material="",
        afm_scan_id=scan_id,
        afm_path=path,
        selected_height_map_path=path,
        channel="ZSensor",
        height_unit_exported="nm",
        scan_size_um=size,
        scan_size_x_um=size,
        scan_size_y_um=size,
        resolution_x=4,
        resolution_y=4,
        rq_nm=rq,
        rq_source="test",
        rq_recomputed_nm=rq,
        ra_nm=rq,
        robust_height_range_nm=rq,
        peak_to_valley_nm=rq,
        qc_flags="",
    )


class ManualPairVisualizationTest(unittest.TestCase):
    def test_valid_select_image_filters_names_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "select_best.JPG"
            bad_prefix = root / "frame_select.jpg"
            bad_video = root / "select.mov"
            hidden = root / "._select.png"
            for path in (good, bad_prefix, bad_video, hidden):
                path.write_bytes(b"x")
            self.assertTrue(valid_select_image(good))
            self.assertFalse(valid_select_image(bad_prefix))
            self.assertFalse(valid_select_image(bad_video))
            self.assertFalse(valid_select_image(hidden))

    def test_discovery_skips_missing_and_prioritizes_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "6001" / "RHEED").mkdir(parents=True)
            (root / "6002" / "RHEED").mkdir(parents=True)
            for name in ("select_best.jpg", "select_final.jpg", "select.jpg"):
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(root / "6001" / "RHEED" / name)
            selections = discover_manual_rheed_images(root)
            by_id = {selection.sample_id: selection for selection in selections}
            self.assertEqual(by_id["6001"].selected_path.name, "select.jpg")
            self.assertIn("multiple_manual_selections", by_id["6001"].warnings)
            self.assertEqual(by_id["6002"].status, "missing_manual_selection")

    def test_afm_selection_uses_primary_median(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for idx in range(3):
                path = root / f"{idx}.npy"
                np.save(path, np.zeros((4, 4)))
                paths.append(path)
            selected, median, distance, reason = select_representative_afm_scan(
                [
                    candidate("a", 1.0, 1.0, paths[0]),
                    candidate("b", 1.0, 2.0, paths[1]),
                    candidate("c", 1.0, 100.0, paths[2]),
                ],
                primary_scan_size_um=1.0,
                tolerance_um=0.1,
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.afm_scan_id, "b")
            self.assertAlmostEqual(median, 2.0)
            self.assertAlmostEqual(distance, 0.0)
            self.assertEqual(reason, "closest_to_primary_1um_subset_median_rq")

    def test_afm_selection_falls_back_to_dominant_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for idx in range(3):
                path = root / f"{idx}.npy"
                np.save(path, np.zeros((4, 4)))
                paths.append(path)
            selected, _, _, reason = select_representative_afm_scan(
                [
                    candidate("a", 2.0, 1.0, paths[0]),
                    candidate("b", 2.0, 3.0, paths[1]),
                    candidate("c", 5.0, 100.0, paths[2]),
                ],
                primary_scan_size_um=1.0,
                tolerance_um=0.1,
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected.afm_scan_id, "a")
            self.assertIn("dominant_size_2", reason)

    def test_rq_recompute_and_scale_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "height.npy"
            np.save(path, np.asarray([[0.0, 2.0], [2.0, 4.0]]))
            stats = recompute_height_stats(path, "nm")
            self.assertAlmostEqual(stats["rq"], np.sqrt(2.0))
            self.assertAlmostEqual(nice_scale_bar_um(1.0), 0.2)

    def test_render_grid_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "select.jpg"
            afm = root / "height.npy"
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(manual)
            np.save(afm, np.zeros((8, 8)))
            paths = ManualVizPaths(
                repo_root=root,
                manual_root=root,
                output_dir=root / "out",
                report_dir=root / "report",
                cache_dir=root / "out" / "cache",
                pages_dir=root / "report" / "pages",
                gallery_assets_dir=root / "report" / "assets",
                plane_corrected_afm_root=root,
                descriptor_csv=root / "missing.csv",
                source_outputs_dir=root / "out",
            )
            paths.report_dir.mkdir()
            pair = __import__("analysis.rheed_roughness.visualize_manual_pairs", fromlist=["SelectedPair"]).SelectedPair(
                sample_id="6001",
                sample_group_id="6001",
                growth_run_id="6001",
                material="",
                manual_folder=root,
                manual_rheed_path=manual,
                manual_rheed_filename=manual.name,
                all_manual_candidates=(manual,),
                manual_warnings=(),
                afm=candidate("a", 1.0, 0.0, afm),
                number_of_candidate_scans=1,
                sample_median_rq_nm=0.0,
                distance_from_median_rq_nm=0.0,
                selection_reason="test",
                native_display_min_nm=-1.0,
                native_display_max_nm=1.0,
            )
            render_pair_grid([pair], paths, output_stem="grid", title="grid", sort_key="rq_nm", cards_per_row=1)
            self.assertTrue((paths.report_dir / "grid.png").is_file())
            self.assertTrue((paths.report_dir / "grid.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
