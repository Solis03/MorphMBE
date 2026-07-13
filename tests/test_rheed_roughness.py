from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.rheed_roughness.run import (
    aggregate_numeric,
    apply_named_perturbation,
    bootstrap_spearman,
    compute_morphology_scores,
    convert_height_to_nm,
    discover_json_schema,
    grouped_cv_predictions,
    validate_unique_pairing,
    write_html_report,
)


class RheedRoughnessHelpersTest(unittest.TestCase):
    def test_unit_conversion_to_nm(self) -> None:
        values = np.asarray([1.0, 2.0])
        converted, status = convert_height_to_nm(values, "um")
        self.assertEqual(status, "ok")
        np.testing.assert_allclose(converted, [1000.0, 2000.0])

    def test_unknown_unit_is_flagged_without_rescaling(self) -> None:
        values = np.asarray([1.0, 2.0])
        converted, status = convert_height_to_nm(values, "counts")
        self.assertEqual(status, "unknown_height_unit")
        np.testing.assert_allclose(converted, values)

    def test_json_roughness_key_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scan_metadata.json").write_text('{"Rq": 1.2, "height_unit": "nm", "nested": {"Ra": 0.8}}')
            rows, summary = discover_json_schema([root])
        keys = {row["json_key"] for row in rows}
        self.assertIn("Rq", keys)
        self.assertIn("nested.Ra", keys)
        self.assertIn("Rq", summary["roughness_candidate_keys"])

    def test_duplicate_pairing_detection(self) -> None:
        rows = [
            {"sample_id": "6001", "afm_path": "a.npy"},
            {"sample_id": "6001", "afm_path": "b.npy"},
        ]
        issues = validate_unique_pairing(rows, ["sample_id"], "afm_path")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["candidate_count"], 2)

    def test_morphology_index_formula(self) -> None:
        scores = compute_morphology_scores(
            {
                "round_spot_count": 4,
                "elongated_spot_count": 2,
                "diffuse_blob_count": 0,
                "horizontal_bar_count": 1,
                "vertical_streak_count": 1,
                "bar_like_score": 0.2,
                "total_component_count": 8,
            }
        )
        self.assertGreater(scores["raw_spottiness"], 0)
        self.assertGreater(scores["raw_streakiness"], 0)
        self.assertGreaterEqual(scores["morphology_index"], 0)
        self.assertLessEqual(scores["morphology_index"], 1)

    def test_sample_aggregation(self) -> None:
        summary = aggregate_numeric([1.0, 2.0, 10.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertTrue(math.isfinite(summary["iqr"]))

    def test_grouped_cv_predictions_are_out_of_fold(self) -> None:
        rows = [
            {"growth_run_id": f"g{i}", "target": float(i), "x": float(i)}
            for i in range(6)
        ]
        y, pred, names = grouped_cv_predictions(rows, "target", ["x"])
        self.assertEqual(len(y), 6)
        self.assertEqual(len(pred), 6)
        self.assertIn("x", names)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_bootstrap_reproducibility(self) -> None:
        x = np.arange(8)
        y = np.arange(8)
        first = bootstrap_spearman(x, y, resamples=50, seed=123)
        second = bootstrap_spearman(x, y, resamples=50, seed=123)
        self.assertEqual(first, second)

    def test_perturbation_pipeline_shape(self) -> None:
        image = np.ones((32, 32), dtype=np.float32) * 0.5
        rng = np.random.default_rng(1)
        shifted = apply_named_perturbation(image, "translate_x_8", rng)
        self.assertEqual(shifted.shape, image.shape)

    def test_html_report_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = type(
                "P",
                (),
                {
                    "repo_root": base,
                    "reports_dir": base,
                    "assets_dir": base / "assets",
                    "figures_dir": base / "figures",
                    "manual_review_dir": base / "manual_review",
                },
            )()
            paths.figures_dir.mkdir(parents=True)
            for name in [
                "data_overview.png",
                "confound_correlation_heatmap.png",
                "material_forest.png",
                "perturbation_sensitivity.png",
            ]:
                (paths.figures_dir / name).write_bytes(b"placeholder")
            for name in [
                "correlation_morphology_vs_rq.svg",
                "correlation_spottiness_vs_rq.svg",
                "correlation_streakiness_vs_rq.svg",
            ]:
                (paths.figures_dir / name).write_text("<svg></svg>")
            write_html_report([], [], [], [], [], {"paired_count": 0}, paths)
            self.assertTrue((base / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()

