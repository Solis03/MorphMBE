from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.rheed_roughness.run import convert_height_to_nm
from analysis.rheed_roughness.visualize_manual_pairs import discover_manual_rheed_images
from analysis.rheed_single_frame.connectivity_features import extract_features_for_image
from analysis.rheed_single_frame.data import ExperimentPaths
from analysis.rheed_single_frame.models import ModelSpec, MedianRegressor, conformal_q, evaluate_fixed_models
from analysis.rheed_single_frame.preprocessing import PreprocessedImage
from analysis.rheed_single_frame.removelist import (
    RemovelistAudit,
    RemovelistRecord,
    assert_no_removed_samples,
    discover_removelist,
    load_removelist_audit,
)
from analysis.rheed_single_frame.run import _apply_perturbation
from analysis.rheed_single_frame.visualization import render_prediction_grid


def _paths(root: Path) -> ExperimentPaths:
    for name in ("out", "report", "report/figures", "report/assets", "manual", "afm"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        repo_root=root,
        outputs_dir=root / "out",
        reports_dir=root / "report",
        figures_dir=root / "report" / "figures",
        assets_dir=root / "report" / "assets",
        manual_root=root / "manual",
        plane_corrected_afm_root=root / "afm",
    )


def _audit(root: Path, ids: tuple[str, ...] = ("9999",)) -> RemovelistAudit:
    records = tuple(RemovelistRecord(sample_id=sid, raw_line=sid, note="", source_path=root / "removelist.txt") for sid in ids)
    return RemovelistAudit(path=root / "removelist.txt", sha256="x", mtime="0", parser="test", sample_ids=ids, records=records)


def _dots(connected: bool = False, vertical: bool = False, scale: float = 1.0) -> np.ndarray:
    img = np.zeros((96, 96), dtype=np.float32)
    yy, xx = np.indices(img.shape)
    centers = [(32, 24), (32, 48), (32, 72), (64, 24), (64, 48), (64, 72)]
    for y, x in centers:
        img[((yy - y) ** 2 + (xx - x) ** 2) <= 7**2] = 1.0
    if connected:
        for y in (32, 64):
            img[y - 2 : y + 3, 24:73] = 1.0
    if vertical:
        for x in (24, 48, 72):
            img[32:65, x - 2 : x + 3] = 1.0
    return np.clip(img * scale, 0.0, 1.0)


def _preprocessed(sample_id: str, arr: np.ndarray, root: Path) -> PreprocessedImage:
    path = root / f"{sample_id}.png"
    Image.fromarray(np.asarray(arr * 255, dtype=np.uint8)).save(path)
    return PreprocessedImage(
        sample_id=sample_id,
        manual_rheed_path=path,
        original_rgb=arr,
        cropped_gray=arr,
        gray_padded=arr,
        normalized=arr,
        valid_mask=np.ones_like(arr, dtype=bool),
        audit_row={"original_height": arr.shape[0], "original_width": arr.shape[1], "background_gradient": 0.0, "sharpness": 1.0},
    )


class SingleFrameManualTest(unittest.TestCase):
    def test_canonical_removelist_discovery_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "removelist.txt"
            path.write_text("6023\n6061 - note\n", encoding="utf-8")
            audit = load_removelist_audit(root, "removelist.txt")
            self.assertEqual(audit.sample_ids, ("6023", "6061"))
            self.assertEqual(discover_removelist(root, None), path.resolve())
            with self.assertRaises(FileNotFoundError):
                discover_removelist(root / "empty", None)

    def test_complete_exclusion_assertion_rejects_removed_sample(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_removed_samples(["6022", "6023"], ["6023"], context="unit_test")

    def test_select_discovery_is_deterministic_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "6022" / "RHEED").mkdir(parents=True)
            (root / "16022" / "RHEED").mkdir(parents=True)
            for name in ("select_best.jpg", "select.jpg"):
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "6022" / "RHEED" / name)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "16022" / "RHEED" / "select.jpg")
            rows = discover_manual_rheed_images(root)
            by_id = {row.sample_id: row for row in rows}
            self.assertEqual(by_id["6022"].selected_path.name, "select.jpg")
            self.assertNotIn("6022", [row.sample_id for row in rows if row.manual_folder.name == "16022"])

    def test_rq_unit_conversion(self) -> None:
        values, status = convert_height_to_nm(np.asarray([[0.0, 0.001], [0.002, 0.003]]), "um")
        self.assertEqual(status, "ok")
        centered = values - values.mean()
        self.assertAlmostEqual(float(np.sqrt(np.mean(centered * centered))), 1.11803398875, places=6)

    def test_connectivity_distinguishes_connected_from_isolated_and_brightness_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            config = {"rheed": {"threshold_percentiles": [75, 85, 90], "adaptive_threshold_block_size": 21}}
            isolated = extract_features_for_image(_preprocessed("6001", _dots(False), root), paths, config).features
            connected = extract_features_for_image(_preprocessed("6002", _dots(True), root), paths, config).features
            bright = extract_features_for_image(_preprocessed("6003", _dots(True, scale=0.8), root), paths, config).features
            self.assertGreater(connected["horizontal_connectivity_score"], isolated["horizontal_connectivity_score"])
            self.assertLess(connected["isolation_score"], isolated["isolation_score"])
            self.assertLess(abs(connected["horizontal_connectivity_score"] - bright["horizontal_connectivity_score"]), 0.25)

    def test_grouped_oof_prediction_integrity(self) -> None:
        rows = [
            {"sample_id": "1", "growth_run_id": "g1", "rq_true_nm": 1.0, "log_rq_true": 0.0, "x": 0.0},
            {"sample_id": "2", "growth_run_id": "g2", "rq_true_nm": 2.0, "log_rq_true": math.log10(2.0), "x": 1.0},
            {"sample_id": "3", "growth_run_id": "g3", "rq_true_nm": 3.0, "log_rq_true": math.log10(3.0), "x": 2.0},
        ]
        spec = ModelSpec("median_baseline", "baseline", tuple(), lambda n, p: MedianRegressor(), 0, "test")
        preds, _ = evaluate_fixed_models(rows, [spec], _audit(Path(tempfile.gettempdir())))
        self.assertEqual(sorted(row["sample_id"] for row in preds), ["1", "2", "3"])
        self.assertTrue(all(row["removelist_checked"] for row in preds))

    def test_conformal_interval_reproducible(self) -> None:
        residuals = [0.1, 0.2, 0.3, 0.4]
        self.assertEqual(conformal_q(residuals, 0.90), conformal_q(residuals, 0.90))
        self.assertAlmostEqual(conformal_q(residuals, 0.90), 0.4)

    def test_perturbation_pipeline_keeps_shape(self) -> None:
        arr = _dots(True)
        mask = np.ones_like(arr, dtype=bool)
        perturbed = _apply_perturbation(arr, mask, "translate_x_4", np.random.default_rng(1))
        self.assertEqual(perturbed.shape, arr.shape)
        self.assertTrue(np.isfinite(perturbed).all())

    def test_prediction_figure_rejects_removelist_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            with self.assertRaises(AssertionError):
                render_prediction_grid(
                    [{"sample_id": "9999", "rq_true_nm": 1.0, "rq_pred_nm": 1.0}],
                    [],
                    paths,
                    _audit(root),
                    output_stem="bad",
                    sort_key="true",
                    common_scale=None,
                )


if __name__ == "__main__":
    unittest.main()

