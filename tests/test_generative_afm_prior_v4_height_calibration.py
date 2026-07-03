from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.common import read_json, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import adapt_external_condition_row
from rheed2morph.generative.evaluate_afm_prior_v4 import evaluate_v4
from rheed2morph.generative.height_calibration_v4 import (
    calibrate_generated_afm,
    compute_height_descriptors,
    fit_affine_height_scale,
)
from rheed2morph.generative.models.afm_autoencoder_v2 import build_afm_autoencoder_v2
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.sample_calibrated_v2_v3 import sample_calibrated
from rheed2morph.generative.train_afm_autoencoder_v2 import save_autoencoder_v2_checkpoint


def _image(size: int = 32, amp: float = 1.0) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    return (amp * (np.sin(np.pi * x) + 0.5 * np.cos(np.pi * y))).astype(np.float32)


class GenerativeAfmPriorV4HeightCalibrationTest(unittest.TestCase):
    def test_closed_form_scale_matches_requested_rq(self) -> None:
        image = _image(64, amp=1.0)
        rq = float(np.std(image))
        fit = fit_affine_height_scale(image, {"rq": rq * 3.0}, weights={"rq": 1.0})
        self.assertAlmostEqual(float(fit["scale_nm_per_unit"]), 3.0, places=5)
        calibrated, result = calibrate_generated_afm(image, {"rq": rq * 2.0}, calibration_mode="rq_only")
        self.assertAlmostEqual(float(np.std(calibrated)), rq * 2.0, places=4)
        self.assertTrue(np.isfinite(result.calibration_error))

    def test_calibration_handles_near_constant_safely(self) -> None:
        image = np.ones((32, 32), dtype=np.float32) * 0.01
        calibrated, result = calibrate_generated_afm(
            image,
            {"rq": "2.0", "ra": "1.0", "robust_range": "3.0"},
            calibration_mode="weighted_rq_ra_range",
            scale_bounds={"scale_low": 0.5, "scale_high": 4.0, "scale_median": 2.0},
        )
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertGreater(result.scale_nm_per_unit, 0.0)

    def test_calibration_clamps_scale_to_bounds(self) -> None:
        image = _image(32, amp=0.01)
        _calibrated, result = calibrate_generated_afm(
            image,
            {"rq": "100.0"},
            calibration_mode="rq_only",
            scale_bounds={"scale_low": 0.5, "scale_high": 2.0, "scale_median": 1.0},
        )
        self.assertEqual(result.scale_nm_per_unit, 2.0)
        self.assertTrue(result.clamped)

    def test_descriptor_computation_before_after_finite(self) -> None:
        image = _image(64, amp=0.5)
        desc = compute_height_descriptors(image)
        self.assertTrue(np.isfinite(desc["rq"]))
        calibrated, _result = calibrate_generated_afm(image, {"rq": "2.0", "ra": "1.5", "robust_range": "6.0"})
        after = compute_height_descriptors(calibrated)
        self.assertTrue(np.isfinite(after["rq"]))

    def _make_tiny_tree(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        mvp3 = root / "mvp3"
        mvp4 = root / "mvp4"
        latents = mvp3 / "latents_v2"
        data = mvp3 / "data"
        schema_dir = mvp4 / "condition_schema_v3"
        latents.mkdir(parents=True)
        data.mkdir(parents=True)
        schema_dir.mkdir(parents=True)
        network = root / "network.npy"
        physical = root / "physical.npy"
        np.save(network, _image(32, amp=0.8))
        np.save(physical, _image(32, amp=2.5))
        row = {
            "row_id": "r0",
            "parent_row_id": "r0",
            "sample_id": "s0",
            "group_id": "g0",
            "split": "val",
            "network_input_path": str(network),
            "descriptor_height_path": str(physical),
            "source_path": str(physical),
            "source_kind": "synthetic",
            "prototype_id": "",
            "rq": "2.0",
            "cond_rq": "0.0",
            "ra": "1.5",
            "cond_ra": "0.0",
            "robust_range": "6.0",
            "cond_robust_range": "0.0",
        }
        train_row = dict(row)
        train_row["row_id"] = "r1"
        train_row["split"] = "train"
        write_csv_rows(latents / "condition_table_v2.csv", [row, train_row])
        write_csv_rows(schema_dir / "condition_table_v3.csv", [row, train_row])
        schema = {
            "descriptor_columns": ["rq", "ra", "robust_range"],
            "condition_columns": ["cond_rq", "cond_ra", "cond_robust_range"],
            "condition_dim": 3,
            "prototype_count": 0,
            "prototype_one_hot": False,
            "descriptor_train_mean": {"rq": 2.0, "ra": 1.5, "robust_range": 6.0},
            "descriptor_train_std": {"rq": 1.0, "ra": 1.0, "robust_range": 2.0},
        }
        write_json(latents / "condition_schema_v2.json", schema)
        write_json(schema_dir / "condition_schema_v3.json", schema)
        np.savez(latents / "latent_standardization_v2.npz", latent_mean=np.zeros((1, 2, 8, 8), dtype=np.float32), latent_std=np.ones((1, 2, 8, 8), dtype=np.float32))
        run = root / "run"
        write_json(run / "height_diagnosis" / "height_scale_summary.json", {"scale_bounds": {"scale_low": 0.5, "scale_high": 8.0, "scale_median": 2.0}})
        ae = build_afm_autoencoder_v2(image_size=32, latent_channels=2, latent_size=8, base_channels=8)
        ae_ckpt = root / "ae.pt"
        save_autoencoder_v2_checkpoint(ae_ckpt, ae, None, 0, 0.0, {"image_size": 32, "latent_channels": 2, "latent_size": 8, "base_channels": 8, "dropout": 0.05})
        for name in ("v2", "v3"):
            model = LatentUNet(latent_channels=2, condition_dim=3, base_channels=8, emb_dim=32)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "latent_channels": 2,
                        "condition_dim": 3,
                        "base_channels": 8,
                        "emb_dim": 32,
                        "timesteps": 4,
                        "beta_schedule": "linear",
                        "prediction_target": "epsilon",
                        "latent_shape": [2, 8, 8],
                        "latents_dir": str(latents),
                    },
                },
                root / f"{name}.pt",
            )
        return mvp3, mvp4, ae_ckpt, root / "v2.pt", root / "v3.pt"

    def test_sample_calibrated_v2_v3_interface_and_v4_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mvp3, mvp4, ae_ckpt, v2_ckpt, v3_ckpt = self._make_tiny_tree(root)
            out = root / "run" / "calibrated"
            sample_calibrated(
                argparse.Namespace(
                    mvp3_root=mvp3,
                    mvp4_root=mvp4,
                    v2_diffusion=v2_ckpt,
                    v3_diffusion=v3_ckpt,
                    autoencoder=ae_ckpt,
                    condition_table=mvp4 / "condition_schema_v3" / "condition_table_v3.csv",
                    condition_schema=mvp4 / "condition_schema_v3" / "condition_schema_v3.json",
                    split="val",
                    num_samples_per_condition=1,
                    keep_top_k=1,
                    ddim_steps=1,
                    guidance_scale=1.0,
                    calibration_mode="weighted_rq_ra_range",
                    rerank=True,
                    max_conditions=1,
                    allow_extrapolation=False,
                    out=out,
                    device="cpu",
                    seed=7,
                )
            )
            self.assertTrue((out / "calibrated_generation_summary.json").is_file())
            eval_out = root / "eval"
            summary = evaluate_v4(argparse.Namespace(v4_root=root / "run", mvp3_root=mvp3, mvp4_root=mvp4, samples_v4=out, out=eval_out))
            self.assertTrue((eval_out / "afm_prior_v4_summary.json").is_file())
            self.assertIn("recommended_primary_prior", summary)

    def test_v4_candidate_reranking_scores_lower_error_candidate(self) -> None:
        image_good = _image(64, amp=1.0)
        image_bad = _image(64, amp=0.1)
        schema = {"descriptor_columns": ["rq", "ra", "robust_range"], "descriptor_train_std": {"rq": 1.0, "ra": 1.0, "robust_range": 1.0}}
        from rheed2morph.generative.sample_calibrated_v2_v3 import _candidate_rows

        target_desc = compute_height_descriptors(image_good)
        top, metrics, _cal = _candidate_rows(
            "test",
            {
                "row_id": "r0",
                "rq": str(float(target_desc["rq"])),
                "ra": str(float(target_desc["ra"])),
                "robust_range": str(float(target_desc["robust_range"])),
            },
            np.stack([image_bad, image_good]),
            schema,
            {"scale_low": 0.5, "scale_high": 3.0, "scale_median": 1.0},
            "rq_only",
            1,
            False,
        )
        rank_one = [row for row in metrics if row.get("rank") == 1][0]
        self.assertEqual(rank_one["candidate_index"], 1)
        self.assertEqual(top.shape[0], 1)

    def test_condition_adapter_refuses_unsafe_mismatch(self) -> None:
        schema = {
            "descriptor_columns": ["rq", "missing_descriptor"],
            "condition_columns": ["cond_rq", "cond_missing_descriptor"],
            "prototype_count": 0,
            "descriptor_train_mean": {"rq": 1.0, "missing_descriptor": 2.0},
            "descriptor_train_std": {"rq": 1.0, "missing_descriptor": 1.0},
        }
        with self.assertRaises(ValueError):
            adapt_external_condition_row({"pred_rq": "1.2"}, schema, mode="predicted", fill_missing_with_train_mean=False)
        mapped, report = adapt_external_condition_row({"pred_rq": "1.2"}, schema, mode="predicted", fill_missing_with_train_mean=True)
        self.assertIn("cond_missing_descriptor", mapped)
        self.assertEqual(report["filled_descriptor_count"], 1)

    def test_v4_path_has_no_knn_markers(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph" / "generative"
        files = [
            source_root / "height_calibration_v4.py",
            source_root / "analyze_height_normalization_v4.py",
            source_root / "sample_calibrated_v2_v3.py",
            source_root / "sample_afm_prior_v4.py",
            source_root / "evaluate_afm_prior_v4.py",
            source_root / "rerun_rheed_conditioned_with_v4_prior.py",
        ]
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)


if __name__ == "__main__":
    unittest.main()
