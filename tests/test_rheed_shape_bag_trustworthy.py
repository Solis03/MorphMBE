from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from rheed2morph.generative.sample_shape_bag_oof_calibrated_v2 import sample as sample_oof
from rheed2morph.rheed.export_shape_bag_oof_predictions import export_predictions
from rheed2morph.rheed.run_shape_bag_strict_descriptor_cv import run_cv
from rheed2morph.rheed.select_production_shape_bag_model import select_model
from rheed2morph.rheed.shape_bag_negative_controls import run_negative_controls
from rheed2morph.rheed.shape_bag_trustworthy_utils import impute_scale_train_val


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    mvp8 = root / "mvp8"
    mvp9 = root / "mvp9"
    data = mvp9 / "data"
    mvp8.mkdir()
    data.mkdir(parents=True)
    condition_schema = root / "condition_schema_v3.json"
    stable_cols = ["weighted_mean_bar_like_score", "weighted_mean_mask_confidence"]
    raw_cols = stable_cols + ["weighted_mean_brightness", "weighted_mean_snr_score"]
    _write(mvp8 / "global_sample_shape_features.csv", "sample_id," + ",".join(raw_cols) + "\n")
    _write(data / "supervised_shape_bag_index.csv", "pair_id,row_id,sample_id,group_id,split,shape_bag_npz,cached_tensor_path,network_input_path,descriptor_height_path,prototype_id," + ",".join(f"shape_feature::{col}" for col in stable_cols) + "\n")
    _write(data / "target_conditions_shape_bag.csv", "pair_id,row_id,sample_id,group_id,split,prototype_id,rq,cond_rq,ra,cond_ra\n")
    _write(data / "strict_fold_assignments.csv", "pair_id,sample_id,group_id,fold_id,strict_split,original_split\n")
    for i in range(8):
        sample = str(7000 + i)
        group = f"g{i // 2}"
        split = "train" if i < 4 else ("val" if i < 6 else "test")
        bag = root / "bags" / sample / "shape_bag.npz"
        bag.parent.mkdir(parents=True)
        frames = np.random.default_rng(i).normal(size=(4, 6, 16, 16)).astype(np.float32)
        np.savez(bag, frames=frames, consensus_maps=frames.mean(axis=0), frame_mask=np.ones(4, dtype=np.float32), frame_weights=np.ones(4, dtype=np.float32) / 4)
        cached = root / "bags" / sample / "cached.npz"
        np.savez(cached, frames=np.random.default_rng(i).normal(size=(4, 1, 16, 16)).astype(np.float32))
        afm = root / "afm" / f"{sample}.npy"
        afm.parent.mkdir(parents=True, exist_ok=True)
        np.save(afm, np.random.default_rng(i).normal(size=(32, 32)).astype(np.float32))
        bar = float(i)
        conf = 1.0 - 0.02 * i
        bright = 100.0 + i
        snr = 0.5 + 0.01 * i
        rq = 1.0 + 0.5 * bar
        ra = 0.25 + 0.1 * i
        with (mvp8 / "global_sample_shape_features.csv").open("a", encoding="utf-8") as handle:
            handle.write(f"{sample},{bar},{conf},{bright},{snr}\n")
        with (data / "supervised_shape_bag_index.csv").open("a", encoding="utf-8") as handle:
            handle.write(f"p{i},r{i},{sample},{group},{split},{bag},{cached},{afm},{afm},{i % 2},{bar},{conf}\n")
        with (data / "target_conditions_shape_bag.csv").open("a", encoding="utf-8") as handle:
            handle.write(f"p{i},r{i},{sample},{group},{split},{i % 2},{rq},{rq - 1.0},{ra},{ra - 0.25}\n")
        with (data / "strict_fold_assignments.csv").open("a", encoding="utf-8") as handle:
            handle.write(f"p{i},{sample},{group},{i % 4},val,{split}\n")
    (data / "feature_schema_shape_bag.json").write_text(
        json.dumps({"stable_feature_columns": stable_cols, "raw_240_feature_columns": raw_cols, "use_raw_240_features_by_default": False}),
        encoding="utf-8",
    )
    schema = {
        "descriptor_columns": ["rq", "ra"],
        "condition_columns": ["cond_rq", "cond_ra"],
        "descriptor_train_mean": {"rq": 1.0, "ra": 0.25},
        "descriptor_train_std": {"rq": 1.0, "ra": 1.0},
        "prototype_count": 2,
    }
    condition_schema.write_text(json.dumps(schema), encoding="utf-8")
    (data / "target_schema_shape_bag.json").write_text(
        json.dumps({"descriptor_columns": ["rq", "ra"], "condition_columns": ["cond_rq", "cond_ra"], "source_condition_schema": str(condition_schema)}),
        encoding="utf-8",
    )
    return mvp8, mvp9, condition_schema


class ShapeBagTrustworthyTest(unittest.TestCase):
    def test_train_fold_scaling_only(self) -> None:
        train = np.asarray([[0.0], [0.0], [0.0]], dtype=np.float32)
        val = np.asarray([[10.0]], dtype=np.float32)
        train_s, val_s, info = impute_scale_train_val(train, val)
        self.assertAlmostEqual(float(info["feature_mean"][0]), 0.0)
        self.assertAlmostEqual(float(info["feature_std"][0]), 1.0)
        self.assertAlmostEqual(float(val_s[0, 0]), 10.0)

    def test_cv_controls_selection_export_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mvp8, mvp9, schema = _make_fixture(root)
            cv_root = root / "mvp10" / "strict_descriptor_cv"
            run_cv(
                argparse.Namespace(
                    mvp8_root=mvp8,
                    mvp9_root=mvp9,
                    condition_schema=schema,
                    out=cv_root,
                    fold_mode="original_mvp9",
                    n_splits=4,
                    n_repeats=1,
                    models="mean,ridge",
                    feature_sets="stable36,brightness_only_diagnostic",
                    target_set="all",
                    nested_hparam=False,
                    save_oof_predictions=True,
                    bootstrap=5,
                    seed=42,
                )
            )
            self.assertTrue((cv_root / "cv_metrics_summary.csv").is_file())
            fold_rows = _rows(cv_root / "cv_predictions_oof.csv")
            self.assertTrue(fold_rows)
            train_groups = {"g0", "g1"}
            val_groups = {row["group_id"] for row in fold_rows}
            self.assertFalse(train_groups & val_groups)
            neg_root = root / "mvp10" / "negative_controls"
            run_negative_controls(
                argparse.Namespace(cv_root=cv_root, mvp8_root=mvp8, mvp9_root=mvp9, out=neg_root, n_permutations=5, seed=42)
            )
            self.assertTrue((neg_root / "negative_control_metrics.csv").is_file())
            fake_neg = root / "mvp10" / "fake_negative_controls"
            fake_neg.mkdir(parents=True)
            (fake_neg / "negative_control_summary.json").write_text(json.dumps({"negative_controls_pass": False}), encoding="utf-8")
            (fake_neg / "negative_control_report.md").write_text("failed\n", encoding="utf-8")
            fi_root = root / "mvp10" / "feature_importance"
            fi_root.mkdir(parents=True)
            _write(fi_root / "grouped_feature_importance.csv", "descriptor,group,mean_importance_delta_mse\nrq,elongated_bar_like,1.0\n")
            prod_root = root / "mvp10" / "production_model_selection"
            select_model(
                argparse.Namespace(cv_root=cv_root, negative_control_root=fake_neg, feature_importance_root=fi_root, out=prod_root)
            )
            selected = json.loads((prod_root / "selected_model_config.json").read_text(encoding="utf-8"))
            self.assertEqual(selected["selected_descriptors"], {})
            policy = json.loads((prod_root / "unsupported_descriptor_policy.json").read_text(encoding="utf-8"))
            self.assertTrue(all(row["production_action"] == "fill_train_mean" for row in policy["policy"]))
            pred_root = root / "mvp10" / "production_predictions"
            export_predictions(
                argparse.Namespace(
                    cv_root=cv_root,
                    production_selection=prod_root / "selected_model_config.json",
                    descriptor_policy=prod_root / "unsupported_descriptor_policy.json",
                    out=pred_root,
                )
            )
            pred_table = pred_root / "predicted_condition_table_oof_production.csv"
            self.assertTrue(pred_table.is_file())
            first = _rows(pred_table)[0]
            self.assertEqual(first["policy_rq"], "filled_by_train_mean")
            gen_root = root / "mvp10" / "trustworthy_generation"
            sample_oof(
                argparse.Namespace(
                    mvp5_root=root,
                    autoencoder=root / "missing.pt",
                    v2_diffusion=root / "missing2.pt",
                    condition_schema=schema,
                    predicted_condition_table=pred_table,
                    shape_bag_index=mvp9 / "data" / "supervised_shape_bag_index.csv",
                    num_samples_per_condition=1,
                    keep_top_k=1,
                    ddim_steps=2,
                    guidance_scale=1.0,
                    calibration_mode="weighted_rq_ra_range",
                    rerank=True,
                    split="val",
                    max_conditions=1,
                    mock=True,
                    device="cpu",
                    seed=42,
                    out=gen_root,
                )
            )
            self.assertTrue((gen_root / "trustworthy_generation_summary.json").is_file())

    def test_no_forbidden_retrieval_api_in_new_path(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph"
        files = [
            source_root / "rheed" / "run_shape_bag_strict_descriptor_cv.py",
            source_root / "rheed" / "shape_bag_feature_importance.py",
            source_root / "rheed" / "shape_bag_negative_controls.py",
            source_root / "rheed" / "compare_manual_vs_auto_shape_bags.py",
            source_root / "rheed" / "select_production_shape_bag_model.py",
            source_root / "rheed" / "export_shape_bag_oof_predictions.py",
            source_root / "rheed" / "generate_shape_bag_evidence_package.py",
            source_root / "generative" / "sample_shape_bag_oof_calibrated_v2.py",
        ]
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)
        self.assertNotIn("knn", joined)


if __name__ == "__main__":
    unittest.main()
