from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.sample_shape_bag_calibrated_v2 import sample as sample_shape_bag_calibrated
from rheed2morph.rheed.build_shape_bag_supervised_dataset import build_dataset
from rheed2morph.rheed.exposure_invariance_training import compute_exposure_invariance_loss
from rheed2morph.rheed.models.shape_bag_morphology_predictor import RHEEDShapeBagMorphologyPredictor, exposure_consistency_loss
from rheed2morph.rheed.predict_shape_bag_conditions import predict_conditions
from rheed2morph.rheed.run_shape_bag_ablation import run_ablations
from rheed2morph.rheed.train_shape_bag_morphology_predictor import train_model


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _shape_bag(path: Path, index: int, k: int = 4, size: int = 32) -> None:
    rng = np.random.default_rng(index)
    frames = rng.normal(size=(k, 6, size, size)).astype(np.float32)
    consensus = frames.mean(axis=0).astype(np.float32)
    mask = np.ones(k, dtype=np.float32)
    weights = np.linspace(1.0, 2.0, k, dtype=np.float32)
    weights = weights / weights.sum()
    np.savez(path, frames=frames, consensus_maps=consensus, frame_mask=mask, frame_weights=weights)


def _afm(path: Path, index: int, size: int = 32) -> None:
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    np.save(path, (np.sin((index + 1) * x) + np.cos((index + 2) * y)).astype(np.float32))


class ShapeBagModelTest(unittest.TestCase):
    def _make_synthetic(self, root: Path, count: int = 6) -> dict[str, Path]:
        shape_root = root / "shape"
        afm_root = root / "afm"
        shape_root.mkdir()
        afm_root.mkdir()
        manifest = root / "rheed_shape_bag_manifest.csv"
        features = root / "global_sample_shape_features.csv"
        stable = root / "default_training_feature_names.txt"
        paired = root / "rheed_supervised_pair_index.csv"
        schema = root / "condition_schema_v3.json"
        condition = root / "condition_table_v3.csv"
        _write(
            manifest,
            "sample_id,shape_bag_npz,shape_input_folder,preview_grid,candidate_csv\n",
        )
        _write(
            features,
            "sample_id,weighted_mean_elongation,std_elongation,weighted_mean_bar_fraction,weighted_mean_brightness,weighted_mean_unstable_count\n",
        )
        _write(
            paired,
            "pair_id,row_id,sample_id,group_id,split,cached_tensor_path,network_input_path,descriptor_height_path,prototype_id,rq,cond_rq,ra,cond_ra,robust_range,cond_robust_range,psd_slope,cond_psd_slope,autocorrelation_length_px,cond_autocorrelation_length_px,gradient_anisotropy,cond_gradient_anisotropy,island_count,cond_island_count\n",
        )
        _write(condition, "row_id,sample_id,group_id,split,prototype_id,rq,cond_rq,ra,cond_ra\n")
        stable.write_text("elongation\nbar_fraction\n", encoding="utf-8")
        for i in range(count):
            sample = str(6000 + i)
            bag_dir = shape_root / sample
            bag_dir.mkdir()
            bag = bag_dir / "shape_bag.npz"
            cached = bag_dir / "cached.npz"
            afm = afm_root / f"{sample}.npy"
            _shape_bag(bag, i)
            np.savez(cached, frames=np.random.default_rng(i).normal(size=(4, 1, 32, 32)).astype(np.float32))
            _afm(afm, i)
            split = "train" if i < count - 2 else "val"
            group = f"growth_{i // 2}"
            rq = 1.0 + 0.2 * i
            ra = 0.5 + 0.1 * i
            rr = 2.0 + 0.3 * i
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample},{bag},{bag_dir},,\n")
            with features.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample},{0.2 + i},{0.02 * i},{0.1 * i},{10 + i},{100 + i}\n")
            with paired.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"p{i},r{i},{sample},{group},{split},{cached},{afm},{afm},{i % 2},"
                    f"{rq},{rq - 1.0},{ra},{ra - 0.5},{rr},{rr - 2.0},{-3.0 + 0.1 * i},{0.1 * i},"
                    f"{5 + i},{0.2 * i},{0.3 + 0.01 * i},{0.05 * i},{10 + i},{0.3 * i}\n"
                )
            with condition.open("a", encoding="utf-8") as handle:
                handle.write(f"r{i},{sample},{group},{split},{i % 2},{rq},{rq - 1.0},{ra},{ra - 0.5}\n")
        schema.write_text(
            json.dumps(
                {
                    "descriptor_columns": ["rq", "ra", "robust_range", "psd_slope", "autocorrelation_length_px", "gradient_anisotropy", "island_count"],
                    "condition_columns": ["cond_rq", "cond_ra", "cond_robust_range", "cond_psd_slope", "cond_autocorrelation_length_px", "cond_gradient_anisotropy", "cond_island_count"],
                    "descriptor_train_mean": {
                        "rq": 1.0,
                        "ra": 0.5,
                        "robust_range": 2.0,
                        "psd_slope": -3.0,
                        "autocorrelation_length_px": 5.0,
                        "gradient_anisotropy": 0.3,
                        "island_count": 10.0,
                    },
                    "descriptor_train_std": {
                        "rq": 1.0,
                        "ra": 1.0,
                        "robust_range": 1.0,
                        "psd_slope": 1.0,
                        "autocorrelation_length_px": 1.0,
                        "gradient_anisotropy": 1.0,
                        "island_count": 1.0,
                    },
                    "prototype_count": 2,
                }
            ),
            encoding="utf-8",
        )
        return {"manifest": manifest, "features": features, "stable": stable, "paired": paired, "schema": schema, "condition": condition}

    def _build(self, root: Path) -> Path:
        files = self._make_synthetic(root)
        out = root / "mvp9" / "data"
        build_dataset(
            argparse.Namespace(
                shape_bag_manifest=files["manifest"],
                shape_features=files["features"],
                stable_feature_list=files["stable"],
                paired_index=files["paired"],
                condition_table=files["condition"],
                condition_schema=files["schema"],
                out=out,
                mvp8_root="",
                mvp5_root="",
                mvp6_root="",
                target_schema="v3",
                split_mode="group_kfold",
                n_splits=3,
                include_metadata=False,
                use_raw_240_features=False,
                strict=False,
                limit=None,
                seed=42,
            )
        )
        return out

    def _train_args(self, data: Path, out: Path) -> argparse.Namespace:
        return argparse.Namespace(
            supervised_index=data / "supervised_shape_bag_index.csv",
            target_table=data / "target_conditions_shape_bag.csv",
            folds=data / "strict_fold_assignments.csv",
            feature_schema=data / "feature_schema_shape_bag.json",
            target_schema=data / "target_schema_shape_bag.json",
            out=out,
            model="shape_bag_fusion",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            device="cpu",
            fold_id="original_split",
            quick=True,
            amp=False,
            predict_uncertainty=True,
            use_frames=False,
            use_consensus=False,
            use_stable_features=True,
            use_raw_240_features=False,
            use_metadata=False,
            freeze_frame_branch=False,
            frame_dropout=0.0,
            channel_dropout=0.0,
            exposure_invariance_weight=0.0,
            loss="mse",
            early_stop_patience=2,
            model_image_size=32,
            hidden_dim=16,
            embedding_dim=32,
            shuffle_labels=False,
            seed=42,
        )

    def test_dataset_builder_pairs_and_group_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = self._build(Path(tmp))
            rows = _csv_rows(data / "supervised_shape_bag_index.csv")
            folds = _csv_rows(data / "strict_fold_assignments.csv")
            self.assertEqual(len(rows), 6)
            for fold in {row["fold_id"] for row in folds}:
                val_groups = {row["group_id"] for row in folds if row["fold_id"] == fold}
                train_groups = {row["group_id"] for row in folds if row["fold_id"] != fold}
                self.assertFalse(val_groups & train_groups)

    def test_stable_features_are_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = self._build(Path(tmp))
            schema = json.loads((data / "feature_schema_shape_bag.json").read_text(encoding="utf-8"))
            self.assertFalse(schema["use_raw_240_features_by_default"])
            self.assertLess(len(schema["stable_feature_columns"]), len(schema["raw_240_feature_columns"]))
            self.assertNotIn("weighted_mean_brightness", schema["stable_feature_columns"])

    def test_model_forward_variable_k_and_metadata(self) -> None:
        model = RHEEDShapeBagMorphologyPredictor(target_dim=3, prototype_count=2, stable_feature_dim=4, metadata_dim=2, hidden_dim=16, embedding_dim=32, use_metadata=True)
        frames = torch.rand(2, 5, 6, 32, 32)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
        weights = torch.ones(2, 5)
        out = model(frames=frames, frame_mask=mask, frame_weights=weights, consensus_maps=torch.rand(2, 6, 32, 32), stable_shape_features=torch.rand(2, 4), metadata=torch.rand(2, 2))
        self.assertEqual(out["descriptor_mean"].shape, (2, 3))
        loss = exposure_consistency_loss(out, out)
        self.assertTrue(torch.isfinite(loss))

    def test_training_prediction_ablation_and_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self._build(root)
            train_out = root / "mvp9" / "shape_bag_predictor"
            train_model(self._train_args(data, train_out))
            self.assertTrue((train_out / "checkpoints" / "best.pt").is_file())
            pred_out = root / "mvp9" / "predicted_conditions"
            predict_conditions(
                argparse.Namespace(
                    checkpoint=train_out / "checkpoints" / "best.pt",
                    supervised_index=data / "supervised_shape_bag_index.csv",
                    target_table=data / "target_conditions_shape_bag.csv",
                    folds=data / "strict_fold_assignments.csv",
                    feature_schema=data / "feature_schema_shape_bag.json",
                    target_schema=data / "target_schema_shape_bag.json",
                    split="val",
                    fold_id=None,
                    out_of_fold=False,
                    use_best_ablation=False,
                    batch_size=2,
                    model_image_size=32,
                    device="cpu",
                    out=pred_out,
                )
            )
            pred_table = pred_out / "predicted_condition_table_val.csv"
            self.assertTrue(pred_table.is_file())
            run_ablations(
                argparse.Namespace(
                    data_root=data,
                    out=root / "mvp9" / "ablations",
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    amp=False,
                    quick=True,
                    full_suite=False,
                    seed=42,
                )
            )
            self.assertTrue((root / "mvp9" / "ablations" / "ablation_metrics_shape_bag.csv").is_file())
            sample_shape_bag_calibrated(
                argparse.Namespace(
                    mvp5_root=root,
                    autoencoder=root / "missing_ae.pt",
                    v2_diffusion=root / "missing_diff.pt",
                    v3_diffusion=None,
                    condition_schema=root / "condition_schema_v3.json",
                    predicted_condition_table=pred_table,
                    shape_bag_index=data / "supervised_shape_bag_index.csv",
                    primary_generator="calibrated_v2",
                    split="val",
                    num_samples_per_condition=1,
                    keep_top_k=1,
                    ddim_steps=2,
                    guidance_scale=1.0,
                    calibration_mode="weighted_rq_ra_range",
                    rerank=True,
                    max_conditions=1,
                    mock=True,
                    device="cpu",
                    seed=42,
                    out=root / "mvp9" / "shape_bag_calibrated_v2_generation",
                )
            )
            self.assertTrue((root / "mvp9" / "shape_bag_calibrated_v2_generation" / "shape_bag_calibrated_v2_grid_val.png").is_file())
            self.assertTrue((root / "mvp9" / "shape_bag_calibrated_v2_generation" / "generation_summary_shape_bag.json").is_file())

    def test_exposure_invariance_loss_is_finite(self) -> None:
        model = RHEEDShapeBagMorphologyPredictor(target_dim=2, prototype_count=0, stable_feature_dim=3, hidden_dim=16, embedding_dim=32)
        batch = {
            "frames": torch.rand(2, 4, 6, 32, 32),
            "frame_mask": torch.ones(2, 4),
            "frame_weights": torch.ones(2, 4) / 4,
            "consensus_maps": torch.rand(2, 6, 32, 32),
            "shape_features": torch.rand(2, 3),
            "metadata": torch.zeros(2, 0),
        }
        loss = compute_exposure_invariance_loss(model, batch, device=torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))

    def test_no_knn_or_retrieval_terms_in_new_path(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph"
        files = [
            root / "rheed" / "build_shape_bag_supervised_dataset.py",
            root / "rheed" / "models" / "shape_bag_morphology_predictor.py",
            root / "rheed" / "train_shape_bag_morphology_predictor.py",
            root / "rheed" / "run_shape_bag_ablation.py",
            root / "rheed" / "predict_shape_bag_conditions.py",
            root / "rheed" / "evaluate_shape_bag_model.py",
            root / "generative" / "sample_shape_bag_calibrated_v2.py",
        ]
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)
        self.assertNotIn("knn", joined)


if __name__ == "__main__":
    unittest.main()
