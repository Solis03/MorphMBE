from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rheed2morph.generative.common import read_csv_rows, write_csv_rows, write_json
from rheed2morph.generative.prepare_rheed_ssl_dataset import prepare_rheed_ssl_dataset
from rheed2morph.generative.pretrain_rheed_frame_mae import pretrain
from rheed2morph.generative.predict_rheed_conditions_v2 import predict_conditions
from rheed2morph.generative.rheed_ssl_augmentations import RheedSafeAugment
from rheed2morph.generative.run_rheed_morphology_ablation_v2 import run_ablations
from rheed2morph.generative.sample_rheed_conditioned_calibrated_v2 import sample
from rheed2morph.generative.train_rheed_morphology_encoder_v2 import train_encoder
from rheed2morph.generative.models.rheed_mae import build_rheed_mae
from rheed2morph.generative.models.rheed_temporal_encoder import RheedTemporalMorphologyEncoder


class RheedSSLTemporalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._write_synthetic_inputs()
        args = argparse.Namespace(
            out=self.root / "mvp6",
            mvp2_root=self.root,
            mvp2_paired_index=self.mvp2_pairs,
            condition_schema=self.schema_path,
            condition_table=self.condition_table,
            manifest=self.manifest,
            rheed_root=self.root,
            frames=4,
            image_size=16,
            final_fraction=1.0,
            sampling="uniform",
            limit=None,
            strict=True,
            seed=7,
        )
        prepare_rheed_ssl_dataset(args)
        self.data = self.root / "mvp6" / "data"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_synthetic_inputs(self) -> None:
        descriptors = ["rq", "ra", "robust_range"]
        schema = {
            "descriptor_columns": descriptors,
            "condition_columns": [f"cond_{name}" for name in descriptors],
            "descriptor_train_mean": {"rq": 1.0, "ra": 0.8, "robust_range": 3.0},
            "descriptor_train_std": {"rq": 0.5, "ra": 0.4, "robust_range": 1.0},
            "prototype_count": 2,
            "prototype_one_hot": True,
            "condition_dim": 5,
        }
        self.schema_path = self.root / "schema.json"
        write_json(self.schema_path, schema)
        condition_rows = []
        pair_rows = []
        manifest_rows = []
        for index, split in enumerate(["train", "train", "val", "test"], start=1):
            sample = f"10{index:02d}"
            rng = np.random.default_rng(index)
            frames = rng.random((6, 1, 16, 16), dtype=np.float32)
            clip = self.root / f"rheed_{sample}.npz"
            np.savez_compressed(clip, frames=frames)
            afm = rng.normal(size=(16, 16)).astype(np.float32)
            afm_path = self.root / f"afm_{sample}.npy"
            np.save(afm_path, afm)
            raw = {"rq": 0.8 + 0.2 * index, "ra": 0.6 + 0.15 * index, "robust_range": 2.0 + index}
            cond = {f"cond_{name}": (raw[name] - schema["descriptor_train_mean"][name]) / schema["descriptor_train_std"][name] for name in descriptors}
            row = {
                "row_id": f"row_{sample}",
                "sample_id": sample,
                "group_id": sample,
                "split": split,
                "network_input_path": afm_path.as_posix(),
                "descriptor_height_path": afm_path.as_posix(),
                "prototype_id": str(index % 2),
                **{name: str(value) for name, value in raw.items()},
                **{name: str(value) for name, value in cond.items()},
            }
            condition_rows.append(row)
            pair_rows.append({"pair_id": f"pair_{index:05d}", "row_id": f"old_{sample}", "sample_id": sample, "group_id": sample, "split": split, "rheed_video_path": clip.as_posix(), "cached_tensor_path": "", "network_input_path": afm_path.as_posix(), "descriptor_height_path": afm_path.as_posix(), "prototype_id": str(index % 2), **{name: str(value) for name, value in raw.items()}})
            manifest_rows.append({"sample_id": sample, "status": "success", "output_video_path": clip.as_posix(), "source_frame_count": "6", "written_frame_count": "6", "fps": "1"})
        unpaired = self.root / "rheed_1999.npz"
        np.savez_compressed(unpaired, frames=np.zeros((6, 1, 16, 16), dtype=np.float32))
        manifest_rows.append({"sample_id": "1999", "status": "success", "output_video_path": unpaired.as_posix(), "source_frame_count": "6", "written_frame_count": "6", "fps": "1"})
        self.condition_table = self.root / "condition_table.csv"
        self.mvp2_pairs = self.root / "mvp2_pairs.csv"
        self.manifest = self.root / "manifest.csv"
        write_csv_rows(self.condition_table, condition_rows)
        write_csv_rows(self.mvp2_pairs, pair_rows)
        write_csv_rows(self.manifest, manifest_rows)

    def test_dataset_prep_indexes_and_cache_shapes(self) -> None:
        video_rows = read_csv_rows(self.data / "rheed_ssl_video_index.csv")
        frame_rows = read_csv_rows(self.data / "rheed_ssl_frame_index.csv")
        pair_rows = read_csv_rows(self.data / "rheed_supervised_pair_index.csv")
        self.assertEqual(len(pair_rows), 4)
        self.assertEqual(sum(int(row["is_paired"]) == 0 for row in video_rows), 1)
        self.assertEqual(len(frame_rows), 20)
        frames = np.load(Path(pair_rows[0]["cached_tensor_path"]))["frames"]
        self.assertEqual(frames.shape, (4, 1, 16, 16))

    def test_augmentations_and_models_forward(self) -> None:
        x = torch.rand(2, 1, 16, 16)
        aug = RheedSafeAugment()(x)
        self.assertEqual(tuple(aug.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(aug).all())
        mae = build_rheed_mae(image_size=16, patch_size=4, embed_dim=32)
        out = mae(x, mask_ratio=0.5)
        self.assertEqual(tuple(out["reconstruction"].shape), tuple(x.shape))
        out["loss"].backward()
        for pooling in ("mean", "attention", "gru"):
            model = RheedTemporalMorphologyEncoder(3, handcrafted_dim=5, metadata_dim=2, prototype_classes=2, visual_embedding_dim=32, temporal_pooling=pooling)
            y = model(torch.rand(2, 4, 1, 16, 16), torch.rand(2, 5), torch.rand(2, 2))
            self.assertEqual(tuple(y["descriptor"].shape), (2, 3))

    def test_training_prediction_ablation_and_mock_sampling(self) -> None:
        mae_args = argparse.Namespace(
            frame_index=self.data / "rheed_ssl_frame_index.csv",
            out=self.root / "mae",
            image_size=16,
            patch_size=4,
            embed_dim=32,
            depth=1,
            decoder_depth=1,
            mask_ratio=0.5,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            weight_decay=0.0,
            model="small_cnn_mae",
            resume=None,
            limit=8,
            num_workers=0,
            quick=True,
            device="cpu",
            amp=False,
            seed=3,
        )
        pretrain(mae_args)
        train_args = argparse.Namespace(
            paired_index=self.data / "rheed_supervised_pair_index.csv",
            condition_schema=self.schema_path,
            mvp5_root=self.root,
            out=self.root / "encoder",
            frames=4,
            image_size=16,
            frame_encoder="small_cnn",
            frame_mae_checkpoint=self.root / "mae" / "checkpoints" / "best.pt",
            freeze_frame_encoder=False,
            temporal_pooling="attention",
            use_visual=True,
            use_handcrafted=True,
            use_metadata=True,
            predict_uncertainty=True,
            target_schema="v3",
            loss="heteroscedastic",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            limit=None,
            label_fraction=1.0,
            shuffle_labels=False,
            shuffle_videos=False,
            quick=True,
            num_workers=0,
            device="cpu",
            amp=False,
            seed=5,
        )
        train_encoder(train_args)
        predict_args = argparse.Namespace(
            checkpoint=self.root / "encoder" / "checkpoints" / "best.pt",
            paired_index=self.data / "rheed_supervised_pair_index.csv",
            condition_schema=self.schema_path,
            split="val",
            out=self.root / "pred",
            batch_size=2,
            device="cpu",
        )
        predict_conditions(predict_args)
        self.assertTrue((self.root / "pred" / "predicted_condition_table_val.csv").is_file())
        ablation_args = argparse.Namespace(
            paired_index=self.data / "rheed_supervised_pair_index.csv",
            condition_schema=self.schema_path,
            mvp5_root=self.root,
            frame_mae_checkpoint=self.root / "mae" / "checkpoints" / "best.pt",
            out=self.root / "ablations",
            epochs=1,
            label_efficiency_epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.0,
            frames=4,
            image_size=16,
            limit=None,
            quick=True,
            full_suite=False,
            device="cpu",
            amp=False,
            seed=11,
        )
        run_ablations(ablation_args)
        self.assertTrue((self.root / "ablations" / "ablation_metrics_v2.csv").is_file())
        sample_args = argparse.Namespace(
            mvp5_root=self.root,
            autoencoder=self.root / "ae.pt",
            v2_diffusion=self.root / "diff.pt",
            v3_diffusion=None,
            predicted_condition_table=self.root / "pred" / "predicted_condition_table_val.csv",
            paired_index=self.data / "rheed_supervised_pair_index.csv",
            condition_schema=self.schema_path,
            primary_generator="calibrated_v2",
            split="val",
            num_samples_per_condition=1,
            keep_top_k=1,
            ddim_steps=1,
            guidance_scale=1.0,
            calibration_mode="weighted_rq_ra_range",
            rerank=True,
            max_conditions=1,
            mock=True,
            out=self.root / "samples",
            device="cpu",
            seed=13,
        )
        sample(sample_args)
        self.assertTrue((self.root / "samples" / "generation_summary_mvp6.json").is_file())

    def test_new_mvp6_path_does_not_call_retrieval_neighbors(self) -> None:
        files = [
            "prepare_rheed_ssl_dataset.py",
            "pretrain_rheed_frame_mae.py",
            "train_rheed_morphology_encoder_v2.py",
            "run_rheed_morphology_ablation_v2.py",
            "predict_rheed_conditions_v2.py",
            "sample_rheed_conditioned_calibrated_v2.py",
            "evaluate_rheed_ssl_temporal.py",
        ]
        root = Path("src/rheed2morph/generative")
        forbidden = ("k" + "neighbors", "nearest" + "neighbors", "nearest" + "_neighbors")
        for name in files:
            text = (root / name).read_text(encoding="utf-8").lower()
            for term in forbidden:
                self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
