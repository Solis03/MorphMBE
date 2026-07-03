from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.models.afm_autoencoder import build_afm_autoencoder
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.models.rheed_condition_encoder import build_rheed_condition_encoder
from rheed2morph.generative.predict_rheed_conditions import predict_conditions
from rheed2morph.generative.prepare_rheed_condition_dataset import prepare_rheed_condition_dataset
from rheed2morph.generative.rheed_features import compute_rheed_features
from rheed2morph.generative.rheed_video import load_rheed_tensor
from rheed2morph.generative.sample_rheed_conditioned_diffusion import sample_rheed_conditioned
from rheed2morph.generative.train_afm_autoencoder import save_autoencoder_checkpoint
from rheed2morph.generative.train_rheed_condition_encoder import train_encoder


def _video(index: int, frames: int = 6, size: int = 32) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    output = []
    for t in range(frames):
        image = np.sin((x + index + t) / 5.0) + np.cos((y - index) / 7.0)
        image += 0.1 * t
        output.append(image.astype(np.float32))
    return np.stack(output, axis=0)


def _afm(index: int, size: int = 128) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    return (np.sin((index + 1) * x) + np.cos((index + 2) * y)).astype(np.float32)


class RheedConditionedDiffusionTest(unittest.TestCase):
    def _make_prepared(self, root: Path) -> Path:
        video_dir = root / "videos"
        afm_dir = root / "afm"
        latents_dir = root / "latents"
        video_dir.mkdir()
        afm_dir.mkdir()
        latents_dir.mkdir()
        manifest = root / "manifest.csv"
        manifest.write_text("sample_id,video_path\n", encoding="utf-8")
        condition = root / "condition_table.csv"
        condition.write_text(
            "row_id,sample_id,group_id,split,network_input_path,descriptor_height_path,prototype_id,rq,cond_rq,ra,cond_ra\n",
            encoding="utf-8",
        )
        for index, sample in enumerate(("6000", "6001", "6002", "6003")):
            video_path = video_dir / f"N{sample}_raw_crop.npz"
            np.savez(video_path, frames=_video(index))
            afm_path = afm_dir / f"{sample}.npy"
            np.save(afm_path, _afm(index))
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(f"{sample},{video_path}\n")
            split = "train" if index < 2 else ("val" if index == 2 else "test")
            with condition.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"r{index},{sample},{sample},{split},{afm_path},{afm_path},{index % 2},"
                    f"{1.0 + index},{0.1 * index},{0.5 + index},{0.2 * index}\n"
                )
        (latents_dir / "latent_stats.json").write_text(
            '{"descriptor_train_mean":{"rq":1.0,"ra":0.5},"descriptor_train_std":{"rq":2.0,"ra":3.0}}',
            encoding="utf-8",
        )
        out = root / "prepared"
        prepare_rheed_condition_dataset(
            argparse.Namespace(
                mvp1_root=root,
                out=out,
                manifest=manifest,
                rheed_root=None,
                afm_data_index=None,
                condition_table=condition,
                latents_dir=latents_dir,
                scan_size_filter="1um",
                frames=4,
                image_size=32,
                final_fraction=0.5,
                sampling="uniform",
                limit=None,
                sample_key_columns="sample_id,group_id,growth_id",
                video_glob="*raw_crop*.mp4",
                allow_unmatched=False,
                strict=True,
                seed=42,
            )
        )
        return out

    def test_prepare_creates_paired_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = self._make_prepared(Path(tmp_dir))
            self.assertTrue((out / "paired_rheed_condition_index.csv").is_file())
            self.assertTrue((out / "condition_schema.json").is_file())
            self.assertTrue((out / "rheed_handcrafted_features.csv").is_file())

    def test_rheed_video_loader_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "video.npz"
            np.savez(path, frames=_video(0))
            tensor, metadata = load_rheed_tensor(path, frames=4, image_size=32, final_fraction=0.5)
            self.assertEqual(tensor.shape, (4, 1, 32, 32))
            self.assertEqual(metadata["frames"], 4)

    def test_handcrafted_features_are_finite(self) -> None:
        tensor = _video(0, frames=4)[..., None].transpose(0, 3, 1, 2)
        features = compute_rheed_features(tensor)
        self.assertTrue(features)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_encoder_forward_and_training_step(self) -> None:
        model = build_rheed_condition_encoder(
            descriptor_dim=2,
            handcrafted_dim=5,
            metadata_dim=0,
            prototype_classes=2,
            visual_backbone="small_cnn",
            temporal_pooling="attention",
            use_visual=True,
            use_handcrafted=True,
            use_metadata=False,
        )
        video = torch.rand(2, 4, 1, 32, 32)
        handcrafted = torch.rand(2, 5)
        target = torch.rand(2, 2)
        output = model(video, handcrafted)
        loss = torch.nn.functional.mse_loss(output["descriptor"], target) + torch.nn.functional.cross_entropy(
            output["prototype_logits"], torch.tensor([0, 1])
        )
        loss.backward()
        self.assertEqual(output["descriptor"].shape, (2, 2))

    def test_training_and_prediction_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared = self._make_prepared(root)
            encoder_out = root / "encoder"
            train_encoder(
                argparse.Namespace(
                    paired_index=prepared / "paired_rheed_condition_index.csv",
                    condition_schema=prepared / "condition_schema.json",
                    out=encoder_out,
                    frames=4,
                    image_size=32,
                    visual_backbone="small_cnn",
                    temporal_pooling="mean",
                    epochs=1,
                    batch_size=2,
                    lr=1e-3,
                    weight_decay=0.0,
                    num_workers=0,
                    limit=None,
                    quick=True,
                    device="cpu",
                    amp=False,
                    use_visual=False,
                    use_handcrafted=True,
                    use_metadata=False,
                    seed=42,
                )
            )
            pred_out = root / "pred"
            predict_conditions(
                argparse.Namespace(
                    checkpoint=encoder_out / "checkpoints" / "best.pt",
                    paired_index=prepared / "paired_rheed_condition_index.csv",
                    condition_schema=prepared / "condition_schema.json",
                    split="val",
                    out=pred_out,
                    batch_size=2,
                    device="cpu",
                )
            )
            self.assertTrue((pred_out / "predicted_condition_table_val.csv").is_file())

    def test_rheed_conditioned_sampling_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared = self._make_prepared(root)
            ae = build_afm_autoencoder(image_size=128, latent_channels=2)
            ae_ckpt = root / "ae.pt"
            save_autoencoder_checkpoint(ae_ckpt, ae, None, 0, 0.0, {"image_size": 128, "latent_channels": 2})
            latents_dir = root / "sample_latents"
            latents_dir.mkdir()
            np.savez(latents_dir / "latent_standardization.npz", latent_mean=np.zeros((1, 2, 16, 16), dtype=np.float32), latent_std=np.ones((1, 2, 16, 16), dtype=np.float32))
            diffusion = LatentUNet(latent_channels=2, condition_dim=4, base_channels=16, emb_dim=64)
            diff_ckpt = root / "diff.pt"
            torch.save(
                {
                    "model_state_dict": diffusion.state_dict(),
                    "config": {
                        "latent_channels": 2,
                        "condition_dim": 4,
                        "base_channels": 16,
                        "emb_dim": 64,
                        "timesteps": 4,
                        "latent_shape": [2, 16, 16],
                        "latents_dir": str(latents_dir),
                        "condition_columns": ["cond_rq", "cond_ra"],
                        "prototype_count": 2,
                    },
                },
                diff_ckpt,
            )
            encoder_out = root / "encoder"
            train_encoder(
                argparse.Namespace(
                    paired_index=prepared / "paired_rheed_condition_index.csv",
                    condition_schema=prepared / "condition_schema.json",
                    out=encoder_out,
                    frames=4,
                    image_size=32,
                    visual_backbone="small_cnn",
                    temporal_pooling="mean",
                    epochs=1,
                    batch_size=2,
                    lr=1e-3,
                    weight_decay=0.0,
                    num_workers=0,
                    limit=None,
                    quick=True,
                    device="cpu",
                    amp=False,
                    use_visual=False,
                    use_handcrafted=True,
                    use_metadata=False,
                    seed=42,
                )
            )
            pred_out = root / "pred"
            predict_conditions(
                argparse.Namespace(
                    checkpoint=encoder_out / "checkpoints" / "best.pt",
                    paired_index=prepared / "paired_rheed_condition_index.csv",
                    condition_schema=prepared / "condition_schema.json",
                    split="val",
                    out=pred_out,
                    batch_size=2,
                    device="cpu",
                )
            )
            sample_out = root / "samples"
            sample_rheed_conditioned(
                argparse.Namespace(
                    diffusion_checkpoint=diff_ckpt,
                    autoencoder_checkpoint=ae_ckpt,
                    predicted_condition_table=pred_out / "predicted_condition_table_val.csv",
                    paired_index=prepared / "paired_rheed_condition_index.csv",
                    split="val",
                    num_samples_per_condition=1,
                    ddim_steps=2,
                    guidance_scale=1.0,
                    max_conditions=1,
                    out=sample_out,
                    device="cpu",
                    seed=42,
                )
            )
            self.assertTrue((sample_out / "generation_summary.json").is_file())
            self.assertTrue((sample_out / "rheed_conditioned_sample_grid_val.png").is_file())

    def test_new_mvp2_code_does_not_use_neighbor_retrieval(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph" / "generative"
        mvp2_files = [
            "prepare_rheed_condition_dataset.py",
            "rheed_video.py",
            "rheed_features.py",
            "train_rheed_condition_encoder.py",
            "predict_rheed_conditions.py",
            "sample_rheed_conditioned_diffusion.py",
            "evaluate_rheed_conditioned_generation.py",
            "models/rheed_condition_encoder.py",
        ]
        joined = "\n".join((source_root / name).read_text(encoding="utf-8").lower() for name in mvp2_files)
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)


if __name__ == "__main__":
    unittest.main()
