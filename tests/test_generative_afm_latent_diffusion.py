from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.afm_descriptors import DESCRIPTOR_NAMES, compute_afm_descriptors
from rheed2morph.generative.common import AFMImageDataset, load_data_index
from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.export_afm_latents import export_latents
from rheed2morph.generative.losses import reconstruction_loss
from rheed2morph.generative.models.afm_autoencoder import build_afm_autoencoder
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.prepare_afm_latent_dataset import prepare_dataset
from rheed2morph.generative.train_afm_autoencoder import save_autoencoder_checkpoint


def _synthetic_afm(index: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(100 + index)
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    image = 0.15 * np.sin((index + 1) * np.pi * x) + 0.10 * np.cos((index + 2) * np.pi * y)
    for blob_index in range(3):
        cx, cy = rng.uniform(-0.6, 0.6, size=2)
        sigma = rng.uniform(0.08, 0.22)
        amp = rng.uniform(0.4, 1.0) * (1.0 + 0.1 * index)
        image += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
    image += rng.normal(0.0, 0.03, size=(size, size))
    return image.astype(np.float32)


class GenerativeAfmLatentDiffusionTest(unittest.TestCase):
    def _make_dataset(self, root: Path) -> tuple[Path, Path]:
        afm_dir = root / "afm"
        afm_dir.mkdir()
        manifest = root / "manifest.csv"
        lines = ["sample_id,group_id,afm_path,scan_size\n"]
        for index in range(8):
            path = afm_dir / f"sample_{index}.npy"
            np.save(path, _synthetic_afm(index))
            lines.append(f"s{index},g{index},{path},1um\n")
        manifest.write_text("".join(lines), encoding="utf-8")
        out = root / "prepared"
        prepare_dataset(
            argparse.Namespace(
                out=out,
                manifest=manifest,
                afm_root=None,
                scan_size_filter="1um",
                image_size=128,
                limit=None,
                seed=7,
            )
        )
        return manifest, out

    def test_data_preparation_produces_index_and_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _manifest, out = self._make_dataset(Path(tmp_dir))
            self.assertTrue((out / "data_index.csv").is_file())
            self.assertTrue((out / "afm_descriptors.csv").is_file())
            self.assertTrue((out / "descriptor_scaler.json").is_file())
            rows = (out / "afm_descriptors.csv").read_text(encoding="utf-8")
            self.assertIn("rq", rows)

    def test_descriptor_extraction_returns_finite_values(self) -> None:
        descriptors = compute_afm_descriptors(_synthetic_afm(0))
        self.assertEqual(sorted(descriptors), sorted(DESCRIPTOR_NAMES))
        self.assertTrue(all(np.isfinite(value) for value in descriptors.values()))

    def test_autoencoder_forward_and_one_training_step(self) -> None:
        model = build_afm_autoencoder(image_size=128, latent_channels=4)
        images = torch.randn(2, 1, 128, 128)
        recon, latent = model(images)
        self.assertEqual(recon.shape, images.shape)
        self.assertEqual(latent.shape, (2, 4, 16, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, parts = reconstruction_loss(recon, images.clamp(-1, 1))
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertIn("gradient_l1", parts)

    def test_latent_export_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _manifest, prepared = self._make_dataset(root)
            model = build_afm_autoencoder(image_size=128, latent_channels=4)
            checkpoint = root / "ae.pt"
            save_autoencoder_checkpoint(
                checkpoint,
                model,
                None,
                epoch=0,
                best_val_loss=0.0,
                config={"image_size": 128, "latent_channels": 4},
            )
            out = root / "latents"
            export_latents(
                argparse.Namespace(
                    checkpoint=checkpoint,
                    data_index=prepared / "data_index.csv",
                    descriptors=prepared / "afm_descriptors.csv",
                    out=out,
                    batch_size=4,
                    limit=None,
                    device="cpu",
                )
            )
            self.assertTrue((out / "latents_train.npz").is_file())
            self.assertTrue((out / "latents_val.npz").is_file())
            self.assertTrue((out / "condition_table.csv").is_file())
            payload = np.load(out / "latents_train.npz")
            self.assertEqual(payload["latents"].shape[1:], (4, 16, 16))

    def test_diffusion_forward_and_sampling_shapes(self) -> None:
        model = LatentUNet(latent_channels=4, condition_dim=5, base_channels=16, emb_dim=64)
        diffusion = GaussianDiffusion(timesteps=10)
        latents = torch.randn(2, 4, 16, 16)
        condition = torch.randn(2, 5)
        loss = diffusion.training_loss(model, latents, condition, cond_dropout=0.0)
        self.assertTrue(torch.isfinite(loss))
        sampled = diffusion.sample_ddim(model, (2, 4, 16, 16), condition, steps=3, guidance_scale=1.0)
        self.assertEqual(sampled.shape, (2, 4, 16, 16))
        autoencoder = build_afm_autoencoder(image_size=128, latent_channels=4)
        decoded = autoencoder.decode(sampled)
        self.assertEqual(decoded.shape, (2, 1, 128, 128))

    def test_dataset_loader_reads_prepared_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _manifest, prepared = self._make_dataset(Path(tmp_dir))
            records = load_data_index(prepared / "data_index.csv", split="train")
            dataset = AFMImageDataset(records)
            image, meta = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 128, 128))
            self.assertIn("sample_id", meta)

    def test_new_generative_path_does_not_use_nearest_neighbor_code(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph" / "generative"
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in source_root.rglob("*.py"))
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)


if __name__ == "__main__":
    unittest.main()
