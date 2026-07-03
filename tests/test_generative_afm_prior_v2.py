from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import (
    AFMPriorV2Dataset,
    V2_DESCRIPTOR_NAMES,
    build_condition_matrix_v2,
    compute_afm_descriptors_v2,
    load_v2_index,
)
from rheed2morph.generative.compare_mvp1_mvp3_generation import adapt_mvp2_row_to_mvp3_condition
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.export_afm_latents_v2 import export_latents
from rheed2morph.generative.models.afm_autoencoder_v2 import build_afm_autoencoder_v2
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.prepare_afm_prior_v2_dataset import prepare_dataset
from rheed2morph.generative.train_afm_autoencoder_v2 import reconstruction_loss_v2, save_autoencoder_v2_checkpoint


def _synthetic_afm(index: int, size: int = 160) -> np.ndarray:
    rng = np.random.default_rng(200 + index)
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    image = 0.2 * np.sin((index % 5 + 1) * np.pi * x) + 0.15 * np.cos((index % 4 + 1) * np.pi * y)
    for _ in range(4):
        cx, cy = rng.uniform(-0.7, 0.7, size=2)
        sigma = rng.uniform(0.06, 0.18)
        amp = rng.uniform(0.3, 1.0)
        image += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
    image += rng.normal(0.0, 0.025, size=(size, size))
    return image.astype(np.float32)


class GenerativeAfmPriorV2Test(unittest.TestCase):
    def _make_tree(self, root: Path, groups: int = 10, files_per_group: int = 2) -> Path:
        afm_root = root / "afm"
        for group in range(groups):
            sample = f"{6000 + group}"
            for item in range(files_per_group):
                file_id = f"N{sample}_1um_{item:03d}"
                directory = afm_root / sample / file_id
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f"{file_id}_height.npy", _synthetic_afm(group * files_per_group + item))
        return afm_root

    def _prepare(self, root: Path, patch_mode: str = "none") -> Path:
        afm_root = self._make_tree(root)
        out = root / "prepared"
        prepare_dataset(
            argparse.Namespace(
                out=out,
                afm_root=afm_root,
                manifest=None,
                include_unpaired_afm=True,
                scan_size_filter="1um",
                image_size=128,
                limit=None,
                group_split=True,
                patch_mode=patch_mode,
                patches_per_image=2,
                patch_size=96,
                min_files_required=5,
                strict=True,
                seed=11,
            )
        )
        return out

    def test_v2_data_prep_indexes_synthetic_height_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = self._prepare(Path(tmp_dir))
            self.assertTrue((out / "afm_prior_v2_index.csv").is_file())
            self.assertTrue((out / "afm_prior_v2_descriptors.csv").is_file())
            self.assertTrue((out / "afm_prior_v2_inventory.json").is_file())
            text = (out / "afm_prior_v2_index.csv").read_text(encoding="utf-8")
            self.assertIn("physical_height", text)

    def test_group_split_prevents_crossing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = self._prepare(Path(tmp_dir), patch_mode="deterministic")
            rows = [line.split(",") for line in (out / "afm_prior_v2_index.csv").read_text(encoding="utf-8").strip().splitlines()]
            header = rows[0]
            group_i = header.index("group_id")
            split_i = header.index("split")
            by_group: dict[str, set[str]] = {}
            for row in rows[1:]:
                by_group.setdefault(row[group_i], set()).add(row[split_i])
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))

    def test_patch_generation_keeps_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = self._prepare(Path(tmp_dir), patch_mode="deterministic")
            rows = (out / "afm_prior_v2_index.csv").read_text(encoding="utf-8")
            self.assertIn("patch_000", rows)
            self.assertIn("patch_001", rows)

    def test_descriptor_extraction_finite(self) -> None:
        descriptors = compute_afm_descriptors_v2(_synthetic_afm(0))
        self.assertTrue(set(V2_DESCRIPTOR_NAMES).issubset(descriptors))
        self.assertTrue(all(np.isfinite(value) for value in descriptors.values()))

    def test_autoencoder_v2_forward_and_train_step(self) -> None:
        model = build_afm_autoencoder_v2(image_size=128, latent_channels=4, latent_size=16, base_channels=16)
        images = torch.randn(2, 1, 128, 128).clamp(-1, 1)
        recon, latent = model(images)
        self.assertEqual(recon.shape, images.shape)
        self.assertEqual(latent.shape, (2, 4, 16, 16))
        loss, parts = reconstruction_loss_v2(recon, images)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertIn("histogram_loss", parts)

    def test_latent_export_writes_v2_condition_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prepared = self._prepare(root)
            model = build_afm_autoencoder_v2(image_size=128, latent_channels=4, latent_size=16, base_channels=16)
            checkpoint = root / "ae_v2.pt"
            save_autoencoder_v2_checkpoint(
                checkpoint,
                model,
                None,
                0,
                0.0,
                {"image_size": 128, "latent_channels": 4, "latent_size": 16, "base_channels": 16, "dropout": 0.05},
            )
            out = root / "latents"
            export_latents(
                argparse.Namespace(
                    checkpoint=checkpoint,
                    data_index=prepared / "afm_prior_v2_index.csv",
                    descriptors=prepared / "afm_prior_v2_descriptors.csv",
                    prototypes=prepared / "morphology_prototypes_v2.csv",
                    out=out,
                    batch_size=4,
                    limit=None,
                    device="cpu",
                )
            )
            self.assertTrue((out / "condition_table_v2.csv").is_file())
            self.assertTrue((out / "latent_stats_v2.json").is_file())
            payload = np.load(out / "latents_train.npz")
            self.assertEqual(payload["latents"].shape[1:], (4, 16, 16))

    def test_diffusion_v2_forward_and_ddim_shapes(self) -> None:
        model = LatentUNet(latent_channels=4, condition_dim=6, base_channels=16, emb_dim=64)
        diffusion = GaussianDiffusionV2(timesteps=12, beta_schedule="cosine", prediction_target="epsilon")
        latents = torch.randn(2, 4, 16, 16)
        conditions = torch.randn(2, 6)
        loss = diffusion.training_loss(model, latents, conditions, cond_dropout=0.0)
        self.assertTrue(torch.isfinite(loss))
        sampled = diffusion.sample_ddim(model, (2, 4, 16, 16), conditions, steps=4, guidance_scale=1.0)
        self.assertEqual(sampled.shape, (2, 4, 16, 16))
        decoded = build_afm_autoencoder_v2(image_size=128, latent_channels=4, latent_size=16, base_channels=16).decode(sampled)
        self.assertEqual(decoded.shape, (2, 1, 128, 128))

    def test_adapter_refuses_descriptor_mismatch_without_fill_mean(self) -> None:
        schema = {
            "descriptor_columns": ["rq", "new_descriptor"],
            "condition_columns": ["cond_rq", "cond_new_descriptor"],
            "prototype_count": 0,
            "descriptor_train_mean": {"rq": 1.0, "new_descriptor": 2.0},
            "descriptor_train_std": {"rq": 0.5, "new_descriptor": 1.0},
        }
        row = {"pred_rq": "1.5"}
        with self.assertRaises(ValueError):
            adapt_mvp2_row_to_mvp3_condition(row, schema, "predicted", fill_missing_with_train_mean=False)
        vector, report = adapt_mvp2_row_to_mvp3_condition(row, schema, "predicted", fill_missing_with_train_mean=True)
        self.assertEqual(vector.shape, (2,))
        self.assertEqual(report["filled_descriptor_count"], 1)

    def test_v2_source_has_no_neighbor_retrieval_markers(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph" / "generative"
        files = [path for path in source_root.glob("*v2*.py")] + [source_root / "compare_mvp1_mvp3_generation.py"]
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in files if path.is_file())
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)


if __name__ == "__main__":
    unittest.main()
