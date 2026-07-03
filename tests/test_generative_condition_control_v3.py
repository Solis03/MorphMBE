from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from rheed2morph.generative.common import read_csv_rows, read_json, write_csv_rows, write_json
from rheed2morph.generative.compare_v2_v3_condition_control import compare
from rheed2morph.generative.condition_control_v3_utils import (
    DEFAULT_V3_DESCRIPTOR_CANDIDATES,
    adapt_external_condition_row,
    condition_row_to_vector,
    create_condition_schema_v3,
)
from rheed2morph.generative.descriptor_guided_sampling import (
    decode_latents,
    ddim_sample_with_descriptor_guidance,
    rerank_decoded_candidates,
)
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.models.afm_autoencoder_v2 import build_afm_autoencoder_v2
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.sample_afm_prior_v3 import _condition_sweep_rows
from rheed2morph.generative.train_latent_descriptor_regressor import LatentDescriptorRegressor


def _schema_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(10):
        split = "train" if index < 8 else "val"
        row = {
            "row_id": f"r{index}",
            "parent_row_id": f"r{index}",
            "sample_id": f"s{index}",
            "group_id": f"g{index // 2}",
            "split": split,
            "network_input_path": f"/tmp/r{index}.npy",
            "descriptor_height_path": f"/tmp/r{index}.npy",
            "source_path": f"/tmp/r{index}.npy",
            "source_kind": "synthetic",
            "prototype_id": str(index % 2),
        }
        for col_index, name in enumerate(DEFAULT_V3_DESCRIPTOR_CANDIDATES):
            value = 0.2 + 0.1 * index + 0.03 * col_index
            row[name] = f"{value:.6f}"
            row[f"cond_{name}"] = "0"
        rows.append(row)
    return rows


def _sine(size: int, amplitude: float) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    return (amplitude * (np.sin(np.pi * x) + np.cos(np.pi * y))).astype(np.float32)


class GenerativeConditionControlV3Test(unittest.TestCase):
    def test_condition_schema_v3_selects_finite_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            table = root / "condition_table_v2.csv"
            prototypes = root / "prototypes.csv"
            descriptors = root / "descriptors.csv"
            write_csv_rows(table, _schema_rows())
            write_csv_rows(prototypes, [{"row_id": f"r{i}", "prototype_id": str(i % 2)} for i in range(10)])
            write_csv_rows(descriptors, [{"row_id": row["row_id"], "rq": row["rq"]} for row in _schema_rows()])
            schema = create_condition_schema_v3(table, descriptors, prototypes, None, root / "schema")
            self.assertTrue((root / "schema" / "condition_schema_v3.json").is_file())
            self.assertTrue((root / "schema" / "condition_table_v3.csv").is_file())
            self.assertIn("rq", schema["descriptor_columns"])
            self.assertEqual(schema["prototype_count"], 2)
            table_rows = read_csv_rows(root / "schema" / "condition_table_v3.csv")
            self.assertIn("cond_rq", table_rows[0])
            self.assertTrue(np.isfinite(float(table_rows[0]["cond_rq"])))

    def test_condition_sweep_changes_target_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            table = root / "condition_table_v2.csv"
            prototypes = root / "prototypes.csv"
            descriptors = root / "descriptors.csv"
            write_csv_rows(table, _schema_rows())
            write_csv_rows(prototypes, [{"row_id": f"r{i}", "prototype_id": str(i % 2)} for i in range(10)])
            write_csv_rows(descriptors, [{"row_id": row["row_id"], "rq": row["rq"]} for row in _schema_rows()])
            schema = create_condition_schema_v3(table, descriptors, prototypes, None, root / "schema")
            rows = read_csv_rows(root / "schema" / "condition_table_v3.csv")
            sweep_rows = _condition_sweep_rows(rows[0], "rq", schema)
            values = [float(row["cond_rq"]) for row in sweep_rows]
            self.assertEqual(len(set(values)), len(values))
            for row in sweep_rows:
                vector = condition_row_to_vector(row, schema)
                self.assertEqual(vector.shape[0], schema["condition_dim"])

    def test_latent_descriptor_regressor_forward_and_train_step(self) -> None:
        model = LatentDescriptorRegressor(latent_channels=4, descriptor_dim=3, prototype_count=2, hidden=16)
        latents = torch.randn(5, 4, 8, 8)
        target = torch.randn(5, 3)
        prototypes = torch.tensor([0, 1, 0, 1, 1])
        pred, logits = model(latents)
        self.assertEqual(pred.shape, (5, 3))
        self.assertIsNotNone(logits)
        loss = torch.nn.functional.mse_loss(pred, target) + torch.nn.functional.cross_entropy(logits, prototypes)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_diffusion_v3_forward_uses_condition_injection(self) -> None:
        model = LatentUNet(latent_channels=4, condition_dim=5, base_channels=16, emb_dim=64)
        diffusion = GaussianDiffusionV2(timesteps=8, beta_schedule="cosine", prediction_target="v")
        latents = torch.randn(2, 4, 8, 8)
        condition = torch.randn(2, 5)
        timesteps = torch.randint(0, diffusion.timesteps, (2,), dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = diffusion.q_sample(latents, timesteps, noise)
        output = model(noisy, timesteps, condition)
        target = diffusion._target(latents, timesteps, noise)
        loss = torch.mean((output - target) ** 2)
        self.assertEqual(output.shape, latents.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.allclose(output, model(noisy, timesteps, torch.zeros_like(condition))))

    def test_descriptor_guided_sampler_and_decode_shapes(self) -> None:
        diffusion_model = LatentUNet(latent_channels=4, condition_dim=3, base_channels=8, emb_dim=32)
        regressor = LatentDescriptorRegressor(latent_channels=4, descriptor_dim=3, prototype_count=0, hidden=16)
        diffusion = GaussianDiffusionV2(timesteps=4, beta_schedule="linear", prediction_target="v")
        condition = torch.randn(2, 3)
        sampled = ddim_sample_with_descriptor_guidance(
            diffusion_model,
            diffusion,
            (2, 4, 8, 8),
            condition,
            descriptor_regressor=regressor,
            descriptor_dim=3,
            steps=2,
            guidance_scale=1.0,
            descriptor_guidance_weight=0.01,
        )
        self.assertEqual(sampled.shape, (2, 4, 8, 8))
        autoencoder = build_afm_autoencoder_v2(image_size=32, latent_channels=4, latent_size=8, base_channels=8)
        decoded = decode_latents(autoencoder, sampled, torch.zeros(1, 4, 8, 8), torch.ones(1, 4, 8, 8))
        self.assertEqual(decoded.shape, (2, 32, 32))

    def test_candidate_reranking_scores_synthetic_candidates(self) -> None:
        good = _sine(64, 0.2)
        bad = _sine(64, 1.0)
        target = {"rq": str(float(np.std(good)))}
        schema = {
            "descriptor_columns": ["rq"],
            "condition_columns": ["cond_rq"],
            "prototype_count": 0,
            "descriptor_train_std": {"rq": 1.0},
        }
        top, metrics = rerank_decoded_candidates(np.stack([bad, good]), target, schema, keep_top_k=1)
        rank_one = [row for row in metrics if row.get("rank") == 1][0]
        self.assertEqual(rank_one["candidate_index"], 1)
        self.assertTrue(np.allclose(top[0], good))

    def test_rheed_condition_adapter_refuses_unsafe_mismatch(self) -> None:
        schema = {
            "descriptor_columns": ["rq", "psd_slope"],
            "condition_columns": ["cond_rq", "cond_psd_slope"],
            "prototype_count": 0,
            "descriptor_train_mean": {"rq": 1.0, "psd_slope": -2.0},
            "descriptor_train_std": {"rq": 0.5, "psd_slope": 1.0},
        }
        row = {"row_id": "r0", "pred_rq": "1.25"}
        with self.assertRaises(ValueError):
            adapt_external_condition_row(row, schema, mode="predicted", fill_missing_with_train_mean=False)
        mapped, report = adapt_external_condition_row(row, schema, mode="predicted", fill_missing_with_train_mean=True)
        self.assertEqual(report["filled_descriptor_count"], 1)
        self.assertIn("cond_psd_slope", mapped)

    def test_v2_v3_comparison_interface_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            v2 = root / "v2"
            v3 = root / "v3"
            v2.mkdir()
            v3.mkdir()
            write_json(
                v2 / "v2_condition_sensitivity_summary.json",
                {"descriptor_summaries": [{"descriptor": "rq", "best_abs_pearson": 0.2, "mae": 0.5, "best_guidance_scale": 1.0}]},
            )
            write_json(
                v3 / "condition_control_summary_v3.json",
                {"reranked_descriptor_error": 0.25, "generated_nonconstant_rate": 1.0},
            )
            write_csv_rows(
                v3 / "condition_control_metrics_v3.csv",
                [{"descriptor": "rq", "pearson": 0.6, "mae": 0.2, "monotonicity": 1.0}],
            )
            summary = compare(argparse.Namespace(v2_sensitivity=v2, v3_evaluation=v3, out=root / "comparison"))
            self.assertEqual(summary["comparable_descriptor_count"], 1)
            self.assertTrue((root / "comparison" / "v2_vs_v3_condition_control_summary.json").is_file())

    def test_v3_condition_control_path_has_no_knn_markers(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src" / "rheed2morph" / "generative"
        files = [
            source_root / "condition_control_v3_utils.py",
            source_root / "analyze_condition_sensitivity_v2.py",
            source_root / "train_latent_descriptor_regressor.py",
            source_root / "train_afm_latent_diffusion_v3.py",
            source_root / "sample_afm_prior_v3.py",
            source_root / "descriptor_guided_sampling.py",
            source_root / "evaluate_condition_control_v3.py",
            source_root / "compare_v2_v3_condition_control.py",
            source_root / "rerun_rheed_conditioned_with_v3_prior.py",
        ]
        joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
        self.assertNotIn("kneighbors", joined)
        self.assertNotIn("nearestneighbors", joined)
        self.assertNotIn("nearest_neighbors", joined)


if __name__ == "__main__":
    unittest.main()
