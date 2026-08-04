from __future__ import annotations

import unittest
import json
from pathlib import Path

import pandas as pd
import torch

from analysis.rheed_single_frame.removelist import load_removelist_audit
from analysis.rheed_to_afm_generation.data import load_rheed_feature_table
from analysis.rheed_to_afm_generation.model import ConditionalAFMVAE, gaussian_kl
from analysis.rheed_to_afm_generation.run import _load_tables


class ConditionalAFMVAETest(unittest.TestCase):
    def test_forward_and_generation_are_finite_and_unit_rq(self) -> None:
        torch.manual_seed(7)
        model = ConditionalAFMVAE(
            condition_dim=5,
            latent_dim=4,
            resolution=64,
            base_channels=4,
        )
        image = torch.randn(2, 1, 64, 64)
        condition = torch.randn(2, 5)
        result = model(image, condition)
        self.assertEqual(result["reconstruction"].shape, image.shape)
        self.assertTrue(torch.isfinite(result["reconstruction"]).all())
        rq = result["reconstruction"].square().mean(dim=(1, 2, 3)).sqrt()
        self.assertTrue(torch.allclose(rq, torch.ones_like(rq), atol=1e-3))

        generated = model.generate(condition)
        self.assertEqual(generated.shape, image.shape)
        self.assertTrue(torch.isfinite(generated).all())

    def test_conditional_prior_changes_with_condition(self) -> None:
        torch.manual_seed(11)
        model = ConditionalAFMVAE(
            condition_dim=3,
            latent_dim=4,
            resolution=64,
            base_channels=4,
        )
        condition_a = torch.zeros(1, 3)
        condition_b = torch.ones(1, 3)
        mean_a, _ = model.conditional_prior(condition_a)
        mean_b, _ = model.conditional_prior(condition_b)
        self.assertFalse(torch.allclose(mean_a, mean_b))

    def test_gaussian_kl_is_zero_for_identical_distributions(self) -> None:
        mean = torch.zeros(3, 4)
        logvar = torch.zeros(3, 4)
        value = gaussian_kl(mean, logvar, mean, logvar)
        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_model_source_has_no_retrieval_dependency(self) -> None:
        import inspect

        source = inspect.getsource(ConditionalAFMVAE).lower()
        self.assertNotIn("nearestneighbor", source)
        self.assertNotIn("retriev", source)

    def test_canonical_removelist_is_applied_before_all_model_tables(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs/rheed_to_afm_generation.json").read_text(
                encoding="utf-8"
            )
        )
        tables = _load_tables(config)
        removed = set(tables["removelist"].sample_ids)
        self.assertEqual(
            tables["removelist"].sha256,
            "87bbce33d0b4e9b9297a8fb447c3581c59a8f3c5402b399042046c094d567c9f",
        )
        self.assertIn("6081", removed)
        for name in ("descriptors", "folds", "physics", "phase1"):
            frame = tables[name]
            for column in ("sample_id", "growth_run_id"):
                if column in frame:
                    self.assertFalse(
                        set(frame[column].dropna().astype(str)) & removed,
                        f"{name}.{column} contains removelist samples",
                    )
        self.assertEqual(
            set(tables["removelist_excluded_rows"]["sample_id"]),
            {"6023", "6081", "6087"},
        )

    def test_removelist_samples_are_removed_from_embedding_payload(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs/rheed_to_afm_generation.json").read_text(
                encoding="utf-8"
            )
        )
        tables = _load_tables(config)
        audit = load_removelist_audit(root, "removelist.txt")
        sample_ids, _, _, _ = load_rheed_feature_table(
            "dino_vits14__centered_8__raw_luminance",
            pd.read_csv(root / config["embedding_registry"]),
            tables["physics"],
            excluded_sample_ids=set(audit.sample_ids),
        )
        self.assertFalse(set(sample_ids) & set(audit.sample_ids))


if __name__ == "__main__":
    unittest.main()
