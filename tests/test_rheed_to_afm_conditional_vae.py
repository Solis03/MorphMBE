from __future__ import annotations

import unittest

import torch

from analysis.rheed_to_afm_generation.model import ConditionalAFMVAE, gaussian_kl


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


if __name__ == "__main__":
    unittest.main()
