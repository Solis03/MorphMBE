from __future__ import annotations

import unittest

import torch

from analysis.rheed_to_afm_island_generation.guided_diffusion import (
    StructureGuidedDiffusion,
    StructureGuidedResidualUNet,
)


class IslandDiffusionTest(unittest.TestCase):
    def test_forward_loss_and_sampling_are_finite(self) -> None:
        model = StructureGuidedResidualUNet(
            base_channels=8, embedding_dim=32
        )
        diffusion = StructureGuidedDiffusion(timesteps=8)
        residual = torch.randn(2, 1, 32, 32)
        guide = torch.randn(2, 1, 32, 32)
        loss = diffusion.training_loss(model, residual, guide)
        self.assertTrue(torch.isfinite(loss))
        sample = diffusion.sample(model, guide, steps=3, seed=17)
        self.assertEqual(sample.shape, residual.shape)
        self.assertTrue(torch.isfinite(sample).all())
        weak = diffusion.sample(
            model, guide, steps=3, seed=17, strength=0.25
        )
        self.assertEqual(weak.shape, residual.shape)
        self.assertTrue(torch.isfinite(weak).all())

    def test_sampling_rejects_invalid_strength(self) -> None:
        model = StructureGuidedResidualUNet(
            base_channels=8, embedding_dim=32
        )
        diffusion = StructureGuidedDiffusion(timesteps=8)
        guide = torch.randn(1, 1, 32, 32)
        with self.assertRaises(ValueError):
            diffusion.sample(
                model, guide, steps=3, seed=17, strength=0.0
            )


if __name__ == "__main__":
    unittest.main()
