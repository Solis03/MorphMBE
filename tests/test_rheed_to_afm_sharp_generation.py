from __future__ import annotations

import unittest

import numpy as np
import torch

from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.adversarial import (
    MorphologyConditionLoss,
    PhysicsSeededAFMRefiner,
    ProjectionDiscriminator,
    calibrate_random_fields,
)
from analysis.rheed_to_afm_sharp_generation.spectral import (
    PSD_BINS,
    QUANTILE_LEVELS,
    shape_parameters,
    synthesize_random_field,
)


class SharpAFMGenerationTest(unittest.TestCase):
    def test_spectral_random_field_is_stochastic_finite_and_unit_rq(self) -> None:
        rng = np.random.default_rng(4)
        reference = rng.normal(size=(64, 64)).astype(np.float32)
        parameters = shape_parameters(reference)
        self.assertEqual(parameters.shape, (PSD_BINS + len(QUANTILE_LEVELS),))
        generated_a = synthesize_random_field(
            parameters, resolution=64, seed=10, iterations=3
        )
        generated_b = synthesize_random_field(
            parameters, resolution=64, seed=11, iterations=3
        )
        self.assertTrue(np.isfinite(generated_a).all())
        self.assertAlmostEqual(float(np.sqrt(np.mean(generated_a**2))), 1.0, places=4)
        self.assertFalse(np.array_equal(generated_a, generated_b))
        self.assertFalse(np.array_equal(generated_a, reference))

    def test_refiner_has_no_upsampling_and_is_translation_equivariant(self) -> None:
        torch.manual_seed(3)
        model = PhysicsSeededAFMRefiner(
            condition_dim=5, channels=8, residual_scale=0.5
        )
        seed = torch.randn(2, 1, 64, 64)
        condition = torch.randn(2, 5)
        generated = model(seed, condition)
        shifted = model(torch.roll(seed, (7, -5), dims=(-2, -1)), condition)
        expected = torch.roll(generated, (7, -5), dims=(-2, -1))
        self.assertEqual(generated.shape, seed.shape)
        self.assertTrue(torch.allclose(shifted, expected, atol=2e-5, rtol=2e-5))
        rq = generated.square().mean(dim=(1, 2, 3)).sqrt()
        self.assertTrue(torch.allclose(rq, torch.ones_like(rq), atol=1e-3))
        self.assertNotIn("interpolate", str(model).lower())

    def test_projection_discriminator_and_morphology_loss_are_finite(self) -> None:
        torch.manual_seed(8)
        discriminator = ProjectionDiscriminator(condition_dim=9, base_channels=4)
        images = torch.randn(3, 1, 64, 64)
        conditions = torch.randn(3, 9)
        score = discriminator(images, conditions)
        self.assertEqual(score.shape, (3,))
        self.assertTrue(torch.isfinite(score).all())
        scaler = ConditionScaler(
            columns=[
                "log_rq_nm",
                "unit_ra",
                "unit_psd_mid_fraction",
                "unit_psd_high_fraction",
                "unit_psd_slope",
                "log_unit_autocorr_length_nm",
                "log_unit_anisotropy_ratio",
                "unit_skewness",
                "unit_kurtosis",
            ],
            mean=np.zeros(9),
            scale=np.ones(9),
            lower=-np.ones(9) * 10,
            upper=np.ones(9) * 10,
        )
        loss = MorphologyConditionLoss(scaler, 64)(images, conditions)
        self.assertTrue(torch.isfinite(loss))

    def test_descriptor_calibration_reduces_condition_loss(self) -> None:
        rng = np.random.default_rng(19)
        fields = rng.normal(size=(2, 32, 32)).astype(np.float32)
        columns = [
            "log_rq_nm",
            "unit_ra",
            "unit_psd_mid_fraction",
            "unit_psd_high_fraction",
            "unit_psd_slope",
            "log_unit_autocorr_length_nm",
            "log_unit_anisotropy_ratio",
            "unit_skewness",
            "unit_kurtosis",
        ]
        scaler = ConditionScaler(
            columns=columns,
            mean=np.array([0.0, 0.8, 0.25, 0.1, -1.0, 3.0, 0.0, 0.0, 3.0]),
            scale=np.array([1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 1.0, 1.0]),
            lower=np.ones(9) * -10,
            upper=np.ones(9) * 10,
        )
        target_raw = np.array(
            [0.0, 0.75, 0.15, 0.03, -2.0, 3.0, 0.0, 0.6, 4.0]
        )
        target = scaler.transform(target_raw[None], clip=False)[0]
        calibrated, history = calibrate_random_fields(
            fields,
            target,
            condition_scaler=scaler,
            device=torch.device("cpu"),
            steps=20,
            learning_rate=0.02,
            content_weight=0.05,
        )
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(
            np.allclose(
                np.sqrt(np.mean(calibrated**2, axis=(1, 2))),
                1.0,
                atol=1e-3,
            )
        )
        self.assertLess(
            history[-1]["condition_loss"], history[0]["condition_loss"]
        )


if __name__ == "__main__":
    unittest.main()
