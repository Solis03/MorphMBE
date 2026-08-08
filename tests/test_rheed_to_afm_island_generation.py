from __future__ import annotations

import unittest

import numpy as np

from analysis.rheed_to_afm_island_generation.islands import (
    ISLAND_FEATURE_COLUMNS,
    IslandPrimitiveGenerator,
    extract_island_features,
)


class IslandGenerationTest(unittest.TestCase):
    def test_feature_extraction_is_finite(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        field = np.exp(-((xx - 20) ** 2 + (yy - 22) ** 2) / 80)
        field += 0.8 * np.exp(-((xx - 45) ** 2 + (yy - 40) ** 2) / 45)
        features = extract_island_features(field)
        self.assertEqual(set(features), set(ISLAND_FEATURE_COLUMNS))
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_all_primitive_modes_are_stochastic_and_unit_rq(self) -> None:
        target = {
            "log_component_count_q55": np.log1p(10),
            "log_component_count_q70": np.log1p(18),
            "log_component_count_q82": np.log1p(22),
            "log_median_area_q55": np.log1p(90),
            "log_median_area_q70": np.log1p(65),
            "log_median_area_q82": np.log1p(40),
            "median_solidity_q70": 0.88,
            "median_eccentricity_q70": 0.80,
            "boundary_gradient_ratio_q55": 1.35,
            "boundary_gradient_ratio_q70": 1.30,
            "boundary_gradient_ratio_q82": 1.20,
            "log_valley_component_count_q18": np.log1p(20),
            "log_valley_median_area_q18": np.log1p(35),
            "gradient_p90": 0.4,
            "laplacian_rms": 0.3,
            "flat_fraction": 0.2,
        }
        generator = IslandPrimitiveGenerator(resolution=64)
        for mode in (
            "superellipse",
            "laguerre",
            "hybrid",
            "separated_ellipse",
            "separated_ellipse_sparse",
            "separated_ellipse_round",
            "separated_ellipse_hierarchical",
            "separated_ellipse_strict_sparse",
        ):
            first = generator.generate(target, seed=17, mode=mode)
            second = generator.generate(target, seed=18, mode=mode)
            self.assertEqual(first.shape, (64, 64))
            self.assertTrue(np.isfinite(first).all())
            self.assertAlmostEqual(float(np.sqrt(np.mean(first**2))), 1.0, places=5)
            self.assertGreater(float(np.mean(np.abs(first - second))), 0.05)

    def test_separated_islands_retain_a_deep_connected_substrate(self) -> None:
        target = {
            "log_component_count_q55": np.log1p(10),
            "log_component_count_q70": np.log1p(18),
            "log_component_count_q82": np.log1p(22),
            "log_median_area_q55": np.log1p(90),
            "log_median_area_q70": np.log1p(65),
            "log_median_area_q82": np.log1p(40),
            "median_solidity_q70": 0.88,
            "median_eccentricity_q70": 0.80,
            "boundary_gradient_ratio_q55": 1.35,
            "boundary_gradient_ratio_q70": 1.30,
            "boundary_gradient_ratio_q82": 1.20,
            "log_valley_component_count_q18": np.log1p(20),
            "log_valley_median_area_q18": np.log1p(35),
            "gradient_p90": 0.4,
            "laplacian_rms": 0.3,
            "flat_fraction": 0.2,
        }
        generator = IslandPrimitiveGenerator(resolution=64)
        generated = generator.generate(
            target, seed=23, mode="separated_ellipse"
        )

        self.assertLess(float(np.quantile(generated, 0.25)), -0.80)
        self.assertGreater(float(np.quantile(generated, 0.90)), 1.15)
        self.assertGreater(float(np.mean(generated < -0.50)), 0.30)

    def test_dense_laguerre_configuration_remains_finite(self) -> None:
        target = {
            "log_component_count_q55": np.log1p(10),
            "log_component_count_q70": np.log1p(18),
            "log_component_count_q82": np.log1p(22),
            "log_median_area_q55": np.log1p(90),
            "log_median_area_q70": np.log1p(65),
            "log_median_area_q82": np.log1p(40),
            "median_solidity_q70": 0.88,
            "median_eccentricity_q70": 0.80,
            "boundary_gradient_ratio_q55": 1.35,
            "boundary_gradient_ratio_q70": 1.30,
            "boundary_gradient_ratio_q82": 1.20,
            "log_valley_component_count_q18": np.log1p(20),
            "log_valley_median_area_q18": np.log1p(35),
            "gradient_p90": 0.4,
            "laplacian_rms": 0.3,
            "flat_fraction": 0.2,
        }
        generator = IslandPrimitiveGenerator(
            resolution=64,
            laguerre_count_factor=6.0,
            fine_count_factor=6.0,
        )
        generated = generator.generate(target, seed=19, mode="laguerre")
        self.assertTrue(np.isfinite(generated).all())
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(generated**2))), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
