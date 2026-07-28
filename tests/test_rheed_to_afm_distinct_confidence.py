from __future__ import annotations

import unittest

import numpy as np

from analysis.rheed_to_afm_distinct_confidence.matern import (
    DescriptorMaternGenerator,
)
from analysis.rheed_to_afm_distinct_confidence.uncertainty import (
    jackknife_plus_interval,
    relative_confidence_index,
)
from analysis.rheed_to_afm_distinct_confidence.variance import (
    VarianceCalibrator,
)
from analysis.rheed_to_afm_distinct_confidence.run import _blend_ensembles
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_video_afm_story.afm_descriptors import describe_map


def _scaler() -> ConditionScaler:
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
    return ConditionScaler(
        columns=columns,
        mean=np.asarray(
            [1.0, 0.78, 0.001, 0.0001, -3.7, np.log(45.0), 0.2, 0.0, 3.3]
        ),
        scale=np.asarray([0.5, 0.05, 0.001, 0.0001, 0.5, 0.4, 0.25, 0.6, 1.0]),
        lower=np.ones(9) * -100,
        upper=np.ones(9) * 100,
    )


class DistinctConfidenceTest(unittest.TestCase):
    def test_matern_generator_is_stochastic_and_condition_sensitive(self) -> None:
        generator = DescriptorMaternGenerator(_scaler(), resolution=64)
        fine = np.asarray([0, 0, 1, 1, 1, -1, 0, 0, 0], dtype=float)
        coarse = np.asarray([0, 0, -1, -1, -1, 1, 1, 1, 1], dtype=float)
        fine_map = generator.generate(fine, seed=9)
        coarse_map = generator.generate(coarse, seed=9)
        second_draw = generator.generate(fine, seed=10)
        self.assertTrue(np.isfinite(fine_map).all())
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(fine_map**2))), 1.0, places=4
        )
        self.assertFalse(np.array_equal(fine_map, second_draw))
        self.assertGreater(float(np.mean(np.abs(fine_map - coarse_map))), 0.2)
        fine_desc = describe_map(fine_map, "unit")
        coarse_desc = describe_map(coarse_map, "unit")
        self.assertLess(
            fine_desc["unit_autocorr_length_nm"],
            coarse_desc["unit_autocorr_length_nm"],
        )

    def test_variance_calibration_expands_nonzero_conditions(self) -> None:
        calibrator = VarianceCalibrator(
            factors=np.asarray([2.0, 1.5]),
            cap=2.0,
            minimum_predicted_std=0.15,
            columns=["a", "b"],
        )
        calibrated = calibrator.transform_z(np.asarray([[0.5, -1.0]]))
        np.testing.assert_allclose(calibrated, [[1.0, -1.5]])

    def test_hybrid_blend_is_generated_finite_and_unit_rq(self) -> None:
        first = np.zeros((16, 16), dtype=np.float32)
        first[3:9, 4:11] = 1.0
        second = np.eye(16, dtype=np.float32)
        blended = _blend_ensembles(
            [first], [second], primary_weight=0.65
        )[0]
        self.assertTrue(np.isfinite(blended).all())
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(blended**2))), 1.0, places=4
        )
        self.assertFalse(np.array_equal(blended, first))
        self.assertFalse(np.array_equal(blended, second))

    def test_jackknife_interval_and_confidence_are_ordered(self) -> None:
        query = np.asarray([[0.0], [0.1], [-0.1], [0.05]])
        calibration_prediction = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        calibration_truth = np.asarray([[0.1], [1.2], [1.7], [3.4]])
        lower, upper = jackknife_plus_interval(
            query,
            calibration_prediction,
            calibration_truth,
            alpha=0.10,
        )
        self.assertLess(float(lower[0]), float(upper[0]))
        confidence = relative_confidence_index(
            np.asarray([1.0, 2.0, 3.0]),
            reference_widths=np.asarray([1.0, 2.0, 3.0]),
        )
        self.assertGreater(float(confidence[0]), float(confidence[-1]))


if __name__ == "__main__":
    unittest.main()
