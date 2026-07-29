from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.afm_metrology_repair.line_flatten import line_flatten, sq_nm
from analysis.rheed_to_afm_functional_morphology.run import _target_series


def test_third_order_line_flatten_removes_row_polynomials() -> None:
    x = np.linspace(-1.0, 1.0, 96)
    y = np.linspace(-1.0, 1.0, 48)[:, None]
    morphology = 0.3 * np.sin(13.0 * x)[None, :] * np.cos(4.0 * y)
    background = (
        7.0
        + 2.0 * y
        + (1.5 + y) * x[None, :]
        - 4.0 * x[None, :] ** 2
        + 3.0 * x[None, :] ** 3
    )
    corrected, fitted = line_flatten(morphology + background, order=3)
    expected, _ = line_flatten(morphology, order=3)
    assert corrected.dtype == np.float32
    assert fitted.shape == morphology.shape
    assert np.allclose(corrected, expected, atol=2e-5)


def test_order_zero_centers_every_scan_line() -> None:
    height = np.arange(35, dtype=float).reshape(5, 7)
    corrected, _ = line_flatten(height, order=0)
    assert np.allclose(corrected.mean(axis=1), 0.0, atol=1e-6)


def test_sq_is_areal_rms_height() -> None:
    height = np.asarray([[-2.0, 0.0], [0.0, 2.0]])
    assert np.isclose(sq_nm(height), np.sqrt(2.0))


def test_target_series_aggregates_sq_in_nm_before_log() -> None:
    descriptors = pd.DataFrame(
        {
            "growth_run_id": ["a", "a", "b"],
            "split": ["train", "train", "train"],
            "rq_nm": [1.0, 9.0, 4.0],
            "log_rq_nm": np.log([1.0, 9.0, 4.0]),
        }
    )
    metrics = pd.DataFrame(
        {
            "growth_run_id": ["a", "b"],
            "split": ["train", "train"],
            "functional_surface_morphology_index_nm": [2.0, 3.0],
        }
    )
    log_sq, _ = _target_series(descriptors, metrics, split="train")
    assert np.isclose(log_sq.loc["a"], np.log(5.0))
