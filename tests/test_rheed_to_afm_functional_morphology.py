from __future__ import annotations

import numpy as np

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    _range_calibrate,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    FSMI_COMPONENTS,
    extract_surface_metrics,
)
from analysis.rheed_to_afm_functional_morphology.render import (
    edge_preserving_terrace_blend,
)


def test_fsmi_has_physical_units_and_scales_with_height() -> None:
    y, x = np.mgrid[:128, :128]
    surface = (
        0.7 * np.sin(2 * np.pi * x / 19)
        + 0.4 * np.cos(2 * np.pi * y / 27)
    )
    base = extract_surface_metrics(surface)
    doubled = extract_surface_metrics(2.0 * surface)

    for component in FSMI_COMPONENTS:
        assert np.isclose(
            doubled[component], 2.0 * base[component], rtol=1e-6
        )
    assert np.isclose(
        doubled["functional_surface_morphology_index_nm"],
        2.0 * base["functional_surface_morphology_index_nm"],
        rtol=1e-6,
    )


def test_fsmi_distinguishes_equal_rq_surfaces_with_different_scales() -> None:
    y, x = np.mgrid[:128, :128]
    coarse = np.sin(2 * np.pi * x / 48) + np.cos(2 * np.pi * y / 52)
    fine = np.sin(2 * np.pi * x / 8) + np.cos(2 * np.pi * y / 9)
    coarse /= np.std(coarse)
    fine /= np.std(fine)

    coarse_metrics = extract_surface_metrics(coarse)
    fine_metrics = extract_surface_metrics(fine)

    assert np.isclose(coarse_metrics["sq_nm"], fine_metrics["sq_nm"])
    assert (
        fine_metrics["functional_surface_morphology_index_nm"]
        > 1.20
        * coarse_metrics["functional_surface_morphology_index_nm"]
    )


def test_range_calibration_expands_compressed_log_spread() -> None:
    predicted = np.asarray([2.0, 2.5, 3.0, 3.5, 4.0])
    truth = np.asarray([1.3, 2.0, 3.0, 4.5, 6.2])
    low, scale = _range_calibrate(predicted, truth, 2.0)
    high, _ = _range_calibrate(predicted, truth, 4.0)

    assert 1.0 <= scale <= 1.20
    assert high / low > 4.0 / 2.0


def test_terrace_renderer_is_novel_finite_and_unit_rq() -> None:
    rng = np.random.default_rng(11)
    structure = rng.normal(size=(128, 128))
    structure = np.cumsum(structure, axis=0)
    prior = rng.normal(size=(128, 128))
    rendered = edge_preserving_terrace_blend(
        structure,
        prior,
        boundary_width_px=0.75,
        structure_weight=0.50,
        plateau_weight=0.14,
        relief_weight=0.18,
        spectral_weight=0.13,
        texture_weight=0.05,
    )

    assert rendered.shape == structure.shape
    assert np.isfinite(rendered).all()
    assert np.isclose(np.std(rendered), 1.0, atol=1e-5)
    assert not np.array_equal(rendered, structure)
    assert not np.array_equal(rendered, prior)
