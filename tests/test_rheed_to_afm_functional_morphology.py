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
    regime_adaptive_separated_island_blend,
    render_ensemble,
    topology_conditioned_sparse_microisland_blend,
)


def test_fsmi_has_physical_units_and_scales_with_height() -> None:
    y, x = np.mgrid[:128, :128]
    surface = 0.7 * np.sin(2 * np.pi * x / 19) + 0.4 * np.cos(2 * np.pi * y / 27)
    base = extract_surface_metrics(surface)
    doubled = extract_surface_metrics(2.0 * surface)

    for component in FSMI_COMPONENTS:
        assert np.isclose(doubled[component], 2.0 * base[component], rtol=1e-6)
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
        > 1.20 * coarse_metrics["functional_surface_morphology_index_nm"]
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


def test_topology_sparse_renderer_is_unit_rq_and_condition_sensitive() -> None:
    rng = np.random.default_rng(43)
    structure = rng.normal(size=(128, 128))
    prior = rng.normal(size=(128, 128))
    sparse = topology_conditioned_sparse_microisland_blend(
        structure,
        prior,
        island_target={"log_component_count_q82": np.log1p(12.0)},
        peak_count_scale=0.5,
    )
    dense = topology_conditioned_sparse_microisland_blend(
        structure,
        prior,
        island_target={"log_component_count_q82": np.log1p(60.0)},
        peak_count_scale=0.5,
    )
    hierarchical = topology_conditioned_sparse_microisland_blend(
        structure,
        prior,
        island_target={"log_component_count_q82": np.log1p(36.0)},
        fine_texture_weight=0.12,
        sparse_peak_weight=0.07,
        shoulder_peak_weight=0.16,
    )

    assert sparse.shape == structure.shape
    assert np.isfinite(sparse).all()
    assert np.isclose(np.std(sparse), 1.0, atol=1e-5)
    assert np.isclose(np.std(dense), 1.0, atol=1e-5)
    assert np.isclose(np.std(hierarchical), 1.0, atol=1e-5)
    assert not np.array_equal(sparse, dense)
    assert not np.array_equal(sparse, hierarchical)


def test_topology_sparse_regime_render_ensemble_preserves_roughness() -> None:
    rng = np.random.default_rng(47)
    structure = [rng.normal(size=(64, 64)) for _ in range(2)]
    prior = [rng.normal(size=(64, 64)) for _ in range(2)]
    rendered = render_ensemble(
        structure,
        prior,
        mode="regime_adaptive_topology_sparse_terrace",
        conditioning_sq_nm=0.9,
        island_target={"log_component_count_q82": np.log1p(24.0)},
        boundary_width_px=0.75,
        structure_weight=0.50,
        plateau_weight=0.14,
        relief_weight=0.18,
        spectral_weight=0.13,
        texture_weight=0.05,
    )

    assert len(rendered) == 2
    assert all(np.isclose(np.std(array), 1.0, atol=1e-5) for array in rendered)


def test_separated_island_regime_keeps_m17_exact_below_rough_threshold() -> None:
    rng = np.random.default_rng(53)
    baseline = [rng.normal(size=(64, 64)) for _ in range(2)]
    separated = [rng.normal(size=(64, 64)) for _ in range(2)]
    prior = [rng.normal(size=(64, 64)) for _ in range(2)]
    common = {
        "conditioning_sq_nm": 1.2,
        "island_target": {"log_component_count_q82": np.log1p(24.0)},
        "boundary_width_px": 0.75,
        "structure_weight": 0.50,
        "plateau_weight": 0.14,
        "relief_weight": 0.18,
        "spectral_weight": 0.13,
        "texture_weight": 0.05,
    }
    frozen_m17 = render_ensemble(
        baseline,
        prior,
        mode="regime_adaptive_topology_sparse_terrace",
        **common,
    )
    redesigned = render_ensemble(
        separated,
        prior,
        baseline_structure=baseline,
        mode="regime_adaptive_separated_islands",
        rough_start_nm=2.2,
        rough_full_nm=3.6,
        **common,
    )

    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(frozen_m17, redesigned, strict=False)
    )


def test_isolated_spots_deepen_rough_substrate_tail() -> None:
    rng = np.random.default_rng(71)
    structure = rng.normal(size=(64, 64))
    prior = rng.normal(size=(64, 64))
    baseline = rng.normal(size=(64, 64))

    bridged = regime_adaptive_separated_island_blend(
        structure,
        prior,
        baseline,
        conditioning_sq_nm=6.0,
        island_target=None,
        rough_structure_weight=1.0,
        rough_texture_weight=0.0,
        rough_tip_sigma_px=0.0,
        rough_isolation_score=0.2,
    )
    isolated = regime_adaptive_separated_island_blend(
        structure,
        prior,
        baseline,
        conditioning_sq_nm=6.0,
        island_target=None,
        rough_structure_weight=1.0,
        rough_texture_weight=0.0,
        rough_tip_sigma_px=0.0,
        rough_isolation_score=0.9,
    )

    assert np.quantile(isolated, 0.05) < np.quantile(bridged, 0.05)
