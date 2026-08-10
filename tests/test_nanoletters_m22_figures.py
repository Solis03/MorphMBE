import numpy as np

from analysis.rheed_rough_island_redesign.nanoletters_m22_figures import (
    CONFIG_PATH,
    METHOD,
    PUBLIC_ID,
    _generated_map,
    _load_data,
)


def test_m22_figure_data_uses_exact_outer_loo_cohort() -> None:
    data = _load_data(CONFIG_PATH.resolve())

    assert len(PUBLIC_ID) == 27
    assert len(set(PUBLIC_ID.values())) == 27
    assert set(data["sq"]["growth_run_id"].astype(str)) == set(PUBLIC_ID)
    assert data["manifest"]["selected_method"] == METHOD
    assert data["manifest"]["retrieval_at_inference"] is False
    assert not data["sq"]["outer_target_used_for_training"].astype(bool).any()


def test_m22_figure_metrics_match_frozen_result() -> None:
    data = _load_data(CONFIG_PATH.resolve())
    truth = data["sq"]["true_target"].to_numpy(float)
    predicted = data["sq"]["predicted_target"].to_numpy(float)

    np.testing.assert_allclose(
        np.corrcoef(truth, predicted)[0, 1],
        0.9234250316048422,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.mean(np.abs(predicted - truth)), 0.6853452351430823, atol=1e-12
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(predicted - truth))),
        0.829067335675652,
        atol=1e-12,
    )


def test_m22_generated_surface_respects_inference_boundary() -> None:
    data = _load_data(CONFIG_PATH.resolve())
    surface, predicted_sq = _generated_map(data["output"], "N6342")

    assert surface.ndim == 2
    assert np.isfinite(surface).all()
    assert predicted_sq > 0
    np.testing.assert_allclose(np.std(surface), predicted_sq, atol=1e-6)
