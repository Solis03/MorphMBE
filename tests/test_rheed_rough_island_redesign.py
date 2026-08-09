from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.rheed_rough_island_redesign.amplitude import (
    apply_rough_tail_rescue,
)
from analysis.rheed_rough_island_redesign.connectivity import (
    CONNECTIVITY_FEATURES,
    crossfit_spot_connectivity_calibration,
)
from analysis.rheed_rough_island_redesign.gwyddion_atlas import (
    gwyddion_net_colormap,
    individual_height_limits,
)
from analysis.rheed_to_afm_full_cohort_loo.run import load_config


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_target": [0.8, 1.0, 4.1, 6.0],
            "streak_expert_nm": [2.2, 2.7, 5.2, 4.8],
            "rough_consensus_gate": [False, False, True, True],
            "interval_radius": [0.4, 0.8, 1.0, 1.2],
            "true_target": [1.1, 3.9, 5.2, 3.1],
        }
    )


def test_tail_rescue_preserves_smooth_and_uses_independent_support() -> None:
    rescued = apply_rough_tail_rescue(_predictions())

    assert np.allclose(rescued["predicted_target"], [0.8, 2.7, 5.2, 6.0])
    assert rescued["rough_tail_rescue_activated"].tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_tail_rescue_decision_is_query_target_blind() -> None:
    original = _predictions()
    changed = original.copy()
    changed["true_target"] = [1000.0, 0.01, 75.0, 0.5]

    first = apply_rough_tail_rescue(original)
    second = apply_rough_tail_rescue(changed)

    assert np.array_equal(
        first["predicted_target"], second["predicted_target"]
    )
    assert np.array_equal(
        first["rough_tail_rescue_activated"],
        second["rough_tail_rescue_activated"],
    )


def test_final_config_inherits_m17_cohort_and_uses_portable_inputs() -> None:
    config = load_config(
        "configs/rheed_m19_separated_rough_islands_full27_v4.json"
    )

    assert config["full_run_suffix"] == "full27_loo"
    assert config["selected_method"].startswith("M19k_")
    for path in config["external_target_predictions"].values():
        assert not path.startswith("/")


def _connectivity_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [f"g{index}" for index in range(6)]
    predictions = pd.DataFrame(
        {
            "growth_run_id": groups,
            "true_target": [3.1, 2.4, 2.2, 4.0, 5.2, 9.4],
            "predicted_target": [6.2, 5.1, 5.5, 2.7, 5.2, 7.4],
            "base_endpoint_prediction_nm": [6.2, 5.1, 4.7, 1.0, 4.1, 5.5],
            "streak_expert_nm": [4.8, 2.5, 5.5, 2.7, 5.2, 7.4],
            "rough_tail_rescue_activated": [True] * 6,
            "interval_radius": [1.0] * 6,
            "interval_lower": [0.0] * 6,
            "interval_upper": [8.0] * 6,
            "absolute_error": [0.0] * 6,
            "interval_covered": [True] * 6,
            "outer_target_used_for_training": [False] * 6,
        }
    )
    physics = pd.DataFrame(
        {
            "growth_run_id": groups,
            CONNECTIVITY_FEATURES[0]: [-4.0, -4.5, -3.0, -2.5, -0.5, 4.0],
            CONNECTIVITY_FEATURES[1]: [4.0, 4.5, 5.0, 2.0, 8.0, 12.0],
            CONNECTIVITY_FEATURES[2]: [0.33, 0.45, 0.27, 0.0, 0.5, 0.5],
            CONNECTIVITY_FEATURES[3]: [0.079, 0.081, 0.067, 0.10, 0.032, 0.018],
        }
    )
    return predictions, physics


def test_connectivity_correction_excludes_query_target() -> None:
    predictions, physics = _connectivity_inputs()
    changed = predictions.copy()
    changed.loc[0, "true_target"] = 1000.0

    first = crossfit_spot_connectivity_calibration(predictions, physics)
    second = crossfit_spot_connectivity_calibration(changed, physics)

    assert first.loc[0, "spot_connectivity_gate"]
    assert first.loc[0, "predicted_target"] == second.loc[0, "predicted_target"]
    assert not first["outer_target_used_for_training"].astype(bool).any()

    changed = predictions.copy()
    changed.loc[5, "true_target"] = 1000.0
    second = crossfit_spot_connectivity_calibration(changed, physics)
    assert first.loc[5, "predicted_target"] == second.loc[5, "predicted_target"]


def test_connectivity_score_orders_bridged_before_isolated_spots() -> None:
    predictions, physics = _connectivity_inputs()
    result = crossfit_spot_connectivity_calibration(predictions, physics)

    assert (
        result.loc[0, "rheed_spot_isolation_score"]
        < result.loc[5, "rheed_spot_isolation_score"]
    )
    assert not result.loc[0, "isolated_spot_uplift_gate"]
    assert result.loc[5, "isolated_spot_uplift_gate"]
    assert result.loc[5, "predicted_target"] > predictions.loc[5, "predicted_target"]


def test_gwyddion_net_gradient_matches_official_endpoints() -> None:
    palette = gwyddion_net_colormap()

    assert np.allclose(palette(0.0)[:3], [0.0, 0.0, 0.0])
    assert np.allclose(palette(1.0)[:3], [1.0, 1.0, 1.0])
    rust = palette(0.344671)[:3]
    assert rust[0] > rust[1] > rust[2]


def test_afm_display_height_limits_are_independent_per_map() -> None:
    first = np.arange(100, dtype=float).reshape(10, 10)
    second = 5.0 * first - 30.0

    first_limits = individual_height_limits(first)
    second_limits = individual_height_limits(second)

    assert np.allclose(second_limits, 5.0 * np.asarray(first_limits) - 30.0)
    assert individual_height_limits(np.ones((4, 4)))[0] < 1.0
    assert individual_height_limits(np.ones((4, 4)))[1] > 1.0
