from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.rheed_rough_island_redesign.amplitude import (
    apply_rough_tail_rescue,
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
