from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_rough_island_redesign.nanoletters_m22_figures import (
    METHOD,
    PUBLIC_ID,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results/m22/sq_outer_loo_predictions.csv"
CONFIG = ROOT / "configs/morphmbe_m22.json"


def test_m22_public_mapping_and_frozen_cohort_are_one_to_one() -> None:
    predictions = pd.read_csv(PREDICTIONS, dtype={"growth_run_id": str})

    assert len(PUBLIC_ID) == 27
    assert len(set(PUBLIC_ID.values())) == 27
    assert set(predictions["growth_run_id"]) == set(PUBLIC_ID)
    assert not predictions["outer_target_used_for_training"].astype(bool).any()
    assert (predictions["outer_fit_growth_count"] == 26).all()


def test_m22_figure_metrics_match_frozen_result() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    truth = predictions["true_target"].to_numpy(float)
    predicted = predictions["predicted_target"].to_numpy(float)

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


def test_m22_publication_configuration_uses_selected_nonretrieval_method() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["selected_method"] == METHOD
    assert config["selected_renderer"]["island_generator_mode"] == (
        "separated_ellipse_growth_layered_gapfill_strong"
    )
    assert config["expected_growth_count"] == 27
    assert "6081" in config["explicitly_excluded_growths"]
    assert "No measured query AFM is used" in config["claim_boundary"]
