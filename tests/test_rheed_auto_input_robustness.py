from __future__ import annotations

import numpy as np

from analysis.rheed_auto_input_robustness.confidence import (
    _empirical_risk,
    _view_summary,
    angular_coverage_risk,
    combine_tta_and_head_confidence,
)
from analysis.rheed_auto_input_robustness.perturbation import (
    PerturbationView,
    perturb_rect,
)
from rheed2morph.rheed.automatic_roi_keyframe import Rect


def test_roi_perturbation_is_small_and_clipped() -> None:
    rect = Rect(10, 20, 100, 160, 140, 220)
    shifted = perturb_rect(
        rect,
        PerturbationView("shift", x_shift_fraction=-0.03, scale=1.06),
    )
    assert shifted.x >= 0
    assert shifted.y >= 0
    assert shifted.x2 <= rect.source_width
    assert shifted.y2 <= rect.source_height
    assert shifted.width > rect.width


def test_tta_centrality_detects_off_center_base_prediction() -> None:
    names = [
        "base",
        "frame_m2",
        "frame_m1",
        "frame_p1",
        "frame_p2",
        "roi_left",
        "roi_right",
        "roi_up",
        "roi_down",
        "roi_tight",
        "roi_wide",
    ]
    predictions = np.asarray(
        [4.0, 2.9, 3.0, 3.1, 3.0, 3.0, 3.1, 2.9, 3.0, 3.1, 3.0]
    )
    summary = _view_summary(predictions, names)
    assert np.isclose(summary.median_prediction, 3.0)
    assert np.isclose(summary.base_to_median_nm, 1.0)
    assert summary.all_std_nm > 0.0
    assert summary.roi_std_nm > 0.0


def test_empirical_risk_increases_with_instability() -> None:
    reference = np.asarray([0.01, 0.03, 0.05, 0.08])
    assert _empirical_risk(0.20, reference) > _empirical_risk(
        0.02, reference
    )


def test_head_disagreement_only_vetoes_extreme_conflict() -> None:
    combined, veto = combine_tta_and_head_confidence(
        np.asarray([0.8, 0.8, 0.8]),
        np.asarray([0.9, 0.2, 0.02]),
    )
    assert np.allclose(combined, [0.8, 0.8, 0.02])
    assert veto.tolist() == [False, False, True]


def test_angular_coverage_risk_requires_both_risk_sources() -> None:
    combined = angular_coverage_risk(
        np.asarray([0.04, 0.64]),
        np.asarray([0.81, 0.25]),
    )
    assert np.allclose(combined, [0.18, 0.40])
