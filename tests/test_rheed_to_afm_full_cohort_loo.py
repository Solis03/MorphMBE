from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import (
    FULL_SPLIT,
    _condition_with_amplitude,
    _load_external_predictions,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_full_cohort_loo.visualization import (
    _external_target_confidence,
)


class _RemovalAudit:
    sample_ids = ("6023", "6087")


def test_prepare_full_cohort_retains_source_split_and_excludes_requested() -> None:
    rows = []
    for group, split in (
        ("6022", "val"),
        ("6028", "train"),
        ("6033", "test"),
    ):
        rows.append(
            {
                "sample_id": group,
                "growth_run_id": group,
                "split": split,
            }
        )
    tables = {
        "descriptors": pd.DataFrame(rows),
        "physics": pd.DataFrame(
            {
                "sample_id": ["6022", "6028", "6033"],
                "growth_run_id": ["6022", "6028", "6033"],
            }
        ),
        "removelist": _RemovalAudit(),
    }
    config = {
        "expected_growth_count": 3,
        "explicitly_excluded_growths": ["6043", "6055"],
    }
    cohort, source = prepare_full_cohort(tables, config)

    assert set(cohort["split"]) == {FULL_SPLIT}
    assert source == {"6022": "val", "6028": "train", "6033": "test"}
    assert not set(cohort["growth_run_id"]) & {"6043", "6055"}


def test_amplitude_override_changes_only_log_rq_condition() -> None:
    columns = ["log_rq_nm", "shape_a", "shape_b"]
    scaler = ConditionScaler(
        columns=columns,
        mean=np.asarray([np.log(2.0), 4.0, 8.0]),
        scale=np.asarray([0.5, 2.0, 3.0]),
        lower=np.asarray([-10.0, -10.0, -10.0]),
        upper=np.asarray([10.0, 10.0, 10.0]),
    )
    initial = np.asarray([0.1, -0.4, 0.7], dtype=np.float32)
    updated = _condition_with_amplitude(initial, scaler, 4.0)

    assert np.isclose(updated[0], np.log(2.0) / 0.5)
    assert np.array_equal(updated[1:], initial[1:])


def test_external_predictions_require_exact_leakage_free_cohort(
    tmp_path,
) -> None:
    groups = ["6022", "6028", "6033"]
    target = pd.Series(
        np.log([1.0, 2.0, 3.0]), index=groups, name="log_rq_nm"
    )
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "growth_run_id": groups,
            "true_target": [1.0, 2.0, 3.0],
            "predicted_target": [1.1, 1.9, 3.1],
            "absolute_error": [0.1, 0.1, 0.1],
            "predicted_absolute_error": [0.2, 0.2, 0.2],
            "interval_lower": [0.5, 1.5, 2.5],
            "interval_upper": [1.5, 2.5, 3.5],
            "interval_covered": [True, True, True],
            "outer_target_used_for_training": [False, False, False],
        }
    ).to_csv(path, index=False)

    loaded = _load_external_predictions(
        path=path, groups=groups, log_target=target
    )

    assert loaded["growth_run_id"].tolist() == groups
    assert not loaded["outer_target_used_for_training"].any()


def test_external_target_confidence_uses_both_target_heads(
    tmp_path,
) -> None:
    groups = [str(6000 + index) for index in range(23)]
    rows = []
    for target, confidence in (("Rq_nm", 0.25), ("FSMI_nm", 1.0)):
        for index, group in enumerate(groups):
            rows.append(
                {
                    "growth_run_id": group,
                    "target": target,
                    "method": "M14i_target_specific_robust",
                    "confidence": confidence,
                    "absolute_error": float(index),
                    "predicted_absolute_error": float(index + 1),
                    "interval_covered": True,
                    "outer_target_used_for_training": False,
                }
            )
    path = tmp_path / "confidence.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    fallback = pd.DataFrame(
        {
            "growth_run_id": groups,
            "realized_island_error_z": 0.0,
            "island_error_90_upper_z": 1.0,
        }
    )

    result = _external_target_confidence(
        path=path, fallback=fallback
    )

    assert np.allclose(result["joint_confidence_index"], 50.0)
    assert result["rq_interval_covered"].all()
