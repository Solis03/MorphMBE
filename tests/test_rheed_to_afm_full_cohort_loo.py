from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import (
    FULL_SPLIT,
    _condition_with_amplitude,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_generation.data import ConditionScaler


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
