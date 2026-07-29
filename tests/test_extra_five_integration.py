from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.extra_five_integration.summarize import _cohort_metrics


REPOSITORY = Path(__file__).resolve().parents[1]


def test_extra_five_config_has_exact_operator_approved_cohort() -> None:
    config = json.loads(
        (
            REPOSITORY / "configs/extra_five_line3_full28_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert set(config["included_samples"]) == {
        "N6342",
        "N6358",
        "N6382",
        "N6389",
        "N6390",
    }
    assert config["excluded_samples"] == ["N6324"]
    assert "N6324" not in config["selected_videos"]
    assert config["expected_combined_growth_count"] == 28
    assert config["raw_data_policy"] == "read_only"
    assert config["standalone_policy"] == "read_only"


def test_orientation_corrected_config_rotates_only_6389_and_6390() -> None:
    path = (
        REPOSITORY
        / "configs/"
        "extra_five_line3_full28_orientation90_keyframe_locked_v3.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    rotations = config["rheed_rotation_clockwise_degrees_by_sample"]

    assert rotations == {"N6389": 90, "N6390": 90}
    assert config["keyframe_override_samples"] == ["N6389", "N6390"]
    assert "full28_v1" in config["keyframe_override_records_root"]
    assert "orientation90_v2" in config["selection_seed_root"]
    assert set(config["included_samples"]) == {
        "N6342",
        "N6358",
        "N6382",
        "N6389",
        "N6390",
    }
    assert config["excluded_samples"] == ["N6324"]


def test_batch_summary_keeps_extra_five_separate() -> None:
    extra = {"N6342", "N6358", "N6382", "N6389", "N6390"}
    groups = ["6022", "6028", *sorted(extra)]
    rows = []
    for target in ("Rq_nm", "FSMI_nm"):
        for index, group in enumerate(groups):
            truth = 1.0 + index * 0.2
            prediction = truth + (-0.1 if index % 2 else 0.1)
            rows.append(
                {
                    "growth_run_id": group,
                    "target": target,
                    "true_target": truth,
                    "predicted_target": prediction,
                    "absolute_error": abs(prediction - truth),
                    "confidence": 0.9 - index * 0.05,
                    "interval_covered": True,
                }
            )
    metrics = _cohort_metrics(pd.DataFrame(rows), extra)

    counts = metrics.set_index(["cohort", "target"])["growth_count"]
    assert counts.loc[("full28", "Rq_nm")] == 7
    assert counts.loc[("original23", "Rq_nm")] == 2
    assert counts.loc[("extra5", "Rq_nm")] == 5
    assert np.isfinite(
        metrics.loc[metrics["cohort"] == "extra5", "mae_nm"]
    ).all()


def test_full28_generation_config_excludes_rejected_extra_sample() -> None:
    config = json.loads(
        (
            REPOSITORY
            / "configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["expected_growth_count"] == 28
    assert "N6324" in config["explicitly_excluded_growths"]
    assert "N6324" not in config["extra_batch_growths"]
    assert len(config["extra_batch_growths"]) == 5
    assert config["full_run_suffix"] == "full28_loo"
