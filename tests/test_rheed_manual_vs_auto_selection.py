from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_manual_vs_auto_selection.statistics import (
    paired_error_statistics,
)


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = "20260729_m14i_human_vs_auto_full23_v1"
OUTPUT = (
    REPO
    / "outputs"
    / "rheed_manual_vs_auto_selection"
    / EXPERIMENT
    / "machine_dataset"
)
REPORT = (
    REPO
    / "reports"
    / "rheed_manual_vs_auto_selection"
    / EXPERIMENT
)


def test_machine_dataset_is_parallel_and_complete() -> None:
    selection = pd.read_csv(
        OUTPUT / "selection_comparison.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    assert len(selection) == selection["growth_run_id"].nunique() == 23
    assert not {"6043", "6055"} & set(selection["growth_run_id"])
    assert not selection["raw_data_modified"].astype(bool).any()
    assert selection["human_roi_coverage"].median() > 0.99
    for group in selection["growth_run_id"]:
        selected = np.load(
            OUTPUT / "clip_variants" / "selected_16" / f"{group}.npz",
            allow_pickle=False,
        )
        assert selected["frames_uint8"].shape == (16, 224, 224)
        assert int(selected["frame_indices"][7]) == int(
            selection.set_index("growth_run_id").loc[
                group, "machine_keyframe_index"
            ]
        )


def test_machine_manifest_points_to_machine_selected_frames() -> None:
    manifest = pd.read_csv(
        OUTPUT / "modeling_manifest.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    for _, row in manifest.iterrows():
        paths = json.loads(row["clip_frame_paths"])
        indices = json.loads(row["clip_frame_indices"])
        assert len(paths) == len(indices) == 16
        assert indices[7] == int(row["keyframe_index"])
        assert all((REPO / path).exists() for path in paths)
        assert str(OUTPUT.relative_to(REPO)) in row["clip_cache_path"]


def test_strict_protocol_predictions_have_no_outer_target_leakage() -> None:
    predictions = pd.read_csv(
        REPORT / "paired_target_predictions.csv",
        dtype={"growth_run_id": str},
    )
    counts = predictions.groupby(["target", "protocol"]).size()
    assert len(counts) == 6
    assert (counts == 23).all()
    assert not predictions["outer_target_used_for_training"].astype(bool).any()
    statistics = paired_error_statistics(predictions, draws=100, seed=7)
    assert len(statistics) == 4
    assert np.isfinite(statistics["mean_absolute_error_change_nm"]).all()


def test_generated_machine_domain_folds_exclude_the_held_growth() -> None:
    audit = pd.read_csv(
        REPORT
        / "auto_input_generation"
        / "full23_loo"
        / "fold_integrity_audit.csv",
        dtype={"held_growth_run_id": str},
    )
    assert len(audit) == audit["held_growth_run_id"].nunique() == 23
    assert not audit["held_overlap_with_fit"].astype(bool).any()
    assert not {"6043", "6055"} & set(audit["held_growth_run_id"])
