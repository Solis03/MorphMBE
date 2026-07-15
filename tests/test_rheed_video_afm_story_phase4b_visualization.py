from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.afm_rendering import scale_bar_pixels
from analysis.rheed_video_afm_story.common import repo_path, sha256_file


CONFIG_PATH = Path("configs/rheed_video_afm_story_phase4b.yaml")
OUTPUT_ROOT = Path("outputs/rheed_video_afm_story/phase4b_visualization")
REPORT_ROOT = Path("reports/rheed_video_afm_story/phase4b_visualization")


def load_config() -> dict:
    return json.loads(repo_path(CONFIG_PATH).read_text())


def sample_table() -> pd.DataFrame:
    return pd.read_csv(repo_path(OUTPUT_ROOT / "sample_level_results.csv"), dtype={"sample_id": str, "growth_run_id": str})


def validation() -> dict:
    return json.loads(repo_path(OUTPUT_ROOT / "visualization_validation.json").read_text())


def test_phase4b_input_hashes_are_frozen_and_old_outputs_unchanged() -> None:
    config = load_config()
    for key, item in config["artifacts"].items():
        assert sha256_file(item["path"]) == item["sha256"], key
    audit = repo_path(REPORT_ROOT / "provenance_and_schema_audit.md").read_text()
    assert "unchanged=True" in audit
    assert "Global/transductive outputs mixed into formal OOF table: False" in audit


def test_master_sample_table_has_exact_primary_oof_cohort() -> None:
    config = load_config()
    df = sample_table()
    assert len(df) == config["cohort"]["expected_primary_n"]
    assert not df["sample_id"].duplicated().any()
    assert not set(df["sample_id"]) & set(config["cohort"]["excluded_samples"])
    assert df["expert_labels_status"].eq("pending").all()
    assert df["ground_truth_afm_path"].map(lambda p: repo_path(p).exists()).all()
    assert df["s4_output_path"].map(lambda p: repo_path(p).exists()).all()
    assert df["rheed_keyframe_path"].map(lambda p: repo_path(p).exists()).all()


def test_oof_rq_join_matches_phase1_targets_and_phase4a_predictions() -> None:
    df = sample_table().set_index("sample_id")
    manifest = pd.read_parquet(repo_path("outputs/rheed_video_afm_story/phase1/modeling_manifest.parquet"))
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest = manifest.set_index("sample_id")
    rq = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/rheed_rq_oof_predictions.csv"), dtype={"sample_id": str})
    rq = rq[rq["model_id"].eq(load_config()["preferred_rq_model"])].set_index("sample_id")
    assert np.allclose(df["true_rq_nm"], manifest.loc[df.index, "primary_rq_nm_median"])
    assert np.allclose(df["predicted_rq_nm"], rq.loc[df.index, "predicted_rq_nm"])
    assert np.allclose(df["rq_absolute_error_nm"], rq.loc[df.index, "absolute_error_nm"])


def test_s4_output_amplitude_tracks_predicted_rq_and_has_no_heldout_source() -> None:
    df = sample_table()
    assert (df["s4_output_rq_nm"] - df["predicted_rq_nm"]).abs().max() < 1e-3
    assert df["s4_heldout_source_contribution"].eq(0).all()
    assert df["s4_exact_pixel_equality"].astype(bool).eq(False).all()
    assert df["s4_largest_source_contribution"].max() <= 0.60


def test_main_sample_selection_is_quantile_based_not_error_based() -> None:
    df = sample_table()
    selection = json.loads(repo_path(OUTPUT_ROOT / "main_figure_sample_selection.json").read_text())
    assert selection["uses_prediction_error"] is False
    assert selection["selection_rule"] == load_config()["sample_selection_rule"]
    assert len(selection["selected_sample_ids"]) == 6
    assert set(selection["selected_sample_ids"]).issubset(set(df["sample_id"]))


def test_afm_rendering_records_colorbars_scale_bars_and_color_modes() -> None:
    v = validation()
    assert v["passed"] is True
    assert v["all_afm_axes_have_colorbar"] is True
    assert v["all_afm_axes_have_125nm_scale_bar"] is True
    assert scale_bar_pixels(scan_size_nm=1000, image_pixels=256, bar_nm=125) == 32
    records = pd.DataFrame(v["afm_render_records"])
    assert not records.empty
    assert set(records["panel"]) == {"gt", "s1", "s4"}
    assert set(records["scale_mode"]) == {"row_shared", "per_image"}
    shared = records[records["scale_mode"].eq("row_shared")]
    for _, group in shared.groupby(["figure", "sample_id"]):
        assert group["vmin"].nunique() == 1
        assert group["vmax"].nunique() == 1
    robust = records[records["scale_mode"].eq("per_image")]
    assert any(group["vmin"].round(6).nunique() > 1 for _, group in robust.groupby(["figure", "sample_id"]))


def test_required_publication_outputs_exist_in_expected_formats() -> None:
    required_stems = [
        "Fig1_pipeline_and_data_flow",
        "Fig2_ground_truth_vs_representative_afm_main_row_shared_scale",
        "Fig2_ground_truth_vs_representative_afm_main_per_image_robust_scale",
        "Fig3_rq_prediction_performance",
        "Fig3_rq_prediction_performance_log",
        "Fig4_rheed_morphology_roughness_continuum",
        "Fig5_afm_descriptor_landscape",
        "Fig6_rheed_afm_feature_relationships",
        "Fig7_afm_output_method_comparison",
        "Fig8_same_growth_similarity_ceiling",
        "Fig9_s4_provenance_and_identity_audit",
        "FigS2_why_neural_decoder_was_not_used",
        "one_page_current_project_summary",
    ]
    for stem in required_stems:
        for suffix in [".png", ".pdf", ".svg"]:
            assert repo_path(REPORT_ROOT / "figures" / f"{stem}{suffix}").exists(), stem
    for order in ["sample_id", "rq"]:
        for mode in ["shared", "robust"]:
            pages = sorted(repo_path(REPORT_ROOT / f"FigS1_all_23_samples_atlas_pages_{order}_{mode}").glob("page_*.png"))
            assert len(pages) == 6


def test_model_summary_and_relationship_tables_are_present() -> None:
    model = pd.read_csv(repo_path(OUTPUT_ROOT / "model_level_summary.csv"))
    assert {"R0_median", "R1_dino_pls", "R4_auto_iso_dino_residual", "R4_high_confidence_subset"}.issubset(set(model["model"]))
    r4 = model[model["model"].eq("R4_auto_iso_dino_residual")].iloc[0]
    assert np.isclose(r4["MAE"], 1.637473, atol=1e-5)
    assert np.isclose(r4["Spearman"], 0.289526, atol=1e-5)
    cmat = pd.read_csv(repo_path(OUTPUT_ROOT / "rheed_afm_spearman_matrix.csv"))
    ci = pd.read_csv(repo_path(OUTPUT_ROOT / "rheed_afm_bootstrap_ci.csv"))
    assert "automatic_spot_streak_index" in set(cmat["rheed_feature"])
    assert ci["N"].eq(23).all()


def test_phase4b_dashboard_reports_and_validation_summary_exist() -> None:
    for rel in [
        REPORT_ROOT / "results_dashboard.html",
        REPORT_ROOT / "figure_captions.md",
        REPORT_ROOT / "current_results_summary.md",
        REPORT_ROOT / "claims_and_limitations.md",
        OUTPUT_ROOT / "all_samples_visual_table.html",
        OUTPUT_ROOT / "phase4b_summary.json",
    ]:
        assert repo_path(rel).exists()
    summary = json.loads(repo_path(OUTPUT_ROOT / "phase4b_summary.json").read_text())
    assert summary["primary_sample_count"] == 23
    assert summary["validation_passed"] is True
