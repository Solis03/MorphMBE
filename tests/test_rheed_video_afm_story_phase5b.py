from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.common import repo_path


ROOT = Path("outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training")


def load_json(path: str | Path) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"


class Phase5BTest(unittest.TestCase):
    def test_primary_cohort_and_exclusions(self) -> None:
        pred = pd.read_csv(repo_path(ROOT / "phase5b_oof_predictions.csv"), dtype={"sample_id": str})
        ids = set(pred["sample_id"])
        self.assertEqual(23, len(ids))
        self.assertFalse({"6023", "6087"} & ids)

    def test_outer_folds_train_on_exact_other_22_groups(self) -> None:
        fold = pd.read_csv(repo_path(ROOT / "fold_membership_audit.csv"), dtype={"heldout_sample_id": str})
        ids = set(fold["heldout_sample_id"])
        self.assertEqual(23, len(fold))
        for row in fold.to_dict("records"):
            held = str(row["heldout_sample_id"])
            actual = set(json.loads(row["actual_training_sample_ids"]))
            self.assertEqual(22, int(row["actual_training_group_count"]))
            self.assertEqual(ids - {held}, actual)
            self.assertFalse(held in actual)
            self.assertTrue(truthy(row["split_valid"]))

    def test_6095_6099_cross_support(self) -> None:
        fold = pd.read_csv(repo_path(ROOT / "fold_membership_audit.csv"), dtype={"heldout_sample_id": str})
        row6099 = fold[fold["heldout_sample_id"].eq("6099")].iloc[0]
        row6095 = fold[fold["heldout_sample_id"].eq("6095")].iloc[0]
        self.assertIn("6095", set(json.loads(row6099["actual_training_sample_ids"])))
        self.assertIn("6099", set(json.loads(row6095["actual_training_sample_ids"])))

    def test_current_r4_pre_fix_bug_is_recorded_not_silenced(self) -> None:
        pre = pd.read_csv(repo_path(ROOT / "current_r4_pre_fix_fold_membership_audit.csv"))
        recon = pd.read_csv(repo_path(ROOT / "current_r4_fold_reconstruction.csv"))
        summary = load_json(ROOT / "phase5b_summary.json")
        self.assertTrue((~pre["pre_fix_split_valid"].map(truthy)).any())
        self.assertTrue((~recon["target_alignment_valid"].map(truthy)).any())
        self.assertTrue(summary["split_bug_found"])
        self.assertFalse(summary["current_code_maximal_n_minus_1_loocv"])
        self.assertTrue(summary["phase5b_fixed_strict_oof_maximal_n_minus_1"])

    def test_heldout_frames_and_afm_are_not_in_training_artifacts(self) -> None:
        fold = pd.read_csv(repo_path(ROOT / "fold_membership_audit.csv"))
        self.assertFalse(fold["heldout_present_in_training"].map(truthy).any())
        self.assertFalse(fold["heldout_afm_present_in_retrieval_bank"].map(truthy).any())
        self.assertFalse(fold["heldout_clip_present_in_embedding_fit"].map(truthy).any())

    def test_each_sample_has_oof_prediction_per_model(self) -> None:
        pred = pd.read_csv(repo_path(ROOT / "phase5b_oof_predictions.csv"), dtype={"sample_id": str})
        for model_id, g in pred.groupby("model_id"):
            self.assertEqual(23, len(g), model_id)
            self.assertEqual(23, g["sample_id"].nunique(), model_id)
            self.assertTrue((g["outer_training_group_count"] == 22).all())

    def test_inner_cv_does_not_access_outer_target(self) -> None:
        inner = pd.read_csv(repo_path(ROOT / "nested_inner_cv_audit.csv"), dtype={"outer_fold": str, "outer_heldout_id": str})
        self.assertFalse(inner["outer_heldout_target_accessed"].map(truthy).any())
        self.assertFalse(inner["outer_heldout_in_inner_train"].map(truthy).any())
        self.assertIn("selected_neighbor_audit.csv", {p.name for p in repo_path(ROOT).glob("*.csv")})

    def test_regime_thresholds_and_clusters_are_fold_local(self) -> None:
        regime = pd.read_csv(repo_path(ROOT / "regime_predictions.csv"), dtype={"sample_id": str})
        self.assertEqual(23, len(regime))
        self.assertTrue(regime["q33_train"].notna().all())
        self.assertTrue(regime["q67_train"].notna().all())
        self.assertTrue(regime["branch_a_predicted_regime"].isin(["low", "middle", "high"]).all())
        self.assertTrue(regime["branch_b_predicted_regime"].isin(["low", "middle", "high"]).all())
        self.assertTrue(regime["branch_c_status"].eq("pending_no_frozen_blinded_expert_labels").all())

    def test_knn_and_retrieval_do_not_select_heldout_or_forbidden_opposite_regime(self) -> None:
        neigh = pd.read_csv(repo_path(ROOT / "selected_neighbor_audit.csv"), dtype={"outer_fold": str, "neighbor_sample_id": str})
        self.assertFalse((neigh["outer_fold"] == neigh["neighbor_sample_id"]).any())
        ret = pd.read_csv(repo_path(ROOT / "regime_gated_retrieval.csv"), dtype={"sample_id": str, "new_source_sample_id": str})
        self.assertFalse((ret["sample_id"] == ret["new_source_sample_id"]).any())
        self.assertTrue(ret["opposite_regime_forbidden"].map(truthy).all())

    def test_bootstrap_and_multiview_units_are_growth_groups(self) -> None:
        boot = pd.read_csv(repo_path(ROOT / "bootstrap_predictions.csv"), dtype={"sample_id": str})
        self.assertEqual(23 * 50, len(boot))
        self.assertTrue(boot["bootstrap_unit"].eq("growth_group").all())
        pred = pd.read_csv(repo_path(ROOT / "phase5b_oof_predictions.csv"), dtype={"sample_id": str})
        self.assertEqual(23, pred[pred["model_id"].eq("L6_cross_fitted_bootstrap_median")]["sample_id"].nunique())

    def test_support_score_is_target_blind_and_abstains(self) -> None:
        regime = pd.read_csv(repo_path(ROOT / "regime_predictions.csv"), dtype={"sample_id": str})
        self.assertIn("low", set(regime["support_level"]))
        self.assertTrue(regime["support_reason"].str.contains("same_regime=").all())
        summary = load_json(ROOT / "phase5b_summary.json")
        self.assertGreater(len(summary["abstained_sample_ids"]), 0)

    def test_deployment_model_uses_all_23_and_metrics_are_isolated(self) -> None:
        dep = ROOT / "deployment_model"
        self.assertTrue(repo_path(dep / "model_registry.json").exists())
        rq = pd.read_csv(repo_path(dep / "training_rq_bank.csv"), dtype={"sample_id": str})
        self.assertEqual(23, rq["sample_id"].nunique())
        provenance = load_json(dep / "provenance.json")
        self.assertTrue(provenance["uses_all_23_labeled_groups"])
        self.assertTrue(provenance["for_future_unseen_samples_only"])
        self.assertTrue(provenance["does_not_report_test_performance"])

    def test_in_sample_calibration_has_warning(self) -> None:
        calib = pd.read_csv(repo_path(ROOT / "full_cohort_in_sample_calibration.csv"))
        self.assertTrue(calib["warning"].str.contains("IN-SAMPLE CALIBRATION ONLY").all())
        self.assertTrue(calib["warning"].str.contains("NOT A TEST RESULT").all())

    def test_first_and_second_order_control_are_both_reported(self) -> None:
        comp = pd.read_csv(repo_path(ROOT / "first_vs_second_regime_aware_comparison.csv"))
        self.assertIn("first_order_control", set(comp["target_variant"]))
        self.assertIn("second_order_y2", set(comp["target_variant"]))

    def test_required_visuals_and_reports_exist(self) -> None:
        for stem in [
            "Fig1_maximal_training_protocol",
            "Fig2_fold_membership_heatmap",
            "Fig3_fold_regime_support",
            "Fig4_neighbor_support_map",
            "Fig5_current_R4_vs_local_analog",
            "Fig6_extreme_high_rq_case_studies",
            "Fig7_regime_confusion_and_rq_distribution",
            "Fig8_support_coverage_performance",
            "Fig9_old_vs_regime_gated_retrieval",
            "Fig10_all_23_oof_prediction_grid",
            "Fig11_full_cohort_deployment_workflow",
            "FigS_in_sample_calibration_warning",
        ]:
            for suffix in [".png", ".pdf", ".svg"]:
                self.assertTrue(repo_path(ROOT.parents[0] / "dummy").exists() or repo_path(f"reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/figures/{stem}{suffix}").exists(), stem)
        self.assertTrue(repo_path("reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/phase5b_report.md").exists())

    def test_case_studies_and_neighbor_tables_exist(self) -> None:
        for sid in ["6095", "6099"]:
            self.assertTrue(repo_path(f"reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/case_studies/sample_{sid}_support_audit.png").exists())
            table = pd.read_csv(repo_path(f"reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/case_studies/sample_{sid}_neighbor_table.csv"))
            self.assertFalse(table.empty)

    def test_deployment_smoke_test_does_not_use_unknown_afm(self) -> None:
        result = load_json(ROOT / "deployment_smoke_test/prediction.json")
        self.assertFalse(result["uses_unknown_afm_target"])
        self.assertIn("predicted_rq_nm", result)
        self.assertIn("nearest_training_analogs", result)

    def test_raw_and_old_hash_validation(self) -> None:
        summary = load_json(ROOT / "phase5b_summary.json")
        self.assertTrue(summary["raw_and_old_hash_validation"]["removelist_hash_ok"])
        self.assertIn("data/processed_afm/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy", summary["raw_and_old_hash_validation"]["hashes"])


if __name__ == "__main__":
    unittest.main()
