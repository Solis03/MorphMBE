from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.common import repo_path


ROOT = Path("outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase6a_exhaustive_discovery")
REPORT = Path("reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase6a_exhaustive_discovery")
REMOVED = {"6023", "6087"}


def load_json(path: str | Path) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"


class Phase6ATest(unittest.TestCase):
    def test_canonical_index_has_23_active_and_no_removed_samples(self) -> None:
        idx = pd.read_csv(repo_path(ROOT / "canonical_index/canonical_sample_index.csv"), dtype={"sample_id": str})
        active = idx[idx["is_primary"].map(truthy)]
        self.assertEqual(23, active["sample_id"].nunique())
        self.assertFalse(REMOVED & set(active["sample_id"]))
        self.assertTrue(active["sample_id"].eq(active["growth_run_id"].astype(str)).all())
        self.assertTrue(active["rheed_metadata_path"].notna().all())
        self.assertTrue(active["second_order_representative_afm_path"].notna().all())

    def test_alignment_audit_and_hashes_are_recorded(self) -> None:
        audit = pd.read_csv(repo_path(ROOT / "canonical_index/canonical_alignment_audit.csv"))
        summary = load_json(ROOT / "phase6a_summary.json")
        self.assertTrue(audit["passed"].map(truthy).all())
        self.assertTrue(summary["canonical_mapping_passed"])
        self.assertTrue(summary["removed_samples_excluded"])
        self.assertEqual("8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b", summary["raw_old_hash_validation"]["removelist.txt"])

    def test_outer_splits_are_strict_n_minus_1_and_cross_support_6095_6099(self) -> None:
        splits = pd.read_csv(repo_path(ROOT / "provenance/outer_splits.csv"), dtype={"heldout_id": str})
        ids = set(splits["heldout_id"])
        self.assertEqual(23, len(splits))
        self.assertEqual(23, len(ids))
        for row in splits.to_dict("records"):
            training = set(json.loads(row["training_ids"]))
            self.assertEqual(22, int(row["training_count"]))
            self.assertEqual(ids - {str(row["heldout_id"])}, training)
            self.assertTrue(truthy(row["split_valid"]))
        self.assertIn("6099", set(json.loads(splits[splits["heldout_id"].eq("6095")].iloc[0]["training_ids"])))
        self.assertIn("6095", set(json.loads(splits[splits["heldout_id"].eq("6099")].iloc[0]["training_ids"])))

    def test_trial_registry_resume_outputs_are_complete_and_unique(self) -> None:
        reg = pd.read_csv(repo_path(ROOT / "trials/trial_registry.csv"))
        leaderboard = pd.read_csv(repo_path(ROOT / "trials/trial_leaderboard.csv"))
        summary = load_json(ROOT / "phase6a_summary.json")
        self.assertEqual(120, len(reg))
        self.assertEqual(120, len(leaderboard))
        self.assertEqual(120, reg["trial_id"].nunique())
        self.assertTrue(reg["status"].eq("completed").all())
        self.assertEqual(120, summary["completed_trial_count"])
        self.assertEqual(0, summary["failed_trial_count"])

    def test_strict_oof_predictions_are_one_per_sample_and_target_blind(self) -> None:
        pred = pd.read_csv(repo_path(ROOT / "strict_oof_predictions.csv"), dtype={"sample_id": str, "heldout_id": str})
        self.assertEqual(23, len(pred))
        self.assertEqual(23, pred["sample_id"].nunique())
        self.assertTrue((pred["sample_id"] == pred["heldout_id"]).all())
        self.assertTrue((pred["training_count"] == 22).all())
        self.assertTrue(pred["true_target_lookup"].eq("heldout_sample_id").all())
        self.assertFalse(pred["outer_target_used_for_selection"].map(truthy).any())
        self.assertFalse(REMOVED & set(pred["sample_id"]))

    def test_all_oof_predictions_cover_each_completed_trial(self) -> None:
        all_pred = pd.read_csv(repo_path(ROOT / "all_oof_predictions.csv"), dtype={"sample_id": str})
        per_trial = all_pred.groupby("trial_id")["sample_id"].nunique()
        self.assertEqual(120, len(per_trial))
        self.assertTrue((per_trial == 23).all())
        self.assertTrue((all_pred["training_count"] == 22).all())
        self.assertFalse(all_pred["outer_target_used_for_selection"].map(truthy).any())

    def test_best_metrics_and_phase5b_reference_are_from_artifacts(self) -> None:
        summary = load_json(ROOT / "phase6a_summary.json")
        best = summary["best_fixed_metrics"]
        self.assertAlmostEqual(1.2857150865731934, best["MAE"], places=9)
        self.assertGreater(best["Spearman"], summary["old_vs_new_comparison"]["phase5b_reconstructed_r4"]["Spearman"])
        self.assertLess(best["MAE"], summary["old_vs_new_comparison"]["phase5b_reconstructed_r4"]["MAE"])
        self.assertFalse(summary["acceptance"]["prediction_target_met"])
        self.assertTrue(summary["acceptance"]["retrieval_target_met"])

    def test_descriptor_and_prototype_predictions_are_cross_fitted(self) -> None:
        desc = pd.read_csv(repo_path(ROOT / "all_descriptor_predictions.csv"), dtype={"sample_id": str})
        proto = pd.read_csv(repo_path(ROOT / "prototype_predictions.csv"), dtype={"sample_id": str})
        self.assertEqual(23, desc["sample_id"].nunique())
        self.assertEqual(23, proto["sample_id"].nunique())
        for row in desc.to_dict("records"):
            self.assertNotIn(str(row["sample_id"]), set(json.loads(row["descriptor_scaler_fit_ids"])))
        for row in proto.to_dict("records"):
            self.assertNotIn(str(row["sample_id"]), set(json.loads(row["prototype_clustering_fit_ids"])))

    def test_retrieval_and_synthesis_do_not_use_heldout_or_removed_bank_entries(self) -> None:
        ret = pd.read_csv(repo_path(ROOT / "retrieval_audit.csv"), dtype={"sample_id": str, "source_sample_id": str})
        syn = pd.read_csv(repo_path(ROOT / "synthesis_metrics.csv"), dtype={"sample_id": str, "source_sample_id": str})
        self.assertEqual(23, len(ret))
        self.assertEqual(23, len(syn))
        self.assertFalse((ret["sample_id"] == ret["source_sample_id"]).any())
        self.assertFalse(ret["heldout_in_retrieval_bank"].map(truthy).any())
        self.assertFalse(ret["patch_bank_contains_heldout"].map(truthy).any())
        self.assertTrue(ret["s1_label"].eq("S1 retrieval").all())
        self.assertFalse(REMOVED & set(ret["source_sample_id"]))
        self.assertTrue(syn["identity_audit_pass"].map(truthy).all())

    def test_full_cohort_deployment_is_labeled_as_not_test_result(self) -> None:
        registry = load_json(ROOT / "deployment/deployment_model/registry.json")
        sample_index = pd.read_csv(repo_path(ROOT / "deployment/deployment_model/sample_index.csv"), dtype={"sample_id": str})
        smoke = load_json(ROOT / "deployment/smoke_test/prediction.json")
        self.assertEqual(23, registry["training_sample_count"])
        self.assertIn("NOT AN INDEPENDENT TEST RESULT", registry["warning"])
        self.assertEqual(23, sample_index["sample_id"].nunique())
        self.assertFalse(REMOVED & set(sample_index["sample_id"]))
        self.assertFalse(smoke["uses_unknown_afm_target"])
        self.assertGreater(len(smoke["predicted_AFM_descriptors"]), 0)

    def test_required_tables_dashboard_reports_and_figures_exist(self) -> None:
        for rel in [
            "trial_leaderboard.csv",
            "finalist_metrics.csv",
            "all_oof_predictions.csv",
            "all_descriptor_predictions.csv",
            "prototype_predictions.csv",
            "retrieval_audit.csv",
            "synthesis_metrics.csv",
            "dashboard/sample_level_master_table.csv",
            "dashboard/sample_level_master_table.html",
        ]:
            self.assertTrue(repo_path(ROOT / rel).exists(), rel)
        for rel in [
            "phase6a_report.md",
            "executive_summary.md",
            "methods_summary.md",
            "claims_and_limitations.md",
            "figure_captions.md",
            "dashboard/results_dashboard.html",
        ]:
            self.assertTrue(repo_path(REPORT / rel).exists(), rel)
        for i in range(1, 13):
            matches = list(repo_path(REPORT / "figures").glob(f"Fig{i}_*.png"))
            self.assertEqual(1, len(matches), f"Fig{i}")

    def test_summary_has_6095_6099_and_claims_limitations(self) -> None:
        summary = load_json(ROOT / "phase6a_summary.json")
        claims = repo_path(REPORT / "claims_and_limitations.md").read_text(encoding="utf-8")
        self.assertEqual("6099", summary["samples_6095_6099_audit"]["6095"]["cross_support_sample_present"])
        self.assertEqual("6095", summary["samples_6095_6099_audit"]["6099"]["cross_support_sample_present"])
        self.assertIn("Cannot claim", claims)
        self.assertIn("independent external validation", claims)


if __name__ == "__main__":
    unittest.main()
