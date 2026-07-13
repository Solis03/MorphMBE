from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.rheed_peak_saddle.concepts import cache_key
from analysis.rheed_peak_saddle.data import infer_growth_stage
from analysis.rheed_peak_saddle.merge_tree import maximum_bottleneck_saddle
from analysis.rheed_peak_saddle.removelist import MISSING_6088_MESSAGE, assert_mandatory_removelist_ids
from analysis.rheed_peak_saddle.run import (
    eligible_truth_pairs,
    match_detected_to_truth_v2,
    measure_synthetic_example_v2,
    prune_lattice_duplicate_spots_v2,
    rank_inversions_v3,
    recovery_audit,
    run_analytical_saddle_tests,
    run_oracle_ladder_v2,
    solver_results_for_split_v3,
    spot_estimate_from_truth,
    synthetic_stage_dependency_audit,
    validate_completed_stage_review,
)
from analysis.rheed_peak_saddle.row_grouping import assign_lattice_indices, form_lattice_adjacent_pairs, group_spot_rows
from analysis.rheed_peak_saddle.synthetic import DEVELOPMENT_SEEDS, DEVELOPMENT_V2_SEEDS, HOLDOUT_SEEDS, HOLDOUT_V2_SEEDS, make_synthetic_split, make_synthetic_split_v2, render_synthetic_rheed
from analysis.rheed_peak_saddle.semantic_v3 import (
    DEVELOPMENT_V3_SEEDS,
    HOLDOUT_V3_SEEDS,
    independent_maximin_saddle,
    make_semantic_templates,
    median_oracle_for_nominal,
    render_semantic_template,
    solve_nominal_for_target,
)
from analysis.rheed_peak_saddle.pair_features import pair_masks
from analysis.rheed_peak_saddle.run import spot_estimate_from_truth_v3_proxy
from analysis.rheed_peak_saddle.metric_audit_v3 import (
    build_metric_lineage_rows,
    calibrated_target_family_metrics,
    corrected_metric_summary,
    family_fraction_ge,
    file_sha256 as audit_file_sha256,
    nominal_control_family_metrics,
    read_csv_rows as audit_read_csv_rows,
)
from analysis.rheed_roughness.run import read_config
from analysis.rheed_single_frame.removelist import RemovelistAudit, RemovelistRecord, load_removelist_audit


def _audit(root: Path, ids: tuple[str, ...]) -> RemovelistAudit:
    records = tuple(RemovelistRecord(sample_id=sid, raw_line=sid, note="", source_path=root / "removelist.txt") for sid in ids)
    return RemovelistAudit(path=root / "removelist.txt", sha256="hash", mtime="0", parser="test", sample_ids=ids, records=records)


class PeakSaddleStage0Test(unittest.TestCase):
    def test_mandatory_6088_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Sample 6088 is not present"):
                assert_mandatory_removelist_ids(_audit(root, ("6023",)))
            assert_mandatory_removelist_ids(_audit(root, ("6023", "6088")))
            try:
                assert_mandatory_removelist_ids(_audit(root, ("6023",)))
            except RuntimeError as exc:
                self.assertEqual(str(exc), MISSING_6088_MESSAGE)

    def test_removelist_parses_leading_numeric_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "removelist.txt"
            path.write_text("6061 -black holes\n6088 - bad\n", encoding="utf-8")
            audit = load_removelist_audit(root, "removelist.txt")
            self.assertEqual(audit.sample_ids, ("6061", "6088"))

    def test_growth_stage_filename_inference(self) -> None:
        self.assertEqual(infer_growth_stage("select_150_min_GaSb.jpg"), "active_growth")
        self.assertEqual(infer_growth_stage("select_rampdown_to_200C.jpg"), "rampdown_or_cooldown")
        self.assertEqual(infer_growth_stage("select_after_ramping_down_GaSb.jpg"), "rampdown_or_cooldown")
        self.assertEqual(infer_growth_stage("select_oxide_desorption.jpg"), "oxide_or_substrate")
        self.assertEqual(infer_growth_stage("select_mystery.jpg"), "unknown")

    def test_cache_key_changes_with_removelist_hash(self) -> None:
        key_a = cache_key(removelist_hash="a", feature_spec={"version": "v1"})
        key_b = cache_key(removelist_hash="b", feature_spec={"version": "v1"})
        key_c = cache_key(removelist_hash="a", feature_spec={"version": "v2"})
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_maximum_bottleneck_saddle_known_corridor(self) -> None:
        image = np.zeros((5, 7), dtype=float)
        image[2, 1] = 1.0
        image[2, 5] = 0.9
        image[2, 2:5] = 0.4
        seed_a = np.zeros_like(image, dtype=bool)
        seed_b = np.zeros_like(image, dtype=bool)
        seed_a[2, 1] = True
        seed_b[2, 5] = True
        corridor = np.zeros_like(image, dtype=bool)
        corridor[2, 1:6] = True
        result = maximum_bottleneck_saddle(image, seed_a, seed_b, corridor)
        self.assertTrue(result.connected)
        self.assertAlmostEqual(result.saddle_intensity, 0.4)

    def test_completed_stage_review_validation_on_checkpoint0_files(self) -> None:
        config_path = Path("configs/rheed_peak_saddle.yaml")
        config = read_config(config_path)
        audit = recovery_audit(config_path, config)
        self.assertEqual(audit.manifest_count, 25)
        self.assertEqual(audit.stage_review.row_count, 25)
        self.assertIn("6088", audit.removelist_payload["parsed_sample_ids"])
        self.assertEqual(audit.stage_review.approved_stage_counts["active_growth"], 7)

    def test_completed_stage_review_file_not_modified_by_validation(self) -> None:
        config = read_config(Path("configs/rheed_peak_saddle.yaml"))
        from analysis.rheed_peak_saddle.data import make_paths

        paths = make_paths(config)
        completed = paths.annotations_dir / "stage_review_completed.csv"
        before = (completed.read_bytes(), completed.stat().st_mtime)
        removelist = load_removelist_audit(paths.repo_root, config.get("removelist_path"))
        validate_completed_stage_review(paths, set(removelist.sample_ids))
        after = (completed.read_bytes(), completed.stat().st_mtime)
        self.assertEqual(before, after)

    def test_synthetic_renderer_is_deterministic(self) -> None:
        a = render_synthetic_rheed(split="development", seed=123, image_id="a", bridge_strength=0.4)
        b = render_synthetic_rheed(split="development", seed=123, image_id="a", bridge_strength=0.4)
        self.assertTrue(np.array_equal(a.image, b.image))
        self.assertEqual(a.spots, b.spots)
        self.assertEqual(a.pairs, b.pairs)

    def test_development_and_holdout_seeds_are_separated(self) -> None:
        self.assertTrue(set(DEVELOPMENT_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)))
        dev = make_synthetic_split("development")
        hold = make_synthetic_split("holdout")
        self.assertGreater(len(dev), 20)
        self.assertEqual(len(dev), len(hold))
        self.assertNotEqual(dev[0].nuisance["seed"], hold[0].nuisance["seed"])

    def test_analytical_saddle_suite_passes(self) -> None:
        rows = run_analytical_saddle_tests()
        self.assertGreaterEqual(len(rows), 9)
        failures = [row for row in rows if not int(row["passed"])]
        self.assertEqual(failures, [])

    def test_synthetic_stage_has_no_afm_rq_dependency(self) -> None:
        audit = synthetic_stage_dependency_audit()
        self.assertTrue(audit["passed"], audit["hits"])

    def test_oracle_pair_evaluator_returns_perfect_precision_recall(self) -> None:
        examples = make_synthetic_split_v2("development_v2")[:4]
        rows, _, _ = run_oracle_ladder_v2(examples)
        oracle_a = next(row for row in rows if row["oracle"] == "A")
        self.assertEqual(oracle_a["adjacent_pair_precision"], 1.0)
        self.assertEqual(oracle_a["adjacent_pair_recall"], 1.0)

    def test_eligible_ineligible_pair_denominator(self) -> None:
        ex = render_synthetic_rheed(
            split="development_v2",
            seed=777,
            image_id="missing",
            bridge_strength=0.5,
            missing_site_indices=((0, 2),),
        )
        eligible, ineligible = eligible_truth_pairs(ex)
        self.assertGreater(len(eligible), 0)
        self.assertTrue(any("missing_endpoint" in pair.ineligible_reason for pair in ineligible.values()))

    def test_one_to_one_detected_true_matching(self) -> None:
        ex = render_synthetic_rheed(split="development_v2", seed=778, image_id="match", bridge_strength=0.3)
        truth = [spot for spot in ex.spots if not spot.missing and not spot.edge_or_crop_flag]
        detected = [spot_estimate_from_truth(truth[0], 0), spot_estimate_from_truth(truth[0], 1)]
        det_to_truth, truth_to_det, _ = match_detected_to_truth_v2(ex, detected)
        self.assertEqual(len(truth_to_det), 1)
        self.assertEqual(len(det_to_truth), 1)

    def test_lattice_assignment_and_no_missing_site_pairing(self) -> None:
        ex = render_synthetic_rheed(
            split="development_v2",
            seed=779,
            image_id="gap",
            bridge_strength=0.4,
            missing_site_indices=((0, 2),),
            row_count=1,
            spots_per_row=6,
            spacing_jitter=0.0,
            read_noise_sigma=0.0,
        )
        measurement = measure_synthetic_example_v2(ex, oracle="D")
        self.assertEqual(measurement["image_row"]["false_adjacency_across_missing_site_rate"], 0.0)

    def test_duplicate_detection_pruning(self) -> None:
        ex = render_synthetic_rheed(split="development_v2", seed=780, image_id="dup", bridge_strength=0.5, duplicate_spot_artifact=True)
        measurement = measure_synthetic_example_v2(ex, oracle="D")
        self.assertGreaterEqual(measurement["image_row"]["spot_detection_precision"], 0.95)

    def test_partial_crop_invalid_pair_rejection(self) -> None:
        ex = render_synthetic_rheed(split="development_v2", seed=781, image_id="crop", bridge_strength=0.5, partial_crop=True, crop_left=8.0)
        measurement = measure_synthetic_example_v2(ex, oracle="D")
        self.assertGreaterEqual(measurement["image_row"]["invalid_pair_rejection_accuracy"], 0.90)

    def test_holdout_v2_seed_separation(self) -> None:
        self.assertTrue(set(HOLDOUT_V2_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)))
        self.assertTrue(set(HOLDOUT_V2_SEEDS).isdisjoint(set(DEVELOPMENT_V2_SEEDS)))

    def test_v3_renderer_component_separation(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        render = render_semantic_template(template, target=0.5, nominal_bridge_control=0.4, image_id="v3")
        self.assertEqual(render.spot_signal_clean.shape, render.morphology_signal_clean.shape)
        self.assertGreater(float(render.spot_signal_clean.max()), 0.0)
        self.assertGreaterEqual(float(render.explicit_bridge_signal_clean.max()), 0.0)
        self.assertGreater(float(render.noisy_observed_linear.max()), float(render.morphology_signal_clean.max()) * 0.5)

    def test_v3_independent_oracle_matches_union_find(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        render = render_semantic_template(template, target=0.5, nominal_bridge_control=0.4, image_id="v3_oracle")
        pair = render.pairs[0]
        estimates = [spot_estimate_from_truth_v3_proxy(spot, i) for i, spot in enumerate(render.spots)]
        masks = pair_masks(render.morphology_signal_clean.shape, estimates[pair.spot_i], estimates[pair.spot_j])
        prod = maximum_bottleneck_saddle(render.morphology_signal_clean, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
        oracle, _ = independent_maximin_saddle(render.morphology_signal_clean, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
        self.assertAlmostEqual(prod.saddle_intensity, oracle, places=5)

    def test_v3_zero_bridge_can_have_nonzero_tail_adhesion(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        value = median_oracle_for_nominal(template, 0.0)
        self.assertGreater(value, 0.0)

    def test_v3_nominal_control_not_assumed_identical_to_visual_adhesion(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        value = median_oracle_for_nominal(template, 0.5)
        self.assertGreater(abs(value - 0.5), 0.01)

    def test_v3_target_solver_converges_or_marks_unattainable(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        solved = solve_nominal_for_target(template, 0.55)
        self.assertIn(solved["solver_status"], {"converged", "max_iter", "unattainable"})
        if solved["solver_status"] != "unattainable":
            self.assertLessEqual(abs(solved["achieved_oracle_visual_adhesion"] - 0.55), 0.02)

    def test_v3_unattainable_target_handling(self) -> None:
        template = make_semantic_templates("development_v3", count=1)[0]
        solved = solve_nominal_for_target(template, 0.0)
        self.assertIn("attainable_min", solved)
        if solved["attainable_min"] > 0.01:
            self.assertEqual(solved["solver_status"], "unattainable")

    def test_v3_solver_table_records_unattainable_targets(self) -> None:
        rows = solver_results_for_split_v3("development_v3", template_count=1)
        statuses = {row["solver_status"] for row in rows}
        self.assertIn("unattainable", statuses)
        self.assertTrue(any(float(row["target_visual_adhesion"]) == 0.0 for row in rows))

    def test_v3_rank_inversion_extraction(self) -> None:
        rows = [
            {"pair_id": "a", "image_id": "ia", "valid": 1, "matched_truth_pair": 1, "oracle_visual_adhesion_clean": 0.1, "estimated_adhesion_observed": 0.8},
            {"pair_id": "b", "image_id": "ib", "valid": 1, "matched_truth_pair": 1, "oracle_visual_adhesion_clean": 0.7, "estimated_adhesion_observed": 0.2},
        ]
        self.assertEqual(len(rank_inversions_v3(rows)), 1)

    def test_v3_holdout_seed_separation(self) -> None:
        self.assertTrue(set(HOLDOUT_V3_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)))
        self.assertTrue(set(HOLDOUT_V3_SEEDS).isdisjoint(set(HOLDOUT_V2_SEEDS)))
        self.assertTrue(set(HOLDOUT_V3_SEEDS).isdisjoint(set(DEVELOPMENT_V3_SEEDS)))

    def test_v3_metric_lineage_identifies_branch_a_columns(self) -> None:
        rows = build_metric_lineage_rows()
        historical = next(row for row in rows if row["metric_name"].startswith("historical"))
        corrected = next(row for row in rows if row["metric_name"].startswith("corrected"))
        self.assertEqual(historical["x_column"], "nominal_bridge_control")
        self.assertEqual(historical["y_column"], "oracle_visual_adhesion_clean")
        self.assertEqual(historical["matches_preregistered_definition"], 0)
        self.assertEqual(corrected["x_column"], "target_visual_adhesion")
        self.assertEqual(corrected["y_column"], "achieved_oracle_visual_adhesion")
        self.assertEqual(corrected["matches_preregistered_definition"], 1)

    def test_v3_calibrated_family_metric_excludes_unattainable_targets(self) -> None:
        rows = audit_read_csv_rows(Path("outputs/rheed_peak_saddle/synthetic_v3/target_adhesion_solver_results.csv"))
        metrics = calibrated_target_family_metrics(rows)
        first = next(row for row in metrics if row["split"] == "development_v3" and row["family_id"] == "development_v3_template_00")
        self.assertEqual(first["number_of_successful_targets"], 12)
        self.assertEqual(first["unattainable_target_count"], 8)
        self.assertEqual(first["solver_failure_count"], 0)
        self.assertAlmostEqual(float(first["spearman_rho"]), 1.0)
        self.assertAlmostEqual(family_fraction_ge(metrics, split="holdout_v3"), 1.0)

    def test_v3_nominal_and_target_family_analyses_use_different_columns(self) -> None:
        old_rows = audit_read_csv_rows(Path("outputs/rheed_peak_saddle/synthetic_v3/old_control_identifiability.csv"))
        solver_rows = audit_read_csv_rows(Path("outputs/rheed_peak_saddle/synthetic_v3/target_adhesion_solver_results.csv"))
        nominal = nominal_control_family_metrics(old_rows)
        calibrated = calibrated_target_family_metrics(solver_rows)
        self.assertEqual(nominal[0]["x_column"], "nominal_bridge_control")
        self.assertEqual(calibrated[0]["x_column"], "target_visual_adhesion")
        self.assertNotEqual(nominal[0]["y_column"], calibrated[0]["y_column"])

    def test_v3_metric_correction_summary_records_branch_a_pass(self) -> None:
        original = audit_read_csv_rows(Path("outputs/rheed_peak_saddle/synthetic_v3/holdout_v3_metrics.csv"))
        solver_rows = audit_read_csv_rows(Path("outputs/rheed_peak_saddle/synthetic_v3/target_adhesion_solver_results.csv"))
        correction, status = corrected_metric_summary(original, calibrated_target_family_metrics(solver_rows))
        row = next(item for item in correction if item["criterion"] == "within_family_fraction_ge_0_99")
        self.assertEqual(status, "STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION")
        self.assertEqual(float(row["historical_reported_value"]), 0.75)
        self.assertEqual(float(row["corrected_preregistered_value"]), 1.0)
        self.assertEqual(row["corrected_pass"], "PASS")

    def test_v3_metric_audit_preserves_immutable_receipt_hash(self) -> None:
        hash_record = Path("outputs/rheed_peak_saddle/synthetic_v3/metric_audit_hashes_before_after.json")
        self.assertTrue(hash_record.is_file())
        payload = __import__("json").loads(hash_record.read_text(encoding="utf-8"))
        self.assertEqual(payload["immutable_hashes_changed"], [])
        receipt = "evaluation_receipt.json"
        self.assertEqual(payload["immutable_hashes_before"][receipt], payload["immutable_hashes_after"][receipt])
        self.assertEqual(payload["immutable_hashes_after"][receipt], audit_file_sha256(Path("outputs/rheed_peak_saddle/synthetic_v3/evaluation_receipt.json")))

    def test_v3_metric_audit_visual_outputs_are_image_panels(self) -> None:
        report_dir = Path("reports/rheed_peak_saddle/synthetic_v3")
        for name in (
            "largest_rank_inversions_visual.png",
            "lattice_indexing_examples_visual.png",
            "high_adhesion_error_cases.png",
        ):
            path = report_dir / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for name in (
            "largest_rank_inversions_visual.pdf",
            "lattice_indexing_examples_visual.pdf",
            "high_adhesion_error_cases.pdf",
        ):
            path = report_dir / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_v3_metric_audit_report_and_review_template(self) -> None:
        report = Path("reports/rheed_peak_saddle/checkpoint_1c_metric_audit.md")
        self.assertTrue(report.is_file())
        text = report.read_text(encoding="utf-8")
        self.assertIn("BRANCH A - CLEAR METRIC-LINEAGE BUG", text)
        self.assertIn("STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION", text)
        index = Path("reports/rheed_peak_saddle/synthetic_v3/human_review_index.html").read_text(encoding="utf-8")
        self.assertIn("lattice_indexing_examples_visual.png", index)
        approval = Path("annotations/rheed_peak_saddle/approvals/checkpoint_1c_visual_review_template.txt")
        self.assertTrue(approval.is_file())
        self.assertNotEqual(approval.read_text(encoding="utf-8").strip(), "APPROVED")


if __name__ == "__main__":
    unittest.main()
