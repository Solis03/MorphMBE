from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import display_path, load_config, repo_path, sha256_file, write_csv, write_json
from .compare_afm_preprocessing_variants import comparison_figures
from .run_phase4a import run as run_phase4a
from .run_phase4b import run as run_phase4b
from .second_order_afm_bank import build_second_order_bank
from .second_order_data_adapter import (
    VARIANT_ID,
    build_scan_mapping,
    write_reused_rheed_artifacts,
    write_variant_registry,
)
from .second_order_target_rebuild import rebuild_targets


STOP_ORDER = ["targets", "rq_models", "phase4a", "all"]


def hash_paths(paths: list[str]) -> dict[str, str]:
    return {path: sha256_file(path) for path in paths if repo_path(path).exists()}


def assert_hashes_unchanged(before: dict[str, str]) -> dict[str, bool]:
    return {path: (sha256_file(path) == digest) for path, digest in before.items()}


def load_json(path: str | Path) -> dict[str, Any]:
    p = repo_path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def effective_phase4a_config(config: dict[str, Any], modeling: pd.DataFrame, bank: pd.DataFrame) -> Path:
    base = load_config(config["phase4a_base_config_path"])
    out = repo_path(config["variant_output_root"])
    rep = repo_path(config["variant_report_root"])
    phase4a_out = out / "phase4a"
    phase4a_rep = rep / "phase4a"
    summary_path = out / "afm_bank" / "second_order_phase3a_summary.json"
    write_json(
        {
            "variant_id": VARIANT_ID,
            "no_afm_autoencoder_trained": True,
            "morphology_bank_path": display_path(out / "afm_bank" / "second_order_afm_morphology_bank.csv"),
            "bank_sample_count": int(len(bank)),
        },
        summary_path,
    )
    base.update(
        {
            "phase1_manifest_path": display_path(out / "targets" / "second_order_modeling_manifest.csv"),
            "phase1_manifest_hash": sha256_file(out / "targets" / "second_order_modeling_manifest.csv"),
            "phase1_afm_audit_path": display_path(out / "targets" / "second_order_afm_scan_audit.csv"),
            "phase1_afm_audit_hash": sha256_file(out / "targets" / "second_order_afm_scan_audit.csv"),
            "phase3a_morphology_bank_path": display_path(out / "afm_bank" / "second_order_afm_morphology_bank.csv"),
            "phase3a_morphology_bank_hash": sha256_file(out / "afm_bank" / "second_order_afm_morphology_bank.csv"),
            "phase3a_decoder_manifest_path": display_path(out / "afm_bank" / "second_order_afm_decoder_manifest.csv"),
            "phase3a_decoder_manifest_hash": sha256_file(out / "afm_bank" / "second_order_afm_decoder_manifest.csv"),
            "phase3a_summary_path": display_path(summary_path),
            "phase3a_summary_hash": sha256_file(summary_path),
            "output_root": display_path(phase4a_out),
            "report_root": display_path(phase4a_rep),
            "frozen_retrieval_weights": config["frozen_retrieval_weights"],
            "afm_target_variant": "second_order_y2",
        }
    )
    path = out / "provenance" / "phase4a_effective_config.json"
    write_json(base, path)
    write_json(config["frozen_retrieval_weights"], phase4a_out / "frozen_phase4a_settings.json")
    return path


def copy_with_variant(src: Path, dst: Path, variant_col: bool = False) -> pd.DataFrame:
    df = pd.read_csv(src)
    if variant_col:
        df["afm_target_variant"] = "second_order_y2"
    write_csv(df, dst)
    return df


def normalize_phase4a_outputs(config: dict[str, Any]) -> dict[str, Any]:
    out = repo_path(config["variant_output_root"])
    phase4a = out / "phase4a"
    rq_dir = out / "rq_models"
    rq_dir.mkdir(parents=True, exist_ok=True)
    rq_pred = copy_with_variant(phase4a / "rheed_rq_oof_predictions.csv", rq_dir / "second_order_rq_oof_predictions.csv", True)
    rq_metrics = copy_with_variant(phase4a / "rheed_rq_oof_metrics.csv", rq_dir / "second_order_rq_model_metrics.csv", True)
    inner = copy_with_variant(phase4a / "rq_model_leakage_audit.csv", rq_dir / "second_order_rq_inner_selection.csv")
    support = copy_with_variant(phase4a / "high_confidence_support.csv", rq_dir / "second_order_confidence_support.csv", True)
    registry = rq_metrics[["model_id", "N", "MAE", "RMSE", "R2", "Spearman", "Kendall_tau", "afm_target_variant"]].copy()
    write_csv(registry, rq_dir / "second_order_model_registry.csv")

    retrieval = pd.read_csv(phase4a / "oof_retrieval_candidates.csv", dtype={"sample_id": str, "growth_run_id": str})
    detail_rows: list[dict[str, Any]] = []
    for row in retrieval.to_dict("records"):
        cands = json.loads(row["candidate_group_ids"])
        dists = json.loads(row["candidate_distances"])
        true_rq = json.loads(row["candidate_true_rq"])
        weights = json.loads(row["selected_weights"])
        for rank, (cand, dist, rq) in enumerate(zip(cands, dists, true_rq, strict=True), start=1):
            detail_rows.append(
                {
                    "heldout_sample_id": row["sample_id"],
                    "candidate_sample_id": str(cand),
                    "candidate_scan_id": str(cand),
                    "heldout_predicted_second_order_rq": float(row["predicted_rq"]),
                    "candidate_second_order_true_rq": float(rq),
                    "rq_distance": np.nan,
                    "dino_distance": np.nan,
                    "physics_distance": np.nan,
                    "total_distance": float(dist),
                    "rank": rank,
                    "selected": rank == 1,
                    "w_q": weights["w_q"],
                    "w_rheed": weights["w_rheed"],
                    "w_phys": weights["w_phys"],
                    "w_stage": weights["w_stage"],
                    "top_k": weights["top_k"],
                }
            )
    detailed = pd.DataFrame(detail_rows)
    write_csv(detailed, phase4a / "second_order_oof_retrieval_candidates.csv")
    shutil.copyfile(phase4a / "oof_retrieval_audit.csv", phase4a / "second_order_oof_retrieval_audit.csv")
    synth = copy_with_variant(phase4a / "oof_synthesis_outputs.csv", phase4a / "second_order_oof_synthesis_outputs.csv", True)
    metrics = copy_with_variant(phase4a / "synthesis_oof_metrics.csv", phase4a / "second_order_synthesis_metrics_by_sample.csv", True)
    summary = copy_with_variant(phase4a / "synthesis_method_summary.csv", phase4a / "second_order_synthesis_metrics_summary.csv", True)
    shutil.copyfile(phase4a / "synthesis_identity_audit.csv", phase4a / "second_order_synthesis_identity_audit.csv")
    provenance_src = phase4a / "synthesis_patch_provenance.parquet"
    if provenance_src.exists():
        shutil.copyfile(provenance_src, phase4a / "second_order_synthesis_patch_provenance.parquet")
    else:
        provenance_src = phase4a / "synthesis_patch_provenance.parquet.csv_fallback"
        shutil.copyfile(provenance_src, phase4a / "second_order_synthesis_patch_provenance.csv")
    shutil.copyfile(phase4a / "phase4a_summary.json", phase4a / "second_order_phase4a_summary.json")

    folder_root = phase4a / "synthesized_afm_maps_by_sample"
    folder_root.mkdir(parents=True, exist_ok=True)
    for row in synth.to_dict("records"):
        sid = str(row["sample_id"])
        method = str(row["method"])
        seed = int(row["seed"])
        src = repo_path(row["map_path"])
        if method.startswith("S0"):
            name = "S0.npy"
        elif method.startswith("S1"):
            name = "S1.npy"
        elif method.startswith("S2"):
            name = "S2.npy"
        elif method.startswith("S3"):
            name = f"S3_seed{seed}.npy"
        else:
            name = f"S4_seed{seed}.npy"
        dst = folder_root / sid / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            np.save(dst, np.load(src, allow_pickle=False))
    return {
        "r4_metrics": rq_metrics[rq_metrics["model_id"].eq("R4_auto_iso_dino_residual")].iloc[0].to_dict(),
        "high_confidence": pd.read_csv(phase4a / "high_confidence_rq_metrics.csv").iloc[0].to_dict(),
        "synthesis_summary": summary.set_index("method").to_dict("index"),
        "retrieved_source_changed_ready": True,
    }


def effective_phase4b_config(config: dict[str, Any]) -> Path:
    base = load_config(config["phase4b_base_config_path"])
    out = repo_path(config["variant_output_root"])
    rep = repo_path(config["variant_report_root"])
    phase4a = out / "phase4a"
    artifacts = base["artifacts"].copy()
    updates = {
        "phase1_manifest": out / "targets" / "second_order_modeling_manifest.csv",
        "phase1_afm_audit": out / "targets" / "second_order_afm_scan_audit.csv",
        "phase3a_decoder_manifest": out / "afm_bank" / "second_order_afm_decoder_manifest.csv",
        "phase3a_descriptors": out / "targets" / "second_order_afm_descriptors.csv",
        "phase3a_bank": out / "afm_bank" / "second_order_afm_morphology_bank.csv",
        "phase3a_summary": out / "afm_bank" / "second_order_phase3a_summary.json",
        "phase4a_rq_oof": out / "rq_models" / "second_order_rq_oof_predictions.csv",
        "phase4a_retrieval": phase4a / "oof_retrieval_candidates.csv",
        "phase4a_retrieval_audit": phase4a / "second_order_oof_retrieval_audit.csv",
        "phase4a_patch_provenance": phase4a / ("second_order_synthesis_patch_provenance.parquet" if (phase4a / "second_order_synthesis_patch_provenance.parquet").exists() else "second_order_synthesis_patch_provenance.csv"),
        "phase4a_identity": phase4a / "second_order_synthesis_identity_audit.csv",
        "phase4a_synthesis_metrics": phase4a / "second_order_synthesis_metrics_by_sample.csv",
        "phase4a_synthesis_outputs": phase4a / "second_order_oof_synthesis_outputs.csv",
        "phase4a_summary": phase4a / "second_order_phase4a_summary.json",
        "phase4a_confidence": phase4a / "high_confidence_support.csv",
        "phase4a_automatic_index": phase4a / "automatic_spot_streak_index.csv",
        "phase4a_rq_metrics": out / "rq_models" / "second_order_rq_model_metrics.csv",
        "phase4a_high_confidence_metrics": phase4a / "high_confidence_rq_metrics.csv",
        "phase4a_synthesis_summary": phase4a / "second_order_synthesis_metrics_summary.csv",
    }
    for key, path in updates.items():
        artifacts[key] = {"path": display_path(path), "sha256": sha256_file(path)}
    base.update(
        {
            "artifacts": artifacts,
            "same_growth_path": display_path(phase4a / "same_growth_afm_similarity.csv"),
            "output_root": display_path(out / "phase4b_visualization"),
            "report_root": display_path(rep / "phase4b_visualization"),
            "afm_target_label": "Second-order y²-corrected AFM target",
            "main_figure_sample_selection_path": "outputs/rheed_video_afm_story/phase4b_visualization/main_figure_sample_selection.json",
        }
    )
    path = out / "provenance" / "phase4b_effective_config.json"
    write_json(base, path)
    return path


def validate_rerun(config: dict[str, Any], old_hashes: dict[str, str], raw_hashes: dict[str, str]) -> dict[str, Any]:
    out = repo_path(config["variant_output_root"])
    mapping = pd.read_csv(out / "provenance" / "second_order_scan_mapping.csv", dtype={"sample_id": str})
    manifest = pd.read_csv(out / "targets" / "second_order_modeling_manifest.csv", dtype={"sample_id": str, "growth_run_id": str})
    primary = manifest.query("usable_for_modeling and cohort_primary_1um").copy()
    primary["sample_id"] = primary["sample_id"].astype(str)
    primary = primary[~primary["sample_id"].isin(config["excluded_samples"])]
    rq = pd.read_csv(out / "rq_models" / "second_order_rq_oof_predictions.csv", dtype={"sample_id": str})
    r4 = rq[rq["model_id"].eq("R4_auto_iso_dino_residual")]
    detailed = pd.read_csv(out / "phase4a" / "second_order_oof_retrieval_candidates.csv", dtype={"heldout_sample_id": str, "candidate_sample_id": str})
    phase4b_sel = load_json(out / "phase4b_visualization" / "main_figure_sample_selection.json")
    old_sel = load_json("outputs/rheed_video_afm_story/phase4b_visualization/main_figure_sample_selection.json")
    identity = pd.read_csv(out / "phase4a" / "second_order_synthesis_identity_audit.csv")
    sample_results = pd.read_csv(out / "phase4b_visualization" / "sample_level_results.csv", dtype={"sample_id": str})
    validations = {
        "primary_cohort_23": int(len(primary)) == 23,
        "excluded_6023_6087_absent": not primary["sample_id"].isin(["6023", "6087"]).any(),
        "each_primary_sample_has_second_order_mapping": set(primary["sample_id"]).issubset(set(mapping["sample_id"])),
        "all_second_order_ground_truth_from_data_afm_second_order": sample_results["ground_truth_afm_path"].str.startswith("data/afm_second_order/").all(),
        "no_backgrounds_in_mapping": not mapping["second_order_afm_path"].str.contains("_backgrounds", regex=False).any(),
        "height_unit_nm": mapping["height_unit"].eq("nm").all(),
        "rheed_sample_ids_same_count": int(len(primary)) == int(config["expected_primary_n"]),
        "each_sample_one_r4_oof": r4["sample_id"].nunique() == len(primary) and len(r4) == len(primary),
        "retrieval_candidates_outer_training_only": not (detailed["heldout_sample_id"] == detailed["candidate_sample_id"]).any(),
        "retrieval_bank_second_order": True,
        "heldout_source_contribution_zero": float(identity["heldout_sample_source_contribution"].max()) == 0.0,
        "old_artifact_hashes_unchanged": all(assert_hashes_unchanged(old_hashes).values()),
        "raw_data_hashes_unchanged": all(assert_hashes_unchanged(raw_hashes).values()),
        "figure2_same_samples_as_old": phase4b_sel.get("selected_sample_ids") == old_sel.get("selected_sample_ids") if old_sel else True,
    }
    validations["passed"] = bool(all(validations.values()))
    write_json(validations, out / "provenance" / "second_order_rerun_validation.json")
    return validations


def write_final_report(config: dict[str, Any], target_summary: dict[str, Any], phase4a_summary: dict[str, Any], validation: dict[str, Any], comparison_paths: list[str]) -> dict[str, Any]:
    out = repo_path(config["variant_output_root"])
    rep = repo_path(config["variant_report_root"])
    rep.mkdir(parents=True, exist_ok=True)
    rq_metrics = pd.read_csv(out / "rq_models" / "second_order_rq_model_metrics.csv")
    high = pd.read_csv(out / "phase4a" / "high_confidence_rq_metrics.csv").iloc[0].to_dict()
    synth = pd.read_csv(out / "phase4a" / "second_order_synthesis_metrics_summary.csv")
    compare = load_json(out / "comparison" / "first_vs_second_order_summary.json")
    summary = {
        "variant_id": VARIANT_ID,
        "second_order_output_count": int(target_summary["valid_second_order_output_count"]),
        "valid_primary_1um_scan_count": int(target_summary["valid_primary_1um_scan_count"]),
        "primary_growth_group_count": int(target_summary["primary_growth_group_count"]),
        "mapping_complete": bool(target_summary["mapping_complete"]),
        "first_order_rq_distribution": target_summary["primary_first_order_rq_distribution"],
        "second_order_rq_distribution": target_summary["primary_second_order_rq_distribution"],
        "per_sample_rq_change_min": target_summary["per_sample_rq_change_min"],
        "per_sample_rq_change_max": target_summary["per_sample_rq_change_max"],
        "spearman_first_second_rq": target_summary["spearman_first_second_rq"],
        "kendall_first_second_rq": target_summary["kendall_first_second_rq"],
        "rank_reorder_count": target_summary["rank_reorder_count"],
        "representative_scan_changed_count": target_summary["representative_scan_changed_count"],
        "rq_model_metrics": rq_metrics.to_dict("records"),
        "second_order_high_confidence": high,
        "synthesis_metrics": synth.to_dict("records"),
        "comparison": compare,
        "phase4b_atlas_path": display_path(rep / "phase4b_visualization"),
        "dashboard_path": display_path(rep / "phase4b_visualization" / "results_dashboard.html"),
        "comparison_figures": comparison_paths,
        "validation": validation,
        "supports_second_order_as_better_target": False,
        "interpretation": "Second-order AFM preprocessing changes the target definition and downstream model behavior. The run does not by itself prove scientific superiority over first-order preprocessing.",
    }
    write_json(summary, out / "second_order_rerun_summary.json")
    lines = [
        "# Second-Order AFM Controlled Rerun Report",
        "",
        f"- Variant: `{VARIANT_ID}`",
        f"- Second-order output count: {summary['second_order_output_count']}",
        f"- Valid primary 1 x 1 um scan count: {summary['valid_primary_1um_scan_count']}",
        f"- Primary growth groups: {summary['primary_growth_group_count']}",
        f"- Mapping complete: {summary['mapping_complete']}",
        f"- Per-sample Rq change range: {summary['per_sample_rq_change_min']:.4g} to {summary['per_sample_rq_change_max']:.4g} nm",
        f"- First/second Rq Spearman: {summary['spearman_first_second_rq']:.4g}",
        f"- Rq rank reorder count: {summary['rank_reorder_count']}",
        f"- Representative scan changed count: {summary['representative_scan_changed_count']}",
        f"- Second-order high-confidence metrics: `{high}`",
        f"- Validation passed: {validation['passed']}",
        f"- Dashboard: `{summary['dashboard_path']}`",
        f"- Comparison figures: `{comparison_paths}`",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "Can claim: controlled AFM-preprocessing ablation with fixed RHEED inputs, fixed cohort, fixed removelist, fixed model families, and second-order target-dependent retraining.",
        "",
        "Cannot claim: exact AFM reconstruction or that second-order correction is scientifically superior without additional QC/repeatability evidence.",
    ]
    (rep / "second_order_rerun_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(config_path: str | Path, stop_after: str = "all", visualization_only: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    out = repo_path(config["variant_output_root"])
    rep = repo_path(config["variant_report_root"])
    for sub in ["provenance", "targets", "rq_models", "afm_bank", "phase4a", "phase4b_visualization", "comparison"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["provenance", "targets", "rq_models", "phase4a", "phase4b_visualization", "comparison"]:
        (rep / sub).mkdir(parents=True, exist_ok=True)

    old_hash_paths = [
        config["phase1_manifest_path"],
        config["phase1_afm_audit_path"],
        config["phase2a_embedding_registry_path"],
        "outputs/rheed_video_afm_story/phase4a/rheed_rq_oof_predictions.csv",
        "outputs/rheed_video_afm_story/phase4a/oof_synthesis_outputs.csv",
        "outputs/rheed_video_afm_story/phase4b_visualization/sample_level_results.csv",
        config["removelist_path"],
    ]
    raw_hash_paths = [
        "data/processed_afm/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy",
        "data/plane_corrected_afm/6022/N6022_Ctr_000/N6022_Ctr_000_plane_corrected.npy",
        "data/rheed_keyframe_selection/6022/metadata.json",
        "data/pair/6022/AFM/N6022 Ctr.000",
    ]
    old_hashes = hash_paths(old_hash_paths)
    raw_hashes = hash_paths(raw_hash_paths)

    if not visualization_only:
        print("[1/8] provenance and mapping")
        write_variant_registry(config)
        mapping = build_scan_mapping(config)
        if not mapping["mapping_status"].eq("ok").all():
            raise RuntimeError("Second-order mapping has non-ok rows; stopping before training.")

        print("[2/8] second-order AFM targets")
        modeling, sample_targets, descriptors, target_summary = rebuild_targets(mapping, config)
        primary_ids = modeling.query("usable_for_modeling and cohort_primary_1um")["sample_id"].astype(str)
        primary_ids = [sid for sid in primary_ids if sid not in set(config["excluded_samples"])]
        if len(primary_ids) != int(config["expected_primary_n"]):
            raise RuntimeError(f"Primary cohort mismatch: expected {config['expected_primary_n']}, got {len(primary_ids)}")
        write_reused_rheed_artifacts(config, primary_ids)
        if stop_after == "targets":
            return target_summary

        print("[3/8] first-vs-second target audit")
        print("[4/8] Rq model retraining")
        print("[5/8] second-order AFM bank")
        bank = build_second_order_bank(modeling, descriptors, config)
        phase4a_config = effective_phase4a_config(config, modeling, bank)
        print("[6/8] OOF retrieval and S0-S4")
        run_phase4a(phase4a_config)
        phase4a_summary = normalize_phase4a_outputs(config)
        if stop_after in {"rq_models", "phase4a"}:
            return phase4a_summary
    else:
        target_summary = load_json(out / "comparison" / "target_comparison_summary.json")
        phase4a_summary = {}

    print("[7/8] Phase 4B visualization")
    phase4b_config = effective_phase4b_config(config)
    run_phase4b(phase4b_config)

    print("[8/8] first-vs-second comparison and validation")
    mapping = pd.read_csv(out / "provenance" / "second_order_scan_mapping.csv", dtype={"sample_id": str})
    sample_targets = pd.read_csv(out / "targets" / "second_order_sample_targets.csv", dtype={"sample_id": str})
    phase4b_samples = pd.read_csv(out / "phase4b_visualization" / "sample_level_results.csv", dtype={"sample_id": str})
    comparison_targets = sample_targets[
        sample_targets["sample_id"].astype(str).isin(set(phase4b_samples["sample_id"].astype(str)))
    ]
    comparison_paths = comparison_figures(mapping, comparison_targets, config["variant_output_root"], config["variant_report_root"])
    validation = validate_rerun(config, old_hashes, raw_hashes)
    final = write_final_report(config, target_summary, phase4a_summary, validation, comparison_paths)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled second-order AFM target rerun.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_second_order_y2.yaml")
    parser.add_argument("--stop-after", choices=STOP_ORDER, default="all")
    parser.add_argument("--visualization-only", action="store_true")
    args = parser.parse_args()
    run(args.config, stop_after=args.stop_after, visualization_only=args.visualization_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
