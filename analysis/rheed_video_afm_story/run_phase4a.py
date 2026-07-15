from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .afm_patch_synthesis import synthesize_all
from .common import load_config, repo_path, sha256_file, write_json
from .monotonic_rq import high_confidence_subset, run_rq_models
from .retrieval_ceiling import oracle_retrieval, same_growth_similarity
from .rheed_conditioned_retrieval import run_retrieval
from .rheed_expert_review import build_expert_review
from .rheed_physics_features import extract_rheed_physics_features
from .synthesis_evaluation import evaluate_synthesis, visual_plausibility_proxy
from .visualize_phase4a import blind_reviews, generate_figures


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    return out, rep


def provenance(config: dict[str, Any], report_root: Path) -> dict[str, bool]:
    checks = {
        "phase1_manifest_hash_ok": sha256_file(config["phase1_manifest_path"]) == config["phase1_manifest_hash"],
        "phase1_afm_audit_hash_ok": sha256_file(config["phase1_afm_audit_path"]) == config["phase1_afm_audit_hash"],
        "phase2a_summary_hash_ok": sha256_file(config["phase2a_summary_path"]) == config["phase2a_summary_hash"],
        "phase2a_embedding_registry_hash_ok": sha256_file(config["phase2a_embedding_registry_path"]) == config["phase2a_embedding_registry_hash"],
        "phase3a_morphology_bank_hash_ok": sha256_file(config["phase3a_morphology_bank_path"]) == config["phase3a_morphology_bank_hash"],
        "phase3a_decoder_manifest_hash_ok": sha256_file(config["phase3a_decoder_manifest_path"]) == config["phase3a_decoder_manifest_hash"],
        "phase3a_summary_hash_ok": sha256_file(config["phase3a_summary_path"]) == config["phase3a_summary_hash"],
        "removelist_hash_ok": sha256_file(config["removelist_path"]) == config["expected_removelist_hash"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 4A provenance check failed: {checks}")
    (report_root / "provenance_check.md").write_text("# Phase 4A Provenance Check\n\n" + "\n".join([f"- {k}: {v}" for k, v in checks.items()]) + "\n", encoding="utf-8")
    return checks


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(repo_path(config["phase1_manifest_path"]))
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["growth_run_id"] = manifest["growth_run_id"].astype(str)
    excluded = set(config["excluded_samples"])
    manifest = manifest.query("usable_for_modeling and cohort_primary_1um").copy()
    manifest = manifest[~manifest["sample_id"].isin(excluded)].reset_index(drop=True)
    bank = pd.read_parquet(repo_path(config["phase3a_morphology_bank_path"]))
    bank["sample_id"] = bank["sample_id"].astype(str)
    bank["growth_run_id"] = bank["growth_run_id"].astype(str)
    bank = bank[~bank["sample_id"].isin(excluded)].reset_index(drop=True)
    decoder = pd.read_parquet(repo_path(config["phase3a_decoder_manifest_path"]))
    decoder["sample_id"] = decoder["sample_id"].astype(str)
    decoder = decoder[~decoder["sample_id"].isin(excluded)].reset_index(drop=True)
    return manifest, bank, decoder


def go_decisions(summary: dict[str, Any]) -> dict[str, bool]:
    r4 = summary["rq_metrics_by_model"].get("R4_auto_iso_dino_residual", {})
    high = summary["high_confidence_metrics"]
    s = summary["synthesis_metrics_by_method"]
    vis = summary["visual_plausibility_proxy"]
    s4 = s.get("S4_calibrated_patch_synthesis", {})
    s0 = s.get("S0_unconditional_median_prototype", {})
    s1 = s.get("S1_top1_real_exemplar_retrieval", {})
    return {
        "Go-RQ-A": bool(r4.get("MAE", 999) < 1.499 or r4.get("Spearman", 0) >= 0.45 or r4.get("pairwise_concordance", 0) >= 0.65),
        "Go-RQ-B": bool(high.get("coverage", 0) >= 0.35 and (high.get("Spearman", 0) >= 0.60 or high.get("low_high_balanced_accuracy", 0) >= 0.75)),
        "Go-VIS-A": bool(vis.get("physically_plausible_rate_proxy", 0) >= 0.9 and vis.get("obvious_artifact_rate_proxy", 1) <= 0.1 and vis.get("obvious_seam_rate_proxy", 1) <= 0.1),
        "Go-VIS-B": bool(s4.get("normalized_psd_log_distance", 999) < s0.get("normalized_psd_log_distance", -999) and s4.get("normalized_psd_log_distance", 999) <= 1.25 * s1.get("normalized_psd_log_distance", 0)),
        "Go-RET": bool(s1.get("normalized_psd_log_distance", 999) < s0.get("normalized_psd_log_distance", -999) or s.get("S2_topk_real_scan_medoid", {}).get("normalized_psd_log_distance", 999) < s0.get("normalized_psd_log_distance", -999)),
    }


def write_report(summary: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# Phase 4A Report",
        "",
        "RHEED-RAMS is evaluated as RHEED-conditioned representative AFM retrieval/synthesis. Outputs are representative morphology, not exact AFM reconstruction.",
        "",
        f"- Same-growth real-to-real ceiling: {summary['same_growth_ceiling']}",
        f"- 90% pixel similarity supported by data: {summary['same_growth_ceiling']['supports_90_percent_pixel_similarity']}",
        f"- Automatic index vs Rq Spearman: {summary['automatic_index_vs_rq_spearman']}",
        f"- Expert review package: {summary['expert_review_package']}",
        f"- Expert branch status: {summary['expert_branch_status']}",
        f"- Rq model metrics: {summary['rq_metrics_by_model']}",
        f"- High-confidence metrics: {summary['high_confidence_metrics']}",
        f"- Abstained sample count: {summary['high_confidence_metrics'].get('abstained_count')}",
        f"- Retrieval final weights: {summary['retrieval_final_weights']}",
        f"- Top-k setting: {summary['retrieval_final_weights'].get('top_k')}",
        f"- S0-S4 descriptor metrics: {summary['synthesis_metrics_by_method']}",
        f"- S1 top-1 retrieval result: {summary['synthesis_metrics_by_method'].get('S1_top1_real_exemplar_retrieval')}",
        f"- S3/S4 patch synthesis result: S3={summary['synthesis_metrics_by_method'].get('S3_patch_synthesis')}; S4={summary['synthesis_metrics_by_method'].get('S4_calibrated_patch_synthesis')}",
        f"- Identity audit: {summary['identity_audit_summary']}",
        f"- Blind review package: {summary['blind_review_package']}",
        f"- Oracle ceiling vs deployable: {summary['oracle_vs_deployable']}",
        f"- Go decisions: {summary['go_decisions']}",
        f"- Final recommendation: {summary['final_recommendation']}",
        "",
        "Can claim: strict group-held-out Rq/retrieval evaluation; representative AFM retrieval/synthesis with predicted-Rq scaling; oracle ceilings separated from deployable results.",
        "Cannot claim: exact AFM reconstruction, RHEED-predicted local AFM, or neural generative AFM superiority.",
    ]
    (repo_path(config["report_root"]) / "phase4a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    out, rep = ensure_dirs(config)
    checks = provenance(config, rep)
    manifest, bank, decoder = load_inputs(config)
    same, same_summary, _ = same_growth_similarity(decoder, config)
    oracle, oracle_summary = oracle_retrieval(manifest, bank, config)
    features = extract_rheed_physics_features(manifest, config)
    expert_status = build_expert_review(manifest, config)
    rq_pred, rq_metrics, idx, rq_audit = run_rq_models(manifest, features, config)
    support, high_metrics = high_confidence_subset(manifest, rq_pred, idx, features, config)
    retrieval, retrieval_audit, final_weights = run_retrieval(manifest, bank, rq_pred, idx, config)
    synth_outputs, provenance_df, identity = synthesize_all(manifest, bank, retrieval, config)
    synth_metrics, synth_summary = evaluate_synthesis(manifest, bank, synth_outputs, config)
    visual_proxy = visual_plausibility_proxy(identity, synth_metrics, config)
    generate_figures(manifest, rq_pred, rq_metrics, features, idx, same, synth_summary, synth_outputs, config)
    blind_reviews(manifest, bank, synth_outputs, config)
    idx_s = idx.merge(manifest[["sample_id", "primary_rq_nm_median"]], on="sample_id")
    rq_metric_map = rq_metrics.set_index("model_id").to_dict("index")
    high_row = high_metrics.iloc[0].to_dict() if len(high_metrics) else {}
    synth_map = synth_summary.set_index("method").to_dict("index")
    visual_row = visual_proxy.iloc[0].to_dict()
    identity_summary = {
        "max_largest_single_source_contribution": float(identity["largest_single_source_contribution"].max()) if len(identity) else np.nan,
        "any_exact_pixel_equality": bool(identity["exact_pixel_equality"].astype(bool).any()) if len(identity) else False,
        "heldout_source_contribution_max": float(identity["heldout_sample_source_contribution"].max()) if len(identity) else 0.0,
    }
    summary = {
        "provenance_checks": checks,
        "same_growth_ceiling": same_summary,
        "automatic_index_vs_rq_spearman": float(idx_s["automatic_spot_streak_index"].corr(idx_s["primary_rq_nm_median"], method="spearman")),
        "expert_review_package": "reports/rheed_video_afm_story/phase4a/rheed_expert_review/index.html",
        "expert_branch_status": expert_status,
        "expert_branch_ran": bool(expert_status["expert_branch_available"]),
        "rq_metrics_by_model": rq_metric_map,
        "high_confidence_metrics": high_row,
        "retrieval_final_weights": final_weights,
        "synthesis_metrics_by_method": synth_map,
        "visual_plausibility_proxy": visual_row,
        "identity_audit_summary": identity_summary,
        "blind_review_package": "reports/rheed_video_afm_story/phase4a/blind_review",
        "oracle_vs_deployable": {"oracle_median_distance": oracle_summary.to_dict("records"), "deployable_s4_psd_distance": synth_map.get("S4_calibrated_patch_synthesis", {}).get("normalized_psd_log_distance")},
        "no_afm_autoencoder_trained": True,
        "no_gan_vae_diffusion_trained": True,
    }
    summary["go_decisions"] = go_decisions(summary)
    if summary["go_decisions"]["Go-VIS-A"] and summary["go_decisions"]["Go-VIS-B"]:
        rec = "RHEED-conditioned representative AFM patch synthesis (S4) with retrieval audit"
    elif summary["go_decisions"]["Go-RET"]:
        rec = "RHEED-conditioned representative AFM retrieval using S1/S2 real training exemplars"
    else:
        rec = "high-confidence Rq screening with abstention; request AFM for morphology"
    summary["final_recommendation"] = rec
    summary["recommended_method_name_for_paper"] = "RHEED-conditioned representative AFM retrieval"
    write_json(summary, out / "phase4a_summary.json")
    write_report(summary, config)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4A RHEED-conditioned representative AFM retrieval/synthesis.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase4a.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
