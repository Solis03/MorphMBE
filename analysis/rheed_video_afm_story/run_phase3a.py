from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .afm_dataset import build_afm_manifest
from .afm_descriptors import write_descriptor_definitions
from .afm_morphology_bank import build_morphology_bank
from .afm_pca_decoder import make_group_folds, run_pca_decoder
from .afm_prototypes import aggregate_scan_latents, run_prototypes
from .afm_training import run_autoencoder_cv, train_global_development_model
from .common import load_config, repo_path, sha256_file, write_csv, write_json
from .visualize_phase3a import generate_phase3a_figures


def ensure_phase3_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)
    return output_root, report_root


def provenance_check(config: dict[str, Any], report_root: Path) -> None:
    checks = {
        "phase1_afm_audit_hash_ok": sha256_file(config["phase1_afm_audit_path"]) == config["phase1_afm_audit_hash"],
        "phase1_manifest_hash_ok": sha256_file(config["phase1_manifest_path"]) == config["phase1_manifest_hash"],
        "phase2a_summary_hash_ok": sha256_file(config["phase2a_summary_path"]) == config["phase2a_summary_hash"],
        "removelist_hash_ok": sha256_file(config["removelist_path"]) == config["expected_removelist_hash"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 3A provenance check failed: {checks}")
    lines = ["# Phase 3A Provenance Check", ""]
    lines.extend([f"- {k}: {v}" for k, v in checks.items()])
    (report_root / "provenance_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_pca_ae(pca: pd.DataFrame, ae: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    if not pca.empty:
        best_pca = pca.sort_values("composite_score_median").iloc[0].to_dict()
        rows.append({"family": "PCA", **best_pca})
    if not ae.empty:
        best_ae = ae.sort_values("composite_score_median").iloc[0].to_dict()
        rows.append({"family": "Autoencoder", **best_ae})
    out = pd.DataFrame(rows)
    if len(out) == 2:
        p = out[out["family"] == "PCA"].iloc[0]["composite_score_median"]
        a = out[out["family"] == "Autoencoder"].iloc[0]["composite_score_median"]
        out["autoencoder_beats_best_pca"] = bool(a < p)
    write_csv(out, output_root / "pca_vs_autoencoder.csv")
    return out


def go_decisions(pca_vs_ae: pd.DataFrame, ae_group: pd.DataFrame, interp: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, bool]:
    ae_beats = bool(pca_vs_ae.get("autoencoder_beats_best_pca", pd.Series([False])).dropna().iloc[0]) if "autoencoder_beats_best_pca" in pca_vs_ae else False
    best_ae = ae_group.sort_values("composite_score").groupby("model_id").median(numeric_only=True).sort_values("composite_score").head(1)
    corr_ok = bool(len(best_ae) and best_ae["correlation_length_relative_error"].iloc[0] <= 0.25)
    desc_ok = bool(len(best_ae) and best_ae["normalized_psd_log_distance"].iloc[0] < ae_group["normalized_psd_log_distance"].median())
    interp_ok = bool(not interp.empty and interp.get("absolute_error_nm", pd.Series([0.0])).dropna().max() < 1e-3)
    proto_ok = bool((candidates["bootstrap_ari_median"].fillna(-1) > 0.2).any() and (candidates["min_cluster_size"] > 1).any())
    return {"Go-A": ae_beats and corr_ok, "Go-B": desc_ok and corr_ok, "Go-C": interp_ok, "Go-D": proto_ok}


def write_report(summary: dict[str, Any], config: dict[str, Any]) -> None:
    report_root = repo_path(config["report_root"])
    lines = [
        "# Phase 3A Report",
        "",
        "RD-AFM-AE is an AFM-side representation/decoder experiment. No RHEED-to-latent model is trained and no RHEED-predicted AFM is generated.",
        "",
        f"- Valid 1 x 1 um AFM scans: {summary['valid_1um_scan_count']}",
        f"- Growth groups: {summary['growth_group_count']}",
        f"- Paired primary scans: {summary['paired_primary_scan_count']}",
        f"- Paired exploratory scans: {summary['paired_exploratory_scan_count']}",
        f"- Unpaired support scans: {summary['unpaired_support_scan_count']}",
        f"- Resolution distribution: {summary['resolution_distribution']}",
        f"- Rq distribution nm: {summary['rq_distribution_nm']}",
        f"- PSD high fraction distribution: {summary['psd_high_fraction_distribution']}",
        f"- Correlation length distribution nm: {summary['correlation_length_distribution_nm']}",
        f"- Best PCA: {summary['best_pca']}",
        f"- Best autoencoder: {summary['best_autoencoder']}",
        f"- Autoencoder beats PCA: {summary['autoencoder_beats_pca']}",
        f"- Global development metrics: {summary['global_development_metrics']}",
        f"- Descriptor preservation: {summary['descriptor_preservation']}",
        f"- Interpolation audit: {summary['interpolation_audit']}",
        f"- Amplitude sweep verification: {summary['amplitude_sweep_verification']}",
        f"- Prototype candidates: {summary['prototype_candidate_summary']}",
        f"- Within-sample morphology variability: {summary['within_sample_variability']}",
        f"- Expert review package: {summary['expert_review_package']}",
        f"- Morphology bank: {summary['morphology_bank_path']}",
        f"- Blind review package: {summary['blind_review_package']}",
        f"- Go decisions: {summary['go_decisions']}",
        f"- Next-stage recommendation: {summary['next_stage_recommendation']}",
        "",
        "Notes:",
        "- Group-held-out metrics use growth_run_id splits; scans from the same growth group do not cross folds.",
        "- Global development metrics are transductive visualization metrics and are not strict test performance.",
        "- Reconstructed physical maps use true q only for AFM-side decoder evaluation, so Rq consistency is architectural, not predictive.",
        "- Original AFM/RHEED/metadata, Phase 1 outputs, and Phase 2A outputs were read only.",
    ]
    (report_root / "phase3a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_root, report_root = ensure_phase3_dirs(config)
    provenance_check(config, report_root)
    write_descriptor_definitions(report_root)
    manifest, decoder_audit, descriptors = build_afm_manifest(config)
    split = make_group_folds(manifest, config)
    stage_a_res = int(config["input_resolutions"]["stage_a"])
    stage_b_res = int(config["input_resolutions"]["stage_b"])
    pca_scan, pca_metrics, pca_registry = run_pca_decoder(manifest, config, stage_a_res, split)
    ae_scan_a, ae_oof_a, ae_group_a, ae_registry_a = run_autoencoder_cv(manifest, config, split, stage_a_res)
    best_a = ae_oof_a.sort_values("composite_score_median").head(1)
    stage_b_scans, stage_b_oofs, stage_b_groups, stage_b_regs = [], [], [], []
    for _, row in best_a.iterrows():
        scans, oofs, groups, regs = run_autoencoder_cv(
            manifest,
            config,
            split,
            stage_b_res,
            latent_dims=[int(row["latent_dim"])],
            loss_presets=[str(row["loss_preset"])],
            seeds=[int(config["seeds"][0])],
        )
        stage_b_scans.append(scans)
        stage_b_oofs.append(oofs)
        stage_b_groups.append(groups)
        stage_b_regs.append(regs)
    ae_scan = pd.concat([ae_scan_a] + stage_b_scans, ignore_index=True)
    ae_oof = pd.concat([ae_oof_a] + stage_b_oofs, ignore_index=True)
    ae_group = pd.concat([ae_group_a] + stage_b_groups, ignore_index=True)
    ae_registry = pd.concat([ae_registry_a] + stage_b_regs, ignore_index=True)
    write_csv(ae_scan, output_root / "autoencoder_scan_metrics.csv")
    write_csv(ae_oof, output_root / "autoencoder_oof_metrics.csv")
    write_csv(ae_group, output_root / "autoencoder_group_metrics.csv")
    write_csv(ae_registry, output_root / "decoder_model_registry.csv")
    pca_vs_ae = compare_pca_ae(pca_metrics, ae_oof, output_root)
    best_ae = ae_oof.sort_values("composite_score_median").iloc[0]
    global_model, global_metrics, global_registry, global_recons, global_latents = train_global_development_model(
        manifest,
        config,
        stage_b_res,
        int(best_ae["latent_dim"]),
        str(best_ae["loss_preset"]),
        int(config["seeds"][0]),
    )
    scan_latents, sample_latents = aggregate_scan_latents(manifest, global_latents, output_root)
    candidates, assignments, stability = run_prototypes(manifest, sample_latents, config)
    bank = build_morphology_bank(manifest, assignments, global_latents, global_recons, config, stage_b_res)
    interp = generate_phase3a_figures(manifest, pca_metrics, ae_scan, ae_oof, sample_latents, assignments, global_model, global_recons, config, stage_b_res)
    go = go_decisions(pca_vs_ae, ae_group, interp, candidates)
    best_pca = pca_metrics.sort_values("composite_score_median").iloc[0].to_dict()
    best_auto = ae_oof.sort_values("composite_score_median").iloc[0].to_dict()
    auto_beats = bool(best_auto["composite_score_median"] < best_pca["composite_score_median"])
    recommendation = "prototype retrieval and PCA decoder preferred for descriptor-critical next-stage work; keep AE as an AFM-side low-frequency development decoder and pair it with retrieved residuals only as a cautious follow-up"
    if auto_beats and go["Go-A"] and go["Go-B"]:
        recommendation = "convolutional decoder plus retrieval residual is supported for next-stage AFM-side synthesis"
    summary = {
        "valid_1um_scan_count": int(len(manifest)),
        "growth_group_count": int(manifest["growth_run_id"].nunique()),
        "paired_primary_scan_count": int(manifest["paired_primary"].sum()),
        "paired_exploratory_scan_count": int(manifest["paired_exploratory"].sum()),
        "unpaired_support_scan_count": int(manifest["unpaired_support"].sum()),
        "resolution_distribution": {str(k): int(v) for k, v in manifest.groupby(["resolution_x", "resolution_y"]).size().to_dict().items()},
        "rq_distribution_nm": manifest["rq_nm"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "psd_high_fraction_distribution": manifest["unit_psd_high_fraction"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "correlation_length_distribution_nm": manifest["unit_autocorr_length_nm"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "pca_component_metrics": pca_metrics.to_dict("records"),
        "autoencoder_metrics": ae_oof.to_dict("records"),
        "group_held_out_metrics_path": "outputs/rheed_video_afm_story/phase3a/autoencoder_group_metrics.csv",
        "global_development_metrics": global_metrics[["unit_l1", "ssim", "normalized_psd_log_distance", "correlation_length_relative_error", "composite_score"]].median(numeric_only=True).to_dict(),
        "best_pca": best_pca,
        "best_autoencoder": best_auto,
        "autoencoder_beats_pca": auto_beats,
        "descriptor_preservation": ae_group[["normalized_psd_log_distance", "correlation_length_relative_error", "height_quantile_error", "anisotropy_error"]].median(numeric_only=True).to_dict(),
        "interpolation_audit": {"rows": int(len(interp)), "max_rq_error_nm": float(interp.get("absolute_error_nm", pd.Series([0.0])).dropna().max() if "absolute_error_nm" in interp else 0.0)},
        "amplitude_sweep_verification": interp[interp.get("target_rq_nm", pd.Series(dtype=float)).notna()][["target_rq_nm", "measured_rq_nm", "absolute_error_nm"]].to_dict("records") if "target_rq_nm" in interp else [],
        "prototype_candidate_summary": candidates.to_dict("records"),
        "within_sample_variability": sample_latents["within_sample_latent_distance_median"].describe().to_dict(),
        "expert_review_package": "reports/rheed_video_afm_story/phase3a/prototype_review/index.html",
        "morphology_bank_path": "outputs/rheed_video_afm_story/phase3a/afm_morphology_bank.parquet",
        "morphology_bank_schema": bank.columns.tolist(),
        "blind_review_package": "reports/rheed_video_afm_story/phase3a/blind_review/review_grid.pdf",
        "go_decisions": go,
        "next_stage_recommendation": recommendation,
        "no_rheed_to_latent_mapping_trained": True,
        "no_rheed_predicted_afm_generated": True,
    }
    write_json(summary, output_root / "phase3a_summary.json")
    write_report(summary, config)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3A AFM-side morphology decoder.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase3a.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
