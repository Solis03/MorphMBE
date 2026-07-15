from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import ensure_dirs, load_config, repo_path, save_parquet, sha256_file, write_csv, write_json
from .confidence_support import bootstrap_ci, conformal_intervals, coverage_performance, permutation_summary, retrieval_audit, support_scores
from .embedding_regression import prediction_metrics, run_metadata_controls, run_regression
from .extreme_screening import regime_labels, run_extreme_screening
from .pairwise_ranking import run_pairwise_ranking
from .phase2_clip_variants import build_clip_variants
from .pretrained_embeddings import extract_embeddings
from .visualize_phase2a import phase2_figures


def phase2_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return out, rep


def load_primary(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(repo_path(config["phase1_manifest_path"]), dtype={"sample_id": str})
    quality = pd.read_csv(repo_path(config["phase1_quality_path"]), dtype={"sample_id": str})
    primary = manifest[(manifest["usable_for_modeling"]) & (manifest["cohort_primary_1um"])].copy()
    primary = primary.sort_values("sample_id").reset_index(drop=True)
    return primary, quality


def provenance_check(config: dict[str, Any], manifest: pd.DataFrame, report_root: Path) -> None:
    errors = []
    actual_hash = sha256_file(config["phase1_manifest_path"])
    rem_hash = sha256_file(config["removelist_path"])
    if actual_hash != config["phase1_manifest_hash"]:
        errors.append(f"Phase 1 manifest hash changed: {actual_hash}")
    if rem_hash != config["expected_removelist_hash"]:
        errors.append(f"removelist hash changed: {rem_hash}")
    excluded = set(config["excluded_samples"])
    if set(manifest["sample_id"]) & excluded:
        errors.append(f"excluded samples present in primary modeling cohort: {sorted(set(manifest['sample_id']) & excluded)}")
    if len(manifest) != int(config["expected_primary_n"]):
        errors.append(f"primary cohort expected {config['expected_primary_n']}, found {len(manifest)}")
    if manifest["growth_run_id"].nunique() != len(manifest):
        errors.append("primary samples are not one growth group each")
    lines = [
        "# Phase 2A Provenance Check",
        "",
        f"- Phase 1 manifest hash: `{actual_hash}`",
        f"- Removelist hash: `{rem_hash}`",
        f"- Excluded samples checked: {sorted(excluded)}",
        f"- Primary cohort N: {len(manifest)}",
        f"- Unique growth groups: {manifest['growth_run_id'].nunique()}",
        f"- Original selected clip lengths: {sorted(manifest['actual_clip_frame_count'].unique().tolist())}",
        f"- AFM target min/median/max: {manifest['primary_rq_nm_median'].min():.6g} / {manifest['primary_rq_nm_median'].median():.6g} / {manifest['primary_rq_nm_median'].max():.6g}",
        f"- Errors: {errors if errors else 'none'}",
    ]
    (report_root / "provenance_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))


def frozen_thresholds(manifest: pd.DataFrame, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = output_root / "frozen_regime_thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    rq = manifest["primary_rq_nm_median"].to_numpy(float)
    payload = {
        "q33": float(np.quantile(rq, config["extreme_class_thresholds"]["q_low"])),
        "q67": float(np.quantile(rq, config["extreme_class_thresholds"]["q_high"])),
        "definition": "low <= q33; middle q33 < Rq < q67; high >= q67",
        "sample_ids": manifest["sample_id"].astype(str).tolist(),
        "created_once": True,
    }
    write_json(payload, path)
    return payload


def summarize_timing(timing: pd.DataFrame) -> dict[str, Any]:
    return {
        "fps_min": float(timing["fps"].min()),
        "fps_max": float(timing["fps"].max()),
        "duration_min_s": float(timing["original_clip_duration_seconds"].min()),
        "duration_max_s": float(timing["original_clip_duration_seconds"].max()),
        "fps_missing_count": int(timing["fps"].isna().sum()),
        "keyframe_position_counts": timing["keyframe_position"].value_counts().to_dict(),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 2A Report",
        "",
        f"- Primary cohort N: {summary['primary_n']}",
        f"- FPS range: {summary['timing']['fps_min']:.3g}-{summary['timing']['fps_max']:.3g}",
        f"- 16-frame duration range: {summary['timing']['duration_min_s']:.3g}-{summary['timing']['duration_max_s']:.3g} s",
        f"- All-primary variants: {', '.join(summary['all_primary_variants'])}",
        f"- Loaded encoders: {summary['loaded_encoders']}",
        f"- Frozen thresholds: q33={summary['thresholds']['q33']:.4g}, q67={summary['thresholds']['q67']:.4g}",
        f"- Best regression: {summary['best_regression']}",
        f"- Best ranking: {summary['best_ranking']}",
        f"- Best extreme: {summary['best_extreme']}",
        f"- Best metadata/control: {summary['best_metadata_control']}",
        f"- Support distribution: {summary['support_distribution']}",
        f"- Prediction interval coverage: {summary['interval_coverage']}",
        f"- Go decisions: {summary['go_decisions']}",
        "",
        "## Notes",
        "",
        "- No AFM decoder was trained and no AFM was generated.",
        "- High-support labels use target-blind domain/support quantities only.",
        "- UMAP/PCA visualization coordinates are descriptive only and are not predictive features.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_root, report_root = phase2_dirs(config)
    manifest, quality = load_primary(config)
    provenance_check(config, manifest, report_root)
    thresholds = frozen_thresholds(manifest, output_root, config)
    variant_manifest, timing = build_clip_variants(manifest, config, output_root, report_root)
    write_csv(timing, output_root / "clip_timing_audit.csv")
    write_csv(variant_manifest, output_root / "clip_variants_manifest.csv")
    save_parquet(variant_manifest, output_root / "clip_variants_manifest.parquet")
    all_primary_variants = sorted(variant_manifest.groupby("clip_variant")["available"].sum().loc[lambda s: s == len(manifest)].index.tolist())
    embedding_registry, model_registry = extract_embeddings(manifest, variant_manifest, config, output_root)
    write_csv(embedding_registry, output_root / "embedding_registry.csv")
    write_csv(model_registry, output_root / "model_registry.csv")
    if embedding_registry.empty:
        raise RuntimeError("No pretrained embeddings were extracted.")
    reg_pred, reg_metrics, reg_sel = run_regression(manifest, embedding_registry, config)
    write_csv(reg_pred, output_root / "regression_oof_predictions.csv")
    write_csv(reg_metrics, output_root / "regression_metrics.csv")
    write_csv(reg_sel, output_root / "regression_inner_selection.csv")
    rank_pred, rank_audit, rank_metrics = run_pairwise_ranking(manifest, embedding_registry, config)
    write_csv(rank_pred, output_root / "ranking_oof_predictions.csv")
    write_csv(rank_audit, output_root / "ranking_pair_audit.csv")
    write_csv(rank_metrics, output_root / "ranking_metrics.csv")
    extreme_pred, extreme_metrics = run_extreme_screening(manifest, embedding_registry, config, thresholds["q33"], thresholds["q67"])
    write_csv(extreme_pred, output_root / "extreme_regime_predictions.csv")
    write_csv(extreme_metrics, output_root / "extreme_regime_metrics.csv")
    best_embedding = reg_metrics.iloc[0]["embedding_id"]
    best_embedding_row = embedding_registry[embedding_registry["embedding_id"] == best_embedding].iloc[0]
    metadata_pred, metadata_metrics = run_metadata_controls(manifest, quality, best_embedding_row, config)
    write_csv(metadata_pred, output_root / "metadata_control_predictions.csv")
    write_csv(metadata_metrics, output_root / "metadata_control_metrics.csv")
    conf = support_scores(manifest, embedding_registry, reg_pred, rank_pred, quality, best_embedding, config)
    best_key = (reg_metrics.iloc[0]["embedding_id"], reg_metrics.iloc[0]["head"])
    intervals = conformal_intervals(reg_pred, manifest, best_key)
    conf = conf.merge(intervals[["sample_id", "pi90_low_nm", "pi90_high_nm"]], on="sample_id", how="left")
    conf["prediction_interval_low"] = conf["pi90_low_nm"]
    conf["prediction_interval_high"] = conf["pi90_high_nm"]
    conf = conf.drop(columns=["pi90_low_nm", "pi90_high_nm"])
    coverage = coverage_performance(conf, reg_pred, best_key)
    retrieval = retrieval_audit(manifest, embedding_registry, reg_metrics["embedding_id"].drop_duplicates().head(3).tolist(), output_root)
    write_csv(conf, output_root / "oof_confidence_support.csv")
    write_csv(intervals, output_root / "cross_conformal_intervals.csv")
    write_csv(coverage, output_root / "coverage_performance.csv")
    write_csv(retrieval, output_root / "embedding_neighbor_audit.csv")
    rng = np.random.default_rng(config["random_seed"])
    best_pred = reg_pred[(reg_pred["embedding_id"] == best_key[0]) & (reg_pred["head"] == best_key[1])]
    y_true = best_pred["true_rq_nm"].to_numpy(float)
    y_pred = best_pred["pred_rq_nm"].to_numpy(float)
    median_mae = float(pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase1/baseline_metrics.csv")).query("model_name == 'B0_training_fold_median'")["MAE_nm"].iloc[0])
    stats = {}
    stats.update(bootstrap_ci(y_true, y_pred, rng, int(config["bootstrap_count"])))
    stats.update(permutation_summary(y_true, y_pred, median_mae, rng, int(config["permutation_count"])))
    write_json(stats, output_root / "statistical_stability.json")
    # Sensitivity summaries from already-frozen OOF outputs.
    largest_stage = manifest["video_stage"].value_counts().idxmax()
    stage_ids = set(manifest[manifest["video_stage"] == largest_stage]["sample_id"].astype(str))
    stage_metrics = prediction_metrics(best_pred[best_pred["sample_id"].astype(str).isin(stage_ids)], ["embedding_id", "head"]) if len(stage_ids) >= 12 else pd.DataFrame()
    stable = manifest[(manifest["primary_afm_scan_count"] >= 3) & ((manifest["primary_rq_nm_mad"] / manifest["primary_rq_nm_median"]) <= 0.25)]
    stable_ids = set(stable["sample_id"].astype(str))
    stable_metrics = prediction_metrics(best_pred[best_pred["sample_id"].astype(str).isin(stable_ids)], ["embedding_id", "head"]) if len(stable_ids) >= 12 else pd.DataFrame()
    write_csv(stage_metrics, output_root / "largest_stage_subset_metrics.csv")
    write_csv(stable_metrics, output_root / "stable_target_subset_metrics.csv")
    phase2_figures(manifest, timing, variant_manifest, embedding_registry, reg_pred, reg_metrics, rank_pred, rank_metrics, extreme_pred, extreme_metrics, metadata_metrics, conf, intervals, coverage, retrieval, report_root)
    best_extreme = extreme_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0].to_dict() if not extreme_metrics.empty else {}
    go = {
        "Go-A": bool(reg_metrics.iloc[0]["MAE_nm"] <= 0.9 * median_mae and reg_metrics.iloc[0]["Spearman"] > 0.35),
        "Go-B": bool(rank_metrics.iloc[0]["pairwise_concordance"] > 0.65 or rank_metrics.iloc[0]["Spearman"] > 0.35),
        "Go-C": bool(best_extreme and (best_extreme.get("AUROC", 0) >= 0.75 or best_extreme.get("balanced_accuracy", 0) >= 0.70)),
        "Go-D": bool(coverage[coverage["coverage_fraction"] <= 0.5]["MAE_nm"].min() < reg_metrics.iloc[0]["MAE_nm"] if len(coverage) else False),
    }
    summary = {
        "primary_n": int(len(manifest)),
        "timing": summarize_timing(timing),
        "all_primary_variants": all_primary_variants,
        "loaded_encoders": model_registry[model_registry["status"] == "loaded"]["encoder"].drop_duplicates().tolist(),
        "thresholds": thresholds,
        "best_regression": reg_metrics.iloc[0].to_dict(),
        "best_ranking": rank_metrics.iloc[0].to_dict() if not rank_metrics.empty else {},
        "best_extreme": best_extreme,
        "best_metadata_control": metadata_metrics.iloc[0].to_dict() if not metadata_metrics.empty else {},
        "support_distribution": conf["support_level"].value_counts().to_dict(),
        "interval_coverage": {
            "coverage_80": float(intervals["covered_80"].mean()),
            "coverage_90": float(intervals["covered_90"].mean()),
            "mean_90_width": float((intervals["pi90_high_nm"] - intervals["pi90_low_nm"]).mean()),
            "median_90_width": float((intervals["pi90_high_nm"] - intervals["pi90_low_nm"]).median()),
        },
        "stage_subset": {"stage": largest_stage, "N": len(stage_ids), "metrics": stage_metrics.to_dict("records")},
        "stable_target_subset": {"N": len(stable_ids), "metrics": stable_metrics.to_dict("records")},
        "statistical_stability": stats,
        "go_decisions": go,
        "recommended_finalists": reg_metrics.head(3)["embedding_id"].tolist(),
    }
    write_json(summary, output_root / "phase2a_summary.json")
    write_report(report_root / "phase2a_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase2a.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
