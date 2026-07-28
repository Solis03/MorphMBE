from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_full_cohort_loo.run import (
    _target_series,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    crossfit_target,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    group_metric_table,
    scan_metric_table,
)
from analysis.rheed_to_afm_functional_morphology.run import _physics_table
from analysis.rheed_to_afm_generation.data import _load_embedding
from analysis.rheed_to_afm_generation.run import _load_tables
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from .prediction import (
    BASELINE,
    DENSITY_WEIGHTED,
    FINAL_TARGET_SPECIFIC,
    MULTIVIEW_20,
    NESTED_SELECTOR,
    CandidateConfig,
    crossfit_robust_candidates,
    training_weight_audit,
)
from .support import leave_one_out_support_audit
from .visualization import (
    plot_confidence_and_coverage,
    plot_exclusion_sensitivity,
    plot_method_summary,
    plot_ood_audit,
    plot_predictions,
    plot_training_weights,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _embedding_frame(
    registry: pd.DataFrame, embedding_id: str
) -> pd.DataFrame:
    row = registry.loc[registry["embedding_id"] == str(embedding_id)]
    if len(row) != 1:
        raise RuntimeError(
            f"embedding id is not uniquely available: {embedding_id}"
        )
    sample_ids, embeddings = _load_embedding(str(row.iloc[0]["path"]))
    frame = pd.DataFrame(
        np.asarray(embeddings, dtype=float),
        index=list(map(str, sample_ids)),
    )
    frame.index.name = "growth_run_id"
    return frame


def _metrics(
    predictions: pd.DataFrame, *, target: str, method: str
) -> dict[str, Any]:
    true = predictions["true_target"].to_numpy(float)
    predicted = predictions["predicted_target"].to_numpy(float)
    error = np.abs(predicted - true)
    pearson = pearsonr(true, predicted)
    spearman = spearmanr(true, predicted)
    confidence_error = spearmanr(
        predictions["confidence"].to_numpy(float), error
    )
    return {
        "target": target,
        "method": method,
        "growth_group_count": len(predictions),
        "mae": float(np.mean(error)),
        "median_absolute_error": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "confidence_vs_absolute_error_spearman": float(
            confidence_error.statistic
        ),
        "confidence_vs_absolute_error_p": float(
            confidence_error.pvalue
        ),
        "interval_coverage": float(
            predictions["interval_covered"].astype(float).mean()
        ),
        "mean_interval_width": float(
            2.0 * predictions["interval_radius"].mean()
        ),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
        "truth_min": float(np.min(true)),
        "truth_max": float(np.max(true)),
    }


def _baseline_metrics(
    predictions: pd.DataFrame,
    *,
    target: str,
    excluded: list[str],
) -> dict[str, Any]:
    true = predictions["true_target"].to_numpy(float)
    predicted = predictions["predicted_target"].to_numpy(float)
    error = np.abs(predicted - true)
    return {
        "target": target,
        "method": BASELINE,
        "excluded_growth_count": len(excluded),
        "excluded_growth_run_ids": ";".join(excluded),
        "growth_group_count": len(predictions),
        "mae": float(np.mean(error)),
        "median_absolute_error": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_r": float(pearsonr(true, predicted).statistic),
        "spearman_rho": float(spearmanr(true, predicted).statistic),
        "selection_used_afm_target": False,
    }


def _risk_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (target, method), rows in predictions.groupby(
        ["target", "method"]
    ):
        ordered = rows.sort_values(
            ["confidence", "growth_run_id"], ascending=[False, True]
        )
        for coverage in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            count = max(2, int(np.ceil(len(rows) * coverage)))
            kept = ordered.iloc[:count]
            true = kept["true_target"].to_numpy(float)
            predicted = kept["predicted_target"].to_numpy(float)
            error = np.abs(predicted - true)
            records.append(
                {
                    "target": target,
                    "method": method,
                    "coverage": count / len(rows),
                    "retained_growth_group_count": count,
                    "mae": float(np.mean(error)),
                    "median_absolute_error": float(np.median(error)),
                    "pearson_r": float(
                        pearsonr(true, predicted).statistic
                    )
                    if count >= 3
                    else np.nan,
                    "retained_growth_run_ids": ";".join(
                        kept["growth_run_id"].astype(str)
                    ),
                }
            )
    return pd.DataFrame(records)


def run(config: dict[str, Any]) -> None:
    started = time.time()
    report = repo_path(config["report_root"])
    figures = report / "figures"
    report.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(config)
    descriptors, source_split = prepare_full_cohort(tables, config)
    scan_metrics = scan_metric_table(
        descriptors,
        scan_size_nm=float(config["scan_size_nm"]),
        analysis_scale_nm=float(config["analysis_scale_nm"]),
    )
    group_metrics = group_metric_table(scan_metrics)
    log_rq, log_fsmi = _target_series(descriptors, group_metrics)
    physics = _physics_table(tables["physics"])
    groups = list(map(str, log_rq.index))
    embeddings = _embedding_frame(
        tables["registry"], str(config["temporal_embedding_id"])
    ).loc[groups]
    audit = leave_one_out_support_audit(physics, groups)
    write_csv(audit, report / "rheed_only_ood_audit.csv")
    write_csv(
        pd.DataFrame(
            {
                "growth_run_id": groups,
                "source_split_provenance": [
                    source_split[group] for group in groups
                ],
                "full23_outer_loo_role": "held_once",
                "canonical_removelist_changed": False,
            }
        ),
        report / "cohort_manifest.csv",
    )

    exclusion_predictions = []
    exclusion_metrics = []
    for count in (0, 2, 3, 4):
        excluded = (
            audit.head(count)["growth_run_id"].astype(str).tolist()
            if count
            else []
        )
        for target, series in (("Rq_nm", log_rq), ("FSMI_nm", log_fsmi)):
            use = series.drop(index=excluded)
            prediction, _ = crossfit_target(
                physics=physics,
                log_target=use,
                alpha=float(config["baseline_ridge_alpha"]),
                morphology_weight=float(
                    config["baseline_morphology_weight"]
                ),
                confidence_alpha=float(config["confidence_alpha"]),
            )
            prediction.insert(0, "target", target)
            prediction.insert(1, "excluded_growth_count", count)
            prediction.insert(
                2, "excluded_growth_run_ids", ";".join(excluded)
            )
            exclusion_predictions.append(prediction)
            exclusion_metrics.append(
                _baseline_metrics(
                    prediction, target=target, excluded=excluded
                )
            )
    exclusion_prediction_table = pd.concat(
        exclusion_predictions, ignore_index=True
    )
    exclusion_metric_table = pd.DataFrame(exclusion_metrics)
    write_csv(
        exclusion_prediction_table,
        report / "exclusion_sensitivity_predictions.csv",
    )
    write_csv(
        exclusion_metric_table,
        report / "exclusion_sensitivity_metrics.csv",
    )

    candidate_config = CandidateConfig(
        density_strength=float(config["density_strength"]),
        density_floor=float(config["density_floor"]),
        residual_strength=float(config["residual_strength"]),
        residual_floor=float(config["residual_floor"]),
        r3d_pca_components=int(config["r3d_pca_components"]),
        ridge_alpha=float(config["robust_ridge_alpha"]),
        baseline_alpha=float(config["baseline_ridge_alpha"]),
        morphology_weight=float(config["baseline_morphology_weight"]),
    )
    prediction_frames = []
    selected_frames = []
    inner_frames = []
    weight_frames = []
    for target, series in (("Rq_nm", log_rq), ("FSMI_nm", log_fsmi)):
        fixed, selected, inner = crossfit_robust_candidates(
            physics=physics,
            embeddings=embeddings,
            log_target=series,
            config=candidate_config,
            confidence_alpha=float(config["confidence_alpha"]),
        )
        for frame in (fixed, selected, inner):
            frame.insert(0, "target", target)
        prediction_frames.append(fixed)
        selected_frames.append(selected)
        inner_frames.append(inner)
        weights = training_weight_audit(
            physics=physics,
            log_target=series,
            config=candidate_config,
        )
        weights.insert(0, "target", target)
        weight_frames.append(weights)
    predictions = pd.concat(
        prediction_frames + selected_frames, ignore_index=True
    )
    selected_methods = {
        str(target): str(method)
        for target, method in config[
            "generation_prediction_methods"
        ].items()
    }
    final_frames = []
    for target, method in selected_methods.items():
        rows = predictions.loc[
            (predictions["target"] == target)
            & (predictions["method"] == method)
        ].copy()
        if len(rows) != len(groups):
            raise RuntimeError(
                f"target-specific selection incomplete: {target} {method}"
            )
        rows["selected_candidate"] = method
        rows["method"] = FINAL_TARGET_SPECIFIC
        final_frames.append(rows)
    predictions = pd.concat(
        [predictions, *final_frames], ignore_index=True
    )
    inner_predictions = pd.concat(inner_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    write_csv(predictions, report / "robust_crossfit_predictions.csv")
    write_csv(
        inner_predictions, report / "robust_nested_inner_predictions.csv"
    )
    write_csv(weights, report / "training_weight_audit.csv")
    metric_records = []
    for (target, method), rows in predictions.groupby(
        ["target", "method"]
    ):
        metric_records.append(
            _metrics(rows, target=target, method=method)
        )
    metrics = pd.DataFrame(metric_records).sort_values(
        ["target", "mae", "method"]
    )
    write_csv(metrics, report / "robust_method_metrics.csv")
    risk_coverage = _risk_coverage(predictions)
    write_csv(risk_coverage, report / "risk_coverage.csv")

    generation_predictions = predictions.loc[
        predictions["method"] == FINAL_TARGET_SPECIFIC
    ].copy()
    if generation_predictions["target"].nunique() != 2:
        raise RuntimeError(
            "target-specific generation selection is missing a target"
        )
    for target, filename in (
        ("Rq_nm", "rq_selected_predictions.csv"),
        ("FSMI_nm", "fsmi_selected_predictions.csv"),
    ):
        write_csv(
            generation_predictions.loc[
                generation_predictions["target"] == target
            ].drop(columns=["target"]),
            report / filename,
        )

    figure_paths: list[str] = []
    figure_paths += plot_ood_audit(audit, figures)
    figure_paths += plot_exclusion_sensitivity(
        exclusion_metric_table, figures
    )
    figure_paths += plot_method_summary(metrics, figures)
    figure_paths += plot_predictions(
        predictions,
        [
            BASELINE,
            DENSITY_WEIGHTED,
            MULTIVIEW_20,
            NESTED_SELECTOR,
            FINAL_TARGET_SPECIFIC,
        ],
        figures,
    )
    figure_paths += plot_confidence_and_coverage(
        predictions,
        risk_coverage,
        FINAL_TARGET_SPECIFIC,
        figures,
    )
    figure_paths += plot_training_weights(
        weights.loc[weights["target"] == "Rq_nm"], figures
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "protocol": (
            "retrospective full23 leave-one-growth-out; every target "
            "prediction is fitted without its held growth"
        ),
        "growth_group_count": len(groups),
        "growth_run_ids": groups,
        "canonical_removelist_changed": False,
        "canonical_removelist_sha256": sha256_file(
            repo_path(config["removelist_path"])
        ),
        "target_blind_exclusion_ranking": audit[
            ["growth_run_id", "rheed_only_ood_rank"]
        ].to_dict(orient="records"),
        "exclusion_sensitivity_counts": [0, 2, 3, 4],
        "generation_prediction_method": FINAL_TARGET_SPECIFIC,
        "generation_prediction_methods_by_target": selected_methods,
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "outer_target_used_for_training": False,
        "method_metrics": metrics.to_dict(orient="records"),
        "figure_paths": figure_paths,
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "The full23 cohort has already been used for retrospective "
            "method development. These LOO results compare methods without "
            "fold leakage but are not a prospective untouched test. "
            "Exclusion cohorts are sensitivity analyses, not a new test set."
        ),
    }
    write_json(manifest, report / "experiment_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_ood_robust.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
