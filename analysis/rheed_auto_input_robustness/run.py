from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_full_cohort_loo.run import (
    _target_series,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    group_metric_table,
    scan_metric_table,
)
from analysis.rheed_to_afm_functional_morphology.run import _physics_table
from analysis.rheed_to_afm_generation.run import _load_tables
from analysis.rheed_to_afm_ood_robust.prediction import (
    R3D_TEMPORAL,
    CandidateConfig,
    crossfit_robust_candidates,
)
from analysis.rheed_video_afm_story.publication_style import (
    set_publication_style,
)

from .confidence import crossfit_r3d_stability_confidence

ROOT = Path(__file__).resolve().parents[2]


def _display_target(target: str) -> str:
    """Paper-facing label; Rq_nm remains only as a legacy schema key."""

    return "Sq" if target == "Rq_nm" else target.replace("_nm", "")


def _load_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    base_path = payload.pop("base_config", None)
    if base_path is None:
        return payload
    base = _load_config(str(base_path))
    base.update(payload)
    return base


def _aurc(errors: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-np.asarray(confidence, dtype=float))
    retained = np.asarray(errors, dtype=float)[order]
    risk = np.cumsum(retained) / np.arange(1, len(retained) + 1)
    return float(np.mean(risk))


def _risk_at_coverage(
    errors: np.ndarray,
    confidence: np.ndarray,
    coverage: float,
) -> float:
    count = max(1, int(np.ceil(float(coverage) * len(errors))))
    order = np.argsort(-np.asarray(confidence, dtype=float))[:count]
    return float(np.mean(np.asarray(errors, dtype=float)[order]))


def _metric_row(
    rows: pd.DataFrame,
    *,
    target: str,
    confidence_column: str = "confidence",
) -> dict[str, Any]:
    truth = rows["true_target"].to_numpy(float)
    predicted = rows["predicted_target"].to_numpy(float)
    errors = np.abs(truth - predicted)
    confidence = rows[confidence_column].to_numpy(float)
    linear = pearsonr(truth, predicted)
    rank = spearmanr(truth, predicted)
    reliability = spearmanr(confidence, errors)
    return {
        "target": target,
        "growth_group_count": len(rows),
        "mae_nm": float(np.mean(errors)),
        "median_absolute_error_nm": float(np.median(errors)),
        "rmse_nm": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_r": float(linear.statistic),
        "pearson_p": float(linear.pvalue),
        "spearman_rho": float(rank.statistic),
        "spearman_p": float(rank.pvalue),
        "confidence_vs_absolute_error_spearman": float(
            reliability.statistic
        ),
        "confidence_vs_absolute_error_p": float(reliability.pvalue),
        "aurc_nm": _aurc(errors, confidence),
        "risk_at_25pct_coverage_nm": _risk_at_coverage(
            errors, confidence, 0.25
        ),
        "risk_at_50pct_coverage_nm": _risk_at_coverage(
            errors, confidence, 0.50
        ),
        "risk_at_75pct_coverage_nm": _risk_at_coverage(
            errors, confidence, 0.75
        ),
        "interval_coverage": float(
            rows["interval_covered"].astype(float).mean()
        ),
    }


def _confidence_ablation_row(
    rows: pd.DataFrame,
    *,
    target: str,
    method: str,
    confidence: np.ndarray | pd.Series,
) -> dict[str, Any]:
    errors = rows["absolute_error"].to_numpy(float)
    values = np.asarray(confidence, dtype=float)
    relation = spearmanr(values, errors)
    return {
        "target": target,
        "confidence_method": method,
        "confidence_vs_absolute_error_spearman": float(
            relation.statistic
        ),
        "p_value": float(relation.pvalue),
        "aurc_nm": _aurc(errors, values),
        "risk_at_50pct_coverage_nm": _risk_at_coverage(
            errors,
            values,
            0.50,
        ),
    }


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        destination.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_predictions(
    predictions: pd.DataFrame,
    figure_root: Path,
) -> None:
    cohort_count = int(predictions["growth_run_id"].nunique())
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.8), constrained_layout=True
    )
    scatter = None
    for axis, target in zip(axes, ("Rq_nm", "FSMI_nm")):
        display = _display_target(target)
        rows = predictions.loc[predictions["target"] == target]
        low = float(min(rows["true_target"].min(), rows["predicted_target"].min()))
        high = float(max(rows["true_target"].max(), rows["predicted_target"].max()))
        pad = 0.05 * max(high - low, 1.0)
        scatter = axis.scatter(
            rows["true_target"],
            rows["predicted_target"],
            c=100.0 * rows["confidence"],
            cmap="viridis",
            vmin=0,
            vmax=100,
            s=52,
            edgecolor="black",
            linewidth=0.5,
        )
        axis.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            "--",
            color="#666666",
            lw=1,
        )
        metric = _metric_row(rows, target=target)
        axis.text(
            0.04,
            0.96,
            f"MAE={metric['mae_nm']:.2f} nm\n"
            f"Pearson r={metric['pearson_r']:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_xlim(low - pad, high + pad)
        axis.set_ylim(low - pad, high + pad)
        axis.set_xlabel(f"Measured {display} (nm)")
        axis.set_ylabel(f"Predicted {display} (nm)")
        axis.set_title(
            f"{display}: strict {cohort_count}-fold automatic-input LOO"
        )
        axis.grid(alpha=0.18)
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axes, pad=0.02)
    colorbar.set_label("Reliability confidence (%)")
    figure.suptitle(
        "M15b causal R3D predicts automatic-input Sq and FSMI",
        fontsize=12,
    )
    _save_figure(figure, figure_root / "Fig1_m15b_target_predictions")


def _plot_confidence(
    predictions: pd.DataFrame,
    figure_root: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.6), constrained_layout=True
    )
    for axis, target in zip(axes, ("Rq_nm", "FSMI_nm")):
        display = _display_target(target)
        rows = predictions.loc[predictions["target"] == target].copy()
        axis.scatter(
            100.0 * rows["confidence"],
            rows["absolute_error"],
            color="#0072B2",
            s=52,
            edgecolor="black",
            linewidth=0.5,
        )
        for row in rows.nlargest(3, "absolute_error").itertuples():
            axis.annotate(
                str(row.growth_run_id),
                (100.0 * row.confidence, row.absolute_error),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        correlation = spearmanr(rows["confidence"], rows["absolute_error"])
        p_label = (
            "<0.001"
            if correlation.pvalue < 0.001
            else f"{correlation.pvalue:.3f}"
        )
        axis.text(
            0.04,
            0.95,
            f"Spearman ρ={correlation.statistic:.2f}\n"
            f"p={p_label}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_xlabel("Confidence (%)")
        axis.set_ylabel("Absolute prediction error (nm)")
        axis.set_title(display)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Confidence is derived without the outer held AFM target",
        fontsize=12,
    )
    _save_figure(figure, figure_root / "Fig2_confidence_vs_error")


def _plot_risk_coverage(
    predictions: pd.DataFrame,
    prior: pd.DataFrame,
    figure_root: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.6), constrained_layout=True
    )
    for axis, target in zip(axes, ("Rq_nm", "FSMI_nm")):
        rows = predictions.loc[predictions["target"] == target].set_index(
            "growth_run_id"
        )
        old = prior.loc[
            (prior["target"] == target)
            & (prior["method"] == R3D_TEMPORAL)
        ].set_index("growth_run_id").loc[rows.index]
        for label, confidence, color in (
            ("M15b range-aware + angular TTA", rows["confidence"], "#0072B2"),
            ("old density/amplitude risk", old["confidence"], "#D55E00"),
        ):
            order = np.argsort(-confidence.to_numpy(float))
            error = rows["absolute_error"].to_numpy(float)[order]
            risk = np.cumsum(error) / np.arange(1, len(error) + 1)
            coverage = np.arange(1, len(error) + 1) / len(error)
            axis.plot(coverage, risk, lw=2, label=label, color=color)
        axis.axhline(
            rows["absolute_error"].mean(),
            color="#666666",
            ls="--",
            lw=1,
            label="all-sample MAE",
        )
        axis.set_xlabel("Retained coverage")
        axis.set_ylabel("MAE of retained predictions (nm)")
        axis.set_title(_display_target(target))
        axis.grid(alpha=0.18)
    axes[0].legend(fontsize=7)
    figure.suptitle(
        "Selective risk–coverage audit (lower is better)",
        fontsize=12,
    )
    _save_figure(figure, figure_root / "Fig3_risk_coverage")


def _plot_method_ablation(
    prior: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_root: Path,
) -> None:
    methods = [
        "M14b_rheed_density_weighted",
        "M14g_multiview_curated60_r3d40",
        R3D_TEMPORAL,
    ]
    labels = ["M14b\nphysics", "M14g\n60% physics", "M15b / M14d\ncausal R3D"]
    figure, axes = plt.subplots(
        1, 2, figsize=(7.9, 3.7), constrained_layout=True
    )
    colors = ["#D55E00", "#E69F00", "#0072B2"]
    for axis, target in zip(axes, ("Rq_nm", "FSMI_nm")):
        maes = []
        correlations = []
        for method in methods:
            rows = prior.loc[
                (prior["target"] == target)
                & (prior["method"] == method)
            ]
            maes.append(float(rows["absolute_error"].mean()))
            correlations.append(
                float(
                    pearsonr(
                        rows["true_target"],
                        rows["predicted_target"],
                    ).statistic
                )
            )
        x = np.arange(len(methods))
        bars = axis.bar(x, maes, color=colors)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel("Strict LOO MAE (nm)")
        axis.set_title(_display_target(target))
        axis.grid(axis="y", alpha=0.18)
        for bar, corr in zip(bars, correlations):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"r={corr:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(
        "Automatic-input head ablation: temporal representation is decisive",
        fontsize=12,
    )
    _save_figure(figure, figure_root / "Fig4_target_head_ablation")


def _plot_all_sample_predictions(
    predictions: pd.DataFrame,
    figure_root: Path,
) -> None:
    """Show every held growth in fixed measured-value order."""

    cohort_count = int(predictions["growth_run_id"].nunique())
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(max(9.4, 0.34 * cohort_count), 6.5),
        constrained_layout=True,
    )
    scatter = None
    for axis, target in zip(axes, ("Rq_nm", "FSMI_nm")):
        display = _display_target(target)
        rows = (
            predictions.loc[predictions["target"] == target]
            .sort_values(["true_target", "growth_run_id"])
            .reset_index(drop=True)
        )
        x = np.arange(len(rows))
        axis.plot(
            x,
            rows["true_target"],
            color="#D55E00",
            marker="o",
            ms=4,
            lw=1.8,
            label="measured",
        )
        axis.plot(
            x,
            rows["predicted_target"],
            color="#56B4E9",
            lw=1.2,
            alpha=0.9,
            label="strict LOO prediction",
        )
        scatter = axis.scatter(
            x,
            rows["predicted_target"],
            c=100.0 * rows["confidence"],
            cmap="viridis",
            vmin=0,
            vmax=100,
            s=48,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        axis.vlines(
            x,
            rows["true_target"],
            rows["predicted_target"],
            color="#999999",
            lw=0.7,
            alpha=0.55,
            zorder=0,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            rows["growth_run_id"],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel(f"{display} (nm)")
        axis.set_title(
            f"{display}: all {cohort_count} automatic-input held folds"
        )
        axis.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="upper left", ncols=2, fontsize=8)
    axes[-1].set_xlabel(
        "Held growth (independently ordered by measured target)"
    )
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axes, pad=0.015)
    colorbar.set_label("Reliability confidence (%)")
    figure.suptitle(
        "No cherry-picking: measured and predicted morphology endpoints",
        fontsize=12,
    )
    _save_figure(
        figure,
        figure_root / f"Fig6_all{cohort_count}_ordered_predictions",
    )


def _expanded_prior_and_targets(
    *,
    config: dict[str, Any],
    payload: Any,
    groups: list[str],
    view_names: list[str],
    physics: pd.DataFrame,
    candidate_config: CandidateConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build same-cohort strict LOO baselines and targets for an expansion.

    This path avoids importing the historical human-selection comparison,
    which cannot cover newly acquired automatic-only growths.  Method
    hyperparameters remain frozen; every candidate head is refit on the other
    growths inside each outer fold.
    """

    cohort_config = _load_config(config["cohort_config"])
    tables = _load_tables(cohort_config)
    descriptors, _ = prepare_full_cohort(tables, cohort_config)
    scan_metrics = scan_metric_table(
        descriptors,
        scan_size_nm=float(cohort_config["scan_size_nm"]),
        analysis_scale_nm=float(cohort_config["analysis_scale_nm"]),
    )
    group_metrics = group_metric_table(scan_metrics)
    log_rq, log_fsmi = _target_series(descriptors, group_metrics)
    targets = {"Rq_nm": log_rq, "FSMI_nm": log_fsmi}
    if set(log_rq.index) != set(groups) or set(log_fsmi.index) != set(groups):
        raise RuntimeError("expanded AFM targets do not match perturbation groups")
    base_index = view_names.index("base")
    embeddings = pd.DataFrame(
        np.asarray(payload["embeddings"][:, base_index], dtype=np.float32),
        index=groups,
    )
    frames: list[pd.DataFrame] = []
    target_records: list[dict[str, Any]] = []
    for target_name, log_target in targets.items():
        log_target = log_target.loc[groups]
        baseline, _, _ = crossfit_robust_candidates(
            physics=physics,
            embeddings=embeddings,
            log_target=log_target,
            config=candidate_config,
            confidence_alpha=float(config.get("confidence_alpha", 0.10)),
        )
        baseline.insert(0, "target", target_name)
        frames.append(baseline)
        for group in groups:
            target_records.append(
                {
                    "growth_run_id": group,
                    "target": target_name,
                    "true_target": float(np.exp(log_target.loc[group])),
                    "outer_fold_unit": "growth_run_id",
                }
            )
    prior = pd.concat(frames, ignore_index=True)
    prior.to_csv(report_root / "expanded_head_baselines.csv", index=False)
    pd.DataFrame(target_records).to_csv(
        report_root / "expanded_afm_targets.csv", index=False
    )
    return prior, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_auto_input_robustness.json",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    output_root = ROOT / config["output_root"]
    report_root = ROOT / config["report_root"]
    figure_root = report_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    perturbation_path = config.get("perturbation_embeddings")
    if perturbation_path is None:
        perturbation_path = output_root / "r3d_causal8_input_perturbations.npz"
    elif isinstance(perturbation_path, str):
        perturbation_path = ROOT / perturbation_path
    source_payload = np.load(perturbation_path, allow_pickle=False)
    source_groups = [
        str(value) for value in source_payload["growth_run_ids"].tolist()
    ]
    excluded_growths = set(map(str, config["excluded_growths"]))
    keep = np.asarray(
        [group not in excluded_growths for group in source_groups],
        dtype=bool,
    )
    groups = [
        group for group, retained in zip(source_groups, keep) if retained
    ]
    payload = {
        "embeddings": np.asarray(source_payload["embeddings"])[keep],
    }
    view_names = [
        str(value) for value in source_payload["view_names"].tolist()
    ]
    if len(groups) != int(config["cohort_count"]):
        raise RuntimeError("unexpected automatic-input cohort size")
    if set(groups) & excluded_growths:
        raise RuntimeError("excluded growth entered robustness cohort")

    target_parameters = json.loads(
        (ROOT / config["target_parameters"]).read_text(encoding="utf-8")
    )
    candidate_config = CandidateConfig(
        density_strength=float(target_parameters["density_strength"]),
        density_floor=float(target_parameters["density_floor"]),
        residual_strength=float(target_parameters["residual_strength"]),
        residual_floor=float(target_parameters["residual_floor"]),
        r3d_pca_components=int(target_parameters["r3d_pca_components"]),
        ridge_alpha=float(target_parameters["robust_ridge_alpha"]),
        baseline_alpha=float(target_parameters["baseline_ridge_alpha"]),
        morphology_weight=float(
            target_parameters["baseline_morphology_weight"]
        ),
    )
    physics_path = config.get("confidence_physics")
    if physics_path is None:
        physics_path = output_root / "physics_roi_strict_loo_features.csv"
    physics = _physics_table(
        pd.read_csv(
            ROOT / physics_path
            if isinstance(physics_path, str)
            else physics_path,
            dtype={"sample_id": str, "growth_run_id": str},
        )
    ).loc[groups]
    if config.get("cohort_config"):
        prior, target_series = _expanded_prior_and_targets(
            config=config,
            payload=payload,
            groups=groups,
            view_names=view_names,
            physics=physics,
            candidate_config=candidate_config,
            report_root=report_root,
        )
    else:
        prior = pd.read_csv(
            ROOT / config["prior_comparison_report"]
            / "machine_robust_crossfit_predictions.csv",
            dtype={"growth_run_id": str},
        )
        target_series = {
            target: np.log(
                prior.loc[
                    (prior["target"] == target)
                    & (prior["method"] == R3D_TEMPORAL)
                ]
                .set_index("growth_run_id")
                .loc[groups, "true_target"]
            )
            for target in ("Rq_nm", "FSMI_nm")
        }
    selection = pd.read_csv(
        ROOT / config["selection_table"],
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id").loc[groups]
    periods = selection["estimated_period_frames"]
    predictions = []
    inner = []
    for target in ("Rq_nm", "FSMI_nm"):
        log_target = target_series[target].loc[groups]
        outer_rows, inner_rows = crossfit_r3d_stability_confidence(
            perturbation_embeddings=payload["embeddings"],
            view_names=view_names,
            groups=groups,
            log_target=log_target,
            physics=physics,
            candidate_config=candidate_config,
            estimated_period_frames=periods,
        )
        outer_rows.insert(0, "target", target)
        inner_rows.insert(0, "target", target)
        predictions.append(outer_rows)
        inner.append(inner_rows)
    prediction_table = pd.concat(predictions, ignore_index=True)
    inner_table = pd.concat(inner, ignore_index=True)
    prediction_table["confidence_method"] = (
        "strictly nested 75% predicted-amplitude support + 25% angular-TTA "
        "risk with a discrete 10% extreme temporal-vs-physics disagreement "
        "veto; outer AFM target excluded"
    )
    prediction_table.to_csv(
        report_root / "m15b_strict_loo_predictions.csv",
        index=False,
    )
    inner_table.to_csv(
        report_root / "m15b_nested_inner_predictions.csv",
        index=False,
    )
    m15a_ablation = prediction_table.copy()
    m15a_ablation["confidence"] = m15a_ablation[
        "tta_centrality_confidence"
    ]
    m15a_ablation["confidence_method"] = (
        "strictly nested keyframe/ROI TTA centrality only"
    )
    m15a_ablation.to_csv(
        report_root / "m15a_tta_centrality_ablation_predictions.csv",
        index=False,
    )

    metrics = pd.DataFrame(
        [
            _metric_row(
                prediction_table.loc[prediction_table["target"] == target],
                target=target,
            )
            for target in ("Rq_nm", "FSMI_nm")
        ]
    )
    metrics.to_csv(report_root / "m15b_metrics.csv", index=False)

    confidence_ablation = []
    baseline_comparison = []
    for target in ("Rq_nm", "FSMI_nm"):
        rows = prediction_table.loc[
            prediction_table["target"] == target
        ].copy()
        old = (
            prior.loc[
                (prior["target"] == target)
                & (prior["method"] == R3D_TEMPORAL)
            ]
            .set_index("growth_run_id")
            .loc[rows["growth_run_id"]]
        )
        variance_risk = rows["tta_all_std_nm"].rank(
            method="average",
            pct=True,
        )
        for method, confidence in (
            ("old_density_amplitude_risk", old["confidence"]),
            ("tta_variance_only", 1.0 - variance_risk),
            (
                "m15a_tta_centrality_only",
                rows["tta_centrality_confidence"],
            ),
            (
                "rotation_period_only",
                1.0 - rows["rotation_period_risk_score"],
            ),
            (
                "m15b_angular_coverage_x_tta",
                rows["angular_coverage_tta_confidence"],
            ),
            ("m15b_final_with_conflict_veto", rows["confidence"]),
        ):
            confidence_ablation.append(
                _confidence_ablation_row(
                    rows,
                    target=target,
                    method=method,
                    confidence=confidence,
                )
            )
        for method, label in (
            ("M14b_rheed_density_weighted", "M14b physics"),
            ("M14g_multiview_curated60_r3d40", "M14g mixed"),
            (R3D_TEMPORAL, "M14d causal R3D / old confidence"),
        ):
            candidate = prior.loc[
                (prior["target"] == target)
                & (prior["method"] == method)
            ].copy()
            metric = _metric_row(candidate, target=target)
            metric.update({"model": label, "method": method})
            baseline_comparison.append(metric)
        for label, candidate in (
            ("M15a causal R3D / TTA centrality", m15a_ablation),
            ("M15b causal R3D / angular TTA", prediction_table),
        ):
            candidate_rows = candidate.loc[candidate["target"] == target]
            metric = _metric_row(candidate_rows, target=target)
            metric.update({"model": label, "method": label.split()[0]})
            baseline_comparison.append(metric)
    pd.DataFrame(confidence_ablation).to_csv(
        report_root / "confidence_method_ablation.csv",
        index=False,
    )
    pd.DataFrame(baseline_comparison).to_csv(
        report_root / "baseline_vs_final_metrics.csv",
        index=False,
    )
    _plot_predictions(prediction_table, figure_root)
    _plot_confidence(prediction_table, figure_root)
    _plot_risk_coverage(prediction_table, prior, figure_root)
    _plot_method_ablation(prior, prediction_table, figure_root)
    _plot_all_sample_predictions(prediction_table, figure_root)

    centrality_ablation = []
    for target, rows in prediction_table.groupby("target"):
        for feature in (
            "base_to_tta_median_nm",
            "tta_frame_std_nm",
            "tta_roi_std_nm",
            "tta_all_std_nm",
            "tta_all_range_nm",
        ):
            relation = spearmanr(rows[feature], rows["absolute_error"])
            centrality_ablation.append(
                {
                    "target": target,
                    "uncertainty_feature": feature,
                    "uncertainty_vs_absolute_error_spearman": float(
                        relation.statistic
                    ),
                    "p_value": float(relation.pvalue),
                }
            )
    pd.DataFrame(centrality_ablation).to_csv(
        report_root / "uncertainty_feature_ablation.csv",
        index=False,
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "final_model_id": "MorphMBE-M15b-AutoR3D-AngularTTA",
        "cohort_count": len(groups),
        "groups": groups,
        "excluded_growths": config["excluded_growths"],
        "point_prediction_method": R3D_TEMPORAL,
        "selection_evidence": config.get(
            "selection_evidence",
            "selected by inner CV in all outer folds for both targets",
        ),
        "confidence_method": (
            "strictly nested 75% predicted-amplitude support + 25% "
            "angular-TTA risk, with a discrete 10% extreme "
            "temporal-vs-physics disagreement veto"
        ),
        "rotation_period_source": (
            "automatic RHEED spot trajectory; AFM-target blind"
        ),
        "confidence_calibration": (
            "true nested inner-fold reference predictions"
        ),
        "outer_target_used_for_training_or_confidence": False,
        "raw_data_modified": False,
        "standalone_modified": False,
    }
    (report_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
