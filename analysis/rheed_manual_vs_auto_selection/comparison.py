from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_full_cohort_loo.run import (
    _target_series,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    _higher_quantile,
    _range_calibrate,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    group_metric_table,
    scan_metric_table,
)
from analysis.rheed_to_afm_functional_morphology.run import _physics_table
from analysis.rheed_to_afm_generation.run import _load_tables
from analysis.rheed_to_afm_ood_robust.prediction import (
    DENSITY_WEIGHTED,
    FINAL_TARGET_SPECIFIC,
    MULTIVIEW_60,
    CandidateConfig,
    _expected_error_and_confidence,
    _nested_calibrated_errors,
    crossfit_robust_candidates,
    predict_candidates,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.publication_style import (
    set_publication_style,
)

from .dataset import load_config


PROTOCOL_HUMAN = "human→human (frozen strict LOO)"
PROTOCOL_AUTO = "auto→auto (strict LOO)"
PROTOCOL_SHIFT = "human→auto (strict cross-domain LOO)"


def _embedding_frame(registry_path: str | Path, embedding_id: str) -> pd.DataFrame:
    registry = pd.read_csv(repo_path(registry_path))
    row = registry.loc[registry["embedding_id"] == embedding_id]
    if len(row) != 1:
        raise RuntimeError(f"embedding is not unique: {embedding_id}")
    payload = np.load(repo_path(str(row.iloc[0]["path"])), allow_pickle=False)
    groups = [str(value) for value in payload["sample_ids"].tolist()]
    return pd.DataFrame(
        np.asarray(payload["embeddings"], dtype=np.float32),
        index=groups,
    )


def _candidate_config(parameters: dict[str, Any]) -> CandidateConfig:
    return CandidateConfig(
        density_strength=float(parameters["density_strength"]),
        density_floor=float(parameters["density_floor"]),
        residual_strength=float(parameters["residual_strength"]),
        residual_floor=float(parameters["residual_floor"]),
        r3d_pca_components=int(parameters["r3d_pca_components"]),
        ridge_alpha=float(parameters["robust_ridge_alpha"]),
        baseline_alpha=float(parameters["baseline_ridge_alpha"]),
        morphology_weight=float(parameters["baseline_morphology_weight"]),
    )


def _cross_domain_target(
    *,
    human_physics: pd.DataFrame,
    human_embeddings: pd.DataFrame,
    auto_physics: pd.DataFrame,
    auto_embeddings: pd.DataFrame,
    log_target: pd.Series,
    method: str,
    config: CandidateConfig,
    confidence_alpha: float,
) -> pd.DataFrame:
    """Fit only human-selected training rows and query the held auto row."""

    groups = list(map(str, log_target.index))
    records: list[dict[str, Any]] = []
    for held in groups:
        fit = [group for group in groups if group != held]
        query = f"__auto_query_{held}__"
        outer_physics = pd.concat(
            [
                human_physics.loc[fit],
                auto_physics.loc[[held]].rename(index={held: query}),
            ]
        )
        outer_embeddings = pd.concat(
            [
                human_embeddings.loc[fit],
                auto_embeddings.loc[[held]].rename(index={held: query}),
            ]
        )
        outer_log, outer_diagnostics = predict_candidates(
            physics=outer_physics,
            embeddings=outer_embeddings,
            log_target=log_target,
            fit_groups=fit,
            query_group=query,
            config=config,
        )
        inner_raw: list[float] = []
        inner_truth: list[float] = []
        inner_diagnostics: list[dict[str, float]] = []
        for inner_held in fit:
            inner_fit = [group for group in fit if group != inner_held]
            predicted, diagnostic = predict_candidates(
                physics=human_physics,
                embeddings=human_embeddings,
                log_target=log_target,
                fit_groups=inner_fit,
                query_group=inner_held,
                config=config,
            )
            inner_raw.append(float(np.exp(predicted[method])))
            inner_truth.append(float(np.exp(log_target.loc[inner_held])))
            inner_diagnostics.append(diagnostic)
        inner_raw_array = np.asarray(inner_raw, dtype=float)
        inner_truth_array = np.asarray(inner_truth, dtype=float)
        calibrated_inner, inner_errors = _nested_calibrated_errors(
            inner_raw_array, inner_truth_array
        )
        raw_outer = float(np.exp(outer_log[method]))
        calibrated_outer, calibration_scale = _range_calibrate(
            inner_raw_array,
            inner_truth_array,
            raw_outer,
        )
        diagnostics_frame = pd.DataFrame(inner_diagnostics)
        expected_error, confidence, risk, inner_risk = (
            _expected_error_and_confidence(
                diagnostics_frame,
                inner_errors,
                calibrated_inner,
                outer_diagnostics,
                calibrated_outer,
            )
        )
        adaptive = inner_errors / (1.0 + inner_risk)
        radius = _higher_quantile(
            adaptive, 1.0 - float(confidence_alpha)
        ) * (1.0 + risk)
        truth = float(np.exp(log_target.loc[held]))
        records.append(
            {
                "growth_run_id": held,
                "method": FINAL_TARGET_SPECIFIC,
                "selected_candidate": method,
                "true_target": truth,
                "raw_predicted_target": raw_outer,
                "predicted_target": calibrated_outer,
                "range_calibration_log_scale": calibration_scale,
                "absolute_error": abs(calibrated_outer - truth),
                "predicted_absolute_error": expected_error,
                "confidence": confidence,
                "interval_lower": max(calibrated_outer - radius, 0.0),
                "interval_upper": calibrated_outer + radius,
                "interval_radius": radius,
                "interval_covered": bool(
                    max(calibrated_outer - radius, 0.0)
                    <= truth
                    <= calibrated_outer + radius
                ),
                "outer_target_used_for_training": False,
                "outer_fit_growth_count": len(fit),
                "training_selection_source": "human",
                "query_selection_source": "automatic_v5_v8",
                "uncertainty_risk_score": risk,
                **outer_diagnostics,
            }
        )
    return pd.DataFrame(records)


def _metric_row(rows: pd.DataFrame, *, target: str, protocol: str) -> dict[str, Any]:
    truth = rows["true_target"].to_numpy(float)
    predicted = rows["predicted_target"].to_numpy(float)
    error = np.abs(truth - predicted)
    confidence = rows["confidence"].to_numpy(float)
    pearson = pearsonr(truth, predicted)
    spearman = spearmanr(truth, predicted)
    confidence_error = spearmanr(confidence, error)
    return {
        "target": target,
        "protocol": protocol,
        "growth_group_count": len(rows),
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
        "confidence_vs_absolute_error_p": float(confidence_error.pvalue),
        "mean_confidence": float(np.mean(confidence)),
        "interval_coverage": float(rows["interval_covered"].astype(float).mean()),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
        "truth_min": float(np.min(truth)),
        "truth_max": float(np.max(truth)),
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _display_contrast(image: np.ndarray) -> np.ndarray:
    """Contrast-normalize RHEED for figures only; model inputs stay untouched."""

    values = np.asarray(image, dtype=float)
    low, high = np.percentile(values, [1.0, 99.7])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _plot_target_protocols(predictions: pd.DataFrame, figure_root: Path) -> None:
    colors = {
        PROTOCOL_HUMAN: "#0072B2",
        PROTOCOL_AUTO: "#009E73",
        PROTOCOL_SHIFT: "#D55E00",
    }
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    for row_index, target in enumerate(("Rq_nm", "FSMI_nm")):
        target_rows = predictions.loc[predictions["target"] == target]
        lo = float(
            min(target_rows["true_target"].min(), target_rows["predicted_target"].min())
        )
        hi = float(
            max(target_rows["true_target"].max(), target_rows["predicted_target"].max())
        )
        padding = 0.05 * (hi - lo)
        for column_index, protocol in enumerate(
            (PROTOCOL_HUMAN, PROTOCOL_AUTO, PROTOCOL_SHIFT)
        ):
            axis = axes[row_index, column_index]
            rows = target_rows.loc[target_rows["protocol"] == protocol]
            scatter = axis.scatter(
                rows["true_target"],
                rows["predicted_target"],
                c=100.0 * rows["confidence"],
                cmap="viridis",
                vmin=0,
                vmax=100,
                s=46,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            axis.plot([lo - padding, hi + padding], [lo - padding, hi + padding], "--", color="#666666", lw=1)
            metric = _metric_row(rows, target=target, protocol=protocol)
            axis.text(
                0.04,
                0.95,
                f"MAE={metric['mae']:.2f} nm\nr={metric['pearson_r']:.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
            axis.set_xlim(lo - padding, hi + padding)
            axis.set_ylim(lo - padding, hi + padding)
            axis.set_xlabel(f"Measured {target.replace('_nm', '')} (nm)")
            axis.set_ylabel(f"Predicted {target.replace('_nm', '')} (nm)")
            axis.set_title(protocol if row_index == 0 else target.replace("_nm", ""))
            axis.grid(alpha=0.18)
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.78, pad=0.015)
    colorbar.set_label("Error-related confidence index")
    figure.suptitle(
        "Effect of automatic keyframe/ROI selection on frozen M14i methodology",
        fontsize=13,
    )
    _save_figure(figure, figure_root / "Fig1_target_protocol_comparison")


def _plot_rq_ordered(predictions: pd.DataFrame, figure_root: Path) -> None:
    rows = predictions.loc[predictions["target"] == "Rq_nm"].copy()
    order = (
        rows.loc[rows["protocol"] == PROTOCOL_HUMAN]
        .sort_values("true_target")["growth_run_id"]
        .tolist()
    )
    x = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(12.2, 4.4), constrained_layout=True)
    truth = (
        rows.loc[rows["protocol"] == PROTOCOL_HUMAN]
        .set_index("growth_run_id")
        .loc[order, "true_target"]
    )
    axis.plot(x, truth, "o-", color="#CC4C02", lw=2, label="Measured Rq")
    for protocol, color, marker in (
        (PROTOCOL_HUMAN, "#0072B2", "o"),
        (PROTOCOL_AUTO, "#009E73", "s"),
        (PROTOCOL_SHIFT, "#D55E00", "^"),
    ):
        values = (
            rows.loc[rows["protocol"] == protocol]
            .set_index("growth_run_id")
            .loc[order]
        )
        axis.plot(
            x,
            values["predicted_target"],
            marker=marker,
            ms=4.5,
            color=color,
            lw=1.3,
            label=protocol,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=55, ha="right")
    axis.set_ylabel("Rq (nm)")
    axis.set_xlabel("Held growth (ordered by measured Rq)")
    axis.set_title("Strict LOO Rq: input protocol changes, AFM targets remain held")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    _save_figure(figure, figure_root / "Fig2_rq_ordered_protocol_comparison")


def _plot_input_shift(
    predictions: pd.DataFrame,
    selection: pd.DataFrame,
    figure_root: Path,
) -> None:
    human = predictions.loc[
        (predictions["target"] == "Rq_nm")
        & (predictions["protocol"] == PROTOCOL_HUMAN)
    ].set_index("growth_run_id")
    shifted = predictions.loc[
        (predictions["target"] == "Rq_nm")
        & (predictions["protocol"] == PROTOCOL_SHIFT)
    ].set_index("growth_run_id")
    table = selection.set_index("growth_run_id").loc[human.index].copy()
    table["absolute_prediction_shift_nm"] = np.abs(
        shifted.loc[human.index, "predicted_target"]
        - human["predicted_target"]
    )
    table["roi_mismatch"] = 1.0 - table["human_roi_coverage"]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    for axis, column, xlabel in (
        (axes[0], "cycle_phase_residual_frames", "Cycle-phase residual (frames)"),
        (axes[1], "roi_mismatch", "Uncovered fraction of human ROI"),
    ):
        axis.scatter(
            table[column],
            table["absolute_prediction_shift_nm"],
            c=100.0 * shifted.loc[table.index, "confidence"],
            cmap="viridis",
            vmin=0,
            vmax=100,
            s=48,
            edgecolor="black",
            linewidth=0.5,
        )
        for group, row in table.nlargest(3, "absolute_prediction_shift_nm").iterrows():
            axis.annotate(group, (row[column], row["absolute_prediction_shift_nm"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rho = spearmanr(table[column], table["absolute_prediction_shift_nm"])
        axis.text(
            0.04,
            0.95,
            f"Spearman ρ={rho.statistic:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("|human→auto − human→human| Rq (nm)")
        axis.grid(alpha=0.18)
    figure.suptitle("Which automatic-input differences move the M14i prediction?")
    _save_figure(figure, figure_root / "Fig3_input_shift_diagnostics")
    write_csv(table.reset_index(), figure_root.parent / "input_shift_diagnostics.csv")


def _plot_clip_atlas(config: dict[str, Any], selection: pd.DataFrame, figure_root: Path) -> None:
    groups = selection["growth_run_id"].astype(str).tolist()
    human_root = repo_path(config["human_clip_root"])
    auto_root = repo_path(config["output_root"]) / "clip_variants"
    for page, start in enumerate(range(0, len(groups), 6), start=1):
        subset = groups[start : start + 6]
        figure, axes = plt.subplots(
            len(subset), 2, figsize=(5.2, 2.25 * len(subset)), constrained_layout=True
        )
        if len(subset) == 1:
            axes = np.asarray([axes])
        for row_index, group in enumerate(subset):
            metadata = selection.set_index("growth_run_id").loc[group]
            human = np.load(
                human_root / "keyframe_1" / f"{group}.npz",
                allow_pickle=False,
            )["frames_uint8"][0]
            auto = np.load(
                auto_root / "keyframe_1" / f"{group}.npz",
                allow_pickle=False,
            )["frames_uint8"][0]
            for axis, image, title in (
                (
                    axes[row_index, 0],
                    human,
                    f"{group} human | k={int(metadata['human_keyframe_index'])}",
                ),
                (
                    axes[row_index, 1],
                    auto,
                    (
                        f"{group} auto | k={int(metadata['machine_keyframe_index'])} "
                        f"| phase Δ={metadata['cycle_phase_residual_frames']:.1f}"
                    ),
                ),
            ):
                axis.imshow(_display_contrast(image), cmap="gray", vmin=0, vmax=1)
                axis.set_title(title, fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            (
                "Model-ready 224×224 RHEED input: human versus automatic "
                f"({page}/4; contrast normalized for display only)"
            ),
            fontsize=11,
        )
        _save_figure(figure, figure_root / f"Fig4_model_input_atlas_page_{page:02d}")


def run(config: dict[str, Any]) -> None:
    set_publication_style()
    report_root = repo_path(config["report_root"])
    figure_root = report_root / "figures"
    report_root.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(
        Path(config["standalone_target_parameters"])
        .expanduser()
        .read_text(encoding="utf-8")
    )
    generator_config_path = config.get(
        "generation_config_override",
        config["standalone_generator_parameters"],
    )
    generator_config = json.loads(
        Path(generator_config_path).expanduser().read_text(encoding="utf-8")
    )
    tables = _load_tables(generator_config)
    descriptors, _ = prepare_full_cohort(tables, generator_config)
    metrics = group_metric_table(
        scan_metric_table(
            descriptors,
            scan_size_nm=float(generator_config["scan_size_nm"]),
            analysis_scale_nm=float(generator_config["analysis_scale_nm"]),
        )
    )
    log_rq, log_fsmi = _target_series(descriptors, metrics)
    groups = list(map(str, log_rq.index))
    human_physics = _physics_table(
        pd.read_csv(
            repo_path(config["human_physics_features"]),
            dtype={"sample_id": str, "growth_run_id": str},
        )
    ).loc[groups]
    auto_root = repo_path(config["output_root"])
    auto_physics = _physics_table(
        pd.read_csv(
            auto_root / "rheed_physics_features.csv",
            dtype={"sample_id": str, "growth_run_id": str},
        )
    ).loc[groups]
    embedding_id = str(parameters["temporal_embedding_id"])
    human_embeddings = _embedding_frame(
        config["human_embedding_registry"], embedding_id
    ).loc[groups]
    auto_embeddings = _embedding_frame(
        auto_root / "embedding_registry.csv", embedding_id
    ).loc[groups]
    candidate_config = _candidate_config(parameters)

    all_predictions: list[pd.DataFrame] = []
    machine_robust_frames: list[pd.DataFrame] = []
    for target, log_target, method, human_path in (
        ("Rq_nm", log_rq, MULTIVIEW_60, config["human_rq_predictions"]),
        ("FSMI_nm", log_fsmi, DENSITY_WEIGHTED, config["human_fsmi_predictions"]),
    ):
        human = pd.read_csv(
            repo_path(human_path), dtype={"growth_run_id": str}
        )
        human.insert(0, "target", target)
        human["protocol"] = PROTOCOL_HUMAN
        human["training_selection_source"] = "human"
        human["query_selection_source"] = "human"
        all_predictions.append(human)

        fixed, _, _ = crossfit_robust_candidates(
            physics=auto_physics,
            embeddings=auto_embeddings,
            log_target=log_target,
            config=candidate_config,
            confidence_alpha=float(parameters["confidence_alpha"]),
        )
        fixed.insert(0, "target", target)
        machine_robust_frames.append(fixed)
        auto = fixed.loc[fixed["method"] == method].copy()
        auto["selected_candidate"] = method
        auto["method"] = FINAL_TARGET_SPECIFIC
        auto["protocol"] = PROTOCOL_AUTO
        auto["training_selection_source"] = "automatic_v5_v8"
        auto["query_selection_source"] = "automatic_v5_v8"
        machine_robust_frames.append(auto.copy())
        all_predictions.append(auto)

        shift = _cross_domain_target(
            human_physics=human_physics,
            human_embeddings=human_embeddings,
            auto_physics=auto_physics,
            auto_embeddings=auto_embeddings,
            log_target=log_target,
            method=method,
            config=candidate_config,
            confidence_alpha=float(parameters["confidence_alpha"]),
        )
        shift.insert(0, "target", target)
        shift["protocol"] = PROTOCOL_SHIFT
        all_predictions.append(shift)

    predictions = pd.concat(all_predictions, ignore_index=True)
    machine_robust = pd.concat(machine_robust_frames, ignore_index=True)
    write_csv(predictions, report_root / "paired_target_predictions.csv")
    write_csv(
        machine_robust, report_root / "machine_robust_crossfit_predictions.csv"
    )
    for target, filename in (
        ("Rq_nm", "machine_rq_selected_predictions.csv"),
        ("FSMI_nm", "machine_fsmi_selected_predictions.csv"),
    ):
        write_csv(
            predictions.loc[
                (predictions["target"] == target)
                & (predictions["protocol"] == PROTOCOL_AUTO)
            ].drop(columns=["target", "protocol"]),
            report_root / filename,
        )
    metric_table = pd.DataFrame(
        [
            _metric_row(rows, target=target, protocol=protocol)
            for (target, protocol), rows in predictions.groupby(
                ["target", "protocol"], sort=False
            )
        ]
    )
    write_csv(metric_table, report_root / "protocol_metrics.csv")
    selection = pd.read_csv(
        auto_root / "selection_comparison.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    _plot_target_protocols(predictions, figure_root)
    _plot_rq_ordered(predictions, figure_root)
    _plot_input_shift(predictions, selection, figure_root)
    _plot_clip_atlas(config, selection, figure_root)
    manifest = {
        "experiment_id": config["experiment_id"],
        "growth_group_count": len(groups),
        "protocols": {
            PROTOCOL_HUMAN: (
                "frozen original M14i outer LOO; fit 22 human-selected inputs"
            ),
            PROTOCOL_AUTO: (
                "same M14i methods/hyperparameters; fit 22 automatic inputs "
                "and predict one held automatic input"
            ),
            PROTOCOL_SHIFT: (
                "fit 22 human-selected inputs; predict the held growth from "
                "its automatic input; inner calibration uses only the 22 "
                "human training growths"
            ),
        },
        "outer_target_used_for_training": False,
        "retrospective_method_development": True,
        "selection_algorithm_previously_developed_on_annotations": True,
        "standalone_modified": False,
        "raw_data_modified": False,
        "metrics": metric_table.to_dict(orient="records"),
    }
    write_json(manifest, report_root / "comparison_manifest.json")
    print(metric_table.to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_manual_vs_auto_selection.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
