from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_video_afm_story.common import write_csv, write_json
from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    predict_groups,
)
from analysis.rheed_to_afm_generation.training import resolve_device

from .adversarial import calibrate_random_fields
from .evaluation import condition_permutation_control, evaluate_method_sets
from .rheed import _fit_hybrid_candidate, _fit_pls_candidate
from .spectral import fit_conditional_spectral_model


METHOD_MEAN = "M0_mean_condition_calibrated_spectral"
METHOD_SPECTRAL = "M2_spectral_rheed_condition"
METHOD_CALIBRATED = "M2b_calibrated_spectral_rheed_condition"


def _generate(
    model: Any,
    condition: np.ndarray,
    *,
    draws: int,
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    return [
        model.generate(
            condition,
            seed=seed + draw,
            iterations=iterations,
        )
        for draw in range(draws)
    ]


def _calibrate(
    fields: list[np.ndarray],
    condition: np.ndarray,
    *,
    scaler: ConditionScaler,
    config: dict[str, Any],
    device: Any,
) -> list[np.ndarray]:
    values, _ = calibrate_random_fields(
        np.stack(fields),
        condition,
        condition_scaler=scaler,
        device=device,
        steps=int(config["descriptor_calibration_steps"]),
        learning_rate=float(config["descriptor_calibration_learning_rate"]),
        content_weight=float(config["descriptor_calibration_content_weight"]),
    )
    return [array for array in values]


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_rows in frame.groupby("method"):
        row: dict[str, Any] = {
            "method": method,
            "held_out_growth_group_count": int(
                method_rows["growth_run_id"].nunique()
            ),
            "texture_gate_pass_fraction": float(
                method_rows["afm_texture_gate_pass"].mean()
            ),
        }
        for column in (
            "rq_absolute_error_nm",
            "condition_descriptor_mae_z",
            "normalized_psd_log_distance",
            "sharpness_ratio",
            "laplacian_rms_relative_error",
            "generated_pairwise_l1",
            "nearest_training_l1",
            "max_training_ssim",
        ):
            row[f"median_{column}"] = float(
                method_rows[column].astype(float).median()
            )
            row[f"iqr_{column}"] = float(
                method_rows[column].astype(float).quantile(0.75)
                - method_rows[column].astype(float).quantile(0.25)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["texture_gate_pass_fraction", "median_condition_descriptor_mae_z"],
        ascending=[False, True],
    )


def _correlation_table(
    predictions: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        truth = predictions[f"true__{column}"].to_numpy(float)
        prediction = predictions[f"predicted__{column}"].to_numpy(float)
        if np.std(truth) <= 1e-12 or np.std(prediction) <= 1e-12:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = float(pearsonr(truth, prediction).statistic)
            spearman = float(spearmanr(truth, prediction).statistic)
        rows.append(
            {
                "descriptor": column,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "mae_raw": float(np.mean(np.abs(prediction - truth))),
            }
        )
    return pd.DataFrame(rows)


def _save_figures(
    *,
    predictions: pd.DataFrame,
    correlations: pd.DataFrame,
    metrics: pd.DataFrame,
    control: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    true_rq = np.exp(predictions["true__log_rq_nm"].to_numpy(float))
    predicted_rq = np.exp(
        predictions["predicted__log_rq_nm"].to_numpy(float)
    )
    limits = (
        float(min(true_rq.min(), predicted_rq.min())),
        float(max(true_rq.max(), predicted_rq.max())),
    )
    axes[0, 0].scatter(true_rq, predicted_rq, color="#2a9d8f", s=42)
    axes[0, 0].plot(limits, limits, "--", color="black", linewidth=1)
    axes[0, 0].set(
        xlabel="Measured group-median Rq (nm)",
        ylabel="Cross-fitted RHEED prediction (nm)",
        title="RHEED → roughness",
    )

    ordered = predictions.sort_values("rheed_condition_mae_z")
    axes[0, 1].bar(
        ordered["growth_run_id"],
        ordered["rheed_condition_mae_z"],
        color="#457b9d",
    )
    axes[0, 1].set(
        xlabel="Held-out training growth group",
        ylabel="Descriptor MAE (training z)",
        title="RHEED condition error",
    )
    axes[0, 1].tick_params(axis="x", rotation=60)

    positions = np.arange(len(correlations))
    axes[1, 0].bar(
        positions - 0.18,
        correlations["pearson_r"],
        width=0.36,
        label="Pearson",
        color="#e9c46a",
    )
    axes[1, 0].bar(
        positions + 0.18,
        correlations["spearman_rho"],
        width=0.36,
        label="Spearman",
        color="#f4a261",
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(
        [
            value.replace("log_unit_", "").replace("unit_", "")
            for value in correlations["descriptor"]
        ],
        rotation=55,
        ha="right",
        fontsize=8,
    )
    axes[1, 0].set(
        ylabel="Correlation",
        ylim=(-1.05, 1.05),
        title="Cross-fitted descriptor correlation",
    )
    axes[1, 0].legend(fontsize=8)

    method_order = [METHOD_MEAN, METHOD_SPECTRAL, METHOD_CALIBRATED]
    values = [
        metrics.loc[metrics["method"] == method, "sharpness_ratio"].to_numpy(
            float
        )
        for method in method_order
    ]
    axes[1, 1].boxplot(
        values,
        tick_labels=["Mean condition", "Spectral", "Calibrated spectral"],
        showmeans=True,
    )
    axes[1, 1].axhspan(0.65, 1.65, color="#2a9d8f", alpha=0.12)
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    win_fraction = float(control["correct_condition_wins"].mean())
    axes[1, 1].set(
        ylabel="Generated / measured mean gradient",
        title=f"AFM sharpness; condition wins {win_fraction:.0%}",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Leave-one-growth-group-out development audit (15 groups)",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "Fig7_training_group_cross_validation.png",
        dpi=240,
        bbox_inches="tight",
    )
    figure.savefig(
        figure_dir / "Fig7_training_group_cross_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def run_training_group_cross_validation(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    selected_predictor: Any,
    output_dir: str | Path,
    report_dir: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """Cross-fit the complete RHEED→spectral generator on 15 train groups.

    Model family and hyperparameters are already frozen from the disjoint
    validation partition. Each row below refits all transforms, the RHEED
    predictor, and the AFM spectral model without the held-out growth group.
    """

    output = Path(output_dir)
    report = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    all_rows = tables["descriptors"]
    development_rows = all_rows.loc[all_rows["split"] == "train"].copy()
    groups = sorted(development_rows["growth_run_id"].astype(str).unique())
    if len(groups) < 5:
        raise ValueError("insufficient training groups for cross-validation")
    selected_family = getattr(selected_predictor, "model_family", "")
    if selected_family not in {
        "PLSRegression",
        "HybridSVRRq_PLSMorphology",
    }:
        raise ValueError(
            f"unsupported cross-validation predictor: {selected_family}"
        )
    device = resolve_device(device_name)
    draws = int(config.get("cross_validation_draws", 4))
    iterations = int(config["spectral_iaaft_iterations"])
    metric_frames: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, held_group in enumerate(groups):
        fit_groups = set(groups) - {held_group}
        train_rows = development_rows.loc[
            development_rows["growth_run_id"].astype(str).isin(fit_groups)
        ].copy()
        held_rows = development_rows.loc[
            development_rows["growth_run_id"].astype(str) == held_group
        ].copy()
        scaler = ConditionScaler.fit(
            development_rows,
            list(config["condition_columns"]),
            fit_groups,
        )
        group_targets = development_rows.groupby("growth_run_id")[
            scaler.columns
        ].median()
        if selected_family == "HybridSVRRq_PLSMorphology":
            predictor = _fit_hybrid_candidate(
                morphology_embedding_id=str(
                    selected_predictor.morphology_predictor.embedding_id
                ),
                roughness_embedding_id=str(
                    selected_predictor.roughness_predictor.embedding_id
                ),
                morphology_pls_components=int(
                    selected_predictor.pls_components
                ),
                embedding_registry=tables["registry"],
                physics_table=tables["physics"],
                group_targets=group_targets,
                condition_scaler=scaler,
                train_groups=sorted(fit_groups),
                pca_dim=int(config["pca_dim"]),
                excluded_sample_ids=set(
                    tables["removelist"].sample_ids
                ),
            )
        else:
            predictor, _ = _fit_pls_candidate(
                embedding_id=str(selected_predictor.embedding_id),
                components=int(getattr(selected_predictor, "pls_components")),
                embedding_registry=tables["registry"],
                physics_table=tables["physics"],
                group_targets=group_targets,
                condition_scaler=scaler,
                train_groups=sorted(fit_groups),
                pca_dim=int(config["pca_dim"]),
                excluded_sample_ids=set(
                    tables["removelist"].sample_ids
                ),
            )
        predicted_raw, predicted_z, _ = predict_groups(
            predictor,
            [held_group],
            tables["registry"],
            tables["physics"],
        )
        predicted_raw = predicted_raw[0]
        predicted_z = predicted_z[0]
        true_raw = held_rows[scaler.columns].median().to_numpy(float)
        true_z = scaler.transform(true_raw[None], clip=False)[0]
        prediction_record: dict[str, Any] = {
            "growth_run_id": held_group,
            "rheed_condition_mae_z": float(
                np.mean(np.abs(predicted_z - true_z))
            ),
        }
        for position, column in enumerate(scaler.columns):
            prediction_record[f"true__{column}"] = float(true_raw[position])
            prediction_record[f"predicted__{column}"] = float(
                predicted_raw[position]
            )
        prediction_rows.append(prediction_record)

        spectral_model, _, _ = fit_conditional_spectral_model(
            train_rows=train_rows,
            condition_scaler=scaler,
            alphas=[float(value) for value in config["spectral_ridge_alphas"]],
            resolution=int(config["resolution"]),
            removelist_sample_ids=tables["removelist"].sample_ids,
        )
        fold_seed = int(config["seed"]) + 700_000 + fold_index * 10_000
        correct = _generate(
            spectral_model,
            predicted_z,
            draws=draws,
            iterations=iterations,
            seed=fold_seed,
        )
        calibrated = _calibrate(
            correct,
            predicted_z,
            scaler=scaler,
            config=config,
            device=device,
        )
        mean_condition = np.zeros_like(predicted_z)
        mean_fields = _generate(
            spectral_model,
            mean_condition,
            draws=draws,
            iterations=iterations,
            seed=fold_seed,
        )
        mean_calibrated = _calibrate(
            mean_fields,
            mean_condition,
            scaler=scaler,
            config=config,
            device=device,
        )
        predicted_rq = float(
            np.exp(predicted_raw[scaler.columns.index("log_rq_nm")])
        )
        mean_rq = float(
            np.exp(
                scaler.mean[scaler.columns.index("log_rq_nm")]
            )
        )
        generated = {
            METHOD_MEAN: {held_group: mean_calibrated},
            METHOD_SPECTRAL: {held_group: correct},
            METHOD_CALIBRATED: {held_group: calibrated},
        }
        generated_rq = {
            METHOD_MEAN: {held_group: mean_rq},
            METHOD_SPECTRAL: {held_group: predicted_rq},
            METHOD_CALIBRATED: {held_group: predicted_rq},
        }
        fold_report = report / "folds" / held_group
        evaluation = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=train_rows,
            condition_scaler=scaler,
            generated=generated,
            generated_rq=generated_rq,
            output_dir=fold_report,
            resolution=int(config["resolution"]),
        )
        frame = evaluation["per_group"].copy()
        frame.insert(0, "cross_validation_fold", fold_index)
        metric_frames.append(frame)

        wrong_group = groups[(fold_index + 1) % len(groups)]
        _, wrong_z, _ = predict_groups(
            predictor,
            [wrong_group],
            tables["registry"],
            tables["physics"],
        )
        wrong = _generate(
            spectral_model,
            wrong_z[0],
            draws=draws,
            iterations=iterations,
            seed=fold_seed,
        )
        wrong_calibrated = _calibrate(
            wrong,
            wrong_z[0],
            scaler=scaler,
            config=config,
            device=device,
        )
        control = condition_permutation_control(
            groups=[held_group],
            split_rows=held_rows,
            condition_scaler=scaler,
            correct_maps={held_group: calibrated},
            wrong_maps={held_group: wrong_calibrated},
            generated_rq={held_group: predicted_rq},
        )
        control["wrong_condition_source_group"] = wrong_group
        controls.append(control)
        fold_output = output / "generated_maps" / held_group
        fold_output.mkdir(parents=True, exist_ok=True)
        for method, arrays in (
            (METHOD_MEAN, mean_calibrated),
            (METHOD_SPECTRAL, correct),
            (METHOD_CALIBRATED, calibrated),
        ):
            np.savez_compressed(
                fold_output / f"{method}.npz",
                generated_unit_shapes=np.stack(arrays).astype(np.float32),
                growth_run_id=np.asarray(held_group),
                method=np.asarray(method),
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    control = pd.concat(controls, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    summary = _summary_table(metrics)
    correlations = _correlation_table(
        predictions, list(config["condition_columns"])
    )
    write_csv(metrics, report / "per_group_metrics.csv")
    write_csv(summary, report / "method_summary.csv")
    write_csv(control, report / "condition_permutation_control.csv")
    write_csv(predictions, report / "rheed_descriptor_predictions.csv")
    write_csv(correlations, report / "descriptor_correlations.csv")
    _save_figures(
        predictions=predictions,
        correlations=correlations,
        metrics=metrics,
        control=control,
        figure_dir=report / "figures",
    )
    manifest = {
        "protocol": "leave_one_growth_group_out_on_training_partition",
        "held_out_growth_group_count": len(groups),
        "growth_groups": groups,
        "validation_partition_used_for_family_selection_only": True,
        "old_test_partition_used": False,
        "refit_inside_each_fold": [
            "condition scaler",
            "RHEED embedding scaler",
            "RHEED PCA",
            "RHEED PLS regression",
            "RHEED roughness SVR head",
            "AFM spectral output scaler",
            "AFM conditional spectral ridge",
        ],
        "selected_embedding_id": selected_predictor.embedding_id,
        "selected_model_family": selected_family,
        "selected_pls_components": int(
            getattr(selected_predictor, "pls_components")
        ),
        "draws_per_group": draws,
        "method_summary": summary.to_dict(orient="records"),
        "rheed_calibrated_beats_mean_condition_descriptor": bool(
            summary.set_index("method").loc[
                METHOD_CALIBRATED,
                "median_condition_descriptor_mae_z",
            ]
            < summary.set_index("method").loc[
                METHOD_MEAN,
                "median_condition_descriptor_mae_z",
            ]
        ),
        "rheed_calibrated_beats_mean_rq": bool(
            summary.set_index("method").loc[
                METHOD_CALIBRATED,
                "median_rq_absolute_error_nm",
            ]
            < summary.set_index("method").loc[
                METHOD_MEAN,
                "median_rq_absolute_error_nm",
            ]
        ),
        "condition_permutation_win_fraction": float(
            control["correct_condition_wins"].mean()
        ),
        "median_rheed_condition_mae_z": float(
            predictions["rheed_condition_mae_z"].median()
        ),
        "removelist_sample_ids": list(tables["removelist"].sample_ids),
        "removelist_overlap": [],
    }
    write_json(manifest, report / "cross_validation_manifest.json")
    return manifest
