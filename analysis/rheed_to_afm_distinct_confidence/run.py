from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    aggregate_group_conditions,
    predict_groups,
)
from analysis.rheed_to_afm_generation.run import _load_tables, _split_manifest
from analysis.rheed_to_afm_sharp_generation.evaluation import evaluate_method_sets
from analysis.rheed_to_afm_sharp_generation.rheed import _fit_hybrid_candidate
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .matern import DescriptorMaternGenerator
from .uncertainty import (
    interval_width_z,
    jackknife_plus_interval,
    relative_confidence_index,
)
from .variance import (
    condition_prediction_metrics,
    fit_variance_calibrator,
)

RAW_METHOD = "M4a_descriptor_matern_raw"
MATERN_METHOD = "M4b_variance_calibrated_descriptor_matern"
PRIOR_METHOD = "M2b_prior_calibrated_spectral"
SELECTED_METHOD = "M5_multiscale_spectral_hybrid"


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _group_targets(descriptors: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return aggregate_group_conditions(descriptors, columns)


def _predictor_factory(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    group_targets: pd.DataFrame,
) -> Callable[[list[str], ConditionScaler], object]:
    excluded = set(tables["removelist"].sample_ids)

    def fit(groups: list[str], scaler: ConditionScaler) -> object:
        return _fit_hybrid_candidate(
            morphology_embedding_id=str(config["hybrid_morphology_embedding_id"]),
            roughness_embedding_id=str(config["hybrid_roughness_embedding_id"]),
            morphology_pls_components=1,
            embedding_registry=tables["registry"],
            physics_table=tables["physics"],
            group_targets=group_targets,
            condition_scaler=scaler,
            train_groups=list(map(str, groups)),
            pca_dim=int(config["pca_dim"]),
            excluded_sample_ids=excluded,
        )

    return fit


def _save_generated(
    root: Path,
    method: str,
    group: str,
    arrays: list[np.ndarray],
    *,
    predicted_rq_nm: float,
    condition_z: np.ndarray,
) -> None:
    path = root / "generated_maps" / method / f"{group}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_unit_shapes=np.stack(arrays).astype(np.float32),
        predicted_rq_nm=np.asarray(float(predicted_rq_nm)),
        condition_z=np.asarray(condition_z, dtype=np.float32),
        growth_run_id=np.asarray(group),
        method=np.asarray(method),
        retrieval_at_inference=np.asarray(False),
    )


def _load_prior_validation(
    root: str | Path, groups: list[str]
) -> dict[str, list[np.ndarray]]:
    base = repo_path(root)
    return {
        group: [
            np.asarray(array, dtype=np.float32)
            for array in np.load(base / f"{group}.npz", allow_pickle=False)[
                "generated_unit_shapes"
            ]
        ]
        for group in groups
    }


def _load_prior_crossfit(root: str | Path, group: str) -> list[np.ndarray]:
    path = repo_path(root) / group / "M2b_calibrated_spectral_rheed_condition.npz"
    return [
        np.asarray(array, dtype=np.float32)
        for array in np.load(path, allow_pickle=False)["generated_unit_shapes"]
    ]


def _blend_ensembles(
    primary: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    primary_weight: float,
) -> list[np.ndarray]:
    weight = float(np.clip(primary_weight, 0.0, 1.0))
    count = max(len(primary), len(spectral))
    return [
        project_unit_rq_np(
            weight * primary[index % len(primary)]
            + (1.0 - weight) * spectral[index % len(spectral)]
        ).astype(np.float32)
        for index in range(count)
    ]


def _aggregate_fold_evaluations(
    fold_frames: list[pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.concat(fold_frames, ignore_index=True)
    summaries: list[dict[str, Any]] = []
    for method, rows in frame.groupby("method"):
        summary: dict[str, Any] = {
            "method": method,
            "held_out_growth_group_count": int(len(rows)),
            "texture_gate_pass_fraction": float(rows["afm_texture_gate_pass"].mean()),
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
            "composite_score",
        ):
            summary[f"median_{column}"] = float(rows[column].median())
            summary[f"iqr_{column}"] = float(
                rows[column].quantile(0.75) - rows[column].quantile(0.25)
            )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["texture_gate_pass_fraction", "median_rq_absolute_error_nm"],
        ascending=[False, True],
    )
    write_csv(frame, output_dir / "per_group_metrics.csv")
    write_csv(summary_frame, output_dir / "method_summary.csv")
    return frame, summary_frame


def _run_crossfit_generation(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    train_rows: pd.DataFrame,
    group_targets: pd.DataFrame,
    fit_predictor: Callable[[list[str], ConditionScaler], object],
    output_root: Path,
    report_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    caps = [float(value) for value in config["variance_caps"]]
    selected_cap = float(config["selected_variance_cap"])
    prediction_records: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    panels: dict[str, dict[str, Any]] = {}
    raw_truths: list[np.ndarray] = []
    z_truths: list[np.ndarray] = []
    by_cap_raw: dict[float, list[np.ndarray]] = {cap: [] for cap in caps}
    by_cap_z: dict[float, list[np.ndarray]] = {cap: [] for cap in caps}
    draws = 2 if smoke else int(config["crossfit_draws"])
    for fold, held in enumerate(groups):
        fit_groups = [group for group in groups if group != held]
        scaler = ConditionScaler.fit(
            train_rows, list(config["condition_columns"]), set(fit_groups)
        )
        calibrator, inner_predictions = fit_variance_calibrator(
            groups=fit_groups,
            group_targets=group_targets,
            enclosing_scaler=scaler,
            fit_predictor=fit_predictor,
            registry=tables["registry"],
            physics=tables["physics"],
            cap=max(caps),
            minimum_predicted_std=float(config["minimum_predicted_std"]),
        )
        predictor = fit_predictor(fit_groups, scaler)
        raw, z, _ = predict_groups(
            predictor, [held], tables["registry"], tables["physics"]
        )
        true_raw = group_targets.loc[held, scaler.columns].to_numpy(float)
        true_z = scaler.transform(true_raw[None], clip=False)[0]
        raw_truths.append(true_raw)
        z_truths.append(true_z)
        cap_conditions: dict[float, np.ndarray] = {}
        for cap in caps:
            factor = np.clip(calibrator.factors, 1.0, cap)
            calibrated_z = (z[0] * factor).astype(np.float32)
            calibrated_raw = scaler.inverse_transform(calibrated_z[None])[0]
            by_cap_z[cap].append(calibrated_z)
            by_cap_raw[cap].append(calibrated_raw)
            cap_conditions[cap] = calibrated_z
        raw_z = cap_conditions[1.0]
        selected_z = cap_conditions[selected_cap]
        rq_index = scaler.columns.index("log_rq_nm")
        raw_rq = float(np.exp(scaler.inverse_transform(raw_z[None])[0, rq_index]))
        selected_rq = float(
            np.exp(scaler.inverse_transform(selected_z[None])[0, rq_index])
        )
        generator = DescriptorMaternGenerator(
            scaler, resolution=int(config["resolution"])
        )
        seed = int(config["seed"]) + fold * 10_000
        generated = {
            RAW_METHOD: {
                held: generator.generate_ensemble(raw_z, draws=draws, seed=seed)
            },
            MATERN_METHOD: {
                held: generator.generate_ensemble(selected_z, draws=draws, seed=seed)
            },
        }
        prior_arrays = _load_prior_crossfit(config["prior_m2b_crossfit_output"], held)
        prior_arrays = prior_arrays[:draws]
        generated[PRIOR_METHOD] = {held: prior_arrays}
        generated[SELECTED_METHOD] = {
            held: _blend_ensembles(
                generated[MATERN_METHOD][held],
                prior_arrays,
                primary_weight=float(config["matern_blend_weight"]),
            )
        }
        rq = {
            RAW_METHOD: {held: raw_rq},
            MATERN_METHOD: {held: selected_rq},
            PRIOR_METHOD: {held: raw_rq},
            SELECTED_METHOD: {held: selected_rq},
        }
        for method in (RAW_METHOD, MATERN_METHOD, SELECTED_METHOD):
            payload = generated[method]
            _save_generated(
                output_root / "training_group_cross_validation",
                method,
                held,
                payload[held],
                predicted_rq_nm=rq[method][held],
                condition_z=raw_z if method == RAW_METHOD else selected_z,
            )
        held_rows = train_rows.loc[train_rows["growth_run_id"].astype(str) == held]
        outer_rows = train_rows.loc[train_rows["growth_run_id"].astype(str) != held]
        evaluation = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=outer_rows,
            condition_scaler=scaler,
            generated=generated,
            generated_rq=rq,
            output_dir=(
                report_root / "training_group_cross_validation" / "folds" / held
            ),
            resolution=int(config["resolution"]),
        )
        fold_frame = evaluation["per_group"].copy()
        fold_frame.insert(0, "cross_validation_fold", fold)
        fold_frames.append(fold_frame)
        panels[held] = evaluation["panels"][held]
        record: dict[str, Any] = {
            "growth_run_id": held,
            "true_rq_nm": float(np.exp(true_raw[rq_index])),
            "raw_predicted_rq_nm": raw_rq,
            "selected_predicted_rq_nm": selected_rq,
            "inner_variance_factors": json.dumps(calibrator.factors.tolist()),
            "inner_group_count": len(inner_predictions),
        }
        for position, column in enumerate(scaler.columns):
            record[f"true_z__{column}"] = float(true_z[position])
            record[f"raw_predicted_z__{column}"] = float(raw_z[position])
            record[f"selected_predicted_z__{column}"] = float(selected_z[position])
            record[f"true_raw__{column}"] = float(true_raw[position])
            record[f"raw_predicted_raw__{column}"] = float(
                scaler.inverse_transform(raw_z[None])[0, position]
            )
            record[f"selected_predicted_raw__{column}"] = float(
                scaler.inverse_transform(selected_z[None])[0, position]
            )
        prediction_records.append(record)
    metrics_dir = report_root / "training_group_cross_validation"
    per_group, method_summary = _aggregate_fold_evaluations(fold_frames, metrics_dir)
    truth_raw_array = np.stack(raw_truths)
    truth_z_array = np.stack(z_truths)
    cap_rows = []
    for cap in caps:
        predicted_raw = np.stack(by_cap_raw[cap])
        predicted_z = np.stack(by_cap_z[cap])
        cap_rows.append(
            {
                "variance_cap": cap,
                **condition_prediction_metrics(
                    truth_z=truth_z_array,
                    predicted_z=predicted_z,
                    truth_raw=truth_raw_array,
                    predicted_raw=predicted_raw,
                    columns=list(config["condition_columns"]),
                ),
                "selected": cap == selected_cap,
            }
        )
    cap_ablation = pd.DataFrame(cap_rows)
    predictions = pd.DataFrame(prediction_records)
    write_csv(predictions, metrics_dir / "condition_predictions.csv")
    write_csv(cap_ablation, metrics_dir / "variance_cap_ablation.csv")
    return {
        "per_group": per_group,
        "summary": method_summary,
        "panels": panels,
        "condition_predictions": predictions,
        "cap_ablation": cap_ablation,
    }


def _run_validation(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    group_targets: pd.DataFrame,
    fit_predictor: Callable[[list[str], ConditionScaler], object],
    output_root: Path,
    report_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    train_groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    groups = sorted(validation_rows["growth_run_id"].astype(str).unique())
    scaler = ConditionScaler.fit(
        train_rows, list(config["condition_columns"]), set(train_groups)
    )
    calibrator, inner = fit_variance_calibrator(
        groups=train_groups,
        group_targets=group_targets,
        enclosing_scaler=scaler,
        fit_predictor=fit_predictor,
        registry=tables["registry"],
        physics=tables["physics"],
        cap=float(config["selected_variance_cap"]),
        minimum_predicted_std=float(config["minimum_predicted_std"]),
    )
    predictor = fit_predictor(train_groups, scaler)
    raw, z, _ = predict_groups(predictor, groups, tables["registry"], tables["physics"])
    calibrated_z = calibrator.transform_z(z)
    calibrated_raw = scaler.inverse_transform(calibrated_z)
    raw_by_group = dict(zip(groups, raw, strict=False))
    z_by_group = dict(zip(groups, z, strict=False))
    selected_raw_by_group = dict(zip(groups, calibrated_raw, strict=False))
    selected_z_by_group = dict(zip(groups, calibrated_z, strict=False))
    generator = DescriptorMaternGenerator(scaler, resolution=int(config["resolution"]))
    draws = 2 if smoke else int(config["draws"])
    generated: dict[str, dict[str, list[np.ndarray]]] = {
        RAW_METHOD: {},
        MATERN_METHOD: {},
        PRIOR_METHOD: _load_prior_validation(config["prior_m2b_output"], groups),
        SELECTED_METHOD: {},
    }
    rq_index = scaler.columns.index("log_rq_nm")
    rq: dict[str, dict[str, float]] = {
        RAW_METHOD: {},
        MATERN_METHOD: {},
        PRIOR_METHOD: {},
        SELECTED_METHOD: {},
    }
    for index, group in enumerate(groups):
        seed = int(config["seed"]) + 500_000 + index * 10_000
        coupled_seed = int(config["seed"]) + 900_000
        generated[RAW_METHOD][group] = generator.generate_ensemble(
            z_by_group[group], draws=draws, seed=seed
        )
        generated[MATERN_METHOD][group] = generator.generate_ensemble(
            selected_z_by_group[group], draws=draws, seed=seed
        )
        # First draw is coupled across groups for a direct condition-response
        # visual control; remaining draws preserve independent ensembles.
        generated[RAW_METHOD][group][0] = generator.generate(
            z_by_group[group], seed=coupled_seed
        )
        generated[MATERN_METHOD][group][0] = generator.generate(
            selected_z_by_group[group], seed=coupled_seed
        )
        rq[RAW_METHOD][group] = float(np.exp(raw_by_group[group][rq_index]))
        rq[MATERN_METHOD][group] = float(np.exp(selected_raw_by_group[group][rq_index]))
        # Prior M2b uses the same raw RHEED condition predictor.
        rq[PRIOR_METHOD][group] = rq[RAW_METHOD][group]
        generated[SELECTED_METHOD][group] = _blend_ensembles(
            generated[MATERN_METHOD][group],
            generated[PRIOR_METHOD][group],
            primary_weight=float(config["matern_blend_weight"]),
        )
        rq[SELECTED_METHOD][group] = rq[MATERN_METHOD][group]
        for method in (RAW_METHOD, MATERN_METHOD, SELECTED_METHOD):
            condition = (
                z_by_group[group]
                if method == RAW_METHOD
                else selected_z_by_group[group]
            )
            _save_generated(
                output_root,
                method,
                group,
                generated[method][group],
                predicted_rq_nm=rq[method][group],
                condition_z=condition,
            )
    evaluation = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=scaler,
        generated=generated,
        generated_rq=rq,
        output_dir=report_root / "validation_evaluation",
        resolution=int(config["resolution"]),
    )
    prediction_rows = []
    for index, group in enumerate(groups):
        truth = group_targets.loc[group, scaler.columns].to_numpy(float)
        truth_z = scaler.transform(truth[None], clip=False)[0]
        prediction_rows.append(
            {
                "growth_run_id": group,
                "condition_error_raw_z": float(np.mean(np.abs(z[index] - truth_z))),
                "condition_error_selected_z": float(
                    np.mean(np.abs(calibrated_z[index] - truth_z))
                ),
                "true_rq_nm": float(np.exp(truth[rq_index])),
                "raw_predicted_rq_nm": rq[RAW_METHOD][group],
                "selected_predicted_rq_nm": rq[SELECTED_METHOD][group],
                **{
                    f"true_z__{column}": float(truth_z[position])
                    for position, column in enumerate(scaler.columns)
                },
                **{
                    f"raw_predicted_z__{column}": float(z[index, position])
                    for position, column in enumerate(scaler.columns)
                },
                **{
                    f"selected_predicted_z__{column}": float(
                        calibrated_z[index, position]
                    )
                    for position, column in enumerate(scaler.columns)
                },
            }
        )
    write_csv(
        pd.DataFrame(prediction_rows),
        report_root / "validation_condition_predictions.csv",
    )
    write_csv(inner, report_root / "full_train_inner_loo_predictions.csv")
    joblib.dump(predictor, output_root / "rheed_descriptor_predictor.joblib")
    write_json(calibrator.to_dict(), output_root / "variance_calibrator.json")
    return {
        **evaluation,
        "scaler": scaler,
        "calibrator": calibrator,
        "predictor": predictor,
        "groups": groups,
        "raw_z": z_by_group,
        "selected_z": selected_z_by_group,
        "rq": rq,
    }


def _cvplus_query(
    *,
    calibration_groups: list[str],
    query_groups: list[str],
    enclosing_scaler: ConditionScaler,
    group_targets: pd.DataFrame,
    fit_predictor: Callable[[list[str], ConditionScaler], object],
    tables: dict[str, Any],
    alpha: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    calibration_prediction: list[np.ndarray] = []
    calibration_truth: list[np.ndarray] = []
    query_predictions: dict[str, list[np.ndarray]] = {
        group: [] for group in query_groups
    }
    for excluded in calibration_groups:
        fit_groups = [group for group in calibration_groups if group != excluded]
        scaler = ConditionScaler.fit(
            group_targets.reset_index(),
            enclosing_scaler.columns,
            set(fit_groups),
        )
        predictor = fit_predictor(fit_groups, scaler)
        queries = [excluded, *query_groups]
        raw, _, _ = predict_groups(
            predictor, queries, tables["registry"], tables["physics"]
        )
        transformed = enclosing_scaler.transform(raw, clip=False)
        calibration_prediction.append(transformed[0])
        true_raw = group_targets.loc[excluded, enclosing_scaler.columns].to_numpy(float)
        calibration_truth.append(
            enclosing_scaler.transform(true_raw[None], clip=False)[0]
        )
        for position, group in enumerate(query_groups, start=1):
            query_predictions[group].append(transformed[position])
    predicted = np.stack(calibration_prediction)
    truth = np.stack(calibration_truth)
    return {
        group: jackknife_plus_interval(
            np.stack(values),
            predicted,
            truth,
            alpha=alpha,
        )
        for group, values in query_predictions.items()
    }


def _run_uncertainty(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    train_rows: pd.DataFrame,
    validation: dict[str, Any],
    group_targets: pd.DataFrame,
    fit_predictor: Callable[[list[str], ConditionScaler], object],
    crossfit_predictions: pd.DataFrame,
    report_root: Path,
    smoke: bool,
) -> dict[str, pd.DataFrame]:
    train_groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    alpha = float(config["confidence_alpha"])
    audit_rows: list[dict[str, Any]] = []
    selected_columns = list(config["condition_columns"])
    audit_groups = train_groups[:3] if smoke else train_groups
    crossfit_lookup = crossfit_predictions.set_index("growth_run_id")
    for held in audit_groups:
        calibration_groups = [group for group in train_groups if group != held]
        enclosing_scaler = ConditionScaler.fit(
            train_rows,
            selected_columns,
            set(calibration_groups),
        )
        interval = _cvplus_query(
            calibration_groups=calibration_groups,
            query_groups=[held],
            enclosing_scaler=enclosing_scaler,
            group_targets=group_targets,
            fit_predictor=fit_predictor,
            tables=tables,
            alpha=alpha,
        )[held]
        lower, upper = interval
        point_z = np.asarray(
            [
                crossfit_lookup.loc[held, f"selected_predicted_z__{column}"]
                for column in selected_columns
            ],
            dtype=float,
        )
        lower = np.minimum(lower, point_z)
        upper = np.maximum(upper, point_z)
        truth_raw = group_targets.loc[held, selected_columns].to_numpy(float)
        truth_z = enclosing_scaler.transform(truth_raw[None], clip=False)[0]
        rq_index = selected_columns.index("log_rq_nm")
        lower_raw = enclosing_scaler.inverse_transform(lower[None])[0]
        upper_raw = enclosing_scaler.inverse_transform(upper[None])[0]
        audit_rows.append(
            {
                "growth_run_id": held,
                "interval_width_z": interval_width_z(lower, upper),
                "point_error_z": float(np.mean(np.abs(point_z - truth_z))),
                "component_coverage": float(
                    np.mean((truth_z >= lower) & (truth_z <= upper))
                ),
                "all_components_covered": bool(
                    np.all((truth_z >= lower) & (truth_z <= upper))
                ),
                "true_rq_nm": float(np.exp(truth_raw[rq_index])),
                "predicted_rq_nm": float(
                    crossfit_lookup.loc[held, "selected_predicted_rq_nm"]
                ),
                "rq_interval_lower_nm": float(np.exp(lower_raw[rq_index])),
                "rq_interval_upper_nm": float(np.exp(upper_raw[rq_index])),
                **{
                    f"lower_z__{column}": float(lower[position])
                    for position, column in enumerate(selected_columns)
                },
                **{
                    f"upper_z__{column}": float(upper[position])
                    for position, column in enumerate(selected_columns)
                },
            }
        )
    audit = pd.DataFrame(audit_rows)
    reference_widths = audit["interval_width_z"].to_numpy(float)
    audit["relative_confidence_index"] = np.asarray(
        [
            relative_confidence_index(
                np.asarray([width]),
                reference_widths=np.delete(reference_widths, index),
            )[0]
            for index, width in enumerate(reference_widths)
        ],
        dtype=np.float32,
    )
    full_scaler: ConditionScaler = validation["scaler"]
    validation_intervals = _cvplus_query(
        calibration_groups=train_groups,
        query_groups=validation["groups"],
        enclosing_scaler=full_scaler,
        group_targets=group_targets,
        fit_predictor=fit_predictor,
        tables=tables,
        alpha=alpha,
    )
    validation_rows = []
    for group in validation["groups"]:
        lower, upper = validation_intervals[group]
        point_z = np.asarray(validation["selected_z"][group], dtype=float)
        lower = np.minimum(lower, point_z)
        upper = np.maximum(upper, point_z)
        truth_raw = group_targets.loc[group, selected_columns].to_numpy(float)
        truth_z = full_scaler.transform(truth_raw[None], clip=False)[0]
        rq_index = selected_columns.index("log_rq_nm")
        lower_raw = full_scaler.inverse_transform(lower[None])[0]
        upper_raw = full_scaler.inverse_transform(upper[None])[0]
        validation_rows.append(
            {
                "growth_run_id": group,
                "interval_width_z": interval_width_z(lower, upper),
                "point_error_z": float(np.mean(np.abs(point_z - truth_z))),
                "component_coverage": float(
                    np.mean((truth_z >= lower) & (truth_z <= upper))
                ),
                "all_components_covered": bool(
                    np.all((truth_z >= lower) & (truth_z <= upper))
                ),
                "true_rq_nm": float(np.exp(truth_raw[rq_index])),
                "predicted_rq_nm": float(validation["rq"][SELECTED_METHOD][group]),
                "rq_interval_lower_nm": float(np.exp(lower_raw[rq_index])),
                "rq_interval_upper_nm": float(np.exp(upper_raw[rq_index])),
            }
        )
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame["relative_confidence_index"] = relative_confidence_index(
        validation_frame["interval_width_z"].to_numpy(float),
        reference_widths=reference_widths,
    )
    write_csv(audit, report_root / "uncertainty_training_audit.csv")
    write_csv(
        validation_frame,
        report_root / "uncertainty_validation_predictions.csv",
    )
    rho = spearmanr(audit["interval_width_z"], audit["point_error_z"]).statistic
    confidence_rho = spearmanr(
        audit["relative_confidence_index"], audit["point_error_z"]
    ).statistic
    rq_coverage = (audit["true_rq_nm"] >= audit["rq_interval_lower_nm"]) & (
        audit["true_rq_nm"] <= audit["rq_interval_upper_nm"]
    )
    summary = pd.DataFrame(
        [
            {
                "nominal_component_coverage": 1.0 - alpha,
                "empirical_component_coverage": float(
                    audit["component_coverage"].mean()
                ),
                "all_component_coverage_fraction": float(
                    audit["all_components_covered"].mean()
                ),
                "interval_width_vs_error_spearman": float(rho),
                "confidence_vs_error_spearman": float(confidence_rho),
                "rq_interval_empirical_coverage": float(rq_coverage.mean()),
                "confidence_index_maximum": float(
                    audit["relative_confidence_index"].max()
                ),
                "training_audit_group_count": int(len(audit)),
                "confidence_is_probability": False,
                "method": (
                    "growth-group CV+/Jackknife+ component intervals; relative "
                    "ranking enters only as a small adjustment to an absolute "
                    "exponential confidence penalty for interval width"
                ),
            }
        ]
    )
    write_csv(summary, report_root / "uncertainty_summary.csv")
    return {
        "audit": audit,
        "validation": validation_frame,
        "summary": summary,
    }


def _run_learning_curve(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    train_rows: pd.DataFrame,
    group_targets: pd.DataFrame,
    fit_predictor: Callable[[list[str], ConditionScaler], object],
    report_root: Path,
    smoke: bool,
) -> pd.DataFrame:
    groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    sizes = [
        int(value)
        for value in config["learning_curve_group_counts"]
        if int(value) < len(groups)
    ]
    repeats = 1 if smoke else int(config["learning_curve_repeats"])
    records: list[dict[str, Any]] = []
    for held_index, held in enumerate(groups):
        candidates = [group for group in groups if group != held]
        for size in sizes:
            actual_repeats = 1 if size == len(candidates) else repeats
            for repeat in range(actual_repeats):
                rng = np.random.default_rng(
                    int(config["seed"]) + 1009 * held_index + 7919 * size + repeat
                )
                fit_groups = sorted(
                    rng.choice(candidates, size=size, replace=False).tolist()
                )
                scaler = ConditionScaler.fit(
                    train_rows,
                    list(config["condition_columns"]),
                    set(fit_groups),
                )
                predictor = fit_predictor(fit_groups, scaler)
                raw, z, _ = predict_groups(
                    predictor, [held], tables["registry"], tables["physics"]
                )
                truth_raw = group_targets.loc[held, scaler.columns].to_numpy(float)
                truth_z = scaler.transform(truth_raw[None], clip=False)[0]
                rq_index = scaler.columns.index("log_rq_nm")
                records.append(
                    {
                        "held_out_growth_group": held,
                        "training_group_count": size,
                        "repeat": repeat,
                        "descriptor_mae_z": float(np.mean(np.abs(z[0] - truth_z))),
                        "rq_absolute_error_nm": float(
                            abs(np.exp(raw[0, rq_index]) - np.exp(truth_raw[rq_index]))
                        ),
                    }
                )
    frame = pd.DataFrame(records)
    write_csv(frame, report_root / "learning_curve_raw.csv")
    summary = (
        frame.groupby("training_group_count")
        .agg(
            descriptor_mae_z_median=("descriptor_mae_z", "median"),
            descriptor_mae_z_mean=("descriptor_mae_z", "mean"),
            descriptor_mae_z_q25=(
                "descriptor_mae_z",
                lambda values: values.quantile(0.25),
            ),
            descriptor_mae_z_q75=(
                "descriptor_mae_z",
                lambda values: values.quantile(0.75),
            ),
            rq_mae_nm_median=("rq_absolute_error_nm", "median"),
            rq_mae_nm_mean=("rq_absolute_error_nm", "mean"),
            evaluation_count=("descriptor_mae_z", "size"),
        )
        .reset_index()
    )
    write_csv(summary, report_root / "learning_curve_summary.csv")
    return summary


def run_experiment(config: dict[str, Any], *, smoke: bool) -> None:
    output_root = repo_path(config["output_root"]) / (
        "smoke" if smoke else "development"
    )
    report_root = repo_path(config["report_root"]) / (
        "smoke" if smoke else "development"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    split_audit = _split_manifest(descriptors, report_root, config)
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[descriptors["split"] == "val"].copy()
    # The historical test partition is deliberately absent from every target
    # table passed into fitting, calibration, uncertainty, or evaluation.
    development_descriptors = descriptors.loc[
        descriptors["split"].isin(["train", "val"])
    ].copy()
    group_targets = _group_targets(
        development_descriptors, list(config["condition_columns"])
    )
    fit_predictor = _predictor_factory(
        config=config, tables=tables, group_targets=group_targets
    )
    crossfit = _run_crossfit_generation(
        config=config,
        tables=tables,
        train_rows=train_rows,
        group_targets=group_targets,
        fit_predictor=fit_predictor,
        output_root=output_root,
        report_root=report_root,
        smoke=smoke,
    )
    validation = _run_validation(
        config=config,
        tables=tables,
        train_rows=train_rows,
        validation_rows=validation_rows,
        group_targets=group_targets,
        fit_predictor=fit_predictor,
        output_root=output_root,
        report_root=report_root,
        smoke=smoke,
    )
    uncertainty = _run_uncertainty(
        config=config,
        tables=tables,
        train_rows=train_rows,
        validation=validation,
        group_targets=group_targets,
        fit_predictor=fit_predictor,
        crossfit_predictions=crossfit["condition_predictions"],
        report_root=report_root,
        smoke=smoke,
    )
    learning_curve = _run_learning_curve(
        config=config,
        tables=tables,
        train_rows=train_rows,
        group_targets=group_targets,
        fit_predictor=fit_predictor,
        report_root=report_root,
        smoke=smoke,
    )
    from .visualization import make_figures

    comparison = make_figures(
        crossfit=crossfit,
        validation=validation,
        uncertainty=uncertainty,
        learning_curve=learning_curve,
        phase1_manifest_path=config["phase1_manifest"],
        prior_crossfit_metrics=config["prior_m2b_crossfit_metrics"],
        figure_dir=report_root / "figures",
    )
    write_csv(
        comparison,
        report_root / "baseline_vs_final_crossfit_metrics.csv",
    )
    manifest = {
        "status": "development_only_no_old_test_reuse",
        "smoke": smoke,
        "selected_method": SELECTED_METHOD,
        "selected_variance_cap": float(config["selected_variance_cap"]),
        "matern_blend_weight": float(config["matern_blend_weight"]),
        "retrieval_at_inference": False,
        "measured_afm_used_at_inference": False,
        "old_test_partition_used_for_modeling_or_evaluation": False,
        "split_audit": split_audit,
        "removelist": {
            "path": str(tables["removelist"].path),
            "sha256": tables["removelist"].sha256,
            "sample_ids": list(tables["removelist"].sample_ids),
            "overlap_after_filtering": [],
        },
        "crossfit_method_summary": crossfit["summary"].to_dict(orient="records"),
        "validation_method_summary": validation["summary"].to_dict(orient="records"),
        "uncertainty_summary": uncertainty["summary"].to_dict(orient="records"),
        "learning_curve": learning_curve.to_dict(orient="records"),
        "figure_directory": str(report_root / "figures"),
        "predictor": {
            "path": str(output_root / "rheed_descriptor_predictor.joblib"),
            "sha256": sha256_file(output_root / "rheed_descriptor_predictor.joblib"),
        },
        "claim_boundary": (
            "All current evidence is strict training-group cross-validation or "
            "pre-existing validation. The consumed historical test partition "
            "is present in the repository split manifest but was not passed to "
            "model fitting, method selection, generation, or evaluation; a "
            "prospective cohort is required for a new final claim."
        ),
    }
    write_json(manifest, report_root / "development_manifest.json")
    print(json.dumps(manifest, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distinct RHEED-to-AFM generation with group confidence"
    )
    parser.add_argument("mode", choices=("smoke", "development"))
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_distinct_confidence.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(load_config(args.config), smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
