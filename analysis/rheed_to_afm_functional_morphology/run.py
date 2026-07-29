from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    aggregate_group_conditions,
)
from analysis.rheed_to_afm_island_generation.evaluation import (
    evaluate_island_methods,
)
from analysis.rheed_to_afm_island_generation.islands import (
    IslandPrimitiveGenerator,
    fit_island_condition_model,
)
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from .amplitude import crossfit_target, predict_external_groups
from .metrics import (
    extract_surface_metrics,
    group_metric_table,
    scan_metric_table,
)
from .render import render_ensemble


M10 = "M10_dense_island_spectral_pareto"


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _arrays(path: Path) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return [
        np.asarray(array, dtype=np.float32)
        for array in payload["generated_unit_shapes"]
    ]


def _physics_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    result = result.set_index("sample_id")
    for column in result.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = values.replace([np.inf, -np.inf], np.nan)
    return result


def _target_series(
    descriptors: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    split: str,
) -> tuple[pd.Series, pd.Series]:
    rows = descriptors.loc[descriptors["split"] == split]
    log_rq = aggregate_group_conditions(
        rows,
        ["log_rq_nm"],
    )["log_rq_nm"]
    log_rq.index = log_rq.index.astype(str)
    log_rq.name = "log_rq_nm"
    fsmi = (
        metrics.loc[metrics["split"] == split]
        .set_index("growth_run_id")[
            "functional_surface_morphology_index_nm"
        ]
        .sort_index()
    )
    fsmi.index = fsmi.index.astype(str)
    log_fsmi = np.log(np.clip(fsmi.astype(float), 1e-6, None))
    log_fsmi.name = "log_fsmi_nm"
    return log_rq.sort_index(), log_fsmi.sort_index()


def _condition_vector(
    row: pd.Series,
    scaler: ConditionScaler,
    *,
    predicted_rq_nm: float,
) -> np.ndarray:
    values = np.asarray(
        [
            row[f"selected_predicted_z__{column}"]
            for column in scaler.columns
        ],
        dtype=np.float32,
    )
    rq_position = scaler.columns.index("log_rq_nm")
    values[rq_position] = float(
        (np.log(max(predicted_rq_nm, 1e-6)) - scaler.mean[rq_position])
        / scaler.scale[rq_position]
    )
    return values


def _save_generated(
    root: Path,
    *,
    method: str,
    group: str,
    arrays: list[np.ndarray],
    predicted_rq_nm: float,
    predicted_fsmi_nm: float,
    condition_z: np.ndarray,
) -> None:
    path = root / "generated_maps" / method / f"{group}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_unit_shapes=np.stack(arrays).astype(np.float32),
        predicted_rq_nm=np.asarray(float(predicted_rq_nm)),
        predicted_fsmi_nm=np.asarray(float(predicted_fsmi_nm)),
        condition_z=np.asarray(condition_z, dtype=np.float32),
        growth_run_id=np.asarray(str(group)),
        method=np.asarray(method),
        retrieval_at_inference=np.asarray(False),
        measured_afm_patch_used_at_inference=np.asarray(False),
    )


def _aggregate(
    frames: list[pd.DataFrame],
    *,
    output: Path,
    stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_group = pd.concat(frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    for method, rows in per_group.groupby("method"):
        record: dict[str, Any] = {
            "method": method,
            "growth_group_count": int(rows["growth_run_id"].nunique()),
        }
        for column in rows.select_dtypes(include=[np.number, bool]):
            record[f"median_{column}"] = float(
                rows[column].astype(float).median()
            )
            record[f"mean_{column}"] = float(
                rows[column].astype(float).mean()
            )
        if "afm_texture_gate_pass" in rows:
            record["texture_gate_pass_fraction"] = float(
                rows["afm_texture_gate_pass"].astype(float).mean()
            )
        records.append(record)
    summary = pd.DataFrame(records).sort_values("method")
    write_csv(per_group, output / f"{stem}_per_group.csv")
    write_csv(summary, output / f"{stem}_summary.csv")
    return per_group, summary


def _surface_method_metrics(
    *,
    group: str,
    true_fsmi_nm: float,
    predicted_rq_nm: dict[str, float],
    arrays: dict[str, list[np.ndarray]],
    scan_size_nm: float,
    analysis_scale_nm: float,
) -> pd.DataFrame:
    records = []
    for method, ensemble in arrays.items():
        generated = [
            extract_surface_metrics(
                np.asarray(array, dtype=float)
                * float(predicted_rq_nm[method]),
                scan_size_nm=scan_size_nm,
                analysis_scale_nm=analysis_scale_nm,
            )
            for array in ensemble
        ]
        row: dict[str, float | str] = {
            "growth_run_id": str(group),
            "method": method,
            "true_fsmi_nm": float(true_fsmi_nm),
        }
        for column in generated[0]:
            row[f"generated_{column}"] = float(
                np.median([record[column] for record in generated])
            )
        row["fsmi_absolute_error_nm"] = abs(
            row["generated_functional_surface_morphology_index_nm"]
            - float(true_fsmi_nm)
        )
        records.append(row)
    return pd.DataFrame(records)


def _source_paths(
    config: dict[str, Any],
    *,
    group: str,
    validation: bool,
) -> tuple[Path, Path]:
    m10_root = repo_path(config["parent_m10_output"])
    island_root = repo_path(config["parent_island_output"])
    if validation:
        old_m10 = (
            m10_root
            / "validation"
            / "generated_maps"
            / M10
            / f"{group}.npz"
        )
        spectral = (
            island_root
            / "generated_maps"
            / "M5_cloudlike_spectral_hybrid"
            / f"{group}.npz"
        )
    else:
        old_m10 = (
            m10_root
            / "crossfit"
            / "generated_maps"
            / M10
            / f"{group}.npz"
        )
        spectral = (
            island_root
            / "crossfit"
            / "generated_maps"
            / "M5_cloudlike_spectral_hybrid"
            / f"{group}.npz"
        )
    return old_m10, spectral


def _candidate_ensembles(
    *,
    config: dict[str, Any],
    generator: IslandPrimitiveGenerator,
    island_target: dict[str, float],
    spectral: list[np.ndarray],
    old_m10: list[np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, list[np.ndarray]]:
    structure = generator.generate_ensemble(
        island_target, draws=draws, seed=seed, mode="laguerre"
    )
    result: dict[str, list[np.ndarray]] = {M10: old_m10[:draws]}
    for method, parameters in config["candidate_renderers"].items():
        result[str(method)] = render_ensemble(
            structure,
            spectral[:draws],
            **parameters,
        )
    return result


def _prediction_metrics(
    predictions: pd.DataFrame,
    *,
    label: str,
) -> dict[str, float | str]:
    truth = predictions["true_target"].to_numpy(float)
    predicted = predictions["predicted_target"].to_numpy(float)
    pearson = pearsonr(truth, predicted)
    spearman = spearmanr(truth, predicted)
    return {
        "target": label,
        "group_count": len(predictions),
        "mean_absolute_error": float(np.mean(np.abs(truth - predicted))),
        "median_absolute_error": float(
            np.median(np.abs(truth - predicted))
        ),
        "rmse": float(np.sqrt(np.mean(np.square(truth - predicted)))),
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "true_min": float(np.min(truth)),
        "true_max": float(np.max(truth)),
        "predicted_min": float(np.min(predicted)),
        "predicted_max": float(np.max(predicted)),
        "standard_deviation_ratio": float(
            np.std(predicted) / max(np.std(truth), 1e-12)
        ),
        "interval_coverage": float(
            predictions["interval_covered"].astype(float).mean()
        ),
    }


def _confidence_table(
    *,
    rq_predictions: pd.DataFrame,
    fsmi_predictions: pd.DataFrame,
    standard: pd.DataFrame,
    island: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from analysis.rheed_to_afm_island_generation.morphology_confidence import (
        FEATURES,
        _fit_predict,
        _loo_predictions,
        _relative_confidence,
        _select_alpha,
    )

    method_standard = standard.loc[standard["method"] == method]
    method_island = island.loc[island["method"] == method]
    cross = method_standard.merge(
        method_island,
        on=["growth_run_id", "method"],
        suffixes=("", "_island"),
    ).merge(
        rq_predictions[
            [
                "growth_run_id",
                "absolute_error",
                "predicted_absolute_error",
                "interval_lower",
                "interval_upper",
                "interval_covered",
            ]
        ],
        on="growth_run_id",
    ).rename(
        columns={
            "absolute_error": "rq_target_absolute_error_nm",
            "predicted_absolute_error": (
                "predicted_rq_absolute_error_nm"
            ),
            "interval_lower": "rq_interval_lower_nm",
            "interval_upper": "rq_interval_upper_nm",
            "interval_covered": "rq_interval_covered",
        }
    ).merge(
        fsmi_predictions[
            [
                "growth_run_id",
                "absolute_error",
                "predicted_absolute_error",
                "interval_lower",
                "interval_upper",
                "interval_covered",
            ]
        ].rename(
            columns={
                "absolute_error": "fsmi_target_absolute_error_nm",
                "predicted_absolute_error": (
                    "predicted_fsmi_absolute_error_nm"
                ),
                "interval_lower": "fsmi_interval_lower_nm",
                "interval_upper": "fsmi_interval_upper_nm",
                "interval_covered": "fsmi_interval_covered",
            }
        ),
        on="growth_run_id",
    )
    x = cross[FEATURES].to_numpy(float)
    y = cross["island_feature_mae_z"].to_numpy(float)
    predicted_morphology_error = np.zeros(len(cross), dtype=float)
    morphology_upper = np.zeros(len(cross), dtype=float)
    for held in range(len(cross)):
        keep = np.arange(len(cross)) != held
        alpha, _ = _select_alpha(x[keep], y[keep])
        inner = _loo_predictions(x[keep], y[keep], alpha=alpha)
        radius = float(
            np.quantile(np.abs(inner - y[keep]), 0.90, method="higher")
        )
        predicted_morphology_error[held] = _fit_predict(
            x[keep], y[keep], x[held : held + 1], alpha=alpha
        )[0]
        morphology_upper[held] = (
            predicted_morphology_error[held] + radius
        )

    # FSMI already contains height amplitude, scale-dependent texture,
    # curvature, bearing-curve relief and island prominence.  Adding Rq again
    # to the joint score would double-count amplitude.  Percentile ranks also
    # put the physical FSMI error (nm) and topology error (z) on a common,
    # target-blind inference-time scale.
    fsmi_uncertainty = (
        cross["predicted_fsmi_absolute_error_nm"]
        .rank(pct=True)
        .to_numpy(float)
    )
    morphology_uncertainty = (
        pd.Series(predicted_morphology_error).rank(pct=True).to_numpy(
            float
        )
    )
    predicted_joint_error = 0.5 * (
        fsmi_uncertainty + morphology_uncertainty
    )
    realized_fsmi_rank = (
        cross["fsmi_target_absolute_error_nm"]
        .rank(pct=True)
        .to_numpy(float)
    )
    realized_morphology_rank = pd.Series(y).rank(pct=True).to_numpy(
        float
    )
    realized_joint_error = 0.5 * (
        realized_fsmi_rank + realized_morphology_rank
    )
    confidence = _relative_confidence(
        predicted_joint_error, predicted_joint_error
    )
    result = pd.DataFrame(
        {
            "growth_run_id": cross["growth_run_id"],
            "predicted_rq_absolute_error_nm": cross[
                "predicted_rq_absolute_error_nm"
            ],
            "realized_rq_absolute_error_nm": cross[
                "rq_target_absolute_error_nm"
            ],
            "rq_interval_lower_nm": cross["rq_interval_lower_nm"],
            "rq_interval_upper_nm": cross["rq_interval_upper_nm"],
            "rq_interval_covered": cross["rq_interval_covered"],
            "predicted_fsmi_absolute_error_nm": cross[
                "predicted_fsmi_absolute_error_nm"
            ],
            "realized_fsmi_absolute_error_nm": cross[
                "fsmi_target_absolute_error_nm"
            ],
            "fsmi_interval_lower_nm": cross["fsmi_interval_lower_nm"],
            "fsmi_interval_upper_nm": cross["fsmi_interval_upper_nm"],
            "fsmi_interval_covered": cross[
                "fsmi_interval_covered"
            ],
            "predicted_island_error_z": predicted_morphology_error,
            "island_error_90_upper_z": morphology_upper,
            "realized_island_error_z": y,
            "predicted_joint_error_index": predicted_joint_error,
            "realized_joint_error_index": realized_joint_error,
            "joint_confidence_index": confidence,
        }
    )
    rho = spearmanr(
        result["joint_confidence_index"],
        result["realized_joint_error_index"],
    )
    manifest = {
        "method": method,
        "joint_confidence_vs_realized_error_spearman": float(
            rho.statistic
        ),
        "joint_confidence_vs_realized_error_pvalue": float(rho.pvalue),
        "rq_expected_error_vs_realized_error_spearman": float(
            spearmanr(
                result["predicted_rq_absolute_error_nm"],
                result["realized_rq_absolute_error_nm"],
            ).statistic
        ),
        "fsmi_expected_error_vs_realized_error_spearman": float(
            spearmanr(
                result["predicted_fsmi_absolute_error_nm"],
                result["realized_fsmi_absolute_error_nm"],
            ).statistic
        ),
        "morphology_expected_error_vs_realized_error_spearman": float(
            spearmanr(
                result["predicted_island_error_z"],
                result["realized_island_error_z"],
            ).statistic
        ),
        "rq_90_interval_coverage": float(
            result["rq_interval_covered"].astype(float).mean()
        ),
        "fsmi_90_interval_coverage": float(
            result["fsmi_interval_covered"].astype(float).mean()
        ),
        "morphology_90_upper_coverage": float(
            np.mean(
                result["realized_island_error_z"]
                <= result["island_error_90_upper_z"]
            )
        ),
        "confidence_is_probability": False,
        "confidence_definition": (
            "Smoothed survival percentile of equal-weight ranks of strictly "
            "cross-fitted expected FSMI and island-topology errors. FSMI "
            "already contains height amplitude, so Rq is not double-counted."
        ),
    }
    return result, manifest


def _validation_confidence_table(
    *,
    cross_confidence: pd.DataFrame,
    cross_fsmi_predictions: pd.DataFrame,
    validation_rq_predictions: pd.DataFrame,
    validation_fsmi_predictions: pd.DataFrame,
    cross_standard: pd.DataFrame,
    cross_island: pd.DataFrame,
    validation_standard: pd.DataFrame,
    validation_island: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    from analysis.rheed_to_afm_island_generation.morphology_confidence import (
        FEATURES,
        _fit_predict,
        _select_alpha,
    )

    train = cross_standard.loc[
        cross_standard["method"] == method
    ].merge(
        cross_island.loc[cross_island["method"] == method],
        on=["growth_run_id", "method"],
        suffixes=("", "_island"),
    )
    validation = validation_standard.loc[
        validation_standard["method"] == method
    ].merge(
        validation_island.loc[validation_island["method"] == method],
        on=["growth_run_id", "method"],
        suffixes=("", "_island"),
    )
    x = train[FEATURES].to_numpy(float)
    y = train["island_feature_mae_z"].to_numpy(float)
    alpha, _ = _select_alpha(x, y)
    predicted_morphology = _fit_predict(
        x,
        y,
        validation[FEATURES].to_numpy(float),
        alpha=alpha,
    )

    cross_fsmi_expected = cross_fsmi_predictions[
        "predicted_absolute_error"
    ].to_numpy(float)
    cross_morphology_expected = cross_confidence[
        "predicted_island_error_z"
    ].to_numpy(float)

    def percentile(value: float, reference: np.ndarray) -> float:
        return float(
            (1.0 + np.sum(reference <= float(value)))
            / (len(reference) + 1.0)
        )

    rows = []
    rq = validation_rq_predictions.set_index("growth_run_id")
    fsmi = validation_fsmi_predictions.set_index("growth_run_id")
    cross_joint = cross_confidence[
        "predicted_joint_error_index"
    ].to_numpy(float)
    for position, (_, row) in enumerate(validation.iterrows()):
        group = str(row["growth_run_id"])
        fsmi_rank = percentile(
            float(fsmi.loc[group, "predicted_absolute_error"]),
            cross_fsmi_expected,
        )
        morphology_rank = percentile(
            float(predicted_morphology[position]),
            cross_morphology_expected,
        )
        joint = 0.5 * (fsmi_rank + morphology_rank)
        confidence = float(
            100.0
            * (1.0 + np.sum(cross_joint >= joint))
            / (len(cross_joint) + 1.0)
        )
        rows.append(
            {
                "growth_run_id": group,
                "joint_confidence_index": confidence,
                "predicted_joint_error_index": joint,
                "predicted_rq_absolute_error_nm": float(
                    rq.loc[group, "predicted_absolute_error"]
                ),
                "realized_rq_absolute_error_nm": float(
                    rq.loc[group, "absolute_error"]
                ),
                "rq_interval_lower_nm": float(
                    rq.loc[group, "interval_lower"]
                ),
                "rq_interval_upper_nm": float(
                    rq.loc[group, "interval_upper"]
                ),
                "rq_interval_covered": bool(
                    rq.loc[group, "interval_covered"]
                ),
                "predicted_fsmi_absolute_error_nm": float(
                    fsmi.loc[group, "predicted_absolute_error"]
                ),
                "realized_fsmi_absolute_error_nm": float(
                    fsmi.loc[group, "absolute_error"]
                ),
                "predicted_island_error_z": float(
                    predicted_morphology[position]
                ),
                "realized_island_error_z": float(
                    row["island_feature_mae_z"]
                ),
                "confidence_is_probability": False,
            }
        )
    return pd.DataFrame(rows)


def run(config: dict[str, Any], *, smoke: bool = False) -> None:
    suffix = "smoke" if smoke else "development"
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(config)
    descriptors = tables["descriptors"].copy()
    # The historical test rows may remain in the source table but never enter
    # metric extraction, fitting, selection, generation or evaluation here.
    development_rows = descriptors.loc[
        descriptors["split"].isin(["train", "val"])
    ].copy()
    scan_metrics = scan_metric_table(
        development_rows,
        scan_size_nm=float(config["scan_size_nm"]),
        analysis_scale_nm=float(config["analysis_scale_nm"]),
    )
    group_metrics = group_metric_table(scan_metrics)
    write_csv(scan_metrics, report / "surface_metrics_per_scan.csv")
    write_csv(group_metrics, report / "surface_metrics_per_group.csv")

    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[
        descriptors["split"] == "val"
    ].copy()
    log_rq, log_fsmi = _target_series(
        descriptors, group_metrics, split="train"
    )
    validation_log_rq, validation_log_fsmi = _target_series(
        descriptors, group_metrics, split="val"
    )
    physics = _physics_table(tables["physics"])

    cross_rq, inner_rq = crossfit_target(
        physics=physics,
        log_target=log_rq,
        alpha=float(config["ridge_alpha"]),
        morphology_weight=float(config["morphology_head_weight"]),
        confidence_alpha=float(config["confidence_alpha"]),
    )
    cross_fsmi, inner_fsmi = crossfit_target(
        physics=physics,
        log_target=log_fsmi,
        alpha=float(config["ridge_alpha"]),
        morphology_weight=float(config["morphology_head_weight"]),
        confidence_alpha=float(config["confidence_alpha"]),
    )
    validation_groups = list(validation_log_rq.index.astype(str))
    validation_rq = predict_external_groups(
        physics=physics,
        log_target=log_rq,
        query_groups=validation_groups,
        crossfit_predictions=cross_rq,
        alpha=float(config["ridge_alpha"]),
        morphology_weight=float(config["morphology_head_weight"]),
        confidence_alpha=float(config["confidence_alpha"]),
    )
    validation_fsmi = predict_external_groups(
        physics=physics,
        log_target=log_fsmi,
        query_groups=validation_groups,
        crossfit_predictions=cross_fsmi,
        alpha=float(config["ridge_alpha"]),
        morphology_weight=float(config["morphology_head_weight"]),
        confidence_alpha=float(config["confidence_alpha"]),
    )
    for prediction, truth in (
        (validation_rq, validation_log_rq),
        (validation_fsmi, validation_log_fsmi),
    ):
        true_lookup = np.exp(truth)
        prediction["true_target"] = prediction["growth_run_id"].map(
            true_lookup
        )
        prediction["absolute_error"] = np.abs(
            prediction["predicted_target"] - prediction["true_target"]
        )
        prediction["interval_covered"] = (
            (prediction["interval_lower"] <= prediction["true_target"])
            & (prediction["true_target"] <= prediction["interval_upper"])
        )
    for table, name in (
        (cross_rq, "rq_crossfit_predictions"),
        (inner_rq, "rq_nested_inner_predictions"),
        (cross_fsmi, "fsmi_crossfit_predictions"),
        (inner_fsmi, "fsmi_nested_inner_predictions"),
        (validation_rq, "rq_validation_predictions"),
        (validation_fsmi, "fsmi_validation_predictions"),
    ):
        write_csv(table, report / f"{name}.csv")

    parent_report = repo_path(config["parent_condition_report"])
    parent_cross = pd.read_csv(
        parent_report
        / "training_group_cross_validation"
        / "condition_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    parent_validation = pd.read_csv(
        parent_report / "validation_condition_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    cross_rq_map = cross_rq.set_index("growth_run_id")
    cross_fsmi_map = cross_fsmi.set_index("growth_run_id")
    validation_rq_map = validation_rq.set_index("growth_run_id")
    validation_fsmi_map = validation_fsmi.set_index("growth_run_id")
    metric_lookup = group_metrics.set_index(["split", "growth_run_id"])

    generator = IslandPrimitiveGenerator(
        resolution=int(config["resolution"]),
        laguerre_count_factor=float(config["laguerre_count_factor"]),
        fine_count_factor=float(config["fine_count_factor"]),
    )
    groups = list(log_rq.index.astype(str))
    if smoke:
        groups = groups[:3]
    cross_standard_frames: list[pd.DataFrame] = []
    cross_island_frames: list[pd.DataFrame] = []
    cross_surface_frames: list[pd.DataFrame] = []
    draws = 2 if smoke else int(config["crossfit_draws"])
    for fold, held in enumerate(groups):
        fit_groups = set(log_rq.index.astype(str))
        fit_groups.remove(held)
        fit_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str).isin(fit_groups)
        ]
        held_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str) == held
        ]
        scaler = ConditionScaler.fit(
            train_rows,
            list(config["condition_columns"]),
            fit_groups,
        )
        island_model, _, _ = fit_island_condition_model(
            train_rows=fit_rows,
            condition_scaler=scaler,
            resolution=int(config["resolution"]),
            alphas=config["island_ridge_alphas"],
        )
        predicted_rq = float(
            cross_rq_map.loc[held, "predicted_target"]
        )
        predicted_fsmi = float(
            cross_fsmi_map.loc[held, "predicted_target"]
        )
        condition_z = _condition_vector(
            parent_cross.loc[held],
            scaler,
            predicted_rq_nm=predicted_rq,
        )
        island_target = island_model.predict(condition_z)
        old_m10_path, spectral_path = _source_paths(
            config, group=held, validation=False
        )
        methods = _candidate_ensembles(
            config=config,
            generator=generator,
            island_target=island_target,
            spectral=_arrays(spectral_path),
            old_m10=_arrays(old_m10_path),
            draws=draws,
            seed=int(config["seed"]) + fold * 10_000,
        )
        for method, arrays in methods.items():
            method_rq = (
                float(
                    np.load(old_m10_path, allow_pickle=False)[
                        "predicted_rq_nm"
                    ]
                )
                if method == M10
                else predicted_rq
            )
            _save_generated(
                output / "crossfit",
                method=method,
                group=held,
                arrays=arrays,
                predicted_rq_nm=method_rq,
                predicted_fsmi_nm=predicted_fsmi,
                condition_z=condition_z,
            )
        rq_by_method = {
            method: {
                held: (
                    float(
                        np.load(old_m10_path, allow_pickle=False)[
                            "predicted_rq_nm"
                        ]
                    )
                    if method == M10
                    else predicted_rq
                )
            }
            for method in methods
        }
        standard = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=fit_rows,
            condition_scaler=scaler,
            generated={
                method: {held: arrays}
                for method, arrays in methods.items()
            },
            generated_rq=rq_by_method,
            output_dir=report
            / "crossfit"
            / "folds"
            / held
            / "standard",
            resolution=int(config["resolution"]),
        )["per_group"]
        standard.insert(0, "cross_validation_fold", fold)
        cross_standard_frames.append(standard)
        island = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=fit_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        island.insert(0, "cross_validation_fold", fold)
        cross_island_frames.append(island)
        true_fsmi = float(
            metric_lookup.loc[
                ("train", held),
                "functional_surface_morphology_index_nm",
            ]
        )
        cross_surface_frames.append(
            _surface_method_metrics(
                group=held,
                true_fsmi_nm=true_fsmi,
                predicted_rq_nm={
                    method: float(values[held])
                    for method, values in rq_by_method.items()
                },
                arrays=methods,
                scan_size_nm=float(config["scan_size_nm"]),
                analysis_scale_nm=float(config["analysis_scale_nm"]),
            )
        )
    cross_standard, cross_standard_summary = _aggregate(
        cross_standard_frames,
        output=report / "crossfit",
        stem="standard",
    )
    cross_island, cross_island_summary = _aggregate(
        cross_island_frames,
        output=report / "crossfit",
        stem="island",
    )
    cross_surface, cross_surface_summary = _aggregate(
        cross_surface_frames,
        output=report / "crossfit",
        stem="functional_surface",
    )

    validation_standard_frames: list[pd.DataFrame] = []
    validation_island_frames: list[pd.DataFrame] = []
    validation_surface_frames: list[pd.DataFrame] = []
    validation_methods: dict[str, dict[str, list[np.ndarray]]] = {
        method: {} for method in [M10, *config["candidate_renderers"]]
    }
    validation_rq_by_method: dict[str, dict[str, float]] = {
        method: {} for method in validation_methods
    }
    validation_scaler = ConditionScaler.fit(
        train_rows,
        list(config["condition_columns"]),
        set(log_rq.index.astype(str)),
    )
    island_model, _, _ = fit_island_condition_model(
        train_rows=train_rows,
        condition_scaler=validation_scaler,
        resolution=int(config["resolution"]),
        alphas=config["island_ridge_alphas"],
    )
    validation_draws = 2 if smoke else int(config["validation_draws"])
    for fold, held in enumerate(validation_groups):
        predicted_rq = float(
            validation_rq_map.loc[held, "predicted_target"]
        )
        predicted_fsmi = float(
            validation_fsmi_map.loc[held, "predicted_target"]
        )
        condition_z = _condition_vector(
            parent_validation.loc[held],
            validation_scaler,
            predicted_rq_nm=predicted_rq,
        )
        island_target = island_model.predict(condition_z)
        old_m10_path, spectral_path = _source_paths(
            config, group=held, validation=True
        )
        methods = _candidate_ensembles(
            config=config,
            generator=generator,
            island_target=island_target,
            spectral=_arrays(spectral_path),
            old_m10=_arrays(old_m10_path),
            draws=validation_draws,
            seed=int(config["seed"]) + 500_000 + fold * 10_000,
        )
        held_rows = validation_rows.loc[
            validation_rows["growth_run_id"].astype(str) == held
        ]
        old_rq = float(
            np.load(old_m10_path, allow_pickle=False)["predicted_rq_nm"]
        )
        for method, arrays in methods.items():
            method_rq = old_rq if method == M10 else predicted_rq
            validation_methods[method][held] = arrays
            validation_rq_by_method[method][held] = method_rq
            _save_generated(
                output / "validation",
                method=method,
                group=held,
                arrays=arrays,
                predicted_rq_nm=method_rq,
                predicted_fsmi_nm=predicted_fsmi,
                condition_z=condition_z,
            )
        validation_island_frames.append(
            evaluate_island_methods(
                held_rows=held_rows,
                train_rows=train_rows,
                generated=methods,
                resolution=int(config["resolution"]),
            )
        )
        true_fsmi = float(
            metric_lookup.loc[
                ("val", held),
                "functional_surface_morphology_index_nm",
            ]
        )
        validation_surface_frames.append(
            _surface_method_metrics(
                group=held,
                true_fsmi_nm=true_fsmi,
                predicted_rq_nm={
                    method: float(
                        validation_rq_by_method[method][held]
                    )
                    for method in methods
                },
                arrays=methods,
                scan_size_nm=float(config["scan_size_nm"]),
                analysis_scale_nm=float(config["analysis_scale_nm"]),
            )
        )
    validation_standard = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=validation_scaler,
        generated=validation_methods,
        generated_rq=validation_rq_by_method,
        output_dir=report / "validation" / "standard_details",
        resolution=int(config["resolution"]),
    )
    validation_standard_frames.append(
        validation_standard["per_group"]
    )
    validation_standard_table, validation_standard_summary = _aggregate(
        validation_standard_frames,
        output=report / "validation",
        stem="standard",
    )
    validation_island, validation_island_summary = _aggregate(
        validation_island_frames,
        output=report / "validation",
        stem="island",
    )
    validation_surface, validation_surface_summary = _aggregate(
        validation_surface_frames,
        output=report / "validation",
        stem="functional_surface",
    )

    selected = str(config["selected_method"])
    confidence, confidence_manifest = _confidence_table(
        rq_predictions=cross_rq,
        fsmi_predictions=cross_fsmi,
        standard=cross_standard,
        island=cross_island,
        method=selected,
    )
    write_csv(confidence, report / "confidence_crossfit.csv")
    write_json(confidence_manifest, report / "confidence_manifest.json")
    validation_confidence = _validation_confidence_table(
        cross_confidence=confidence,
        cross_fsmi_predictions=cross_fsmi,
        validation_rq_predictions=validation_rq,
        validation_fsmi_predictions=validation_fsmi,
        cross_standard=cross_standard,
        cross_island=cross_island,
        validation_standard=validation_standard_table,
        validation_island=validation_island,
        method=selected,
    )
    write_csv(
        validation_confidence, report / "confidence_validation.csv"
    )

    target_summary = pd.DataFrame(
        [
            _prediction_metrics(cross_rq, label="Rq_nm"),
            _prediction_metrics(cross_fsmi, label="FSMI_nm"),
        ]
    )
    write_csv(target_summary, report / "target_prediction_summary.csv")
    summary = (
        cross_standard_summary.merge(
            cross_island_summary,
            on="method",
            suffixes=("_standard", "_island"),
        )
        .merge(
            cross_surface_summary,
            on="method",
            suffixes=("", "_surface"),
        )
    )
    write_csv(summary, report / "candidate_summary.csv")

    baseline_rows = []
    for split, standard_summary, island_summary, surface_summary in (
        (
            "strict_15_growth_loo",
            cross_standard_summary,
            cross_island_summary,
            cross_surface_summary,
        ),
        (
            "preexisting_validation",
            validation_standard_summary,
            validation_island_summary,
            validation_surface_summary,
        ),
    ):
        for method in (M10, selected):
            s = standard_summary.loc[
                standard_summary["method"] == method
            ].iloc[0]
            i = island_summary.loc[
                island_summary["method"] == method
            ].iloc[0]
            f = surface_summary.loc[
                surface_summary["method"] == method
            ].iloc[0]
            baseline_rows.append(
                {
                    "split": split,
                    "method": method,
                    "median_rq_absolute_error_nm": s[
                        "median_rq_absolute_error_nm"
                    ],
                    "median_condition_descriptor_mae_z": s[
                        "median_condition_descriptor_mae_z"
                    ],
                    "median_psd_log_distance": s[
                        "median_normalized_psd_log_distance"
                    ],
                    "median_sharpness_ratio": s[
                        "median_sharpness_ratio"
                    ],
                    "median_composite_error": s[
                        "median_composite_score"
                    ],
                    "texture_gate_pass_fraction": s[
                        "texture_gate_pass_fraction"
                    ],
                    "median_island_feature_mae_z": i[
                        "median_island_feature_mae_z"
                    ],
                    "median_afm_prior_mahalanobis": i[
                        "median_afm_prior_mahalanobis"
                    ],
                    "median_q70_area_log_error": i[
                        "median_median_area_q70_log_absolute_error"
                    ],
                    "median_q70_count_log_error": i[
                        "median_component_count_q70_log_absolute_error"
                    ],
                    "median_fsmi_absolute_error_nm": f[
                        "median_fsmi_absolute_error_nm"
                    ],
                    "median_generated_boundary_contrast": f[
                        "median_generated_island_boundary_contrast"
                    ],
                }
            )
    baseline = pd.DataFrame(baseline_rows)
    write_csv(baseline, report / "baseline_vs_final_metrics.csv")

    manifest = {
        "experiment_id": config["experiment_id"],
        "selected_method": selected,
        "training_growth_groups": list(log_rq.index.astype(str)),
        "validation_growth_groups": validation_groups,
        "historical_test_used": False,
        "test_rows_present_but_unselected": int(
            (descriptors["split"] == "test").sum()
        ),
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "removelist_sha256": sha256_file(
            repo_path(config["removelist_path"])
        ),
        "target_prediction_summary": target_summary.to_dict(
            orient="records"
        ),
        "confidence": confidence_manifest,
        "candidate_summary": summary.to_dict(orient="records"),
        "baseline_vs_final": baseline.to_dict(orient="records"),
        "fsmi_definition": (
            "RMS of five height-equivalent terms: Sq, 31.25 nm RMS height "
            "increment, 31.25 nm second-difference curvature relief, one "
            "quarter p90-p10 bearing span, and q70 island prominence."
        ),
        "fsmi_is_standardized_parameter": False,
        "claim_boundary": (
            "Development evidence only: strict training-growth LOO and the "
            "pre-existing validation cohort. The historical test remains "
            "closed; prospective growth groups are required for confirmation."
        ),
    }
    write_json(manifest, report / "best_model_manifest.json")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_functional_morphology.json",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config), smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
