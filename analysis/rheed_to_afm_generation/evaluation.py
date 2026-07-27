from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from skimage.metrics import structural_similarity
import torch

from analysis.rheed_video_afm_story.afm_descriptors import describe_map
from analysis.rheed_video_afm_story.afm_evaluation import reconstruction_metrics
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .data import ConditionScaler
from .model import ConditionalAFMVAE


def _load_unit_map(row: pd.Series, resolution: int) -> np.ndarray:
    paths = json.loads(str(row["unit_shape_paths"]))
    return project_unit_rq_np(
        np.load(repo_path(paths[str(resolution)]), allow_pickle=False).astype(
            np.float32
        )
    )


def _condition_from_map(
    unit_map: np.ndarray,
    rq_nm: float,
    columns: list[str],
) -> np.ndarray:
    descriptor = describe_map(project_unit_rq_np(unit_map), "unit")
    values = {
        "log_rq_nm": float(np.log(max(rq_nm, 1e-6))),
        "unit_ra": descriptor["unit_ra"],
        "unit_psd_mid_fraction": descriptor["unit_psd_mid_fraction"],
        "unit_psd_high_fraction": descriptor["unit_psd_high_fraction"],
        "unit_psd_slope": descriptor["unit_psd_slope"],
        "log_unit_autocorr_length_nm": float(
            np.log(max(descriptor["unit_autocorr_length_nm"], 1e-6))
        ),
        "log_unit_anisotropy_ratio": float(
            np.log(max(descriptor["unit_anisotropy_ratio"], 1.0))
        ),
        "unit_skewness": descriptor["unit_skewness"],
        "unit_kurtosis": descriptor["unit_kurtosis"],
    }
    return np.asarray([values[column] for column in columns], dtype=float)


def _map_medoid(
    maps: list[np.ndarray],
    rq_values: list[float],
    scaler: ConditionScaler,
) -> int:
    descriptors = np.vstack(
        [
            _condition_from_map(array, rq, scaler.columns)
            for array, rq in zip(maps, rq_values)
        ]
    )
    standardized = scaler.transform(descriptors, clip=False)
    center = np.nanmedian(standardized, axis=0)
    return int(np.argmin(np.linalg.norm(standardized - center, axis=1)))


def _pairwise_l1(maps: list[np.ndarray]) -> float:
    if len(maps) < 2:
        return 0.0
    values = [
        float(np.mean(np.abs(maps[i] - maps[j])))
        for i in range(len(maps))
        for j in range(i + 1, len(maps))
    ]
    return float(np.median(values))


def _data_range(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        max(
            np.percentile(a, 99) - np.percentile(a, 1),
            np.percentile(b, 99) - np.percentile(b, 1),
            1e-6,
        )
    )


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=_data_range(a, b)))


def _nearest_training_audit(
    generated: np.ndarray,
    training_maps: list[np.ndarray],
) -> dict[str, float | bool]:
    l1_values = [float(np.mean(np.abs(generated - real))) for real in training_maps]
    ssim_values = [_similarity(generated, real) for real in training_maps]
    exact = any(np.array_equal(generated, real) for real in training_maps)
    return {
        "nearest_training_l1": float(np.min(l1_values)),
        "max_training_ssim": float(np.max(ssim_values)),
        "exact_training_pixel_equality": bool(exact),
    }


def _method_metrics(
    *,
    group_id: str,
    method: str,
    target_unit: np.ndarray,
    target_rq_nm: float,
    generated_unit: np.ndarray,
    generated_rq_nm: float,
    generated_set: list[np.ndarray],
    real_set: list[np.ndarray],
    condition_scaler: ConditionScaler,
    true_condition_raw: np.ndarray,
    training_maps: list[np.ndarray],
    source_group: str | None = None,
) -> dict[str, Any]:
    shape_metrics = reconstruction_metrics(
        target_unit, generated_unit, target_rq_nm
    )
    target_physical = target_rq_nm * project_unit_rq_np(target_unit)
    generated_physical = generated_rq_nm * project_unit_rq_np(generated_unit)
    generated_conditions = np.vstack(
        [
            _condition_from_map(sample, generated_rq_nm, condition_scaler.columns)
            for sample in generated_set
        ]
    )
    generated_condition_median = np.nanmedian(generated_conditions, axis=0)
    target_z = condition_scaler.transform(true_condition_raw[None], clip=False)[0]
    generated_z = condition_scaler.transform(
        generated_condition_median[None], clip=False
    )[0]
    row: dict[str, Any] = {
        "growth_run_id": group_id,
        "method": method,
        "source_group": source_group or "",
        "true_rq_nm": target_rq_nm,
        "generated_rq_nm": generated_rq_nm,
        "rq_absolute_error_nm": abs(generated_rq_nm - target_rq_nm),
        "physical_histogram_wasserstein_nm": float(
            wasserstein_distance(target_physical.ravel(), generated_physical.ravel())
        ),
        "generated_pairwise_l1": _pairwise_l1(generated_set),
        "real_pairwise_l1": _pairwise_l1(real_set),
        "diversity_ratio": _pairwise_l1(generated_set)
        / max(_pairwise_l1(real_set), 1e-8),
        "condition_descriptor_mae_z": float(
            np.nanmean(np.abs(generated_z - target_z))
        ),
    }
    row.update(shape_metrics)
    row.update(_nearest_training_audit(generated_unit, training_maps))
    return row


def _generate_with_noise(
    model: ConditionalAFMVAE,
    condition_z: np.ndarray,
    sample_count: int,
    seed: int,
    device: torch.device,
) -> list[np.ndarray]:
    condition = torch.as_tensor(
        np.repeat(condition_z[None], sample_count, axis=0),
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        prior_mean, prior_logvar = model.conditional_prior(condition)
        noise = torch.randn(
            prior_mean.shape,
            dtype=prior_mean.dtype,
            device=device,
            generator=generator,
        )
        latent = prior_mean + torch.exp(0.5 * prior_logvar) * noise
        generated = model.decode(latent, condition).detach().cpu().numpy()[:, 0]
    return [project_unit_rq_np(array) for array in generated]


def _bootstrap_summary(
    metrics: pd.DataFrame,
    *,
    seed: int,
    repetitions: int = 2000,
) -> pd.DataFrame:
    metric_columns = [
        "rq_absolute_error_nm",
        "unit_l1",
        "ssim",
        "normalized_psd_log_distance",
        "correlation_length_relative_error",
        "height_quantile_error",
        "physical_histogram_wasserstein_nm",
        "condition_descriptor_mae_z",
        "diversity_ratio",
        "nearest_training_l1",
        "max_training_ssim",
        "composite_score",
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method"):
        for column in metric_columns:
            values = group[column].to_numpy(float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
            bootstrap = np.median(draws, axis=1)
            rows.append(
                {
                    "method": method,
                    "metric": column,
                    "N_groups": len(values),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "ci95_low": float(np.percentile(bootstrap, 2.5)),
                    "ci95_high": float(np.percentile(bootstrap, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_split(
    *,
    split_name: str,
    model: ConditionalAFMVAE,
    device: torch.device,
    split_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    predicted_raw: dict[str, np.ndarray],
    predicted_standardized: dict[str, np.ndarray],
    transformed_features: dict[str, np.ndarray],
    train_transformed_features: dict[str, np.ndarray],
    output_dir: str | Path,
    resolution: int,
    samples_per_condition: int,
    seed: int,
) -> dict[str, Any]:
    output = repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_dir = output / "generated_samples"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_maps = [
        _load_unit_map(row, resolution) for _, row in train_rows.iterrows()
    ]
    unconditional_shape = project_unit_rq_np(np.mean(np.stack(train_maps), axis=0))
    unconditional_rq = float(train_rows.groupby("growth_run_id")["rq_nm"].median().median())

    train_group_representatives: dict[str, tuple[np.ndarray, float]] = {}
    for group_id, group_rows in train_rows.groupby("growth_run_id"):
        maps = [_load_unit_map(row, resolution) for _, row in group_rows.iterrows()]
        rqs = [float(value) for value in group_rows["rq_nm"]]
        medoid = _map_medoid(maps, rqs, condition_scaler)
        train_group_representatives[str(group_id)] = (
            maps[medoid],
            float(np.median(rqs)),
        )

    groups = sorted(split_rows["growth_run_id"].astype(str).unique())
    permutation = {
        group: groups[(index + 1) % len(groups)]
        for index, group in enumerate(groups)
    }
    metric_rows: list[dict[str, Any]] = []
    condition_control_rows: list[dict[str, Any]] = []
    sample_manifest_rows: list[dict[str, Any]] = []
    representative_payload: dict[str, dict[str, Any]] = {}

    for group_index, group in enumerate(groups):
        group_rows = split_rows.loc[
            split_rows["growth_run_id"].astype(str) == group
        ]
        real_maps = [
            _load_unit_map(row, resolution) for _, row in group_rows.iterrows()
        ]
        real_rqs = [float(value) for value in group_rows["rq_nm"]]
        real_medoid_index = _map_medoid(real_maps, real_rqs, condition_scaler)
        real_medoid = real_maps[real_medoid_index]
        true_rq = float(np.median(real_rqs))
        true_condition_raw = (
            group_rows[condition_scaler.columns].median().to_numpy(float)
        )
        predicted_rq = float(
            np.exp(
                predicted_raw[group][
                    condition_scaler.columns.index("log_rq_nm")
                ]
            )
        )
        generated_maps = _generate_with_noise(
            model,
            predicted_standardized[group],
            samples_per_condition,
            seed + 1009 * (group_index + 1),
            device,
        )
        generated_rqs = [predicted_rq] * len(generated_maps)
        generated_medoid_index = _map_medoid(
            generated_maps, generated_rqs, condition_scaler
        )
        generated_medoid = generated_maps[generated_medoid_index]

        wrong_group = permutation[group]
        wrong_maps = _generate_with_noise(
            model,
            predicted_standardized[wrong_group],
            samples_per_condition,
            seed + 1009 * (group_index + 1),
            device,
        )
        wrong_rq = float(
            np.exp(
                predicted_raw[wrong_group][
                    condition_scaler.columns.index("log_rq_nm")
                ]
            )
        )
        correct_condition_error = float(
            np.mean(
                np.abs(
                    condition_scaler.transform(
                        np.nanmedian(
                            np.vstack(
                                [
                                    _condition_from_map(
                                        sample, predicted_rq, condition_scaler.columns
                                    )
                                    for sample in generated_maps
                                ]
                            ),
                            axis=0,
                        )[None],
                        clip=False,
                    )[0]
                    - condition_scaler.transform(
                        true_condition_raw[None], clip=False
                    )[0]
                )
            )
        )
        wrong_condition_error = float(
            np.mean(
                np.abs(
                    condition_scaler.transform(
                        np.nanmedian(
                            np.vstack(
                                [
                                    _condition_from_map(
                                        sample, wrong_rq, condition_scaler.columns
                                    )
                                    for sample in wrong_maps
                                ]
                            ),
                            axis=0,
                        )[None],
                        clip=False,
                    )[0]
                    - condition_scaler.transform(
                        true_condition_raw[None], clip=False
                    )[0]
                )
            )
        )
        condition_control_rows.append(
            {
                "growth_run_id": group,
                "permuted_condition_source": wrong_group,
                "correct_condition_descriptor_mae_z": correct_condition_error,
                "permuted_condition_descriptor_mae_z": wrong_condition_error,
                "permutation_error_increase": wrong_condition_error
                - correct_condition_error,
                "correct_better_than_permuted": correct_condition_error
                < wrong_condition_error,
            }
        )

        train_groups = sorted(train_transformed_features)
        distances = [
            float(
                np.linalg.norm(
                    transformed_features[group]
                    - train_transformed_features[train_group]
                )
            )
            for train_group in train_groups
        ]
        retrieved_group = train_groups[int(np.argmin(distances))]
        retrieved_shape, retrieved_rq = train_group_representatives[retrieved_group]

        methods = [
            (
                "unconditional_train_mean",
                unconditional_shape,
                unconditional_rq,
                [unconditional_shape],
                None,
            ),
            (
                "nearest_rheed_retrieval",
                retrieved_shape,
                retrieved_rq,
                [retrieved_shape],
                retrieved_group,
            ),
            (
                "rheed_conditional_cvae",
                generated_medoid,
                predicted_rq,
                generated_maps,
                None,
            ),
        ]
        for method, shape, rq, generated_set, source_group in methods:
            metric_rows.append(
                _method_metrics(
                    group_id=group,
                    method=method,
                    target_unit=real_medoid,
                    target_rq_nm=true_rq,
                    generated_unit=shape,
                    generated_rq_nm=rq,
                    generated_set=generated_set,
                    real_set=real_maps,
                    condition_scaler=condition_scaler,
                    true_condition_raw=true_condition_raw,
                    training_maps=train_maps,
                    source_group=source_group,
                )
            )

        sample_path = generated_dir / f"{group}_samples.npz"
        np.savez_compressed(
            sample_path,
            generated_unit_shapes=np.stack(generated_maps).astype(np.float32),
            predicted_rq_nm=np.asarray(predicted_rq, dtype=np.float32),
            predicted_condition_raw=predicted_raw[group].astype(np.float32),
            predicted_condition_standardized=predicted_standardized[group].astype(
                np.float32
            ),
            real_medoid_unit_shape=real_medoid.astype(np.float32),
            real_rq_nm=np.asarray(true_rq, dtype=np.float32),
            retrieved_unit_shape=retrieved_shape.astype(np.float32),
            retrieved_rq_nm=np.asarray(retrieved_rq, dtype=np.float32),
            retrieved_group=np.asarray(retrieved_group),
        )
        sample_manifest_rows.append(
            {
                "growth_run_id": group,
                "split": split_name,
                "generated_npz": str(sample_path.relative_to(repo_path("."))),
                "sample_count": len(generated_maps),
                "real_scan_count": len(real_maps),
                "real_medoid_afm_file_id": str(
                    group_rows.iloc[real_medoid_index]["afm_file_id"]
                ),
                "retrieved_training_group": retrieved_group,
                "retrieval_feature_distance": min(distances),
            }
        )
        representative_payload[group] = {
            "real_medoid": real_medoid,
            "real_rq": true_rq,
            "generated_medoid": generated_medoid,
            "generated_samples": generated_maps,
            "generated_rq": predicted_rq,
            "retrieved": retrieved_shape,
            "retrieved_rq": retrieved_rq,
            "retrieved_group": retrieved_group,
        }

    metrics = pd.DataFrame(metric_rows)
    condition_control = pd.DataFrame(condition_control_rows)
    sample_manifest = pd.DataFrame(sample_manifest_rows)
    summary = _bootstrap_summary(metrics, seed=seed)
    write_csv(metrics, output / "per_group_metrics.csv")
    write_csv(summary, output / "metric_summary.csv")
    write_csv(condition_control, output / "condition_permutation_control.csv")
    write_csv(sample_manifest, output / "generated_sample_manifest.csv")

    cvae = metrics.loc[metrics["method"] == "rheed_conditional_cvae"]
    descriptor_predictions = []
    for group in groups:
        truth = (
            split_rows.loc[
                split_rows["growth_run_id"].astype(str) == group,
                condition_scaler.columns,
            ]
            .median()
            .to_numpy(float)
        )
        for position, column in enumerate(condition_scaler.columns):
            descriptor_predictions.append(
                {
                    "growth_run_id": group,
                    "descriptor": column,
                    "true": truth[position],
                    "predicted": float(predicted_raw[group][position]),
                    "absolute_error": abs(
                        float(predicted_raw[group][position]) - truth[position]
                    ),
                }
            )
    descriptor_predictions_frame = pd.DataFrame(descriptor_predictions)
    write_csv(
        descriptor_predictions_frame, output / "descriptor_predictions.csv"
    )
    rq_rows = descriptor_predictions_frame.loc[
        descriptor_predictions_frame["descriptor"] == "log_rq_nm"
    ]
    rq_correlation = (
        float(spearmanr(rq_rows["true"], rq_rows["predicted"]).statistic)
        if len(rq_rows) >= 3
        else float("nan")
    )
    result = {
        "split": split_name,
        "group_count": len(groups),
        "scan_count": len(split_rows),
        "methods": sorted(metrics["method"].unique()),
        "cvae_median_rq_mae_nm": float(cvae["rq_absolute_error_nm"].median()),
        "cvae_median_ssim": float(cvae["ssim"].median()),
        "cvae_median_psd_log_distance": float(
            cvae["normalized_psd_log_distance"].median()
        ),
        "cvae_median_condition_descriptor_mae_z": float(
            cvae["condition_descriptor_mae_z"].median()
        ),
        "cvae_median_diversity_ratio": float(cvae["diversity_ratio"].median()),
        "cvae_exact_training_identity_count": int(
            cvae["exact_training_pixel_equality"].sum()
        ),
        "condition_permutation_correct_win_fraction": float(
            condition_control["correct_better_than_permuted"].mean()
        ),
        "condition_permutation_median_error_increase": float(
            condition_control["permutation_error_increase"].median()
        ),
        "log_rq_spearman": rq_correlation,
    }
    write_json(result, output / "evaluation_summary.json")
    return {
        "summary": result,
        "metrics": metrics,
        "metric_summary": summary,
        "condition_control": condition_control,
        "descriptor_predictions": descriptor_predictions_frame,
        "representatives": representative_payload,
        "sample_manifest": sample_manifest,
    }
