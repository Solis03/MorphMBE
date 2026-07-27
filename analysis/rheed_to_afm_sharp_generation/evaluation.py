from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_video_afm_story.afm_descriptors import describe_map
from analysis.rheed_video_afm_story.afm_evaluation import reconstruction_metrics
from analysis.rheed_video_afm_story.common import write_csv, write_json
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_generation.evaluation import (
    _condition_from_map,
    _map_medoid,
    _nearest_training_audit,
    _pairwise_l1,
)

from .spectral import load_unit_map


def texture_features(array: np.ndarray, border: int = 8) -> dict[str, float]:
    unit = project_unit_rq_np(array)
    gy, gx = np.gradient(unit)
    gradient = np.hypot(gx, gy)
    laplacian = (
        np.roll(unit, 1, 0)
        + np.roll(unit, -1, 0)
        + np.roll(unit, 1, 1)
        + np.roll(unit, -1, 1)
        - 4.0 * unit
    )
    edge_mask = np.zeros_like(unit, dtype=bool)
    edge_mask[:border] = True
    edge_mask[-border:] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    interior = ~edge_mask
    wrap = np.concatenate(
        [
            np.abs(unit[0] - unit[-1]),
            np.abs(unit[:, 0] - unit[:, -1]),
        ]
    )
    adjacent = np.concatenate(
        [np.abs(np.diff(unit, axis=0)).ravel(), np.abs(np.diff(unit, axis=1)).ravel()]
    )
    descriptor = describe_map(unit, "unit")
    return {
        "mean_gradient": float(np.mean(gradient)),
        "rms_gradient": float(np.sqrt(np.mean(gradient**2))),
        "laplacian_rms": float(np.sqrt(np.mean(laplacian**2))),
        "edge_energy_ratio": float(
            np.mean(gradient[edge_mask])
            / max(float(np.mean(gradient[interior])), 1e-8)
        ),
        "wrap_discontinuity_ratio": float(
            np.mean(wrap) / max(float(np.mean(adjacent)), 1e-8)
        ),
        "high_psd_fraction": float(descriptor["unit_psd_high_fraction"]),
        "robust_height_range": float(descriptor["unit_robust_height_range"]),
        "skewness": float(descriptor["unit_skewness"]),
        "kurtosis": float(descriptor["unit_kurtosis"]),
    }


def _median_feature_table(maps: list[np.ndarray]) -> dict[str, float]:
    table = pd.DataFrame([texture_features(array) for array in maps])
    return {str(column): float(table[column].median()) for column in table}


def _true_group_condition(
    group_rows: pd.DataFrame, scaler: ConditionScaler
) -> tuple[np.ndarray, np.ndarray]:
    raw = group_rows[scaler.columns].median().to_numpy(float)
    standardized = scaler.transform(raw[None], clip=False)[0]
    return raw, standardized


def evaluate_method_sets(
    *,
    split_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    generated: dict[str, dict[str, list[np.ndarray]]],
    generated_rq: dict[str, dict[str, float]],
    output_dir: str | Path,
    resolution: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_maps = [
        load_unit_map(row, resolution) for _, row in train_rows.iterrows()
    ]
    per_group: list[dict[str, Any]] = []
    panels: dict[str, dict[str, Any]] = {}
    for group_id, group_rows in split_rows.groupby("growth_run_id"):
        group = str(group_id)
        real_maps = [
            load_unit_map(row, resolution) for _, row in group_rows.iterrows()
        ]
        real_rqs = [float(value) for value in group_rows["rq_nm"]]
        real_index = _map_medoid(real_maps, real_rqs, condition_scaler)
        real_medoid = real_maps[real_index]
        true_rq = float(np.median(real_rqs))
        true_raw, true_z = _true_group_condition(group_rows, condition_scaler)
        real_texture = _median_feature_table(real_maps)
        panels[group] = {
            "real_medoid": real_medoid,
            "real_maps": real_maps,
            "true_rq": true_rq,
            "methods": {},
        }
        for method, method_groups in generated.items():
            samples = [
                project_unit_rq_np(array) for array in method_groups[group]
            ]
            rq = float(generated_rq[method][group])
            sample_conditions = np.stack(
                [
                    _condition_from_map(sample, rq, condition_scaler.columns)
                    for sample in samples
                ]
            )
            sample_z = condition_scaler.transform(
                np.median(sample_conditions, axis=0, keepdims=True),
                clip=False,
            )[0]
            index = _map_medoid(samples, [rq] * len(samples), condition_scaler)
            medoid = samples[index]
            morphology = reconstruction_metrics(real_medoid, medoid, true_rq)
            generated_texture = _median_feature_table(samples)
            row: dict[str, Any] = {
                "growth_run_id": group,
                "method": method,
                "true_rq_nm": true_rq,
                "generated_rq_nm": rq,
                "rq_absolute_error_nm": abs(rq - true_rq),
                "condition_descriptor_mae_z": float(
                    np.mean(np.abs(sample_z - true_z))
                ),
                "generated_pairwise_l1": _pairwise_l1(samples),
                "real_pairwise_l1": _pairwise_l1(real_maps),
                "diversity_ratio": _pairwise_l1(samples)
                / max(_pairwise_l1(real_maps), 1e-8),
            }
            row.update(morphology)
            row.update(_nearest_training_audit(medoid, training_maps))
            for feature, value in generated_texture.items():
                target = real_texture[feature]
                row[f"generated_{feature}"] = value
                row[f"real_{feature}"] = target
                row[f"{feature}_relative_error"] = abs(value - target) / max(
                    abs(target), 1e-8
                )
            row["sharpness_ratio"] = generated_texture["mean_gradient"] / max(
                real_texture["mean_gradient"], 1e-8
            )
            row["afm_texture_gate_pass"] = bool(
                0.65 <= row["sharpness_ratio"] <= 1.65
                and 0.50 <= generated_texture["edge_energy_ratio"] <= 1.80
                and row["laplacian_rms_relative_error"] <= 1.0
                and not row["exact_training_pixel_equality"]
            )
            per_group.append(row)
            panels[group]["methods"][method] = {
                "samples": samples,
                "medoid": medoid,
                "rq": rq,
            }
    frame = pd.DataFrame(per_group)
    summary_rows: list[dict[str, Any]] = []
    for method, method_rows in frame.groupby("method"):
        numeric = method_rows.select_dtypes(include=[np.number, bool])
        summary: dict[str, Any] = {
            "method": method,
            "group_count": int(len(method_rows)),
        }
        for column in numeric:
            summary[f"median_{column}"] = float(
                np.median(method_rows[column].astype(float))
            )
        summary["texture_gate_pass_fraction"] = float(
            method_rows["afm_texture_gate_pass"].mean()
        )
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["texture_gate_pass_fraction", "median_composite_score"],
        ascending=[False, True],
    )
    write_csv(frame, output / "per_group_metrics.csv")
    write_csv(summary_frame, output / "method_summary.csv")
    summary = {
        "group_count": int(split_rows["growth_run_id"].nunique()),
        "scan_count": int(len(split_rows)),
        "methods": list(summary_frame["method"]),
        "selection_policy": (
            "AFM texture gate first; among passing methods use morphology "
            "composite and condition-control evidence. No old test reuse."
        ),
    }
    write_json(summary, output / "evaluation_manifest.json")
    return {
        "per_group": frame,
        "summary": summary_frame,
        "panels": panels,
        "manifest": summary,
    }


def condition_permutation_control(
    *,
    groups: list[str],
    split_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    correct_maps: dict[str, list[np.ndarray]],
    wrong_maps: dict[str, list[np.ndarray]],
    generated_rq: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, group in enumerate(groups):
        wrong_group = groups[(position + 1) % len(groups)]
        truth = split_rows.loc[
            split_rows["growth_run_id"].astype(str) == group,
            condition_scaler.columns,
        ].median().to_numpy(float)
        truth_z = condition_scaler.transform(truth[None], clip=False)[0]

        def error(maps: list[np.ndarray], rq: float) -> float:
            values = np.stack(
                [
                    _condition_from_map(
                        array, rq, condition_scaler.columns
                    )
                    for array in maps
                ]
            )
            median = np.median(values, axis=0)
            z = condition_scaler.transform(median[None], clip=False)[0]
            return float(np.mean(np.abs(z - truth_z)))

        correct_error = error(correct_maps[group], generated_rq[group])
        wrong_error = error(wrong_maps[group], generated_rq[group])
        records.append(
            {
                "growth_run_id": group,
                "wrong_condition_source_group": wrong_group,
                "correct_condition_error_z": correct_error,
                "wrong_condition_error_z": wrong_error,
                "wrong_minus_correct": wrong_error - correct_error,
                "correct_condition_wins": wrong_error > correct_error,
            }
        )
    return pd.DataFrame(records)
