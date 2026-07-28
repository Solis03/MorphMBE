from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .evaluation import evaluate_island_methods
from .islands import (
    ISLAND_FEATURE_COLUMNS,
    IslandPrimitiveGenerator,
    fit_island_condition_model,
    group_island_feature_table,
)


M5 = "M5_cloudlike_spectral_hybrid"
M6A = "M6a_superellipse_island_primitives"
M6B = "M6b_multiscale_laguerre_terraces"
M6C = "M6c_island_structure_plus_spectral_prior"


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(repo_path(path).read_text(encoding="utf-8"))
    parent = json.loads(
        repo_path(config["parent_config"]).read_text(encoding="utf-8")
    )
    return {**parent, **config}


def _prediction_vector(
    row: pd.Series, columns: list[str]
) -> np.ndarray:
    return np.asarray(
        [row[f"selected_predicted_z__{column}"] for column in columns],
        dtype=np.float32,
    )


def _load_parent_arrays(
    config: dict[str, Any], group: str, *, validation: bool
) -> list[np.ndarray]:
    base = repo_path(config["parent_output"])
    if validation:
        path = base / "generated_maps" / (
            "M5_multiscale_spectral_hybrid"
        ) / f"{group}.npz"
    else:
        path = base / "training_group_cross_validation" / "generated_maps" / (
            "M5_multiscale_spectral_hybrid"
        ) / f"{group}.npz"
    payload = np.load(path, allow_pickle=False)
    return [
        np.asarray(array, dtype=np.float32)
        for array in payload["generated_unit_shapes"]
    ]


def _save_arrays(
    root: Path,
    *,
    method: str,
    group: str,
    arrays: list[np.ndarray],
    predicted_rq_nm: float,
    island_target: dict[str, float],
    condition_z: np.ndarray,
) -> None:
    path = root / "generated_maps" / method / f"{group}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_unit_shapes=np.stack(arrays).astype(np.float32),
        predicted_rq_nm=np.asarray(float(predicted_rq_nm)),
        condition_z=np.asarray(condition_z, dtype=np.float32),
        island_feature_target=np.asarray(
            [island_target[column] for column in ISLAND_FEATURE_COLUMNS],
            dtype=np.float32,
        ),
        island_feature_columns=np.asarray(ISLAND_FEATURE_COLUMNS),
        growth_run_id=np.asarray(group),
        method=np.asarray(method),
        retrieval_at_inference=np.asarray(False),
        measured_afm_patch_used_at_inference=np.asarray(False),
    )


def _blend(
    islands: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    weight: float,
) -> list[np.ndarray]:
    count = max(len(islands), len(spectral))
    return [
        project_unit_rq_np(
            weight * islands[index % len(islands)]
            + (1.0 - weight) * spectral[index % len(spectral)]
        ).astype(np.float32)
        for index in range(count)
    ]


def _method_ensembles(
    *,
    generator: IslandPrimitiveGenerator,
    island_target: dict[str, float],
    parent: list[np.ndarray],
    draws: int,
    seed: int,
    blend_weight: float,
) -> dict[str, list[np.ndarray]]:
    parent = parent[:draws]
    superellipse = generator.generate_ensemble(
        island_target,
        draws=draws,
        seed=seed,
        mode="superellipse",
    )
    laguerre = generator.generate_ensemble(
        island_target,
        draws=draws,
        seed=seed + 1000,
        mode="laguerre",
    )
    return {
        M5: parent,
        M6A: superellipse,
        M6B: laguerre,
        M6C: _blend(laguerre, parent, weight=blend_weight),
    }


def _aggregate(
    frames: list[pd.DataFrame],
    *,
    output: Path,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_group = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for method, values in per_group.groupby("method"):
        record: dict[str, Any] = {
            "method": method,
            "growth_group_count": int(values["growth_run_id"].nunique()),
        }
        numeric = values.select_dtypes(include=[np.number, bool])
        for column in numeric:
            record[f"median_{column}"] = float(
                values[column].astype(float).median()
            )
        rows.append(record)
    summary = pd.DataFrame(rows).sort_values("method")
    write_csv(per_group, output / f"{name}_per_group.csv")
    write_csv(summary, output / f"{name}_summary.csv")
    return per_group, summary


def run_experiment(config: dict[str, Any], *, smoke: bool) -> None:
    suffix = "smoke" if smoke else "development"
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(config)
    descriptors = tables["descriptors"].copy()
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[
        descriptors["split"] == "val"
    ].copy()
    if (descriptors["split"] == "test").any():
        # Test rows may exist in the manifest, but are never passed below.
        pass
    columns = list(config["condition_columns"])
    cross_predictions = pd.read_csv(
        repo_path(config["parent_report"])
        / "training_group_cross_validation"
        / "condition_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    validation_predictions = pd.read_csv(
        repo_path(config["parent_report"])
        / "validation_condition_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    all_crossfit_groups = sorted(
        train_rows["growth_run_id"].astype(str).unique()
    )
    groups = list(all_crossfit_groups)
    if smoke:
        groups = groups[:3]
    draws = 2 if smoke else int(config["crossfit_draws"])
    generator = IslandPrimitiveGenerator(
        resolution=int(config["resolution"]),
        laguerre_count_factor=float(
            config.get("laguerre_count_factor", 3.0)
        ),
        fine_count_factor=float(config.get("fine_count_factor", 3.0)),
    )
    cross_standard_frames: list[pd.DataFrame] = []
    cross_island_frames: list[pd.DataFrame] = []
    target_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    for fold, held in enumerate(groups):
        fit_groups = set(all_crossfit_groups)
        fit_groups.remove(held)
        fit_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str).isin(fit_groups)
        ]
        held_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str) == held
        ]
        scaler = ConditionScaler.fit(train_rows, columns, fit_groups)
        island_model, cv, targets = fit_island_condition_model(
            train_rows=fit_rows,
            condition_scaler=scaler,
            resolution=int(config["resolution"]),
            alphas=config["island_ridge_alphas"],
        )
        condition_z = _prediction_vector(
            cross_predictions.loc[held], columns
        )
        island_target = island_model.predict(condition_z)
        parent = _load_parent_arrays(config, held, validation=False)
        predicted_rq = float(
            cross_predictions.loc[held, "selected_predicted_rq_nm"]
        )
        seed = int(config["seed"]) + fold * 10_000
        methods = _method_ensembles(
            generator=generator,
            island_target=island_target,
            parent=parent,
            draws=draws,
            seed=seed,
            blend_weight=float(config["selected_blend_weight"]),
        )
        rq = {
            method: {held: predicted_rq} for method in methods
        }
        nested = {method: {held: arrays} for method, arrays in methods.items()}
        evaluation = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=fit_rows,
            condition_scaler=scaler,
            generated=nested,
            generated_rq=rq,
            output_dir=report / "crossfit" / "folds" / held / "standard",
            resolution=int(config["resolution"]),
        )
        standard = evaluation["per_group"].copy()
        standard.insert(0, "cross_validation_fold", fold)
        cross_standard_frames.append(standard)
        island = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=fit_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        island.insert(0, "cross_validation_fold", fold)
        write_csv(
            island,
            report / "crossfit" / "folds" / held / "island_metrics.csv",
        )
        write_csv(
            cv,
            report / "crossfit" / "folds" / held / "island_ridge_cv.csv",
        )
        cross_island_frames.append(island)
        true_target = group_island_feature_table(
            held_rows, resolution=int(config["resolution"])
        ).iloc[0]
        target_record: dict[str, Any] = {
            "growth_run_id": held,
            "ridge_alpha": island_model.alpha,
        }
        for column in ISLAND_FEATURE_COLUMNS:
            target_record[f"predicted__{column}"] = island_target[column]
            target_record[f"true__{column}"] = float(true_target[column])
        target_records.append(target_record)
        for weight in config["blend_weights"]:
            arrays = _blend(
                methods[M6B], parent[:draws], weight=float(weight)
            )
            frame = evaluate_island_methods(
                held_rows=held_rows,
                train_rows=fit_rows,
                generated={f"weight_{float(weight):.2f}": arrays},
                resolution=int(config["resolution"]),
            )
            frame["blend_weight"] = float(weight)
            weight_frames.append(frame)
        for method, arrays in methods.items():
            _save_arrays(
                output / "crossfit",
                method=method,
                group=held,
                arrays=arrays,
                predicted_rq_nm=predicted_rq,
                island_target=island_target,
                condition_z=condition_z,
            )
    cross_standard, standard_summary = _aggregate(
        cross_standard_frames, output=report / "crossfit", name="standard"
    )
    cross_island, island_summary = _aggregate(
        cross_island_frames, output=report / "crossfit", name="island"
    )
    write_csv(
        pd.DataFrame(target_records),
        report / "crossfit" / "island_target_predictions.csv",
    )
    weight_table = pd.concat(weight_frames, ignore_index=True)
    write_csv(weight_table, report / "crossfit" / "blend_ablation.csv")
    weight_summary = (
        weight_table.groupby("blend_weight")
        .agg(
            median_island_feature_mae_z=(
                "island_feature_mae_z",
                "median",
            ),
            median_afm_prior_mahalanobis=(
                "afm_prior_mahalanobis",
                "median",
            ),
            median_afm_likeness_percentile=(
                "afm_likeness_percentile",
                "median",
            ),
        )
        .reset_index()
    )
    write_csv(weight_summary, report / "crossfit" / "blend_ablation_summary.csv")

    # Pre-existing validation: one frozen training fit, no test rows.
    all_train_groups = set(
        train_rows["growth_run_id"].astype(str).unique()
    )
    scaler = ConditionScaler.fit(train_rows, columns, all_train_groups)
    island_model, validation_cv, train_targets = fit_island_condition_model(
        train_rows=train_rows,
        condition_scaler=scaler,
        resolution=int(config["resolution"]),
        alphas=config["island_ridge_alphas"],
    )
    validation_methods: dict[str, dict[str, list[np.ndarray]]] = {
        method: {} for method in (M5, M6A, M6B, M6C)
    }
    validation_rq: dict[str, dict[str, float]] = {
        method: {} for method in validation_methods
    }
    validation_island_frames = []
    validation_target_records = []
    validation_groups = sorted(
        validation_rows["growth_run_id"].astype(str).unique()
    )
    for index, group in enumerate(validation_groups):
        condition_z = _prediction_vector(
            validation_predictions.loc[group], columns
        )
        island_target = island_model.predict(condition_z)
        parent = _load_parent_arrays(config, group, validation=True)
        methods = _method_ensembles(
            generator=generator,
            island_target=island_target,
            parent=parent,
            draws=2 if smoke else int(config["draws"]),
            seed=int(config["seed"]) + 500_000 + index * 10_000,
            blend_weight=float(config["selected_blend_weight"]),
        )
        predicted_rq = float(
            validation_predictions.loc[
                group, "selected_predicted_rq_nm"
            ]
        )
        for method, arrays in methods.items():
            validation_methods[method][group] = arrays
            validation_rq[method][group] = predicted_rq
            _save_arrays(
                output,
                method=method,
                group=group,
                arrays=arrays,
                predicted_rq_nm=predicted_rq,
                island_target=island_target,
                condition_z=condition_z,
            )
        held_rows = validation_rows.loc[
            validation_rows["growth_run_id"].astype(str) == group
        ]
        frame = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=train_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        validation_island_frames.append(frame)
        true_target = group_island_feature_table(
            held_rows, resolution=int(config["resolution"])
        ).iloc[0]
        record: dict[str, Any] = {"growth_run_id": group}
        for column in ISLAND_FEATURE_COLUMNS:
            record[f"predicted__{column}"] = island_target[column]
            record[f"true__{column}"] = float(true_target[column])
        validation_target_records.append(record)
    validation_standard = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=scaler,
        generated=validation_methods,
        generated_rq=validation_rq,
        output_dir=report / "validation" / "standard",
        resolution=int(config["resolution"]),
    )
    validation_island = pd.concat(
        validation_island_frames, ignore_index=True
    )
    write_csv(
        validation_island,
        report / "validation" / "island_per_group.csv",
    )
    _, validation_island_summary = _aggregate(
        validation_island_frames,
        output=report / "validation",
        name="island_aggregate",
    )
    write_csv(
        pd.DataFrame(validation_target_records),
        report / "validation" / "island_target_predictions.csv",
    )
    write_csv(
        validation_cv, report / "validation" / "island_ridge_cv.csv"
    )
    write_csv(
        train_targets.reset_index(),
        report / "validation" / "training_island_targets.csv",
    )
    manifest = {
        "experiment": "RHEED-conditioned island-aware AFM generation",
        "smoke": smoke,
        "methods": [M5, M6A, M6B, M6C],
        "selected_blend_weight": float(config["selected_blend_weight"]),
        "training_growth_groups": sorted(all_train_groups),
        "validation_growth_groups": validation_groups,
        "historical_test_used": False,
        "removelist_sha256": sha256_file(
            repo_path(config["removelist_path"])
        ),
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "standard_crossfit_summary": standard_summary.to_dict(
            orient="records"
        ),
        "island_crossfit_summary": island_summary.to_dict(
            orient="records"
        ),
        "validation_standard_summary": validation_standard[
            "summary"
        ].to_dict(orient="records"),
        "validation_island_summary": validation_island_summary.to_dict(
            orient="records"
        ),
        "claim_boundary": (
            "Method selection uses strict training-growth-group crossfit and "
            "the pre-existing validation cohort. The consumed historical "
            "test cohort remains closed."
        ),
    }
    write_json(manifest, report / "experiment_manifest.json")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation.json",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(load_config(args.config), smoke=bool(args.smoke))


if __name__ == "__main__":
    main()
