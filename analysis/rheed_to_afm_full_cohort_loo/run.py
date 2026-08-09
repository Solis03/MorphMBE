from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_distinct_confidence.matern import (
    DescriptorMaternGenerator,
)
from analysis.rheed_to_afm_distinct_confidence.run import (
    _blend_ensembles,
    _group_targets,
    _predictor_factory,
)
from analysis.rheed_to_afm_distinct_confidence.run import (
    _load_tables as load_source_tables,
)
from analysis.rheed_to_afm_distinct_confidence.variance import (
    fit_variance_calibrator,
)
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    crossfit_target,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    group_metric_table,
    scan_metric_table,
)
from analysis.rheed_to_afm_functional_morphology.render import (
    render_ensemble,
)
from analysis.rheed_to_afm_functional_morphology.run import (
    M10,
    _aggregate,
    _confidence_table,
    _physics_table,
    _prediction_metrics,
    _save_generated,
    _surface_method_metrics,
)
from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    aggregate_group_conditions,
    predict_groups,
)
from analysis.rheed_to_afm_generation.training import resolve_device
from analysis.rheed_to_afm_island_generation.evaluate_topology_renderers import (
    _structure_blend,
)
from analysis.rheed_to_afm_island_generation.evaluation import (
    evaluate_island_methods,
)
from analysis.rheed_to_afm_island_generation.islands import (
    IslandPrimitiveGenerator,
    fit_island_condition_model,
)
from analysis.rheed_to_afm_sharp_generation.cross_validation import (
    _calibrate,
    _generate,
)
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_to_afm_sharp_generation.spectral import (
    fit_conditional_spectral_model,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

FULL_SPLIT = "retrospective_full23_loo"


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    base_path = payload.pop("base_config", None)
    if base_path is None:
        return payload
    base = load_config(str(base_path))
    base.update(payload)
    return base


def prepare_full_cohort(
    tables: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the configured cohort and retain old splits as provenance."""

    descriptors = tables["descriptors"].copy()
    descriptors["sample_id"] = descriptors["sample_id"].astype(str)
    descriptors["growth_run_id"] = descriptors["growth_run_id"].astype(str)
    descriptors["source_split"] = descriptors["split"].astype(str)
    descriptors["split"] = str(config.get("split_label", FULL_SPLIT))
    groups = sorted(descriptors["growth_run_id"].unique())
    expected = int(config["expected_growth_count"])
    if len(groups) != expected:
        raise RuntimeError(
            f"expected {expected} harmonized growths, found {len(groups)}"
        )
    excluded = set(map(str, config["explicitly_excluded_growths"]))
    overlap = sorted(set(groups) & excluded)
    if overlap:
        raise RuntimeError(
            f"explicitly excluded growths entered full cohort: {overlap}"
        )
    removed_overlap = sorted(
        set(groups) & set(map(str, tables["removelist"].sample_ids))
    )
    if removed_overlap:
        raise RuntimeError(
            f"removelist growths entered full cohort: {removed_overlap}"
        )
    source_split = (
        descriptors[["growth_run_id", "source_split"]]
        .drop_duplicates()
        .set_index("growth_run_id")["source_split"]
        .to_dict()
    )
    if len(source_split) != expected:
        raise RuntimeError("growth-to-source-split mapping is not unique")
    physics_groups = set(tables["physics"]["growth_run_id"].astype(str))
    missing_physics = sorted(set(groups) - physics_groups)
    if missing_physics:
        raise RuntimeError(
            f"full-cohort RHEED physics missing: {missing_physics}"
        )
    return descriptors, source_split


def _target_series(
    descriptors: pd.DataFrame, group_metrics: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    log_rq = aggregate_group_conditions(
        descriptors,
        ["log_rq_nm"],
    )["log_rq_nm"]
    log_rq.index = log_rq.index.astype(str)
    log_rq.name = "log_rq_nm"
    fsmi = group_metrics.set_index("growth_run_id")[
        "functional_surface_morphology_index_nm"
    ].sort_index()
    fsmi.index = fsmi.index.astype(str)
    log_fsmi = np.log(np.clip(fsmi.astype(float), 1e-6, None))
    log_fsmi.name = "log_fsmi_nm"
    return log_rq.sort_index(), log_fsmi.sort_index()


def _load_external_predictions(
    *,
    path: str | Path,
    groups: list[str],
    log_target: pd.Series,
) -> pd.DataFrame:
    frame = pd.read_csv(
        repo_path(path), dtype={"growth_run_id": str}
    )
    if "target" in frame.columns:
        target_lookup = {
            "log_rq_nm": "Rq_nm",
            "log_fsmi_nm": "FSMI_nm",
        }
        target = target_lookup.get(str(log_target.name))
        if target is None:
            raise RuntimeError(
                "cannot infer the target row from external predictions for "
                f"series {log_target.name!r}"
            )
        frame = frame.loc[frame["target"] == target].copy()
    required = {
        "growth_run_id",
        "true_target",
        "predicted_target",
        "absolute_error",
        "predicted_absolute_error",
        "interval_lower",
        "interval_upper",
        "interval_covered",
        "outer_target_used_for_training",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"external prediction columns missing from {path}: {missing}"
        )
    if set(frame["growth_run_id"]) != set(groups) or len(frame) != len(
        groups
    ):
        raise RuntimeError(
            f"external predictions do not match full cohort: {path}"
        )
    expected = np.exp(log_target.loc[groups].to_numpy(float))
    actual = (
        frame.set_index("growth_run_id")
        .loc[groups, "true_target"]
        .to_numpy(float)
    )
    if not np.allclose(expected, actual, rtol=1e-7, atol=1e-7):
        raise RuntimeError(
            f"external prediction truth values do not match: {path}"
        )
    if frame["outer_target_used_for_training"].astype(bool).any():
        raise RuntimeError(
            f"external predictions report outer target leakage: {path}"
        )
    return frame.sort_values("growth_run_id").reset_index(drop=True)


def _condition_with_amplitude(
    selected_z: np.ndarray,
    scaler: ConditionScaler,
    predicted_rq_nm: float,
) -> np.ndarray:
    result = np.asarray(selected_z, dtype=np.float32).copy()
    rq_position = scaler.columns.index("log_rq_nm")
    result[rq_position] = float(
        (np.log(max(float(predicted_rq_nm), 1e-6)) - scaler.mean[rq_position])
        / scaler.scale[rq_position]
    )
    return result


def _summarize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, rows in frame.groupby("method"):
        record: dict[str, Any] = {
            "method": method,
            "growth_group_count": int(rows["growth_run_id"].nunique()),
        }
        for column in rows.select_dtypes(include=[np.number, bool]).columns:
            record[f"median_{column}"] = float(
                rows[column].astype(float).median()
            )
            record[f"mean_{column}"] = float(
                rows[column].astype(float).mean()
            )
        records.append(record)
    return pd.DataFrame(records).sort_values("method").reset_index(drop=True)


def _comparison_target_rows(
    *,
    current_rq: pd.DataFrame,
    current_fsmi: pd.DataFrame,
    prior_report: Path,
    current_protocol: str = "current_full23_train22_all23",
    current_same_protocol: str = "current_full23_train22_same15",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_rq = pd.read_csv(
        prior_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    prior_fsmi = pd.read_csv(
        prior_report / "fsmi_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    prior_groups = set(prior_rq["growth_run_id"].astype(str))
    current_rq_same = current_rq.loc[
        current_rq["growth_run_id"].astype(str).isin(prior_groups)
    ].copy()
    current_fsmi_same = current_fsmi.loc[
        current_fsmi["growth_run_id"].astype(str).isin(prior_groups)
    ].copy()
    rows = []
    for protocol, rq, fsmi in (
        ("prior_M12_strict15_train14", prior_rq, prior_fsmi),
        (current_protocol, current_rq, current_fsmi),
        (
            current_same_protocol,
            current_rq_same,
            current_fsmi_same,
        ),
    ):
        for target, table in (("Rq_nm", rq), ("FSMI_nm", fsmi)):
            rows.append(
                {
                    "protocol": protocol,
                    **_prediction_metrics(table, label=target),
                }
            )
    per_group = prior_rq[
        ["growth_run_id", "true_target", "predicted_target", "absolute_error"]
    ].rename(
        columns={
            "predicted_target": "prior_predicted_rq_nm",
            "absolute_error": "prior_rq_absolute_error_nm",
        }
    ).merge(
        current_rq_same[
            [
                "growth_run_id",
                "predicted_target",
                "absolute_error",
            ]
        ].rename(
            columns={
                "predicted_target": "current_predicted_rq_nm",
                "absolute_error": "current_rq_absolute_error_nm",
            }
        ),
        on="growth_run_id",
    )
    per_group["rq_absolute_error_change_nm"] = (
        per_group["current_rq_absolute_error_nm"]
        - per_group["prior_rq_absolute_error_nm"]
    )
    return pd.DataFrame(rows), per_group


def run(config: dict[str, Any], *, smoke: bool, device_name: str) -> None:
    started = time.time()
    suffix = "smoke" if smoke else str(
        config.get("full_run_suffix", "full23_loo")
    )
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    tables = load_source_tables(config)
    descriptors, source_split = prepare_full_cohort(tables, config)
    groups_all = sorted(descriptors["growth_run_id"].astype(str).unique())
    extra_batch = set(map(str, config.get("extra_batch_growths", [])))
    if not extra_batch.issubset(set(groups_all)):
        raise RuntimeError(
            "configured extra-batch growths are absent from the cohort: "
            f"{sorted(extra_batch - set(groups_all))}"
        )
    write_csv(
        pd.DataFrame(
            {
                "growth_run_id": groups_all,
                "source_split": [source_split[group] for group in groups_all],
                "cohort_origin": [
                    "extra_five_batch"
                    if group in extra_batch
                    else "original_23_batch"
                    for group in groups_all
                ],
                "full_cohort_role": "outer_loo_once",
            }
        ),
        report / "cohort_manifest.csv",
    )

    scan_metrics = scan_metric_table(
        descriptors,
        scan_size_nm=float(config["scan_size_nm"]),
        analysis_scale_nm=float(config["analysis_scale_nm"]),
    )
    group_metrics = group_metric_table(scan_metrics)
    write_csv(scan_metrics, report / "surface_metrics_per_scan.csv")
    write_csv(group_metrics, report / "surface_metrics_per_group.csv")
    log_rq, log_fsmi = _target_series(descriptors, group_metrics)
    physics = _physics_table(tables["physics"])
    external = config.get("external_target_predictions")
    if external:
        cross_rq_all = _load_external_predictions(
            path=external["Rq_nm"],
            groups=groups_all,
            log_target=log_rq,
        )
        cross_fsmi_all = _load_external_predictions(
            path=external["FSMI_nm"],
            groups=groups_all,
            log_target=log_fsmi,
        )
        inner_rq = pd.DataFrame()
        inner_fsmi = pd.DataFrame()
    else:
        cross_rq_all, inner_rq = crossfit_target(
            physics=physics,
            log_target=log_rq,
            alpha=float(config["ridge_alpha"]),
            morphology_weight=float(config["morphology_head_weight"]),
            confidence_alpha=float(config["confidence_alpha"]),
        )
        cross_fsmi_all, inner_fsmi = crossfit_target(
            physics=physics,
            log_target=log_fsmi,
            alpha=float(config["ridge_alpha"]),
            morphology_weight=float(config["morphology_head_weight"]),
            confidence_alpha=float(config["confidence_alpha"]),
        )
    if smoke and config.get("smoke_growths"):
        requested_smoke_groups = [
            str(group) for group in config["smoke_growths"]
        ]
        unknown_smoke_groups = sorted(
            set(requested_smoke_groups) - set(groups_all)
        )
        if unknown_smoke_groups:
            raise RuntimeError(
                "smoke_growths contains unavailable or excluded growths: "
                f"{unknown_smoke_groups}"
            )
        groups = requested_smoke_groups
    else:
        groups = groups_all[
            : int(config["smoke_growth_count"])
        ] if smoke else groups_all
    cross_rq = cross_rq_all.loc[
        cross_rq_all["growth_run_id"].isin(groups)
    ].copy()
    cross_fsmi = cross_fsmi_all.loc[
        cross_fsmi_all["growth_run_id"].isin(groups)
    ].copy()
    for frame, stem in (
        (cross_rq, "rq_crossfit_predictions"),
        (cross_fsmi, "fsmi_crossfit_predictions"),
        (inner_rq, "rq_nested_inner_predictions"),
        (inner_fsmi, "fsmi_nested_inner_predictions"),
    ):
        write_csv(frame, report / f"{stem}.csv")

    rq_lookup = cross_rq.set_index("growth_run_id")
    fsmi_lookup = cross_fsmi.set_index("growth_run_id")
    metric_lookup = group_metrics.set_index("growth_run_id")
    condition_targets = _group_targets(
        descriptors, list(config["condition_columns"])
    )
    fit_predictor = _predictor_factory(
        config=config,
        tables=tables,
        group_targets=condition_targets,
    )
    island_generator = IslandPrimitiveGenerator(
        resolution=int(config["resolution"]),
        laguerre_count_factor=float(config["laguerre_count_factor"]),
        fine_count_factor=float(config["fine_count_factor"]),
    )
    device = resolve_device(device_name)
    standard_frames: list[pd.DataFrame] = []
    island_frames: list[pd.DataFrame] = []
    surface_frames: list[pd.DataFrame] = []
    condition_records: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []

    for evaluation_index, held in enumerate(groups):
        fold = groups_all.index(held)
        fold_started = time.time()
        fit_groups = [group for group in groups_all if group != held]
        fit_group_set = set(fit_groups)
        fit_rows = descriptors.loc[
            descriptors["growth_run_id"].isin(fit_group_set)
        ].copy()
        held_rows = descriptors.loc[
            descriptors["growth_run_id"] == held
        ].copy()
        scaler = ConditionScaler.fit(
            descriptors,
            list(config["condition_columns"]),
            fit_group_set,
        )

        variance_calibrator, inner_conditions = fit_variance_calibrator(
            groups=fit_groups,
            group_targets=condition_targets,
            enclosing_scaler=scaler,
            fit_predictor=fit_predictor,
            registry=tables["registry"],
            physics=tables["physics"],
            cap=float(config["variance_cap"]),
            minimum_predicted_std=float(config["minimum_predicted_std"]),
        )
        predictor = fit_predictor(fit_groups, scaler)
        raw_condition, raw_z, _ = predict_groups(
            predictor,
            [held],
            tables["registry"],
            tables["physics"],
        )
        selected_z = variance_calibrator.transform_z(raw_z)[0]
        predicted_rq = float(rq_lookup.loc[held, "predicted_target"])
        predicted_fsmi = float(fsmi_lookup.loc[held, "predicted_target"])
        isolation_value = rq_lookup.loc[held].get(
            "rheed_spot_isolation_score", 0.50
        )
        predicted_isolation = (
            0.50 if pd.isna(isolation_value) else float(isolation_value)
        )
        condition_z = _condition_with_amplitude(
            selected_z, scaler, predicted_rq
        )

        matern_generator = DescriptorMaternGenerator(
            scaler, resolution=int(config["resolution"])
        )
        fold_seed = int(config["seed"]) + fold * 10_000
        matern = matern_generator.generate_ensemble(
            selected_z,
            draws=int(config["draws"]),
            seed=fold_seed,
        )
        spectral_model, spectral_cv, _ = fit_conditional_spectral_model(
            train_rows=fit_rows,
            condition_scaler=scaler,
            alphas=[float(value) for value in config["spectral_ridge_alphas"]],
            resolution=int(config["resolution"]),
            removelist_sample_ids=tables["removelist"].sample_ids,
        )
        spectral_raw = _generate(
            spectral_model,
            raw_z[0],
            draws=int(config["draws"]),
            iterations=int(config["spectral_iaaft_iterations"]),
            seed=fold_seed + 700_000,
        )
        spectral = _calibrate(
            spectral_raw,
            raw_z[0],
            scaler=scaler,
            config=config,
            device=device,
        )
        m5 = _blend_ensembles(
            matern,
            spectral,
            primary_weight=float(config["matern_blend_weight"]),
        )

        island_model, island_cv, _ = fit_island_condition_model(
            train_rows=fit_rows,
            condition_scaler=scaler,
            resolution=int(config["resolution"]),
            alphas=config["island_ridge_alphas"],
        )
        island_target = dict(island_model.predict(condition_z))
        island_target["rheed_spot_isolation_score"] = predicted_isolation
        baseline_structure = island_generator.generate_ensemble(
            island_target,
            draws=int(config["draws"]),
            seed=fold_seed + 300_000,
            mode="laguerre",
        )
        m10 = _structure_blend(
            baseline_structure,
            m5,
            weight=float(config["m10_structure_weight"]),
        )
        selected_method = str(config["selected_method"])
        renderer_definitions = config.get("candidate_renderers")
        if renderer_definitions is None:
            renderer_definitions = {
                selected_method: config["selected_renderer"]
            }
        if selected_method not in renderer_definitions:
            raise RuntimeError(
                "selected method is absent from candidate_renderers"
            )
        structure_cache: dict[str, list[np.ndarray]] = {
            "laguerre": baseline_structure
        }
        requested_modes = sorted(
            {
                str(
                    renderer_parameters.get(
                        "island_generator_mode", "laguerre"
                    )
                )
                for renderer_parameters in renderer_definitions.values()
            }
            - {"laguerre"}
        )
        for mode_index, generator_mode in enumerate(
            requested_modes, start=1
        ):
            structure_cache[generator_mode] = (
                island_generator.generate_ensemble(
                    island_target,
                    draws=int(config["draws"]),
                    seed=fold_seed + 300_000 + mode_index * 100_000,
                    mode=generator_mode,
                )
            )
        methods = {M10: m10}
        for renderer_name, renderer_parameters in renderer_definitions.items():
            parameters = dict(
                config.get("candidate_renderer_defaults", {})
            )
            parameters.update(renderer_parameters)
            generator_mode = str(
                parameters.pop("island_generator_mode", "laguerre")
            )
            methods[str(renderer_name)] = render_ensemble(
                structure_cache[generator_mode],
                m5,
                baseline_structure=baseline_structure,
                conditioning_sq_nm=predicted_rq,
                island_target=island_target,
                rough_isolation_score=predicted_isolation,
                **parameters,
            )
        for method, arrays in methods.items():
            _save_generated(
                output / "crossfit",
                method=method,
                group=held,
                arrays=arrays,
                predicted_rq_nm=predicted_rq,
                predicted_fsmi_nm=predicted_fsmi,
                condition_z=condition_z,
            )

        generated_rq = {
            method: {held: predicted_rq} for method in methods
        }
        standard = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=fit_rows,
            condition_scaler=scaler,
            generated={
                method: {held: arrays}
                for method, arrays in methods.items()
            },
            generated_rq=generated_rq,
            output_dir=report / "crossfit" / "folds" / held / "standard",
            resolution=int(config["resolution"]),
        )["per_group"]
        standard.insert(0, "cross_validation_fold", fold)
        standard_frames.append(standard)
        island = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=fit_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        island.insert(0, "cross_validation_fold", fold)
        island_frames.append(island)
        true_fsmi = float(
            metric_lookup.loc[
                held, "functional_surface_morphology_index_nm"
            ]
        )
        surface_frames.append(
            _surface_method_metrics(
                group=held,
                true_fsmi_nm=true_fsmi,
                predicted_rq_nm={
                    method: predicted_rq for method in methods
                },
                arrays=methods,
                scan_size_nm=float(config["scan_size_nm"]),
                analysis_scale_nm=float(config["analysis_scale_nm"]),
            )
        )

        true_raw = condition_targets.loc[
            held, scaler.columns
        ].to_numpy(float)
        true_z = scaler.transform(true_raw[None], clip=False)[0]
        record: dict[str, Any] = {
            "growth_run_id": held,
            "source_split": source_split[held],
            "outer_fit_growth_count": len(fit_groups),
            "true_rq_nm": float(np.exp(true_raw[0])),
            "amplitude_predicted_rq_nm": predicted_rq,
            "rheed_spot_isolation_score": predicted_isolation,
            "amplitude_predicted_fsmi_nm": predicted_fsmi,
            "condition_descriptor_mae_z": float(
                np.mean(np.abs(condition_z - true_z))
            ),
            "variance_factors": json.dumps(
                variance_calibrator.factors.tolist()
            ),
        }
        for position, column in enumerate(scaler.columns):
            record[f"true_z__{column}"] = float(true_z[position])
            record[f"raw_predicted_z__{column}"] = float(raw_z[0, position])
            record[f"selected_predicted_z__{column}"] = float(
                condition_z[position]
            )
            record[f"raw_predicted_raw__{column}"] = float(
                raw_condition[0, position]
            )
        condition_records.append(record)
        fold_audit = {
            "outer_fold": fold,
            "held_growth_run_id": held,
            "held_source_split": source_split[held],
            "fit_growth_count": len(fit_groups),
            "fit_growth_run_ids": fit_groups,
            "held_overlap_with_fit": bool(held in fit_group_set),
            "condition_inner_growth_count": int(
                inner_conditions["growth_run_id"].nunique()
            ),
            "spectral_train_growth_count": len(spectral_model.train_groups),
            "spectral_train_growth_run_ids": spectral_model.train_groups,
            "island_train_growth_count": len(island_model.train_groups),
            "island_train_growth_run_ids": island_model.train_groups,
            "spectral_selected_alpha": float(spectral_model.alpha),
            "island_selected_alpha": float(island_model.alpha),
            "runtime_seconds": time.time() - fold_started,
        }
        if (
            fold_audit["held_overlap_with_fit"]
            or held in fold_audit["spectral_train_growth_run_ids"]
            or held in fold_audit["island_train_growth_run_ids"]
        ):
            raise RuntimeError(f"outer-fold leakage detected for {held}")
        fold_audits.append(fold_audit)
        fold_dir = report / "crossfit" / "folds" / held
        write_csv(spectral_cv, fold_dir / "spectral_inner_cv.csv")
        write_csv(island_cv, fold_dir / "island_inner_cv.csv")
        write_csv(
            inner_conditions,
            fold_dir / "condition_variance_inner_loo.csv",
        )
        write_json(fold_audit, fold_dir / "fold_manifest.json")
        print(
            f"[{evaluation_index + 1:02d}/{len(groups):02d}] held={held} "
            f"fit={len(fit_groups)} runtime={fold_audit['runtime_seconds']:.1f}s",
            flush=True,
        )

    standard, standard_summary = _aggregate(
        standard_frames, output=report / "crossfit", stem="standard"
    )
    island, island_summary = _aggregate(
        island_frames, output=report / "crossfit", stem="island"
    )
    _surface, surface_summary = _aggregate(
        surface_frames,
        output=report / "crossfit",
        stem="functional_surface",
    )
    write_csv(
        pd.DataFrame(condition_records),
        report / "condition_predictions.csv",
    )
    write_csv(pd.DataFrame(fold_audits), report / "fold_integrity_audit.csv")

    selected_method = str(config["selected_method"])
    confidence, confidence_manifest = _confidence_table(
        rq_predictions=cross_rq,
        fsmi_predictions=cross_fsmi,
        standard=standard,
        island=island,
        method=selected_method,
    )
    write_csv(confidence, report / "confidence_crossfit.csv")
    write_json(confidence_manifest, report / "confidence_manifest.json")
    target_summary = pd.DataFrame(
        [
            _prediction_metrics(cross_rq, label="Rq_nm"),
            _prediction_metrics(cross_fsmi, label="FSMI_nm"),
        ]
    )
    write_csv(target_summary, report / "target_prediction_summary.csv")
    method_summary = (
        standard_summary.merge(
            island_summary, on="method", suffixes=("_standard", "_island")
        )
        .merge(surface_summary, on="method", suffixes=("", "_surface"))
    )
    write_csv(method_summary, report / "method_summary.csv")

    cohort_count = len(groups_all)
    fit_count = cohort_count - 1
    current_protocol = (
        f"current_full{cohort_count}_train{fit_count}_all{cohort_count}"
    )
    current_same_protocol = (
        f"current_full{cohort_count}_train{fit_count}_same15"
    )
    comparison, paired = _comparison_target_rows(
        current_rq=cross_rq,
        current_fsmi=cross_fsmi,
        prior_report=repo_path(config["prior_m12_report"]),
        current_protocol=current_protocol,
        current_same_protocol=current_same_protocol,
    )
    write_csv(comparison, report / "comparison_to_prior15_targets.csv")
    write_csv(paired, report / "comparison_to_prior15_per_group.csv")
    prior_standard = pd.read_csv(
        repo_path(config["prior_m12_report"])
        / "crossfit"
        / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    current_prior_groups = standard.loc[
        standard["growth_run_id"].isin(paired["growth_run_id"])
    ]
    image_comparison = pd.concat(
        [
            _summarize_methods(
                prior_standard.loc[
                    prior_standard["method"].isin([M10, selected_method])
                ]
            ).assign(protocol="prior_M12_strict15_train14"),
            _summarize_methods(current_prior_groups).assign(
                protocol=current_same_protocol
            ),
            _summarize_methods(standard).assign(
                protocol=current_protocol
            ),
        ],
        ignore_index=True,
    )
    write_csv(
        image_comparison, report / "comparison_to_prior15_image_metrics.csv"
    )

    manifest = {
        "experiment_id": config["experiment_id"],
        "mode": suffix,
        "protocol": (
            "retrospective nested leave-one-growth-out over the fixed "
            f"harmonized {cohort_count}-growth cohort"
        ),
        "outer_growth_group_count": len(groups),
        "full_available_growth_group_count": len(groups_all),
        "outer_fit_growth_count": len(groups_all) - 1,
        "growth_run_ids": groups,
        "source_split_provenance": source_split,
        "explicitly_excluded_growths": list(
            map(str, config["explicitly_excluded_growths"])
        ),
        "selected_method": selected_method,
        "target_prediction_method": config.get(
            "target_prediction_method", "M12a_frozen_alpha_head"
        ),
        "external_target_prediction_files": (
            {
                target: {
                    "path": str(path),
                    "sha256": sha256_file(repo_path(path)),
                }
                for target, path in external.items()
            }
            if external
            else None
        ),
        "selected_method_frozen_before_expanded_cohort_run": bool(
            config.get(
                "selected_method_frozen_before_expanded_cohort_run",
                True,
            )
        ),
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "all_outer_fold_leakage_checks_passed": bool(
            all(not row["held_overlap_with_fit"] for row in fold_audits)
        ),
        "removelist_sha256": sha256_file(
            repo_path(config["removelist_path"])
        ),
        "target_prediction_summary": target_summary.to_dict(
            orient="records"
        ),
        "confidence": confidence_manifest,
        "method_summary": method_summary.to_dict(orient="records"),
        "comparison_to_prior15_targets": comparison.to_dict(
            orient="records"
        ),
        "runtime_seconds": time.time() - started,
        "claim_boundary": str(
            config.get(
                "claim_boundary",
                "Retrospective full-cohort cross-validation. Every displayed "
                "growth target, AFM texture model and island model is excluded "
                "from its own outer fit, but the M12a method family was "
                "developed using earlier partitions. This is not a "
                "prospective untouched test.",
            )
        ),
    }
    write_json(manifest, report / "best_model_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_full_cohort_loo.json",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        load_config(args.config),
        smoke=bool(args.smoke),
        device_name=str(args.device),
    )


if __name__ == "__main__":
    main()
