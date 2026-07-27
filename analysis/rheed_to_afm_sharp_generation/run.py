from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    predict_groups,
)
from analysis.rheed_to_afm_generation.run import (
    _load_tables,
    _split_manifest,
)
from analysis.rheed_to_afm_generation.training import resolve_device

from .adversarial import calibrate_random_fields
from .cross_validation import run_training_group_cross_validation
from .evaluation import condition_permutation_control, evaluate_method_sets
from .rheed import select_sharp_rheed_predictor
from .spectral import (
    ConditionalSpectralModel,
    fit_conditional_spectral_model,
    save_spectral_model,
)
from .training import load_refiner, train_adversarial_refiner
from .visualization import make_sharp_generation_figures


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _condition_maps(
    *,
    predictor: Any,
    groups: list[str],
    registry: pd.DataFrame,
    physics: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw, standardized, _ = predict_groups(
        predictor, groups, registry, physics
    )
    return dict(zip(groups, raw)), dict(zip(groups, standardized))


def _true_conditions(
    rows: pd.DataFrame,
    scaler: ConditionScaler,
) -> dict[str, np.ndarray]:
    return {
        str(group): scaler.transform(
            group_rows[scaler.columns].median().to_numpy(float)[None],
            clip=False,
        )[0]
        for group, group_rows in rows.groupby("growth_run_id")
    }


def _generate_spectral(
    *,
    model: ConditionalSpectralModel,
    groups: list[str],
    conditions: dict[str, np.ndarray],
    sample_count: int,
    seed: int,
    iterations: int,
) -> dict[str, list[np.ndarray]]:
    return {
        group: [
            model.generate(
                conditions[group],
                seed=seed + group_index * 10_000 + draw,
                iterations=iterations,
            )
            for draw in range(sample_count)
        ]
        for group_index, group in enumerate(groups)
    }


def _calibrate_sets(
    *,
    spectral_sets: dict[str, list[np.ndarray]],
    conditions: dict[str, np.ndarray],
    condition_scaler: ConditionScaler,
    config: dict[str, Any],
    device_name: str,
    history_path: Path | None = None,
    smoke: bool = False,
) -> dict[str, list[np.ndarray]]:
    device = resolve_device(device_name)
    calibrated: dict[str, list[np.ndarray]] = {}
    histories: list[dict[str, Any]] = []
    steps = 2 if smoke else int(config["descriptor_calibration_steps"])
    for group, arrays in spectral_sets.items():
        result, history = calibrate_random_fields(
            np.stack(arrays),
            conditions[group],
            condition_scaler=condition_scaler,
            device=device,
            steps=steps,
            learning_rate=float(
                config["descriptor_calibration_learning_rate"]
            ),
            content_weight=float(
                config["descriptor_calibration_content_weight"]
            ),
        )
        calibrated[group] = [array for array in result]
        histories.extend(
            {
                "growth_run_id": group,
                "device": str(device),
                **record,
            }
            for record in history
        )
    if history_path is not None:
        write_csv(pd.DataFrame(histories), history_path)
    return calibrated


@torch.no_grad()
def _refine_sets(
    *,
    checkpoint_path: Path,
    spectral_sets: dict[str, list[np.ndarray]],
    conditions: dict[str, np.ndarray],
    device_name: str,
) -> dict[str, list[np.ndarray]]:
    model, _, device = load_refiner(checkpoint_path, device_name)
    result: dict[str, list[np.ndarray]] = {}
    for group, arrays in spectral_sets.items():
        tensor = torch.from_numpy(
            np.stack(arrays)[:, None].astype(np.float32)
        ).to(device)
        condition = torch.from_numpy(
            np.repeat(
                conditions[group][None], len(arrays), axis=0
            ).astype(np.float32)
        ).to(device)
        result[group] = [
            array
            for array in model(tensor, condition).cpu().numpy()[:, 0]
        ]
    return result


def _prior_cvae_sets(
    directory: str | Path, groups: list[str]
) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    root = repo_path(directory)
    maps: dict[str, list[np.ndarray]] = {}
    rq: dict[str, float] = {}
    for group in groups:
        payload = np.load(root / f"{group}_samples.npz", allow_pickle=False)
        maps[group] = [
            np.asarray(array, dtype=np.float32)
            for array in payload["generated_unit_shapes"]
        ]
        rq[group] = float(payload["predicted_rq_nm"])
    return maps, rq


def _save_generated_sets(
    generated: dict[str, dict[str, list[np.ndarray]]],
    output_root: Path,
) -> None:
    for method, groups in generated.items():
        method_root = output_root / "generated_maps" / method
        method_root.mkdir(parents=True, exist_ok=True)
        for group, arrays in groups.items():
            np.savez_compressed(
                method_root / f"{group}.npz",
                generated_unit_shapes=np.stack(arrays).astype(np.float32),
                growth_run_id=np.asarray(group),
                method=np.asarray(method),
            )


def run_experiment(
    config: dict[str, Any],
    *,
    include_gan: bool,
    smoke: bool,
    device_name: str,
) -> None:
    suffix = "smoke" if smoke else "development"
    output_root = repo_path(config["output_root"]) / suffix
    report_root = repo_path(config["report_root"]) / suffix
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    method_root = report_root / "methods"
    method_root.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    split_audit = _split_manifest(descriptors, report_root, config)
    write_csv(
        tables["removelist_excluded_rows"],
        report_root / "excluded_by_removelist.csv",
    )
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[descriptors["split"] == "val"].copy()
    train_groups = set(train_rows["growth_run_id"].astype(str))
    validation_groups = sorted(
        validation_rows["growth_run_id"].astype(str).unique()
    )
    condition_scaler = ConditionScaler.fit(
        descriptors, list(config["condition_columns"]), train_groups
    )
    predictor, temporal_ablation, predictor_selection = select_sharp_rheed_predictor(
        config=config,
        tables=tables,
        condition_scaler=condition_scaler,
        output_root=output_root / "rheed_condition_model",
        report_root=report_root / "rheed_condition_model",
    )
    predicted_raw, predicted_conditions = _condition_maps(
        predictor=predictor,
        groups=validation_groups,
        registry=tables["registry"],
        physics=tables["physics"],
    )
    oracle_conditions = _true_conditions(validation_rows, condition_scaler)

    spectral_model, spectral_cv, shape_targets = fit_conditional_spectral_model(
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        alphas=[float(value) for value in config["spectral_ridge_alphas"]],
        resolution=int(config["resolution"]),
        removelist_sample_ids=tables["removelist"].sample_ids,
    )
    spectral_path = output_root / "M2_learned_spectral_field" / "model.joblib"
    save_spectral_model(spectral_model, spectral_path)
    (method_root / "M2_learned_spectral_field").mkdir(
        parents=True, exist_ok=True
    )
    write_csv(
        spectral_cv,
        method_root
        / "M2_learned_spectral_field"
        / "ridge_leave_one_group_out.csv",
    )
    write_csv(
        shape_targets.reset_index(),
        method_root
        / "M2_learned_spectral_field"
        / "training_group_shape_parameters.csv",
    )

    sample_count = 2 if smoke else int(config["samples_per_condition"])
    iaaft_iterations = (
        4 if smoke else int(config["spectral_iaaft_iterations"])
    )
    spectral_oracle = _generate_spectral(
        model=spectral_model,
        groups=validation_groups,
        conditions=oracle_conditions,
        sample_count=sample_count,
        seed=int(config["seed"]) + 100_000,
        iterations=iaaft_iterations,
    )
    spectral_rheed = _generate_spectral(
        model=spectral_model,
        groups=validation_groups,
        conditions=predicted_conditions,
        sample_count=sample_count,
        seed=int(config["seed"]) + 200_000,
        iterations=iaaft_iterations,
    )
    calibrated_root = (
        method_root / "M2b_descriptor_calibrated_spectral_field"
    )
    calibrated_oracle = _calibrate_sets(
        spectral_sets=spectral_oracle,
        conditions=oracle_conditions,
        condition_scaler=condition_scaler,
        config=config,
        device_name=device_name,
        history_path=calibrated_root / "oracle_calibration_history.csv",
        smoke=smoke,
    )
    calibrated_rheed = _calibrate_sets(
        spectral_sets=spectral_rheed,
        conditions=predicted_conditions,
        condition_scaler=condition_scaler,
        config=config,
        device_name=device_name,
        history_path=calibrated_root / "rheed_calibration_history.csv",
        smoke=smoke,
    )
    mean_conditions = {
        group: np.zeros_like(predicted_conditions[group])
        for group in validation_groups
    }
    mean_spectral = _generate_spectral(
        model=spectral_model,
        groups=validation_groups,
        conditions=mean_conditions,
        sample_count=sample_count,
        seed=int(config["seed"]) + 200_000,
        iterations=iaaft_iterations,
    )
    mean_calibrated = _calibrate_sets(
        spectral_sets=mean_spectral,
        conditions=mean_conditions,
        condition_scaler=condition_scaler,
        config=config,
        device_name=device_name,
        history_path=calibrated_root / "mean_calibration_history.csv",
        smoke=smoke,
    )
    prior_cvae, prior_cvae_rq = _prior_cvae_sets(
        config["prior_cvae_validation_dir"], validation_groups
    )
    if smoke:
        prior_cvae = {
            group: arrays[:sample_count] for group, arrays in prior_cvae.items()
        }
    log_rq_index = condition_scaler.columns.index("log_rq_nm")
    predicted_rq = {
        group: float(np.exp(predicted_raw[group][log_rq_index]))
        for group in validation_groups
    }
    oracle_rq = {
        str(group): float(rows["rq_nm"].median())
        for group, rows in validation_rows.groupby("growth_run_id")
    }
    mean_rq_value = float(
        np.exp(
            condition_scaler.mean[
                condition_scaler.columns.index("log_rq_nm")
            ]
        )
    )
    mean_rq = {group: mean_rq_value for group in validation_groups}
    generated: dict[str, dict[str, list[np.ndarray]]] = {
        "M0_mean_condition_calibrated_spectral": mean_calibrated,
        "M1_cvae_blur_baseline": prior_cvae,
        "M2_spectral_oracle_condition": spectral_oracle,
        "M2_spectral_rheed_condition": spectral_rheed,
        "M2b_calibrated_spectral_oracle_condition": calibrated_oracle,
        "M2b_calibrated_spectral_rheed_condition": calibrated_rheed,
    }
    rq_by_method: dict[str, dict[str, float]] = {
        "M0_mean_condition_calibrated_spectral": mean_rq,
        "M1_cvae_blur_baseline": prior_cvae_rq,
        "M2_spectral_oracle_condition": oracle_rq,
        "M2_spectral_rheed_condition": predicted_rq,
        "M2b_calibrated_spectral_oracle_condition": oracle_rq,
        "M2b_calibrated_spectral_rheed_condition": predicted_rq,
    }
    adversarial_result = None
    if include_gan:
        adversarial_dir = output_root / "M3_adversarial_spectral_refiner"
        adversarial_result = train_adversarial_refiner(
            train_rows=train_rows,
            validation_rows=validation_rows,
            condition_scaler=condition_scaler,
            spectral_model=spectral_model,
            validation_predicted_conditions=predicted_conditions,
            config=config,
            output_dir=adversarial_dir,
            smoke=smoke,
            device_name=device_name,
        )
        generated["M3_refiner_oracle_condition"] = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=spectral_oracle,
            conditions=oracle_conditions,
            device_name=device_name,
        )
        generated["M3_refiner_rheed_condition"] = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=spectral_rheed,
            conditions=predicted_conditions,
            device_name=device_name,
        )
        generated["M3b_calibrated_refiner_oracle_condition"] = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=calibrated_oracle,
            conditions=oracle_conditions,
            device_name=device_name,
        )
        generated["M3b_calibrated_refiner_rheed_condition"] = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=calibrated_rheed,
            conditions=predicted_conditions,
            device_name=device_name,
        )
        generated["M0b_mean_condition_calibrated_refiner"] = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=mean_calibrated,
            conditions=mean_conditions,
            device_name=device_name,
        )
        rq_by_method["M3_refiner_oracle_condition"] = oracle_rq
        rq_by_method["M3_refiner_rheed_condition"] = predicted_rq
        rq_by_method["M3b_calibrated_refiner_oracle_condition"] = oracle_rq
        rq_by_method["M3b_calibrated_refiner_rheed_condition"] = predicted_rq
        rq_by_method["M0b_mean_condition_calibrated_refiner"] = mean_rq

    _save_generated_sets(generated, output_root)
    evaluation = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        generated=generated,
        generated_rq=rq_by_method,
        output_dir=report_root / "validation_evaluation",
        resolution=int(config["resolution"]),
    )

    wrong_conditions = {
        group: predicted_conditions[
            validation_groups[(index + 1) % len(validation_groups)]
        ]
        for index, group in enumerate(validation_groups)
    }
    wrong_spectral = _generate_spectral(
        model=spectral_model,
        groups=validation_groups,
        conditions=wrong_conditions,
        sample_count=sample_count,
        seed=int(config["seed"]) + 200_000,
        iterations=iaaft_iterations,
    )
    wrong_calibrated = _calibrate_sets(
        spectral_sets=wrong_spectral,
        conditions=wrong_conditions,
        condition_scaler=condition_scaler,
        config=config,
        device_name=device_name,
        history_path=calibrated_root / "permuted_calibration_history.csv",
        smoke=smoke,
    )
    controls: list[pd.DataFrame] = []
    spectral_control = condition_permutation_control(
        groups=validation_groups,
        split_rows=validation_rows,
        condition_scaler=condition_scaler,
        correct_maps=spectral_rheed,
        wrong_maps=wrong_spectral,
        generated_rq=predicted_rq,
    )
    spectral_control.insert(1, "method", "M2_spectral_rheed_condition")
    controls.append(spectral_control)
    calibrated_control = condition_permutation_control(
        groups=validation_groups,
        split_rows=validation_rows,
        condition_scaler=condition_scaler,
        correct_maps=calibrated_rheed,
        wrong_maps=wrong_calibrated,
        generated_rq=predicted_rq,
    )
    calibrated_control.insert(
        1, "method", "M2b_calibrated_spectral_rheed_condition"
    )
    controls.append(calibrated_control)
    if include_gan and adversarial_result is not None:
        wrong_refined = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=wrong_spectral,
            conditions=wrong_conditions,
            device_name=device_name,
        )
        refiner_control = condition_permutation_control(
            groups=validation_groups,
            split_rows=validation_rows,
            condition_scaler=condition_scaler,
            correct_maps=generated["M3_refiner_rheed_condition"],
            wrong_maps=wrong_refined,
            generated_rq=predicted_rq,
        )
        refiner_control.insert(1, "method", "M3_refiner_rheed_condition")
        controls.append(refiner_control)
        wrong_calibrated_refined = _refine_sets(
            checkpoint_path=adversarial_result.checkpoint_path,
            spectral_sets=wrong_calibrated,
            conditions=wrong_conditions,
            device_name=device_name,
        )
        calibrated_refiner_control = condition_permutation_control(
            groups=validation_groups,
            split_rows=validation_rows,
            condition_scaler=condition_scaler,
            correct_maps=generated[
                "M3b_calibrated_refiner_rheed_condition"
            ],
            wrong_maps=wrong_calibrated_refined,
            generated_rq=predicted_rq,
        )
        calibrated_refiner_control.insert(
            1, "method", "M3b_calibrated_refiner_rheed_condition"
        )
        controls.append(calibrated_refiner_control)
    control = pd.concat(controls, ignore_index=True)
    write_csv(control, report_root / "condition_permutation_control.csv")

    figure_dir = report_root / "figures"
    make_sharp_generation_figures(
        evaluation=evaluation,
        condition_control=control,
        training_history_path=(
            adversarial_result.history_path
            if adversarial_result is not None
            else None
        ),
        figure_dir=figure_dir,
        phase1_manifest_path=config["phase1_manifest"],
    )
    cross_validation_manifest = None
    if bool(config.get("run_train_group_cross_validation", False)) and not smoke:
        cross_validation_manifest = run_training_group_cross_validation(
            config=config,
            tables=tables,
            selected_predictor=predictor,
            output_dir=output_root / "training_group_cross_validation",
            report_dir=report_root / "training_group_cross_validation",
            device_name=device_name,
        )
    manifest = {
        "status": "validation_only_development",
        "smoke": smoke,
        "old_test_partition_reused": False,
        "split_audit": split_audit,
        "removelist": {
            "path": str(tables["removelist"].path.relative_to(repo_path("."))),
            "sha256": tables["removelist"].sha256,
            "sample_ids": list(tables["removelist"].sample_ids),
            "overlap_after_filtering": [],
        },
        "rheed_predictor_selection": predictor_selection,
        "spectral_model": {
            "path": str(spectral_path.relative_to(repo_path("."))),
            "sha256": sha256_file(spectral_path),
            "ridge_alpha": spectral_model.alpha,
            "retrieval_at_inference": False,
        },
        "descriptor_calibration": {
            "steps": 2
            if smoke
            else int(config["descriptor_calibration_steps"]),
            "learning_rate": float(
                config["descriptor_calibration_learning_rate"]
            ),
            "content_weight": float(
                config["descriptor_calibration_content_weight"]
            ),
            "measured_afm_target_used_at_inference": False,
            "retrieval_at_inference": False,
        },
        "adversarial_refiner": (
            {
                "path": str(
                    adversarial_result.checkpoint_path.relative_to(repo_path("."))
                ),
                "sha256": sha256_file(adversarial_result.checkpoint_path),
                "best_step": adversarial_result.best_step,
                "best_validation_score": adversarial_result.best_validation_score,
                "runtime_seconds": adversarial_result.runtime_seconds,
                "retrieval_at_inference": False,
            }
            if adversarial_result is not None
            else None
        ),
        "method_summary": evaluation["summary"].to_dict(orient="records"),
        "condition_control": control.to_dict(orient="records"),
        "training_group_cross_validation": cross_validation_manifest,
    }
    write_json(manifest, report_root / "development_manifest.json")
    joblib.dump(
        predictor,
        output_root / "rheed_descriptor_predictor.joblib",
    )
    write_csv(temporal_ablation, report_root / "temporal_window_ablation.csv")
    print(json.dumps(manifest, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sharp non-retrieval RHEED-to-AFM generation"
    )
    parser.add_argument(
        "mode",
        choices=("smoke", "spectral", "gan"),
        help="smoke includes a short GAN; spectral skips GAN; gan runs all methods",
    )
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_sharp_generation.json",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_experiment(
        config,
        include_gan=args.mode in {"smoke", "gan"},
        smoke=args.mode == "smoke",
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
