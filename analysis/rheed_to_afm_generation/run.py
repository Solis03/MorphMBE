from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.rheed_single_frame.removelist import (
    assert_no_removed_samples,
    excluded_rows_for_present_samples,
    load_removelist_audit,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    sha256_object,
    write_csv,
    write_json,
)

from .data import (
    ConditionScaler,
    build_fixed_split,
    derive_condition_table,
    fit_rheed_descriptor_predictor,
    load_predictor,
    predict_groups,
    save_predictor,
)
from .evaluation import evaluate_split
from .training import load_model_checkpoint, train_conditional_vae
from .visualization import make_all_figures


def load_config(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    removelist = load_removelist_audit(
        repo_path("."), config.get("removelist_path", "removelist.txt")
    )
    expected_hash = config.get("removelist_sha256")
    if expected_hash is not None and removelist.sha256 != str(expected_hash):
        raise RuntimeError(
            "canonical removelist hash mismatch: "
            f"expected {expected_hash}, found {removelist.sha256}"
        )
    excluded = set(removelist.sample_ids)
    descriptors = derive_condition_table(
        pd.read_csv(
            repo_path(config["afm_descriptors"]),
            dtype={"sample_id": str, "growth_run_id": str},
        )
    )
    descriptors = descriptors.loc[
        ~descriptors["sample_id"].isin(excluded)
        & ~descriptors["growth_run_id"].isin(excluded)
    ].copy()
    folds = pd.read_csv(
        repo_path(config["group_folds"]), dtype={"growth_run_id": str}
    )
    folds = folds.loc[~folds["growth_run_id"].isin(excluded)].copy()
    split_descriptors, groups = build_fixed_split(
        descriptors,
        folds,
        validation_fold=int(config["validation_fold"]),
        test_fold=int(config["test_fold"]),
    )
    registry = pd.read_csv(repo_path(config["embedding_registry"]))
    physics = pd.read_csv(
        repo_path(config["rheed_physics_features"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    physics = physics.loc[
        ~physics["sample_id"].isin(excluded)
        & ~physics["growth_run_id"].isin(excluded)
    ].copy()
    phase1 = pd.read_csv(
        repo_path(config["phase1_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    phase1_all = phase1.copy()
    phase1 = phase1.loc[
        ~phase1["sample_id"].isin(excluded)
        & ~phase1["growth_run_id"].isin(excluded)
    ].copy()
    for name, frame in {
        "AFM descriptors": split_descriptors,
        "fold table": folds,
        "RHEED physics": physics,
        "phase-1 manifest": phase1,
    }.items():
        identifiers: list[str] = []
        for column in ("sample_id", "growth_run_id"):
            if column in frame:
                identifiers.extend(frame[column].dropna().astype(str))
        assert_no_removed_samples(identifiers, excluded, context=name)
    return {
        "descriptors": split_descriptors,
        "folds": folds,
        "registry": registry,
        "physics": physics,
        "phase1": phase1,
        "groups": groups,
        "removelist": removelist,
        "removelist_excluded_rows": pd.DataFrame(
            excluded_rows_for_present_samples(
                removelist.records,
                phase1_all["sample_id"].dropna().astype(str),
            )
        ),
    }


def _split_manifest(
    descriptors: pd.DataFrame,
    report_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    columns = [
        "sample_id",
        "growth_run_id",
        "afm_file_id",
        "plane_corrected_array_path",
        "rq_nm",
        "split",
    ]
    manifest = descriptors[columns].sort_values(
        ["split", "growth_run_id", "afm_file_id"]
    )
    removelist = load_removelist_audit(
        repo_path("."), config.get("removelist_path", "removelist.txt")
    )
    write_csv(manifest, report_root / "split_manifest.csv")
    group_table = (
        descriptors.groupby(["split", "growth_run_id"])
        .size()
        .rename("afm_scan_count")
        .reset_index()
    )
    write_csv(group_table, report_root / "split_group_summary.csv")
    group_sets = {
        split: set(group["growth_run_id"].astype(str))
        for split, group in group_table.groupby("split")
    }
    leakage = {
        "train_val_overlap": sorted(group_sets["train"] & group_sets["val"]),
        "train_test_overlap": sorted(group_sets["train"] & group_sets["test"]),
        "val_test_overlap": sorted(group_sets["val"] & group_sets["test"]),
    }
    result = {
        "validation_fold": int(config["validation_fold"]),
        "test_fold": int(config["test_fold"]),
        "split_counts_scans": descriptors["split"].value_counts().to_dict(),
        "split_counts_groups": group_table["split"].value_counts().to_dict(),
        "groups": {
            split: sorted(values) for split, values in group_sets.items()
        },
        "leakage": leakage,
        "leakage_check_passed": not any(leakage.values()),
        "removelist_path": str(removelist.path.relative_to(repo_path("."))),
        "removelist_sha256": removelist.sha256,
        "removelist_sample_ids": list(removelist.sample_ids),
        "removelist_overlap_after_filtering": sorted(
            set(manifest["sample_id"].astype(str))
            & set(removelist.sample_ids)
        ),
        "selection_policy": (
            "RHEED embedding family, ridge alpha, and CVAE epoch are selected "
            "with train/validation data only. Test fold 1 is evaluated only "
            "after best_model_manifest.json is frozen."
        ),
    }
    if not result["leakage_check_passed"]:
        raise RuntimeError(f"growth-group leakage: {leakage}")
    if result["removelist_overlap_after_filtering"]:
        raise RuntimeError(
            "removelist samples survived filtering: "
            f"{result['removelist_overlap_after_filtering']}"
        )
    write_json(result, report_root / "split_integrity_audit.json")
    return result


def _evaluate_predictor_candidate(
    *,
    predictor: Any,
    validation_groups: list[str],
    group_targets: pd.DataFrame,
    registry: pd.DataFrame,
    physics: pd.DataFrame,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    raw, standardized, features = predict_groups(
        predictor, validation_groups, registry, physics
    )
    truth_raw = group_targets.loc[
        validation_groups, predictor.condition_scaler.columns
    ].to_numpy(float)
    truth_z = predictor.condition_scaler.transform(truth_raw, clip=False)
    condition_mae = float(np.mean(np.abs(standardized - truth_z)))
    log_rq_position = predictor.condition_scaler.columns.index("log_rq_nm")
    rq_true = np.exp(truth_raw[:, log_rq_position])
    rq_predicted = np.exp(raw[:, log_rq_position])
    rq_mae = float(np.mean(np.abs(rq_predicted - rq_true)))
    rq_scale = float(
        np.median(
            np.exp(
                group_targets.loc[
                    predictor.train_groups, "log_rq_nm"
                ].to_numpy(float)
            )
        )
    )
    selection_score = 0.75 * condition_mae + 0.25 * rq_mae / max(rq_scale, 1e-6)
    metrics = {
        "val_condition_mae_z": condition_mae,
        "val_rq_mae_nm": rq_mae,
        "val_selection_score": selection_score,
    }
    return metrics, raw, standardized, features


def select_rheed_predictor(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    condition_scaler: ConditionScaler,
    output_root: Path,
    report_root: Path,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    descriptors = tables["descriptors"]
    removelist_ids = set(tables["removelist"].sample_ids)
    group_targets = descriptors.groupby("growth_run_id")[
        condition_scaler.columns
    ].median()
    train_groups = set(
        descriptors.loc[descriptors["split"] == "train", "growth_run_id"].astype(
            str
        )
    )
    validation_groups = sorted(
        descriptors.loc[descriptors["split"] == "val", "growth_run_id"]
        .astype(str)
        .unique()
    )
    ablation_rows: list[dict[str, Any]] = []
    predictors: dict[str, Any] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for embedding_id in config["embedding_candidates"]:
        predictor, cv_table, metadata = fit_rheed_descriptor_predictor(
            embedding_id=embedding_id,
            embedding_registry=tables["registry"],
            physics_table=tables["physics"],
            group_targets=group_targets,
            condition_scaler=condition_scaler,
            train_groups=train_groups,
            pca_dim=int(config["pca_dim"]),
            alphas=[float(value) for value in config["descriptor_ridge_alphas"]],
            excluded_sample_ids=removelist_ids,
        )
        metrics, _, _, _ = _evaluate_predictor_candidate(
            predictor=predictor,
            validation_groups=validation_groups,
            group_targets=group_targets,
            registry=tables["registry"],
            physics=tables["physics"],
        )
        row = dict(metadata)
        row.update(metrics)
        ablation_rows.append(row)
        predictors[embedding_id] = predictor
        metadata_by_id[embedding_id] = metadata
        candidate_dir = report_root / "rheed_predictor_ablations" / embedding_id
        write_csv(cv_table, candidate_dir / "inner_leave_one_group_out.csv")
        write_json(row, candidate_dir / "validation_metrics.json")

    ablation = pd.DataFrame(ablation_rows).sort_values(
        ["val_selection_score", "val_condition_mae_z", "val_rq_mae_nm"]
    )
    write_csv(ablation, report_root / "temporal_window_ablation.csv")
    selected_id = str(ablation.iloc[0]["embedding_id"])
    selected = predictors[selected_id]
    predictor_path = output_root / "rheed_descriptor_predictor.joblib"
    save_predictor(selected, predictor_path)
    selection = {
        "selected_embedding_id": selected_id,
        "selection_basis": "lowest validation-only weighted descriptor/Rq score",
        "validation_score": float(ablation.iloc[0]["val_selection_score"]),
        "condition_scaler": condition_scaler.to_dict(),
        "predictor_path": str(predictor_path.relative_to(repo_path("."))),
        "predictor_sha256": sha256_file(predictor_path),
        "candidate_count": len(ablation),
        "test_targets_accessed_for_selection": False,
        "metadata": metadata_by_id[selected_id],
    }
    write_json(selection, report_root / "rheed_predictor_selection.json")
    return selected, ablation, selection


def _prediction_maps(
    predictor: Any,
    groups: list[str],
    registry: pd.DataFrame,
    physics: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    raw, standardized, features = predict_groups(
        predictor, groups, registry, physics
    )
    return (
        dict(zip(groups, raw)),
        dict(zip(groups, standardized)),
        dict(zip(groups, features)),
    )


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "device_policy": "MPS when available, otherwise CUDA, otherwise CPU",
    }


def run_development(
    config: dict[str, Any],
    *,
    smoke: bool,
    device: str,
) -> None:
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    base_output = repo_path(config["output_root"])
    base_report = repo_path(config["report_root"])
    output_root = base_output / "smoke" if smoke else base_output
    report_root = base_report / "smoke" if smoke else base_report
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    split_audit = _split_manifest(descriptors, report_root, config)
    write_csv(
        tables["removelist_excluded_rows"],
        report_root / "excluded_by_removelist.csv",
    )
    train_groups = set(split_audit["groups"]["train"])
    condition_scaler = ConditionScaler.fit(
        descriptors, list(config["condition_columns"]), train_groups
    )
    write_json(condition_scaler.to_dict(), report_root / "condition_scaler.json")
    predictor, ablation, predictor_selection = select_rheed_predictor(
        config=config,
        tables=tables,
        condition_scaler=condition_scaler,
        output_root=output_root,
        report_root=report_root,
    )
    validation_groups = sorted(split_audit["groups"]["val"])
    train_group_list = sorted(split_audit["groups"]["train"])
    val_raw, val_z, val_features = _prediction_maps(
        predictor, validation_groups, tables["registry"], tables["physics"]
    )
    _, _, train_features = _prediction_maps(
        predictor, train_group_list, tables["registry"], tables["physics"]
    )
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[descriptors["split"] == "val"].copy()
    epochs = int(config["smoke_epochs"] if smoke else config["epochs"])
    result = train_conditional_vae(
        train_rows=train_rows,
        validation_rows=validation_rows,
        condition_scaler=condition_scaler,
        validation_predicted_conditions=val_z,
        validation_predicted_raw_conditions=val_raw,
        output_dir=output_root / "conditional_vae",
        config=config,
        epochs=epochs,
        device_name=device,
    )
    model, checkpoint_payload, model_device = load_model_checkpoint(
        result.checkpoint_path, device
    )
    evaluation = evaluate_split(
        split_name="validation",
        model=model,
        device=model_device,
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        predicted_raw=val_raw,
        predicted_standardized=val_z,
        transformed_features=val_features,
        train_transformed_features=train_features,
        output_dir=output_root / "validation_evaluation",
        resolution=int(config["resolution"]),
        samples_per_condition=(
            2 if smoke else int(config["samples_per_condition"])
        ),
        seed=int(config["seed"]),
    )
    make_all_figures(
        evaluation=evaluation,
        ablation=ablation,
        training_history_path=result.history_path,
        phase1_manifest=tables["phase1"],
        figure_dir=figure_dir,
    )
    run_manifest = {
        "mode": "smoke" if smoke else "development",
        "config": config,
        "config_sha256": sha256_object(config),
        "environment": _environment(),
        "split_audit": split_audit,
        "predictor_selection": predictor_selection,
        "training": {
            "checkpoint_path": str(
                result.checkpoint_path.relative_to(repo_path("."))
            ),
            "checkpoint_sha256": sha256_file(result.checkpoint_path),
            "best_epoch": result.best_epoch,
            "best_validation_selection_score": result.best_selection_score,
            "runtime_seconds": result.runtime_seconds,
        },
        "validation_summary": evaluation["summary"],
        "test_partition_evaluated": False,
    }
    write_json(run_manifest, report_root / "development_run_manifest.json")
    if not smoke:
        best_manifest = {
            "status": "frozen_before_test",
            "method": "two-stage RHEED descriptor predictor + conditional AFM VAE",
            "is_true_generator": True,
            "uses_retrieval_at_inference": False,
            "checkpoint_path": run_manifest["training"]["checkpoint_path"],
            "checkpoint_sha256": run_manifest["training"]["checkpoint_sha256"],
            "predictor_path": predictor_selection["predictor_path"],
            "predictor_sha256": predictor_selection["predictor_sha256"],
            "selected_embedding_id": predictor_selection[
                "selected_embedding_id"
            ],
            "selection_basis": (
                "RHEED embedding family and ridge regularization selected on "
                "validation descriptors; CVAE checkpoint selected on validation "
                "prior-generation morphology composite, Rq error, and a "
                "diversity-ratio penalty."
            ),
            "train_groups": split_audit["groups"]["train"],
            "validation_groups": split_audit["groups"]["val"],
            "test_groups": split_audit["groups"]["test"],
            "test_targets_accessed_for_selection": False,
            "config_path": "configs/rheed_to_afm_generation.json",
            "config_sha256": sha256_object(config),
            "split_manifest_path": str(
                (report_root / "split_manifest.csv").relative_to(repo_path("."))
            ),
            "split_manifest_sha256": sha256_file(
                report_root / "split_manifest.csv"
            ),
        }
        write_json(best_manifest, report_root / "best_model_manifest.json")
    print(json.dumps(run_manifest, indent=2, default=str))


def run_test(config: dict[str, Any], *, device: str) -> None:
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    best_path = report_root / "best_model_manifest.json"
    if not best_path.exists():
        raise FileNotFoundError(
            "development selection must be frozen before test evaluation"
        )
    test_manifest_path = report_root / "test_evaluation_manifest.json"
    if test_manifest_path.exists():
        raise FileExistsError(
            "test evaluation already exists; refusing to overwrite the held-out result"
        )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if sha256_file(best["checkpoint_path"]) != best["checkpoint_sha256"]:
        raise RuntimeError("frozen CVAE checkpoint hash mismatch")
    if sha256_file(best["predictor_path"]) != best["predictor_sha256"]:
        raise RuntimeError("frozen RHEED predictor hash mismatch")
    if sha256_object(config) != best["config_sha256"]:
        raise RuntimeError("config changed after model freeze")
    predictor = load_predictor(best["predictor_path"])
    condition_scaler = predictor.condition_scaler
    test_groups = sorted(best["test_groups"])
    train_groups = sorted(best["train_groups"])
    test_raw, test_z, test_features = _prediction_maps(
        predictor, test_groups, tables["registry"], tables["physics"]
    )
    _, _, train_features = _prediction_maps(
        predictor, train_groups, tables["registry"], tables["physics"]
    )
    model, _, model_device = load_model_checkpoint(best["checkpoint_path"], device)
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    test_rows = descriptors.loc[descriptors["split"] == "test"].copy()
    evaluation = evaluate_split(
        split_name="test",
        model=model,
        device=model_device,
        split_rows=test_rows,
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        predicted_raw=test_raw,
        predicted_standardized=test_z,
        transformed_features=test_features,
        train_transformed_features=train_features,
        output_dir=output_root / "test_evaluation",
        resolution=int(config["resolution"]),
        samples_per_condition=int(config["samples_per_condition"]),
        seed=int(config["seed"]) + 100_000,
    )
    ablation = pd.read_csv(report_root / "temporal_window_ablation.csv")
    make_all_figures(
        evaluation=evaluation,
        ablation=ablation,
        training_history_path=output_root
        / "conditional_vae"
        / "training_history.csv",
        phase1_manifest=tables["phase1"],
        figure_dir=report_root / "test_figures",
    )
    manifest = {
        "status": "held_out_test_evaluated_once",
        "frozen_model_manifest": str(best_path.relative_to(repo_path("."))),
        "frozen_model_manifest_sha256": sha256_file(best_path),
        "test_groups": test_groups,
        "test_scan_count": len(test_rows),
        "test_summary": evaluation["summary"],
        "test_output_dir": str(
            (output_root / "test_evaluation").relative_to(repo_path("."))
        ),
        "test_figure_dir": str(
            (report_root / "test_figures").relative_to(repo_path("."))
        ),
        "selection_changed_after_test": False,
    }
    write_json(manifest, test_manifest_path)
    print(json.dumps(manifest, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe RHEED-conditioned AFM generative experiment"
    )
    parser.add_argument(
        "mode", choices=["smoke", "develop", "test"], help="experiment phase"
    )
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_generation.json",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "smoke":
        run_development(config, smoke=True, device=args.device)
    elif args.mode == "develop":
        run_development(config, smoke=False, device=args.device)
    else:
        run_test(config, device=args.device)


if __name__ == "__main__":
    main()
