from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from analysis.rheed_to_afm_generation.data import (
    PHYSICS_COLUMNS,
    ConditionScaler,
    RheedDescriptorPredictor,
    fit_rheed_descriptor_predictor,
    load_rheed_feature_table,
    predict_groups,
)
from analysis.rheed_video_afm_story.common import sha256_file, write_csv, write_json


@dataclass
class RheedRoughnessSVR:
    embedding_id: str
    pca: PCA
    embedding_scaler: StandardScaler
    physics_scaler: StandardScaler
    estimator: SVR
    train_groups: list[str]
    excluded_sample_ids: list[str]

    def predict_z(self, embeddings: np.ndarray, physics: np.ndarray) -> np.ndarray:
        scaled = self.embedding_scaler.transform(embeddings)
        features = np.concatenate(
            [
                self.pca.transform(scaled),
                self.physics_scaler.transform(physics),
            ],
            axis=1,
        )
        return np.asarray(self.estimator.predict(features), dtype=np.float32)


@dataclass
class HybridRheedDescriptorPredictor:
    """Separate amplitude and unit-shape heads for small-data robustness."""

    morphology_predictor: RheedDescriptorPredictor
    roughness_predictor: RheedRoughnessSVR
    condition_scaler: ConditionScaler
    train_groups: list[str]
    excluded_sample_ids: list[str]
    model_family: str = "HybridSVRRq_PLSMorphology"

    @property
    def embedding_id(self) -> str:
        return self.morphology_predictor.embedding_id

    @property
    def pls_components(self) -> int:
        return int(self.morphology_predictor.pls_components)

    def predict_from_tables(
        self,
        groups: list[str],
        embedding_registry: pd.DataFrame,
        physics_table: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, standardized, morphology_features = predict_groups(
            self.morphology_predictor,
            groups,
            embedding_registry,
            physics_table,
        )
        sample_ids, embeddings, physics, _ = load_rheed_feature_table(
            self.roughness_predictor.embedding_id,
            embedding_registry,
            physics_table,
            excluded_sample_ids=set(self.excluded_sample_ids),
        )
        index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
        positions = [index[group] for group in groups]
        rq_z = self.roughness_predictor.predict_z(
            embeddings[positions], physics[positions]
        )
        combined = np.asarray(standardized, dtype=np.float32).copy()
        rq_position = self.condition_scaler.columns.index("log_rq_nm")
        combined[:, rq_position] = rq_z
        raw = self.condition_scaler.inverse_transform(combined)
        return raw, combined, morphology_features


def _validation_metrics(
    *,
    predictor: Any,
    validation_groups: list[str],
    group_targets: pd.DataFrame,
    registry: pd.DataFrame,
    physics: pd.DataFrame,
) -> dict[str, Any]:
    raw, standardized, _ = predict_groups(
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
    train_rq_scale = float(
        np.median(
            np.exp(
                group_targets.loc[predictor.train_groups, "log_rq_nm"].to_numpy(float)
            )
        )
    )
    pairwise = [
        float(np.mean(np.abs(standardized[i] - standardized[j])))
        for i in range(len(standardized))
        for j in range(i)
    ]
    wins = [
        float(np.mean(np.abs(standardized[i] - truth_z[i])))
        < float(np.mean(np.abs(standardized[(i + 1) % len(standardized)] - truth_z[i])))
        for i in range(len(standardized))
    ]
    sensitivity = float(np.median(pairwise)) if pairwise else 0.0
    correct_wins = int(sum(wins))
    passes = bool(
        sensitivity >= 0.20
        and correct_wins >= int(np.ceil(2 * len(validation_groups) / 3))
    )
    return {
        "val_condition_mae_z": condition_mae,
        "val_rq_mae_nm": rq_mae,
        "val_selection_score": 0.75 * condition_mae
        + 0.25 * rq_mae / max(train_rq_scale, 1e-8),
        "val_condition_sensitivity": sensitivity,
        "val_correct_condition_wins": correct_wins,
        "val_condition_group_count": len(validation_groups),
        "condition_gate_passed": passes,
    }


def _fit_roughness_svr(
    *,
    embedding_id: str,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    group_targets: pd.DataFrame,
    condition_scaler: ConditionScaler,
    train_groups: list[str],
    excluded_sample_ids: set[str],
    pca_dim: int = 1,
    c: float = 0.1,
) -> RheedRoughnessSVR:
    sample_ids, embeddings, physics, _ = load_rheed_feature_table(
        embedding_id,
        embedding_registry,
        physics_table,
        excluded_sample_ids=excluded_sample_ids,
    )
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    positions = [index[group] for group in train_groups]
    embedding_scaler = StandardScaler().fit(embeddings[positions])
    scaled = embedding_scaler.transform(embeddings[positions])
    components = min(int(pca_dim), len(train_groups) - 2, scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(scaled)
    physics_scaler = StandardScaler().fit(physics[positions])
    features = np.concatenate(
        [
            pca.transform(scaled),
            physics_scaler.transform(physics[positions]),
        ],
        axis=1,
    )
    targets = condition_scaler.transform(
        group_targets.loc[train_groups, condition_scaler.columns].to_numpy(float),
        clip=False,
    )
    rq_position = condition_scaler.columns.index("log_rq_nm")
    estimator = SVR(C=float(c), epsilon=0.15).fit(features, targets[:, rq_position])
    return RheedRoughnessSVR(
        embedding_id=embedding_id,
        pca=pca,
        embedding_scaler=embedding_scaler,
        physics_scaler=physics_scaler,
        estimator=estimator,
        train_groups=train_groups,
        excluded_sample_ids=sorted(excluded_sample_ids),
    )


def _fit_hybrid_candidate(
    *,
    morphology_embedding_id: str,
    roughness_embedding_id: str,
    morphology_pls_components: int,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    group_targets: pd.DataFrame,
    condition_scaler: ConditionScaler,
    train_groups: list[str],
    pca_dim: int,
    excluded_sample_ids: set[str],
) -> HybridRheedDescriptorPredictor:
    morphology, _ = _fit_pls_candidate(
        embedding_id=morphology_embedding_id,
        components=int(morphology_pls_components),
        embedding_registry=embedding_registry,
        physics_table=physics_table,
        group_targets=group_targets,
        condition_scaler=condition_scaler,
        train_groups=train_groups,
        pca_dim=pca_dim,
        excluded_sample_ids=excluded_sample_ids,
    )
    roughness = _fit_roughness_svr(
        embedding_id=roughness_embedding_id,
        embedding_registry=embedding_registry,
        physics_table=physics_table,
        group_targets=group_targets,
        condition_scaler=condition_scaler,
        train_groups=train_groups,
        excluded_sample_ids=excluded_sample_ids,
        pca_dim=1,
        c=0.1,
    )
    return HybridRheedDescriptorPredictor(
        morphology_predictor=morphology,
        roughness_predictor=roughness,
        condition_scaler=condition_scaler,
        train_groups=train_groups,
        excluded_sample_ids=sorted(excluded_sample_ids),
    )


def _fit_pls_candidate(
    *,
    embedding_id: str,
    components: int,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    group_targets: pd.DataFrame,
    condition_scaler: ConditionScaler,
    train_groups: list[str],
    pca_dim: int,
    excluded_sample_ids: set[str],
) -> tuple[RheedDescriptorPredictor, pd.DataFrame]:
    sample_ids, embeddings, physics, _ = load_rheed_feature_table(
        embedding_id,
        embedding_registry,
        physics_table,
        excluded_sample_ids=excluded_sample_ids,
    )
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    positions = [index[group] for group in train_groups]
    embedding_scaler = StandardScaler().fit(embeddings[positions])
    scaled = embedding_scaler.transform(embeddings[positions])
    pca_components = min(int(pca_dim), len(train_groups) - 2, scaled.shape[1])
    pca = PCA(n_components=pca_components, svd_solver="full").fit(scaled)
    physics_scaler = StandardScaler().fit(physics[positions])
    features = np.concatenate(
        [
            pca.transform(scaled),
            physics_scaler.transform(physics[positions]),
        ],
        axis=1,
    )
    targets = condition_scaler.transform(
        group_targets.loc[train_groups, condition_scaler.columns].to_numpy(float)
    )
    predictions = np.zeros_like(targets)
    for held in range(len(train_groups)):
        keep = np.arange(len(train_groups)) != held
        model = PLSRegression(n_components=int(components), scale=False, max_iter=1000)
        model.fit(features[keep], targets[keep])
        predictions[held] = model.predict(features[held : held + 1])[0]
    cv = pd.DataFrame(
        {
            "component": np.arange(targets.shape[1]),
            "loo_standardized_mae": np.mean(np.abs(predictions - targets), axis=0),
        }
    )
    estimator = PLSRegression(
        n_components=int(components), scale=False, max_iter=1000
    ).fit(features, targets)
    predictor = RheedDescriptorPredictor(
        embedding_id=embedding_id,
        pca=pca,
        embedding_scaler=embedding_scaler,
        physics_scaler=physics_scaler,
        ridge=estimator,
        condition_scaler=condition_scaler,
        alpha=float(components),
        train_groups=train_groups,
        excluded_sample_ids=sorted(excluded_sample_ids),
    )
    predictor.model_family = "PLSRegression"
    predictor.pls_components = int(components)
    return predictor, cv


def _crossfit_fixed_candidates(
    *,
    descriptors: pd.DataFrame,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    condition_columns: list[str],
    morphology_embedding_id: str,
    roughness_embedding_id: str,
    pca_dim: int,
    excluded_sample_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict LOO comparison used to choose PLS versus the hybrid head."""

    groups = sorted(descriptors["growth_run_id"].astype(str).unique())
    group_targets = descriptors.groupby("growth_run_id")[condition_columns].median()
    predictions: dict[str, list[np.ndarray]] = {
        "r3d_pls1": [],
        "hybrid_svr_rq_pls_shape": [],
    }
    raw_predictions: dict[str, list[np.ndarray]] = {
        "r3d_pls1": [],
        "hybrid_svr_rq_pls_shape": [],
    }
    truths: list[np.ndarray] = []
    raw_truths: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for held_group in groups:
        fit_groups = [group for group in groups if group != held_group]
        scaler = ConditionScaler.fit(descriptors, condition_columns, set(fit_groups))
        pls, _ = _fit_pls_candidate(
            embedding_id=morphology_embedding_id,
            components=1,
            embedding_registry=embedding_registry,
            physics_table=physics_table,
            group_targets=group_targets,
            condition_scaler=scaler,
            train_groups=fit_groups,
            pca_dim=pca_dim,
            excluded_sample_ids=excluded_sample_ids,
        )
        hybrid = _fit_hybrid_candidate(
            morphology_embedding_id=morphology_embedding_id,
            roughness_embedding_id=roughness_embedding_id,
            morphology_pls_components=1,
            embedding_registry=embedding_registry,
            physics_table=physics_table,
            group_targets=group_targets,
            condition_scaler=scaler,
            train_groups=fit_groups,
            pca_dim=pca_dim,
            excluded_sample_ids=excluded_sample_ids,
        )
        true_raw = group_targets.loc[held_group, condition_columns].to_numpy(float)
        truth = scaler.transform(true_raw[None], clip=False)[0]
        truths.append(truth)
        raw_truths.append(true_raw)
        for candidate_id, predictor in (
            ("r3d_pls1", pls),
            ("hybrid_svr_rq_pls_shape", hybrid),
        ):
            predicted_raw, predicted, _ = predict_groups(
                predictor,
                [held_group],
                embedding_registry,
                physics_table,
            )
            predictions[candidate_id].append(predicted[0])
            raw_predictions[candidate_id].append(predicted_raw[0])
            records.append(
                {
                    "growth_run_id": held_group,
                    "candidate_id": candidate_id,
                    "condition_mae_z": float(np.mean(np.abs(predicted[0] - truth))),
                    **{
                        f"true__{column}": float(truth[position])
                        for position, column in enumerate(condition_columns)
                    },
                    **{
                        f"predicted__{column}": float(predicted[0, position])
                        for position, column in enumerate(condition_columns)
                    },
                    **{
                        f"true_raw__{column}": float(true_raw[position])
                        for position, column in enumerate(condition_columns)
                    },
                    **{
                        f"predicted_raw__{column}": float(predicted_raw[0, position])
                        for position, column in enumerate(condition_columns)
                    },
                }
            )
    truth_array = np.stack(truths)
    raw_truth_array = np.stack(raw_truths)
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, values in predictions.items():
        array = np.stack(values)
        raw_array = np.stack(raw_predictions[candidate_id])
        correlations = [
            float(spearmanr(truth_array[:, position], array[:, position]).statistic)
            for position in range(len(condition_columns))
        ]
        per_group_mae = np.mean(np.abs(array - truth_array), axis=1)
        pairwise = [
            float(np.mean(np.abs(array[first] - array[second])))
            for first in range(len(array))
            for second in range(first)
        ]
        summary_rows.append(
            {
                "crossfit_candidate": candidate_id,
                "crossfit_group_count": len(groups),
                "crossfit_median_condition_mae_z": float(np.median(per_group_mae)),
                "crossfit_mean_condition_mae_z": float(np.mean(per_group_mae)),
                "crossfit_fold_standardized_rq_spearman": correlations[
                    condition_columns.index("log_rq_nm")
                ],
                "crossfit_raw_rq_spearman": float(
                    spearmanr(
                        raw_truth_array[:, condition_columns.index("log_rq_nm")],
                        raw_array[:, condition_columns.index("log_rq_nm")],
                    ).statistic
                ),
                "crossfit_median_rq_mae_nm": float(
                    np.median(
                        np.abs(
                            np.exp(
                                raw_truth_array[
                                    :,
                                    condition_columns.index("log_rq_nm"),
                                ]
                            )
                            - np.exp(
                                raw_array[
                                    :,
                                    condition_columns.index("log_rq_nm"),
                                ]
                            )
                        )
                    )
                ),
                "crossfit_shape_spearman_median": float(
                    np.median(
                        [
                            correlation
                            for position, correlation in enumerate(correlations)
                            if condition_columns[position] != "log_rq_nm"
                        ]
                    )
                ),
                "crossfit_all_spearman_median": float(np.median(correlations)),
                "crossfit_condition_sensitivity": float(np.median(pairwise)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(records)


def select_sharp_rheed_predictor(
    *,
    config: dict[str, Any],
    tables: dict[str, Any],
    condition_scaler: ConditionScaler,
    output_root: Path,
    report_root: Path,
) -> tuple[RheedDescriptorPredictor, pd.DataFrame, dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    descriptors = tables["descriptors"]
    group_targets = descriptors.groupby("growth_run_id")[
        condition_scaler.columns
    ].median()
    train_groups = sorted(
        descriptors.loc[descriptors["split"] == "train", "growth_run_id"].astype(str)
    )
    train_groups = sorted(set(train_groups))
    validation_groups = sorted(
        descriptors.loc[descriptors["split"] == "val", "growth_run_id"].astype(str)
    )
    validation_groups = sorted(set(validation_groups))
    excluded = set(tables["removelist"].sample_ids)
    rows: list[dict[str, Any]] = []
    predictors: dict[str, RheedDescriptorPredictor] = {}
    for embedding_id in config["embedding_candidates"]:
        ridge, ridge_cv, _ = fit_rheed_descriptor_predictor(
            embedding_id=str(embedding_id),
            embedding_registry=tables["registry"],
            physics_table=tables["physics"],
            group_targets=group_targets,
            condition_scaler=condition_scaler,
            train_groups=set(train_groups),
            pca_dim=int(config["pca_dim"]),
            alphas=[float(value) for value in config["descriptor_ridge_alphas"]],
            excluded_sample_ids=excluded,
        )
        ridge.model_family = "Ridge"
        ridge_id = f"{embedding_id}__ridge_alpha_{ridge.alpha:g}"
        predictors[ridge_id] = ridge
        metrics = _validation_metrics(
            predictor=ridge,
            validation_groups=validation_groups,
            group_targets=group_targets,
            registry=tables["registry"],
            physics=tables["physics"],
        )
        rows.append(
            {
                "candidate_id": ridge_id,
                "embedding_id": embedding_id,
                "model_family": "Ridge",
                "model_hyperparameter": ridge.alpha,
                "inner_loo_mae_z": float(
                    ridge_cv.loc[
                        ridge_cv["alpha"] == ridge.alpha, "standardized_mae"
                    ].iloc[0]
                ),
                **metrics,
            }
        )
        candidate_dir = report_root / ridge_id
        write_csv(ridge_cv, candidate_dir / "inner_leave_one_group_out.csv")
        for components in config.get("pls_components", [1, 2, 3]):
            predictor, cv = _fit_pls_candidate(
                embedding_id=str(embedding_id),
                components=int(components),
                embedding_registry=tables["registry"],
                physics_table=tables["physics"],
                group_targets=group_targets,
                condition_scaler=condition_scaler,
                train_groups=train_groups,
                pca_dim=int(config["pca_dim"]),
                excluded_sample_ids=excluded,
            )
            candidate_id = f"{embedding_id}__pls_{int(components)}"
            predictors[candidate_id] = predictor
            metrics = _validation_metrics(
                predictor=predictor,
                validation_groups=validation_groups,
                group_targets=group_targets,
                registry=tables["registry"],
                physics=tables["physics"],
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "embedding_id": embedding_id,
                    "model_family": "PLSRegression",
                    "model_hyperparameter": int(components),
                    "inner_loo_mae_z": float(cv["loo_standardized_mae"].mean()),
                    **metrics,
                }
            )
            write_csv(
                cv,
                report_root / candidate_id / "inner_leave_one_group_out.csv",
            )
    morphology_embedding = str(
        config.get(
            "hybrid_morphology_embedding_id",
            "r3d_18__selected_16__raw_luminance",
        )
    )
    roughness_embedding = str(
        config.get(
            "hybrid_roughness_embedding_id",
            "dino_vits14__keyframe_1__raw_luminance",
        )
    )
    hybrid = _fit_hybrid_candidate(
        morphology_embedding_id=morphology_embedding,
        roughness_embedding_id=roughness_embedding,
        morphology_pls_components=1,
        embedding_registry=tables["registry"],
        physics_table=tables["physics"],
        group_targets=group_targets,
        condition_scaler=condition_scaler,
        train_groups=train_groups,
        pca_dim=int(config["pca_dim"]),
        excluded_sample_ids=excluded,
    )
    hybrid_id = (
        f"hybrid__rq_{roughness_embedding}__svr_c_0.1"
        f"__shape_{morphology_embedding}__pls_1"
    )
    predictors[hybrid_id] = hybrid
    rows.append(
        {
            "candidate_id": hybrid_id,
            "embedding_id": (f"Rq:{roughness_embedding};shape:{morphology_embedding}"),
            "model_family": hybrid.model_family,
            "model_hyperparameter": "Rq SVR C=0.1; shape PLS=1",
            "inner_loo_mae_z": np.nan,
            **_validation_metrics(
                predictor=hybrid,
                validation_groups=validation_groups,
                group_targets=group_targets,
                registry=tables["registry"],
                physics=tables["physics"],
            ),
        }
    )
    table = pd.DataFrame(rows)
    crossfit_summary, crossfit_predictions = _crossfit_fixed_candidates(
        descriptors=descriptors.loc[descriptors["split"] == "train"].copy(),
        embedding_registry=tables["registry"],
        physics_table=tables["physics"],
        condition_columns=condition_scaler.columns,
        morphology_embedding_id=morphology_embedding,
        roughness_embedding_id=roughness_embedding,
        pca_dim=int(config["pca_dim"]),
        excluded_sample_ids=excluded,
    )
    crossfit_lookup = {
        "r3d_pls1": (f"{morphology_embedding}__pls_1"),
        "hybrid_svr_rq_pls_shape": hybrid_id,
    }
    crossfit_summary["candidate_id"] = crossfit_summary["crossfit_candidate"].map(
        crossfit_lookup
    )
    table = table.merge(
        crossfit_summary.drop(columns=["crossfit_candidate"]),
        on="candidate_id",
        how="left",
    )
    write_csv(
        crossfit_summary,
        report_root / "fixed_candidate_crossfit_summary.csv",
    )
    write_csv(
        crossfit_predictions,
        report_root / "fixed_candidate_crossfit_predictions.csv",
    )
    eligible = table.loc[
        table["condition_gate_passed"] & table["crossfit_all_spearman_median"].notna()
    ].copy()
    if eligible.empty:
        raise RuntimeError("no RHEED condition candidate passed sensitivity control")
    eligible = eligible.sort_values(
        [
            "crossfit_median_condition_mae_z",
            "crossfit_median_rq_mae_nm",
            "val_selection_score",
        ],
        ascending=[True, True, True],
    )
    selected_id = str(eligible.iloc[0]["candidate_id"])
    selected = predictors[selected_id]
    path = output_root / "rheed_descriptor_predictor.joblib"
    joblib.dump(selected, path)
    table["selected"] = table["candidate_id"] == selected_id
    table = table.sort_values(
        ["condition_gate_passed", "val_selection_score"],
        ascending=[False, True],
    )
    write_csv(table, report_root / "candidate_summary.csv")
    selection = {
        "selected_candidate_id": selected_id,
        "selected_embedding_id": selected.embedding_id,
        "selected_model_family": getattr(selected, "model_family", "unknown"),
        "development_only_selection": True,
        "selection_policy": (
            "Require noncollapsed validation sensitivity and correct-vs-cyclic "
            "wins for at least 2/3 validation groups; then minimize strict "
            "15-group cross-fitted condition MAE and Rq MAE. Raw cross-fold "
            "Rq rank correlation is reported but is not used to imply a "
            "relationship when the small-data predictor shrinks to the mean."
        ),
        "validation_metrics": eligible.iloc[0].to_dict(),
        "strict_training_group_crossfit": (
            crossfit_summary.loc[crossfit_summary["candidate_id"] == selected_id]
            .iloc[0]
            .to_dict()
        ),
        "predictor_path": str(path),
        "predictor_sha256": sha256_file(path),
        "test_targets_accessed_for_selection": False,
        "physics_columns": PHYSICS_COLUMNS,
        "excluded_sample_ids": sorted(excluded),
    }
    write_json(selection, report_root / "selection.json")
    return selected, table, selection
