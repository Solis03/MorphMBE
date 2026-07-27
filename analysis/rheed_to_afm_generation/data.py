from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from analysis.rheed_single_frame.removelist import assert_no_removed_samples
from analysis.rheed_video_afm_story.common import repo_path


PHYSICS_COLUMNS = [
    "spot_summary_raw",
    "streak_summary_raw",
    "connection_summary_raw",
    "diffuse_summary_raw",
    "temporal_brightness_drift",
]


def _finite_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        finite = values[np.isfinite(values)]
        fill = float(finite.median()) if len(finite) else 0.0
        result[column] = values.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return result


def derive_condition_table(descriptors: pd.DataFrame) -> pd.DataFrame:
    result = descriptors.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    result["growth_run_id"] = result["growth_run_id"].astype(str)
    result["log_rq_nm"] = np.log(np.clip(result["rq_nm"].astype(float), 1e-4, None))
    result["log_unit_autocorr_length_nm"] = np.log(
        np.clip(result["unit_autocorr_length_nm"].astype(float), 1e-4, None)
    )
    result["log_unit_anisotropy_ratio"] = np.log(
        np.clip(result["unit_anisotropy_ratio"].astype(float), 1.0, None)
    )
    return result


def build_fixed_split(
    descriptor_table: pd.DataFrame,
    fold_table: pd.DataFrame,
    *,
    validation_fold: int,
    test_fold: int,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    if validation_fold == test_fold:
        raise ValueError("validation and test folds must differ")
    folds = fold_table.copy()
    folds["growth_run_id"] = folds["growth_run_id"].astype(str)
    folds["fold"] = folds["fold"].astype(int)
    all_groups = set(descriptor_table["growth_run_id"].astype(str))
    validation_groups = set(
        folds.loc[
            (folds["fold"] == int(validation_fold)) & (folds["split"] == "test"),
            "growth_run_id",
        ]
    )
    test_groups = set(
        folds.loc[
            (folds["fold"] == int(test_fold)) & (folds["split"] == "test"),
            "growth_run_id",
        ]
    )
    train_groups = all_groups - validation_groups - test_groups
    if not train_groups or not validation_groups or not test_groups:
        raise ValueError("fixed split contains an empty partition")
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise ValueError("growth-group leakage detected in fixed split")

    split_lookup = {
        **{group: "train" for group in train_groups},
        **{group: "val" for group in validation_groups},
        **{group: "test" for group in test_groups},
    }
    result = descriptor_table.copy()
    result["split"] = result["growth_run_id"].astype(str).map(split_lookup)
    if result["split"].isna().any():
        raise ValueError("some AFM rows were not assigned to a split")
    groups = {
        "train": train_groups,
        "val": validation_groups,
        "test": test_groups,
    }
    return result, groups


@dataclass
class ConditionScaler:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(
        cls,
        descriptors: pd.DataFrame,
        columns: list[str],
        train_groups: set[str],
    ) -> "ConditionScaler":
        group_medians = (
            descriptors.loc[descriptors["growth_run_id"].isin(train_groups)]
            .groupby("growth_run_id")[columns]
            .median()
        )
        values = group_medians.to_numpy(float)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        lower = np.nanpercentile(values, 1, axis=0)
        upper = np.nanpercentile(values, 99, axis=0)
        return cls(columns=list(columns), mean=mean, scale=scale, lower=lower, upper=upper)

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if clip:
            array = np.clip(array, self.lower, self.upper)
        return np.clip((array - self.mean) / self.scale, -5.0, 5.0).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "lower_train_p01": self.lower.tolist(),
            "upper_train_p99": self.upper.tolist(),
            "fit_unit": "growth-group medians from training partition only",
        }


@dataclass
class RheedDescriptorPredictor:
    embedding_id: str
    pca: PCA
    embedding_scaler: StandardScaler
    physics_scaler: StandardScaler
    ridge: Ridge
    condition_scaler: ConditionScaler
    alpha: float
    train_groups: list[str]
    excluded_sample_ids: list[str]

    def transform_features(
        self, embeddings: np.ndarray, physics: np.ndarray
    ) -> np.ndarray:
        embedded = self.pca.transform(self.embedding_scaler.transform(embeddings))
        physical = self.physics_scaler.transform(physics)
        return np.concatenate([embedded, physical], axis=1).astype(np.float32)

    def predict(
        self, embeddings: np.ndarray, physics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self.transform_features(embeddings, physics)
        standardized = self.ridge.predict(features)
        raw = self.condition_scaler.inverse_transform(standardized)
        return raw.astype(np.float32), standardized.astype(np.float32)


def _load_embedding(embedding_path: str | Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(repo_path(embedding_path), allow_pickle=False)
    sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(sample_ids) != len(embeddings):
        raise ValueError(f"invalid embedding payload: {embedding_path}")
    return sample_ids, embeddings


def load_rheed_feature_table(
    embedding_id: str,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    excluded_sample_ids: set[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    row = embedding_registry.loc[embedding_registry["embedding_id"] == embedding_id]
    if len(row) != 1:
        raise ValueError(f"embedding id not uniquely available: {embedding_id}")
    path = str(row.iloc[0]["path"])
    sample_ids, embeddings = _load_embedding(path)
    excluded = set(map(str, excluded_sample_ids or set()))
    keep = np.asarray(
        [sample_id not in excluded for sample_id in sample_ids], dtype=bool
    )
    sample_ids = [
        sample_id for sample_id, keep_sample in zip(sample_ids, keep) if keep_sample
    ]
    embeddings = embeddings[keep]
    assert_no_removed_samples(
        sample_ids, excluded, context=f"RHEED embedding {embedding_id}"
    )
    physics = physics_table.copy()
    physics["sample_id"] = physics["sample_id"].astype(str)
    physics = _finite_frame(physics, PHYSICS_COLUMNS).set_index("sample_id")
    missing = sorted(set(sample_ids) - set(physics.index))
    if missing:
        raise ValueError(f"physics features missing for samples: {missing}")
    physics_values = physics.loc[sample_ids, PHYSICS_COLUMNS].to_numpy(np.float32)
    return sample_ids, embeddings, physics_values, path


def _inner_cv_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: list[float],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    splitter = LeaveOneOut()
    for alpha in alphas:
        predictions = np.zeros_like(targets, dtype=float)
        for train_index, held_index in splitter.split(features):
            model = Ridge(alpha=float(alpha))
            model.fit(features[train_index], targets[train_index])
            predictions[held_index] = model.predict(features[held_index])
        per_dimension = np.mean(np.abs(predictions - targets), axis=0)
        rows.append(
            {
                "alpha": float(alpha),
                "standardized_mae": float(np.mean(per_dimension)),
                "log_rq_standardized_mae": float(per_dimension[0]),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["standardized_mae", "log_rq_standardized_mae", "alpha"]
    )
    return float(table.iloc[0]["alpha"]), table


def fit_rheed_descriptor_predictor(
    *,
    embedding_id: str,
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
    group_targets: pd.DataFrame,
    condition_scaler: ConditionScaler,
    train_groups: set[str],
    pca_dim: int,
    alphas: list[float],
    excluded_sample_ids: set[str] | None = None,
) -> tuple[RheedDescriptorPredictor, pd.DataFrame, dict[str, Any]]:
    excluded = set(map(str, excluded_sample_ids or set()))
    assert_no_removed_samples(
        train_groups, excluded, context="RHEED descriptor predictor training groups"
    )
    sample_ids, embeddings, physics, path = load_rheed_feature_table(
        embedding_id,
        embedding_registry,
        physics_table,
        excluded_sample_ids=excluded,
    )
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    ordered_groups = sorted(train_groups)
    missing = sorted(set(ordered_groups) - set(index))
    if missing:
        raise ValueError(f"RHEED embeddings missing for train groups: {missing}")
    positions = [index[group] for group in ordered_groups]
    train_embeddings = embeddings[positions]
    train_physics = physics[positions]

    embedding_scaler = StandardScaler().fit(train_embeddings)
    embedding_scaled = embedding_scaler.transform(train_embeddings)
    components = min(int(pca_dim), len(ordered_groups) - 2, embedding_scaled.shape[1])
    if components < 1:
        raise ValueError("insufficient groups for embedding PCA")
    pca = PCA(n_components=components, svd_solver="full").fit(embedding_scaled)
    physics_scaler = StandardScaler().fit(train_physics)
    features = np.concatenate(
        [pca.transform(embedding_scaled), physics_scaler.transform(train_physics)],
        axis=1,
    )
    target_values = group_targets.loc[
        ordered_groups, condition_scaler.columns
    ].to_numpy(float)
    targets = condition_scaler.transform(target_values)
    alpha, cv_table = _inner_cv_alpha(features, targets, alphas)
    ridge = Ridge(alpha=alpha).fit(features, targets)
    predictor = RheedDescriptorPredictor(
        embedding_id=embedding_id,
        pca=pca,
        embedding_scaler=embedding_scaler,
        physics_scaler=physics_scaler,
        ridge=ridge,
        condition_scaler=condition_scaler,
        alpha=alpha,
        train_groups=ordered_groups,
        excluded_sample_ids=sorted(excluded),
    )
    metadata = {
        "embedding_id": embedding_id,
        "embedding_path": path,
        "embedding_pca_components": components,
        "embedding_pca_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "physics_columns": PHYSICS_COLUMNS,
        "ridge_alpha": alpha,
        "train_groups": ordered_groups,
        "excluded_sample_ids": sorted(excluded),
        "feature_dim": int(features.shape[1]),
    }
    return predictor, cv_table, metadata


def predict_groups(
    predictor: RheedDescriptorPredictor,
    groups: list[str],
    embedding_registry: pd.DataFrame,
    physics_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    custom_predict = getattr(predictor, "predict_from_tables", None)
    if custom_predict is not None:
        return custom_predict(groups, embedding_registry, physics_table)
    sample_ids, embeddings, physics, _ = load_rheed_feature_table(
        predictor.embedding_id,
        embedding_registry,
        physics_table,
        excluded_sample_ids=set(
            map(str, getattr(predictor, "excluded_sample_ids", []))
        ),
    )
    assert_no_removed_samples(
        groups,
        getattr(predictor, "excluded_sample_ids", []),
        context="RHEED descriptor prediction groups",
    )
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    missing = sorted(set(groups) - set(index))
    if missing:
        raise ValueError(f"RHEED embeddings missing for groups: {missing}")
    positions = [index[group] for group in groups]
    raw, standardized = predictor.predict(embeddings[positions], physics[positions])
    transformed_features = predictor.transform_features(
        embeddings[positions], physics[positions]
    )
    return raw, standardized, transformed_features


def save_predictor(
    predictor: RheedDescriptorPredictor,
    path: str | Path,
) -> None:
    file_path = repo_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, file_path)


def load_predictor(path: str | Path) -> RheedDescriptorPredictor:
    return joblib.load(repo_path(path))


class AFMConditionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        condition_scaler: ConditionScaler,
        resolution: int,
        *,
        augment: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.condition_scaler = condition_scaler
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.conditions = condition_scaler.transform(
            self.rows[condition_scaler.columns].to_numpy(float)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.rows.iloc[int(index)]
        paths = json.loads(str(row["unit_shape_paths"]))
        key = str(self.resolution)
        if key not in paths:
            raise KeyError(f"resolution {key} missing for {row['afm_file_id']}")
        array = np.load(repo_path(paths[key]), allow_pickle=False).astype(np.float32)
        if array.shape != (self.resolution, self.resolution):
            raise ValueError(f"unexpected AFM shape {array.shape} for {paths[key]}")
        image = torch.from_numpy(array[None])
        if self.augment:
            # Translation is compatible with approximately stationary AFM
            # morphology and preserves height histograms and Fourier power.
            shift_y = int(torch.randint(-8, 9, (1,)).item())
            shift_x = int(torch.randint(-8, 9, (1,)).item())
            image = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
        condition = torch.from_numpy(self.conditions[int(index)])
        return (
            image,
            condition,
            str(row["growth_run_id"]),
            str(row["afm_file_id"]),
        )
