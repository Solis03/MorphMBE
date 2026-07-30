"""Leakage-aware endpoint ensemble for areal AFM roughness prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .endpoint_models import calibrate_positive


@dataclass
class FittedR3DExpert:
    scaler: StandardScaler
    pca: PCA
    ridge: Ridge

    def predict_raw(self, embedding: np.ndarray) -> float:
        latent = self.pca.transform(
            self.scaler.transform(np.asarray(embedding)[None])
        )
        return float(np.exp(self.ridge.predict(latent)[0]))


@dataclass
class FittedStreakExpert:
    embedding_scaler: StandardScaler
    pca: PCA
    streak_scaler: StandardScaler
    ridge: Ridge

    def predict_raw(
        self, embedding: np.ndarray, streak_feature: float
    ) -> float:
        latent = self.pca.transform(
            self.embedding_scaler.transform(np.asarray(embedding)[None])
        )
        streak = self.streak_scaler.transform(
            np.asarray([[streak_feature]], dtype=float)
        )
        return float(np.exp(self.ridge.predict(np.c_[latent, streak])[0]))


@dataclass(frozen=True)
class EndpointPrediction:
    value_nm: float
    temporal_5_nm: float
    temporal_8_nm: float
    streak_expert_nm: float
    streak_gate: bool
    rough_consensus_gate: bool
    streak_threshold: float
    upper_threshold_nm: float
    expert_log_range: float
    nearest_embedding_distance: float
    streak_robust_z: float


def _fit_r3d(
    embeddings: np.ndarray,
    log_target: np.ndarray,
    indices: np.ndarray,
    *,
    components: int,
    alpha: float,
) -> FittedR3DExpert:
    scaler = StandardScaler().fit(embeddings[indices])
    standardized = scaler.transform(embeddings[indices])
    pca = PCA(n_components=min(int(components), len(indices) - 1)).fit(
        standardized
    )
    ridge = Ridge(alpha=float(alpha)).fit(
        pca.transform(standardized), log_target[indices]
    )
    return FittedR3DExpert(scaler, pca, ridge)


def _fit_streak(
    embeddings: np.ndarray,
    streak: np.ndarray,
    log_target: np.ndarray,
    indices: np.ndarray,
    *,
    components: int = 3,
    alpha: float = 3.0,
) -> FittedStreakExpert:
    embedding_scaler = StandardScaler().fit(embeddings[indices])
    standardized = embedding_scaler.transform(embeddings[indices])
    pca = PCA(n_components=min(int(components), len(indices) - 1)).fit(
        standardized
    )
    streak_scaler = StandardScaler().fit(streak[indices, None])
    design = np.c_[
        pca.transform(standardized),
        streak_scaler.transform(streak[indices, None]),
    ]
    ridge = Ridge(alpha=float(alpha)).fit(design, log_target[indices])
    return FittedStreakExpert(
        embedding_scaler, pca, streak_scaler, ridge
    )


def _raw_reference(
    embeddings: np.ndarray,
    streak: np.ndarray,
    log_target: np.ndarray,
    train: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    raw = np.empty(len(train), dtype=float)
    for position, query in enumerate(train):
        fit = train[train != query]
        if kind == "temporal5":
            model = _fit_r3d(
                embeddings, log_target, fit, components=5, alpha=30.0
            )
            raw[position] = model.predict_raw(embeddings[query])
        elif kind == "temporal8":
            model = _fit_r3d(
                embeddings, log_target, fit, components=8, alpha=30.0
            )
            raw[position] = model.predict_raw(embeddings[query])
        elif kind == "streak":
            model = _fit_streak(
                embeddings, streak, log_target, fit
            )
            raw[position] = model.predict_raw(
                embeddings[query], streak[query]
            )
        else:
            raise ValueError(f"unknown expert kind: {kind}")
    return raw


def _calibrated_expert(
    embeddings: np.ndarray,
    streak: np.ndarray,
    target_nm: np.ndarray,
    train: np.ndarray,
    query_embedding: np.ndarray,
    query_streak: float,
    *,
    kind: str,
) -> float:
    log_target = np.log(np.clip(target_nm, 1e-8, None))
    reference = _raw_reference(
        embeddings, streak, log_target, train, kind=kind
    )
    if kind == "temporal5":
        model = _fit_r3d(
            embeddings, log_target, train, components=5, alpha=30.0
        )
        raw_query = model.predict_raw(query_embedding)
        calibration = "range30"
    elif kind == "temporal8":
        model = _fit_r3d(
            embeddings, log_target, train, components=8, alpha=30.0
        )
        raw_query = model.predict_raw(query_embedding)
        calibration = "range30"
    elif kind == "streak":
        model = _fit_streak(embeddings, streak, log_target, train)
        raw_query = model.predict_raw(query_embedding, query_streak)
        calibration = "range20"
    else:
        raise ValueError(f"unknown expert kind: {kind}")
    return calibrate_positive(
        reference,
        target_nm[train],
        raw_query,
        calibration,
    )


def predict_endpoint(
    *,
    embeddings: np.ndarray,
    streak: np.ndarray,
    target_nm: np.ndarray,
    train: np.ndarray,
    query_embedding: np.ndarray,
    query_streak: float,
    streak_gate_quantile: float = 0.80,
    rough_gate_quantile: float = 0.75,
) -> EndpointPrediction:
    """Predict one query without using its AFM target.

    Two temporal experts capture different latent resolutions.  A causal
    diffraction-peak elongation expert is activated only above the upper
    quintile of the training streak distribution.  Conversely, both temporal
    heads must exceed the training upper quartile before the rough endpoint is
    expanded.  This prevents either a single unstable head or a single
    hand-crafted feature from forcing an endpoint prediction.
    """

    train = np.asarray(train, dtype=int)
    temporal_5 = _calibrated_expert(
        embeddings,
        streak,
        target_nm,
        train,
        query_embedding,
        query_streak,
        kind="temporal5",
    )
    temporal_8 = _calibrated_expert(
        embeddings,
        streak,
        target_nm,
        train,
        query_embedding,
        query_streak,
        kind="temporal8",
    )
    streak_expert = _calibrated_expert(
        embeddings,
        streak,
        target_nm,
        train,
        query_embedding,
        query_streak,
        kind="streak",
    )
    streak_threshold = float(
        np.quantile(streak[train], float(streak_gate_quantile))
    )
    upper_threshold = float(
        np.quantile(target_nm[train], float(rough_gate_quantile))
    )
    streak_gate = bool(query_streak > streak_threshold)
    rough_gate = bool(
        temporal_5 > upper_threshold and temporal_8 > upper_threshold
    )
    if streak_gate:
        value = streak_expert
    elif rough_gate:
        value = max(temporal_5, temporal_8)
    else:
        value = temporal_8
    expert_values = np.asarray(
        [temporal_5, temporal_8, streak_expert], dtype=float
    )
    distance_scaler = StandardScaler().fit(embeddings[train])
    standardized = distance_scaler.transform(embeddings[train])
    query_standardized = distance_scaler.transform(
        np.asarray(query_embedding)[None]
    )[0]
    nearest = float(
        np.min(
            np.sqrt(
                np.mean(
                    np.square(standardized - query_standardized), axis=1
                )
            )
        )
    )
    streak_median = float(np.median(streak[train]))
    streak_scale = max(
        float(
            (
                np.quantile(streak[train], 0.75)
                - np.quantile(streak[train], 0.25)
            )
            / 1.349
        ),
        1e-6,
    )
    return EndpointPrediction(
        value_nm=float(value),
        temporal_5_nm=float(temporal_5),
        temporal_8_nm=float(temporal_8),
        streak_expert_nm=float(streak_expert),
        streak_gate=streak_gate,
        rough_consensus_gate=rough_gate,
        streak_threshold=streak_threshold,
        upper_threshold_nm=upper_threshold,
        expert_log_range=float(
            np.ptp(np.log(np.clip(expert_values, 1e-8, None)))
        ),
        nearest_embedding_distance=nearest,
        streak_robust_z=float(
            abs(query_streak - streak_median) / streak_scale
        ),
    )
