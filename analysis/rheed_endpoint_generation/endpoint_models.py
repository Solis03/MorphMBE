from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler


PHYSICS_ENDPOINT_FEATURES = [
    # Key-frame spot/streak geometry.
    "keyframe_1__round_component_fraction_p97_median",
    "keyframe_1__component_eccentricity_mean_p90_median",
    "keyframe_1__component_area_median_p95_median",
    "keyframe_1__component_count_p97_median",
    "keyframe_1__largest_component_area_fraction_p97_median",
    "keyframe_1__vertical_percolation_fraction_median",
    "keyframe_1__horizontal_percolation_fraction_median",
    # Causal evolution around the accepted rotational vertex.
    "causal_8__component_area_median_p97_slope",
    "causal_8__component_circularity_mean_p95_median",
    "causal_8__line_response_q50_slope",
    # Longer-window spot density, line response and diffuse background.
    "selected_16__round_component_fraction_p97_median",
    "selected_16__component_count_p97_median",
    "selected_16__component_count_p97_std",
    "selected_16__local_maximum_density_slope",
    "selected_16__blob_response_energy_slope",
    "selected_16__spot_peak_width_proxy_slope",
    "selected_16__horizontal_percolation_fraction_slope",
    "selected_16__background_spatial_smoothness_slope",
    "selected_16__fft_horizontal_vertical_anisotropy_first_last_diff",
    "selected_16__diffuse_to_peak_intensity_ratio_iqr",
    "selected_16__structure_tensor_coherence_first_last_diff",
    "temporal_brightness_drift",
]


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, float | int]

    @property
    def identifier(self) -> str:
        suffix = "_".join(
            f"{key}{value}" for key, value in sorted(self.parameters.items())
        )
        return f"{self.family}__{suffix}" if suffix else self.family


class _R3DRidge:
    def __init__(self, components: int, alpha: float) -> None:
        self.components = int(components)
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.model = Ridge(alpha=self.alpha)

    def fit(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        log_target: np.ndarray,
    ) -> "_R3DRidge":
        scaled = self.scaler.fit_transform(r3d)
        count = min(self.components, len(r3d) - 2, r3d.shape[1])
        self.pca = PCA(n_components=count, svd_solver="full").fit(scaled)
        self.model.fit(self.pca.transform(scaled), log_target)
        return self

    def predict(self, r3d: np.ndarray, physics: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        return self.model.predict(
            self.pca.transform(self.scaler.transform(r3d))
        )


class _R3DPLS:
    def __init__(self, components: int) -> None:
        self.components = int(components)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model: PLSRegression | None = None

    def fit(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        log_target: np.ndarray,
    ) -> "_R3DPLS":
        values = self.scaler.fit_transform(self.imputer.fit_transform(r3d))
        count = min(self.components, len(r3d) - 2, values.shape[1])
        self.model = PLSRegression(
            n_components=count,
            scale=False,
            max_iter=1000,
        ).fit(values, log_target)
        return self

    def predict(self, r3d: np.ndarray, physics: np.ndarray) -> np.ndarray:
        assert self.model is not None
        values = self.scaler.transform(self.imputer.transform(r3d))
        return np.ravel(self.model.predict(values))


class _PhysicsRidge:
    def __init__(self, alpha: float) -> None:
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.model = Ridge(alpha=float(alpha))

    def fit(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        log_target: np.ndarray,
    ) -> "_PhysicsRidge":
        values = self.scaler.fit_transform(
            self.imputer.fit_transform(physics)
        )
        self.model.fit(values, log_target)
        return self

    def predict(self, r3d: np.ndarray, physics: np.ndarray) -> np.ndarray:
        return self.model.predict(
            self.scaler.transform(self.imputer.transform(physics))
        )


class _FusedRidge:
    def __init__(
        self,
        components: int,
        alpha: float,
        physics_weight: float,
    ) -> None:
        self.components = int(components)
        self.alpha = float(alpha)
        self.physics_weight = float(physics_weight)
        self.r3d_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.pca_scaler = StandardScaler()
        self.physics_imputer = SimpleImputer(strategy="median")
        self.physics_scaler = RobustScaler()
        self.model = Ridge(alpha=self.alpha)

    def _transform(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        *,
        fit: bool,
    ) -> np.ndarray:
        if fit:
            r3d_scaled = self.r3d_scaler.fit_transform(r3d)
            count = min(
                self.components,
                len(r3d) - 2,
                r3d_scaled.shape[1],
            )
            self.pca = PCA(
                n_components=count,
                svd_solver="full",
            ).fit(r3d_scaled)
            latent = self.pca_scaler.fit_transform(
                self.pca.transform(r3d_scaled)
            )
            physical = self.physics_scaler.fit_transform(
                self.physics_imputer.fit_transform(physics)
            )
        else:
            assert self.pca is not None
            latent = self.pca_scaler.transform(
                self.pca.transform(self.r3d_scaler.transform(r3d))
            )
            physical = self.physics_scaler.transform(
                self.physics_imputer.transform(physics)
            )
        return np.concatenate(
            [latent, self.physics_weight * physical],
            axis=1,
        )

    def fit(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        log_target: np.ndarray,
    ) -> "_FusedRidge":
        self.model.fit(
            self._transform(r3d, physics, fit=True),
            log_target,
        )
        return self

    def predict(self, r3d: np.ndarray, physics: np.ndarray) -> np.ndarray:
        return self.model.predict(self._transform(r3d, physics, fit=False))


class _TailMixture:
    """R3D regression with train-only smooth/rough morphology gates."""

    def __init__(
        self,
        components: int,
        alpha: float,
        classifier_c: float,
        tail_fraction: float,
        gate_strength: float,
    ) -> None:
        self.components = int(components)
        self.alpha = float(alpha)
        self.classifier_c = float(classifier_c)
        self.tail_fraction = float(tail_fraction)
        self.gate_strength = float(gate_strength)
        self.r3d_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.pca_scaler = StandardScaler()
        self.physics_imputer = SimpleImputer(strategy="median")
        self.physics_scaler = RobustScaler()
        self.regressor = Ridge(alpha=self.alpha)
        self.low_gate: LogisticRegression | None = None
        self.high_gate: LogisticRegression | None = None
        self.low_center = 0.0
        self.high_center = 0.0

    def _transform(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        *,
        fit: bool,
    ) -> np.ndarray:
        if fit:
            scaled = self.r3d_scaler.fit_transform(r3d)
            count = min(
                self.components,
                len(r3d) - 2,
                scaled.shape[1],
            )
            self.pca = PCA(
                n_components=count,
                svd_solver="full",
            ).fit(scaled)
            latent = self.pca_scaler.fit_transform(
                self.pca.transform(scaled)
            )
            physical = self.physics_scaler.fit_transform(
                self.physics_imputer.fit_transform(physics)
            )
        else:
            assert self.pca is not None
            latent = self.pca_scaler.transform(
                self.pca.transform(self.r3d_scaler.transform(r3d))
            )
            physical = self.physics_scaler.transform(
                self.physics_imputer.transform(physics)
            )
        return np.concatenate([latent, physical], axis=1)

    def fit(
        self,
        r3d: np.ndarray,
        physics: np.ndarray,
        log_target: np.ndarray,
    ) -> "_TailMixture":
        values = self._transform(r3d, physics, fit=True)
        self.regressor.fit(values[:, : self.components], log_target)
        low_cut = float(np.quantile(log_target, self.tail_fraction))
        high_cut = float(np.quantile(log_target, 1.0 - self.tail_fraction))
        low = (log_target <= low_cut).astype(int)
        high = (log_target >= high_cut).astype(int)
        self.low_gate = LogisticRegression(
            C=self.classifier_c,
            class_weight="balanced",
            max_iter=5000,
        ).fit(values, low)
        self.high_gate = LogisticRegression(
            C=self.classifier_c,
            class_weight="balanced",
            max_iter=5000,
        ).fit(values, high)
        self.low_center = float(np.median(log_target[low.astype(bool)]))
        self.high_center = float(np.median(log_target[high.astype(bool)]))
        return self

    def predict(self, r3d: np.ndarray, physics: np.ndarray) -> np.ndarray:
        assert self.low_gate is not None and self.high_gate is not None
        values = self._transform(r3d, physics, fit=False)
        base = self.regressor.predict(values[:, : self.components])
        p_low = self.low_gate.predict_proba(values)[:, 1]
        p_high = self.high_gate.predict_proba(values)[:, 1]
        normalizer = np.maximum(p_low + p_high, 1.0)
        p_low = self.gate_strength * p_low / normalizer
        p_high = self.gate_strength * p_high / normalizer
        center_weight = np.clip(p_low + p_high, 0.0, 0.95)
        return (
            (1.0 - center_weight) * base
            + p_low * self.low_center
            + p_high * self.high_center
        )


def candidate_grid() -> list[Candidate]:
    """Small hypothesis-driven grid; not an unconstrained sweep."""

    candidates: list[Candidate] = []
    for components in (3, 5, 8):
        for alpha in (3.0, 10.0, 30.0):
            candidates.append(
                Candidate(
                    "r3d_ridge",
                    {"components": components, "alpha": alpha},
                )
            )
    for components in (1, 2, 3, 4):
        candidates.append(Candidate("r3d_pls", {"components": components}))
    for alpha in (3.0, 10.0, 30.0):
        candidates.append(Candidate("physics_ridge", {"alpha": alpha}))
    for components in (3, 5):
        for alpha in (10.0, 30.0):
            for weight in (0.35, 0.70):
                candidates.append(
                    Candidate(
                        "fused_ridge",
                        {
                            "components": components,
                            "alpha": alpha,
                            "physics_weight": weight,
                        },
                    )
                )
    for components in (3, 5):
        for classifier_c in (0.1, 1.0):
            candidates.append(
                Candidate(
                    "tail_mixture",
                    {
                        "components": components,
                        "alpha": 10.0,
                        "classifier_c": classifier_c,
                        "tail_fraction": 0.22,
                        "gate_strength": 0.85,
                    },
                )
            )
    return candidates


def build_model(candidate: Candidate) -> Any:
    family = candidate.family
    parameters = candidate.parameters
    if family == "r3d_ridge":
        return _R3DRidge(**parameters)
    if family == "r3d_pls":
        return _R3DPLS(**parameters)
    if family == "physics_ridge":
        return _PhysicsRidge(**parameters)
    if family == "fused_ridge":
        return _FusedRidge(**parameters)
    if family == "tail_mixture":
        return _TailMixture(**parameters)
    raise ValueError(f"unknown candidate family: {family}")


def calibrate_positive(
    reference_raw: np.ndarray,
    reference_truth: np.ndarray,
    query_raw: float,
    method: str,
) -> float:
    raw_log = np.log(np.clip(reference_raw, 1e-8, None))
    true_log = np.log(np.clip(reference_truth, 1e-8, None))
    query_log = float(np.log(max(float(query_raw), 1e-8)))
    if method == "range12":
        scale_cap, blend = 1.20, 0.75
        scale = float(
            np.clip(
                np.std(true_log) / max(np.std(raw_log), 1e-8),
                1.0 / scale_cap,
                scale_cap,
            )
        )
        prediction = (1.0 - blend) * query_log + blend * (
            (query_log - np.mean(raw_log)) * scale + np.mean(true_log)
        )
    elif method in {"range20", "range25", "range30"}:
        scale_cap, blend = {
            "range20": (2.0, 0.90),
            "range25": (2.5, 0.95),
            "range30": (3.0, 1.00),
        }[method]
        scale = float(
            np.clip(
                np.std(true_log) / max(np.std(raw_log), 1e-8),
                1.0 / scale_cap,
                scale_cap,
            )
        )
        prediction = (1.0 - blend) * query_log + blend * (
            (query_log - np.mean(raw_log)) * scale + np.mean(true_log)
        )
    elif method == "huber_affine":
        model = HuberRegressor(
            alpha=0.05,
            epsilon=1.5,
            max_iter=2000,
        ).fit(raw_log[:, None], true_log)
        slope = float(np.clip(model.coef_[0], 0.50, 2.50))
        intercept = float(
            np.median(true_log - slope * raw_log)
        )
        prediction = intercept + slope * query_log
    else:
        raise ValueError(f"unknown calibration method: {method}")
    training_low = float(np.min(true_log) - 0.35)
    training_high = float(np.max(true_log) + 0.35)
    return float(np.exp(np.clip(prediction, training_low, training_high)))


CALIBRATION_METHODS = (
    "range12",
    "range20",
    "range25",
    "range30",
    "huber_affine",
)


def cross_calibrated_predictions(
    raw: np.ndarray,
    truth: np.ndarray,
    method: str,
) -> np.ndarray:
    values = np.empty_like(raw, dtype=float)
    for index in range(len(raw)):
        keep = np.arange(len(raw)) != index
        values[index] = calibrate_positive(
            raw[keep],
            truth[keep],
            float(raw[index]),
            method,
        )
    return values


def endpoint_objective(truth: np.ndarray, prediction: np.ndarray) -> float:
    error = np.abs(np.asarray(prediction) - np.asarray(truth))
    low = truth <= np.quantile(truth, 0.20)
    high = truth >= np.quantile(truth, 0.80)
    tail = 0.5 * float(np.mean(error[low])) + 0.5 * float(
        np.mean(error[high])
    )
    bias = abs(float(np.mean(prediction - truth)))
    return float(np.mean(error) + 0.35 * tail + 0.15 * bias)
