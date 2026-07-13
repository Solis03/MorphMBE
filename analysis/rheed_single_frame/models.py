"""Small-data model evaluation, nested selection, and uncertainty."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from analysis.rheed_roughness.run import display_path, safe_float
from analysis.rheed_single_frame.connectivity_features import PHYSICAL_INTERPRETABLE_FEATURES
from analysis.rheed_single_frame.data import ExperimentPaths, write_csv_rows
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples


class MedianRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MedianRegressor":
        self.value_ = float(np.median(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value_, dtype=float)


class OneDimensionalIsotonic(BaseEstimator, RegressorMixin):
    def __init__(self, increasing: bool = True) -> None:
        self.increasing = increasing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneDimensionalIsotonic":
        x = np.asarray(X, dtype=float)[:, 0]
        if np.unique(x).size < 2:
            self.fallback_ = float(np.median(y))
            self.model_ = None
        else:
            self.fallback_ = float(np.median(y))
            self.model_ = IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
            self.model_.fit(x, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.full(X.shape[0], self.fallback_, dtype=float)
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)[:, 0]), dtype=float)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    feature_names: tuple[str, ...]
    builder: Callable[[int, int], Any]
    simplicity_rank: int
    description: str


def _pipeline(regressor: Any, *, pca_components: int | None = None) -> Callable[[int, int], Pipeline]:
    def build(n_train: int, n_features: int) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
        if pca_components is not None and n_features > 1 and n_train > 2:
            n_components = max(1, min(int(pca_components), n_features, n_train - 1))
            steps.append(("pca", PCA(n_components=n_components, random_state=0)))
        steps.append(("regressor", clone(regressor)))
        return Pipeline(steps)

    return build


def _pls_builder(components: int) -> Callable[[int, int], Pipeline]:
    def build(n_train: int, n_features: int) -> Pipeline:
        n_components = max(1, min(int(components), n_features, n_train - 1))
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("regressor", PLSRegression(n_components=n_components)),
            ]
        )

    return build


def _gpr_builder() -> Callable[[int, int], Pipeline]:
    def build(n_train: int, n_features: int) -> Pipeline:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(
            noise_level=0.05, noise_level_bounds=(1e-5, 1.0)
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=max(1, min(4, n_features, n_train - 1)), random_state=0)),
                ("regressor", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)),
            ]
        )

    return build


def _extra_trees_builder() -> Callable[[int, int], Pipeline]:
    def build(n_train: int, n_features: int) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=128,
                        max_depth=3,
                        min_samples_leaf=max(2, int(math.ceil(n_train * 0.08))),
                        random_state=0,
                    ),
                ),
            ]
        )

    return build


def build_model_specs(
    all_feature_names: Sequence[str],
    physics_names: Sequence[str],
    nuisance_names: Sequence[str],
    embedding_names: Sequence[str],
    config: dict[str, Any],
) -> list[ModelSpec]:
    all_set = set(all_feature_names)
    interpret = tuple(name for name in PHYSICAL_INTERPRETABLE_FEATURES if name in all_set)
    extended_physics = tuple(name for name in physics_names if name in all_set and not name.endswith("_threshold_std"))
    nuisance = tuple(name for name in nuisance_names if name in all_set)
    morphology = tuple(name for name in ("morphology_index", "existing_morphology_index") if name in all_set)[:1]
    embedding = tuple(name for name in embedding_names if name in all_set)
    hybrid = extended_physics + embedding
    specs: list[ModelSpec] = [
        ModelSpec("median_baseline", "baseline", tuple(), lambda n, p: MedianRegressor(), 0, "training-fold median log Rq"),
    ]
    if morphology:
        specs.extend(
            [
                ModelSpec("morphology_linear", "existing_morphology", morphology, _pipeline(LinearRegression()), 1, "old morphology score linear"),
                ModelSpec("morphology_ridge", "existing_morphology", morphology, _pipeline(Ridge(alpha=1.0)), 2, "old morphology score ridge"),
                ModelSpec("morphology_isotonic_increasing", "existing_morphology", morphology, lambda n, p: OneDimensionalIsotonic(True), 2, "pre-specified increasing isotonic"),
            ]
        )
    if nuisance:
        specs.append(ModelSpec("nuisance_ridge", "nuisance", nuisance, _pipeline(Ridge(alpha=3.0)), 3, "acquisition nuisance ridge"))
    if interpret:
        specs.append(ModelSpec("connectivity_interpretable_ridge", "physics_connectivity", interpret, _pipeline(Ridge(alpha=3.0)), 4, "compact pre-registered connectivity features"))
    if extended_physics:
        specs.extend(
            [
                ModelSpec("physics_ridge", "physics", extended_physics, _pipeline(Ridge(alpha=10.0)), 5, "all physics features ridge"),
                ModelSpec(
                    "physics_elasticnet",
                    "physics",
                    extended_physics,
                    _pipeline(ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=20000, random_state=0)),
                    6,
                    "all physics features elastic net",
                ),
                ModelSpec("physics_pls2", "physics", extended_physics, _pls_builder(2), 6, "all physics features PLS"),
                ModelSpec("physics_svr_pca4", "physics", extended_physics, _pipeline(SVR(C=2.0, epsilon=0.05, gamma="scale"), pca_components=4), 7, "physics PCA + RBF SVR"),
                ModelSpec("physics_gpr_pca4", "physics", extended_physics, _gpr_builder(), 8, "physics PCA + Matern GP"),
                ModelSpec("physics_extra_trees_shallow", "physics", extended_physics, _extra_trees_builder(), 9, "strongly regularized shallow ExtraTrees"),
            ]
        )
    if embedding:
        for comp in config.get("models", {}).get("embedding_pca_components", [2, 4, 8]):
            specs.append(
                ModelSpec(
                    f"frozen_resnet50_ridge_pca{comp}",
                    "frozen_embedding",
                    embedding,
                    _pipeline(Ridge(alpha=10.0), pca_components=int(comp)),
                    8 + int(comp),
                    "frozen local ResNet50 embedding + PCA + ridge",
                )
            )
        specs.append(ModelSpec("frozen_resnet50_pls2", "frozen_embedding", embedding, _pls_builder(2), 10, "frozen local ResNet50 embedding + PLS"))
    if hybrid:
        specs.extend(
            [
                ModelSpec("hybrid_ridge_pca4", "hybrid", hybrid, _pipeline(Ridge(alpha=10.0), pca_components=4), 11, "physics + embedding PCA ridge"),
                ModelSpec("hybrid_pls2", "hybrid", hybrid, _pls_builder(2), 11, "physics + embedding PLS"),
                ModelSpec("hybrid_svr_pca4", "hybrid", hybrid, _pipeline(SVR(C=1.5, epsilon=0.06, gamma="scale"), pca_components=4), 12, "physics + embedding PCA SVR"),
            ]
        )
    return specs


def make_model_table(
    target_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    embedding_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    features_by_id = {str(row["sample_id"]): row for row in feature_rows}
    embeddings_by_id = {str(row["sample_id"]): row for row in embedding_rows}
    rows: list[dict[str, Any]] = []
    for target in target_rows:
        sid = str(target["sample_id"])
        rq = safe_float(target.get("rq_nm"), math.nan)
        if not math.isfinite(rq) or rq <= 0:
            continue
        row = {
            "sample_id": sid,
            "sample_group_id": target.get("sample_group_id", sid),
            "growth_run_id": target.get("growth_run_id", sid),
            "selected_afm_scan_id": target.get("selected_afm_scan_id", ""),
            "selected_height_map_path": target.get("selected_height_map_path", ""),
            "manual_rheed_path": features_by_id.get(sid, {}).get("manual_rheed_path", ""),
            "rq_true_nm": rq,
            "log_rq_true": math.log10(rq),
            **features_by_id.get(sid, {}),
            **embeddings_by_id.get(sid, {}),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def feature_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    if not feature_names:
        return np.zeros((len(rows), 1), dtype=float)
    values = []
    for row in rows:
        values.append([safe_float(row.get(name), math.nan) for name in feature_names])
    return np.asarray(values, dtype=float)


def _outer_splits(rows: Sequence[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    groups = np.asarray([str(row.get("growth_run_id") or row.get("sample_group_id") or row["sample_id"]) for row in rows])
    y = np.asarray([safe_float(row["log_rq_true"]) for row in rows], dtype=float)
    if len(np.unique(groups)) >= 3:
        splitter = LeaveOneGroupOut()
        return [(train, test, f"group_{groups[test][0]}") for train, test in splitter.split(np.zeros_like(y), y, groups)]
    return [
        (
            np.asarray([j for j in range(len(rows)) if j != i], dtype=int),
            np.asarray([i], dtype=int),
            f"sample_{rows[i]['sample_id']}",
        )
        for i in range(len(rows))
    ]


def _inner_splits(train_rows: Sequence[dict[str, Any]], max_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([str(row.get("growth_run_id") or row.get("sample_group_id") or row["sample_id"]) for row in train_rows])
    y = np.asarray([safe_float(row["log_rq_true"]) for row in train_rows], dtype=float)
    unique = np.unique(groups)
    if unique.size >= 3:
        n_splits = min(int(max_splits), unique.size)
        splitter = GroupKFold(n_splits=n_splits)
        return [(train, test) for train, test in splitter.split(np.zeros_like(y), y, groups)]
    if len(train_rows) >= 3:
        return [(np.asarray([j for j in range(len(train_rows)) if j != i]), np.asarray([i])) for i in range(len(train_rows))]
    return []


def fit_predict(spec: ModelSpec, train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, Any]:
    X_train = feature_matrix(train_rows, spec.feature_names)
    y_train = np.asarray([safe_float(row["log_rq_true"]) for row in train_rows], dtype=float)
    X_test = feature_matrix(test_rows, spec.feature_names)
    model = spec.builder(len(train_rows), X_train.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    return pred, model


def _safe_fit_predict(spec: ModelSpec, train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, Any | None, str]:
    try:
        pred, model = fit_predict(spec, train_rows, test_rows)
        if not np.all(np.isfinite(pred)):
            raise ValueError("non-finite prediction")
        return pred, model, "ok"
    except Exception as exc:
        fallback = float(np.median([safe_float(row["log_rq_true"]) for row in train_rows]))
        return np.full(len(test_rows), fallback), None, f"fallback_median_after_{type(exc).__name__}"


def metrics_from_predictions(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_log) & np.isfinite(pred_log)
    if mask.sum() < 3:
        return {key: math.nan for key in ["mae_log", "rmse_log", "r2_log", "spearman_log", "mae_nm", "median_ae_nm", "rmse_nm", "r2_nm", "spearman_nm", "pearson_nm", "pairwise_rank_concordance"]}
    y = y_log[mask]
    p = pred_log[mask]
    y_nm = np.power(10.0, y)
    p_nm = np.power(10.0, p)
    spearman_log = stats.spearmanr(y, p).statistic if np.unique(y).size > 1 and np.unique(p).size > 1 else math.nan
    spearman_nm = stats.spearmanr(y_nm, p_nm).statistic if np.unique(y_nm).size > 1 and np.unique(p_nm).size > 1 else math.nan
    pearson_nm = stats.pearsonr(y_nm, p_nm).statistic if np.unique(y_nm).size > 1 and np.unique(p_nm).size > 1 else math.nan
    concordant = 0
    total = 0
    for i in range(len(y_nm)):
        for j in range(i + 1, len(y_nm)):
            true_order = np.sign(y_nm[i] - y_nm[j])
            pred_order = np.sign(p_nm[i] - p_nm[j])
            if true_order != 0:
                total += 1
                concordant += int(true_order == pred_order)
    return {
        "mae_log": float(mean_absolute_error(y, p)),
        "rmse_log": float(math.sqrt(mean_squared_error(y, p))),
        "r2_log": float(r2_score(y, p)),
        "spearman_log": safe_float(spearman_log, math.nan),
        "mae_nm": float(mean_absolute_error(y_nm, p_nm)),
        "median_ae_nm": float(np.median(np.abs(y_nm - p_nm))),
        "rmse_nm": float(math.sqrt(mean_squared_error(y_nm, p_nm))),
        "r2_nm": float(r2_score(y_nm, p_nm)),
        "spearman_nm": safe_float(spearman_nm, math.nan),
        "pearson_nm": safe_float(pearson_nm, math.nan),
        "pairwise_rank_concordance": float(concordant / total) if total else math.nan,
    }


def evaluate_fixed_models(rows: Sequence[dict[str, Any]], specs: Sequence[ModelSpec], removelist: RemovelistAudit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert_no_removed_samples((row["sample_id"] for row in rows), removelist.sample_ids, context="fixed-model CV")
    splits = _outer_splits(rows)
    predictions: list[dict[str, Any]] = []
    for spec in specs:
        for train_idx, test_idx, fold_id in splits:
            train_rows = [rows[i] for i in train_idx]
            test_rows = [rows[i] for i in test_idx]
            pred, _, status = _safe_fit_predict(spec, train_rows, test_rows)
            for row, pred_log in zip(test_rows, pred):
                predictions.append(
                    {
                        "model_name": spec.name,
                        "model_family": spec.family,
                        "sample_id": row["sample_id"],
                        "sample_group_id": row.get("sample_group_id", row["sample_id"]),
                        "growth_run_id": row.get("growth_run_id", row["sample_id"]),
                        "rq_true_nm": row["rq_true_nm"],
                        "log_rq_true": row["log_rq_true"],
                        "rq_pred_nm": float(10**pred_log),
                        "log_rq_pred": float(pred_log),
                        "absolute_error_nm": float(abs(10**pred_log - row["rq_true_nm"])),
                        "squared_error": float((10**pred_log - row["rq_true_nm"]) ** 2),
                        "outer_fold_id": fold_id,
                        "fit_status": status,
                        "removelist_checked": 1,
                    }
                )
    comparison = comparison_rows_from_prediction_rows(predictions)
    return predictions, comparison


def comparison_rows_from_prediction_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_model.setdefault(str(row.get("model_name")), []).append(row)
    comparison = []
    median_mae = math.nan
    for model_name, model_rows in sorted(by_model.items()):
        y = np.asarray([safe_float(row["log_rq_true"]) for row in model_rows], dtype=float)
        p = np.asarray([safe_float(row["log_rq_pred"]) for row in model_rows], dtype=float)
        metrics = metrics_from_predictions(y, p)
        if model_name == "median_baseline":
            median_mae = metrics["mae_log"]
        comparison.append(
            {
                "model_name": model_name,
                "model_family": model_rows[0].get("model_family", ""),
                "n_samples": len(model_rows),
                **metrics,
                "mae_log_improvement_vs_median": median_mae - metrics["mae_log"] if math.isfinite(median_mae) else math.nan,
            }
        )
    return comparison


def inner_cv_score(spec: ModelSpec, train_rows: Sequence[dict[str, Any]], max_splits: int) -> tuple[float, float, list[float]]:
    splits = _inner_splits(train_rows, max_splits)
    if not splits:
        return math.inf, math.inf, []
    preds = np.full(len(train_rows), np.nan, dtype=float)
    y = np.asarray([safe_float(row["log_rq_true"]) for row in train_rows], dtype=float)
    for inner_train_idx, inner_test_idx in splits:
        inner_train = [train_rows[i] for i in inner_train_idx]
        inner_test = [train_rows[i] for i in inner_test_idx]
        pred, _, _ = _safe_fit_predict(spec, inner_train, inner_test)
        preds[inner_test_idx] = pred
    residuals = np.abs(preds - y)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return math.inf, math.inf, []
    mean = float(np.mean(residuals))
    se = float(np.std(residuals, ddof=1) / math.sqrt(residuals.size)) if residuals.size > 1 else 0.0
    return mean, se, residuals.tolist()


def select_one_se(rows: Sequence[dict[str, Any]], specs: Sequence[ModelSpec], max_splits: int) -> tuple[ModelSpec, list[dict[str, Any]], list[float]]:
    results = []
    residuals_by_name: dict[str, list[float]] = {}
    for spec in specs:
        mean, se, residuals = inner_cv_score(spec, rows, max_splits)
        residuals_by_name[spec.name] = residuals
        results.append({"model_name": spec.name, "inner_mae_log": mean, "inner_se_log": se, "simplicity_rank": spec.simplicity_rank})
    finite = [row for row in results if math.isfinite(row["inner_mae_log"])]
    if not finite:
        return specs[0], results, []
    best = min(finite, key=lambda row: row["inner_mae_log"])
    threshold = best["inner_mae_log"] + best["inner_se_log"]
    eligible = [row for row in finite if row["inner_mae_log"] <= threshold]
    chosen_row = min(eligible, key=lambda row: (row["simplicity_rank"], row["inner_mae_log"]))
    chosen = next(spec for spec in specs if spec.name == chosen_row["model_name"])
    return chosen, results, residuals_by_name.get(chosen.name, [])


def conformal_q(residuals: Sequence[float], nominal: float) -> float:
    arr = np.asarray([value for value in residuals if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return 0.0
    alpha = 1.0 - float(nominal)
    rank = min(arr.size, int(math.ceil((arr.size + 1) * (1.0 - alpha)))) - 1
    return float(np.sort(arr)[rank])


def domain_distance(train_rows: Sequence[dict[str, Any]], test_row: dict[str, Any], feature_names: Sequence[str]) -> tuple[float, float]:
    X_train = feature_matrix(train_rows, feature_names)
    X_test = feature_matrix([test_row], feature_names)
    if X_train.shape[1] == 0:
        return 0.0, 100.0
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    Xt = scaler.fit_transform(imputer.fit_transform(X_train))
    xv = scaler.transform(imputer.transform(X_test))
    distances = np.linalg.norm(Xt - xv, axis=1)
    nearest = float(np.min(distances)) if distances.size else 0.0
    reference = float(np.median(np.linalg.norm(Xt - np.median(Xt, axis=0), axis=1))) if Xt.size else 1.0
    score = 100.0 / (1.0 + nearest / max(reference, 1e-8))
    return nearest, float(np.clip(score, 0.0, 100.0))


def bootstrap_predictions(spec: ModelSpec, train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]], seed: int, n_boot: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    preds = []
    n = len(train_rows)
    for _ in range(max(1, int(n_boot))):
        indices = rng.integers(0, n, size=n)
        boot = [train_rows[int(i)] for i in indices]
        pred, _, status = _safe_fit_predict(spec, boot, test_rows)
        if status == "ok":
            preds.append(pred)
    if not preds:
        pred, _, _ = _safe_fit_predict(spec, train_rows, test_rows)
        preds.append(pred)
    return np.vstack(preds)


def _prediction_sensitivity(
    model: Any,
    spec: ModelSpec,
    perturbed_rows: Sequence[dict[str, Any]],
    original_pred_log: float,
    rq_iqr_nm: float,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    if not perturbed_rows or model is None:
        return 0.0, rows
    Xp = feature_matrix(perturbed_rows, spec.feature_names)
    try:
        preds_log = np.asarray(model.predict(Xp), dtype=float).reshape(-1)
    except Exception:
        return 0.0, rows
    original_nm = 10**original_pred_log
    deltas = np.abs(np.power(10.0, preds_log) - original_nm)
    rel = deltas / max(float(rq_iqr_nm), 1e-8)
    for row, pred_log, delta, rel_delta in zip(perturbed_rows, preds_log, deltas, rel):
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "perturbation": row.get("perturbation", ""),
                "log_rq_pred": float(pred_log),
                "rq_pred_nm": float(10**pred_log),
                "prediction_change_nm": float(delta),
                "prediction_change_relative_to_dataset_iqr": float(rel_delta),
                "selected_model_name": spec.name,
            }
        )
    return float(np.max(rel)) if rel.size else 0.0, rows


def evaluate_nested_selector(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[ModelSpec],
    config: dict[str, Any],
    removelist: RemovelistAudit,
    perturbation_rows_by_sample: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assert_no_removed_samples((row["sample_id"] for row in rows), removelist.sample_ids, context="nested CV")
    splits = _outer_splits(rows)
    y_nm_all = np.asarray([safe_float(row["rq_true_nm"]) for row in rows], dtype=float)
    rq_iqr_nm = float(np.percentile(y_nm_all, 75) - np.percentile(y_nm_all, 25)) if y_nm_all.size else 1.0
    nominal = float(config.get("models", {}).get("confidence_interval_nominal", 0.90))
    max_splits = int(config.get("models", {}).get("max_inner_splits", 5))
    n_boot = int(config.get("models", {}).get("bootstrap_ensembles", 32))
    seed = int(config.get("random_seed", 0))
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx, fold_id) in enumerate(splits):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        chosen, inner_results, residuals = select_one_se(train_rows, specs, max_splits)
        pred, model, status = _safe_fit_predict(chosen, train_rows, test_rows)
        q = conformal_q(residuals, nominal)
        boot = bootstrap_predictions(chosen, train_rows, test_rows, seed + fold_idx, n_boot)
        for inner in inner_results:
            selections.append(
                {
                    "outer_fold_id": fold_id,
                    "held_out_sample_id": ";".join(str(row["sample_id"]) for row in test_rows),
                    "candidate_model": inner["model_name"],
                    "selected_model": chosen.name,
                    "inner_mae_log": inner["inner_mae_log"],
                    "inner_se_log": inner["inner_se_log"],
                    "one_standard_error_selected": int(inner["model_name"] == chosen.name),
                }
            )
        for local_idx, (row, pred_log) in enumerate(zip(test_rows, pred)):
            lower_log = float(pred_log - q)
            upper_log = float(pred_log + q)
            boot_std_log = float(np.std(boot[:, local_idx])) if boot.shape[0] > 1 else 0.0
            nearest, support = domain_distance(train_rows, row, chosen.feature_names)
            pert_map = perturbation_rows_by_sample or {}
            sensitivity, pert_rows = _prediction_sensitivity(model, chosen, pert_map.get(str(row["sample_id"]), []), float(pred_log), rq_iqr_nm)
            perturbation_rows.extend(pert_rows)
            pred_nm = float(10**pred_log)
            lower_nm = float(10**lower_log)
            upper_nm = float(10**upper_log)
            width_nm = upper_nm - lower_nm
            width_score = 100.0 / (1.0 + width_nm / max(rq_iqr_nm, 1e-8))
            ensemble_std_nm = float(abs(math.log(10) * pred_nm * boot_std_log))
            ensemble_score = 100.0 / (1.0 + ensemble_std_nm / max(rq_iqr_nm, 1e-8))
            sensitivity_score = 100.0 / (1.0 + sensitivity)
            confidence = float(np.clip(np.mean([width_score, support, ensemble_score, sensitivity_score]), 0.0, 100.0))
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "sample_group_id": row.get("sample_group_id", row["sample_id"]),
                    "growth_run_id": row.get("growth_run_id", row["sample_id"]),
                    "manual_rheed_path": row.get("manual_rheed_path", ""),
                    "selected_afm_scan_id": row.get("selected_afm_scan_id", ""),
                    "selected_height_map_path": row.get("selected_height_map_path", ""),
                    "rq_true_nm": row["rq_true_nm"],
                    "log_rq_true": row["log_rq_true"],
                    "rq_pred_nm": pred_nm,
                    "log_rq_pred": float(pred_log),
                    "absolute_error_nm": float(abs(pred_nm - row["rq_true_nm"])),
                    "squared_error": float((pred_nm - row["rq_true_nm"]) ** 2),
                    "prediction_interval_lower_nm": lower_nm,
                    "prediction_interval_upper_nm": upper_nm,
                    "prediction_interval_width_nm": width_nm,
                    "confidence_score": confidence,
                    "ensemble_std": ensemble_std_nm,
                    "domain_support_score": support,
                    "ood_distance": nearest,
                    "perturbation_sensitivity": sensitivity,
                    "selected_model_family": chosen.family,
                    "selected_model_name": chosen.name,
                    "selected_hyperparameters": json.dumps({"model": chosen.name, "features": len(chosen.feature_names), "fit_status": status}, sort_keys=True),
                    "outer_fold_id": fold_id,
                    "removelist_checked": 1,
                    "qc_flags": "" if status == "ok" else status,
                    "prediction_interval_nominal": nominal,
                }
            )
    return predictions, selections, perturbation_rows


def confidence_calibration_rows(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not predictions:
        return []
    rows = []
    y = np.asarray([safe_float(row["rq_true_nm"]) for row in predictions], dtype=float)
    lower = np.asarray([safe_float(row["prediction_interval_lower_nm"]) for row in predictions], dtype=float)
    upper = np.asarray([safe_float(row["prediction_interval_upper_nm"]) for row in predictions], dtype=float)
    conf = np.asarray([safe_float(row["confidence_score"]) for row in predictions], dtype=float)
    err = np.asarray([safe_float(row["absolute_error_nm"]) for row in predictions], dtype=float)
    width = upper - lower
    for nominal in (0.80, 0.90):
        # The stored interval is 90%; the 80% row is a central shrink diagnostic.
        center = (lower + upper) / 2.0
        scale = nominal / 0.90
        lo = center - (center - lower) * scale
        hi = center + (upper - center) * scale
        rows.append(
            {
                "interval_nominal": nominal,
                "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                "mean_interval_width_nm": float(np.mean(hi - lo)),
                "n_samples": len(predictions),
            }
        )
    quantiles = np.quantile(conf, [0, 1 / 3, 2 / 3, 1]) if len(conf) >= 3 else np.asarray([0, 33, 67, 100], dtype=float)
    for idx in range(3):
        if idx == 2:
            mask = (conf >= quantiles[idx]) & (conf <= quantiles[idx + 1])
        else:
            mask = (conf >= quantiles[idx]) & (conf < quantiles[idx + 1])
        if mask.any():
            rows.append(
                {
                    "confidence_bin": idx + 1,
                    "confidence_min": float(np.min(conf[mask])),
                    "confidence_max": float(np.max(conf[mask])),
                    "mean_confidence": float(np.mean(conf[mask])),
                    "mean_absolute_error_nm": float(np.mean(err[mask])),
                    "mean_interval_width_nm": float(np.mean(width[mask])),
                    "n_samples": int(mask.sum()),
                }
            )
    return rows


def final_model_comparison(fixed_comparison: Sequence[dict[str, Any]], nested_predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(fixed_comparison)
    y = np.asarray([safe_float(row["log_rq_true"]) for row in nested_predictions], dtype=float)
    p = np.asarray([safe_float(row["log_rq_pred"]) for row in nested_predictions], dtype=float)
    metrics = metrics_from_predictions(y, p)
    cover = np.mean(
        [
            safe_float(row["prediction_interval_lower_nm"]) <= safe_float(row["rq_true_nm"]) <= safe_float(row["prediction_interval_upper_nm"])
            for row in nested_predictions
        ]
    )
    rows.append(
        {
            "model_name": "nested_one_se_selector",
            "model_family": "nested_selector",
            "n_samples": len(nested_predictions),
            **metrics,
            "interval_90_coverage": float(cover),
            "mean_interval_width_nm": float(np.mean([safe_float(row["prediction_interval_width_nm"]) for row in nested_predictions])) if nested_predictions else math.nan,
        }
    )
    baseline = next((row for row in rows if row.get("model_name") == "median_baseline"), None)
    if baseline is not None:
        base_mae = safe_float(baseline.get("mae_log"), math.nan)
        for row in rows:
            row["mae_log_improvement_vs_median"] = base_mae - safe_float(row.get("mae_log"), math.nan)
    return rows


def influence_analysis(predictions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for removed in predictions:
        subset = [row for row in predictions if row["sample_id"] != removed["sample_id"]]
        y = np.asarray([safe_float(row["log_rq_true"]) for row in subset], dtype=float)
        p = np.asarray([safe_float(row["log_rq_pred"]) for row in subset], dtype=float)
        metrics = metrics_from_predictions(y, p)
        rows.append({"removed_sample_id": removed["sample_id"], "n_remaining": len(subset), **metrics})
    return rows


def sensitivity_without_6023(predictions: Sequence[dict[str, Any]], removelist: RemovelistAudit) -> list[dict[str, Any]]:
    if "6023" in set(removelist.sample_ids):
        return [{"analysis": "without_6023", "status": "not_applicable_6023_removed_by_canonical_removelist"}]
    all_y = np.asarray([safe_float(row["log_rq_true"]) for row in predictions], dtype=float)
    all_p = np.asarray([safe_float(row["log_rq_pred"]) for row in predictions], dtype=float)
    subset = [row for row in predictions if row["sample_id"] != "6023"]
    sub_y = np.asarray([safe_float(row["log_rq_true"]) for row in subset], dtype=float)
    sub_p = np.asarray([safe_float(row["log_rq_pred"]) for row in subset], dtype=float)
    return [
        {"analysis": "all_valid_non_removelist", "status": "ok", **metrics_from_predictions(all_y, all_p)},
        {"analysis": "excluding_6023", "status": "ok", **metrics_from_predictions(sub_y, sub_p)},
    ]


def permutation_tests(predictions: Sequence[dict[str, Any]], fixed_predictions: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    nested_y = np.asarray([safe_float(row["rq_true_nm"]) for row in predictions], dtype=float)
    nested_p = np.asarray([safe_float(row["rq_pred_nm"]) for row in predictions], dtype=float)
    observed_spearman = stats.spearmanr(nested_y, nested_p).statistic if len(predictions) >= 3 else math.nan
    median_rows = [row for row in fixed_predictions if row.get("model_name") == "median_baseline"]
    base_by_id = {row["sample_id"]: safe_float(row["rq_pred_nm"]) for row in median_rows}
    nested_mae = np.mean(np.abs(nested_y - nested_p))
    base_mae = np.mean([abs(safe_float(row["rq_true_nm"]) - base_by_id.get(row["sample_id"], math.nan)) for row in predictions])
    observed_improvement = base_mae - nested_mae
    rng = np.random.default_rng(int(config.get("random_seed", 0)) + 33)
    n = int(config.get("models", {}).get("permutation_resamples", 2000))
    spearman_extreme = 0
    improvement_extreme = 0
    total = 0
    for _ in range(n):
        perm = rng.permutation(nested_y)
        stat = stats.spearmanr(perm, nested_p).statistic if np.unique(perm).size > 1 and np.unique(nested_p).size > 1 else math.nan
        if math.isfinite(stat) and math.isfinite(observed_spearman):
            spearman_extreme += int(abs(stat) >= abs(observed_spearman))
        perm_improve = base_mae - np.mean(np.abs(perm - nested_p))
        if math.isfinite(observed_improvement):
            improvement_extreme += int(perm_improve >= observed_improvement)
        total += 1
    return {
        "nested_spearman_nm": safe_float(observed_spearman, math.nan),
        "nested_spearman_permutation_p": float((spearman_extreme + 1) / (total + 1)),
        "mae_improvement_vs_median_nm": safe_float(observed_improvement, math.nan),
        "mae_improvement_permutation_p": float((improvement_extreme + 1) / (total + 1)),
        "permutation_resamples": n,
    }


def feature_importance_from_ridge(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    removelist: RemovelistAudit,
) -> list[dict[str, Any]]:
    assert_no_removed_samples((row["sample_id"] for row in rows), removelist.sample_ids, context="feature importance")
    splits = _outer_splits(rows)
    coefs: list[np.ndarray] = []
    for train_idx, _, _ in splits:
        train_rows = [rows[i] for i in train_idx]
        X = feature_matrix(train_rows, feature_names)
        y = np.asarray([safe_float(row["log_rq_true"]) for row in train_rows], dtype=float)
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("regressor", Ridge(alpha=10.0))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        coefs.append(np.asarray(model.named_steps["regressor"].coef_, dtype=float).reshape(-1))
    if not coefs:
        return []
    arr = np.vstack(coefs)
    rows_out = []
    for idx, name in enumerate(feature_names):
        values = arr[:, idx]
        rows_out.append(
            {
                "feature": name,
                "feature_group": "nuisance" if name.startswith(("mean_", "median_", "intensity_", "dynamic_", "saturation_", "underexposure_", "background_", "sharpness", "image_", "valid_roi", "pattern_", "bright_pixel_centroid")) else "physics",
                "mean_coefficient": float(np.mean(values)),
                "mean_abs_coefficient": float(np.mean(np.abs(values))),
                "std_coefficient": float(np.std(values)),
                "fold_count": int(arr.shape[0]),
            }
        )
    return sorted(rows_out, key=lambda row: row["mean_abs_coefficient"], reverse=True)


def write_model_outputs(
    paths: ExperimentPaths,
    fixed_predictions: Sequence[dict[str, Any]],
    nested_predictions: Sequence[dict[str, Any]],
    model_comparison: Sequence[dict[str, Any]],
    hyperparameter_rows: Sequence[dict[str, Any]],
    confidence_rows: Sequence[dict[str, Any]],
    perturbation_rows: Sequence[dict[str, Any]],
    importance_rows: Sequence[dict[str, Any]],
    influence_rows: Sequence[dict[str, Any]],
    sensitivity_rows: Sequence[dict[str, Any]],
    removelist: RemovelistAudit,
) -> None:
    assert_no_removed_samples((row["sample_id"] for row in nested_predictions), removelist.sample_ids, context="prediction output writing")
    write_csv_rows(paths.outputs_dir / "fixed_model_oof_predictions.csv", fixed_predictions)
    write_csv_rows(paths.outputs_dir / "nested_selector_oof_predictions.csv", nested_predictions)
    write_csv_rows(paths.outputs_dir / "model_comparison.csv", model_comparison)
    write_csv_rows(paths.outputs_dir / "hyperparameter_selection.csv", hyperparameter_rows)
    write_csv_rows(paths.outputs_dir / "confidence_calibration.csv", confidence_rows)
    write_csv_rows(paths.outputs_dir / "perturbation_stability.csv", perturbation_rows)
    write_csv_rows(paths.outputs_dir / "feature_importance.csv", importance_rows)
    write_csv_rows(paths.outputs_dir / "influence_analysis.csv", influence_rows)
    write_csv_rows(paths.outputs_dir / "sensitivity_without_6023.csv", sensitivity_rows)

