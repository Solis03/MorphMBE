"""Shared utilities for MVP-10 shape-bag trustworthiness analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def numeric_id(value: str) -> str:
    match = re.search(r"(\d{4,})", str(value))
    return match.group(1) if match else str(value)


def stable_hash_float(value: str) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


@dataclass
class ShapeBagTrustDataset:
    mvp8_root: Path
    mvp9_root: Path
    index_rows: list[dict[str, str]]
    target_rows: list[dict[str, str]]
    fold_rows: list[dict[str, str]]
    feature_schema: dict[str, Any]
    target_schema: dict[str, Any]
    condition_schema: dict[str, Any]
    global_features_by_sample: dict[str, dict[str, str]]
    target_by_pair: dict[str, dict[str, str]]

    @property
    def descriptor_columns(self) -> list[str]:
        wanted = list(self.condition_schema.get("descriptor_columns", [])) or list(self.target_schema.get("descriptor_columns", []))
        available = set(self.target_rows[0].keys()) if self.target_rows else set()
        return [name for name in wanted if name in available]

    @property
    def stable_columns(self) -> list[str]:
        return list(self.feature_schema.get("stable_feature_columns", []))

    @property
    def raw_columns(self) -> list[str]:
        cols = list(self.feature_schema.get("raw_240_feature_columns", []))
        if cols:
            return cols
        if self.global_features_by_sample:
            row = next(iter(self.global_features_by_sample.values()))
            return [name for name in row if name != "sample_id"]
        return []


def load_dataset(mvp8_root: str | Path, mvp9_root: str | Path, condition_schema: str | Path | None = None) -> ShapeBagTrustDataset:
    mvp8 = resolve_path(mvp8_root)
    mvp9 = resolve_path(mvp9_root)
    data = mvp9 / "data"
    cond_path = resolve_path(condition_schema) if condition_schema is not None else resolve_path(data / "target_schema_shape_bag.json")
    index_rows = read_csv(data / "supervised_shape_bag_index.csv")
    target_rows = read_csv(data / "target_conditions_shape_bag.csv")
    fold_rows = read_csv(data / "strict_fold_assignments.csv")
    feature_schema = read_json(data / "feature_schema_shape_bag.json")
    target_schema = read_json(data / "target_schema_shape_bag.json")
    cond_schema = read_json(cond_path)
    global_rows = read_csv(mvp8 / "global_sample_shape_features.csv")
    global_by_sample = {numeric_id(row.get("sample_id", "")): row for row in global_rows}
    target_by_pair = {row["pair_id"]: row for row in target_rows}
    return ShapeBagTrustDataset(mvp8, mvp9, index_rows, target_rows, fold_rows, feature_schema, target_schema, cond_schema, global_by_sample, target_by_pair)


def parse_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def make_folds(
    rows: Sequence[dict[str, str]],
    fold_rows: Sequence[dict[str, str]],
    *,
    fold_mode: str,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    all_ids = [row["pair_id"] for row in rows]
    if fold_mode == "original_mvp9":
        train = [row["pair_id"] for row in rows if row.get("split") == "train"]
        val = [row["pair_id"] for row in rows if row.get("split") in {"val", "test"}]
        if not val:
            val = [row["pair_id"] for row in rows if row.get("split") != "train"]
        return [{"fold_id": "original_mvp9", "repeat": 0, "train_ids": train, "val_ids": val}]
    if fold_mode == "leave_one_group_out":
        folds = []
        groups = sorted({row.get("group_id", row["sample_id"]) for row in rows})
        for group in groups:
            val = [row["pair_id"] for row in rows if row.get("group_id", row["sample_id"]) == group]
            train = [row["pair_id"] for row in rows if row["pair_id"] not in set(val)]
            folds.append({"fold_id": f"group_{group}", "repeat": 0, "train_ids": train, "val_ids": val})
        return folds
    if fold_mode == "repeated_group_kfold":
        groups = sorted({row.get("group_id", row["sample_id"]) for row in rows})
        rng = np.random.default_rng(seed)
        folds = []
        for repeat in range(max(1, n_repeats)):
            shuffled = list(groups)
            rng.shuffle(shuffled)
            group_to_fold = {group: idx % max(1, n_splits) for idx, group in enumerate(shuffled)}
            for fold in range(max(1, n_splits)):
                val = [row["pair_id"] for row in rows if group_to_fold[row.get("group_id", row["sample_id"])] == fold]
                train = [pair_id for pair_id in all_ids if pair_id not in set(val)]
                folds.append({"fold_id": str(fold), "repeat": repeat, "train_ids": train, "val_ids": val})
        return folds
    strict_by_pair = {row["pair_id"]: int(float(row.get("fold_id", 0))) for row in fold_rows}
    fold_ids = sorted(set(strict_by_pair.values()))
    folds = []
    for fold in fold_ids:
        val = [pair for pair, fold_value in strict_by_pair.items() if fold_value == fold]
        train = [pair_id for pair_id in all_ids if pair_id not in set(val)]
        folds.append({"fold_id": str(fold), "repeat": 0, "train_ids": train, "val_ids": val})
    return folds


def _row_features(row: dict[str, str], feature_row: dict[str, str], columns: Sequence[str]) -> list[float]:
    out = []
    for name in columns:
        value = row.get(f"shape_feature::{name}", "")
        if value == "":
            value = feature_row.get(name, "")
        out.append(finite_float(value, 0.0))
    return out


def _consensus_summary(path: str | Path) -> tuple[list[float], list[str]]:
    with np.load(resolve_path(path), allow_pickle=False) as data:
        maps = np.asarray(data["consensus_maps"], dtype=np.float32)
    features: list[float] = []
    names: list[str] = []
    for channel in range(maps.shape[0]):
        arr = maps[channel]
        vals = [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.percentile(arr, 10)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
        ]
        suffixes = ["mean", "std", "p10", "p50", "p90"]
        features.extend(vals)
        names.extend([f"consensus_c{channel}_{suffix}" for suffix in suffixes])
    return features, names


def feature_matrix(bundle: ShapeBagTrustDataset, feature_set: str, pair_ids: Sequence[str], *, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    rows_by_pair = {row["pair_id"]: row for row in bundle.index_rows}
    selected_rows = [rows_by_pair[pair] for pair in pair_ids if pair in rows_by_pair]
    rng = np.random.default_rng(seed)
    if feature_set == "stable36":
        columns = bundle.stable_columns
        matrix = [
            _row_features(row, bundle.global_features_by_sample.get(row["sample_id"], {}), columns)
            for row in selected_rows
        ]
        return np.asarray(matrix, dtype=np.float32), list(columns)
    if feature_set == "raw240_diagnostic":
        columns = bundle.raw_columns
        matrix = [
            _row_features(row, bundle.global_features_by_sample.get(row["sample_id"], {}), columns)
            for row in selected_rows
        ]
        return np.asarray(matrix, dtype=np.float32), list(columns)
    if feature_set == "brightness_only_diagnostic" or feature_set == "exposure_only_diagnostic":
        raw = bundle.raw_columns
        tokens = ("brightness", "contrast", "saturation", "shadow", "snr_score", "mask_confidence", "artifact")
        columns = [col for col in raw if any(token in col for token in tokens)]
        if not columns:
            columns = [col for col in raw if "snr" in col or "confidence" in col][:8]
        matrix = [
            _row_features(row, bundle.global_features_by_sample.get(row["sample_id"], {}), columns)
            for row in selected_rows
        ]
        return np.asarray(matrix, dtype=np.float32), list(columns)
    if feature_set == "consensus_summary":
        matrix = []
        names: list[str] = []
        for row in selected_rows:
            vals, current_names = _consensus_summary(row["shape_bag_npz"])
            if not names:
                names = current_names
            matrix.append(vals)
        return np.asarray(matrix, dtype=np.float32), names
    if feature_set == "stable36_plus_consensus_summary":
        x_a, names_a = feature_matrix(bundle, "stable36", pair_ids, seed=seed)
        x_b, names_b = feature_matrix(bundle, "consensus_summary", pair_ids, seed=seed)
        return np.concatenate([x_a, x_b], axis=1), names_a + names_b
    if feature_set == "stable36_plus_metadata":
        x_a, names_a = feature_matrix(bundle, "stable36", pair_ids, seed=seed)
        metadata = np.asarray(
            [
                [finite_float(row.get("split") == "train", 0.0), finite_float(row.get("split") == "val", 0.0), finite_float(row.get("split") == "test", 0.0)]
                for row in selected_rows
            ],
            dtype=np.float32,
        )
        return np.concatenate([x_a, metadata], axis=1), names_a + ["meta_is_train", "meta_is_val", "meta_is_test"]
    if feature_set == "random_gaussian_diagnostic":
        dim = max(1, len(bundle.stable_columns))
        return rng.normal(size=(len(selected_rows), dim)).astype(np.float32), [f"random_{idx}" for idx in range(dim)]
    if feature_set == "forbidden_id_path_diagnostic":
        matrix = np.asarray(
            [
                [
                    finite_float(row.get("sample_id", "0"), 0.0),
                    stable_hash_float(row.get("group_id", "")),
                    stable_hash_float(row.get("shape_bag_npz", "")),
                    stable_hash_float(row.get("network_input_path", "")),
                ]
                for row in selected_rows
            ],
            dtype=np.float32,
        )
        return matrix, ["forbidden_sample_id", "forbidden_group_id_hash", "forbidden_shape_path_hash", "forbidden_afm_path_hash"]
    raise ValueError(f"Unsupported feature set: {feature_set}")


def target_vector(bundle: ShapeBagTrustDataset, descriptor: str, pair_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([finite_float(bundle.target_by_pair[pair].get(descriptor, "nan")) for pair in pair_ids if pair in bundle.target_by_pair], dtype=np.float64)


def impute_scale_train_val(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    med = np.nanmedian(np.where(np.isfinite(x_train), x_train, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train_i = np.where(np.isfinite(x_train), x_train, med)
    x_val_i = np.where(np.isfinite(x_val), x_val, med)
    mean = x_train_i.mean(axis=0) if x_train_i.size else np.zeros(x_train_i.shape[1], dtype=np.float32)
    std = x_train_i.std(axis=0) if x_train_i.size else np.ones(x_train_i.shape[1], dtype=np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return (x_train_i - mean) / std, (x_val_i - mean) / std, {"feature_median": med, "feature_mean": mean, "feature_std": std}


def fit_predict_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, *, seed: int = 42) -> np.ndarray:
    y_mean = float(np.mean(y_train)) if y_train.size else 0.0
    y_std = float(np.std(y_train)) if y_train.size and float(np.std(y_train)) > 1e-8 else 1.0
    if model_name == "mean":
        return np.full(x_val.shape[0], y_mean, dtype=np.float64)
    if model_name == "median":
        return np.full(x_val.shape[0], float(np.median(y_train)) if y_train.size else y_mean, dtype=np.float64)
    y_scaled = (y_train - y_mean) / y_std
    if model_name == "ridge":
        try:
            from sklearn.linear_model import Ridge  # type: ignore

            model = Ridge(alpha=1.0)
            model.fit(x_train, y_scaled)
            return np.asarray(model.predict(x_val), dtype=np.float64) * y_std + y_mean
        except Exception:
            x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
            coef = np.linalg.pinv(x_aug.T @ x_aug + np.eye(x_aug.shape[1]) * 1e-3) @ x_aug.T @ y_scaled
            val_aug = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1)
            return np.asarray(val_aug @ coef, dtype=np.float64) * y_std + y_mean
    if model_name == "elasticnet":
        from sklearn.linear_model import ElasticNet  # type: ignore

        model = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=seed)
        model.fit(x_train, y_scaled)
        return np.asarray(model.predict(x_val), dtype=np.float64) * y_std + y_mean
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        model = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=seed)
        model.fit(x_train, y_scaled)
        return np.asarray(model.predict(x_val), dtype=np.float64) * y_std + y_mean
    if model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

        model = GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=seed)
        model.fit(x_train, y_scaled)
        return np.asarray(model.predict(x_val), dtype=np.float64) * y_std + y_mean
    if model_name == "stable_features_mlp":
        from sklearn.neural_network import MLPRegressor  # type: ignore

        model = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, learning_rate_init=1e-3, max_iter=500, random_state=seed, early_stopping=False)
        model.fit(x_train, y_scaled)
        return np.asarray(model.predict(x_val), dtype=np.float64) * y_std + y_mean
    raise ValueError(f"Unsupported model: {model_name}")


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        value = spearmanr(x, y, nan_policy="omit").correlation
        return float(value) if value is not None and math.isfinite(float(value)) else float("nan")
    except Exception:
        return correlation(np.argsort(np.argsort(np.asarray(x))), np.argsort(np.argsort(np.asarray(y))))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray, *, train_std: float) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(baseline)
    if not np.any(mask):
        return {
            "row_count": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "normalized_mae": float("nan"),
            "r2_vs_train_mean": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mean_baseline_mse": float("nan"),
            "paired_improvement_over_mean_mse": float("nan"),
        }
    true = y_true[mask]
    pred = y_pred[mask]
    base = baseline[mask]
    err = pred - true
    base_err = base - true
    mse = float(np.mean(err * err))
    base_mse = float(np.mean(base_err * base_err))
    denom = float(np.sum(base_err * base_err))
    return {
        "row_count": int(true.size),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "normalized_mae": float(np.mean(np.abs(err)) / max(train_std, 1e-8)),
        "r2_vs_train_mean": 1.0 - float(np.sum(err * err)) / max(denom, 1e-8),
        "pearson": correlation(true, pred),
        "spearman": spearman(true, pred),
        "mean_baseline_mse": base_mse,
        "paired_improvement_over_mean_mse": base_mse - mse,
    }


def bootstrap_ci(values: np.ndarray, *, bootstrap: int = 1000, seed: int = 42) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if bootstrap <= 1:
        return float(np.mean(vals)), float(np.mean(vals))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(vals[rng.integers(0, vals.size, size=vals.size)])) for _ in range(int(bootstrap))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def trust_label(summary: dict[str, Any], negative_controls_pass: bool = True) -> str:
    r2 = finite_float(summary.get("r2_vs_train_mean", "nan"))
    improvement = finite_float(summary.get("paired_improvement_over_mean_mse", "nan"))
    fold_rate = finite_float(summary.get("beats_mean_fold_rate", "nan"), 0.0)
    ci_low = finite_float(summary.get("improvement_ci_low", "nan"))
    if not negative_controls_pass:
        return "UNRELIABLE"
    if fold_rate >= 0.6 and r2 > 0.0 and improvement > 0.0 and (not math.isfinite(ci_low) or ci_low >= 0.0):
        return "SUPPORTED"
    if fold_rate >= 0.5 and improvement > 0.0:
        return "WEAK"
    return "NOT_SUPPORTED"


def write_bar_plot(path: Path, rows: Sequence[dict[str, Any]], label_key: str, value_key: str, *, title: str = "") -> None:
    finite = [row for row in rows if math.isfinite(finite_float(row.get(value_key, "nan")))]
    if not finite:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return
    fig, axis = plt.subplots(figsize=(max(7, 0.35 * len(finite)), 4))
    axis.bar(range(len(finite)), [finite_float(row[value_key]) for row in finite])
    axis.set_xticks(range(len(finite)))
    axis.set_xticklabels([str(row.get(label_key, "")) for row in finite], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(value_key)
    if title:
        axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_heatmap(path: Path, matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], *, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if matrix.size == 0:
        path.touch()
        return
    fig, axis = plt.subplots(figsize=(max(5, 0.4 * len(col_labels)), max(3, 0.25 * len(row_labels))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(col_labels)))
    axis.set_xticklabels(list(col_labels), rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(list(row_labels), fontsize=7)
    if title:
        axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def descriptor_to_condition(name: str, raw_value: float, schema: dict[str, Any]) -> float:
    mean = float(schema.get("descriptor_train_mean", {}).get(name, 0.0))
    std = float(schema.get("descriptor_train_std", {}).get(name, 1.0) or 1.0)
    return (float(raw_value) - mean) / std


def condition_to_descriptor(name: str, cond_value: float, schema: dict[str, Any]) -> float:
    mean = float(schema.get("descriptor_train_mean", {}).get(name, 0.0))
    std = float(schema.get("descriptor_train_std", {}).get(name, 1.0) or 1.0)
    return float(cond_value) * std + mean


def pair_rows_by_id(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["pair_id"]: row for row in rows}
