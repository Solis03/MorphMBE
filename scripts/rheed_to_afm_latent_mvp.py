#!/usr/bin/env python3
"""Predict or retrieve AFM latents from RHEED embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rheed2morph.afm.mvp import (
    display_path,
    load_afm_array,
    load_autoencoder_checkpoint,
    preprocess_afm_array,
    read_csv,
    resolve_existing_path,
    write_csv,
)
from rheed2morph.rheed.mvp import decode_video_frames, infer_target_columns, sample_uniform_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RHEED-to-AFM latent MVP.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rheed-embeddings", type=Path, required=True)
    parser.add_argument("--rheed-embedding-index", type=Path, default=None)
    parser.add_argument("--afm-latents", type=Path, required=True)
    parser.add_argument("--afm-latent-index", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, default=None)
    parser.add_argument("--descriptor-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def load_matrix(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got shape {array.shape}")
    return np.asarray(array, dtype=np.float32)


def load_rheed_embedding_table(
    rheed_embeddings_path: Path,
    rheed_embedding_index_path: Path | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if rheed_embeddings_path.suffix.lower() == ".csv":
        rows = read_csv(rheed_embeddings_path)
        if not rows:
            raise ValueError(f"RHEED embedding CSV is empty: {rheed_embeddings_path}")
        embedding_columns = [column for column in rows[0] if column.startswith("embedding_")]
        matrix = np.asarray(
            [[float(row[column]) for column in embedding_columns] for row in rows],
            dtype=np.float32,
        )
        index = {row["sample_id"]: idx for idx, row in enumerate(rows)}
        return matrix, index
    if rheed_embedding_index_path is None:
        raise ValueError("A RHEED embedding index is required when embeddings are provided as .npy.")
    matrix = load_matrix(rheed_embeddings_path)
    index = {row["sample_id"]: idx for idx, row in enumerate(read_csv(rheed_embedding_index_path))}
    return matrix, index


def load_joined_dataset(
    manifest_path: Path,
    rheed_embeddings_path: Path,
    rheed_embedding_index_path: Path | None,
    afm_latents: Path,
    afm_latent_index: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    manifest_rows = read_csv(manifest_path)
    rheed_matrix, rheed_index = load_rheed_embedding_table(rheed_embeddings_path, rheed_embedding_index_path)
    afm_matrix = load_matrix(afm_latents)
    afm_index = {resolve_existing_path(Path(row["afm_path"])).as_posix(): idx for idx, row in enumerate(read_csv(afm_latent_index))}

    joined_rows: list[dict[str, Any]] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        afm_key = resolve_existing_path(Path(row["afm_path"])).as_posix()
        if sample_id not in rheed_index or afm_key not in afm_index:
            continue
        joined_rows.append(row)
        x_rows.append(rheed_matrix[rheed_index[sample_id]])
        y_rows.append(afm_matrix[afm_index[afm_key]])
    if not joined_rows:
        raise ValueError("No rows could be joined across manifest, RHEED embeddings, and AFM latents.")
    return joined_rows, np.stack(x_rows, axis=0), np.stack(y_rows, axis=0)


def split_groups(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([row.get("group_id", row["sample_id"]) for row in rows])
    indices = np.arange(len(rows))
    if len(sorted(set(groups))) < 2:
        return indices, indices
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    return train_idx.astype(int), test_idx.astype(int)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denominator = np.where(denominator > 0, denominator, 1.0)
    return numerator / denominator


def candidate_models() -> list[tuple[str, Any]]:
    return [
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ("knn", KNeighborsRegressor(n_neighbors=3)),
        ("mlp", MLPRegressor(hidden_layer_sizes=(128,), max_iter=1000, random_state=42, early_stopping=True)),
    ]


def fit_predict(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.transform(y_train)
    model.fit(x_train_scaled, y_train_scaled)
    pred_scaled = np.asarray(model.predict(x_test_scaled), dtype=np.float32)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled[:, None]
    return y_scaler.inverse_transform(pred_scaled)


def select_model(x_train: np.ndarray, y_train: np.ndarray, train_groups: Sequence[str]) -> tuple[str, Any, list[dict[str, Any]]]:
    unique_groups = sorted(set(train_groups))
    if len(unique_groups) < 2:
        return "ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)), []
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    best_name = ""
    best_model = None
    best_loss = float("inf")
    rows: list[dict[str, Any]] = []
    for name, model in candidate_models():
        losses: list[float] = []
        failed = False
        for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x_train, groups=train_groups), start=1):
            try:
                pred = fit_predict(model, x_train[fit_idx], y_train[fit_idx], x_train[val_idx])
                loss = float(mean_squared_error(y_train[val_idx], pred))
                losses.append(loss)
                rows.append({"model_name": name, "fold": fold_index, "latent_mse": loss, "status": "ok"})
            except Exception as exc:
                failed = True
                rows.append({"model_name": name, "fold": fold_index, "latent_mse": "", "status": f"failed: {exc}"})
                break
        if failed or not losses:
            continue
        score = float(np.mean(losses))
        if score < best_loss:
            best_loss = score
            best_name = name
            best_model = model
    if best_model is None:
        return "ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)), rows
    return best_name, best_model, rows


def nearest_train_indices(y_train: np.ndarray, y_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    distances = []
    for row in y_query:
        all_distances = np.sum((y_train - row[None, :]) ** 2, axis=1)
        nearest_index = int(np.argmin(all_distances))
        indices.append(nearest_index)
        distances.append(float(np.sqrt(all_distances[nearest_index])))
    return np.asarray(indices, dtype=int), np.asarray(distances, dtype=np.float32)


def topk_hit_rate(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    hits = 0
    for truth, pred in zip(y_true, y_pred):
        pred_distances = np.sum((y_train - pred[None, :]) ** 2, axis=1)
        truth_distances = np.sum((y_train - truth[None, :]) ** 2, axis=1)
        nearest_truth = int(np.argmin(truth_distances))
        topk = np.argsort(pred_distances)[: min(k, y_train.shape[0])]
        hits += int(nearest_truth in set(topk.tolist()))
    return hits / y_true.shape[0] if y_true.shape[0] else float("nan")


def rheed_thumbnail(path: Path) -> np.ndarray:
    frames = decode_video_frames(path)
    sampled = sample_uniform_frames(frames, 1)
    return sampled[0]


def decode_latents(model_ae: Any, latents: np.ndarray) -> np.ndarray:
    device = next(model_ae.parameters()).device
    with torch.no_grad():
        decoded = model_ae.decode(torch.from_numpy(latents.astype(np.float32)).to(device)).detach().cpu().numpy()
    return decoded[:, 0]


def write_afm_prediction_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    test_idx: np.ndarray,
    images: np.ndarray,
    title_prefix: str,
) -> None:
    count = min(test_idx.size, 8)
    selected = np.linspace(0, test_idx.size - 1, count, dtype=int)
    figure, axes = plt.subplots(1, count, figsize=(2.6 * count, 3), dpi=150)
    axes = np.atleast_1d(axes)
    for axis, offset in zip(axes, selected):
        row = rows[int(test_idx[offset])]
        axis.imshow(images[offset], cmap="viridis")
        axis.axis("off")
        axis.set_title(f"{title_prefix}\n{row['sample_id']}", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_nearest_latent_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    test_idx: np.ndarray,
    nearest_idx: np.ndarray,
    nearest_distances: np.ndarray,
    nearest_cosines: np.ndarray,
    decoded_pred: np.ndarray | None,
    train_rows: Sequence[dict[str, Any]],
    model_name: str,
    image_size: int,
) -> None:
    count = min(test_idx.size, 8)
    selected = np.linspace(0, test_idx.size - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 4 if decoded_pred is not None else 3, figsize=(12, 3 * count), dpi=150)
    axes = np.atleast_2d(axes)
    for grid_row, offset in zip(axes, selected):
        test_row = rows[int(test_idx[offset])]
        train_row = train_rows[int(nearest_idx[offset])]
        rheed = rheed_thumbnail(resolve_existing_path(Path(test_row["rheed_path"])))
        true_afm = preprocess_afm_array(load_afm_array(resolve_existing_path(Path(test_row["afm_path"]))), image_size)[0]
        nn_afm = preprocess_afm_array(load_afm_array(resolve_existing_path(Path(train_row["afm_path"]))), image_size)[0]
        panels: list[tuple[np.ndarray, str]] = [
            (rheed, f"RHEED\n{test_row['sample_id']}"),
            (true_afm, f"True AFM\n{test_row['group_id']}"),
            (
                nn_afm,
                (
                    f"{model_name} nearest prototype\n"
                    f"{train_row['sample_id']} | dist={nearest_distances[offset]:.3f} "
                    f"cos={nearest_cosines[offset]:.3f}"
                ),
            ),
        ]
        if decoded_pred is not None:
            panels.append((decoded_pred[offset], f"Decoded predicted\n{model_name}"))
        for axis, (image, title) in zip(grid_row, panels):
            axis.axis("off")
            axis.set_title(title, fontsize=8)
            if image.ndim == 3:
                axis.imshow(image)
            else:
                axis.imshow(image, cmap="viridis")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def load_descriptor_lookup(descriptor_csv: Path | None) -> tuple[dict[str, np.ndarray], list[str]] | None:
    if descriptor_csv is None:
        return None
    rows = read_csv(resolve_existing_path(descriptor_csv))
    target_names = infer_target_columns(rows)
    lookup = {resolve_existing_path(Path(row["afm_path"])).as_posix(): np.asarray([float(row[name]) for name in target_names], dtype=np.float32) for row in rows}
    return lookup, target_names


def collect_descriptor_distance(
    descriptor_lookup: tuple[dict[str, np.ndarray], list[str]] | None,
    true_path: str,
    retrieved_path: str,
) -> float | None:
    if descriptor_lookup is None:
        return None
    lookup, _ = descriptor_lookup
    true_vec = lookup.get(resolve_existing_path(Path(true_path)).as_posix())
    retrieved_vec = lookup.get(resolve_existing_path(Path(retrieved_path)).as_posix())
    if true_vec is None or retrieved_vec is None:
        return None
    return float(np.linalg.norm(true_vec - retrieved_vec))


def evaluate_prediction(
    name: str,
    y_train: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: Sequence[dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    descriptor_lookup: tuple[dict[str, np.ndarray], list[str]] | None,
    top_k: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    nearest_idx, nearest_distances = nearest_train_indices(y_train, y_pred)
    retrieved = y_train[nearest_idx]
    nn_cosines = cosine_similarity(y_true, retrieved)
    descriptor_distances: list[float] = []
    for offset, dataset_index in enumerate(test_idx):
        train_dataset_index = int(train_idx[nearest_idx[offset]])
        descriptor_distance = collect_descriptor_distance(
            descriptor_lookup,
            rows[int(dataset_index)]["afm_path"],
            rows[train_dataset_index]["afm_path"],
        )
        if descriptor_distance is not None:
            descriptor_distances.append(descriptor_distance)
    metrics = {
        "method": name,
        "latent_mse": float(mean_squared_error(y_true, y_pred)),
        "latent_cosine_similarity": float(np.mean(cosine_similarity(y_true, y_pred))),
        "nearest_neighbor_latent_distance": float(np.mean(nearest_distances)),
        "nearest_neighbor_cosine_similarity": float(np.mean(nn_cosines)),
        "retrieved_latent_mse": float(mean_squared_error(y_true, retrieved)),
        "topk_retrieval_hit_rate": float(topk_hit_rate(y_train, y_true, y_pred, top_k)),
        "descriptor_distance_mean": float(np.mean(descriptor_distances)) if descriptor_distances else None,
    }
    return metrics, nearest_idx, nearest_distances, retrieved


def write_summary(
    path: Path,
    metrics_by_method: list[dict[str, Any]],
    selected_model_name: str,
    interpretability_warning: str,
    sample_regime_note: str,
) -> None:
    best_row = next(row for row in metrics_by_method if row["method"] == selected_model_name)
    mean_row = next(row for row in metrics_by_method if row["method"] == "train_mean_latent")
    random_row = next(row for row in metrics_by_method if row["method"] == "random_train_latent")
    lines = [
        "# RHEED-to-AFM Latent MVP",
        "",
        f"- Selected model: `{selected_model_name}`",
        f"- Learned latent MSE / cosine: `{best_row['latent_mse']:.6f}` / `{best_row['latent_cosine_similarity']:.6f}`",
        f"- Mean-latent baseline MSE / cosine: `{mean_row['latent_mse']:.6f}` / `{mean_row['latent_cosine_similarity']:.6f}`",
        f"- Random-train baseline MSE / cosine: `{random_row['latent_mse']:.6f}` / `{random_row['latent_cosine_similarity']:.6f}`",
        "",
        "## Interpretation",
        "",
    ]
    if interpretability_warning:
        lines.append(f"- Warning: {interpretability_warning}")
    else:
        lines.append("- AFM autoencoder does not trigger the collapse warning; qualitative retrieval inspection remains essential.")
    if sample_regime_note:
        lines.append(f"- Sample-regime note: {sample_regime_note}")
    lines.append("- Use nearest_latent_grid.png as the main scientific decision aid.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = resolve_existing_path(args.manifest)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, x, y = load_joined_dataset(
        manifest_path,
        resolve_existing_path(args.rheed_embeddings),
        resolve_existing_path(args.rheed_embedding_index) if args.rheed_embedding_index is not None else None,
        resolve_existing_path(args.afm_latents),
        resolve_existing_path(args.afm_latent_index),
    )
    train_idx, test_idx = split_groups(rows)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_true = y[test_idx]
    train_groups = [rows[index].get("group_id", rows[index]["sample_id"]) for index in train_idx]
    model_name, model, selection_rows = select_model(x_train, y_train, train_groups)
    y_pred = fit_predict(model, x_train, y_train, x_test)

    descriptor_lookup = load_descriptor_lookup(args.descriptor_csv)
    metrics_rows: list[dict[str, Any]] = []

    learned_metrics, learned_nearest_idx, learned_nearest_distances, _ = evaluate_prediction(
        model_name,
        y_train,
        y_true,
        y_pred,
        rows,
        train_idx,
        test_idx,
        descriptor_lookup,
        args.top_k,
    )
    metrics_rows.append(learned_metrics)

    mean_latent = np.repeat(np.mean(y_train, axis=0, keepdims=True), y_true.shape[0], axis=0).astype(np.float32)
    mean_metrics, _, _, _ = evaluate_prediction(
        "train_mean_latent",
        y_train,
        y_true,
        mean_latent,
        rows,
        train_idx,
        test_idx,
        descriptor_lookup,
        args.top_k,
    )
    metrics_rows.append(mean_metrics)

    rng = np.random.default_rng(42)
    random_indices = rng.integers(0, y_train.shape[0], size=y_true.shape[0])
    random_latent = y_train[random_indices]
    random_metrics, _, _, _ = evaluate_prediction(
        "random_train_latent",
        y_train,
        y_true,
        random_latent,
        rows,
        train_idx,
        test_idx,
        descriptor_lookup,
        args.top_k,
    )
    metrics_rows.append(random_metrics)

    predictions = {
        model_name: y_pred.astype(np.float32),
        "train_mean_latent": mean_latent.astype(np.float32),
        "random_train_latent": random_latent.astype(np.float32),
    }
    np.save(out_dir / "predictions.npy", predictions, allow_pickle=True)

    autoencoder_metrics = None
    interpretability_warning = ""
    sample_regime_note = ""
    decoded_pred = None
    if args.autoencoder_checkpoint is not None:
        checkpoint_path = resolve_existing_path(args.autoencoder_checkpoint)
        model_ae, payload = load_autoencoder_checkpoint(checkpoint_path, args.device)
        decoded_pred = decode_latents(model_ae, y_pred)
        autoencoder_metrics_path = checkpoint_path.parent / "metrics.json"
        if autoencoder_metrics_path.exists():
            autoencoder_metrics = json.loads(autoencoder_metrics_path.read_text(encoding="utf-8"))
            interpretability_warning = str(autoencoder_metrics.get("quality_warning", "") or "")
        write_afm_prediction_grid(out_dir / "generated_afm_grid.png", rows, test_idx, decoded_pred, model_name)

    if len(rows) <= 6:
        sample_regime_note = "Smoke/qualitative only due to very small subset size."

    learned_nearest_idx, learned_nearest_distances = nearest_train_indices(y_train, y_pred)
    learned_retrieved = y_train[learned_nearest_idx]
    learned_nearest_cosines = cosine_similarity(y_true, learned_retrieved)
    write_nearest_latent_grid(
        out_dir / "nearest_latent_grid.png",
        rows,
        test_idx,
        learned_nearest_idx,
        learned_nearest_distances,
        learned_nearest_cosines,
        decoded_pred,
        [rows[index] for index in train_idx],
        model_name,
        image_size=128,
    )

    prediction_rows = []
    for offset, dataset_index in enumerate(test_idx):
        train_dataset_index = int(train_idx[learned_nearest_idx[offset]])
        prediction_rows.append(
            {
                "row_id": rows[int(dataset_index)].get("row_id", ""),
                "sample_id": rows[int(dataset_index)]["sample_id"],
                "group_id": rows[int(dataset_index)].get("group_id", rows[int(dataset_index)]["sample_id"]),
                "nearest_train_sample_id": rows[train_dataset_index]["sample_id"],
                "nearest_train_group_id": rows[train_dataset_index].get("group_id", rows[train_dataset_index]["sample_id"]),
                "nearest_neighbor_latent_distance": f"{float(learned_nearest_distances[offset]):.8f}",
                "nearest_neighbor_cosine_similarity": f"{float(learned_nearest_cosines[offset]):.8f}",
            }
        )
    write_csv(
        out_dir / "predicted_latent_index.csv",
        prediction_rows,
        [
            "row_id",
            "sample_id",
            "group_id",
            "nearest_train_sample_id",
            "nearest_train_group_id",
            "nearest_neighbor_latent_distance",
            "nearest_neighbor_cosine_similarity",
        ],
    )

    metrics_payload = {
        "manifest": str(manifest_path),
        "selected_model_name": model_name,
        "train_row_count": int(train_idx.size),
        "test_row_count": int(test_idx.size),
        "train_group_count": len(set(train_groups)),
        "test_group_count": len({rows[index].get("group_id", rows[index]["sample_id"]) for index in test_idx}),
        "interpretability_warning": interpretability_warning,
        "sample_regime_note": sample_regime_note,
        "methods": metrics_rows,
    }
    if autoencoder_metrics is not None:
        metrics_payload["autoencoder_metrics_path"] = str(checkpoint_path.parent / "metrics.json")
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_rows = []
    for row in metrics_rows:
        csv_rows.append(
            {
                "method": row["method"],
                "latent_mse": f"{row['latent_mse']:.8f}",
                "latent_cosine_similarity": f"{row['latent_cosine_similarity']:.8f}",
                "nearest_neighbor_latent_distance": f"{row['nearest_neighbor_latent_distance']:.8f}",
                "nearest_neighbor_cosine_similarity": f"{row['nearest_neighbor_cosine_similarity']:.8f}",
                "retrieved_latent_mse": f"{row['retrieved_latent_mse']:.8f}",
                "topk_retrieval_hit_rate": f"{row['topk_retrieval_hit_rate']:.8f}",
                "descriptor_distance_mean": "" if row["descriptor_distance_mean"] is None else f"{row['descriptor_distance_mean']:.8f}",
            }
        )
    write_csv(
        out_dir / "metrics_summary.csv",
        csv_rows,
        [
            "method",
            "latent_mse",
            "latent_cosine_similarity",
            "nearest_neighbor_latent_distance",
            "nearest_neighbor_cosine_similarity",
            "retrieved_latent_mse",
            "topk_retrieval_hit_rate",
            "descriptor_distance_mean",
        ],
    )
    if selection_rows:
        write_csv(out_dir / "model_selection.csv", selection_rows, ["model_name", "fold", "latent_mse", "status"])
    write_summary(out_dir / "summary.md", metrics_rows, model_name, interpretability_warning, sample_regime_note)
    print(f"Wrote latent MVP outputs to {display_path(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
