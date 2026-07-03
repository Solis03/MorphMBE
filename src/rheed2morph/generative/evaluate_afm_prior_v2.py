"""Evaluate AFM prior v2 generated samples."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.afm_prior_v2_utils import V2_DESCRIPTOR_NAMES, compute_afm_descriptors_v2
from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_csv_rows,
    read_json,
    resolve_repo_path,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AFM prior v2 samples.")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--real-index", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _safe_float(value: str | float | int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _descriptor_array(rows: list[dict[str, Any]], columns: list[str], prefix: str = "") -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for col in columns:
            key = f"{prefix}{col}" if prefix else col
            values.append(_safe_float(row.get(key, "nan")))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


def _distribution_rows(real_rows: list[dict[str, str]], generated_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    real = _descriptor_array(real_rows, columns)
    generated = _descriptor_array(generated_rows, columns, prefix="generated_")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(columns):
        real_col = real[:, index]
        gen_col = generated[:, index]
        real_col = real_col[np.isfinite(real_col)]
        gen_col = gen_col[np.isfinite(gen_col)]
        if real_col.size == 0 or gen_col.size == 0:
            continue
        rows.append(
            {
                "descriptor": name,
                "real_mean": f"{float(np.mean(real_col)):.10g}",
                "generated_mean": f"{float(np.mean(gen_col)):.10g}",
                "abs_mean_delta": f"{abs(float(np.mean(real_col)) - float(np.mean(gen_col))):.10g}",
                "real_std": f"{float(np.std(real_col)):.10g}",
                "generated_std": f"{float(np.std(gen_col)):.10g}",
                "abs_std_delta": f"{abs(float(np.std(real_col)) - float(np.std(gen_col))):.10g}",
            }
        )
    return rows


def _condition_consistency_rows(generated_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in columns:
        pred = []
        target = []
        for row in generated_rows:
            gen = _safe_float(row.get(f"generated_{name}", "nan"))
            req = _safe_float(row.get(f"requested_{name}", "nan"))
            if np.isfinite(gen) and np.isfinite(req):
                pred.append(gen)
                target.append(req)
        if not pred:
            continue
        x = np.asarray(target, dtype=np.float32)
        y = np.asarray(pred, dtype=np.float32)
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 and np.std(x) > 1e-8 and np.std(y) > 1e-8 else float("nan")
        rows.append(
            {
                "descriptor": name,
                "mae": f"{float(np.mean(np.abs(y - x))):.10g}",
                "rmse": f"{float(np.sqrt(np.mean((y - x) ** 2))):.10g}",
                "correlation": "" if not np.isfinite(corr) else f"{corr:.10g}",
            }
        )
    return rows


def _two_sample_descriptor_accuracy(real_matrix: np.ndarray, generated_matrix: np.ndarray) -> float:
    if real_matrix.shape[0] < 2 or generated_matrix.shape[0] < 2:
        return float("nan")
    real = np.nan_to_num(real_matrix, nan=0.0)
    generated = np.nan_to_num(generated_matrix, nan=0.0)
    all_points = np.concatenate([real, generated], axis=0)
    labels = np.asarray([0] * real.shape[0] + [1] * generated.shape[0])
    distances = np.sqrt(np.sum((all_points[:, None, :] - all_points[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(distances, np.inf)
    closest = np.argmin(distances, axis=1)
    return float(np.mean(labels[closest] == labels))


def _write_distribution_plots(out_dir: Path, real_rows: list[dict[str, str]], generated_rows: list[dict[str, Any]], columns: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [name for name in ("rq", "ra", "psd_low_power", "psd_mid_power", "psd_high_power", "psd_slope", "autocorrelation_length_px") if name in columns]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=150, squeeze=False)
    for axis, name in zip(axes.ravel(), selected):
        real = [_safe_float(row.get(name, "nan")) for row in real_rows]
        gen = [_safe_float(row.get(f"generated_{name}", "nan")) for row in generated_rows]
        real = [value for value in real if np.isfinite(value)]
        gen = [value for value in gen if np.isfinite(value)]
        axis.hist(real, bins=20, alpha=0.6, label="real")
        axis.hist(gen, bins=20, alpha=0.6, label="generated")
        axis.set_title(name, fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "descriptor_distribution_comparison.png")
    fig.savefig(out_dir / "psd_distribution_comparison.png")
    plt.close(fig)
    matrix_real = _descriptor_array(real_rows, selected)
    matrix_gen = _descriptor_array(generated_rows, selected, prefix="generated_")
    points = np.concatenate([np.nan_to_num(matrix_real, nan=0.0), np.nan_to_num(matrix_gen, nan=0.0)], axis=0)
    labels = np.asarray(["real"] * matrix_real.shape[0] + ["generated"] * matrix_gen.shape[0])
    if points.shape[0] >= 2:
        points = (points - points.mean(axis=0, keepdims=True)) / np.maximum(points.std(axis=0, keepdims=True), 1e-6)
        try:
            from sklearn.decomposition import PCA

            reduced = PCA(n_components=2, random_state=42).fit_transform(points)
        except Exception:
            reduced = points[:, :2] if points.shape[1] >= 2 else np.pad(points, ((0, 0), (0, 1)))
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        for label in ("real", "generated"):
            mask = labels == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1], s=12, alpha=0.7, label=label)
        ax.legend()
        ax.set_title("real vs generated descriptor PCA")
        fig.tight_layout()
        fig.savefig(out_dir / "real_vs_generated_umap_or_pca.png")
        plt.close(fig)


def _write_diagnostic_grids(out_dir: Path, samples_dir: Path, real_index: Path, generated_rows: list[dict[str, Any]]) -> None:
    samples_npz = samples_dir / "generated_samples_v2.npz"
    if not samples_npz.is_file():
        return
    payload = np.load(samples_npz, allow_pickle=True)
    images = np.asarray(payload["images"], dtype=np.float32)
    if images.shape[0] == 0:
        return
    real_rows = read_csv_rows(real_index)
    real_images: list[np.ndarray] = []
    real_titles: list[str] = []
    for row in real_rows[: min(40, len(real_rows))]:
        try:
            real_images.append(load_height_array(resolve_repo_path(Path(row["network_input_path"]))))
            real_titles.append(str(row.get("sample_id", row["row_id"])))
        except Exception:
            continue
    if not real_images:
        return
    real_flat = np.asarray([image.ravel() for image in real_images], dtype=np.float32)
    selected = images[: min(6, images.shape[0])]
    grid_rows: list[list[np.ndarray]] = []
    titles: list[str] = []
    for index, image in enumerate(selected):
        image_flat = image.ravel()[None]
        distances = np.mean((real_flat - image_flat) ** 2, axis=1)
        closest_index = int(np.argmin(distances))
        grid_rows.append([image, real_images[closest_index], np.abs(image - real_images[closest_index])])
        titles.append(f"generated {index} vs {real_titles[closest_index]}")
    write_panel_grid(out_dir / "nearest_real_generated_grid.png", grid_rows, ["generated", "closest real diagnostic", "absolute diff"], titles)
    stds = [_safe_float(row.get("generated_std", "nan")) for row in generated_rows]
    order = np.argsort(np.nan_to_num(np.asarray(stds), nan=0.0))[: min(6, len(stds))]
    failure_rows = [[images[int(i)]] for i in order if int(i) < images.shape[0]]
    write_panel_grid(out_dir / "failure_cases_grid.png", failure_rows, ["low-std generated"], [str(i) for i in order])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    samples_dir = resolve_repo_path(args.samples_dir)
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_index = resolve_repo_path(args.real_index)
    descriptors_path = resolve_repo_path(args.descriptors)
    generated_metric_path = samples_dir / "generation_metrics_v2.csv"
    generated_rows = read_csv_rows(generated_metric_path)
    if not generated_rows:
        raise RuntimeError(f"No generated metrics found at {generated_metric_path}")
    real_rows = read_csv_rows(descriptors_path)
    columns = [name for name in V2_DESCRIPTOR_NAMES if real_rows and name in real_rows[0]]
    distribution = _distribution_rows(real_rows, generated_rows, columns)
    consistency = _condition_consistency_rows(generated_rows, columns)
    write_csv_rows(out_dir / "descriptor_distribution_distance_v2.csv", distribution)
    write_csv_rows(out_dir / "condition_consistency_v2.csv", consistency)
    real_matrix = _descriptor_array(real_rows, columns)
    generated_matrix = _descriptor_array(generated_rows, columns, prefix="generated_")
    two_sample_acc = _two_sample_descriptor_accuracy(real_matrix, generated_matrix)
    generated_stds = np.asarray([_safe_float(row.get("generated_std", "nan")) for row in generated_rows], dtype=np.float32)
    generated_stds = generated_stds[np.isfinite(generated_stds)]
    modes = sorted({row.get("mode", "") for row in generated_rows})
    diversity: dict[str, Any] = {}
    for mode in modes:
        mode_rows = [row for row in generated_rows if row.get("mode", "") == mode]
        mat = _descriptor_array(mode_rows, columns, prefix="generated_")
        diversity[f"{mode}_descriptor_variance_mean"] = float(np.nanmean(np.nanvar(mat, axis=0))) if mat.size else 0.0
    if generated_matrix.shape[0] >= 2:
        centered = np.nan_to_num(generated_matrix, nan=0.0)
        distances = np.sqrt(np.sum((centered[:, None, :] - centered[None, :, :]) ** 2, axis=-1))
        mask = ~np.eye(distances.shape[0], dtype=bool)
        pairwise_mean = float(np.mean(distances[mask]))
        near_duplicate_rate = float(np.mean(distances[mask] < 1e-5))
    else:
        pairwise_mean = 0.0
        near_duplicate_rate = 0.0
    _write_distribution_plots(out_dir, real_rows, generated_rows, columns)
    _write_diagnostic_grids(out_dir, samples_dir, real_index, generated_rows)
    summary = {
        "generated_count": len(generated_rows),
        "generated_nonconstant_rate": float(np.mean(generated_stds > 1e-4)) if generated_stds.size else 0.0,
        "generated_std_mean": float(np.mean(generated_stds)) if generated_stds.size else 0.0,
        "generated_std_min": float(np.min(generated_stds)) if generated_stds.size else 0.0,
        "descriptor_distribution_rows": len(distribution),
        "mean_abs_descriptor_mean_delta": float(np.mean([_safe_float(row["abs_mean_delta"]) for row in distribution])) if distribution else 0.0,
        "two_sample_descriptor_accuracy": two_sample_acc,
        "pairwise_generated_descriptor_distance_mean": pairwise_mean,
        "near_duplicate_rate": near_duplicate_rate,
        **diversity,
        "generation_metrics": display_path(generated_metric_path),
        "descriptor_distribution_comparison": display_path(out_dir / "descriptor_distribution_comparison.png"),
        "psd_distribution_comparison": display_path(out_dir / "psd_distribution_comparison.png"),
        "real_vs_generated_umap_or_pca": display_path(out_dir / "real_vs_generated_umap_or_pca.png"),
        "nearest_real_generated_grid": display_path(out_dir / "nearest_real_generated_grid.png"),
        "failure_cases_grid": display_path(out_dir / "failure_cases_grid.png"),
        "note": "Closest-real visualization is diagnostic only; generated samples come from diffusion sampling.",
    }
    write_json(out_dir / "generation_summary_v2.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(f"Wrote AFM prior v2 evaluation to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
