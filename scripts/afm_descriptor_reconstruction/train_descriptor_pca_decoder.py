#!/usr/bin/env python3
"""Train a descriptor-based PCA morphology reconstruction baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:  # pragma: no cover - depends on optional local environment.
    ssim = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_1um_manifest.csv"
DEFAULT_SELECTED_DESCRIPTOR_CSV = (
    REPO_ROOT
    / "data"
    / "afm_descriptor_reconstruction"
    / "selected_descriptors"
    / "selected_descriptors.csv"
)
DEFAULT_NETWORK_INPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "network_inputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "pca_decoder"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction" / "pca_decoder"
DEFAULT_COMPONENTS = (8, 16, 24, 32)
DEFAULT_IMAGE_SIZE = 128


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required for GPU-enabled PCA training. "
            "Install it with: python -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )


def resolve_device(device_name: str) -> tuple[str, torch.device]:
    require_torch()
    normalized = device_name.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested for PCA training, but PyTorch cannot see a GPU."
        )
    return normalized, torch.device(normalized)


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_selected_descriptor_csv(path: Path) -> tuple[Path, str | None]:
    resolved = resolve_existing_path(path)
    if resolved.exists():
        return resolved, None
    fallback = resolved.with_name("selected_descriptor_table.csv")
    if fallback.exists():
        return fallback, f"requested descriptor CSV missing; used fallback {display_path(fallback)}"
    return resolved, None


def network_input_path(row: dict[str, str], network_input_dir: Path) -> Path:
    afm_path = Path(row["afm_path"])
    sample_id = row.get("sample_id") or afm_path.parent.parent.name
    base = afm_path.name.removesuffix("_plane_corrected.npy")
    return network_input_dir / f"{sample_id}_{base}_network_input.npy"


def resize_bilinear(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D image to size x size using separable linear interpolation."""
    if image.shape == (size, size):
        return image.astype(np.float32, copy=False)
    rows, cols = image.shape
    target_y = np.linspace(0, rows - 1, size)
    target_x = np.linspace(0, cols - 1, size)
    source_y = np.arange(rows)
    source_x = np.arange(cols)
    resized_rows = np.vstack([np.interp(target_x, source_x, row) for row in image])
    resized = np.vstack(
        [np.interp(target_y, source_y, resized_rows[:, col]) for col in range(size)]
    ).T
    return resized.astype(np.float32)


def load_dataset(
    selected_rows: list[dict[str, str]],
    network_input_dir: Path,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[str], list[str]]:
    id_columns = {"row_id", "sample_id", "afm_path"}
    descriptor_columns = [column for column in selected_rows[0] if column not in id_columns]
    descriptors: list[list[float]] = []
    images: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for row in selected_rows:
        path = network_input_path(row, network_input_dir)
        if not path.exists():
            warnings.append(f"missing network input for row {row.get('row_id', '')}: {display_path(path)}")
            continue
        try:
            image = np.asarray(np.load(path), dtype=float)
        except Exception as exc:  # noqa: BLE001 - keep batch run robust.
            warnings.append(f"failed loading row {row.get('row_id', '')}: {exc}")
            continue
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            warnings.append(f"invalid image for row {row.get('row_id', '')}: shape={image.shape}")
            continue
        try:
            feature_values = [float(row[column]) for column in descriptor_columns]
        except ValueError as exc:
            warnings.append(f"invalid descriptor row {row.get('row_id', '')}: {exc}")
            continue
        descriptors.append(feature_values)
        images.append(resize_bilinear(image, image_size))
        kept_rows.append(row)

    if not images:
        raise SystemExit("No valid image/descriptor pairs were loaded.")
    return (
        np.asarray(descriptors, dtype=np.float32),
        np.asarray(images, dtype=np.float32),
        kept_rows,
        descriptor_columns,
        warnings,
    )


def roughness_triplet(image: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(image))
    centered = image - mean
    ra = float(np.mean(np.abs(centered)))
    rq = float(np.sqrt(np.mean(centered**2)))
    peak_to_valley = float(np.max(image) - np.min(image))
    return ra, rq, peak_to_valley


def image_metrics(true_image: np.ndarray, pred_image: np.ndarray) -> dict[str, float]:
    error = true_image - pred_image
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    true_flat = true_image.ravel()
    pred_flat = pred_image.ravel()
    if float(np.std(true_flat)) > 0 and float(np.std(pred_flat)) > 0:
        pearson = float(np.corrcoef(true_flat, pred_flat)[0, 1])
    else:
        pearson = 0.0
    ssim_value = np.nan
    if ssim is not None:
        ssim_value = float(ssim(true_image, pred_image, data_range=2.0))

    true_ra, true_rq, true_pv = roughness_triplet(true_image)
    pred_ra, pred_rq, pred_pv = roughness_triplet(pred_image)
    return {
        "mse": mse,
        "mae": mae,
        "ssim": ssim_value,
        "pearson": pearson,
        "ra_abs_error": abs(pred_ra - true_ra),
        "rq_abs_error": abs(pred_rq - true_rq),
        "peak_to_valley_abs_error": abs(pred_pv - true_pv),
    }


def average_metrics(true_images: np.ndarray, pred_images: np.ndarray) -> dict[str, float]:
    per_image = [image_metrics(true, pred) for true, pred in zip(true_images, pred_images)]
    return {
        key: float(np.nanmean([metrics[key] for metrics in per_image]))
        for key in per_image[0]
    }


def ridge_fit_predict_torch(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    x_mean = train_x.mean(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    x_centered = train_x - x_mean
    y_centered = train_y - y_mean
    xtx = x_centered.T @ x_centered
    eye = torch.eye(xtx.shape[0], device=train_x.device, dtype=train_x.dtype)
    weights = torch.linalg.solve(xtx + alpha * eye, x_centered.T @ y_centered)
    intercept = y_mean - x_mean @ weights
    pred = test_x @ weights + intercept
    return pred, {
        "coef": weights.detach().cpu().numpy(),
        "intercept": intercept.detach().cpu().numpy().ravel(),
        "alpha": np.asarray(alpha, dtype=np.float32),
    }


def fit_pca_torch(
    flat_images: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, dict[str, np.ndarray], float]:
    mean = flat_images.mean(dim=0, keepdim=True)
    centered = flat_images - mean
    q = min(n_components, centered.shape[0], centered.shape[1])
    u, s, v = torch.pca_lowrank(centered, q=q, center=False)
    components = v[:, :n_components].T.contiguous()
    coeffs = centered @ components.T
    denom = max(centered.shape[0] - 1, 1)
    total_var = centered.var(dim=0, unbiased=True).sum()
    explained = (s[:n_components].square() / denom).sum()
    explained_ratio = float((explained / total_var).item()) if float(total_var.item()) > 0 else np.nan
    model = {
        "mean": mean.detach().cpu().numpy().ravel().astype(np.float32),
        "components": components.detach().cpu().numpy().astype(np.float32),
    }
    return coeffs, model, explained_ratio


def inverse_pca_torch(coeffs: torch.Tensor, pca_model: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    components = torch.from_numpy(pca_model["components"]).to(device=device, dtype=torch.float32)
    mean = torch.from_numpy(pca_model["mean"]).to(device=device, dtype=torch.float32)
    return coeffs @ components + mean


def fit_predict_insample(
    descriptors: np.ndarray,
    flat_images: np.ndarray,
    n_components: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    require_torch()
    x = torch.from_numpy(descriptors).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(flat_images).to(device=device, dtype=torch.float32)
    coeffs, pca_model, explained_ratio = fit_pca_torch(y, n_components)
    pred_coeffs, ridge_model = ridge_fit_predict_torch(x, coeffs, x)
    pred_flat = inverse_pca_torch(pred_coeffs, pca_model, device).detach().cpu().numpy().astype(np.float32)
    metrics = average_metrics(flat_images, pred_flat)
    metrics["pca_explained_variance_ratio_sum"] = explained_ratio
    return pred_flat, pca_model, ridge_model, metrics


def loocv_predictions(
    descriptors: np.ndarray,
    flat_images: np.ndarray,
    component_counts: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    require_torch()
    n_samples = descriptors.shape[0]
    predictions = {
        n_components: np.zeros_like(flat_images, dtype=np.float32)
        for n_components in component_counts
    }
    x_all = torch.from_numpy(descriptors).to(device=device, dtype=torch.float32)
    y_all = torch.from_numpy(flat_images).to(device=device, dtype=torch.float32)

    for test_index in range(n_samples):
        train_mask = torch.ones(n_samples, device=device, dtype=torch.bool)
        train_mask[test_index] = False
        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        test_x = x_all[test_index : test_index + 1]
        max_components = min(max(component_counts), y_train.shape[0], y_train.shape[1])
        coeffs, pca_model, _ = fit_pca_torch(y_train, max_components)
        for n_components in component_counts:
            limited_pca_model = {
                "mean": pca_model["mean"],
                "components": pca_model["components"][:n_components],
            }
            pred_coeffs, _ = ridge_fit_predict_torch(x_train, coeffs[:, :n_components], test_x)
            pred_flat = inverse_pca_torch(pred_coeffs, limited_pca_model, device)
            predictions[n_components][test_index] = pred_flat.detach().cpu().numpy().astype(np.float32)
    return predictions


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "model",
        "n_components",
        "n_images",
        "mse",
        "mae",
        "ssim",
        "pearson",
        "ra_abs_error",
        "rq_abs_error",
        "peak_to_valley_abs_error",
        "pca_explained_variance_ratio_sum",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: f"{row[field]:.10g}" if isinstance(row.get(field), float) else row.get(field, "")
                    for field in fieldnames
                }
            )


def save_reconstructions(
    output_dir: Path,
    rows: list[dict[str, str]],
    predictions: np.ndarray,
    mode: str,
) -> None:
    target_dir = output_dir / f"reconstructions_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row, image in zip(rows, predictions):
        afm_path = Path(row["afm_path"])
        sample_id = row.get("sample_id") or afm_path.parent.parent.name
        base = afm_path.name.removesuffix("_plane_corrected.npy")
        np.save(target_dir / f"{sample_id}_{base}_reconstructed.npy", image.astype(np.float32))


def write_grid(path: Path, true_images: np.ndarray, pred_images: np.ndarray, rows: list[dict[str, str]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(12, true_images.shape[0])
    indices = np.linspace(0, true_images.shape[0] - 1, count, dtype=int)
    fig, axes = plt.subplots(count, 3, figsize=(8.5, 2.2 * count), dpi=150)
    if count == 1:
        axes = np.asarray([axes])
    fig.suptitle(title, fontsize=12)
    for row_index, image_index in enumerate(indices):
        true_image = true_images[image_index]
        pred_image = pred_images[image_index]
        error = np.abs(true_image - pred_image)
        label = f"row {rows[image_index].get('row_id', '')} | sample {rows[image_index].get('sample_id', '')}"
        panels = (
            (true_image, "true", "gray", -1.0, 1.0),
            (pred_image, "reconstructed", "gray", -1.0, 1.0),
            (error, "absolute error", "magma", 0.0, float(np.percentile(error, 99))),
        )
        for col_index, (panel, panel_title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_index, col_index]
            ax.imshow(panel, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
            ax.set_axis_off()
            ax.set_title(f"{label}\n{panel_title}" if col_index == 0 else panel_title, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def valid_component_counts(requested: tuple[int, ...], n_samples: int, n_features: int) -> list[int]:
    max_allowed = min(n_samples - 1, n_features)
    return [value for value in requested if value < n_samples and value <= max_allowed]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a descriptor-based morphology reconstruction PCA baseline."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected_descriptor_csv", type=Path, default=DEFAULT_SELECTED_DESCRIPTOR_CSV)
    parser.add_argument("--network_input_dir", type=Path, default=DEFAULT_NETWORK_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:0.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_existing_path(args.manifest)
    selected_csv, fallback_warning = resolve_selected_descriptor_csv(args.selected_descriptor_csv)
    network_input_dir = resolve_existing_path(args.network_input_dir)
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    if not selected_csv.is_file():
        raise SystemExit(f"Selected descriptor CSV does not exist: {selected_csv}")
    if not network_input_dir.is_dir():
        raise SystemExit(f"Network input directory does not exist: {network_input_dir}")
    if args.image_size <= 0:
        raise SystemExit("--image_size must be positive.")

    resolved_device_name, device = resolve_device(args.device)
    warnings = [fallback_warning] if fallback_warning else []
    selected_rows = read_csv(selected_csv)
    descriptors, images, kept_rows, descriptor_columns, load_warnings = load_dataset(
        selected_rows, network_input_dir, args.image_size
    )
    warnings.extend(load_warnings)

    flat_images = images.reshape(images.shape[0], -1)
    component_counts = valid_component_counts(DEFAULT_COMPONENTS, images.shape[0], flat_images.shape[1])
    if not component_counts:
        raise SystemExit("No valid PCA component counts for this dataset.")

    metrics_rows: list[dict[str, Any]] = []
    insample_predictions: dict[int, np.ndarray] = {}
    insample_models: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for n_components in component_counts:
        pred_flat, pca_model, regression_model, metrics = fit_predict_insample(
            descriptors, flat_images, n_components, device
        )
        insample_predictions[n_components] = pred_flat.reshape(images.shape)
        insample_models[n_components] = (pca_model, regression_model)
        metrics_rows.append(
            {
                "mode": "insample",
                "model": "torch_ridge",
                "n_components": n_components,
                "n_images": images.shape[0],
                **metrics,
            }
        )

    loocv_by_components = loocv_predictions(descriptors, flat_images, component_counts, device)
    for n_components, pred_flat in loocv_by_components.items():
        metrics = average_metrics(flat_images, pred_flat)
        metrics_rows.append(
            {
                "mode": "loocv",
                "model": "torch_ridge",
                "n_components": n_components,
                "n_images": images.shape[0],
                **metrics,
                "pca_explained_variance_ratio_sum": np.nan,
            }
        )

    loocv_rows = [row for row in metrics_rows if row["mode"] == "loocv"]
    best_row = min(loocv_rows, key=lambda row: float(row["mse"]))
    best_components = int(best_row["n_components"])
    best_pca, best_regression = insample_models[best_components]
    best_insample = insample_predictions[best_components]
    best_loocv = loocv_by_components[best_components].reshape(images.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "best_model_summary.json"
    pca_path = output_dir / "best_pca_model.joblib"
    regression_path = output_dir / "best_ridge_regression_model.joblib"
    insample_grid_path = report_dir / "reconstruction_grid_insample.png"
    loocv_grid_path = report_dir / "reconstruction_grid_loocv.png"

    write_metrics_csv(metrics_path, metrics_rows)
    joblib.dump(best_pca, pca_path)
    joblib.dump(best_regression, regression_path)
    save_reconstructions(output_dir, kept_rows, best_insample, "insample")
    save_reconstructions(output_dir, kept_rows, best_loocv, "loocv")
    write_grid(
        insample_grid_path,
        images,
        best_insample,
        kept_rows,
        "Descriptor-based morphology reconstruction baseline: in-sample",
    )
    write_grid(
        loocv_grid_path,
        images,
        best_loocv,
        kept_rows,
        "Descriptor-based morphology reconstruction baseline: leave-one-out",
    )
    write_summary(
        summary_path,
        {
            "framing": (
                "Descriptor-based morphology reconstruction baseline; this is not an exact "
                "AFM reconstruction claim."
            ),
            "n_images": int(images.shape[0]),
            "image_size": [int(args.image_size), int(args.image_size)],
            "descriptor_csv": display_path(selected_csv),
            "descriptor_count": len(descriptor_columns),
            "descriptor_columns": descriptor_columns,
            "component_counts": component_counts,
            "device": resolved_device_name,
            "best_selection_metric": "lowest LOOCV MSE",
            "best_model": best_row,
            "outputs": {
                "metrics_csv": display_path(metrics_path),
                "pca_model": display_path(pca_path),
                "regression_model": display_path(regression_path),
                "insample_grid": display_path(insample_grid_path),
                "loocv_grid": display_path(loocv_grid_path),
            },
            "warnings": warnings,
        },
    )

    print("AFM descriptor PCA decoder summary")
    print(f"  images: {images.shape[0]}")
    print(f"  resized image shape: {args.image_size}x{args.image_size}")
    print(f"  descriptor count: {len(descriptor_columns)}")
    print(f"  training device: {resolved_device_name}")
    if warnings:
        print(f"  warnings: {len(warnings)}")
        for warning in warnings:
            print(f"    - {warning}")
    print("  PCA component results:")
    for row in metrics_rows:
        print(
            "    "
            f"{row['mode']} k={row['n_components']}: "
            f"MSE={row['mse']:.6g}, MAE={row['mae']:.6g}, "
            f"SSIM={row['ssim']:.6g}, Pearson={row['pearson']:.6g}"
        )
    print(
        "  best setting: "
        f"torch ridge with {best_components} PCA components by LOOCV MSE={best_row['mse']:.6g}"
    )
    print(f"  metrics: {display_path(metrics_path)}")
    print(f"  summary: {display_path(summary_path)}")
    print(f"  in-sample grid: {display_path(insample_grid_path)}")
    print(f"  LOOCV grid: {display_path(loocv_grid_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
