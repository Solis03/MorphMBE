#!/usr/bin/env python3
"""Train/evaluate the large-dataset AFM descriptor-to-image MLP baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_SCRIPT_DIR = REPO_ROOT / "scripts" / "afm_descriptor_reconstruction"
sys.path.insert(0, str(SMALL_SCRIPT_DIR))

from train_descriptor_mlp_decoder import (  # noqa: E402
    average_metrics,
    best_pca_loocv_metrics,
    display_path,
    image_metrics,
    load_dataset,
    predict,
    read_csv,
    resolve_existing_path,
    save_checkpoint,
    train_model,
    write_loss_plot,
    write_metrics_csv,
)


DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "large_afm_manifest.csv"
)
DEFAULT_SELECTED_DESCRIPTOR_CSV = (
    REPO_ROOT
    / "data"
    / "afm_descriptor_reconstruction_large"
    / "selected_descriptors"
    / "selected_descriptors.csv"
)
DEFAULT_NETWORK_INPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "network_inputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "mlp_decoder"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction_large" / "mlp_decoder"
SUMMARY_REPORT = REPO_ROOT / "reports" / "afm_descriptor_reconstruction_large" / "summary.md"
RUN_COMMANDS = REPO_ROOT / "reports" / "afm_descriptor_reconstruction_large" / "RUN_COMMANDS.md"


def manifest_by_row(path: Path) -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in read_csv(path)}


def split_random_folds(n: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [fold.astype(int) for fold in np.array_split(rng.permutation(n), folds)]


def split_group_folds(rows: list[dict[str, str]], folds: int) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["sample_id"], []).append(index)
    fold_lists = [[] for _ in range(folds)]
    for fold_index, sample_id in enumerate(sorted(groups)):
        fold_lists[fold_index % folds].extend(groups[sample_id])
    return [np.asarray(indices, dtype=int) for indices in fold_lists if indices]


def run_cv(
    x: np.ndarray,
    y: np.ndarray,
    image_size: int,
    folds: list[np.ndarray],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str = "auto",
    use_amp: bool = True,
    compile_model: bool = False,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    predictions = np.zeros_like(y, dtype=np.float32)
    summaries = []
    all_indices = np.arange(x.shape[0])
    for fold_number, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        model, history = train_model(
            x[train_idx],
            y[train_idx],
            image_size,
            epochs,
            patience,
            batch_size,
            learning_rate,
            seed + fold_number,
            device_name=device_name,
            use_amp=use_amp,
            compile_model=compile_model,
        )
        predictions[test_idx] = predict(model, x[test_idx], batch_size)
        summaries.append(
            {
                "fold": float(fold_number),
                "train_count": float(train_idx.size),
                "test_count": float(test_idx.size),
                "epochs": float(len(history)),
                "final_loss": history[-1]["loss"],
            }
        )
    return predictions, summaries


def mean_baseline_cv(y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    predictions = np.zeros_like(y, dtype=np.float32)
    all_indices = np.arange(y.shape[0])
    for test_idx in folds:
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        predictions[test_idx] = np.mean(y[train_idx], axis=0, keepdims=True)
    return predictions


def nearest_neighbor_cv(x: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(y, dtype=np.float32)
    distances = np.full(x.shape[0], np.nan, dtype=float)
    all_indices = np.arange(x.shape[0])
    for test_idx in folds:
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        for index in test_idx:
            diff = x[train_idx] - x[index]
            dist = np.sqrt(np.sum(diff**2, axis=1))
            nearest = int(np.argmin(dist))
            predictions[index] = y[train_idx[nearest]]
            distances[index] = float(dist[nearest])
    return predictions, distances


def train_test_prediction(
    x: np.ndarray,
    y: np.ndarray,
    image_size: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str = "auto",
    use_amp: bool = True,
    compile_model: bool = False,
) -> np.ndarray:
    model, _ = train_model(
        x[train_idx],
        y[train_idx],
        image_size,
        epochs,
        patience,
        batch_size,
        learning_rate,
        seed,
        device_name=device_name,
        use_amp=use_amp,
        compile_model=compile_model,
    )
    return predict(model, x[test_idx], batch_size)


def per_sample_metric_rows(
    rows: list[dict[str, str]],
    manifest: dict[str, dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_size: int,
    mode: str,
    nearest_distances: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    true_images = y_true.reshape((-1, image_size, image_size))
    pred_images = y_pred.reshape((-1, image_size, image_size))
    out = []
    for index, (row, true_image, pred_image) in enumerate(zip(rows, true_images, pred_images)):
        meta = manifest[row["row_id"]]
        metrics = image_metrics(true_image, pred_image)
        out.append(
            {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "scan_size_x_um": meta["scan_size_x_um"],
                "scan_size_y_um": meta["scan_size_y_um"],
                "area_um2": meta["area_um2"],
                "is_1um_scan": meta["is_1um_scan"],
                "evaluation_mode": mode,
                **metrics,
                "nearest_train_descriptor_distance": (
                    nearest_distances[index] if nearest_distances is not None else ""
                ),
            }
        )
    return out


def write_per_sample_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "row_id",
        "sample_id",
        "scan_size_x_um",
        "scan_size_y_um",
        "area_um2",
        "is_1um_scan",
        "evaluation_mode",
        "mse",
        "mae",
        "ssim",
        "pearson",
        "ra_abs_error",
        "rq_abs_error",
        "peak_to_valley_abs_error",
        "nearest_train_descriptor_distance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_reconstructions(path: Path, rows: list[dict[str, str]], pred: np.ndarray, image_size: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = pred.reshape((-1, image_size, image_size))
    for row, image in zip(rows, images):
        afm = Path(row["afm_path"])
        stem = f"{row['sample_id']}_{afm.name.removesuffix('_plane_corrected.npy')}"
        np.save(path / f"{stem}_reconstructed.npy", image.astype(np.float32))


def write_reconstruction_grid(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: list[dict[str, str]],
    image_size: int,
    title: str,
) -> None:
    """Write a reconstruction grid with an explicit evaluation-mode title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_images = y_true.reshape((-1, image_size, image_size))
    pred_images = y_pred.reshape((-1, image_size, image_size))
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
            (true_image, "true AFM", "gray", -1.0, 1.0),
            (pred_image, "reconstructed AFM", "gray", -1.0, 1.0),
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


def metrics_by_scan_size_plot(path: Path, per_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_rows = [row for row in per_rows if row["evaluation_mode"] == "group5fold_mlp"]
    areas = np.asarray([float(row["area_um2"]) for row in group_rows])
    mses = np.asarray([float(row["mse"]) for row in group_rows])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.scatter(areas, mses, s=20, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("scan area (um^2)")
    ax.set_ylabel("GroupKFold MLP MSE")
    ax.set_title("Large MLP error by scan area")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def comparison_plot(path: Path, metrics_rows: list[dict[str, Any]], previous: dict[str, str] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    values = []
    if previous:
        labels.append("1um MLP\n5-fold")
        values.append(float(previous["mse"]))
    for row in metrics_rows:
        if row["mode"] in {"insample_mlp", "random5fold_mlp", "group5fold_mlp", "mean_baseline_group5fold", "nearest_neighbor_group5fold"}:
            labels.append(row["mode"].replace("_", "\n"))
            values.append(float(row["mse"]))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("1um-only vs larger-dataset descriptor MLP")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def previous_1um_mlp_metrics() -> dict[str, str] | None:
    path = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "mlp_decoder" / "metrics.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    for row in rows:
        if row["mode"] == "5-fold_cv":
            return row
    return None


def write_summary_report(
    path: Path,
    manifest_summary: dict[str, Any],
    descriptor_count: int,
    metrics_rows: list[dict[str, Any]],
    previous_mlp: dict[str, str] | None,
) -> None:
    metric_lines = ["| Experiment | MSE | MAE | SSIM | Pearson |", "|---|---:|---:|---:|---:|"]
    if previous_mlp:
        metric_lines.append(
            f"| previous 1um-only MLP 5-fold | {previous_mlp['mse']} | {previous_mlp['mae']} | {previous_mlp['ssim']} | {previous_mlp['pearson']} |"
        )
    for row in metrics_rows:
        metric_lines.append(
            f"| {row['mode']} | {row['mse']:.6g} | {row['mae']:.6g} | {row['ssim']:.6g} | {row['pearson']:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Large AFM Descriptor Reconstruction Summary\n\n"
        "## Goal\n\n"
        "Test whether using more AFM data improves descriptor-to-image MLP reconstruction without changing model architecture.\n\n"
        "## Dataset\n\n"
        f"- Total valid AFM files: {manifest_summary['valid_files_included']}\n"
        f"- Unique sample_id count: {manifest_summary['unique_sample_id_count']}\n"
        f"- 1um scans: {manifest_summary['one_um_scan_count']}\n"
        f"- Non-1um scans: {manifest_summary['non_one_um_scan_count']}\n"
        f"- Scan size distribution: `{manifest_summary['scan_size_distribution']}`\n"
        f"- Original resolution distribution: `{manifest_summary['resolution_distribution']}`\n\n"
        "## Feature Changes\n\n"
        "Scan size, pixel size, area, log area, aspect ratio, and `is_1um_scan` were added as descriptors so resized 64x64 images remain physically conditioned.\n\n"
        "## Model\n\n"
        "The MLP architecture was intentionally kept unchanged: descriptor input -> 128 -> 256 -> 512 -> image pixels.\n\n"
        "## Metrics\n\n"
        + "\n".join(metric_lines)
        + "\n\n## Interpretation\n\n"
        "The in-sample result tests capacity and is expected to overfit. GroupKFold is more trustworthy than random row-level CV because scans from the same sample remain in the same fold. Mixing scan sizes gives more rows but also makes the resized image target less physically uniform; the scan-size-conditioned features help but do not make different scan areas directly equivalent. Errors by scan size should be read from `metrics_by_scan_size.png`; higher errors at larger areas would suggest the model is learning common texture patterns more than scan-size-transferable morphology.\n\n"
        "## Limitations\n\n"
        "- Different scan sizes are not directly equivalent after resizing.\n"
        "- Image normalization may remove absolute height scale.\n"
        "- Descriptor-to-image reconstruction is one-to-many.\n"
        "- A direct MLP pixel decoder may fail on rare morphology types.\n"
        "- GroupKFold is more trustworthy than random row-level CV.\n\n"
        "## Next Steps\n\n"
        "- Train separate models per scan-size group.\n"
        "- Use descriptor-to-PCA latent instead of direct pixels.\n"
        "- Use a convolutional decoder or autoencoder latent.\n"
        "- Condition the decoder on scan size.\n"
        "- Use morphology clustering before reconstruction.\n",
        encoding="utf-8",
    )


def write_run_commands(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Large AFM Descriptor Reconstruction Run Commands\n\n"
        "```bash\n"
        "uv run python scripts/afm_descriptor_reconstruction_large_dataset/build_large_afm_manifest.py --input_dir plane_corrected_afm\n"
        "uv run python scripts/afm_descriptor_reconstruction_large_dataset/extract_large_afm_descriptors.py --manifest data/afm_descriptor_reconstruction_large/large_afm_manifest.csv\n"
        "uv run python scripts/afm_descriptor_reconstruction_large_dataset/select_large_descriptors.py --descriptor_csv data/afm_descriptor_reconstruction_large/large_afm_descriptors.csv\n"
        "uv run python scripts/afm_descriptor_reconstruction_large_dataset/train_large_mlp_decoder.py --manifest data/afm_descriptor_reconstruction_large/large_afm_manifest.csv --selected_descriptor_csv data/afm_descriptor_reconstruction_large/selected_descriptors/selected_descriptors.csv --network_input_dir data/afm_descriptor_reconstruction_large/network_inputs --output_dir data/afm_descriptor_reconstruction_large/mlp_decoder --report_dir reports/afm_descriptor_reconstruction_large/mlp_decoder\n"
        "# report is generated by the training command\n"
        "```\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected_descriptor_csv", type=Path, default=DEFAULT_SELECTED_DESCRIPTOR_CSV)
    parser.add_argument("--network_input_dir", type=Path, default=DEFAULT_NETWORK_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cv_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision on CUDA. Use --no-amp to disable it.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the PyTorch model when supported.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_existing_path(args.manifest)
    selected_csv = resolve_existing_path(args.selected_descriptor_csv)
    network_input_dir = resolve_existing_path(args.network_input_dir)
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    manifest = manifest_by_row(manifest_path)
    manifest_summary_path = manifest_path.with_name("large_afm_manifest_summary.json")
    manifest_summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))

    rows = read_csv(selected_csv)
    x, y, kept_rows, descriptor_columns, warnings = load_dataset(rows, network_input_dir, args.image_size)
    model, history = train_model(
        x,
        y,
        args.image_size,
        args.epochs,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.seed,
        device_name=args.device,
        use_amp=args.amp,
        compile_model=args.compile,
    )
    insample_pred = predict(model, x, args.batch_size)

    random_folds = split_random_folds(x.shape[0], 5, args.seed)
    group_folds = split_group_folds(kept_rows, 5)
    random_pred, random_summary = run_cv(
        x,
        y,
        args.image_size,
        random_folds,
        args.cv_epochs,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.seed + 100,
        device_name=args.device,
        use_amp=args.amp,
        compile_model=args.compile,
    )
    group_pred, group_summary = run_cv(
        x,
        y,
        args.image_size,
        group_folds,
        args.cv_epochs,
        args.patience,
        args.batch_size,
        args.learning_rate,
        args.seed + 200,
        device_name=args.device,
        use_amp=args.amp,
        compile_model=args.compile,
    )
    mean_pred = mean_baseline_cv(y, group_folds)
    nn_pred, nn_dist = nearest_neighbor_cv(x, y, group_folds)

    metrics_rows: list[dict[str, Any]] = [
        {"mode": "insample_mlp", "model": "torch_mlp", "n_images": x.shape[0], **average_metrics(y, insample_pred, args.image_size)},
        {"mode": "random5fold_mlp", "model": "torch_mlp", "n_images": x.shape[0], **average_metrics(y, random_pred, args.image_size)},
        {"mode": "group5fold_mlp", "model": "torch_mlp", "n_images": x.shape[0], **average_metrics(y, group_pred, args.image_size)},
        {"mode": "mean_baseline_group5fold", "model": "mean_image", "n_images": x.shape[0], **average_metrics(y, mean_pred, args.image_size)},
        {"mode": "nearest_neighbor_group5fold", "model": "nearest_neighbor", "n_images": x.shape[0], **average_metrics(y, nn_pred, args.image_size)},
    ]

    one_um = np.asarray([manifest[row["row_id"]]["is_1um_scan"].lower() == "true" for row in kept_rows])
    if np.count_nonzero(one_um) >= 10 and np.count_nonzero(~one_um) >= 10:
        train_1um = np.flatnonzero(one_um)
        test_non = np.flatnonzero(~one_um)
        pred_non = train_test_prediction(
            x,
            y,
            args.image_size,
            train_1um,
            test_non,
            args.cv_epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.seed + 300,
            device_name=args.device,
            use_amp=args.amp,
            compile_model=args.compile,
        )
        metrics_rows.append({"mode": "train_1um_test_non1um", "model": "torch_mlp", "n_images": int(test_non.size), **average_metrics(y[test_non], pred_non, args.image_size)})
        train_non = np.flatnonzero(~one_um)
        test_1um = np.flatnonzero(one_um)
        pred_1um = train_test_prediction(
            x,
            y,
            args.image_size,
            train_non,
            test_1um,
            args.cv_epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.seed + 400,
            device_name=args.device,
            use_amp=args.amp,
            compile_model=args.compile,
        )
        metrics_rows.append({"mode": "train_non1um_test_1um", "model": "torch_mlp", "n_images": int(test_1um.size), **average_metrics(y[test_1um], pred_1um, args.image_size)})

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    per_sample_path = output_dir / "per_sample_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    per_rows = []
    per_rows.extend(per_sample_metric_rows(kept_rows, manifest, y, insample_pred, args.image_size, "insample_mlp"))
    per_rows.extend(per_sample_metric_rows(kept_rows, manifest, y, random_pred, args.image_size, "random5fold_mlp"))
    per_rows.extend(per_sample_metric_rows(kept_rows, manifest, y, group_pred, args.image_size, "group5fold_mlp"))
    per_rows.extend(per_sample_metric_rows(kept_rows, manifest, y, mean_pred, args.image_size, "mean_baseline_group5fold"))
    per_rows.extend(per_sample_metric_rows(kept_rows, manifest, y, nn_pred, args.image_size, "nearest_neighbor_group5fold", nn_dist))
    write_per_sample_csv(per_sample_path, per_rows)

    save_checkpoint(
        output_dir / "large_mlp_decoder_checkpoint.npz",
        model,
        {
            "architecture": [x.shape[1], 128, 256, 512, y.shape[1]],
            "descriptor_columns": descriptor_columns,
            "device": args.device,
            "amp_enabled": bool(args.amp),
        },
    )
    save_reconstructions(output_dir / "reconstructions_group5fold", kept_rows, group_pred, args.image_size)
    write_loss_plot(report_dir / "training_loss.png", history)
    write_loss_plot(report_dir / "train_vs_val_loss.png", history)
    write_reconstruction_grid(
        report_dir / "reconstruction_grid_insample.png",
        y,
        insample_pred,
        kept_rows,
        args.image_size,
        "Large-dataset MLP reconstruction: in-sample sanity check",
    )
    write_reconstruction_grid(
        report_dir / "reconstruction_grid_random5fold.png",
        y,
        random_pred,
        kept_rows,
        args.image_size,
        "Large-dataset MLP reconstruction: random 5-fold CV",
    )
    write_reconstruction_grid(
        report_dir / "reconstruction_grid_groupkfold.png",
        y,
        group_pred,
        kept_rows,
        args.image_size,
        "Large-dataset MLP reconstruction: GroupKFold by sample_id",
    )
    write_reconstruction_grid(
        report_dir / "mean_baseline_grid.png",
        y,
        mean_pred,
        kept_rows,
        args.image_size,
        "Large-dataset baseline: mean training image prediction",
    )
    write_reconstruction_grid(
        report_dir / "nearest_neighbor_grid.png",
        y,
        nn_pred,
        kept_rows,
        args.image_size,
        "Large-dataset baseline: nearest descriptor neighbor",
    )
    metrics_by_scan_size_plot(report_dir / "metrics_by_scan_size.png", per_rows)
    previous = previous_1um_mlp_metrics()
    comparison_plot(report_dir / "metrics_1um_vs_all_dataset_comparison.png", metrics_rows, previous)
    write_summary_report(SUMMARY_REPORT, manifest_summary, len(descriptor_columns), metrics_rows, previous)
    write_run_commands(RUN_COMMANDS)
    (output_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "n_images": int(x.shape[0]),
                "descriptor_count": len(descriptor_columns),
                "epochs_ran": len(history),
                "device": getattr(model, "training_device", args.device),
                "amp_enabled": bool(getattr(model, "amp_enabled", False)),
                "random_fold_summaries": random_summary,
                "group_fold_summaries": group_summary,
                "warnings": warnings,
                "metrics": metrics_rows,
                "pca_reference": best_pca_loocv_metrics(),
                "previous_1um_mlp": previous,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Large AFM MLP decoder summary")
    print(f"  images: {x.shape[0]}")
    print(f"  descriptor count: {len(descriptor_columns)}")
    print(f"  training device: {getattr(model, 'training_device', args.device)}")
    print(f"  mixed precision: {bool(getattr(model, 'amp_enabled', False))}")
    print(f"  1um scans: {manifest_summary['one_um_scan_count']}")
    print(f"  non-1um scans: {manifest_summary['non_one_um_scan_count']}")
    print("  metrics:")
    for row in metrics_rows:
        print(f"    {row['mode']}: MSE={row['mse']:.6g}, MAE={row['mae']:.6g}, SSIM={row['ssim']:.6g}, Pearson={row['pearson']:.6g}")
    print(f"  metrics CSV: {display_path(metrics_path)}")
    print(f"  per-sample metrics: {display_path(per_sample_path)}")
    print(f"  figures: {display_path(report_dir)}")
    print(f"  summary report: {display_path(SUMMARY_REPORT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
