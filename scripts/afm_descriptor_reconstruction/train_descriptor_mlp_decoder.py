#!/usr/bin/env python3
"""Train a descriptor-to-AFM-image MLP baseline with PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None
    nn = None

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:  # pragma: no cover - optional environment detail.
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "mlp_decoder"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction" / "mlp_decoder"


if nn is not None:
    class MLPDecoder(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
else:  # pragma: no cover - only used when torch is unavailable.
    class MLPDecoder:  # type: ignore[no-redef]
        pass


def require_torch() -> None:
    if torch is None or nn is None:
        raise SystemExit(
            "PyTorch is required for the MLP decoder. Install it with a CUDA-enabled wheel, "
            "for example: python -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )


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
    if image.shape == (size, size):
        return image.astype(np.float32, copy=False)
    rows, cols = image.shape
    target_y = np.linspace(0, rows - 1, size)
    target_x = np.linspace(0, cols - 1, size)
    source_y = np.arange(rows)
    source_x = np.arange(cols)
    resized_rows = np.vstack([np.interp(target_x, source_x, row) for row in image])
    return np.vstack([np.interp(target_y, source_y, resized_rows[:, col]) for col in range(size)]).T.astype(
        np.float32
    )


def load_dataset(
    selected_rows: list[dict[str, str]],
    network_input_dir: Path,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], list[str], list[str]]:
    id_columns = {"row_id", "sample_id", "afm_path"}
    descriptor_columns = [column for column in selected_rows[0] if column not in id_columns]
    x_values: list[list[float]] = []
    y_values: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for row in selected_rows:
        path = network_input_path(row, network_input_dir)
        if not path.exists():
            warnings.append(f"missing network input for row {row.get('row_id', '')}: {display_path(path)}")
            continue
        image = np.asarray(np.load(path), dtype=np.float32)
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            warnings.append(f"invalid image for row {row.get('row_id', '')}: shape={image.shape}")
            continue
        try:
            descriptors = [float(row[column]) for column in descriptor_columns]
        except ValueError as exc:
            warnings.append(f"invalid descriptors for row {row.get('row_id', '')}: {exc}")
            continue
        x_values.append(descriptors)
        y_values.append(resize_bilinear(image, image_size).ravel())
        kept_rows.append(row)
    if not y_values:
        raise SystemExit("No valid descriptor/image pairs loaded.")
    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32),
        kept_rows,
        descriptor_columns,
        warnings,
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> tuple[str, torch.device]:
    require_torch()
    normalized = device_name.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested for MLP training, but PyTorch cannot see a GPU. "
            "Check the CUDA-enabled torch install and run outside restrictive sandboxes."
        )
    return normalized, torch.device(normalized)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def gradient_loss_torch(pred: torch.Tensor, target: torch.Tensor, image_size: int) -> torch.Tensor:
    pred_img = pred.view(-1, image_size, image_size)
    target_img = target.view(-1, image_size, image_size)
    dx_error = (pred_img[:, :, 1:] - pred_img[:, :, :-1]) - (
        target_img[:, :, 1:] - target_img[:, :, :-1]
    )
    dy_error = (pred_img[:, 1:, :] - pred_img[:, :-1, :]) - (
        target_img[:, 1:, :] - target_img[:, :-1, :]
    )
    return torch.mean(dx_error.square()) + torch.mean(dy_error.square())


def loss_terms(pred: torch.Tensor, target: torch.Tensor, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = torch.mean((pred - target).square())
    grad_loss = gradient_loss_torch(pred, target, image_size)
    total = mse + 0.1 * grad_loss
    return total, mse, grad_loss


def evaluate_model(
    model: nn.Module,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    image_size: int,
    batch_size: int,
) -> tuple[float, float, float]:
    totals: list[float] = []
    mses: list[float] = []
    grad_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            batch_x = x_tensor[start : start + batch_size]
            batch_y = y_tensor[start : start + batch_size]
            pred = model(batch_x)
            total, mse, grad_loss = loss_terms(pred.float(), batch_y, image_size)
            totals.append(float(total.item()))
            mses.append(float(mse.item()))
            grad_losses.append(float(grad_loss.item()))
    return float(np.mean(totals)), float(np.mean(mses)), float(np.mean(grad_losses))


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    image_size: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str = "auto",
    use_amp: bool = True,
    compile_model: bool = False,
) -> tuple[nn.Module, list[dict[str, float]]]:
    require_torch()
    resolved_device_name, device = resolve_device(device_name)
    set_random_seed(seed)

    x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    y_tensor = torch.from_numpy(y).to(device=device, dtype=torch.float32)
    model = MLPDecoder(x.shape[1], y.shape[1]).to(device)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(x_tensor.shape[0], device=device)
        for start in range(0, x_tensor.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_x = x_tensor.index_select(0, indices)
            batch_y = y_tensor.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                pred = model(batch_x)
                total, _, _ = loss_terms(pred.float(), batch_y, image_size)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

        total, mse, grad_loss = evaluate_model(model, x_tensor, y_tensor, image_size, batch_size)
        history.append({"epoch": float(epoch), "loss": total, "mse": mse, "gradient_loss": grad_loss})
        if total < best_loss - 1e-7:
            best_loss = total
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    setattr(model, "training_device", resolved_device_name)
    setattr(model, "amp_enabled", bool(use_amp and device.type == "cuda"))
    return model, history


def predict(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    require_torch()
    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            pred = model(x_tensor[start : start + batch_size]).float().cpu().numpy()
            outputs.append(pred)
    return np.vstack(outputs).astype(np.float32)


def roughness_triplet(image: np.ndarray) -> tuple[float, float, float]:
    centered = image - float(np.mean(image))
    return (
        float(np.mean(np.abs(centered))),
        float(np.sqrt(np.mean(centered**2))),
        float(np.max(image) - np.min(image)),
    )


def image_metrics(true_image: np.ndarray, pred_image: np.ndarray) -> dict[str, float]:
    error = true_image - pred_image
    true_flat = true_image.ravel()
    pred_flat = pred_image.ravel()
    pearson = 0.0
    if float(np.std(true_flat)) > 0 and float(np.std(pred_flat)) > 0:
        pearson = float(np.corrcoef(true_flat, pred_flat)[0, 1])
    ssim_value = np.nan
    if ssim is not None:
        ssim_value = float(ssim(true_image, pred_image, data_range=2.0))
    true_ra, true_rq, true_pv = roughness_triplet(true_image)
    pred_ra, pred_rq, pred_pv = roughness_triplet(pred_image)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "ssim": ssim_value,
        "pearson": pearson,
        "ra_abs_error": abs(pred_ra - true_ra),
        "rq_abs_error": abs(pred_rq - true_rq),
        "peak_to_valley_abs_error": abs(pred_pv - true_pv),
    }


def average_metrics(y_true: np.ndarray, y_pred: np.ndarray, image_size: int) -> dict[str, float]:
    true_images = y_true.reshape((-1, image_size, image_size))
    pred_images = y_pred.reshape((-1, image_size, image_size))
    metrics = [image_metrics(true, pred) for true, pred in zip(true_images, pred_images)]
    return {key: float(np.nanmean([item[key] for item in metrics])) for key in metrics[0]}


def kfold_predictions(
    x: np.ndarray,
    y: np.ndarray,
    image_size: int,
    folds: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str = "auto",
    use_amp: bool = True,
    compile_model: bool = False,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    fold_indices = np.array_split(indices, folds)
    predictions = np.zeros_like(y, dtype=np.float32)
    fold_histories: list[dict[str, float]] = []
    for fold_index, test_indices in enumerate(fold_indices, start=1):
        train_indices = np.setdiff1d(indices, test_indices, assume_unique=False)
        model, history = train_model(
            x[train_indices],
            y[train_indices],
            image_size,
            epochs,
            patience,
            batch_size,
            learning_rate,
            seed + fold_index,
            device_name=device_name,
            use_amp=use_amp,
            compile_model=compile_model,
        )
        predictions[test_indices] = predict(model, x[test_indices], batch_size)
        fold_histories.append(
            {
                "fold": float(fold_index),
                "epochs": float(len(history)),
                "final_loss": history[-1]["loss"],
            }
        )
    return predictions, fold_histories


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "model",
        "n_images",
        "mse",
        "mae",
        "ssim",
        "pearson",
        "ra_abs_error",
        "rq_abs_error",
        "peak_to_valley_abs_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.10g}" if isinstance(row.get(key), float) else row.get(key, "")
                    for key in fieldnames
                }
            )


def write_loss_plot(path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in history]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(epochs, [item["loss"] for item in history], label="total")
    ax.plot(epochs, [item["mse"] for item in history], label="mse")
    ax.plot(epochs, [item["gradient_loss"] for item in history], label="gradient")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_grid(path: Path, y_true: np.ndarray, y_pred: np.ndarray, rows: list[dict[str, str]], image_size: int) -> None:
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
    fig.suptitle("MLP descriptor-based morphology reconstruction baseline", fontsize=12)
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


def save_reconstructions(output_dir: Path, rows: list[dict[str, str]], predictions: np.ndarray, image_size: int) -> None:
    target_dir = output_dir / "reconstructions"
    target_dir.mkdir(parents=True, exist_ok=True)
    images = predictions.reshape((-1, image_size, image_size))
    for row, image in zip(rows, images):
        afm_path = Path(row["afm_path"])
        sample_id = row.get("sample_id") or afm_path.parent.parent.name
        base = afm_path.name.removesuffix("_plane_corrected.npy")
        np.save(target_dir / f"{sample_id}_{base}_mlp_reconstructed.npy", image.astype(np.float32))


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)


def write_report(path: Path, metrics_rows: list[dict[str, Any]], pca_metrics: dict[str, str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MLP Descriptor Decoder Baseline",
        "",
        "This PyTorch MLP decoder is an exploratory baseline. The PCA decoder is the more stable reference.",
        "",
        "The model maps selected AFM descriptors directly to normalized plane-corrected ZSensor AFM height maps. It should not be interpreted as exact reconstruction.",
        "",
        "## Metrics",
        "",
    ]
    for row in metrics_rows:
        lines.append(
            f"- {row['mode']}: MSE={row['mse']:.6g}, MAE={row['mae']:.6g}, "
            f"SSIM={row['ssim']:.6g}, Pearson={row['pearson']:.6g}"
        )
    if pca_metrics:
        lines.extend(
            [
                "",
                "## PCA Reference",
                "",
                f"- best PCA LOOCV row: k={pca_metrics.get('n_components')}, "
                f"MSE={pca_metrics.get('mse')}, MAE={pca_metrics.get('mae')}, "
                f"SSIM={pca_metrics.get('ssim')}, Pearson={pca_metrics.get('pearson')}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_pca_loocv_metrics() -> dict[str, str] | None:
    path = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "pca_decoder" / "metrics.csv"
    if not path.exists():
        return None
    rows = [row for row in read_csv(path) if row.get("mode") == "loocv"]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["mse"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small MLP descriptor-to-image baseline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected_descriptor_csv", type=Path, default=DEFAULT_SELECTED_DESCRIPTOR_CSV)
    parser.add_argument("--network_input_dir", type=Path, default=DEFAULT_NETWORK_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cv_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cv_folds", type=int, default=5)
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
    selected_csv, fallback_warning = resolve_selected_descriptor_csv(args.selected_descriptor_csv)
    network_input_dir = resolve_existing_path(args.network_input_dir)
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")
    if not selected_csv.exists():
        raise SystemExit(f"Selected descriptor CSV does not exist: {selected_csv}")
    if not network_input_dir.is_dir():
        raise SystemExit(f"Network input directory does not exist: {network_input_dir}")
    if args.image_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("image size, epochs, and batch size must be positive.")

    warnings = [fallback_warning] if fallback_warning else []
    selected_rows = read_csv(selected_csv)
    x, y, kept_rows, descriptor_columns, load_warnings = load_dataset(
        selected_rows, network_input_dir, args.image_size
    )
    warnings.extend(load_warnings)

    resolved_device_name, _ = resolve_device(args.device)
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
    metrics_rows: list[dict[str, Any]] = [
        {"mode": "insample", "model": "torch_mlp", "n_images": x.shape[0], **average_metrics(y, insample_pred, args.image_size)}
    ]

    cv_pred = None
    fold_histories: list[dict[str, float]] = []
    if args.cv_folds >= 2 and args.cv_folds <= x.shape[0]:
        cv_pred, fold_histories = kfold_predictions(
            x,
            y,
            args.image_size,
            args.cv_folds,
            args.cv_epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.seed + 1000,
            device_name=args.device,
            use_amp=args.amp,
            compile_model=args.compile,
        )
        metrics_rows.append(
            {
                "mode": f"{args.cv_folds}-fold_cv",
                "model": "torch_mlp",
                "n_images": x.shape[0],
                **average_metrics(y, cv_pred, args.image_size),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mlp_decoder_checkpoint.npz"
    metrics_path = output_dir / "metrics.csv"
    loss_path = report_dir / "training_loss.png"
    grid_path = report_dir / "reconstruction_grid.png"
    report_path = report_dir / "README.md"
    summary_path = output_dir / "model_summary.json"

    save_checkpoint(
        checkpoint_path,
        model,
        {
            "architecture": [x.shape[1], 128, 256, 512, y.shape[1]],
            "image_size": args.image_size,
            "descriptor_columns": descriptor_columns,
            "framing": "exploratory PyTorch MLP baseline; PCA decoder is the more stable reference",
            "device": resolved_device_name,
            "amp_enabled": bool(args.amp and resolved_device_name.startswith("cuda")),
            "torch_version": torch.__version__ if torch is not None else "",
        },
    )
    write_metrics_csv(metrics_path, metrics_rows)
    write_loss_plot(loss_path, history)
    write_grid(grid_path, y, cv_pred if cv_pred is not None else insample_pred, kept_rows, args.image_size)
    save_reconstructions(output_dir, kept_rows, cv_pred if cv_pred is not None else insample_pred, args.image_size)
    pca_reference = best_pca_loocv_metrics()
    write_report(report_path, metrics_rows, pca_reference)
    summary_path.write_text(
        json.dumps(
            {
                "n_images": int(x.shape[0]),
                "image_size": [args.image_size, args.image_size],
                "descriptor_count": len(descriptor_columns),
                "descriptor_columns": descriptor_columns,
                "epochs_ran": len(history),
                "device": resolved_device_name,
                "amp_enabled": bool(args.amp and resolved_device_name.startswith("cuda")),
                "torch_version": torch.__version__ if torch is not None else "",
                "cv_fold_summaries": fold_histories,
                "warnings": warnings,
                "outputs": {
                    "checkpoint": display_path(checkpoint_path),
                    "metrics": display_path(metrics_path),
                    "training_loss": display_path(loss_path),
                    "reconstruction_grid": display_path(grid_path),
                    "report": display_path(report_path),
                },
                "pca_reference": pca_reference,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("AFM descriptor MLP decoder summary")
    print(f"  images: {x.shape[0]}")
    print(f"  resized image shape: {args.image_size}x{args.image_size}")
    print(f"  descriptor count: {len(descriptor_columns)}")
    print(f"  training device: {resolved_device_name}")
    print(f"  mixed precision: {bool(args.amp and resolved_device_name.startswith('cuda'))}")
    print(f"  epochs run: {len(history)}")
    if warnings:
        print(f"  warnings: {len(warnings)}")
        for warning in warnings:
            print(f"    - {warning}")
    print("  MLP metrics:")
    for row in metrics_rows:
        print(
            f"    {row['mode']}: MSE={row['mse']:.6g}, MAE={row['mae']:.6g}, "
            f"SSIM={row['ssim']:.6g}, Pearson={row['pearson']:.6g}"
        )
    if pca_reference:
        print(
            "  PCA reference best LOOCV: "
            f"k={pca_reference['n_components']}, MSE={pca_reference['mse']}, "
            f"MAE={pca_reference['mae']}, SSIM={pca_reference['ssim']}, "
            f"Pearson={pca_reference['pearson']}"
        )
    print(f"  checkpoint: {display_path(checkpoint_path)}")
    print(f"  metrics: {display_path(metrics_path)}")
    print(f"  training loss: {display_path(loss_path)}")
    print(f"  reconstruction grid: {display_path(grid_path)}")
    print(f"  reconstructed arrays: {display_path(output_dir / 'reconstructions')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
