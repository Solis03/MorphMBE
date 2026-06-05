#!/usr/bin/env python3
"""Shared AFM MVP utilities for autoencoder and latent experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AFMExample:
    row_id: str
    sample_id: str
    group_id: str
    material: str
    rheed_path: Path
    afm_path: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_manifest_examples(manifest_path: Path) -> list[AFMExample]:
    rows = read_csv(manifest_path)
    examples: list[AFMExample] = []
    for index, row in enumerate(rows, start=1):
        examples.append(
            AFMExample(
                row_id=row.get("row_id", str(index)),
                sample_id=row["sample_id"].strip(),
                group_id=(row.get("group_id", "").strip() or row["sample_id"].strip()),
                material=row.get("material", "").strip() or "unknown",
                rheed_path=resolve_existing_path(Path(row["rheed_path"])),
                afm_path=resolve_existing_path(Path(row["afm_path"])),
            )
        )
    return examples


def load_afm_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D AFM array, got shape {array.shape} for {path}")
        return np.asarray(array, dtype=np.float32)
    if suffix in {".png", ".tif", ".tiff"}:
        with Image.open(path) as image:
            grayscale = image.convert("F")
            return np.asarray(grayscale, dtype=np.float32)
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D AFM text array, got shape {array.shape} for {path}")
        return np.asarray(array, dtype=np.float32)
    raise ValueError(f"Unsupported AFM file type: {path}")


def preprocess_afm_array(
    array: np.ndarray,
    image_size: int,
    normalize_mode: str = "per_image_zscore",
) -> tuple[np.ndarray, dict[str, float]]:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None, :, :]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
    output = resized.numpy().astype(np.float32)
    stats = {
        "mean": float(np.mean(output)),
        "std": float(np.std(output)),
        "min": float(np.min(output)),
        "max": float(np.max(output)),
    }
    if normalize_mode == "per_image_zscore":
        std = stats["std"] if stats["std"] > 1e-6 else 1.0
        normalized = ((output - stats["mean"]) / std).astype(np.float32)
    elif normalize_mode == "per_image_minmax":
        scale = stats["max"] - stats["min"]
        if scale <= 1e-6:
            normalized = np.zeros_like(output, dtype=np.float32)
        else:
            normalized = ((output - stats["min"]) / scale * 2.0 - 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    return normalized, stats


class ConvAutoencoder(torch.nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 8
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.encoder_fc = torch.nn.Linear(64 * self.feature_size * self.feature_size, latent_dim)
        self.decoder_fc = torch.nn.Linear(latent_dim, 64 * self.feature_size * self.feature_size)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.encoder_fc(features.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_fc(z).view(z.shape[0], 64, self.feature_size, self.feature_size)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class ResidualConvAutoencoder(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 64) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(128),
        )
        self.encoder_fc = nn.Linear(128 * self.feature_size * self.feature_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * self.feature_size * self.feature_size)
        self.decoder = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.encoder_fc(features.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_fc(z).view(z.shape[0], 128, self.feature_size, self.feature_size)
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


def build_autoencoder(model_type: str, image_size: int, latent_dim: int) -> nn.Module:
    if model_type == "baseline":
        return ConvAutoencoder(image_size=image_size, latent_dim=latent_dim)
    if model_type == "residual":
        return ResidualConvAutoencoder(image_size=image_size, latent_dim=latent_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")


def image_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    return grad_x, grad_y


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    pixel_loss: str = "smooth_l1",
    edge_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pixel_loss == "mse":
        pixel = torch.mean((recon - target) ** 2)
    elif pixel_loss == "smooth_l1":
        pixel = F.smooth_l1_loss(recon, target)
    else:
        raise ValueError(f"Unsupported pixel_loss: {pixel_loss}")
    grad_x_recon, grad_y_recon = image_gradients(recon)
    grad_x_target, grad_y_target = image_gradients(target)
    edge = F.l1_loss(grad_x_recon, grad_x_target) + F.l1_loss(grad_y_recon, grad_y_target)
    total = pixel + edge_weight * edge
    return total, {
        "pixel_loss": float(pixel.detach().cpu()),
        "edge_loss": float(edge.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def write_loss_plot(path: Path, history: Sequence[dict[str, float]]) -> None:
    figure, axis = plt.subplots(figsize=(6, 4), dpi=150)
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="val")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.set_title("AFM Autoencoder Loss")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_reconstruction_grid(
    path: Path,
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    titles: Sequence[str],
    error_maps: np.ndarray | None = None,
) -> None:
    count = min(inputs.shape[0], 8)
    column_count = 3 if error_maps is not None else 2
    figure, axes = plt.subplots(count, column_count, figsize=(4 * column_count, 2.8 * count), dpi=150)
    axes = np.atleast_2d(axes)
    for row_index in range(count):
        panels: list[tuple[str, np.ndarray, str]] = [
            ("Original", inputs[row_index], "viridis"),
            ("Reconstruction", reconstructions[row_index], "viridis"),
        ]
        if error_maps is not None:
            panels.append(("Absolute error", error_maps[row_index], "magma"))
        for column_index, (label, image, cmap_name) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap_name)
            axis.axis("off")
            axis.set_title(f"{label}\n{titles[row_index]}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_latent_pca_plot(
    path: Path,
    latents: np.ndarray,
    labels: Sequence[str],
    splits: Sequence[str],
) -> None:
    if latents.ndim != 2 or latents.shape[0] < 2:
        return
    pca = PCA(n_components=2)
    coords = pca.fit_transform(latents)
    figure, axis = plt.subplots(figsize=(6, 5), dpi=150)
    for split_name, color in (("train", "tab:blue"), ("val", "tab:orange")):
        mask = np.asarray([split == split_name for split in splits], dtype=bool)
        if not np.any(mask):
            continue
        axis.scatter(coords[mask, 0], coords[mask, 1], s=40, alpha=0.8, label=split_name, color=color)
    for label, x_coord, y_coord in zip(labels, coords[:, 0], coords[:, 1]):
        axis.text(x_coord, y_coord, label, fontsize=6, alpha=0.8)
    axis.set_title("AFM Latent PCA")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    image_size: int,
    latent_dim: int,
    normalize_mode: str,
    model_type: str = "baseline",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "image_size": image_size,
            "latent_dim": latent_dim,
            "normalize_mode": normalize_mode,
            "model_type": model_type,
        },
        path,
    )


def load_autoencoder_checkpoint(path: Path, device_name: str = "auto") -> tuple[torch.nn.Module, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(path, map_location=device)
    model = build_autoencoder(
        model_type=str(payload.get("model_type", "baseline")),
        image_size=int(payload["image_size"]),
        latent_dim=int(payload["latent_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
