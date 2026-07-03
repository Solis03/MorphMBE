"""Pretrain a compact RHEED frame masked autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import display_path, read_csv_rows, resolve_repo_path, resolve_torch_device, set_seed, write_json
from rheed2morph.generative.models.rheed_mae import build_rheed_mae
from rheed2morph.generative.rheed_ssl_augmentations import RheedAugmentationConfig, RheedSafeAugment
from rheed2morph.generative.visualization import write_panel_grid


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain RHEED frame MAE.")
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="small_cnn_mae", choices=["small_cnn_mae", "vit_mae"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class RheedFrameDataset(Dataset[torch.Tensor]):
    def __init__(self, frame_index: Path, limit: int | None = None, augment: bool = True) -> None:
        rows = read_csv_rows(resolve_repo_path(frame_index))
        if limit is not None:
            rows = rows[: int(limit)]
        self.rows = rows
        self.augment = bool(augment)
        self.augmenter = RheedSafeAugment(RheedAugmentationConfig(patch_mask_ratio=0.0))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        frames = np.asarray(np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"], dtype=np.float32)
        frame = torch.from_numpy(frames[int(row["frame_index"])])
        return self.augmenter(frame) if self.augment else frame


def _write_curves(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fig, axis = plt.subplots(figsize=(5, 3), dpi=150)
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history])
    axis.set_xlabel("epoch")
    axis.set_ylabel("masked reconstruction loss")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def _write_recon_grid(path: Path, model: torch.nn.Module, loader: DataLoader, device: torch.device, mask_ratio: float) -> None:
    try:
        batch = next(iter(loader))[:8].to(device)
    except StopIteration:
        return
    out = model(batch, mask_ratio=mask_ratio)
    rows = []
    titles = []
    for index in range(batch.shape[0]):
        rows.append([batch[index, 0].detach().cpu().numpy(), out["masked"][index, 0].detach().cpu().numpy(), out["reconstruction"][index, 0].detach().cpu().numpy()])
        titles.append(f"frame {index}")
    write_panel_grid(path, rows, ["input", "masked", "reconstruction"], titles)


@torch.no_grad()
def _write_embedding_preview(path: Path, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> None:
    embeddings = []
    for batch_index, batch in enumerate(loader):
        out = model(batch.to(device), mask_ratio=0.0)
        embeddings.append(out["embedding"].detach().cpu().numpy())
        if batch_index >= 8:
            break
    if not embeddings:
        return
    matrix = np.concatenate(embeddings, axis=0)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] >= 2:
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
        coords = matrix @ vh[:2].T
    else:
        coords = np.zeros((matrix.shape[0], 2), dtype=np.float32)
    fig, axis = plt.subplots(figsize=(4, 4), dpi=150)
    axis.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.8)
    axis.set_title("RHEED MAE embedding PCA")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": int(epoch), "best_loss": float(best_loss), "config": config}, path)


def pretrain(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.quick else int(args.epochs)
    limit = args.limit if not args.quick else (args.limit or 128)
    device = resolve_torch_device(args.device)
    dataset = RheedFrameDataset(args.frame_index, limit=limit, augment=True)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    eval_loader = DataLoader(RheedFrameDataset(args.frame_index, limit=min(limit or len(dataset), 32), augment=False), batch_size=min(8, int(args.batch_size)), shuffle=False)
    model = build_rheed_mae(args.model, image_size=int(args.image_size), patch_size=int(args.patch_size), embed_dim=int(args.embed_dim), depth=int(args.depth), decoder_depth=int(args.decoder_depth)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if args.resume is not None and resolve_repo_path(args.resume).is_file():
        payload = torch.load(resolve_repo_path(args.resume), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    config = {
        "frame_index": display_path(resolve_repo_path(args.frame_index)),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "embed_dim": int(args.embed_dim),
        "mask_ratio": float(args.mask_ratio),
        "epochs": epochs,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "model": args.model,
        "device": str(device),
        "frame_count": len(dataset),
    }
    write_json(out_dir / "config.json", config)
    history: list[dict[str, float]] = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(batch, mask_ratio=float(args.mask_ratio))
                loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("RHEED MAE loss became non-finite.")
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.shape[0])
            count += int(batch.shape[0])
        train_loss = total / max(count, 1)
        history.append({"epoch": float(epoch), "train_loss": train_loss})
        if train_loss < best:
            best = train_loss
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best, config)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best, config)
    _write_recon_grid(out_dir / "mae_reconstruction_grid.png", model.eval(), eval_loader, device, float(args.mask_ratio))
    _write_curves(out_dir / "ssl_loss_curves.png", history)
    _write_embedding_preview(out_dir / "embedding_pca.png", model.eval(), eval_loader, device)
    metrics = {"history": history, "best_loss": best, "last_loss": history[-1]["train_loss"] if history else float("nan"), "reconstruction_grid": display_path(out_dir / "mae_reconstruction_grid.png"), "embedding_preview": display_path(out_dir / "embedding_pca.png")}
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = pretrain(args)
    print(f"Wrote RHEED frame MAE outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"best_loss={metrics['best_loss']:.6f}")


if __name__ == "__main__":
    main()
