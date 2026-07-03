"""Optional lightweight temporal SSL pretraining for RHEED clips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import display_path, read_csv_rows, resolve_repo_path, resolve_torch_device, set_seed, write_json
from rheed2morph.generative.models.rheed_mae import SmallCNNFrameEncoder, load_mae_encoder_state


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional temporal SSL pretraining for RHEED.")
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--frame-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class ClipDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, video_index: Path, limit: int | None = None) -> None:
        rows = [row for row in read_csv_rows(resolve_repo_path(video_index)) if row.get("cached_tensor_path", "")]
        if limit is not None:
            rows = rows[: int(limit)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = np.asarray(np.load(resolve_repo_path(Path(self.rows[index]["cached_tensor_path"])))["frames"], dtype=np.float32)
        means = frames[:, 0].mean(axis=(1, 2))
        label = int(means[-1] >= means[0]) if len(means) >= 2 else 0
        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)


class TemporalSSLModel(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.frame_encoder = SmallCNNFrameEncoder(embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(embedding_dim, 2)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = video.shape
        emb = self.frame_encoder(video.reshape(b * t, c, h, w)).reshape(b, t, -1)
        seq, hidden = self.gru(emb)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head(pooled), seq.mean(dim=1)


def _plot(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fig, axis = plt.subplots(figsize=(5, 3), dpi=150)
    axis.plot([row["epoch"] for row in history], [row["loss"] for row in history])
    axis.set_xlabel("epoch")
    axis.set_ylabel("temporal SSL loss")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def _embedding_plot(path: Path, model: TemporalSSLModel, loader: DataLoader, device: torch.device) -> None:
    values = []
    for batch_index, (video, _label) in enumerate(loader):
        _logits, emb = model(video.to(device))
        values.append(emb.detach().cpu().numpy())
        if batch_index >= 6:
            break
    if not values:
        return
    matrix = np.concatenate(values, axis=0)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] >= 2:
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
        coords = matrix @ vh[:2].T
    else:
        coords = np.zeros((matrix.shape[0], 2), dtype=np.float32)
    fig, axis = plt.subplots(figsize=(4, 4), dpi=150)
    axis.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.8)
    axis.set_title("Temporal SSL embedding PCA")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def pretrain_temporal(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.quick else int(args.epochs)
    limit = args.limit if not args.quick else (args.limit or 64)
    device = resolve_torch_device(args.device)
    dataset = ClipDataset(args.video_index, limit=limit)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    model = TemporalSSLModel().to(device)
    loaded = False
    if args.frame_encoder_checkpoint is not None and resolve_repo_path(args.frame_encoder_checkpoint).is_file():
        loaded = load_mae_encoder_state(model.frame_encoder, resolve_repo_path(args.frame_encoder_checkpoint).as_posix(), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        model.train()
        for video, label in loader:
            video = video.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits, _emb = model(video)
                loss = F.cross_entropy(logits, label)
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * int(video.shape[0])
            count += int(video.shape[0])
        row = {"epoch": float(epoch), "loss": total / max(count, 1)}
        history.append(row)
        if row["loss"] < best:
            best = row["loss"]
            (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_loss": best, "config": {"mae_encoder_loaded": loaded}}, out_dir / "checkpoints" / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "epoch": epochs, "best_loss": best, "config": {"mae_encoder_loaded": loaded}}, out_dir / "checkpoints" / "last.pt")
    _plot(out_dir / "ssl_loss_curves.png", history)
    _embedding_plot(out_dir / "temporal_embedding_umap_or_pca.png", model.eval(), loader, device)
    metrics = {"history": history, "best_loss": best, "mae_encoder_loaded": loaded, "objective": "time-bin prediction plus temporal sequence pooling"}
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = pretrain_temporal(args)
    print(f"Wrote temporal SSL outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"best_loss={metrics['best_loss']:.6f}")


if __name__ == "__main__":
    main()
