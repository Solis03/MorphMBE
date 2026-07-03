"""Extract compact RHEED SSL frame embeddings for diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.common import display_path, read_csv_rows, resolve_repo_path, resolve_torch_device, write_csv_rows, write_json
from rheed2morph.generative.models.rheed_mae import build_rheed_mae
from rheed2morph.generative.pretrain_rheed_frame_mae import RheedFrameDataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract RHEED SSL frame embeddings.")
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


@torch.no_grad()
def extract(args: argparse.Namespace) -> dict[str, object]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    payload = torch.load(resolve_repo_path(args.checkpoint), map_location=device)
    config = dict(payload.get("config", {}))
    model = build_rheed_mae(config.get("model", "small_cnn_mae"), image_size=int(config.get("image_size", 224)), patch_size=int(config.get("patch_size", 16)), embed_dim=int(config.get("embed_dim", 256))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    dataset = RheedFrameDataset(args.frame_index, limit=args.limit, augment=False)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
    source_rows = read_csv_rows(resolve_repo_path(args.frame_index))[: len(dataset)]
    embeddings = []
    for batch in loader:
        out = model(batch.to(device), mask_ratio=0.0)
        embeddings.append(out["embedding"].detach().cpu().numpy())
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, int(config.get("embed_dim", 256))), dtype=np.float32)
    rows = []
    for index, emb in enumerate(matrix):
        row = {key: source_rows[index].get(key, "") for key in ("frame_id", "video_id", "sample_id", "split", "frame_index")}
        row.update({f"emb_{dim:03d}": float(value) for dim, value in enumerate(emb)})
        rows.append(row)
    write_csv_rows(out_dir / "rheed_ssl_frame_embeddings.csv", rows)
    coords = np.zeros((matrix.shape[0], 2), dtype=np.float32)
    if matrix.shape[0] >= 2:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:2].T
    fig, axis = plt.subplots(figsize=(4, 4), dpi=150)
    if coords.size:
        axis.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.8)
    axis.set_title("Frame embedding PCA")
    fig.tight_layout()
    fig.savefig(out_dir / "frame_embedding_pca.png")
    plt.close(fig)
    summary = {"row_count": len(rows), "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0, "embeddings": display_path(out_dir / "rheed_ssl_frame_embeddings.csv")}
    write_json(out_dir / "embedding_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = extract(args)
    print(f"Wrote {summary['row_count']} RHEED embeddings to {display_path(resolve_repo_path(args.out))}")


if __name__ == "__main__":
    main()
