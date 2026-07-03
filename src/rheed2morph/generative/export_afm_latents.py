"""Export standardized AFM spatial latents and condition tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_descriptors import DESCRIPTOR_NAMES
from rheed2morph.generative.common import (
    AFMImageDataset,
    display_path,
    load_data_index,
    read_csv_rows,
    resolve_repo_path,
    resolve_torch_device,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint


META_COLUMNS = {"row_id", "sample_id", "group_id", "split", "source_kind"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export AFM autoencoder spatial latents.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _encode_records(model: torch.nn.Module, records: list[Any], device: torch.device, batch_size: int) -> tuple[np.ndarray, list[dict[str, str]]]:
    if not records:
        return np.zeros((0, 1, 1, 1), dtype=np.float32), []
    loader = torch.utils.data.DataLoader(AFMImageDataset(records), batch_size=batch_size, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    metas: list[dict[str, str]] = []
    with torch.no_grad():
        for images, metadata in loader:
            latent = model.encode(images.to(device)).detach().cpu().numpy().astype(np.float32)
            latents.append(latent)
            for index in range(latent.shape[0]):
                metas.append({key: str(value[index]) for key, value in metadata.items()})
    return np.concatenate(latents, axis=0), metas


def _descriptor_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    names = [name for name in DESCRIPTOR_NAMES if name in rows[0]]
    if names:
        return names
    columns: list[str] = []
    for key in rows[0]:
        if key in META_COLUMNS:
            continue
        try:
            for row in rows:
                float(row.get(key, "nan"))
        except ValueError:
            continue
        columns.append(key)
    return columns


def _prototype_by_row(descriptor_path: Path) -> dict[str, str]:
    path = descriptor_path.parent / "prototype_labels.csv"
    if not path.is_file():
        return {}
    return {row["row_id"]: row.get("prototype_id", "") for row in read_csv_rows(path)}


def export_latents(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_index = resolve_repo_path(args.data_index)
    descriptor_path = resolve_repo_path(args.descriptors)
    device = resolve_torch_device(args.device)
    model, checkpoint = load_autoencoder_checkpoint(args.checkpoint, args.device)
    model.to(device).eval()
    split_payloads: dict[str, tuple[np.ndarray, list[dict[str, str]]]] = {}
    for split in ("train", "val", "test"):
        records = load_data_index(data_index, split=split, limit=args.limit)
        split_payloads[split] = _encode_records(model, records, device, int(args.batch_size))
    train_latents = split_payloads["train"][0]
    if train_latents.shape[0] == 0:
        raise RuntimeError("Cannot standardize latents because the train split is empty.")
    mean = train_latents.mean(axis=0, keepdims=True)
    std = train_latents.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    mean = mean.astype(np.float32)
    for split, (latents, metas) in split_payloads.items():
        standardized = ((latents - mean) / std).astype(np.float32) if latents.size else latents.astype(np.float32)
        np.savez_compressed(
            out_dir / f"latents_{split}.npz",
            latents=standardized,
            latents_raw=latents.astype(np.float32),
            row_ids=np.asarray([meta["row_id"] for meta in metas]),
            sample_ids=np.asarray([meta["sample_id"] for meta in metas]),
            group_ids=np.asarray([meta["group_id"] for meta in metas]),
            splits=np.asarray([split] * len(metas)),
        )
    descriptor_rows = read_csv_rows(descriptor_path)
    descriptor_by_row = {row["row_id"]: row for row in descriptor_rows}
    descriptor_cols = _descriptor_columns(descriptor_rows)
    train_descriptor_rows = [row for row in descriptor_rows if row.get("split") == "train"]
    if not train_descriptor_rows:
        train_descriptor_rows = descriptor_rows
    desc_train = np.asarray([[float(row[col]) for col in descriptor_cols] for row in train_descriptor_rows], dtype=np.float32)
    desc_mean = desc_train.mean(axis=0)
    desc_std = desc_train.std(axis=0)
    desc_std = np.where(desc_std > 1e-8, desc_std, 1.0).astype(np.float32)
    prototype_by_row = _prototype_by_row(descriptor_path)
    data_rows = read_csv_rows(data_index)
    data_by_row = {row["row_id"]: row for row in data_rows}
    condition_rows: list[dict[str, Any]] = []
    for split, (_latents, metas) in split_payloads.items():
        for meta in metas:
            row_id = meta["row_id"]
            descriptor = descriptor_by_row[row_id]
            data_row = data_by_row.get(row_id, {})
            out_row: dict[str, Any] = {
                "row_id": row_id,
                "sample_id": meta["sample_id"],
                "group_id": meta["group_id"],
                "split": split,
                "network_input_path": data_row.get("network_input_path", ""),
                "descriptor_height_path": data_row.get("descriptor_height_path", ""),
                "prototype_id": prototype_by_row.get(row_id, ""),
            }
            for index, col in enumerate(descriptor_cols):
                value = float(descriptor[col])
                out_row[col] = f"{value:.10g}"
                out_row[f"cond_{col}"] = f"{((value - float(desc_mean[index])) / float(desc_std[index])):.10g}"
            condition_rows.append(out_row)
    write_csv_rows(out_dir / "condition_table.csv", condition_rows)
    stats = {
        "autoencoder_checkpoint": display_path(resolve_repo_path(args.checkpoint)),
        "autoencoder_config": checkpoint.get("config", {}),
        "latent_shape": list(train_latents.shape[1:]),
        "latent_train_count": int(train_latents.shape[0]),
        "latent_mean_shape": list(mean.shape[1:]),
        "latent_mean_scalar": float(np.mean(mean)),
        "latent_std_scalar": float(np.mean(std)),
        "latent_mean_path_key": "latent_mean",
        "latent_std_path_key": "latent_std",
        "descriptor_columns": descriptor_cols,
        "descriptor_train_mean": {col: float(desc_mean[i]) for i, col in enumerate(descriptor_cols)},
        "descriptor_train_std": {col: float(desc_std[i]) for i, col in enumerate(descriptor_cols)},
        "split_counts": {split: int(split_payloads[split][0].shape[0]) for split in ("train", "val", "test")},
    }
    np.savez_compressed(out_dir / "latent_standardization.npz", latent_mean=mean, latent_std=std)
    write_json(out_dir / "latent_stats.json", stats)
    return stats


def main() -> None:
    args = build_parser().parse_args()
    stats = export_latents(args)
    print(f"Wrote AFM latents to {display_path(resolve_repo_path(args.out))}")
    print(f"latent_shape={stats['latent_shape']} train_count={stats['latent_train_count']}")


if __name__ == "__main__":
    main()
