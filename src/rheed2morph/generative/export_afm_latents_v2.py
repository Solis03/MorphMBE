"""Export AFM autoencoder v2 latents and v2 condition schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.afm_prior_v2_utils import (
    AFMPriorV2Dataset,
    V2_DESCRIPTOR_NAMES,
    build_condition_matrix_v2,
    load_v2_index,
    write_npz_latents,
)
from rheed2morph.generative.common import (
    display_path,
    read_csv_rows,
    read_json,
    resolve_repo_path,
    resolve_torch_device,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export AFM prior v2 spatial latents.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _encode_records(model: torch.nn.Module, records: list[Any], device: torch.device, batch_size: int) -> tuple[np.ndarray, list[dict[str, str]]]:
    if not records:
        return np.zeros((0, 1, 1, 1), dtype=np.float32), []
    loader = DataLoader(AFMPriorV2Dataset(records), batch_size=batch_size, shuffle=False, num_workers=0)
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
        return V2_DESCRIPTOR_NAMES
    return [name for name in V2_DESCRIPTOR_NAMES if name in rows[0]]


def _prototype_map(path: Path) -> tuple[dict[str, str], int]:
    if not path.is_file():
        return {}, 0
    rows = read_csv_rows(path)
    mapping = {row["row_id"]: row.get("prototype_id", "") for row in rows}
    values = [int(float(value)) for value in mapping.values() if value != ""]
    return mapping, (max(values) + 1 if values else 0)


def export_latents(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_index = resolve_repo_path(args.data_index)
    descriptor_path = resolve_repo_path(args.descriptors)
    prototype_path = resolve_repo_path(args.prototypes)
    device = resolve_torch_device(args.device)
    model, checkpoint = load_autoencoder_v2_checkpoint(args.checkpoint, args.device)
    model.to(device).eval()
    split_payloads: dict[str, tuple[np.ndarray, list[dict[str, str]]]] = {}
    for split in ("train", "val", "test"):
        records = load_v2_index(data_index, split=split, limit=args.limit)
        split_payloads[split] = _encode_records(model, records, device, int(args.batch_size))
    train_latents = split_payloads["train"][0]
    if train_latents.shape[0] == 0:
        raise RuntimeError("Cannot export AFM prior v2 latents because the train split is empty.")
    latent_mean = train_latents.mean(axis=0, keepdims=True).astype(np.float32)
    latent_std = train_latents.std(axis=0, keepdims=True).astype(np.float32)
    latent_std = np.where(latent_std > 1e-6, latent_std, 1.0).astype(np.float32)
    for split, (latents_raw, metas) in split_payloads.items():
        standardized = ((latents_raw - latent_mean) / latent_std).astype(np.float32) if latents_raw.size else latents_raw.astype(np.float32)
        write_npz_latents(out_dir / f"latents_{split}.npz", standardized, latents_raw, metas, split)
    descriptor_rows = read_csv_rows(descriptor_path)
    descriptor_by_row = {row["row_id"]: row for row in descriptor_rows}
    descriptor_cols = _descriptor_columns(descriptor_rows)
    train_descriptor_rows = [row for row in descriptor_rows if row.get("split") == "train"] or descriptor_rows
    desc_train = np.asarray([[float(row[col]) for col in descriptor_cols] for row in train_descriptor_rows], dtype=np.float32)
    desc_mean = desc_train.mean(axis=0)
    desc_std = desc_train.std(axis=0)
    desc_std = np.where(desc_std > 1e-8, desc_std, 1.0).astype(np.float32)
    prototype_by_row, prototype_count = _prototype_map(prototype_path)
    data_rows = read_csv_rows(data_index)
    data_by_row = {row["row_id"]: row for row in data_rows}
    condition_rows: list[dict[str, Any]] = []
    for split, (_latents_raw, metas) in split_payloads.items():
        for meta in metas:
            row_id = meta["row_id"]
            descriptor = descriptor_by_row[row_id]
            data_row = data_by_row.get(row_id, {})
            out_row: dict[str, Any] = {
                "row_id": row_id,
                "parent_row_id": meta.get("parent_row_id", row_id),
                "sample_id": meta["sample_id"],
                "group_id": meta["group_id"],
                "split": split,
                "network_input_path": data_row.get("network_input_path", ""),
                "descriptor_height_path": data_row.get("descriptor_height_path", ""),
                "source_path": data_row.get("source_path", ""),
                "source_kind": data_row.get("source_kind", ""),
                "prototype_id": prototype_by_row.get(row_id, ""),
            }
            for index, col in enumerate(descriptor_cols):
                value = float(descriptor[col])
                out_row[col] = f"{value:.10g}"
                out_row[f"cond_{col}"] = f"{((value - float(desc_mean[index])) / float(desc_std[index])):.10g}"
            condition_rows.append(out_row)
    write_csv_rows(out_dir / "condition_table_v2.csv", condition_rows)
    condition_columns = [f"cond_{col}" for col in descriptor_cols]
    schema = {
        "descriptor_columns": descriptor_cols,
        "condition_columns": condition_columns,
        "prototype_count": int(prototype_count),
        "condition_dim": len(condition_columns) + int(prototype_count),
        "descriptor_train_mean": {col: float(desc_mean[i]) for i, col in enumerate(descriptor_cols)},
        "descriptor_train_std": {col: float(desc_std[i]) for i, col in enumerate(descriptor_cols)},
        "prototype_source": display_path(prototype_path),
        "prototype_one_hot": bool(prototype_count > 0),
    }
    np.savez_compressed(out_dir / "latent_standardization_v2.npz", latent_mean=latent_mean, latent_std=latent_std)
    stats = {
        "autoencoder_checkpoint": display_path(resolve_repo_path(args.checkpoint)),
        "autoencoder_config": checkpoint.get("config", {}),
        "latent_shape": list(train_latents.shape[1:]),
        "latent_train_count": int(train_latents.shape[0]),
        "latent_mean_shape": list(latent_mean.shape[1:]),
        "latent_mean_scalar": float(np.mean(latent_mean)),
        "latent_std_scalar": float(np.mean(latent_std)),
        "latent_standardization": display_path(out_dir / "latent_standardization_v2.npz"),
        "descriptor_columns": descriptor_cols,
        "condition_columns": condition_columns,
        "prototype_count": int(prototype_count),
        "split_counts": {split: int(split_payloads[split][0].shape[0]) for split in ("train", "val", "test")},
    }
    write_json(out_dir / "condition_schema_v2.json", schema)
    write_json(out_dir / "latent_stats_v2.json", stats)
    return stats


def main() -> None:
    args = build_parser().parse_args()
    stats = export_latents(args)
    print(f"Wrote AFM prior v2 latents to {display_path(resolve_repo_path(args.out))}")
    print(f"latent_shape={stats['latent_shape']} train_count={stats['latent_train_count']}")


if __name__ == "__main__":
    main()
