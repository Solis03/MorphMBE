#!/usr/bin/env python3
"""Train a lightweight AFM autoencoder MVP on one-to-one manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rheed2morph.afm.mvp import (
    AFMExample,
    build_autoencoder,
    display_path,
    load_afm_array,
    load_manifest_examples,
    preprocess_afm_array,
    reconstruction_loss,
    resolve_torch_device,
    save_checkpoint,
    write_csv,
    write_latent_pca_plot,
    write_loss_plot,
    write_reconstruction_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AFM autoencoder MVP.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--normalize-mode", default="per_image_zscore", choices=["per_image_zscore", "per_image_minmax", "global_zscore"])
    parser.add_argument("--model-type", default="residual", choices=["baseline", "residual"])
    parser.add_argument("--pixel-loss", default="smooth_l1", choices=["mse", "smooth_l1"])
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def load_examples(
    manifest_path: Path,
    image_size: int,
    normalize_mode: str,
) -> tuple[list[AFMExample], np.ndarray, list[dict[str, float]], dict[str, float]]:
    examples = load_manifest_examples(manifest_path)
    resized_images: list[np.ndarray] = []
    stats: list[dict[str, float]] = []
    for example in examples:
        array = load_afm_array(example.afm_path)
        resized, item_stats = preprocess_afm_array(array, image_size=image_size, normalize_mode="per_image_zscore")
        resized_images.append(resized.astype(np.float32))
        stats.append(item_stats)

    stacked = np.stack(resized_images, axis=0)
    global_stats = {
        "mean": float(np.mean(stacked)),
        "std": float(np.std(stacked)),
    }
    if normalize_mode == "global_zscore":
        denom = global_stats["std"] if global_stats["std"] > 1e-6 else 1.0
        normalized = ((stacked - global_stats["mean"]) / denom).astype(np.float32)
    elif normalize_mode in {"per_image_zscore", "per_image_minmax"}:
        if normalize_mode == "per_image_zscore":
            normalized = stacked
        else:
            remapped: list[np.ndarray] = []
            for example in examples:
                array = load_afm_array(example.afm_path)
                item, _ = preprocess_afm_array(array, image_size=image_size, normalize_mode=normalize_mode)
                remapped.append(item.astype(np.float32))
            normalized = np.stack(remapped, axis=0)
    else:
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
    return examples, normalized, stats, global_stats


def split_groups(examples: list[AFMExample], val_fraction: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([example.group_id for example in examples])
    indices = np.arange(len(examples))
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return indices, indices
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    return train_idx.astype(int), val_idx.astype(int)


def iterate_minibatches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, indices.size, batch_size)]


def evaluate(
    model: torch.nn.Module,
    images: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    pixel_loss: str,
    edge_loss_weight: float,
) -> tuple[float, float, float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    pixel_losses: list[float] = []
    edge_losses: list[float] = []
    reconstructions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx in iterate_minibatches(indices, batch_size):
            batch = images[batch_idx]
            recon, _ = model(batch)
            loss, parts = reconstruction_loss(recon, batch, pixel_loss=pixel_loss, edge_weight=edge_loss_weight)
            losses.append(float(loss.detach().cpu()))
            pixel_losses.append(parts["pixel_loss"])
            edge_losses.append(parts["edge_loss"])
            reconstructions.append(recon.detach().cpu().numpy())
    return (
        float(np.mean(losses)),
        float(np.mean(pixel_losses)),
        float(np.mean(edge_losses)),
        np.concatenate(reconstructions, axis=0),
    )


def summarize_reconstruction_quality(inputs: np.ndarray, reconstructions: np.ndarray) -> tuple[dict[str, float], bool, str]:
    abs_error = np.abs(reconstructions - inputs)
    mae = float(np.mean(abs_error))
    mse = float(np.mean((reconstructions - inputs) ** 2))
    input_std = float(np.std(inputs))
    recon_std = float(np.std(reconstructions))
    std_ratio = recon_std / input_std if input_std > 1e-6 else 0.0
    collapsed = std_ratio < 0.35 or mae > 0.75 or mse > 0.9
    warning = (
        "AFM latent space may not yet be morphology-preserving; RHEED-to-latent metrics should not be overinterpreted."
        if collapsed
        else ""
    )
    return {
        "reconstruction_mae": mae,
        "reconstruction_mse": mse,
        "input_std": input_std,
        "reconstruction_std": recon_std,
        "reconstruction_std_ratio": std_ratio,
    }, collapsed, warning


def write_summary(path: Path, metrics: dict[str, object]) -> None:
    warning = metrics.get("quality_warning", "")
    lines = [
        "# AFM Autoencoder MVP",
        "",
        f"- Manifest: `{metrics['manifest']}`",
        f"- Model type: `{metrics['model_type']}`",
        f"- Normalization: `{metrics['normalize_mode']}`",
        f"- Pixel loss / edge weight: `{metrics['pixel_loss']}` / `{float(metrics['edge_loss_weight']):.3f}`",
        f"- Train rows / groups: `{metrics['train_row_count']}` / `{metrics['train_group_count']}`",
        f"- Val rows / groups: `{metrics['val_row_count']}` / `{metrics['val_group_count']}`",
        f"- Best epoch: `{metrics['best_epoch']}`",
        f"- Best val reconstruction loss: `{float(metrics['best_val_loss']):.6f}`",
        f"- Final val reconstruction loss: `{float(metrics['final_val_loss']):.6f}`",
        f"- Reconstruction MAE: `{float(metrics['reconstruction_mae']):.6f}`",
        f"- Reconstruction std ratio: `{float(metrics['reconstruction_std_ratio']):.4f}`",
        "",
        "## Interpretation",
        "",
    ]
    if warning:
        lines.append(f"- Warning: {warning}")
    else:
        lines.append("- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else (Path.cwd() / args.manifest)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples, image_array, image_stats, global_stats = load_examples(manifest_path, args.image_size, args.normalize_mode)
    train_idx, val_idx = split_groups(examples, args.val_fraction, args.random_state)
    train_groups = {examples[index].group_id for index in train_idx}
    val_groups = {examples[index].group_id for index in val_idx}
    print(f"Train groups: {len(train_groups)} | Val groups: {len(val_groups)}")

    device = resolve_torch_device(args.device)
    images = torch.from_numpy(image_array[:, None, :, :]).to(device)
    model = build_autoencoder(args.model_type, image_size=args.image_size, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, float | int]] = []
    checkpoint_path = out_dir / "autoencoder_checkpoint.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        shuffled = np.array(train_idx, copy=True)
        rng = np.random.default_rng(args.random_state + epoch)
        rng.shuffle(shuffled)
        for batch_idx in iterate_minibatches(shuffled, args.batch_size):
            batch = images[batch_idx]
            recon, _ = model(batch)
            loss, parts = reconstruction_loss(
                recon,
                batch,
                pixel_loss=args.pixel_loss,
                edge_weight=args.edge_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(parts["total_loss"]))

        val_loss, val_pixel_loss, val_edge_loss, _ = evaluate(
            model,
            images,
            val_idx,
            args.batch_size,
            pixel_loss=args.pixel_loss,
            edge_loss_weight=args.edge_loss_weight,
        )
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        improved = val_loss < (best_val_loss - args.min_delta)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "val_pixel_loss": float(val_pixel_loss),
                "val_edge_loss": float(val_edge_loss),
                "best_so_far": min(best_val_loss, val_loss) if np.isfinite(best_val_loss) else float(val_loss),
                "improved": int(improved),
            }
        )
        if improved:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                args.image_size,
                args.latent_dim,
                args.normalize_mode,
                model_type=args.model_type,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                stopped_early = True
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    final_val_loss, final_val_pixel_loss, final_val_edge_loss, val_recons = evaluate(
        model,
        images,
        val_idx,
        args.batch_size,
        pixel_loss=args.pixel_loss,
        edge_loss_weight=args.edge_loss_weight,
    )
    _, _, _, _ = evaluate(
        model,
        images,
        train_idx,
        args.batch_size,
        pixel_loss=args.pixel_loss,
        edge_loss_weight=args.edge_loss_weight,
    )
    with torch.no_grad():
        latents = model.encode(images).detach().cpu().numpy().astype(np.float32)

    val_inputs = image_array[val_idx]
    val_recon_images = val_recons[:, 0]
    error_maps = np.abs(val_inputs - val_recon_images)
    quality_metrics, quality_warning_triggered, quality_warning = summarize_reconstruction_quality(val_inputs, val_recon_images)

    np.save(out_dir / "afm_latents.npy", latents)
    split_lookup = set(train_idx.tolist())
    latent_rows = []
    latent_labels: list[str] = []
    latent_splits: list[str] = []
    for index, (example, stats) in enumerate(zip(examples, image_stats)):
        split_name = "train" if index in split_lookup else "val"
        latent_rows.append(
            {
                "row_id": example.row_id,
                "sample_id": example.sample_id,
                "group_id": example.group_id,
                "material": example.material,
                "rheed_path": display_path(example.rheed_path),
                "afm_path": display_path(example.afm_path),
                "split": split_name,
                "norm_mean": f"{stats['mean']:.8f}",
                "norm_std": f"{stats['std']:.8f}",
                "norm_min": f"{stats['min']:.8f}",
                "norm_max": f"{stats['max']:.8f}",
            }
        )
        latent_labels.append(example.sample_id)
        latent_splits.append(split_name)
    write_csv(
        out_dir / "afm_latent_index.csv",
        latent_rows,
        [
            "row_id",
            "sample_id",
            "group_id",
            "material",
            "rheed_path",
            "afm_path",
            "split",
            "norm_mean",
            "norm_std",
            "norm_min",
            "norm_max",
        ],
    )
    write_csv(
        out_dir / "training_history.csv",
        history,
        ["epoch", "train_loss", "val_loss", "val_pixel_loss", "val_edge_loss", "best_so_far", "improved"],
    )
    write_loss_plot(out_dir / "training_curve.png", history)
    write_reconstruction_grid(
        out_dir / "recon_grid.png",
        val_inputs,
        val_recon_images,
        [f"{examples[index].sample_id} | {examples[index].group_id}" for index in val_idx],
        error_maps=error_maps,
    )
    write_latent_pca_plot(out_dir / "latent_pca.png", latents, latent_labels, latent_splits)

    metrics = {
        "manifest": str(manifest_path),
        "image_size": args.image_size,
        "latent_dim": args.latent_dim,
        "model_type": args.model_type,
        "pixel_loss": args.pixel_loss,
        "edge_loss_weight": args.edge_loss_weight,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "stopped_early": stopped_early,
        "normalize_mode": args.normalize_mode,
        "global_normalization_mean": global_stats["mean"],
        "global_normalization_std": global_stats["std"],
        "train_row_count": int(train_idx.size),
        "val_row_count": int(val_idx.size),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_pixel_loss": float(final_val_pixel_loss),
        "final_val_edge_loss": float(final_val_edge_loss),
        "quality_warning": quality_warning,
        "quality_warning_triggered": quality_warning_triggered,
        **quality_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_dir / "summary.md", metrics)
    print(f"Wrote AFM autoencoder outputs to {display_path(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
