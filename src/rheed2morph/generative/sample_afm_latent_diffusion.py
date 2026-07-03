"""Sample AFM-like height maps from descriptor-conditioned latent diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_descriptors import compute_afm_descriptors
from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_csv_rows,
    resolve_repo_path,
    resolve_torch_device,
    set_seed,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion import build_condition_matrix, load_diffusion_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample descriptor-conditioned AFM latent diffusion.")
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-conditions", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _condition_rows(path: Path, split: str, max_conditions: int) -> list[dict[str, str]]:
    rows = [row for row in read_csv_rows(path) if row.get("split") == split]
    if not rows and split != "train":
        rows = [row for row in read_csv_rows(path) if row.get("split") == "train"]
    return rows[:max_conditions]


def _encode_decode_true(autoencoder: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        recon, _latent = autoencoder(tensor)
    return recon[0, 0].detach().cpu().numpy()


def sample_diffusion(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    model, diffusion_payload = load_diffusion_checkpoint(args.diffusion_checkpoint, str(device))
    config = dict(diffusion_payload["config"])
    model.to(device).eval()
    autoencoder, _ae_payload = load_autoencoder_checkpoint(args.autoencoder_checkpoint, str(device))
    autoencoder.to(device).eval()
    condition_table = resolve_repo_path(args.condition_table)
    all_condition_rows = read_csv_rows(condition_table)
    selected_rows = _condition_rows(condition_table, str(args.split), int(args.max_conditions))
    if not selected_rows:
        raise RuntimeError(f"No rows for split={args.split} in {condition_table}")
    latent_stats_path = resolve_repo_path(Path(config["latents_dir"])) / "latent_standardization.npz"
    latent_stats = np.load(latent_stats_path)
    latent_mean = torch.from_numpy(np.asarray(latent_stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(latent_stats["latent_std"], dtype=np.float32)).to(device)
    diffusion = GaussianDiffusion(timesteps=int(config["timesteps"]), device=device)
    condition_columns = list(config["condition_columns"])
    prototype_count = int(config.get("prototype_count", 0))
    sample_count = int(args.num_samples_per_condition)
    latent_shape = tuple(int(value) for value in config["latent_shape"])
    metric_rows: list[dict[str, Any]] = []
    grid_rows: list[list[np.ndarray]] = []
    grid_titles: list[str] = []
    for row in selected_rows:
        row_ids = np.asarray([row["row_id"]] * sample_count)
        condition, _cols, _proto = build_condition_matrix(all_condition_rows, row_ids, condition_columns, prototype_count)
        cond_tensor = torch.from_numpy(condition).to(device)
        sampled = diffusion.sample_ddim(
            model,
            (sample_count, *latent_shape),
            cond_tensor,
            steps=int(args.ddim_steps),
            guidance_scale=float(args.guidance_scale),
        )
        sampled_raw = sampled * latent_std + latent_mean
        with torch.no_grad():
            decoded = autoencoder.decode(sampled_raw).detach().cpu().numpy()
        true_path = row.get("network_input_path", "")
        true_image = load_height_array(resolve_repo_path(Path(true_path))) if true_path else np.zeros((128, 128), dtype=np.float32)
        reconstruction = _encode_decode_true(autoencoder, true_image, device)
        panels = [true_image, reconstruction]
        for sample_index in range(sample_count):
            image = decoded[sample_index, 0]
            panels.append(image)
            desc = compute_afm_descriptors(image)
            metric_rows.append(
                {
                    "row_id": row["row_id"],
                    "sample_id": row.get("sample_id", ""),
                    "generated_index": sample_index,
                    "generated_std": f"{float(np.std(image)):.10g}",
                    "generated_min": f"{float(np.min(image)):.10g}",
                    "generated_max": f"{float(np.max(image)):.10g}",
                    "generated_rq": f"{float(desc['rq']):.10g}",
                    "condition_rq": row.get("rq", ""),
                    "generated_ra": f"{float(desc['ra']):.10g}",
                    "condition_ra": row.get("ra", ""),
                    "generated_psd_low_power": f"{float(desc['psd_low_power']):.10g}",
                    "condition_psd_low_power": row.get("psd_low_power", ""),
                    "generated_psd_mid_power": f"{float(desc['psd_mid_power']):.10g}",
                    "condition_psd_mid_power": row.get("psd_mid_power", ""),
                    "generated_psd_high_power": f"{float(desc['psd_high_power']):.10g}",
                    "condition_psd_high_power": row.get("psd_high_power", ""),
                }
            )
        grid_rows.append(panels)
        grid_titles.append(str(row.get("sample_id", row["row_id"])))
    column_titles = ["true AFM", "AE reconstruction"] + [f"generated sample {i + 1}" for i in range(sample_count)]
    grid_path = out_dir / f"sample_grid_{args.split}.png"
    write_panel_grid(grid_path, grid_rows, column_titles, grid_titles)
    write_csv_rows(out_dir / "generation_metrics.csv", metric_rows)
    generated_stds = [float(row["generated_std"]) for row in metric_rows]
    summary = {
        "split": str(args.split),
        "condition_count": len(selected_rows),
        "num_samples_per_condition": sample_count,
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale": float(args.guidance_scale),
        "sample_grid": display_path(grid_path),
        "generation_metrics": display_path(out_dir / "generation_metrics.csv"),
        "generated_std_mean": float(np.mean(generated_stds)) if generated_stds else 0.0,
        "generated_std_min": float(np.min(generated_stds)) if generated_stds else 0.0,
        "generated_nonconstant": bool(generated_stds and np.min(generated_stds) > 1e-4),
        "note": "Generated images are decoded from diffusion-sampled latents conditioned on AFM descriptor oracle vectors.",
    }
    write_json(out_dir / "generation_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_diffusion(args)
    print(f"Wrote diffusion samples to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_std_mean={summary['generated_std_mean']:.6f}")


if __name__ == "__main__":
    main()
