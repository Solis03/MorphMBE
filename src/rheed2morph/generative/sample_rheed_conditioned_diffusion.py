"""Generate AFM-like maps from RHEED-predicted MVP-1 diffusion conditions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_descriptors import compute_afm_descriptors
from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion import load_diffusion_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample AFM morphology from RHEED-predicted diffusion conditions.")
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-condition-table", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _condition_vector(row: dict[str, str], config: dict[str, Any], oracle: bool) -> np.ndarray:
    values = []
    for col in config["condition_columns"]:
        key = f"true_{col}" if oracle and row.get(f"true_{col}", "") != "" else col
        values.append(float(row[key]))
    proto_count = int(config.get("prototype_count", 0))
    if proto_count > 0:
        proto_key = "true_prototype_id" if oracle and row.get("true_prototype_id", "") != "" else "prototype_id"
        one_hot = [0.0] * proto_count
        if row.get(proto_key, "") != "":
            proto = int(float(row[proto_key]))
            if 0 <= proto < proto_count:
                one_hot[proto] = 1.0
        values.extend(one_hot)
    return np.asarray(values, dtype=np.float32)


def _mean_condition_vector(paired_rows: list[dict[str, str]], config: dict[str, Any]) -> np.ndarray:
    train_rows = [row for row in paired_rows if row.get("split") == "train"] or paired_rows
    values = []
    for col in config["condition_columns"]:
        col_values = [float(row[col]) for row in train_rows if row.get(col, "") != ""]
        values.append(float(np.mean(col_values)) if col_values else 0.0)
    proto_count = int(config.get("prototype_count", 0))
    if proto_count > 0:
        proto_values = [int(float(row["prototype_id"])) for row in train_rows if row.get("prototype_id", "") != ""]
        one_hot = [0.0] * proto_count
        if proto_values:
            counts = np.bincount(np.asarray(proto_values, dtype=np.int64), minlength=proto_count)
            one_hot[int(np.argmax(counts))] = 1.0
        values.extend(one_hot)
    return np.asarray(values, dtype=np.float32)


def _decode_latents(
    diffusion_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: GaussianDiffusion,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    condition: np.ndarray,
    latent_shape: tuple[int, int, int],
    sample_count: int,
    steps: int,
    guidance: float,
    device: torch.device,
) -> np.ndarray:
    cond = torch.from_numpy(np.repeat(condition[None], sample_count, axis=0)).to(device)
    sampled = diffusion.sample_ddim(diffusion_model, (sample_count, *latent_shape), cond, steps=steps, guidance_scale=guidance)
    raw = sampled * latent_std + latent_mean
    with torch.no_grad():
        decoded = autoencoder.decode(raw).detach().cpu().numpy()
    return decoded[:, 0]


def _ae_reconstruction(autoencoder: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        recon, _latent = autoencoder(torch.from_numpy(image[None, None].astype(np.float32)).to(device))
    return recon[0, 0].detach().cpu().numpy()


def sample_rheed_conditioned(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    diffusion_model, diffusion_payload = load_diffusion_checkpoint(args.diffusion_checkpoint, str(device))
    diffusion_config = dict(diffusion_payload["config"])
    autoencoder, _ae_payload = load_autoencoder_checkpoint(args.autoencoder_checkpoint, str(device))
    diffusion_model.to(device).eval()
    autoencoder.to(device).eval()
    latent_stats = np.load(resolve_repo_path(Path(diffusion_config["latents_dir"])) / "latent_standardization.npz")
    latent_mean = torch.from_numpy(np.asarray(latent_stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(latent_stats["latent_std"], dtype=np.float32)).to(device)
    diffusion = GaussianDiffusion(timesteps=int(diffusion_config["timesteps"]), device=device)
    predicted_rows = [row for row in read_csv_rows(resolve_repo_path(args.predicted_condition_table)) if row.get("split") == args.split]
    if not predicted_rows:
        predicted_rows = read_csv_rows(resolve_repo_path(args.predicted_condition_table))
    predicted_rows = predicted_rows[: int(args.max_conditions)]
    paired_rows = read_csv_rows(resolve_repo_path(args.paired_index))
    paired_by_pair = {row["pair_id"]: row for row in paired_rows}
    mean_condition = _mean_condition_vector(paired_rows, diffusion_config)
    latent_shape = tuple(int(value) for value in diffusion_config["latent_shape"])
    sample_count = int(args.num_samples_per_condition)
    grid_rows: list[list[np.ndarray]] = []
    row_titles: list[str] = []
    metric_rows: list[dict[str, Any]] = []
    for row in predicted_rows:
        pair = paired_by_pair.get(row.get("pair_id", ""), {})
        rheed = np.load(resolve_repo_path(Path(row.get("cached_tensor_path", pair.get("cached_tensor_path", "")))))["frames"]
        rheed_frame = rheed[-1, 0]
        true_afm = load_height_array(resolve_repo_path(Path(row.get("network_input_path", pair.get("network_input_path", "")))))
        recon = _ae_reconstruction(autoencoder, true_afm, device)
        oracle_condition = _condition_vector(row, diffusion_config, oracle=True)
        predicted_condition = _condition_vector(row, diffusion_config, oracle=False)
        oracle_images = _decode_latents(
            diffusion_model,
            autoencoder,
            diffusion,
            latent_mean,
            latent_std,
            oracle_condition,
            latent_shape,
            1,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )
        predicted_images = _decode_latents(
            diffusion_model,
            autoencoder,
            diffusion,
            latent_mean,
            latent_std,
            predicted_condition,
            latent_shape,
            sample_count,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )
        mean_images = _decode_latents(
            diffusion_model,
            autoencoder,
            diffusion,
            latent_mean,
            latent_std,
            mean_condition,
            latent_shape,
            1,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )
        grid_rows.append([rheed_frame, true_afm, recon, oracle_images[0], *[predicted_images[i] for i in range(sample_count)]])
        row_titles.append(str(row.get("sample_id", row.get("row_id", ""))))
        for mode, images in (("mean", mean_images), ("oracle", oracle_images), ("predicted", predicted_images)):
            for image_index, image in enumerate(images):
                desc = compute_afm_descriptors(image)
                metric_rows.append(
                    {
                        "row_id": row.get("row_id", ""),
                        "pair_id": row.get("pair_id", ""),
                        "sample_id": row.get("sample_id", ""),
                        "mode": mode,
                        "generated_index": image_index,
                        "generated_std": f"{float(np.std(image)):.10g}",
                        "generated_min": f"{float(np.min(image)):.10g}",
                        "generated_max": f"{float(np.max(image)):.10g}",
                        "generated_rq": f"{float(desc['rq']):.10g}",
                        "generated_ra": f"{float(desc['ra']):.10g}",
                        "generated_psd_low_power": f"{float(desc['psd_low_power']):.10g}",
                        "true_cond_rq": row.get("true_cond_rq", ""),
                        "pred_cond_rq": row.get("cond_rq", ""),
                        "true_cond_ra": row.get("true_cond_ra", ""),
                        "pred_cond_ra": row.get("cond_ra", ""),
                    }
                )
    column_titles = ["RHEED final frame", "true AFM", "AE reconstruction", "oracle-conditioned generation"] + [
        f"RHEED-pred sample {i + 1}" for i in range(sample_count)
    ]
    grid_path = out_dir / f"rheed_conditioned_sample_grid_{args.split}.png"
    write_panel_grid(grid_path, grid_rows, column_titles, row_titles)
    write_csv_rows(out_dir / "generation_metrics.csv", metric_rows)
    pred_stds = [float(row["generated_std"]) for row in metric_rows if row["mode"] == "predicted"]
    oracle_stds = [float(row["generated_std"]) for row in metric_rows if row["mode"] == "oracle"]
    mean_stds = [float(row["generated_std"]) for row in metric_rows if row["mode"] == "mean"]
    summary = {
        "split": args.split,
        "condition_count": len(predicted_rows),
        "num_samples_per_condition": sample_count,
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale": float(args.guidance_scale),
        "sample_grid": display_path(grid_path),
        "generation_metrics": display_path(out_dir / "generation_metrics.csv"),
        "predicted_generated_std_mean": float(np.mean(pred_stds)) if pred_stds else 0.0,
        "predicted_generated_std_min": float(np.min(pred_stds)) if pred_stds else 0.0,
        "oracle_generated_std_mean": float(np.mean(oracle_stds)) if oracle_stds else 0.0,
        "mean_condition_generated_std_mean": float(np.mean(mean_stds)) if mean_stds else 0.0,
        "mean_condition_generated_nonconstant": bool(mean_stds and np.min(mean_stds) > 1e-4),
        "predicted_generated_nonconstant": bool(pred_stds and np.min(pred_stds) > 1e-4),
        "oracle_generated_nonconstant": bool(oracle_stds and np.min(oracle_stds) > 1e-4),
        "note": "Predicted-mode images use RHEED-predicted standardized descriptor/prototype conditions; oracle-mode images use true AFM conditions.",
    }
    write_json(out_dir / "generation_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_rheed_conditioned(args)
    print(f"Wrote RHEED-conditioned samples to {display_path(resolve_repo_path(args.out))}")
    print(f"predicted_generated_std_mean={summary['predicted_generated_std_mean']:.6f}")


if __name__ == "__main__":
    main()
