"""Sample AFM prior v3 with descriptor guidance and reranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import compute_afm_descriptors_v2
from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import bool_arg, condition_row_to_vector, raw_descriptor_to_condition
from rheed2morph.generative.descriptor_guided_sampling import decode_latents, ddim_sample_with_descriptor_guidance, rerank_decoded_candidates
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v3 import load_diffusion_v3_checkpoint
from rheed2morph.generative.train_latent_descriptor_regressor import load_regressor_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample AFM prior v3.")
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--latent-descriptor-regressor", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=16)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--descriptor-guidance-weight", type=float, default=0.1)
    parser.add_argument("--rerank", type=bool_arg, default=True)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_v2_context(autoencoder_checkpoint: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any], dict[str, Any], list[dict[str, str]], torch.Tensor, torch.Tensor] | None:
    root = resolve_repo_path(autoencoder_checkpoint).parents[2]
    diffusion_path = root / "latent_diffusion_v2" / "checkpoints" / "ema_last.pt"
    table_path = root / "latents_v2" / "condition_table_v2.csv"
    schema_path = root / "latents_v2" / "condition_schema_v2.json"
    stats_path = root / "latents_v2" / "latent_standardization_v2.npz"
    if not diffusion_path.is_file() or not table_path.is_file() or not schema_path.is_file() or not stats_path.is_file():
        return None
    model, payload = load_diffusion_v2_checkpoint(diffusion_path, str(device))
    config = dict(payload["config"])
    schema = read_json(schema_path)
    rows = read_csv_rows(table_path)
    stats = np.load(stats_path)
    latent_mean = torch.from_numpy(np.asarray(stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(stats["latent_std"], dtype=np.float32)).to(device)
    model.to(device).eval()
    return model, config, schema, rows, latent_mean, latent_std


def _sample_images(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: GaussianDiffusionV2,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    regressor: torch.nn.Module | None,
    condition: np.ndarray,
    latent_shape: tuple[int, int, int],
    descriptor_dim: int,
    count: int,
    steps: int,
    guidance_scale: float,
    descriptor_guidance_weight: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    cond = torch.from_numpy(np.repeat(condition[None], count, axis=0).astype(np.float32)).to(device)
    latents = ddim_sample_with_descriptor_guidance(
        model,
        diffusion,
        (count, *latent_shape),
        cond,
        descriptor_regressor=regressor,
        descriptor_dim=descriptor_dim,
        steps=steps,
        guidance_scale=guidance_scale,
        descriptor_guidance_weight=descriptor_guidance_weight,
    )
    return decode_latents(autoencoder, latents, latent_mean, latent_std), latents


def _metric_row(mode: str, row: dict[str, str], image: np.ndarray, index: int, schema: dict[str, Any], rank: int | str = "") -> dict[str, Any]:
    desc = compute_afm_descriptors_v2(image)
    out: dict[str, Any] = {
        "mode": mode,
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "group_id": row.get("group_id", ""),
        "prototype_id": row.get("prototype_id", ""),
        "generated_index": index,
        "rank": rank,
        "generated_std": float(np.std(image)),
        "generated_min": float(np.min(image)),
        "generated_max": float(np.max(image)),
    }
    for name in schema["descriptor_columns"]:
        if name in desc:
            out[f"generated_{name}"] = float(desc[name])
        if row.get(name, "") != "":
            out[f"requested_{name}"] = float(row[name])
    return out


def _ae_recon(autoencoder: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        recon, _latent = autoencoder(torch.from_numpy(image[None, None].astype(np.float32)).to(device))
    return recon[0, 0].detach().cpu().numpy()


def _condition_sweep_rows(base_row: dict[str, str], descriptor: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    mean = float(schema["descriptor_train_mean"][descriptor])
    std = float(schema["descriptor_train_std"].get(descriptor, 1.0) or 1.0)
    rows = []
    for z in (-2.0, -1.0, 0.0, 1.0, 2.0):
        raw = mean + z * std
        row = dict(base_row)
        row[descriptor] = f"{raw:.10g}"
        row[f"cond_{descriptor}"] = f"{raw_descriptor_to_condition(descriptor, raw, schema):.10g}"
        rows.append(row)
    return rows


def sample_v3(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    rows = read_csv_rows(resolve_repo_path(args.condition_table))
    schema = read_json(resolve_repo_path(args.condition_schema))
    selected = [row for row in rows if row.get("split") == args.split][: int(args.max_conditions)]
    if not selected:
        selected = rows[: int(args.max_conditions)]
    model, payload = load_diffusion_v3_checkpoint(args.diffusion_checkpoint, str(device))
    config = dict(payload["config"])
    model.to(device).eval()
    autoencoder, _ae = load_autoencoder_v2_checkpoint(args.autoencoder_checkpoint, str(device))
    autoencoder.to(device).eval()
    regressor, _reg = load_regressor_checkpoint(args.latent_descriptor_regressor, str(device))
    regressor.to(device).eval()
    stats = np.load(resolve_repo_path(Path(config["latents_dir"])) / "latent_standardization_v2.npz")
    latent_mean = torch.from_numpy(np.asarray(stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(stats["latent_std"], dtype=np.float32)).to(device)
    diffusion = GaussianDiffusionV2(int(config["timesteps"]), str(config["beta_schedule"]), str(config["prediction_target"]), device)
    latent_shape = tuple(int(value) for value in config["latent_shape"])
    descriptor_dim = len(schema["condition_columns"])
    v2_context = _load_v2_context(args.autoencoder_checkpoint, device)
    metric_rows: list[dict[str, Any]] = []
    rerank_rows: list[dict[str, Any]] = []
    all_images: list[np.ndarray] = []
    all_modes: list[str] = []
    grid_rows: list[list[np.ndarray]] = []
    row_titles: list[str] = []
    for row in selected:
        condition = condition_row_to_vector(row, schema)
        plain, _plain_latents = _sample_images(model, autoencoder, diffusion, latent_mean, latent_std, None, condition, latent_shape, descriptor_dim, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), 0.0, device)
        guided, _guided_latents = _sample_images(model, autoencoder, diffusion, latent_mean, latent_std, regressor, condition, latent_shape, descriptor_dim, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
        top_images = guided[: min(int(args.keep_top_k), guided.shape[0])]
        candidate_metrics: list[dict[str, Any]] = []
        if bool(args.rerank):
            top_images, candidate_metrics = rerank_decoded_candidates(guided, row, schema, int(args.keep_top_k))
            for candidate in candidate_metrics:
                out = {"row_id": row["row_id"], "sample_id": row.get("sample_id", ""), **candidate}
                rerank_rows.append(out)
        true_image = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        recon = _ae_recon(autoencoder, true_image, device)
        v2_image = np.zeros_like(true_image)
        if v2_context is not None:
            v2_model, v2_config, v2_schema, v2_rows, v2_mean, v2_std = v2_context
            v2_by_row = {item["row_id"]: item for item in v2_rows}
            if row["row_id"] in v2_by_row:
                v2_condition = condition_row_to_vector(v2_by_row[row["row_id"]], v2_schema)
                v2_diff = GaussianDiffusionV2(int(v2_config["timesteps"]), str(v2_config.get("beta_schedule", "cosine")), str(v2_config.get("prediction_target", "epsilon")), device)
                v2_latent_shape = tuple(int(value) for value in v2_config["latent_shape"])
                v2_image, _ = _sample_images(v2_model, autoencoder, v2_diff, v2_mean, v2_std, None, v2_condition, v2_latent_shape, len(v2_schema["condition_columns"]), 1, int(args.ddim_steps), 1.5, 0.0, device)
                v2_image = v2_image[0]
        panels = [true_image, recon, v2_image, plain[0], guided[0], top_images[0], top_images[min(1, top_images.shape[0] - 1)]]
        grid_rows.append(panels)
        row_titles.append(str(row.get("sample_id", row["row_id"])))
        for mode, images in (("v3_plain", plain), ("v3_guided", guided), ("v3_reranked", top_images)):
            for index, image in enumerate(images):
                metric_rows.append(_metric_row(mode, row, image, index, schema, rank=(index + 1 if mode == "v3_reranked" else "")))
                all_images.append(image)
                all_modes.append(mode)
    oracle_grid = out_dir / f"afm_prior_v3_oracle_grid_{args.split}.png"
    write_panel_grid(
        oracle_grid,
        grid_rows,
        ["true AFM", "AE recon", "v2 gen best", "v3 plain", "v3 guided", "v3 reranked top1", "v3 reranked top2"],
        row_titles,
    )
    sweep_metrics: list[dict[str, Any]] = []
    sweep_files = {
        "rq": "afm_prior_v3_condition_sweep_rq.png",
        "psd_slope": "afm_prior_v3_condition_sweep_psd.png",
        "autocorrelation_length_px": "afm_prior_v3_condition_sweep_autocorr.png",
        "gradient_anisotropy": "afm_prior_v3_condition_sweep_anisotropy.png",
    }
    if selected:
        base = selected[0]
        for descriptor, filename in sweep_files.items():
            if descriptor not in schema["descriptor_columns"]:
                continue
            panels = []
            titles = []
            for sweep_row in _condition_sweep_rows(base, descriptor, schema):
                condition = condition_row_to_vector(sweep_row, schema)
                images, _ = _sample_images(model, autoencoder, diffusion, latent_mean, latent_std, regressor, condition, latent_shape, descriptor_dim, 1, int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
                panels.append(images[0])
                titles.append(f"{descriptor}={float(sweep_row[descriptor]):.2g}")
                out = _metric_row(f"sweep_{descriptor}", sweep_row, images[0], 0, schema)
                out["sweep_descriptor"] = descriptor
                sweep_metrics.append(out)
            write_panel_grid(out_dir / filename, [panels], titles)
    proto_rows = []
    seen = set()
    for row in rows:
        proto = row.get("prototype_id", "")
        if proto != "" and proto not in seen:
            proto_rows.append(row)
            seen.add(proto)
    if proto_rows:
        proto_panels = []
        proto_titles = []
        for row in proto_rows[: int(args.max_conditions)]:
            condition = condition_row_to_vector(row, schema)
            images, _ = _sample_images(model, autoencoder, diffusion, latent_mean, latent_std, regressor, condition, latent_shape, descriptor_dim, 1, int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
            proto_panels.append(images[0])
            proto_titles.append(f"prototype {row.get('prototype_id')}")
        write_panel_grid(out_dir / "afm_prior_v3_prototype_grid.png", [proto_panels], proto_titles)
    rng = np.random.default_rng(int(args.seed))
    random_rows = [rows[int(i)] for i in rng.choice(len(rows), size=min(int(args.max_conditions), len(rows)), replace=False)]
    random_panels = []
    random_titles = []
    for row in random_rows:
        condition = condition_row_to_vector(row, schema)
        images, _ = _sample_images(model, autoencoder, diffusion, latent_mean, latent_std, regressor, condition, latent_shape, descriptor_dim, 1, int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
        random_panels.append(images[0])
        random_titles.append(str(row.get("sample_id", row["row_id"])))
    write_panel_grid(out_dir / "afm_prior_v3_random_grid.png", [random_panels], random_titles)
    write_csv_rows(out_dir / "generation_metrics_v3.csv", metric_rows)
    write_csv_rows(out_dir / "reranking_metrics_v3.csv", rerank_rows)
    write_csv_rows(out_dir / "condition_sweep_metrics_v3.csv", sweep_metrics)
    np.savez_compressed(out_dir / "generated_candidates_v3.npz", images=np.asarray(all_images, dtype=np.float32), modes=np.asarray(all_modes))
    stds = [float(row["generated_std"]) for row in metric_rows]
    summary = {
        "split": args.split,
        "condition_count": len(selected),
        "num_samples_per_condition": int(args.num_samples_per_condition),
        "keep_top_k": int(args.keep_top_k),
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale": float(args.guidance_scale),
        "descriptor_guidance_weight": float(args.descriptor_guidance_weight),
        "rerank": bool(args.rerank),
        "generated_count": len(metric_rows),
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "generated_std_min": float(np.min(stds)) if stds else 0.0,
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "oracle_grid": display_path(oracle_grid),
        "generation_metrics": display_path(out_dir / "generation_metrics_v3.csv"),
        "reranking_metrics": display_path(out_dir / "reranking_metrics_v3.csv"),
        "condition_sweep_metrics": display_path(out_dir / "condition_sweep_metrics_v3.csv"),
        "generated_candidates": display_path(out_dir / "generated_candidates_v3.npz"),
    }
    write_json(out_dir / "generation_summary_v3.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_v3(args)
    print(f"Wrote AFM prior v3 samples to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
