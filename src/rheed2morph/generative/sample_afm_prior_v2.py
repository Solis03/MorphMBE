"""Sample AFM-like maps from AFM prior v2 latent diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import build_condition_matrix_v2, compute_afm_descriptors_v2
from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_csv_rows,
    read_json,
    resolve_repo_path,
    resolve_torch_device,
    set_seed,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample AFM prior v2.")
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--max-conditions", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _rows_for_split(rows: list[dict[str, str]], split: str, limit: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split") == split]
    if not selected and split != "train":
        selected = [row for row in rows if row.get("split") == "train"]
    return selected[:limit]


def _condition_from_values(values: list[float], prototype_id: int | None, schema: dict[str, Any]) -> np.ndarray:
    proto_count = int(schema.get("prototype_count", 0))
    out = list(values)
    if proto_count > 0:
        one_hot = [0.0] * proto_count
        if prototype_id is not None and 0 <= prototype_id < proto_count:
            one_hot[prototype_id] = 1.0
        out.extend(one_hot)
    return np.asarray(out, dtype=np.float32)


def _mean_condition(rows: list[dict[str, str]], schema: dict[str, Any]) -> np.ndarray:
    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    values = []
    for col in schema["condition_columns"]:
        col_values = [float(row[col]) for row in train_rows if row.get(col, "") != ""]
        values.append(float(np.mean(col_values)) if col_values else 0.0)
    proto_values = [int(float(row["prototype_id"])) for row in train_rows if row.get("prototype_id", "") != ""]
    prototype_id = None
    if proto_values:
        counts = np.bincount(np.asarray(proto_values, dtype=np.int64), minlength=int(schema.get("prototype_count", 0)))
        prototype_id = int(np.argmax(counts))
    return _condition_from_values(values, prototype_id, schema)


def _decode_samples(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: GaussianDiffusionV2,
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
    sampled = diffusion.sample_ddim(model, (sample_count, *latent_shape), cond, steps=steps, guidance_scale=guidance)
    raw = sampled * latent_std + latent_mean
    with torch.no_grad():
        decoded = autoencoder.decode(raw).detach().cpu().numpy()
    return decoded[:, 0]


def _ae_recon(autoencoder: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        recon, _latent = autoencoder(torch.from_numpy(image[None, None].astype(np.float32)).to(device))
    return recon[0, 0].detach().cpu().numpy()


def _record_metrics(
    rows: list[dict[str, Any]],
    mode: str,
    source_row: dict[str, str],
    images: np.ndarray,
    schema: dict[str, Any],
) -> None:
    for index, image in enumerate(images):
        descriptors = compute_afm_descriptors_v2(image)
        out: dict[str, Any] = {
            "mode": mode,
            "row_id": source_row.get("row_id", ""),
            "sample_id": source_row.get("sample_id", ""),
            "group_id": source_row.get("group_id", ""),
            "prototype_id": source_row.get("prototype_id", ""),
            "generated_index": index,
            "generated_std": f"{float(np.std(image)):.10g}",
            "generated_min": f"{float(np.min(image)):.10g}",
            "generated_max": f"{float(np.max(image)):.10g}",
        }
        for name in schema["descriptor_columns"]:
            if name in descriptors:
                out[f"generated_{name}"] = f"{float(descriptors[name]):.10g}"
            if source_row.get(name, "") != "":
                out[f"requested_{name}"] = source_row[name]
        rows.append(out)


def sample_prior(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    model, diffusion_payload = load_diffusion_v2_checkpoint(args.diffusion_checkpoint, str(device))
    config = dict(diffusion_payload["config"])
    model.to(device).eval()
    autoencoder, _ae_payload = load_autoencoder_v2_checkpoint(args.autoencoder_checkpoint, str(device))
    autoencoder.to(device).eval()
    condition_table = resolve_repo_path(args.condition_table)
    rows = read_csv_rows(condition_table)
    schema = read_json(condition_table.parent / "condition_schema_v2.json")
    latent_stats = np.load(resolve_repo_path(Path(config["latents_dir"])) / "latent_standardization_v2.npz")
    latent_mean = torch.from_numpy(np.asarray(latent_stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(latent_stats["latent_std"], dtype=np.float32)).to(device)
    diffusion = GaussianDiffusionV2(
        timesteps=int(config["timesteps"]),
        beta_schedule=str(config.get("beta_schedule", "cosine")),
        prediction_target=str(config.get("prediction_target", "epsilon")),
        device=device,
    )
    latent_shape = tuple(int(value) for value in config["latent_shape"])
    selected = _rows_for_split(rows, str(args.split), int(args.max_conditions))
    if not selected:
        raise RuntimeError(f"No condition rows available for split={args.split} in {condition_table}")
    sample_count = int(args.num_samples_per_condition)
    metric_rows: list[dict[str, Any]] = []
    all_images: list[np.ndarray] = []
    all_modes: list[str] = []
    all_row_ids: list[str] = []

    oracle_grid_rows: list[list[np.ndarray]] = []
    oracle_titles: list[str] = []
    for row in selected:
        condition = build_condition_matrix_v2(rows, [row["row_id"]], schema)[0]
        generated = _decode_samples(
            model,
            autoencoder,
            diffusion,
            latent_mean,
            latent_std,
            condition,
            latent_shape,
            sample_count,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )
        true_image = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        recon = _ae_recon(autoencoder, true_image, device)
        oracle_grid_rows.append([true_image, recon, *[generated[i] for i in range(min(4, sample_count))]])
        oracle_titles.append(str(row.get("sample_id", row["row_id"])))
        _record_metrics(metric_rows, "oracle", row, generated, schema)
        all_images.extend([image for image in generated])
        all_modes.extend(["oracle"] * generated.shape[0])
        all_row_ids.extend([row["row_id"]] * generated.shape[0])
    oracle_columns = ["true AFM", "AE recon"] + [f"gen{i + 1}" for i in range(min(4, sample_count))]
    oracle_grid = out_dir / f"afm_prior_v2_oracle_grid_{args.split}.png"
    write_panel_grid(oracle_grid, oracle_grid_rows, oracle_columns, oracle_titles)

    train_rows = [row for row in rows if row.get("split") == "train"] or rows
    rng = np.random.default_rng(int(args.seed))
    random_rows = [train_rows[int(i)] for i in rng.choice(len(train_rows), size=min(int(args.max_conditions), len(train_rows)), replace=False)]
    random_grid_rows: list[list[np.ndarray]] = []
    random_titles: list[str] = []
    for row in random_rows:
        condition = build_condition_matrix_v2(rows, [row["row_id"]], schema)[0]
        generated = _decode_samples(model, autoencoder, diffusion, latent_mean, latent_std, condition, latent_shape, 4, int(args.ddim_steps), float(args.guidance_scale), device)
        random_grid_rows.append([generated[i] for i in range(generated.shape[0])])
        random_titles.append(str(row.get("sample_id", row["row_id"])))
        _record_metrics(metric_rows, "random_train_condition", row, generated, schema)
        all_images.extend([image for image in generated])
        all_modes.extend(["random_train_condition"] * generated.shape[0])
        all_row_ids.extend([row["row_id"]] * generated.shape[0])
    random_grid = out_dir / "afm_prior_v2_random_grid.png"
    write_panel_grid(random_grid, random_grid_rows, [f"gen{i + 1}" for i in range(4)], random_titles)

    proto_rows: list[dict[str, str]] = []
    seen_proto: set[str] = set()
    for row in train_rows:
        proto = row.get("prototype_id", "")
        if proto != "" and proto not in seen_proto:
            proto_rows.append(row)
            seen_proto.add(proto)
    prototype_grid_rows: list[list[np.ndarray]] = []
    prototype_titles: list[str] = []
    for row in proto_rows[: max(1, int(args.max_conditions))]:
        condition = build_condition_matrix_v2(rows, [row["row_id"]], schema)[0]
        generated = _decode_samples(model, autoencoder, diffusion, latent_mean, latent_std, condition, latent_shape, 4, int(args.ddim_steps), float(args.guidance_scale), device)
        true_image = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        prototype_grid_rows.append([true_image, *[generated[i] for i in range(generated.shape[0])]])
        prototype_titles.append(f"prototype {row.get('prototype_id', '')}")
        _record_metrics(metric_rows, "prototype_balanced", row, generated, schema)
        all_images.extend([image for image in generated])
        all_modes.extend(["prototype_balanced"] * generated.shape[0])
        all_row_ids.extend([row["row_id"]] * generated.shape[0])
    prototype_grid = out_dir / "afm_prior_v2_prototype_grid.png"
    write_panel_grid(prototype_grid, prototype_grid_rows, ["prototype example", "gen1", "gen2", "gen3", "gen4"], prototype_titles)

    mean_condition = _mean_condition(rows, schema)
    mean_source = dict(selected[0])
    mean_source["row_id"] = "mean_condition"
    mean_generated = _decode_samples(model, autoencoder, diffusion, latent_mean, latent_std, mean_condition, latent_shape, 8, int(args.ddim_steps), float(args.guidance_scale), device)
    _record_metrics(metric_rows, "mean_condition", mean_source, mean_generated, schema)
    all_images.extend([image for image in mean_generated])
    all_modes.extend(["mean_condition"] * mean_generated.shape[0])
    all_row_ids.extend(["mean_condition"] * mean_generated.shape[0])

    write_csv_rows(out_dir / "generation_metrics_v2.csv", metric_rows)
    np.savez_compressed(
        out_dir / "generated_samples_v2.npz",
        images=np.asarray(all_images, dtype=np.float32),
        modes=np.asarray(all_modes),
        row_ids=np.asarray(all_row_ids),
    )
    generated_stds = [float(row["generated_std"]) for row in metric_rows]
    summary = {
        "split": str(args.split),
        "condition_count": len(selected),
        "num_samples_per_condition": sample_count,
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale": float(args.guidance_scale),
        "generated_count": len(metric_rows),
        "generated_std_mean": float(np.mean(generated_stds)) if generated_stds else 0.0,
        "generated_std_min": float(np.min(generated_stds)) if generated_stds else 0.0,
        "generated_nonconstant_rate": float(np.mean(np.asarray(generated_stds) > 1e-4)) if generated_stds else 0.0,
        "oracle_grid": display_path(oracle_grid),
        "prototype_grid": display_path(prototype_grid),
        "random_grid": display_path(random_grid),
        "generation_metrics": display_path(out_dir / "generation_metrics_v2.csv"),
        "generated_samples": display_path(out_dir / "generated_samples_v2.npz"),
        "note": "Generated samples are decoded from diffusion-sampled latents conditioned on AFM descriptor/prototype vectors.",
    }
    write_json(out_dir / "generation_summary_v2.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_prior(args)
    print(f"Wrote AFM prior v2 samples to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_std_mean={summary['generated_std_mean']:.6f}")


if __name__ == "__main__":
    main()
