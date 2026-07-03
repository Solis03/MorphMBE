"""Rerun MVP-2 RHEED-conditioned generation using the v3 AFM prior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import adapt_external_condition_row, bool_arg, condition_row_to_vector
from rheed2morph.generative.descriptor_guided_sampling import decode_latents, ddim_sample_with_descriptor_guidance, rerank_decoded_candidates
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.sample_afm_prior_v3 import _metric_row
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v3 import load_diffusion_v3_checkpoint
from rheed2morph.generative.train_latent_descriptor_regressor import load_regressor_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun RHEED-conditioned generation with v3 prior.")
    parser.add_argument("--mvp2-root", type=Path, required=True)
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v3-diffusion", type=Path, required=True)
    parser.add_argument("--v3-autoencoder", type=Path, required=True)
    parser.add_argument("--v3-condition-schema", type=Path, required=True)
    parser.add_argument("--latent-descriptor-regressor", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=16)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--descriptor-guidance-weight", type=float, default=0.1)
    parser.add_argument("--rerank", type=bool_arg, default=True)
    parser.add_argument("--fill-missing-with-train-mean", type=bool_arg, default=False)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _predicted_table(mvp2_root: Path, split: str) -> Path:
    candidates = [
        mvp2_root / "predicted_conditions_10epoch_visual_handcrafted" / f"predicted_condition_table_{split}.csv",
        mvp2_root / "predicted_conditions" / f"predicted_condition_table_{split}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(mvp2_root.glob(f"**/predicted_condition_table_{split}.csv"))
    if not matches:
        raise FileNotFoundError(f"No predicted condition table for split={split} under {mvp2_root}")
    return matches[0]


def _sample(
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
) -> np.ndarray:
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
    return decode_latents(autoencoder, latents, latent_mean, latent_std)


def rerun(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    mvp2_root = resolve_repo_path(args.mvp2_root)
    mvp3_root = resolve_repo_path(args.mvp3_root)
    v3_root = resolve_repo_path(args.v3_root)
    table_path = _predicted_table(mvp2_root, args.split)
    pred_rows = [row for row in read_csv_rows(table_path) if row.get("split") == args.split][: int(args.max_conditions)]
    if not pred_rows:
        pred_rows = read_csv_rows(table_path)[: int(args.max_conditions)]
    schema = read_json(resolve_repo_path(args.v3_condition_schema))
    v3_model, v3_payload = load_diffusion_v3_checkpoint(args.v3_diffusion, str(device))
    v3_config = dict(v3_payload["config"])
    autoencoder, _ae = load_autoencoder_v2_checkpoint(args.v3_autoencoder, str(device))
    regressor, _reg = load_regressor_checkpoint(args.latent_descriptor_regressor, str(device))
    v3_model.to(device).eval()
    autoencoder.to(device).eval()
    regressor.to(device).eval()
    v3_stats = np.load(resolve_repo_path(Path(v3_config["latents_dir"])) / "latent_standardization_v2.npz")
    v3_mean = torch.from_numpy(np.asarray(v3_stats["latent_mean"], dtype=np.float32)).to(device)
    v3_std = torch.from_numpy(np.asarray(v3_stats["latent_std"], dtype=np.float32)).to(device)
    v3_diffusion = GaussianDiffusionV2(int(v3_config["timesteps"]), str(v3_config["beta_schedule"]), str(v3_config["prediction_target"]), device)
    v3_latent_shape = tuple(int(value) for value in v3_config["latent_shape"])
    descriptor_dim = len(schema["condition_columns"])
    v2_model, v2_payload = load_diffusion_v2_checkpoint(mvp3_root / "latent_diffusion_v2" / "checkpoints" / "ema_last.pt", str(device))
    v2_config = dict(v2_payload["config"])
    v2_schema = read_json(mvp3_root / "latents_v2" / "condition_schema_v2.json")
    v2_stats = np.load(mvp3_root / "latents_v2" / "latent_standardization_v2.npz")
    v2_mean = torch.from_numpy(np.asarray(v2_stats["latent_mean"], dtype=np.float32)).to(device)
    v2_std = torch.from_numpy(np.asarray(v2_stats["latent_std"], dtype=np.float32)).to(device)
    v2_diffusion = GaussianDiffusionV2(int(v2_config["timesteps"]), str(v2_config.get("beta_schedule", "cosine")), str(v2_config.get("prediction_target", "epsilon")), device)
    v2_latent_shape = tuple(int(value) for value in v2_config["latent_shape"])
    v2_model.to(device).eval()
    grid_rows: list[list[np.ndarray]] = []
    grid_titles: list[str] = []
    metrics: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    mean_row, mean_report = adapt_external_condition_row(pred_rows[0], schema, mode="mean", fill_missing_with_train_mean=True)
    adapter_reports.append(mean_report)
    for row in pred_rows:
        pred_v3_row, pred_report = adapt_external_condition_row(row, schema, mode="predicted", fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean))
        oracle_v3_row, oracle_report = adapt_external_condition_row(row, schema, mode="oracle", fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean))
        adapter_reports.extend([pred_report, oracle_report])
        pred_condition = condition_row_to_vector(pred_v3_row, schema)
        oracle_condition = condition_row_to_vector(oracle_v3_row, schema)
        mean_condition = condition_row_to_vector(mean_row, schema)
        v3_plain = _sample(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, None, pred_condition, v3_latent_shape, descriptor_dim, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), 0.0, device)
        v3_guided = _sample(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, regressor, pred_condition, v3_latent_shape, descriptor_dim, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
        v3_top, rerank_metrics = rerank_decoded_candidates(v3_guided, pred_v3_row, schema, int(args.keep_top_k)) if bool(args.rerank) else (v3_guided[: int(args.keep_top_k)], [])
        v3_oracle = _sample(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, regressor, oracle_condition, v3_latent_shape, descriptor_dim, int(args.keep_top_k), int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
        v3_mean_img = _sample(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, regressor, mean_condition, v3_latent_shape, descriptor_dim, 1, int(args.ddim_steps), float(args.guidance_scale), float(args.descriptor_guidance_weight), device)
        v2_row, _v2_report = adapt_external_condition_row(row, v2_schema, mode="predicted", fill_missing_with_train_mean=True)
        v2_condition = condition_row_to_vector(v2_row, v2_schema)
        v2_image = _sample(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, None, v2_condition, v2_latent_shape, len(v2_schema["condition_columns"]), 1, int(args.ddim_steps), 1.5, 0.0, device)[0]
        rheed = np.zeros((128, 128), dtype=np.float32)
        if row.get("cached_tensor_path", ""):
            try:
                rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"][-1, 0]
            except Exception:
                pass
        true_afm = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        with torch.no_grad():
            recon, _latent = autoencoder(torch.from_numpy(true_afm[None, None].astype(np.float32)).to(device))
        recon_image = recon[0, 0].detach().cpu().numpy()
        grid_rows.append([rheed, true_afm, recon_image, v2_image, v3_plain[0], v3_guided[0], v3_top[0], v3_oracle[0], v3_mean_img[0]])
        grid_titles.append(str(row.get("sample_id", row["row_id"])))
        for mode, images, target in (
            ("mvp3_v2_predicted", np.asarray([v2_image]), pred_v3_row),
            ("v3_predicted_plain", v3_plain, pred_v3_row),
            ("v3_predicted_guided", v3_guided, pred_v3_row),
            ("v3_predicted_reranked", v3_top, pred_v3_row),
            ("v3_oracle_reranked", v3_oracle, oracle_v3_row),
            ("v3_mean_condition", v3_mean_img, mean_row),
        ):
            for index, image in enumerate(images):
                metrics.append(_metric_row(mode, target, image, index, schema, rank=(index + 1 if "reranked" in mode else "")))
    grid_path = out_dir / "rheed_conditioned_v3_prior_grid.png"
    write_panel_grid(
        grid_path,
        grid_rows,
        ["RHEED final frame", "true AFM", "AE recon", "MVP-3 v2 predicted gen", "V3 predicted plain", "V3 predicted guided", "V3 reranked top1", "V3 oracle reranked", "V3 mean condition"],
        grid_titles,
    )
    write_csv_rows(out_dir / "rheed_conditioned_v3_metrics.csv", metrics)
    filled_by_mode: dict[str, list[str]] = {}
    for report in adapter_reports:
        mode = str(report.get("mode", ""))
        for name in report.get("filled_descriptors", []):
            filled_by_mode.setdefault(mode, [])
            if name not in filled_by_mode[mode]:
                filled_by_mode[mode].append(name)
    for mode in list(filled_by_mode):
        filled_by_mode[mode] = sorted(filled_by_mode[mode])
    unsafe_filled = sorted(
        {
            name
            for mode, names in filled_by_mode.items()
            if mode != "mean"
            for name in names
        }
    )
    adapter_text = [
        "# RHEED Condition Adapter Report",
        "",
        f"Predicted table: `{display_path(table_path)}`",
        f"Fill missing with train mean: `{bool(args.fill_missing_with_train_mean)}`",
        f"Filled predicted/oracle descriptors: `{unsafe_filled}`",
        f"Filled descriptors by mode: `{filled_by_mode}`",
        "Mappings use exact descriptor names only.",
    ]
    (out_dir / "condition_adapter_report.md").write_text("\n".join(adapter_text) + "\n", encoding="utf-8")
    stds = [float(row["generated_std"]) for row in metrics]
    summary = {
        "predicted_table": display_path(table_path),
        "comparison_grid": display_path(grid_path),
        "metrics": display_path(out_dir / "rheed_conditioned_v3_metrics.csv"),
        "condition_adapter_report": display_path(out_dir / "condition_adapter_report.md"),
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "adapter_filled_predicted_oracle_descriptors": unsafe_filled,
        "adapter_filled_descriptors_by_mode": filled_by_mode,
        "note": "This remains two-stage RHEED-conditioned generation and does not retrain the RHEED encoder.",
    }
    write_json(out_dir / "rheed_conditioned_v3_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = rerun(args)
    print(f"Wrote RHEED-conditioned v3 outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
