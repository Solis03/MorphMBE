"""Sample existing v2/v3 AFM priors with physical height-scale calibration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

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
from rheed2morph.generative.condition_control_v3_utils import condition_row_to_vector, finite_float
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.height_calibration_v4 import calibrate_generated_afm, descriptor_error
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v3 import load_diffusion_v3_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


ROUGHNESS_NAMES = ("rq", "ra", "robust_range")
TEXTURE_NAMES = ("psd_slope", "autocorrelation_length_px", "gradient_anisotropy", "island_count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample calibrated v2/v3 AFM priors.")
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--mvp4-root", type=Path, required=True)
    parser.add_argument("--v2-diffusion", type=Path, required=True)
    parser.add_argument("--v3-diffusion", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=16)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--calibration-mode", type=str, default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_scale_bounds(out_dir: Path, mvp3_root: Path) -> dict[str, float]:
    candidates = [
        out_dir.parent / "height_diagnosis" / "height_scale_summary.json",
        mvp3_root / "height_diagnosis" / "height_scale_summary.json",
    ]
    for path in candidates:
        if path.is_file():
            summary = read_json(path)
            bounds = summary.get("scale_bounds", {})
            if bounds:
                return {key: float(value) for key, value in bounds.items() if isinstance(value, (int, float))}
    return {"scale_low": 0.1, "scale_high": 100.0, "scale_median": 5.0}


def _diffusion_from_config(config: Mapping[str, Any], device: torch.device) -> GaussianDiffusionV2:
    return GaussianDiffusionV2(
        timesteps=int(config["timesteps"]),
        beta_schedule=str(config.get("beta_schedule", "cosine")),
        prediction_target=str(config.get("prediction_target", "epsilon")),
        device=device,
    )


@torch.no_grad()
def _decode(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: GaussianDiffusionV2,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    latent_shape: tuple[int, int, int],
    condition: np.ndarray,
    count: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> np.ndarray:
    cond = torch.from_numpy(np.repeat(condition[None], count, axis=0).astype(np.float32)).to(device)
    latents = diffusion.sample_ddim(model, (count, *latent_shape), cond, steps=int(steps), guidance_scale=float(guidance_scale))
    raw = latents * latent_std + latent_mean
    decoded = autoencoder.decode(raw).detach().cpu().numpy()
    return decoded[:, 0]


def _ae_recon(autoencoder: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        recon, _latent = autoencoder(torch.from_numpy(image[None, None].astype(np.float32)).to(device))
    return recon[0, 0].detach().cpu().numpy()


def _score_candidate(before: Mapping[str, float], after: Mapping[str, float], target: Mapping[str, str], schema: Mapping[str, Any]) -> dict[str, float]:
    rough = descriptor_error(after, target, ROUGHNESS_NAMES, schema.get("descriptor_train_std", {}))
    texture = descriptor_error(after, target, TEXTURE_NAMES, schema.get("descriptor_train_std", {}))
    if not math.isfinite(rough):
        rough = 100.0
    if not math.isfinite(texture):
        texture = 0.0
    normalized_std = finite_float(before.get("rq", float("nan")))
    gradient_std = finite_float(before.get("gradient_std", float("nan")))
    psd_high = finite_float(before.get("psd_high_power", float("nan")))
    realness_penalty = 0.0
    if not math.isfinite(normalized_std) or normalized_std < 1e-4:
        realness_penalty += 100.0
    elif normalized_std < 0.2:
        realness_penalty += 0.5 * (0.2 - normalized_std) / 0.2
    visual_richness_bonus = 0.0
    if math.isfinite(gradient_std):
        visual_richness_bonus += min(max(gradient_std, 0.0), 1.0)
    if math.isfinite(psd_high):
        visual_richness_bonus += min(max(psd_high / 8.0, 0.0), 1.0)
    score = rough + 0.8 * texture + 0.3 * realness_penalty - 0.2 * visual_richness_bonus
    return {
        "score": float(score),
        "roughness_error": float(rough),
        "texture_error": float(texture),
        "realness_penalty": float(realness_penalty),
        "visual_richness_bonus": float(visual_richness_bonus),
    }


def _candidate_rows(
    prior: str,
    target_row: dict[str, str],
    images: np.ndarray,
    schema: Mapping[str, Any],
    scale_bounds: Mapping[str, float],
    calibration_mode: str,
    keep_top_k: int,
    allow_extrapolation: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    scored: list[tuple[float, int, np.ndarray]] = []
    flattened = images.reshape(images.shape[0], -1)
    for index, image in enumerate(images):
        before = compute_afm_descriptors_v2(image)
        calibrated, result = calibrate_generated_afm(
            image,
            target_row,
            schema,
            calibration_mode=calibration_mode,
            scale_bounds=scale_bounds,
            allow_extrapolation=allow_extrapolation,
        )
        after = result.descriptors_after
        scores = _score_candidate(before, after, target_row, schema)
        if index > 0:
            distances = np.mean((flattened[:index] - flattened[index][None]) ** 2, axis=1)
            scores["duplicate_penalty"] = float(np.mean(distances < 1e-7))
            scores["score"] += 0.2 * scores["duplicate_penalty"]
        else:
            scores["duplicate_penalty"] = 0.0
        scored.append((scores["score"], index, calibrated))
        base: dict[str, Any] = {
            "prior": prior,
            "row_id": target_row.get("row_id", ""),
            "sample_id": target_row.get("sample_id", ""),
            "group_id": target_row.get("group_id", ""),
            "candidate_index": index,
            "calibration_mode": calibration_mode,
            "scale_nm_per_unit": result.scale_nm_per_unit,
            "offset_nm": result.offset_nm,
            "scale_clamped": result.clamped,
            "unclamped_scale_nm_per_unit": result.unclamped_scale_nm_per_unit,
            **scores,
            "normalized_std": float(np.std(image)),
            "calibrated_std_nm": float(np.std(calibrated)),
            "nonconstant": bool(np.std(image) > 1e-4),
        }
        for name in schema.get("descriptor_columns", []):
            if target_row.get(name, "") != "":
                base[f"requested_{name}"] = finite_float(target_row[name])
            if name in before:
                base[f"uncalibrated_{name}"] = finite_float(before[name])
            if name in after:
                base[f"calibrated_{name}"] = finite_float(after[name])
        metric_rows.append(base)
        calibration_rows.append(
            {
                "prior": prior,
                "row_id": target_row.get("row_id", ""),
                "candidate_index": index,
                "scale_nm_per_unit": result.scale_nm_per_unit,
                "offset_nm": result.offset_nm,
                "clamped": result.clamped,
                "calibration_error": result.calibration_error,
                "target_rq": finite_float(target_row.get("rq", "nan")),
                "before_rq": finite_float(before.get("rq", "nan")),
                "after_rq": finite_float(after.get("rq", "nan")),
                "target_ra": finite_float(target_row.get("ra", "nan")),
                "before_ra": finite_float(before.get("ra", "nan")),
                "after_ra": finite_float(after.get("ra", "nan")),
                "target_robust_range": finite_float(target_row.get("robust_range", "nan")),
                "before_robust_range": finite_float(before.get("robust_range", "nan")),
                "after_robust_range": finite_float(after.get("robust_range", "nan")),
            }
        )
    order = [index for _score, index, _image in sorted(scored, key=lambda item: item[0])[: int(keep_top_k)]]
    for rank, candidate_index in enumerate(order, start=1):
        metric_rows[candidate_index]["rank"] = rank
        calibration_rows[candidate_index]["rank"] = rank
    top = np.stack([images[index] for index in order], axis=0)
    top_calibrated = np.stack([scored[[idx for _s, idx, _img in scored].index(index)][2] for index in order], axis=0)
    for row in metric_rows:
        row.setdefault("rank", "")
    for row in calibration_rows:
        row.setdefault("rank", "")
    return top_calibrated, metric_rows, calibration_rows


def _mae(rows: list[dict[str, Any]], prior: str, calibrated: bool, name: str, top_only: bool = False) -> float:
    values = []
    key = f"{'calibrated' if calibrated else 'uncalibrated'}_{name}"
    for row in rows:
        if row.get("prior") != prior:
            continue
        if top_only and str(row.get("rank", "")) != "1":
            continue
        gen = finite_float(row.get(key, "nan"))
        req = finite_float(row.get(f"requested_{name}", "nan"))
        if math.isfinite(gen) and math.isfinite(req):
            values.append(abs(gen - req))
    return float(np.mean(values)) if values else float("nan")


def sample_calibrated(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    mvp3_root = resolve_repo_path(args.mvp3_root)
    mvp4_root = resolve_repo_path(args.mvp4_root)
    schema = read_json(resolve_repo_path(args.condition_schema))
    v3_rows = read_csv_rows(resolve_repo_path(args.condition_table))
    selected = [row for row in v3_rows if row.get("split") == args.split][: int(args.max_conditions)]
    if not selected:
        selected = v3_rows[: int(args.max_conditions)]
    v2_rows = read_csv_rows(mvp3_root / "latents_v2" / "condition_table_v2.csv")
    v2_schema = read_json(mvp3_root / "latents_v2" / "condition_schema_v2.json")
    v2_by_row = {row["row_id"]: row for row in v2_rows}
    v2_model, v2_payload = load_diffusion_v2_checkpoint(args.v2_diffusion, str(device))
    v3_model, v3_payload = load_diffusion_v3_checkpoint(args.v3_diffusion, str(device))
    autoencoder, _ae = load_autoencoder_v2_checkpoint(args.autoencoder, str(device))
    v2_model.to(device).eval()
    v3_model.to(device).eval()
    autoencoder.to(device).eval()
    v2_config = dict(v2_payload["config"])
    v3_config = dict(v3_payload["config"])
    v2_stats = np.load(resolve_repo_path(Path(v2_config["latents_dir"])) / "latent_standardization_v2.npz")
    v3_stats = np.load(resolve_repo_path(Path(v3_config["latents_dir"])) / "latent_standardization_v2.npz")
    v2_mean = torch.from_numpy(np.asarray(v2_stats["latent_mean"], dtype=np.float32)).to(device)
    v2_std = torch.from_numpy(np.asarray(v2_stats["latent_std"], dtype=np.float32)).to(device)
    v3_mean = torch.from_numpy(np.asarray(v3_stats["latent_mean"], dtype=np.float32)).to(device)
    v3_std = torch.from_numpy(np.asarray(v3_stats["latent_std"], dtype=np.float32)).to(device)
    v2_diffusion = _diffusion_from_config(v2_config, device)
    v3_diffusion = _diffusion_from_config(v3_config, device)
    v2_latent_shape = tuple(int(value) for value in v2_config["latent_shape"])
    v3_latent_shape = tuple(int(value) for value in v3_config["latent_shape"])
    scale_bounds = _load_scale_bounds(out_dir, mvp3_root)
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    grid_rows: list[list[np.ndarray]] = []
    grid_titles: list[str] = []
    all_images: list[np.ndarray] = []
    all_modes: list[str] = []
    for row in selected:
        if row["row_id"] not in v2_by_row:
            continue
        v2_condition = build_condition_matrix_v2(v2_rows, [row["row_id"]], v2_schema)[0]
        v3_condition = condition_row_to_vector(row, schema)
        v2_images = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, v2_condition, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), device)
        v3_images = _decode(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, v3_latent_shape, v3_condition, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), device)
        v2_top, v2_metrics, v2_calibration = _candidate_rows("v2", row, v2_images, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), bool(args.allow_extrapolation))
        v3_top, v3_metrics, v3_calibration = _candidate_rows("v3", row, v3_images, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), bool(args.allow_extrapolation))
        metric_rows.extend(v2_metrics)
        metric_rows.extend(v3_metrics)
        calibration_rows.extend(v2_calibration)
        calibration_rows.extend(v3_calibration)
        true_physical = load_height_array(resolve_repo_path(Path(row.get("descriptor_height_path", row.get("network_input_path", "")))))
        true_network = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        recon = _ae_recon(autoencoder, true_network, device)
        grid_rows.append([true_physical, recon, v2_images[0], v2_top[0], v3_images[0], v3_top[0], v3_top[min(1, v3_top.shape[0] - 1)]])
        grid_titles.append(str(row.get("sample_id", row["row_id"])))
        all_images.extend([*v2_images, *v3_images, *v2_top, *v3_top])
        all_modes.extend(["v2_uncalibrated"] * len(v2_images) + ["v3_uncalibrated"] * len(v3_images) + ["v2_calibrated_top"] * len(v2_top) + ["v3_calibrated_top"] * len(v3_top))
    grid_path = out_dir / f"calibrated_v2_v3_oracle_grid_{args.split}.png"
    write_panel_grid(
        grid_path,
        grid_rows,
        ["true AFM", "AE recon", "v2 uncal", "v2 calibrated top1", "v3 uncal", "v3 calibrated top1", "v3 reranked"],
        grid_titles,
    )
    if grid_rows:
        write_panel_grid(out_dir / "roughness_calibration_examples.png", [[row[2], row[3], row[4], row[5]] for row in grid_rows[:4]], ["v2 before", "v2 after", "v3 before", "v3 after"], grid_titles[:4])
        worst = sorted(metric_rows, key=lambda item: finite_float(item.get("score", "nan")), reverse=True)[: min(4, len(metric_rows))]
        failure_panels = []
        for row in worst:
            source = v3_rows[0]
            try:
                failure_panels.append(load_height_array(resolve_repo_path(Path(source.get("network_input_path", "")))))
            except Exception:
                failure_panels.append(np.zeros((128, 128), dtype=np.float32))
        write_panel_grid(out_dir / "calibration_failure_cases.png", [failure_panels], [str(row.get("prior", "")) for row in worst])
    write_csv_rows(out_dir / "calibrated_generation_metrics.csv", metric_rows)
    write_csv_rows(out_dir / "height_calibration_metrics_v4.csv", calibration_rows)
    np.savez_compressed(out_dir / "generated_candidates_calibrated_v2_v3.npz", images=np.asarray(all_images, dtype=np.float32), modes=np.asarray(all_modes))
    stds = [finite_float(row.get("normalized_std", "nan")) for row in metric_rows]
    clamp_rate = float(np.mean([bool(row.get("scale_clamped", False)) for row in metric_rows])) if metric_rows else 0.0
    summary = {
        "split": args.split,
        "condition_count": len(selected),
        "num_samples_per_condition": int(args.num_samples_per_condition),
        "keep_top_k": int(args.keep_top_k),
        "ddim_steps": int(args.ddim_steps),
        "guidance_scale": float(args.guidance_scale),
        "calibration_mode": args.calibration_mode,
        "scale_bounds": scale_bounds,
        "scale_clamp_rate": clamp_rate,
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "generated_normalized_std_mean": float(np.nanmean(stds)) if stds else 0.0,
        "v2_rq_mae_before": _mae(metric_rows, "v2", False, "rq"),
        "v2_rq_mae_after_top1": _mae(metric_rows, "v2", True, "rq", top_only=True),
        "v3_rq_mae_before": _mae(metric_rows, "v3", False, "rq"),
        "v3_rq_mae_after_top1": _mae(metric_rows, "v3", True, "rq", top_only=True),
        "v2_ra_mae_before": _mae(metric_rows, "v2", False, "ra"),
        "v2_ra_mae_after_top1": _mae(metric_rows, "v2", True, "ra", top_only=True),
        "v2_robust_range_mae_before": _mae(metric_rows, "v2", False, "robust_range"),
        "v2_robust_range_mae_after_top1": _mae(metric_rows, "v2", True, "robust_range", top_only=True),
        "recommended_primary_prior": "calibrated_v2",
        "oracle_grid": display_path(grid_path),
        "calibrated_generation_metrics": display_path(out_dir / "calibrated_generation_metrics.csv"),
        "height_calibration_metrics": display_path(out_dir / "height_calibration_metrics_v4.csv"),
    }
    if summary["v3_rq_mae_after_top1"] < summary["v2_rq_mae_after_top1"] and summary["generated_normalized_std_mean"] < 0.45:
        summary["recommended_primary_prior"] = "calibrated_v3_auxiliary"
    write_json(out_dir / "calibrated_generation_summary.json", summary)
    report = [
        "# Calibrated V2/V3 Sampling Report",
        "",
        f"Recommended primary prior: `{summary['recommended_primary_prior']}`",
        f"V2 Rq MAE before/after top1: `{summary['v2_rq_mae_before']:.6g}` / `{summary['v2_rq_mae_after_top1']:.6g}`",
        f"V3 Rq MAE before/after top1: `{summary['v3_rq_mae_before']:.6g}` / `{summary['v3_rq_mae_after_top1']:.6g}`",
        f"Scale clamp rate: `{summary['scale_clamp_rate']:.3f}`",
    ]
    (out_dir / "calibration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_calibrated(args)
    print(f"Wrote calibrated v2/v3 samples to {display_path(resolve_repo_path(args.out))}")
    print(f"recommended_primary_prior={summary['recommended_primary_prior']} nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
