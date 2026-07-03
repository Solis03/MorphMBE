"""Rerun RHEED-conditioned generation with the v4 height-calibrated prior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import adapt_external_condition_row, bool_arg, condition_row_to_vector, finite_float
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.sample_calibrated_v2_v3 import _candidate_rows, _decode, _diffusion_from_config
from rheed2morph.generative.sample_afm_prior_v3 import _metric_row
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v3 import load_diffusion_v3_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerun RHEED-conditioned generation with v4 calibrated prior.")
    parser.add_argument("--mvp2-root", type=Path, required=True)
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--mvp4-root", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--primary-generator", choices=["auto", "calibrated_v2", "calibrated_v3"], default="auto")
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--v2-diffusion", type=Path, required=True)
    parser.add_argument("--v3-diffusion", type=Path, required=True)
    parser.add_argument("--v4-diffusion", type=Path, default=None)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--descriptor-guidance-weight", type=float, default=0.03)
    parser.add_argument("--calibration-mode", type=str, default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=bool_arg, default=True)
    parser.add_argument("--fill-missing-with-train-mean", type=bool_arg, nargs="?", const=True, default=False)
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


def _latent_stats(config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], GaussianDiffusionV2]:
    stats = np.load(resolve_repo_path(Path(config["latents_dir"])) / "latent_standardization_v2.npz")
    mean = torch.from_numpy(np.asarray(stats["latent_mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(stats["latent_std"], dtype=np.float32)).to(device)
    return mean, std, tuple(int(value) for value in config["latent_shape"]), _diffusion_from_config(config, device)


def rerun(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    mvp2_root = resolve_repo_path(args.mvp2_root)
    mvp3_root = resolve_repo_path(args.mvp3_root)
    mvp4_root = resolve_repo_path(args.mvp4_root)
    v4_root = resolve_repo_path(args.v4_root)
    table_path = _predicted_table(mvp2_root, args.split)
    pred_rows = [row for row in read_csv_rows(table_path) if row.get("split") == args.split][: int(args.max_conditions)]
    if not pred_rows:
        pred_rows = read_csv_rows(table_path)[: int(args.max_conditions)]
    schema = read_json(resolve_repo_path(args.condition_schema))
    v2_schema = read_json(mvp3_root / "latents_v2" / "condition_schema_v2.json")
    v2_model, v2_payload = load_diffusion_v2_checkpoint(args.v2_diffusion, str(device))
    v3_model, v3_payload = load_diffusion_v3_checkpoint(args.v3_diffusion, str(device))
    autoencoder, _ae = load_autoencoder_v2_checkpoint(args.autoencoder, str(device))
    v2_model.to(device).eval()
    v3_model.to(device).eval()
    autoencoder.to(device).eval()
    v2_mean, v2_std, v2_latent_shape, v2_diffusion = _latent_stats(dict(v2_payload["config"]), device)
    v3_mean, v3_std, v3_latent_shape, v3_diffusion = _latent_stats(dict(v3_payload["config"]), device)
    scale_summary_path = v4_root / "height_diagnosis" / "height_scale_summary.json"
    scale_bounds = read_json(scale_summary_path).get("scale_bounds", {}) if scale_summary_path.is_file() else {"scale_low": 0.1, "scale_high": 100.0, "scale_median": 5.0}
    primary = "calibrated_v2" if args.primary_generator == "auto" else args.primary_generator
    grid_rows: list[list[np.ndarray]] = []
    grid_titles: list[str] = []
    metrics: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    mean_row, mean_report = adapt_external_condition_row(pred_rows[0], schema, mode="mean", fill_missing_with_train_mean=True)
    adapter_reports.append(mean_report)
    for row in pred_rows:
        pred_v3_row, pred_report = adapt_external_condition_row(row, schema, mode="predicted", fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean))
        oracle_v3_row, oracle_report = adapt_external_condition_row(row, schema, mode="oracle", fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean))
        pred_v2_row, _v2_report = adapt_external_condition_row(row, v2_schema, mode="predicted", fill_missing_with_train_mean=True)
        adapter_reports.extend([pred_report, oracle_report])
        pred_v2_condition = condition_row_to_vector(pred_v2_row, v2_schema)
        pred_v3_condition = condition_row_to_vector(pred_v3_row, schema)
        oracle_condition = condition_row_to_vector(oracle_v3_row, schema)
        mean_condition = condition_row_to_vector(mean_row, schema)
        v2_uncal = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, pred_v2_condition, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), device)
        v3_uncal = _decode(v3_model, autoencoder, v3_diffusion, v3_mean, v3_std, v3_latent_shape, pred_v3_condition, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), device)
        v2_top, v2_metrics, v2_cal = _candidate_rows("v4_calibrated_v2_predicted", pred_v3_row, v2_uncal, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), False)
        v3_top, v3_metrics, v3_cal = _candidate_rows("v4_calibrated_v3_predicted", pred_v3_row, v3_uncal, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), False)
        oracle_uncal = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, pred_v2_condition, int(args.keep_top_k), int(args.ddim_steps), float(args.guidance_scale), device)
        oracle_top, oracle_metrics, oracle_cal = _candidate_rows("v4_calibrated_oracle", oracle_v3_row, oracle_uncal, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), False)
        mean_uncal = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, pred_v2_condition * 0.0, 1, int(args.ddim_steps), float(args.guidance_scale), device)
        mean_top, mean_metrics, mean_cal = _candidate_rows("v4_mean_condition", mean_row, mean_uncal, schema, scale_bounds, args.calibration_mode, 1, False)
        metrics.extend(v2_metrics + v3_metrics + oracle_metrics + mean_metrics)
        calibration_rows.extend(v2_cal + v3_cal + oracle_cal + mean_cal)
        true_afm = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        true_physical = load_height_array(resolve_repo_path(Path(row.get("descriptor_height_path", row.get("network_input_path", "")))))
        with torch.no_grad():
            recon, _latent = autoencoder(torch.from_numpy(true_afm[None, None].astype(np.float32)).to(device))
        recon_image = recon[0, 0].detach().cpu().numpy()
        rheed = np.zeros_like(true_afm)
        if row.get("cached_tensor_path", ""):
            try:
                rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"][-1, 0]
            except Exception:
                pass
        primary_top = v2_top if primary == "calibrated_v2" else v3_top
        grid_rows.append([rheed, true_physical, recon_image, v2_uncal[0], v3_uncal[0], primary_top[0], primary_top[min(1, primary_top.shape[0] - 1)], oracle_top[0], mean_top[0]])
        grid_titles.append(str(row.get("sample_id", row["row_id"])))
        metrics.append(_metric_row("mvp3_v2_predicted_uncalibrated", pred_v3_row, v2_uncal[0], 0, schema))
        metrics.append(_metric_row("mvp4_v3_predicted_uncalibrated", pred_v3_row, v3_uncal[0], 0, schema))
    grid_path = out_dir / "rheed_conditioned_v4_prior_grid.png"
    write_panel_grid(
        grid_path,
        grid_rows,
        ["RHEED final frame", "true AFM", "AE recon", "MVP-3 v2 predicted gen", "MVP-4 v3 predicted gen", "V4 predicted calibrated top1", "V4 predicted calibrated top2", "V4 oracle calibrated", "V4 mean condition"],
        grid_titles,
    )
    write_csv_rows(out_dir / "rheed_conditioned_v4_metrics.csv", metrics)
    write_csv_rows(out_dir / "rheed_conditioned_v4_calibration_metrics.csv", calibration_rows)
    filled_by_mode: dict[str, list[str]] = {}
    for report in adapter_reports:
        mode = str(report.get("mode", ""))
        for name in report.get("filled_descriptors", []):
            filled_by_mode.setdefault(mode, [])
            if name not in filled_by_mode[mode]:
                filled_by_mode[mode].append(name)
    for mode in list(filled_by_mode):
        filled_by_mode[mode] = sorted(filled_by_mode[mode])
    unsafe_filled = sorted({name for mode, names in filled_by_mode.items() if mode != "mean" for name in names})
    adapter_text = [
        "# RHEED V4 Condition Adapter Report",
        "",
        f"Predicted table: `{display_path(table_path)}`",
        f"Fill missing with train mean: `{bool(args.fill_missing_with_train_mean)}`",
        f"Filled predicted/oracle descriptors: `{unsafe_filled}`",
        f"Filled descriptors by mode: `{filled_by_mode}`",
        "Mappings use exact descriptor names only.",
    ]
    (out_dir / "condition_adapter_report.md").write_text("\n".join(adapter_text) + "\n", encoding="utf-8")
    stds = [finite_float(row.get("normalized_std", row.get("generated_std", "nan"))) for row in metrics]
    stds = [value for value in stds if np.isfinite(value)]
    summary = {
        "predicted_table": display_path(table_path),
        "primary_generator": primary,
        "comparison_grid": display_path(grid_path),
        "metrics": display_path(out_dir / "rheed_conditioned_v4_metrics.csv"),
        "calibration_metrics": display_path(out_dir / "rheed_conditioned_v4_calibration_metrics.csv"),
        "condition_adapter_report": display_path(out_dir / "condition_adapter_report.md"),
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "adapter_filled_predicted_oracle_descriptors": unsafe_filled,
        "adapter_filled_descriptors_by_mode": filled_by_mode,
        "note": "This remains two-stage RHEED-conditioned generation and does not retrain the RHEED encoder.",
    }
    write_json(out_dir / "rheed_conditioned_v4_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = rerun(args)
    print(f"Wrote RHEED-conditioned v4 outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"primary_generator={summary['primary_generator']} nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
