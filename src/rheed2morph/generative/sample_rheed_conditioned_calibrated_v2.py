"""Generate RHEED-conditioned samples with the MVP-5 calibrated_v2 prior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import adapt_external_condition_row, condition_row_to_vector, finite_float
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.sample_calibrated_v2_v3 import _candidate_rows, _decode, _diffusion_from_config
from rheed2morph.generative.sample_afm_prior_v3 import _metric_row
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample RHEED-conditioned calibrated_v2 AFM prior.")
    parser.add_argument("--mvp5-root", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--v2-diffusion", type=Path, required=True)
    parser.add_argument("--v3-diffusion", type=Path, default=None)
    parser.add_argument("--predicted-condition-table", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--primary-generator", choices=["calibrated_v2"], default="calibrated_v2")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--calibration-mode", type=str, default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _latent_stats(config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], GaussianDiffusionV2]:
    stats = np.load(resolve_repo_path(Path(config["latents_dir"])) / "latent_standardization_v2.npz")
    mean = torch.from_numpy(np.asarray(stats["latent_mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(stats["latent_std"], dtype=np.float32)).to(device)
    return mean, std, tuple(int(value) for value in config["latent_shape"]), _diffusion_from_config(config, device)


def _load_scale_bounds(mvp5_root: Path) -> dict[str, float]:
    path = mvp5_root / "height_diagnosis" / "height_scale_summary.json"
    if path.is_file():
        bounds = read_json(path).get("scale_bounds", {})
        if bounds:
            return {key: float(value) for key, value in bounds.items() if isinstance(value, (int, float))}
    return {"scale_low": 0.1, "scale_high": 100.0, "scale_median": 5.0}


def _mock_outputs(args: argparse.Namespace, out_dir: Path, rows: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
    grid_rows = []
    titles = []
    metric_rows = []
    for row in rows:
        rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"]
        true = load_height_array(resolve_repo_path(Path(row.get("descriptor_height_path", row.get("network_input_path", "")))))
        pred_image = true - float(np.mean(true))
        grid_rows.append([rheed[-1, 0], rheed[:, 0].mean(axis=0), true, true, np.zeros_like(true), pred_image, pred_image, true, np.zeros_like(true)])
        titles.append(str(row.get("sample_id", row["row_id"])))
        metric_rows.append({"prior": "mock_calibrated_v2_predicted", "row_id": row["row_id"], "sample_id": row.get("sample_id", ""), "rank": 1, "normalized_std": float(np.std(pred_image)), "nonconstant": bool(np.std(pred_image) > 1e-4)})
    grid_path = out_dir / f"rheed_conditioned_calibrated_v2_grid_{args.split}.png"
    write_panel_grid(grid_path, grid_rows, ["RHEED final", "RHEED temporal", "true AFM", "AE recon", "old MVP-2 gen", "MVP-6 top1", "MVP-6 top2", "oracle", "mean"], titles)
    write_csv_rows(out_dir / "generation_metrics_mvp6.csv", metric_rows)
    write_panel_grid(out_dir / "failure_cases_grid_mvp6.png", [row[:3] for row in grid_rows[:2]], ["RHEED final", "RHEED temporal", "true AFM"], titles[:2])
    summary = {"mock": True, "primary_generator": "calibrated_v2", "grid": display_path(grid_path), "generated_nonconstant_rate": 1.0 if metric_rows else 0.0, "generated_std_mean": float(np.mean([row["normalized_std"] for row in metric_rows])) if metric_rows else 0.0}
    write_json(out_dir / "generation_summary_mvp6.json", summary)
    return summary


def sample(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mvp5_root = resolve_repo_path(args.mvp5_root)
    schema = read_json(resolve_repo_path(args.condition_schema))
    v2_schema = read_json(resolve_repo_path(Path("reports/afm_prior_v2/20260703_052537/latents_v2/condition_schema_v2.json")))
    rows = [row for row in read_csv_rows(resolve_repo_path(args.predicted_condition_table)) if row.get("split", args.split) == args.split]
    if not rows:
        rows = read_csv_rows(resolve_repo_path(args.predicted_condition_table))
    rows = rows[: int(args.max_conditions)]
    if args.mock:
        return _mock_outputs(args, out_dir, rows, schema)
    device = resolve_torch_device(args.device)
    v2_model, v2_payload = load_diffusion_v2_checkpoint(args.v2_diffusion, str(device))
    autoencoder, _ae = load_autoencoder_v2_checkpoint(args.autoencoder, str(device))
    v2_model.to(device).eval()
    autoencoder.to(device).eval()
    v2_mean, v2_std, v2_latent_shape, v2_diffusion = _latent_stats(dict(v2_payload["config"]), device)
    scale_bounds = _load_scale_bounds(mvp5_root)
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    grid_rows: list[list[np.ndarray]] = []
    grid_titles: list[str] = []
    mean_v3, mean_report = adapt_external_condition_row(rows[0], schema, mode="mean", fill_missing_with_train_mean=True)
    mean_report["schema"] = "v3"
    mean_v2, _ = adapt_external_condition_row(rows[0], v2_schema, mode="mean", fill_missing_with_train_mean=True)
    adapter_reports.append(mean_report)
    for row in rows:
        pred_v3, pred_report = adapt_external_condition_row(row, schema, mode="predicted", fill_missing_with_train_mean=True)
        pred_report["schema"] = "v3"
        oracle_v3, oracle_report = adapt_external_condition_row(row, schema, mode="oracle", fill_missing_with_train_mean=True)
        oracle_report["schema"] = "v3"
        pred_v2, pred_v2_report = adapt_external_condition_row(row, v2_schema, mode="predicted", fill_missing_with_train_mean=True)
        pred_v2_report["schema"] = "v2"
        oracle_v2, _ = adapt_external_condition_row(row, v2_schema, mode="oracle", fill_missing_with_train_mean=True)
        adapter_reports.extend([pred_report, oracle_report, pred_v2_report])
        pred_condition = condition_row_to_vector(pred_v2, v2_schema)
        oracle_condition = condition_row_to_vector(oracle_v2, v2_schema)
        mean_condition = condition_row_to_vector(mean_v2, v2_schema)
        pred_images = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, pred_condition, int(args.num_samples_per_condition), int(args.ddim_steps), float(args.guidance_scale), device)
        oracle_images = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, oracle_condition, max(1, int(args.keep_top_k)), int(args.ddim_steps), float(args.guidance_scale), device)
        mean_images = _decode(v2_model, autoencoder, v2_diffusion, v2_mean, v2_std, v2_latent_shape, mean_condition, 1, int(args.ddim_steps), float(args.guidance_scale), device)
        pred_top, pred_metrics, pred_cal = _candidate_rows("mvp6_predicted_calibrated_v2", pred_v3, pred_images, schema, scale_bounds, args.calibration_mode, int(args.keep_top_k), False)
        oracle_top, oracle_metrics, oracle_cal = _candidate_rows("mvp6_oracle_calibrated_v2", oracle_v3, oracle_images, schema, scale_bounds, args.calibration_mode, max(1, int(args.keep_top_k)), False)
        mean_top, mean_metrics, mean_cal = _candidate_rows("mvp6_mean_condition_calibrated_v2", mean_v3, mean_images, schema, scale_bounds, args.calibration_mode, 1, False)
        metric_rows.extend(pred_metrics + oracle_metrics + mean_metrics)
        calibration_rows.extend(pred_cal + oracle_cal + mean_cal)
        true_network = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        true_physical = load_height_array(resolve_repo_path(Path(row.get("descriptor_height_path", row.get("network_input_path", "")))))
        with torch.no_grad():
            recon, _latent = autoencoder(torch.from_numpy(true_network[None, None].astype(np.float32)).to(device))
        recon_image = recon[0, 0].detach().cpu().numpy()
        try:
            rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"]
            final = rheed[-1, 0]
            temporal = rheed[:, 0].mean(axis=0)
        except Exception:
            final = np.zeros_like(true_network)
            temporal = np.zeros_like(true_network)
        old = np.zeros_like(true_network)
        grid_rows.append([final, temporal, true_physical, recon_image, old, pred_top[0], pred_top[min(1, len(pred_top) - 1)], oracle_top[0], mean_top[0]])
        grid_titles.append(str(row.get("sample_id", row["row_id"])))
        metric_rows.append(_metric_row("mvp6_predicted_uncalibrated_v2_top0", pred_v3, pred_images[0], 0, schema))
    grid_path = out_dir / f"rheed_conditioned_calibrated_v2_grid_{args.split}.png"
    write_panel_grid(
        grid_path,
        grid_rows,
        ["RHEED final frame", "RHEED temporal summary", "true AFM", "AE recon", "old MVP-2 gen", "MVP-6 predicted top1", "MVP-6 predicted top2", "oracle calibrated_v2", "mean calibrated_v2"],
        grid_titles,
    )
    write_csv_rows(out_dir / "generation_metrics_mvp6.csv", metric_rows)
    write_csv_rows(out_dir / "height_calibration_metrics_mvp6.csv", calibration_rows)
    if grid_rows:
        write_panel_grid(
            out_dir / "failure_cases_grid_mvp6.png",
            [[row[0], row[1], row[2], row[3], row[5]] for row in grid_rows[: min(4, len(grid_rows))]],
            ["RHEED final", "RHEED temporal", "true AFM", "AE recon", "MVP-6 top1"],
            grid_titles[: min(4, len(grid_titles))],
        )
    stds = [finite_float(row.get("normalized_std", "nan")) for row in metric_rows]
    stds = [value for value in stds if np.isfinite(value)]
    filled_v3_pred_oracle = sorted(
        {
            name
            for report in adapter_reports
            if report.get("schema") == "v3" and report.get("mode") != "mean"
            for name in report.get("filled_descriptors", [])
        }
    )
    filled_by_schema_mode: dict[str, list[str]] = {}
    for report in adapter_reports:
        key = f"{report.get('schema', 'unknown')}:{report.get('mode', '')}"
        values = filled_by_schema_mode.setdefault(key, [])
        for name in report.get("filled_descriptors", []):
            if name not in values:
                values.append(name)
    for key in list(filled_by_schema_mode):
        filled_by_schema_mode[key] = sorted(filled_by_schema_mode[key])
    summary = {
        "primary_generator": "calibrated_v2",
        "predicted_condition_table": display_path(resolve_repo_path(args.predicted_condition_table)),
        "grid": display_path(grid_path),
        "generation_metrics": display_path(out_dir / "generation_metrics_mvp6.csv"),
        "calibration_metrics": display_path(out_dir / "height_calibration_metrics_mvp6.csv"),
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "adapter_filled_predicted_oracle_v3_descriptors": filled_v3_pred_oracle,
        "adapter_filled_descriptors_by_schema_mode": filled_by_schema_mode,
        "calibration_mode": args.calibration_mode,
        "note": "Generated AFM maps are representative morphology conditioned on predicted descriptors, not exact pixel-level reconstruction.",
    }
    write_json(out_dir / "generation_summary_mvp6.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample(args)
    print(f"Wrote MVP-6 calibrated_v2 RHEED-conditioned samples to {display_path(resolve_repo_path(args.out))}")
    print(f"nonconstant_rate={summary['generated_nonconstant_rate']:.3f} primary={summary['primary_generator']}")


if __name__ == "__main__":
    main()
