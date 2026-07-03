"""Compare MVP-1 and MVP-3 priors under MVP-2 RHEED-predicted conditions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import bool_arg, compute_afm_descriptors_v2
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
from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion import load_diffusion_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MVP-1 and MVP-3 generation under MVP-2 predicted conditions.")
    parser.add_argument("--mvp2-root", type=Path, required=True)
    parser.add_argument("--mvp1-diffusion", type=Path, required=True)
    parser.add_argument("--mvp1-autoencoder", type=Path, required=True)
    parser.add_argument("--mvp3-diffusion", type=Path, required=True)
    parser.add_argument("--mvp3-autoencoder", type=Path, required=True)
    parser.add_argument("--mvp3-condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--fill-missing-with-train-mean", type=bool_arg, default=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _find_predicted_table(mvp2_root: Path, split: str) -> Path:
    preferred = [
        mvp2_root / "predicted_conditions_10epoch_visual_handcrafted" / f"predicted_condition_table_{split}.csv",
        mvp2_root / "predicted_conditions" / f"predicted_condition_table_{split}.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path
    matches = sorted(mvp2_root.glob(f"**/predicted_condition_table_{split}.csv"))
    if not matches:
        raise FileNotFoundError(f"No MVP-2 predicted condition table for split={split} under {mvp2_root}")
    return matches[0]


def _mvp1_condition(row: dict[str, str], config: dict[str, Any]) -> np.ndarray:
    values = [float(row[col]) for col in config["condition_columns"]]
    proto_count = int(config.get("prototype_count", 0))
    if proto_count > 0:
        one_hot = [0.0] * proto_count
        proto = row.get("prototype_id", "")
        if proto != "":
            index = int(float(proto))
            if 0 <= index < proto_count:
                one_hot[index] = 1.0
        values.extend(one_hot)
    return np.asarray(values, dtype=np.float32)


def adapt_mvp2_row_to_mvp3_condition(
    row: dict[str, str],
    schema: dict[str, Any],
    mode: str,
    fill_missing_with_train_mean: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode not in {"predicted", "oracle", "mean"}:
        raise ValueError("mode must be predicted, oracle, or mean.")
    descriptor_columns = list(schema["descriptor_columns"])
    means = dict(schema["descriptor_train_mean"])
    stds = dict(schema["descriptor_train_std"])
    values: list[float] = []
    missing: list[str] = []
    mapped: list[str] = []
    for name in descriptor_columns:
        if mode == "mean":
            raw_value = float(means[name])
            mapped.append(name)
        else:
            key = f"pred_{name}" if mode == "predicted" else name
            if row.get(key, "") != "":
                raw_value = float(row[key])
                mapped.append(name)
            elif row.get(name, "") != "" and mode == "predicted":
                raw_value = float(row[name])
                mapped.append(name)
            elif fill_missing_with_train_mean:
                raw_value = float(means[name])
                missing.append(name)
            else:
                raise ValueError(
                    f"Cannot adapt MVP-2 {mode} row to MVP-3 schema; missing descriptor {name}. "
                    "Enable fill_missing_with_train_mean to use v2 train means."
                )
        values.append((raw_value - float(means[name])) / float(stds.get(name, 1.0) or 1.0))
    proto_count = int(schema.get("prototype_count", 0))
    if proto_count > 0:
        values.extend([0.0] * proto_count)
    extra_predicted = sorted(
        key[5:]
        for key in row
        if key.startswith("pred_") and key[5:] not in descriptor_columns and not key.startswith("pred_cond_")
    )
    report = {
        "mode": mode,
        "mapped_descriptor_count": len(mapped),
        "filled_descriptor_count": len(missing),
        "filled_descriptors": missing,
        "extra_predicted_descriptors": extra_predicted,
        "prototype_policy": "zero prototype vector because MVP-1 and MVP-3 prototype ids are not semantically aligned",
    }
    return np.asarray(values, dtype=np.float32), report


def _decode(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: Any,
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


def compare_generation(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    mvp2_root = resolve_repo_path(args.mvp2_root)
    predicted_table = _find_predicted_table(mvp2_root, str(args.split))
    predicted_rows = [row for row in read_csv_rows(predicted_table) if row.get("split") == args.split]
    if not predicted_rows:
        predicted_rows = read_csv_rows(predicted_table)
    predicted_rows = predicted_rows[: int(args.max_conditions)]

    mvp1_model, mvp1_payload = load_diffusion_checkpoint(args.mvp1_diffusion, str(device))
    mvp1_config = dict(mvp1_payload["config"])
    mvp1_autoencoder, _mvp1_ae = load_autoencoder_checkpoint(args.mvp1_autoencoder, str(device))
    mvp1_model.to(device).eval()
    mvp1_autoencoder.to(device).eval()
    mvp1_latent_stats = np.load(resolve_repo_path(Path(mvp1_config["latents_dir"])) / "latent_standardization.npz")
    mvp1_latent_mean = torch.from_numpy(np.asarray(mvp1_latent_stats["latent_mean"], dtype=np.float32)).to(device)
    mvp1_latent_std = torch.from_numpy(np.asarray(mvp1_latent_stats["latent_std"], dtype=np.float32)).to(device)
    mvp1_diffusion = GaussianDiffusion(timesteps=int(mvp1_config["timesteps"]), device=device)
    mvp1_latent_shape = tuple(int(value) for value in mvp1_config["latent_shape"])

    mvp3_model, mvp3_payload = load_diffusion_v2_checkpoint(args.mvp3_diffusion, str(device))
    mvp3_config = dict(mvp3_payload["config"])
    mvp3_schema = read_json(resolve_repo_path(args.mvp3_condition_schema))
    mvp3_autoencoder, _mvp3_ae = load_autoencoder_v2_checkpoint(args.mvp3_autoencoder, str(device))
    mvp3_model.to(device).eval()
    mvp3_autoencoder.to(device).eval()
    mvp3_latent_stats = np.load(resolve_repo_path(Path(mvp3_config["latents_dir"])) / "latent_standardization_v2.npz")
    mvp3_latent_mean = torch.from_numpy(np.asarray(mvp3_latent_stats["latent_mean"], dtype=np.float32)).to(device)
    mvp3_latent_std = torch.from_numpy(np.asarray(mvp3_latent_stats["latent_std"], dtype=np.float32)).to(device)
    mvp3_diffusion = GaussianDiffusionV2(
        timesteps=int(mvp3_config["timesteps"]),
        beta_schedule=str(mvp3_config.get("beta_schedule", "cosine")),
        prediction_target=str(mvp3_config.get("prediction_target", "epsilon")),
        device=device,
    )
    mvp3_latent_shape = tuple(int(value) for value in mvp3_config["latent_shape"])

    grid_rows: list[list[np.ndarray]] = []
    row_titles: list[str] = []
    metric_rows: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    mean_condition, mean_report = adapt_mvp2_row_to_mvp3_condition(
        predicted_rows[0],
        mvp3_schema,
        "mean",
        fill_missing_with_train_mean=True,
    )
    adapter_reports.append(mean_report)
    for row in predicted_rows:
        rheed_path = row.get("cached_tensor_path", "")
        if rheed_path:
            try:
                rheed = np.load(resolve_repo_path(Path(rheed_path)))["frames"][-1, 0]
            except Exception:
                rheed = np.zeros((128, 128), dtype=np.float32)
        else:
            rheed = np.zeros((128, 128), dtype=np.float32)
        true_afm = load_height_array(resolve_repo_path(Path(row.get("network_input_path", ""))))
        mvp1_condition = _mvp1_condition(row, mvp1_config)
        mvp1_image = _decode(
            mvp1_model,
            mvp1_autoencoder,
            mvp1_diffusion,
            mvp1_latent_mean,
            mvp1_latent_std,
            mvp1_condition,
            mvp1_latent_shape,
            1,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )[0]
        pred_condition, pred_report = adapt_mvp2_row_to_mvp3_condition(
            row,
            mvp3_schema,
            "predicted",
            fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean),
        )
        oracle_condition, oracle_report = adapt_mvp2_row_to_mvp3_condition(
            row,
            mvp3_schema,
            "oracle",
            fill_missing_with_train_mean=bool(args.fill_missing_with_train_mean),
        )
        adapter_reports.extend([pred_report, oracle_report])
        mvp3_pred = _decode(
            mvp3_model,
            mvp3_autoencoder,
            mvp3_diffusion,
            mvp3_latent_mean,
            mvp3_latent_std,
            pred_condition,
            mvp3_latent_shape,
            int(args.num_samples_per_condition),
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )
        mvp3_oracle = _decode(
            mvp3_model,
            mvp3_autoencoder,
            mvp3_diffusion,
            mvp3_latent_mean,
            mvp3_latent_std,
            oracle_condition,
            mvp3_latent_shape,
            1,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )[0]
        mvp3_mean = _decode(
            mvp3_model,
            mvp3_autoencoder,
            mvp3_diffusion,
            mvp3_latent_mean,
            mvp3_latent_std,
            mean_condition,
            mvp3_latent_shape,
            1,
            int(args.ddim_steps),
            float(args.guidance_scale),
            device,
        )[0]
        grid_rows.append([rheed, true_afm, mvp1_image, *[mvp3_pred[i] for i in range(min(2, mvp3_pred.shape[0]))], mvp3_oracle, mvp3_mean])
        row_titles.append(str(row.get("sample_id", row["row_id"])))
        for mode, images in (
            ("mvp1_predicted", np.asarray([mvp1_image])),
            ("mvp3_predicted", mvp3_pred),
            ("mvp3_oracle", np.asarray([mvp3_oracle])),
            ("mvp3_mean", np.asarray([mvp3_mean])),
        ):
            for image_index, image in enumerate(images):
                descriptors = compute_afm_descriptors_v2(image)
                metric_rows.append(
                    {
                        "row_id": row.get("row_id", ""),
                        "sample_id": row.get("sample_id", ""),
                        "mode": mode,
                        "generated_index": image_index,
                        "generated_std": f"{float(np.std(image)):.10g}",
                        "generated_rq": f"{float(descriptors['rq']):.10g}",
                        "generated_ra": f"{float(descriptors['ra']):.10g}",
                        "generated_psd_slope": f"{float(descriptors['psd_slope']):.10g}",
                        "requested_rq": row.get("pred_rq", row.get("rq", "")),
                        "requested_ra": row.get("pred_ra", row.get("ra", "")),
                    }
                )
    grid_path = out_dir / "mvp1_vs_mvp3_rheed_conditioned_grid.png"
    write_panel_grid(
        grid_path,
        grid_rows,
        [
            "RHEED final frame",
            "true AFM",
            "MVP-1 pred-conditioned gen",
            "MVP-3 pred gen 1",
            "MVP-3 pred gen 2",
            "oracle MVP-3 gen",
            "mean MVP-3 gen",
        ],
        row_titles,
    )
    write_csv_rows(out_dir / "comparison_generation_metrics.csv", metric_rows)
    report_lines = [
        "# MVP-2 To MVP-3 Condition Adapter Report",
        "",
        f"Predicted table: `{display_path(predicted_table)}`",
        f"Fill missing descriptors with MVP-3 train mean: `{bool(args.fill_missing_with_train_mean)}`",
        "",
        "MVP-1 and MVP-3 prototype ids are not semantically aligned, so MVP-3 adapted vectors use a zero prototype vector.",
        "",
    ]
    filled = sorted({name for report in adapter_reports for name in report.get("filled_descriptors", [])})
    report_lines.append(f"Filled descriptor names: `{filled}`")
    mapped_counts = [int(report.get("mapped_descriptor_count", 0)) for report in adapter_reports if report.get("mode") != "mean"]
    report_lines.append(f"Mapped descriptor count range: `{min(mapped_counts) if mapped_counts else 0}` to `{max(mapped_counts) if mapped_counts else 0}`")
    (out_dir / "condition_adapter_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    stds = [float(row["generated_std"]) for row in metric_rows]
    summary = {
        "predicted_table": display_path(predicted_table),
        "comparison_grid": display_path(grid_path),
        "condition_adapter_report": display_path(out_dir / "condition_adapter_report.md"),
        "comparison_generation_metrics": display_path(out_dir / "comparison_generation_metrics.csv"),
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "adapter_filled_descriptors": filled,
        "note": "This comparison reuses MVP-2 predicted conditions and does not retrain the RHEED encoder.",
    }
    write_json(out_dir / "comparison_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_generation(args)
    print(f"Wrote MVP-1 vs MVP-3 comparison to {display_path(resolve_repo_path(args.out))}")
    print(f"generated_nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
