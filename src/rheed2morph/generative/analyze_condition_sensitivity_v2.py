"""Diagnose MVP-3 v2 diffusion condition sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.afm_prior_v2_utils import compute_afm_descriptors_v2
from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import (
    SWEEP_DESCRIPTOR_NAMES,
    condition_row_to_vector,
    correlation,
    finite_float,
    format_float,
    monotonicity_score,
    rank_correlation,
    raw_descriptor_to_condition,
)
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_afm_latent_diffusion_v2 import load_diffusion_v2_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze MVP-3 v2 condition sensitivity.")
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-base-conditions", type=int, default=8)
    parser.add_argument("--num-samples-per-condition", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scales", type=str, default="0.0,0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_guidance(text: str) -> list[float]:
    return [float(item) for item in str(text).split(",") if item.strip()]


def _sweep_values(name: str, schema: dict[str, Any]) -> list[float]:
    mean = float(schema["descriptor_train_mean"].get(name, 0.0))
    std = float(schema["descriptor_train_std"].get(name, 1.0) or 1.0)
    if name == "island_coverage":
        return [0.10, 0.18, 0.25, 0.32, 0.40]
    return [mean + z * std for z in (-2.0, -1.0, 0.0, 1.0, 2.0)]


def _decode_samples(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    diffusion: GaussianDiffusionV2,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    condition: np.ndarray,
    latent_shape: tuple[int, int, int],
    count: int,
    steps: int,
    guidance: float,
    device: torch.device,
) -> np.ndarray:
    cond = torch.from_numpy(np.repeat(condition[None], count, axis=0).astype(np.float32)).to(device)
    sampled = diffusion.sample_ddim(model, (count, *latent_shape), cond, steps=steps, guidance_scale=guidance)
    raw = sampled * latent_std + latent_mean
    with torch.no_grad():
        decoded = autoencoder.decode(raw).detach().cpu().numpy()
    return decoded[:, 0]


def _write_scatter_plot(out_dir: Path, metric_rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    descriptors = [name for name in SWEEP_DESCRIPTOR_NAMES if name in schema["descriptor_columns"]]
    if not descriptors:
        return
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), dpi=150, squeeze=False)
    for axis, name in zip(axes.ravel(), descriptors):
        rows = [row for row in metric_rows if row["sweep_descriptor"] == name]
        x = [finite_float(row["requested_raw"]) for row in rows]
        y = [finite_float(row.get(f"generated_{name}", "nan")) for row in rows]
        axis.scatter(x, y, s=8, alpha=0.5)
        axis.set_title(name, fontsize=8)
        axis.set_xlabel("requested")
        axis.set_ylabel("generated")
    fig.tight_layout()
    fig.savefig(out_dir / "requested_vs_generated_scatter_v2.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for name in descriptors:
        rows = [row for row in metric_rows if row["sweep_descriptor"] == name]
        grouped: dict[float, list[float]] = {}
        for row in rows:
            requested = finite_float(row["requested_raw"])
            generated = finite_float(row.get(f"generated_{name}", "nan"))
            if np.isfinite(requested) and np.isfinite(generated):
                grouped.setdefault(requested, []).append(generated)
        if grouped:
            xs = sorted(grouped)
            ys = [float(np.mean(grouped[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=name)
    ax.legend(fontsize=7)
    ax.set_xlabel("requested raw descriptor")
    ax.set_ylabel("generated descriptor mean")
    fig.tight_layout()
    fig.savefig(out_dir / "monotonicity_curves_v2.png")
    plt.close(fig)


def _write_sweep_grids(
    out_dir: Path,
    grid_examples: dict[str, list[tuple[str, np.ndarray]]],
    prototypes: list[tuple[str, np.ndarray]],
) -> None:
    name_to_file = {
        "rq": "condition_sweep_rq.png",
        "psd_slope": "condition_sweep_psd_slope.png",
        "autocorrelation_length_px": "condition_sweep_autocorr.png",
        "gradient_anisotropy": "condition_sweep_anisotropy.png",
    }
    for name, filename in name_to_file.items():
        panels = grid_examples.get(name, [])
        if panels:
            write_panel_grid(out_dir / filename, [[image for _title, image in panels]], [title for title, _image in panels])
    if prototypes:
        write_panel_grid(out_dir / "condition_sweep_prototype.png", [[image for _title, image in prototypes]], [title for title, _image in prototypes])


def analyze_condition_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    condition_rows = read_csv_rows(resolve_repo_path(args.condition_table))
    schema = read_json(resolve_repo_path(args.condition_schema))
    base_rows = [row for row in condition_rows if row.get("split") == args.split][: int(args.num_base_conditions)]
    if not base_rows:
        base_rows = condition_rows[: int(args.num_base_conditions)]
    model, payload = load_diffusion_v2_checkpoint(args.diffusion_checkpoint, str(device))
    config = dict(payload["config"])
    model.to(device).eval()
    autoencoder, _ae_payload = load_autoencoder_v2_checkpoint(args.autoencoder_checkpoint, str(device))
    autoencoder.to(device).eval()
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
    guidance_values = _parse_guidance(args.guidance_scales)
    descriptor_names = [name for name in SWEEP_DESCRIPTOR_NAMES if name in schema["descriptor_columns"]]
    metric_rows: list[dict[str, Any]] = []
    grid_examples: dict[str, list[tuple[str, np.ndarray]]] = {}
    for descriptor in descriptor_names:
        values = _sweep_values(descriptor, schema)
        for guidance in guidance_values:
            for base_index, base in enumerate(base_rows):
                for value_index, raw_value in enumerate(values):
                    row = dict(base)
                    row[descriptor] = format_float(raw_value)
                    row[f"cond_{descriptor}"] = format_float(raw_descriptor_to_condition(descriptor, raw_value, schema))
                    condition = condition_row_to_vector(row, schema)
                    decoded = _decode_samples(
                        model,
                        autoencoder,
                        diffusion,
                        latent_mean,
                        latent_std,
                        condition,
                        latent_shape,
                        int(args.num_samples_per_condition),
                        int(args.ddim_steps),
                        float(guidance),
                        device,
                    )
                    if guidance == guidance_values[-1] and base_index == 0:
                        grid_examples.setdefault(descriptor, []).append((f"{descriptor}={raw_value:.2g}", decoded[0]))
                    for sample_index, image in enumerate(decoded):
                        generated_desc = compute_afm_descriptors_v2(image)
                        out = {
                            "sweep_descriptor": descriptor,
                            "base_row_id": base["row_id"],
                            "sample_id": base.get("sample_id", ""),
                            "guidance_scale": guidance,
                            "requested_raw": format_float(raw_value),
                            "requested_cond": format_float(raw_descriptor_to_condition(descriptor, raw_value, schema)),
                            "sample_index": sample_index,
                            "generated_std": format_float(float(np.std(image))),
                            "generated_min": format_float(float(np.min(image))),
                            "generated_max": format_float(float(np.max(image))),
                        }
                        for name in schema["descriptor_columns"]:
                            if name in generated_desc:
                                out[f"generated_{name}"] = format_float(float(generated_desc[name]))
                        metric_rows.append(out)
    prototype_examples: list[tuple[str, np.ndarray]] = []
    proto_count = int(schema.get("prototype_count", 0))
    if proto_count > 0 and base_rows:
        base = dict(base_rows[0])
        for proto in range(proto_count):
            base["prototype_id"] = str(proto)
            condition = condition_row_to_vector(base, schema)
            decoded = _decode_samples(model, autoencoder, diffusion, latent_mean, latent_std, condition, latent_shape, 1, int(args.ddim_steps), guidance_values[-1], device)
            prototype_examples.append((f"prototype {proto}", decoded[0]))
            desc = compute_afm_descriptors_v2(decoded[0])
            metric_rows.append(
                {
                    "sweep_descriptor": "prototype_id",
                    "base_row_id": base["row_id"],
                    "sample_id": base.get("sample_id", ""),
                    "guidance_scale": guidance_values[-1],
                    "requested_raw": proto,
                    "requested_cond": proto,
                    "sample_index": 0,
                    "generated_std": format_float(float(np.std(decoded[0]))),
                    **{f"generated_{name}": format_float(float(desc[name])) for name in schema["descriptor_columns"] if name in desc},
                }
            )
    write_csv_rows(out_dir / "v2_condition_sensitivity_metrics.csv", metric_rows)
    descriptor_summaries: list[dict[str, Any]] = []
    for descriptor in descriptor_names:
        best: dict[str, Any] | None = None
        for guidance in guidance_values:
            rows = [row for row in metric_rows if row["sweep_descriptor"] == descriptor and float(row["guidance_scale"]) == float(guidance)]
            requested = [finite_float(row["requested_raw"]) for row in rows]
            generated = [finite_float(row.get(f"generated_{descriptor}", "nan")) for row in rows]
            pearson = correlation(requested, generated)
            spearman = rank_correlation(requested, generated)
            req = np.asarray(requested, dtype=np.float64)
            gen = np.asarray(generated, dtype=np.float64)
            mask = np.isfinite(req) & np.isfinite(gen)
            mae = float(np.mean(np.abs(gen[mask] - req[mask]))) if np.any(mask) else float("nan")
            rmse = float(np.sqrt(np.mean((gen[mask] - req[mask]) ** 2))) if np.any(mask) else float("nan")
            summary = {
                "descriptor": descriptor,
                "guidance_scale": guidance,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else float("nan"),
                "mae": mae,
                "rmse": rmse,
                "monotonicity": monotonicity_score(requested, generated),
            }
            if best is None or (np.nan_to_num(summary["abs_pearson"], nan=-1.0) > np.nan_to_num(best["abs_pearson"], nan=-1.0)):
                best = summary
        if best is not None:
            descriptor_summaries.append({**best, "best_guidance_scale": best["guidance_scale"], "best_abs_pearson": best["abs_pearson"]})
    stds = [finite_float(row.get("generated_std", "nan")) for row in metric_rows]
    stds = [value for value in stds if np.isfinite(value)]
    cfg_rows = []
    for descriptor in descriptor_names:
        for guidance in guidance_values:
            rows = [row for row in metric_rows if row["sweep_descriptor"] == descriptor and float(row["guidance_scale"]) == float(guidance)]
            cfg_rows.append({"descriptor": descriptor, "guidance_scale": guidance, "generated_std_mean": float(np.mean([finite_float(row["generated_std"]) for row in rows])) if rows else 0.0})
    summary = {
        "mvp3_root": display_path(resolve_repo_path(args.mvp3_root)),
        "row_count": len(metric_rows),
        "descriptor_summaries": descriptor_summaries,
        "generated_std_mean": float(np.mean(stds)) if stds else 0.0,
        "generated_std_min": float(np.min(stds)) if stds else 0.0,
        "generated_nonconstant_rate": float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0,
        "guidance_scales": guidance_values,
        "cfg_effect_rows": cfg_rows,
        "sweep_descriptors": descriptor_names,
    }
    write_json(out_dir / "v2_condition_sensitivity_summary.json", summary)
    _write_scatter_plot(out_dir, metric_rows, schema)
    _write_sweep_grids(out_dir, grid_examples, prototype_examples)
    report_lines = [
        "# V2 Condition Sensitivity Report",
        "",
        f"Metric rows: `{len(metric_rows)}`",
        f"Generated nonconstant rate: `{summary['generated_nonconstant_rate']:.3f}`",
        "",
        "## Descriptor Summaries",
        "",
    ]
    for item in descriptor_summaries:
        report_lines.append(
            f"- `{item['descriptor']}`: best guidance `{item['best_guidance_scale']}`, "
            f"Pearson `{item['pearson']:.4g}`, MAE `{item['mae']:.4g}`, monotonicity `{item['monotonicity']:.4g}`"
        )
    (out_dir / "v2_condition_sensitivity_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_condition_sensitivity(args)
    print(f"Wrote v2 condition sensitivity outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"rows={summary['row_count']} nonconstant_rate={summary['generated_nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
