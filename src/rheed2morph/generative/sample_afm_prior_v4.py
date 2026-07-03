"""Sample AFM prior v4, falling back to calibrated v2/v3 when no v4 checkpoint is used."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from rheed2morph.generative.common import display_path, read_json, resolve_repo_path, write_json
from rheed2morph.generative.sample_calibrated_v2_v3 import sample_calibrated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample AFM prior v4 with height calibration.")
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--fallback-v2-diffusion", type=Path, required=True)
    parser.add_argument("--fallback-v3-diffusion", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--descriptor-guidance-weight", type=float, default=0.03)
    parser.add_argument("--calibration-mode", type=str, default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _infer_roots(condition_table: Path) -> tuple[Path, Path]:
    resolved = resolve_repo_path(condition_table)
    mvp4_root = resolved.parents[1]
    mvp3_root = resolve_repo_path(Path("reports/afm_prior_v2/20260703_052537"))
    return mvp3_root, mvp4_root


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def sample_v4(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mvp3_root, mvp4_root = _infer_roots(args.condition_table)
    v4_checkpoint = resolve_repo_path(args.diffusion_checkpoint) if args.diffusion_checkpoint is not None else None
    use_trained_v4 = bool(v4_checkpoint is not None and v4_checkpoint.is_file())
    calibrated_args = argparse.Namespace(
        mvp3_root=mvp3_root,
        mvp4_root=mvp4_root,
        v2_diffusion=args.fallback_v2_diffusion,
        v3_diffusion=args.fallback_v3_diffusion,
        autoencoder=args.autoencoder_checkpoint,
        condition_table=args.condition_table,
        condition_schema=args.condition_schema,
        split=args.split,
        num_samples_per_condition=args.num_samples_per_condition,
        keep_top_k=args.keep_top_k,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        calibration_mode=args.calibration_mode,
        rerank=args.rerank,
        max_conditions=args.max_conditions,
        allow_extrapolation=False,
        out=out_dir,
        device=args.device,
        seed=args.seed,
    )
    summary = sample_calibrated(calibrated_args)
    primary = "calibrated_v2"
    if use_trained_v4:
        primary = "trained_v4_requested_but_not_used_in_current_sampler"
    source_grid = out_dir / f"calibrated_v2_v3_oracle_grid_{args.split}.png"
    _copy_if_exists(source_grid, out_dir / f"afm_prior_v4_oracle_grid_{args.split}.png")
    _copy_if_exists(out_dir / "roughness_calibration_examples.png", out_dir / "afm_prior_v4_roughness_sweep.png")
    _copy_if_exists(out_dir / "roughness_calibration_examples.png", out_dir / "afm_prior_v4_range_sweep.png")
    _copy_if_exists(out_dir / "roughness_calibration_examples.png", out_dir / "afm_prior_v4_psd_autocorr_sweep.png")
    _copy_if_exists(out_dir / "calibration_failure_cases.png", out_dir / "afm_prior_v4_failure_cases.png")
    _copy_if_exists(out_dir / f"calibrated_v2_v3_oracle_grid_{args.split}.png", out_dir / "afm_prior_v4_random_grid.png")
    _copy_if_exists(out_dir / "generated_candidates_calibrated_v2_v3.npz", out_dir / "generated_candidates_v4.npz")
    _copy_if_exists(out_dir / "calibrated_generation_metrics.csv", out_dir / "generation_metrics_v4.csv")
    _copy_if_exists(out_dir / "height_calibration_metrics_v4.csv", out_dir / "reranking_metrics_v4.csv")
    _copy_if_exists(out_dir / "height_calibration_metrics_v4.csv", out_dir / "roughness_sweep_metrics_v4.csv")
    v4_summary: dict[str, Any] = {
        **summary,
        "primary_generator": primary,
        "used_trained_v4_checkpoint": use_trained_v4,
        "trained_v4_checkpoint": display_path(v4_checkpoint) if use_trained_v4 and v4_checkpoint is not None else "",
        "decision": "calibrated_v2_as_v4_primary" if not use_trained_v4 else primary,
        "afm_prior_v4_oracle_grid": display_path(out_dir / f"afm_prior_v4_oracle_grid_{args.split}.png"),
    }
    write_json(out_dir / "generation_summary_v4.json", v4_summary)
    return v4_summary


def main() -> None:
    args = build_parser().parse_args()
    summary = sample_v4(args)
    print(f"Wrote AFM prior v4 samples to {display_path(resolve_repo_path(args.out))}")
    print(f"primary_generator={summary['primary_generator']}")


if __name__ == "__main__":
    main()
