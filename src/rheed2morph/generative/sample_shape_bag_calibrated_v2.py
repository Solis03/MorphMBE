"""Generate representative calibrated_v2 AFM samples from MVP-9 shape-bag predictions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.sample_rheed_conditioned_calibrated_v2 import sample as sample_rheed_calibrated_v2


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp5-root", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--v2-diffusion", type=Path, required=True)
    parser.add_argument("--v3-diffusion", type=Path, default=None)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--predicted-condition-table", type=Path, required=True)
    parser.add_argument("--shape-bag-index", type=Path, required=True)
    parser.add_argument("--primary-generator", choices=["calibrated_v2"], default="calibrated_v2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples-per-condition", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--calibration-mode", default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=str_to_bool, default=True)
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        dst.touch()


def _comparison_table(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.is_file():
        write_csv_rows(out_path, [])
        return
    rows = read_csv_rows(metrics_path)
    selected = [
        row
        for row in rows
        if any(token in row.get("prior", "") for token in ["predicted", "oracle", "mean"])
    ]
    write_csv_rows(out_path, selected or rows[: min(12, len(rows))])


def sample(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_args = SimpleNamespace(
        mvp5_root=args.mvp5_root,
        autoencoder=args.autoencoder,
        v2_diffusion=args.v2_diffusion,
        v3_diffusion=args.v3_diffusion,
        predicted_condition_table=args.predicted_condition_table,
        paired_index=args.shape_bag_index,
        condition_schema=args.condition_schema,
        primary_generator=args.primary_generator,
        split=args.split,
        num_samples_per_condition=args.num_samples_per_condition,
        keep_top_k=args.keep_top_k,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        calibration_mode=args.calibration_mode,
        rerank=args.rerank,
        max_conditions=args.max_conditions,
        mock=args.mock,
        out=out_dir,
        device=args.device,
        seed=args.seed,
    )
    summary = sample_rheed_calibrated_v2(base_args)
    grid_src = out_dir / f"rheed_conditioned_calibrated_v2_grid_{args.split}.png"
    grid_dst = out_dir / f"shape_bag_calibrated_v2_grid_{args.split}.png"
    _copy_if_exists(grid_src, grid_dst)
    _copy_if_exists(out_dir / "generation_metrics_mvp6.csv", out_dir / "generation_metrics_shape_bag.csv")
    _copy_if_exists(out_dir / "failure_cases_grid_mvp6.png", out_dir / "generation_failure_cases_shape_bag.png")
    _comparison_table(out_dir / "generation_metrics_shape_bag.csv", out_dir / "oracle_vs_predicted_vs_mean_table.csv")
    base_summary_path = out_dir / "generation_summary_mvp6.json"
    if base_summary_path.is_file():
        summary = read_json(base_summary_path)
    summary.update(
        {
            "mvp": "mvp9_shape_bag",
            "grid": display_path(grid_dst),
            "generation_metrics": display_path(out_dir / "generation_metrics_shape_bag.csv"),
            "oracle_vs_predicted_vs_mean_table": display_path(out_dir / "oracle_vs_predicted_vs_mean_table.csv"),
            "note": "Generated AFM maps are representative calibrated_v2 diffusion samples from predicted morphology conditions, not exact pixel-level reconstruction.",
        }
    )
    write_json(out_dir / "generation_summary_shape_bag.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = sample(args)
    print(f"Wrote MVP-9 shape-bag calibrated_v2 samples to {display_path(resolve_repo_path(args.out))}")
    print(f"grid={summary.get('grid', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
