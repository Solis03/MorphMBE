"""Sample calibrated_v2 AFM maps from MVP-10 policy-adjusted OOF predictions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.sample_shape_bag_calibrated_v2 import sample as sample_shape_bag


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _copy(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        dst.touch()


def sample(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = SimpleNamespace(
        mvp5_root=args.mvp5_root,
        autoencoder=args.autoencoder,
        v2_diffusion=args.v2_diffusion,
        v3_diffusion=None,
        condition_schema=args.condition_schema,
        predicted_condition_table=args.predicted_condition_table,
        shape_bag_index=args.shape_bag_index,
        primary_generator="calibrated_v2",
        split=args.split,
        num_samples_per_condition=args.num_samples_per_condition,
        keep_top_k=args.keep_top_k,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        calibration_mode=args.calibration_mode,
        rerank=args.rerank,
        max_conditions=args.max_conditions,
        mock=args.mock,
        device=args.device,
        seed=args.seed,
        out=out,
    )
    summary = sample_shape_bag(base)
    _copy(out / f"shape_bag_calibrated_v2_grid_{args.split}.png", out / "trustworthy_shape_bag_calibrated_v2_grid.png")
    _copy(out / "generation_metrics_shape_bag.csv", out / "trustworthy_generation_metrics.csv")
    _copy(out / "generation_summary_shape_bag.json", out / "base_generation_summary_shape_bag.json")
    _copy(out / "generation_failure_cases_shape_bag.png", out / "generation_failure_cases.png")
    _copy(out / "oracle_vs_predicted_vs_mean_table.csv", out / "oracle_vs_predicted_vs_mean_supported_descriptors.csv")
    base_summary = read_json(out / "base_generation_summary_shape_bag.json") if (out / "base_generation_summary_shape_bag.json").is_file() else summary
    rows = read_csv_rows(resolve_repo_path(args.predicted_condition_table))
    policy_cols = [key for key in rows[0].keys() if key.startswith("policy_")] if rows else []
    supported_counts = {
        "predicted_by_rheed": sum(1 for row in rows for key in policy_cols if row.get(key) == "predicted_by_rheed"),
        "filled_by_train_mean": sum(1 for row in rows for key in policy_cols if row.get(key) == "filled_by_train_mean"),
    }
    base_summary.update(
        {
            "mvp": "mvp10_trustworthy_shape_bag",
            "grid": display_path(out / "trustworthy_shape_bag_calibrated_v2_grid.png"),
            "generation_metrics": display_path(out / "trustworthy_generation_metrics.csv"),
            "descriptor_policy_counts": supported_counts,
            "note": "Generated AFM maps are representative calibrated_v2 diffusion samples from policy-adjusted descriptor conditions, not exact pixel-level reconstruction.",
        }
    )
    write_json(out / "trustworthy_generation_summary.json", base_summary)
    return base_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp5-root", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--v2-diffusion", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--predicted-condition-table", type=Path, required=True)
    parser.add_argument("--shape-bag-index", type=Path, required=True)
    parser.add_argument("--num-samples-per-condition", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--calibration-mode", default="weighted_rq_ra_range")
    parser.add_argument("--rerank", type=str_to_bool, default=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-conditions", type=int, default=4)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = sample(args)
    print(f"Wrote trustworthy calibrated_v2 generation to {display_path(resolve_repo_path(args.out))}")
    print(f"grid={summary.get('grid', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
