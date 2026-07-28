from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json

from .evaluation import evaluate_island_methods
from .sample_guided_diffusion import _arrays, _blend
from .train_guided_diffusion import _load_config


M6C = "M6c_island_structure_plus_spectral_prior"
M7 = "M7_structure_guided_residual_diffusion"


def _name(weight: float) -> str:
    return f"diffusion_weight_{float(weight):.2f}"


def _aggregate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.concat(frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    for method, group in values.groupby("method"):
        record: dict[str, Any] = {"method": method}
        for column in group.select_dtypes(include=[np.number, bool]):
            record[f"median_{column}"] = float(
                group[column].astype(float).median()
            )
        if "afm_texture_gate_pass" in group:
            record["texture_gate_pass_fraction"] = float(
                group["afm_texture_gate_pass"].astype(float).mean()
            )
        records.append(record)
    return values, pd.DataFrame(records).sort_values("method")


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    source = repo_path(args.source)
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    standard_frames = []
    island_frames = []
    for fold, held in enumerate(groups):
        fit_groups = set(groups)
        fit_groups.remove(held)
        fit_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str).isin(fit_groups)
        ]
        held_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str) == held
        ]
        m6c = _arrays(
            source / "generated_maps" / M6C / f"{held}.npz"
        )
        m7 = _arrays(
            source / "generated_maps" / M7 / f"{held}.npz"
        )
        methods = {
            _name(weight): _blend(
                m7,
                m6c,
                weight=float(weight),
            )
            for weight in args.weights
        }
        predicted_rq = float(
            np.load(
                source / "generated_maps" / M6C / f"{held}.npz",
                allow_pickle=False,
            )["predicted_rq_nm"]
        )
        scaler = ConditionScaler.fit(
            train_rows,
            list(config["condition_columns"]),
            fit_groups,
        )
        standard = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=fit_rows,
            condition_scaler=scaler,
            generated={
                method: {held: arrays}
                for method, arrays in methods.items()
            },
            generated_rq={
                method: {held: predicted_rq} for method in methods
            },
            output_dir=output / "folds" / held / "standard",
            resolution=int(config["resolution"]),
        )["per_group"]
        standard.insert(0, "cross_validation_fold", fold)
        standard_frames.append(standard)
        island = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=fit_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        island.insert(0, "cross_validation_fold", fold)
        island_frames.append(island)
    standard, standard_summary = _aggregate(standard_frames)
    island, island_summary = _aggregate(island_frames)
    write_csv(standard, output / "standard_per_group.csv")
    write_csv(standard_summary, output / "standard_summary.csv")
    write_csv(island, output / "island_per_group.csv")
    write_csv(island_summary, output / "island_summary.csv")
    merged = standard_summary.merge(
        island_summary, on="method", suffixes=("_standard", "_island")
    )
    write_csv(merged, output / "pareto_summary.csv")
    write_json(
        {
            "experiment": "Strict LOO diffusion-weight ablation",
            "weights": [float(value) for value in args.weights],
            "growth_groups": groups,
            "historical_test_used": False,
            "validation_used": False,
            "retrieval_at_inference": False,
            "measured_afm_patch_used_at_inference": False,
        },
        output / "experiment_manifest.json",
    )
    print(json.dumps(merged.to_dict(orient="records"), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
