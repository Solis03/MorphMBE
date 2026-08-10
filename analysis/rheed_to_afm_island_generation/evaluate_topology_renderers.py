from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .evaluation import evaluate_island_methods
from .sample_guided_diffusion import _arrays
from .train_guided_diffusion import _load_config

M5 = "M5_cloudlike_spectral_hybrid"
M6B = "M6b_multiscale_laguerre_terraces"
M6C = "M6c_island_structure_plus_spectral_prior"


def _aggregate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.concat(frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    for method, group in values.groupby("method"):
        record: dict[str, Any] = {"method": method}
        for column in group.select_dtypes(include=[np.number, bool]):
            record[f"median_{column}"] = float(group[column].astype(float).median())
        if "afm_texture_gate_pass" in group:
            record["texture_gate_pass_fraction"] = float(
                group["afm_texture_gate_pass"].astype(float).mean()
            )
        records.append(record)
    return values, pd.DataFrame(records).sort_values("method")


def _edge_render(
    structure: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    edge_gain: float,
) -> list[np.ndarray]:
    result = []
    for index in range(max(len(structure), len(spectral))):
        island = structure[index % len(structure)]
        base = spectral[index % len(spectral)]
        edge = island - ndimage.gaussian_filter(island, sigma=1.35, mode="wrap")
        result.append(
            project_unit_rq_np(base + float(edge_gain) * edge).astype(np.float32)
        )
    return result


def _quantized(array: np.ndarray, *, levels: int, smooth_sigma: float) -> np.ndarray:
    thresholds = np.quantile(array, np.linspace(0.0, 1.0, int(levels) + 1))
    centers = 0.5 * (thresholds[:-1] + thresholds[1:])
    labels = np.clip(
        np.digitize(array, thresholds[1:-1]),
        0,
        len(centers) - 1,
    )
    terrace = centers[labels]
    return ndimage.gaussian_filter(terrace, sigma=float(smooth_sigma), mode="wrap")


def _terrace_render(
    structure: list[np.ndarray],
    cloud: list[np.ndarray],
    *,
    levels: int,
    structure_weight: float,
) -> list[np.ndarray]:
    result = []
    for index in range(max(len(structure), len(cloud))):
        terrace = _quantized(
            structure[index % len(structure)],
            levels=int(levels),
            smooth_sigma=0.45,
        )
        base = cloud[index % len(cloud)]
        result.append(
            project_unit_rq_np(
                float(structure_weight) * terrace
                + (1.0 - float(structure_weight)) * base
            ).astype(np.float32)
        )
    return result


def _structure_blend(
    structure: list[np.ndarray],
    cloud: list[np.ndarray],
    *,
    weight: float,
) -> list[np.ndarray]:
    return [
        project_unit_rq_np(
            float(weight) * structure[index % len(structure)]
            + (1.0 - float(weight)) * cloud[index % len(cloud)]
        ).astype(np.float32)
        for index in range(max(len(structure), len(cloud)))
    ]


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
        held_rows = train_rows.loc[train_rows["growth_run_id"].astype(str) == held]
        m5 = _arrays(source / M5 / f"{held}.npz")
        m6b = _arrays(source / M6B / f"{held}.npz")
        m6c = _arrays(source / M6C / f"{held}.npz")
        methods = {M6C: m6c}
        for gain in args.edge_gains:
            methods[f"M9a_edge_gain_{float(gain):.2f}"] = _edge_render(
                m6b, m6c, edge_gain=float(gain)
            )
        for levels in args.terrace_levels:
            methods[f"M9b_quantized_{int(levels)}_levels"] = _terrace_render(
                m6b,
                m5,
                levels=int(levels),
                structure_weight=float(args.structure_weight),
            )
        for weight in args.structure_weights:
            methods[f"M10_structure_weight_{float(weight):.2f}"] = _structure_blend(
                m6b, m5, weight=float(weight)
            )
        predicted_rq = float(
            np.load(source / M6C / f"{held}.npz", allow_pickle=False)["predicted_rq_nm"]
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
            generated={method: {held: arrays} for method, arrays in methods.items()},
            generated_rq={method: {held: predicted_rq} for method in methods},
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
    write_csv(
        standard_summary.merge(
            island_summary, on="method", suffixes=("_standard", "_island")
        ),
        output / "pareto_summary.csv",
    )
    write_json(
        {
            "experiment": "Strict LOO topology-renderer ablation",
            "edge_gains": [float(value) for value in args.edge_gains],
            "terrace_levels": [int(value) for value in args.terrace_levels],
            "structure_weight": float(args.structure_weight),
            "structure_weights": [float(value) for value in args.structure_weights],
            "growth_groups": groups,
            "historical_test_used": False,
            "validation_used": False,
            "retrieval_at_inference": False,
            "measured_afm_patch_used_at_inference": False,
        },
        output / "experiment_manifest.json",
    )
    print(json.dumps(standard_summary.to_dict(orient="records"), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--edge-gains", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--terrace-levels", type=int, nargs="+", default=[5, 7, 9])
    parser.add_argument("--structure-weight", type=float, default=0.60)
    parser.add_argument(
        "--structure-weights",
        type=float,
        nargs="*",
        default=[],
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
