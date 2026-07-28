from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from .evaluation import evaluate_island_methods
from .guided_diffusion import (
    StructureGuidedDiffusion,
    StructureGuidedResidualUNet,
)
from .sample_guided_diffusion import _arrays, _blend, _refine
from .train_guided_diffusion import (
    _batch,
    _device,
    _guides,
    _load_config,
    _maps,
)


M5 = "M5_cloudlike_spectral_hybrid"
M6C = "M6c_island_structure_plus_spectral_prior"
M6B = "M6b_multiscale_laguerre_terraces"
M7 = "M7_structure_guided_residual_diffusion"
M8 = "M8_island_diffusion_spectral_pareto"


def _aggregate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.concat(frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    for method, group in values.groupby("method"):
        record: dict[str, Any] = {
            "method": method,
            "growth_group_count": int(group["growth_run_id"].nunique()),
        }
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


def _save(
    root: Path,
    *,
    group: str,
    method: str,
    arrays: list[np.ndarray],
    predicted_rq_nm: float,
) -> None:
    path = root / "generated_maps" / method / f"{group}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_unit_shapes=np.stack(arrays).astype(np.float32),
        predicted_rq_nm=np.asarray(predicted_rq_nm),
        growth_run_id=np.asarray(group),
        method=np.asarray(method),
        retrieval_at_inference=np.asarray(False),
        measured_afm_patch_used_at_inference=np.asarray(False),
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    output = repo_path(args.output)
    report = repo_path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    guide_root = repo_path(args.guide_root)
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    device = _device(args.device)
    standard_frames: list[pd.DataFrame] = []
    island_frames: list[pd.DataFrame] = []
    training_records: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for fold, held in enumerate(groups):
        fold_start = time.perf_counter()
        fit_groups = set(groups)
        fit_groups.remove(held)
        fit_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str).isin(fit_groups)
        ]
        held_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str) == held
        ]
        fit_arrays, _ = _maps(fit_rows, int(config["resolution"]))
        fit_guides = _guides(fit_arrays, float(args.guide_sigma))
        residual_scale = max(
            float(np.std(fit_arrays - fit_guides)),
            1e-3,
        )
        fold_seed = int(args.seed) + 100_000 * fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        rng = np.random.default_rng(fold_seed)
        model = StructureGuidedResidualUNet(
            base_channels=int(args.base_channels),
            embedding_dim=int(args.embedding_dim),
        ).to(device)
        diffusion = StructureGuidedDiffusion(
            timesteps=int(args.timesteps), device=device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=1e-4,
        )
        losses: list[float] = []
        for step in range(1, int(args.steps) + 1):
            model.train()
            residual, guide = _batch(
                fit_arrays,
                fit_guides,
                batch_size=int(args.batch_size),
                patch_size=int(args.patch_size),
                residual_scale=residual_scale,
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(
                model, residual.to(device), guide.to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite diffusion loss in fold {held}, step {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        checkpoint = output / "checkpoints" / f"{held}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "held_growth_group": held,
                "fit_growth_groups": sorted(fit_groups),
                "config": {
                    "base_channels": int(args.base_channels),
                    "embedding_dim": int(args.embedding_dim),
                    "timesteps": int(args.timesteps),
                    "guide_sigma": float(args.guide_sigma),
                    "residual_scale": residual_scale,
                    "resolution": int(config["resolution"]),
                    "strength": float(args.strength),
                    "diffusion_spectral_blend_weight": float(
                        args.blend_weight
                    ),
                },
                "training_steps": int(args.steps),
            },
            checkpoint,
        )
        source = {
            method: _arrays(guide_root / method / f"{held}.npz")[
                : int(args.draws)
            ]
            for method in (M5, M6B, M6C)
        }
        m7 = _refine(
            guides=source[M6B],
            model=model,
            diffusion=diffusion,
            residual_scale=residual_scale,
            steps=int(args.sampling_steps),
            seed=fold_seed + 77,
            strength=float(args.strength),
            device=device,
        )
        m8 = _blend(
            m7,
            source[M6C],
            weight=float(args.blend_weight),
        )
        methods = {
            M5: source[M5],
            M6C: source[M6C],
            M7: m7,
            M8: m8,
        }
        predicted_rq_nm = float(
            np.load(
                guide_root / M6C / f"{held}.npz",
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
                method: {held: predicted_rq_nm} for method in methods
            },
            output_dir=report / "folds" / held / "standard",
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
        for method, arrays in methods.items():
            _save(
                output,
                group=held,
                method=method,
                arrays=arrays,
                predicted_rq_nm=predicted_rq_nm,
            )
        record = {
            "cross_validation_fold": fold,
            "held_growth_group": held,
            "fit_growth_group_count": len(fit_groups),
            "training_steps": int(args.steps),
            "final_training_loss": losses[-1],
            "median_final_100_training_loss": float(
                np.median(losses[-100:])
            ),
            "residual_scale": residual_scale,
            "runtime_seconds": time.perf_counter() - fold_start,
        }
        training_records.append(record)
        print(json.dumps(record), flush=True)
        del model, optimizer, diffusion
        if device.type == "mps":
            torch.mps.empty_cache()
    standard, standard_summary = _aggregate(standard_frames)
    island, island_summary = _aggregate(island_frames)
    write_csv(standard, report / "standard_per_group.csv")
    write_csv(standard_summary, report / "standard_summary.csv")
    write_csv(island, report / "island_per_group.csv")
    write_csv(island_summary, report / "island_summary.csv")
    write_csv(
        pd.DataFrame(training_records), report / "training_by_fold.csv"
    )
    write_json(
        {
            "experiment": "Strict growth-group LOO guided diffusion",
            "methods": [M5, M6C, M7, M8],
            "device": str(device),
            "training_growth_groups": groups,
            "fold_count": len(groups),
            "historical_test_used": False,
            "validation_used": False,
            "removelist_sha256": sha256_file(
                repo_path(config["removelist_path"])
            ),
            "retrieval_at_inference": False,
            "measured_afm_patch_used_at_inference": False,
            "diffusion_steps": int(args.steps),
            "sampling_steps": int(args.sampling_steps),
            "strength": float(args.strength),
            "diffusion_spectral_blend_weight": float(args.blend_weight),
            "runtime_seconds": time.perf_counter() - start_all,
            "standard_summary": standard_summary.to_dict(orient="records"),
            "island_summary": island_summary.to_dict(orient="records"),
        },
        report / "experiment_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--guide-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--guide-sigma", type=float, default=2.2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--sampling-steps", type=int, default=48)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--blend-weight", type=float, default=0.50)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=90210)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
