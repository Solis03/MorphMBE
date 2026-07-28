from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import ndimage
import torch

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_sharp_generation.spectral import load_unit_map
from analysis.rheed_video_afm_story.common import (
    repo_path,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .guided_diffusion import (
    StructureGuidedDiffusion,
    StructureGuidedResidualUNet,
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config(path: str | Path) -> dict:
    config = json.loads(repo_path(path).read_text(encoding="utf-8"))
    parent = json.loads(
        repo_path(config["parent_config"]).read_text(encoding="utf-8")
    )
    return {**parent, **config}


def _maps(rows: pd.DataFrame, resolution: int) -> tuple[np.ndarray, list[str]]:
    arrays = []
    groups = []
    for _, row in rows.iterrows():
        arrays.append(load_unit_map(row, resolution))
        groups.append(str(row["growth_run_id"]))
    return np.stack(arrays).astype(np.float32), groups


def _guides(arrays: np.ndarray, sigma: float) -> np.ndarray:
    result = np.stack(
        [
            project_unit_rq_np(
                ndimage.gaussian_filter(array, sigma=sigma, mode="reflect")
            )
            for array in arrays
        ]
    )
    return result.astype(np.float32)


def _batch(
    arrays: np.ndarray,
    guides: np.ndarray,
    *,
    batch_size: int,
    patch_size: int,
    residual_scale: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    conditions = []
    height, width = arrays.shape[-2:]
    for _ in range(int(batch_size)):
        index = int(rng.integers(0, len(arrays)))
        y = int(rng.integers(0, height - patch_size + 1))
        x = int(rng.integers(0, width - patch_size + 1))
        target = arrays[index, y : y + patch_size, x : x + patch_size]
        guide = guides[index, y : y + patch_size, x : x + patch_size]
        rotations = int(rng.integers(0, 4))
        target = np.rot90(target, rotations)
        guide = np.rot90(guide, rotations)
        if rng.random() < 0.5:
            target = np.flip(target, axis=0)
            guide = np.flip(guide, axis=0)
        if rng.random() < 0.5:
            target = np.flip(target, axis=1)
            guide = np.flip(guide, axis=1)
        targets.append(np.ascontiguousarray((target - guide) / residual_scale))
        conditions.append(np.ascontiguousarray(guide))
    residual = torch.from_numpy(np.stack(targets)[:, None].astype(np.float32))
    guide_tensor = torch.from_numpy(
        np.stack(conditions)[:, None].astype(np.float32)
    )
    return residual, guide_tensor


@torch.no_grad()
def _validation_loss(
    *,
    model: StructureGuidedResidualUNet,
    diffusion: StructureGuidedDiffusion,
    arrays: np.ndarray,
    guides: np.ndarray,
    batch_size: int,
    patch_size: int,
    residual_scale: float,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    model.eval()
    residual, guide = _batch(
        arrays,
        guides,
        batch_size=batch_size,
        patch_size=patch_size,
        residual_scale=residual_scale,
        rng=rng,
    )
    return float(
        diffusion.training_loss(
            model, residual.to(device), guide.to(device)
        )
        .detach()
        .cpu()
    )


def train(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    all_groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    validation_groups = [] if args.fit_all else all_groups[3::5]
    fit_rows = train_rows.loc[
        ~train_rows["growth_run_id"].astype(str).isin(validation_groups)
    ]
    refiner_validation_rows = train_rows.loc[
        train_rows["growth_run_id"].astype(str).isin(validation_groups)
    ]
    fit_arrays, fit_array_groups = _maps(
        fit_rows, int(config["resolution"])
    )
    validation_arrays = None
    validation_array_groups: list[str] = []
    if not args.fit_all:
        validation_arrays, validation_array_groups = _maps(
            refiner_validation_rows, int(config["resolution"])
        )
    fit_guides = _guides(fit_arrays, float(args.guide_sigma))
    validation_guides = (
        None
        if validation_arrays is None
        else _guides(validation_arrays, float(args.guide_sigma))
    )
    residual_scale = float(
        np.std(fit_arrays - fit_guides)
    )
    residual_scale = max(residual_scale, 1e-3)
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
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
    records = []
    best = float("inf")
    start = time.perf_counter()
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
        residual = residual.to(device)
        guide = guide.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(model, residual, guide)
        if not torch.isfinite(loss):
            raise RuntimeError("guided diffusion loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % int(args.evaluate_every) == 0:
            validation_loss = (
                float("nan")
                if validation_arrays is None or validation_guides is None
                else _validation_loss(
                    model=model,
                    diffusion=diffusion,
                    arrays=validation_arrays,
                    guides=validation_guides,
                    batch_size=int(args.batch_size),
                    patch_size=int(args.patch_size),
                    residual_scale=residual_scale,
                    rng=rng,
                    device=device,
                )
            )
            record = {
                "step": step,
                "training_loss": float(loss.detach().cpu()),
                "validation_loss": validation_loss,
                "elapsed_seconds": time.perf_counter() - start,
            }
            records.append(record)
            if not args.fit_all and validation_loss < best:
                best = validation_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {
                            "base_channels": int(args.base_channels),
                            "embedding_dim": int(args.embedding_dim),
                            "timesteps": int(args.timesteps),
                            "guide_sigma": float(args.guide_sigma),
                            "residual_scale": residual_scale,
                            "resolution": int(config["resolution"]),
                        },
                        "fit_growth_groups": sorted(
                            set(fit_array_groups)
                        ),
                        "refiner_validation_growth_groups": validation_groups,
                        "step": step,
                        "best_validation_loss": best,
                    },
                    output / "best_guided_diffusion.pt",
                )
            print(json.dumps(record), flush=True)
    if args.fit_all:
        best = float(records[-1]["training_loss"])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "base_channels": int(args.base_channels),
                    "embedding_dim": int(args.embedding_dim),
                    "timesteps": int(args.timesteps),
                    "guide_sigma": float(args.guide_sigma),
                    "residual_scale": residual_scale,
                    "resolution": int(config["resolution"]),
                },
                "fit_growth_groups": sorted(set(fit_array_groups)),
                "refiner_validation_growth_groups": [],
                "step": int(args.steps),
                "best_validation_loss": float("nan"),
                "fixed_step_final_fit": True,
            },
            output / "best_guided_diffusion.pt",
        )
    write_csv(pd.DataFrame(records), output / "training_curves.csv")
    write_json(
        {
            "device": str(device),
            "steps": int(args.steps),
            "runtime_seconds": time.perf_counter() - start,
            "best_validation_loss": best,
            "fit_growth_groups": sorted(set(fit_array_groups)),
            "refiner_validation_growth_groups": validation_groups,
            "refiner_validation_scan_groups": sorted(
                set(validation_array_groups)
            ),
            "fit_all_training_growth_groups": bool(args.fit_all),
            "historical_test_used": False,
            "residual_scale": residual_scale,
            "guide_sigma": float(args.guide_sigma),
        },
        output / "training_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--guide-sigma", type=float, default=2.2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--fit-all", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
