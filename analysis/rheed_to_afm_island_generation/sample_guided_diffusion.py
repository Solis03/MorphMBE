from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_to_afm_sharp_generation.spectral import load_unit_map
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

from .evaluation import evaluate_island_methods
from .guided_diffusion import (
    StructureGuidedDiffusion,
    StructureGuidedResidualUNet,
)
from .train_guided_diffusion import _device, _load_config


M6B = "M6b_multiscale_laguerre_terraces"
M6C = "M6c_island_structure_plus_spectral_prior"
M7 = "M7_structure_guided_residual_diffusion"


def _blend_name(weight: float) -> str:
    return f"M8_diffusion_spectral_blend_w{float(weight):.2f}"


def _blend(
    diffusion: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    weight: float,
) -> list[np.ndarray]:
    return [
        project_unit_rq_np(
            float(weight) * diffusion[index % len(diffusion)]
            + (1.0 - float(weight)) * spectral[index % len(spectral)]
        ).astype(np.float32)
        for index in range(max(len(diffusion), len(spectral)))
    ]


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[StructureGuidedResidualUNet, StructureGuidedDiffusion, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model_config = checkpoint["config"]
    model = StructureGuidedResidualUNet(
        base_channels=int(model_config["base_channels"]),
        embedding_dim=int(model_config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = StructureGuidedDiffusion(
        timesteps=int(model_config["timesteps"]), device=device
    )
    return model, diffusion, checkpoint


def _arrays(path: Path) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return [
        np.asarray(value, dtype=np.float32)
        for value in payload["generated_unit_shapes"]
    ]


@torch.no_grad()
def _refine(
    *,
    guides: list[np.ndarray],
    model: StructureGuidedResidualUNet,
    diffusion: StructureGuidedDiffusion,
    residual_scale: float,
    steps: int,
    seed: int,
    strength: float,
    device: torch.device,
) -> list[np.ndarray]:
    refined = []
    for index, guide in enumerate(guides):
        tensor = torch.from_numpy(guide[None, None]).to(
            device=device, dtype=torch.float32
        )
        residual = diffusion.sample(
            model,
            tensor,
            steps=int(steps),
            seed=int(seed) + index,
            strength=float(strength),
        )
        generated = (
            guide + float(residual_scale) * residual[0, 0].cpu().numpy()
        )
        refined.append(project_unit_rq_np(generated).astype(np.float32))
    return refined


def _figure(
    *,
    groups: list[str],
    rows: pd.DataFrame,
    guides: dict[str, list[np.ndarray]],
    refined: dict[str, list[np.ndarray]],
    output: Path,
    resolution: int,
) -> None:
    figure, axes = plt.subplots(
        len(groups), 3, figsize=(8.2, 2.45 * len(groups)), squeeze=False
    )
    for index, group in enumerate(groups):
        held = rows.loc[rows["growth_run_id"].astype(str) == group].iloc[0]
        real = load_unit_map(held, resolution)
        images = [guides[group][0], refined[group][0], real]
        titles = ["M6b structure guide", "M7 diffusion", "Real AFM"]
        low, high = np.percentile(np.concatenate([x.ravel() for x in images]), [1, 99])
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[index, column].imshow(
                image, cmap="afmhot", vmin=low, vmax=high
            )
            axes[index, column].set_axis_off()
            axes[index, column].set_title(
                f"{title}\nGrowth {group}" if column == 0 else title,
                fontsize=9,
            )
    figure.suptitle(
        "Structure-guided residual diffusion smoke evaluation "
        "(unit-Rq morphology)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def sample(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    model, diffusion, checkpoint = _load_model(
        repo_path(args.checkpoint), device
    )
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[descriptors["split"] == "val"].copy()
    condition_scaler = ConditionScaler.fit(
        train_rows,
        list(config["condition_columns"]),
        set(train_rows["growth_run_id"].astype(str).unique()),
    )
    groups = sorted(validation_rows["growth_run_id"].astype(str).unique())
    guide_root = repo_path(args.guide_root)
    guides: dict[str, list[np.ndarray]] = {}
    reference: dict[str, list[np.ndarray]] = {}
    refined: dict[str, list[np.ndarray]] = {}
    predicted_rq: dict[str, float] = {}
    for group_index, group in enumerate(groups):
        guide_path = guide_root / M6B / f"{group}.npz"
        reference_path = guide_root / M6C / f"{group}.npz"
        guides[group] = _arrays(guide_path)[: int(args.draws)]
        reference[group] = _arrays(reference_path)[: int(args.draws)]
        refined[group] = _refine(
            guides=guides[group],
            model=model,
            diffusion=diffusion,
            residual_scale=float(checkpoint["config"]["residual_scale"]),
            steps=int(args.sampling_steps),
            seed=int(args.seed) + 10_000 * group_index,
            strength=float(args.strength),
            device=device,
        )
        predicted_rq[group] = float(
            np.load(guide_path, allow_pickle=False)["predicted_rq_nm"]
        )
        np.savez_compressed(
            output / f"{group}.npz",
            generated_unit_shapes=np.stack(refined[group]).astype(np.float32),
            predicted_rq_nm=np.asarray(predicted_rq[group]),
            method=np.asarray(M7),
            retrieval_at_inference=np.asarray(False),
            measured_afm_patch_used_at_inference=np.asarray(False),
        )
    generated = {M6B: guides, M6C: reference, M7: refined}
    for weight in args.reference_blend_weights:
        generated[_blend_name(float(weight))] = {
            group: _blend(
                refined[group],
                reference[group],
                weight=float(weight),
            )
            for group in groups
        }
    for method, method_groups in generated.items():
        for group, arrays in method_groups.items():
            path = output / "generated_maps" / method / f"{group}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                generated_unit_shapes=np.stack(arrays).astype(np.float32),
                predicted_rq_nm=np.asarray(predicted_rq[group]),
                growth_run_id=np.asarray(group),
                method=np.asarray(method),
                retrieval_at_inference=np.asarray(False),
                measured_afm_patch_used_at_inference=np.asarray(False),
            )
    rq = {
        method: {group: predicted_rq[group] for group in groups}
        for method in generated
    }
    standard = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        generated=generated,
        generated_rq=rq,
        output_dir=output / "standard",
        resolution=int(config["resolution"]),
    )
    island_frames = []
    for group in groups:
        held = validation_rows.loc[
            validation_rows["growth_run_id"].astype(str) == group
        ]
        island_frames.append(
            evaluate_island_methods(
                held_rows=held,
                train_rows=train_rows,
                generated={
                    method: values[group] for method, values in generated.items()
                },
                resolution=int(config["resolution"]),
            )
        )
    island = pd.concat(island_frames, ignore_index=True)
    write_csv(island, output / "island_per_group.csv")
    write_csv(
        island.groupby("method").median(numeric_only=True).reset_index(),
        output / "island_summary.csv",
    )
    _figure(
        groups=groups,
        rows=validation_rows,
        guides=guides,
        refined=refined,
        output=output / "validation_visual_smoke.png",
        resolution=int(config["resolution"]),
    )
    write_json(
        {
            "checkpoint": str(repo_path(args.checkpoint)),
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_validation_loss": float(
                checkpoint["best_validation_loss"]
            ),
            "device": str(device),
            "sampling_steps": int(args.sampling_steps),
            "strength": float(args.strength),
            "reference_blend_weights": [
                float(value) for value in args.reference_blend_weights
            ],
            "growth_groups": groups,
            "historical_test_used": False,
            "retrieval_at_inference": False,
            "measured_afm_patch_used_at_inference": False,
            "standard_summary": standard["summary"].to_dict(orient="records"),
        },
        output / "sampling_manifest.json",
    )
    print(json.dumps(standard["summary"].to_dict(orient="records"), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--guide-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--reference-blend-weights",
        type=float,
        nargs="*",
        default=[],
    )
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=81173)
    return parser


def main() -> None:
    sample(build_parser().parse_args())


if __name__ == "__main__":
    main()
