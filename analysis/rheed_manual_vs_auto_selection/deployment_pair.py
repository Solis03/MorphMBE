from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rheed_video_afm_story.common import (
    repo_path,
    write_csv,
    write_json,
)
from analysis.rheed_video_afm_story.publication_style import (
    set_publication_style,
)
from rheed2morph.realtime.model import RealtimeMorphologyPredictor

from .comparison import _display_contrast, _save_figure
from .dataset import load_config


def _clip(root: Path, group: str) -> np.ndarray:
    return np.asarray(
        np.load(
            root / "selected_16" / f"{group}.npz",
            allow_pickle=False,
        )["frames_uint8"],
        dtype=np.uint8,
    )


def _standardized_l1(first: np.ndarray, second: np.ndarray) -> float:
    a = (np.asarray(first, dtype=float) - float(np.mean(first))) / max(
        float(np.std(first)), 1e-8
    )
    b = (np.asarray(second, dtype=float) - float(np.mean(second))) / max(
        float(np.std(second)), 1e-8
    )
    return float(np.mean(np.abs(a - b)))


def _plot_deployment_atlas(
    *,
    records: pd.DataFrame,
    manifest: pd.DataFrame,
    human_root: Path,
    auto_root: Path,
    map_root: Path,
    figure_root: Path,
) -> None:
    order = records.sort_values("true_rq_nm")["growth_run_id"].tolist()
    manifest_index = manifest.set_index("growth_run_id")
    record_index = records.set_index("growth_run_id")
    for page, start in enumerate(range(0, len(order), 5), start=1):
        groups = order[start : start + 5]
        figure, axes = plt.subplots(
            len(groups), 5, figsize=(12.4, 2.25 * len(groups)), constrained_layout=True
        )
        if len(groups) == 1:
            axes = np.asarray([axes])
        for row_index, group in enumerate(groups):
            row = record_index.loc[group]
            human_clip = _clip(human_root, group)
            auto_clip = _clip(auto_root, group)
            human_map = np.load(
                map_root / "human" / f"{group}.npz", allow_pickle=False
            )["height_nm"]
            auto_map = np.load(
                map_root / "auto" / f"{group}.npz", allow_pickle=False
            )["height_nm"]
            real_map = np.load(
                repo_path(manifest_index.loc[group, "representative_afm_height_array"]),
                allow_pickle=False,
            )
            low = float(np.percentile(np.concatenate([human_map.ravel(), auto_map.ravel(), real_map.ravel()]), 1))
            high = float(np.percentile(np.concatenate([human_map.ravel(), auto_map.ravel(), real_map.ravel()]), 99))
            panels = [
                (
                    _display_contrast(human_clip[7]),
                    "gray",
                    0,
                    1,
                    f"{group} human RHEED",
                ),
                (
                    _display_contrast(auto_clip[7]),
                    "gray",
                    0,
                    1,
                    f"auto RHEED | Cq={100*row['auto_keyframe_quality']:.0f}",
                ),
                (
                    human_map,
                    "afmhot",
                    low,
                    high,
                    f"same model: human\nRq={row['human_predicted_rq_nm']:.2f}",
                ),
                (
                    auto_map,
                    "afmhot",
                    low,
                    high,
                    (
                        f"same model: auto\nRq={row['auto_predicted_rq_nm']:.2f}, "
                        f"C={100*row['auto_model_confidence']:.0f}"
                    ),
                ),
                (
                    real_map,
                    "afmhot",
                    low,
                    high,
                    f"measured AFM\nRq={row['true_rq_nm']:.2f}",
                ),
            ]
            for axis, (image, cmap, vmin, vmax, title) in zip(axes[row_index], panels):
                axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_title(title, fontsize=7.5)
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            (
                "Paired sensitivity of the same all-23 deployment weights "
                f"to human versus automatic RHEED inputs ({page}/5; "
                "RHEED contrast normalized for display only)"
            ),
            fontsize=11,
        )
        _save_figure(
            figure,
            figure_root / f"Fig5_same_weights_deployment_atlas_page_{page:02d}",
        )


def run(config: dict[str, Any], *, device_name: str) -> None:
    set_publication_style()
    started = time.time()
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    map_root = output_root.parent / "paired_deployment_maps"
    figure_root = report_root / "figures"
    human_root = repo_path(config["human_clip_root"])
    auto_root = output_root / "clip_variants"
    manifest = pd.read_csv(
        repo_path(config["human_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    selection = pd.read_csv(
        output_root / "selection_comparison.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    groups = selection["growth_run_id"].astype(str).tolist()
    manifest = (
        manifest.loc[manifest["growth_run_id"].isin(groups)]
        .set_index("growth_run_id")
        .loc[groups]
        .reset_index()
    )
    predictor = RealtimeMorphologyPredictor.from_path(
        "outputs/rheed_realtime_ui/morphmbe_m14i_m12a_live_v1.joblib",
        device=device_name,
    )
    selection_index = selection.set_index("growth_run_id")
    records: list[dict[str, Any]] = []
    for position, group in enumerate(groups, start=1):
        print(f"[{position:02d}/23] paired all-23 deployment {group}", flush=True)
        seed = 141421 + position * 10_000
        human = predictor.predict(
            _clip(human_root, group),
            keyframe_quality=1.0,
            seed=seed,
        )
        auto_quality = float(
            selection_index.loc[group, "machine_keyframe_quality"]
        )
        automatic = predictor.predict(
            _clip(auto_root, group),
            keyframe_quality=auto_quality,
            seed=seed,
        )
        for label, prediction in (("human", human), ("auto", automatic)):
            path = map_root / label / f"{group}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                height_nm=prediction.height_nm.astype(np.float32),
                unit_shape=prediction.unit_shape.astype(np.float32),
                predicted_rq_nm=np.asarray(prediction.rq.value),
                predicted_fsmi_nm=np.asarray(prediction.fsmi.value),
                model_confidence=np.asarray(prediction.model_confidence),
                selection_source=np.asarray(label),
                same_deployment_weights=np.asarray(True),
                strict_held_out=np.asarray(False),
            )
        manifest_row = manifest.set_index("growth_run_id").loc[group]
        records.append(
            {
                "growth_run_id": group,
                "true_rq_nm": float(manifest_row["primary_rq_nm_median"]),
                "human_predicted_rq_nm": human.rq.value,
                "auto_predicted_rq_nm": automatic.rq.value,
                "absolute_rq_input_shift_nm": abs(
                    automatic.rq.value - human.rq.value
                ),
                "human_predicted_fsmi_nm": human.fsmi.value,
                "auto_predicted_fsmi_nm": automatic.fsmi.value,
                "absolute_fsmi_input_shift_nm": abs(
                    automatic.fsmi.value - human.fsmi.value
                ),
                "human_model_confidence": human.model_confidence,
                "auto_model_confidence": automatic.model_confidence,
                "auto_combined_confidence": automatic.combined_confidence,
                "auto_keyframe_quality": auto_quality,
                "generated_map_standardized_l1": _standardized_l1(
                    human.height_nm, automatic.height_nm
                ),
                "same_deployment_weights": True,
                "strict_held_out": False,
                "purpose": "paired input-domain sensitivity only",
            }
        )
    table = pd.DataFrame(records)
    write_csv(table, report_root / "same_weights_deployment_sensitivity.csv")
    _plot_deployment_atlas(
        records=table,
        manifest=manifest,
        human_root=human_root,
        auto_root=auto_root,
        map_root=map_root,
        figure_root=figure_root,
    )
    summary = {
        "protocol": (
            "same frozen all-23 deployment weights and paired random seed; "
            "only human versus automatic model input changes"
        ),
        "strict_held_out": False,
        "claim_boundary": (
            "This paired deployment experiment isolates input sensitivity. "
            "It is not held-out evidence because the all-23 deployment bundle "
            "was fitted using each growth's human-selected training row."
        ),
        "growth_group_count": len(table),
        "median_absolute_rq_input_shift_nm": float(
            table["absolute_rq_input_shift_nm"].median()
        ),
        "median_absolute_fsmi_input_shift_nm": float(
            table["absolute_fsmi_input_shift_nm"].median()
        ),
        "median_generated_map_standardized_l1": float(
            table["generated_map_standardized_l1"].median()
        ),
        "median_auto_model_confidence": float(
            table["auto_model_confidence"].median()
        ),
        "runtime_seconds": time.time() - started,
    }
    write_json(summary, report_root / "same_weights_deployment_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_manual_vs_auto_selection.json",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config), device_name=str(args.device))


if __name__ == "__main__":
    main()
