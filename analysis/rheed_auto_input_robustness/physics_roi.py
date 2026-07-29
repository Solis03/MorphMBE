from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_auto_roi_keyframe.train_full_lattice_roi import (
    fit_bundle,
    rect_iou,
)
from analysis.rheed_manual_vs_auto_selection.dataset import (
    _decode_selected16,
)
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    _range_calibrate,
)
from analysis.rheed_to_afm_ood_robust.prediction import _ridge_prediction
from analysis.rheed_to_afm_ood_robust.support import density_weights
from rheed2morph.rheed.automatic_roi_keyframe import (
    Rect,
    analyze_aperture,
    sample_frames,
)
from rheed2morph.rheed.lattice_roi import (
    normalized_roi_bounds,
    orientation_group,
    predict_full_lattice_roi,
)
from rheed2morph.realtime.clips import build_model_clip, live_physics_row


PHYSICS_ROI_QUANTILES = (0.50, 0.50, 0.50, 0.50)


def _roi_config() -> dict[str, float]:
    return {
        "arc_margin_analysis_pixels": 2.0,
        "right_boundary_margin_analysis_pixels": 1.0,
        "left_padding_aperture_fraction": 0.0,
        "right_padding_aperture_fraction": 0.0,
        "top_padding_aperture_fraction": 0.0,
        "bottom_padding_aperture_fraction": 0.0,
        "minimum_width_fraction": 0.40,
    }


def _manual_rect(row: pd.Series) -> Rect:
    return Rect(
        int(row["roi_x"]),
        int(row["roi_y"]),
        int(row["roi_width"]),
        int(row["roi_height"]),
        int(row["source_width"]),
        int(row["source_height"]),
    )


def _coverage(predicted: Rect, manual: Rect) -> float:
    x0 = max(predicted.x, manual.x)
    y0 = max(predicted.y, manual.y)
    x1 = min(predicted.x2, manual.x2)
    y1 = min(predicted.y2, manual.y2)
    area = max(x1 - x0, 0) * max(y1 - y0, 0)
    return float(area / max(manual.width * manual.height, 1))


def build_physics_roi_dataset(
    *,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    output_root: Path,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Build strict held-video physics ROIs and a final 23-video bundle."""

    rows = selection.copy()
    rows["growth_run_id"] = rows["growth_run_id"].astype(str)
    rows = rows.set_index("growth_run_id")
    metadata = manifest.copy()
    metadata["growth_run_id"] = metadata["growth_run_id"].astype(str)
    metadata = metadata.set_index("growth_run_id").loc[rows.index]
    if (
        "selection_source" in metadata
        and metadata["selection_source"]
        .astype(str)
        .str.contains("automatic", case=False)
        .any()
    ):
        raise ValueError(
            "physics ROI calibration requires the original human ROI "
            "manifest, not a machine-updated manifest"
        )
    groups = rows.index.tolist()
    if len(groups) != 23 or set(groups) & {"6043", "6055"}:
        raise RuntimeError("physics ROI requires the fixed allowed Full23 cohort")

    analyses = {}
    targets = []
    manual_rects = {}
    for position, group in enumerate(groups, start=1):
        sampled, _ = sample_frames(
            Path(str(rows.loc[group, "source_video"])),
            maximum=48,
        )
        analysis = analyze_aperture(sampled)
        analyses[group] = analysis
        manual = _manual_rect(metadata.loc[group])
        manual_rects[group] = manual
        targets.append(
            {
                "sample_id": group,
                "orientation": orientation_group(analysis),
                **normalized_roi_bounds(manual, analysis),
            }
        )
        if progress:
            print(
                f"[physics ROI aperture {position:02d}/{len(groups):02d}] "
                f"{group}",
                flush=True,
            )
    target_table = pd.DataFrame(targets)
    output_root.mkdir(parents=True, exist_ok=True)
    target_table.to_csv(
        output_root / "normalized_physics_roi_targets.csv",
        index=False,
    )

    config = _roi_config()
    roi_records = []
    physics_records = []
    for position, group in enumerate(groups, start=1):
        training = target_table.loc[target_table["sample_id"] != group]
        bundle = fit_bundle(
            training,
            grouped=True,
            quantiles=PHYSICS_ROI_QUANTILES,
            config=config,
        )
        predicted = predict_full_lattice_roi(
            analyses[group],
            bundle,
        ).rect
        manual = manual_rects[group]
        selection_row = rows.loc[group]
        decoded = _decode_selected16(
            Path(str(selection_row["source_video"])),
            int(selection_row["machine_keyframe_index"]),
        )
        clip = build_model_clip(decoded, predicted)
        physics = live_physics_row(
            clip,
            sample_id=group,
        ).reset_index(drop=True)
        physics["growth_run_id"] = group
        physics["video_stage"] = "machine_keyframe_physics_roi_q50"
        physics_records.append(physics)
        roi_records.append(
            {
                "growth_run_id": group,
                "physics_roi_x": predicted.x,
                "physics_roi_y": predicted.y,
                "physics_roi_width": predicted.width,
                "physics_roi_height": predicted.height,
                "manual_roi_iou": rect_iou(predicted, manual),
                "manual_roi_coverage": _coverage(predicted, manual),
                "held_roi_annotation_used_for_fit": False,
                "roi_fit_growth_count": len(training),
            }
        )
        if progress:
            print(
                f"[physics ROI features {position:02d}/{len(groups):02d}] "
                f"{group}",
                flush=True,
            )
    roi_table = pd.DataFrame(roi_records)
    physics_table = pd.concat(physics_records, ignore_index=True)
    roi_table.to_csv(
        output_root / "physics_roi_strict_loo_predictions.csv",
        index=False,
    )
    physics_table.to_csv(
        output_root / "physics_roi_strict_loo_features.csv",
        index=False,
    )

    final_bundle = fit_bundle(
        target_table,
        grouped=True,
        quantiles=PHYSICS_ROI_QUANTILES,
        config=config,
    )
    final_bundle.update(
        {
            "model_family": (
                "orientation_conditioned_median_geometry_physics_roi"
            ),
            "training_video_count": len(groups),
            "training_sample_ids": groups,
            "excluded_growths": ["6043", "6055"],
            "validation_protocol": "strict_leave_one_video_out",
            "role": (
                "handcrafted physics features only; never replaces the "
                "complete-lattice R3D/generator ROI"
            ),
            "afm_targets_used_for_roi_fit": False,
        }
    )
    bundle_path = output_root / "physics_roi_calibration.joblib"
    joblib.dump(final_bundle, bundle_path, compress=3)
    return roi_table, physics_table, bundle_path


def crossfit_density_physics(
    *,
    physics: pd.DataFrame,
    true_target: pd.Series,
) -> pd.DataFrame:
    """Reproduce the M14b physics-only head on a supplied feature domain."""

    frame = physics.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame = frame.set_index("sample_id", drop=False)
    target = true_target.copy()
    target.index = target.index.astype(str)
    groups = target.index.tolist()
    log_target = np.log(target)
    records = []
    for held in groups:
        fit = [group for group in groups if group != held]
        weights = (
            density_weights(
                frame,
                fit,
                strength=0.5,
                floor=0.25,
            )
            .set_index("growth_run_id")
            .loc[fit, "density_sample_weight"]
            .to_numpy(float)
        )
        outer_log = _ridge_prediction(
            physics=frame,
            log_target=log_target,
            fit_groups=fit,
            query_group=held,
            alpha=10.0,
            morphology_weight=0.75,
            sample_weights=weights,
        )
        inner_raw = []
        inner_truth = []
        for inner_held in fit:
            inner_fit = [
                group for group in fit if group != inner_held
            ]
            inner_weights = (
                density_weights(
                    frame,
                    inner_fit,
                    strength=0.5,
                    floor=0.25,
                )
                .set_index("growth_run_id")
                .loc[inner_fit, "density_sample_weight"]
                .to_numpy(float)
            )
            inner_log = _ridge_prediction(
                physics=frame,
                log_target=log_target,
                fit_groups=inner_fit,
                query_group=inner_held,
                alpha=10.0,
                morphology_weight=0.75,
                sample_weights=inner_weights,
            )
            inner_raw.append(float(np.exp(inner_log)))
            inner_truth.append(float(target.loc[inner_held]))
        predicted, _ = _range_calibrate(
            np.asarray(inner_raw),
            np.asarray(inner_truth),
            float(np.exp(outer_log)),
        )
        records.append(
            {
                "growth_run_id": held,
                "true_target": float(target.loc[held]),
                "predicted_target": predicted,
                "absolute_error": abs(predicted - float(target.loc[held])),
                "outer_target_used_for_training": False,
            }
        )
    return pd.DataFrame(records)


def save_physics_roi_figure(
    roi_table: pd.DataFrame,
    *,
    current_selection: pd.DataFrame,
    destination: Path,
) -> None:
    rows = roi_table.set_index("growth_run_id")
    current = current_selection.copy()
    current["growth_run_id"] = current["growth_run_id"].astype(str)
    current = current.set_index("growth_run_id").loc[rows.index]
    current_iou = current["roi_iou"].to_numpy(float)
    physics_iou = rows["manual_roi_iou"].to_numpy(float)
    figure, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.6), constrained_layout=True
    )
    axes[0].boxplot(
        [current_iou, physics_iou],
        tick_labels=["V8 complete\nmodel input", "Q50 dedicated\nphysics ROI"],
        showmeans=True,
    )
    axes[0].set_ylabel("IoU with human physics crop")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.18)
    order = np.argsort(physics_iou)
    axes[1].plot(
        np.arange(len(rows)),
        current_iou[order],
        "o-",
        ms=3,
        label="V8 complete ROI",
    )
    axes[1].plot(
        np.arange(len(rows)),
        physics_iou[order],
        "o-",
        ms=3,
        label="physics ROI",
    )
    axes[1].set_xticks(np.arange(len(rows)))
    axes[1].set_xticklabels(
        rows.index.to_numpy()[order],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    axes[1].set_ylabel("IoU with human physics crop")
    axes[1].set_xlabel("Held growth")
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=7)
    figure.suptitle(
        "Role-separated automatic ROIs: complete pattern vs stable physics",
        fontsize=12,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination.with_suffix(".png"), dpi=300, bbox_inches="tight"
    )
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
