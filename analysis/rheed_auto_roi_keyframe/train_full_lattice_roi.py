#!/usr/bin/env python3
"""Train and evaluate complete-lattice RHEED ROI calibration.

Each outer fold excludes one complete video.  Four independent ROI boundaries
are calibrated only from the remaining videos.  The proposed rectangle is
then constrained to start inside the circular eyepiece arc while retaining
the stable right light/shadow boundary.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage
from skimage.feature import peak_local_max


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rheed2morph.rheed.automatic_roi_keyframe import (  # noqa: E402
    ApertureAnalysis,
    Rect,
    analyze_aperture,
    load_frame,
    sample_frames,
)
from rheed2morph.rheed.lattice_roi import (  # noqa: E402
    BOUNDARY_NAMES,
    circular_arc_intrusion_fraction,
    normalized_roi_bounds,
    orientation_group,
    predict_full_lattice_roi,
)


METHODS = {
    "v7_global_median": {
        "grouped": False,
        "quantiles": (0.50, 0.50, 0.50, 0.50),
    },
    "v8_orientation_model_input_q25_q75": {
        "grouped": True,
        "quantiles": (0.25, 0.75, 0.25, 0.75),
    },
    "v8_orientation_model_input_q20_q80": {
        "grouped": True,
        "quantiles": (0.20, 0.80, 0.20, 0.80),
    },
    "v8_orientation_model_input_q15_q85": {
        "grouped": True,
        "quantiles": (0.15, 0.85, 0.15, 0.85),
    },
    "v7_global_conservative_q10_q90": {
        "grouped": False,
        "quantiles": (0.10, 0.90, 0.10, 0.90),
    },
    "v7_orientation_conservative_q10_q90": {
        "grouped": True,
        "quantiles": (0.10, 0.90, 0.10, 0.90),
    },
    "v7_orientation_conservative_q05_q95": {
        "grouped": True,
        "quantiles": (0.05, 0.95, 0.05, 0.95),
    },
    "v7_orientation_extreme": {
        "grouped": True,
        "quantiles": (0.00, 1.00, 0.00, 1.00),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_auto_roi_keyframe_v7.json",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_removed_ids(config: dict[str, Any]) -> set[str]:
    lines = (ROOT / config["removelist"]).read_text(
        encoding="utf-8"
    ).splitlines()
    return {
        line.split(maxsplit=1)[0]
        for line in lines
        if line.strip() and line.split(maxsplit=1)[0].isdigit()
    }


def load_manifest(config: dict[str, Any]) -> pd.DataFrame:
    manifest = pd.read_csv(
        ROOT / config["manifest"], dtype={"sample_id": str}
    )
    excluded = manifest["excluded_by_removelist"]
    if excluded.dtype != bool:
        excluded = excluded.astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    removed_ids = load_removed_ids(config)
    flagged_ids = set(manifest.loc[excluded, "sample_id"])
    expected_ids = set(config["expected_excluded_sample_ids"])
    in_cohort = set(manifest["sample_id"]) & removed_ids
    if flagged_ids != expected_ids or in_cohort != expected_ids:
        raise ValueError(
            "Manifest/removelist exclusion mismatch: "
            f"flagged={sorted(flagged_ids)}, "
            f"in_cohort={sorted(in_cohort)}, "
            f"expected={sorted(expected_ids)}"
        )
    retained = manifest.loc[~excluded].copy()
    if len(retained) != int(config["expected_video_count"]):
        raise ValueError(
            f"Expected {config['expected_video_count']} videos, "
            f"found {len(retained)}"
        )
    return retained.sort_values("sample_id").reset_index(drop=True)


def rect_iou(first: Rect, second: Rect) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x2, second.x2)
    y1 = min(first.y2, second.y2)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    return float(intersection / max(union, 1))


def manual_area_coverage(predicted: Rect, manual: Rect) -> float:
    x0 = max(predicted.x, manual.x)
    y0 = max(predicted.y, manual.y)
    x1 = min(predicted.x2, manual.x2)
    y1 = min(predicted.y2, manual.y2)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    return float(intersection / max(manual.width * manual.height, 1))


def _brightness(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame)
    if values.ndim == 2:
        result = values.astype(np.float32)
    else:
        result = values[..., :3].astype(np.float32).max(axis=2)
    if float(result.max()) > 1.5:
        result /= 255.0
    return np.clip(result, 0.0, 1.0)


def spot_support_metrics(
    frame: np.ndarray,
    predicted: Rect,
    manual: Rect,
    *,
    maximum_side: int = 320,
) -> tuple[float, float, int]:
    """Measure retained compact-spot response inside the human ROI.

    This is evaluation-only.  The manual rectangle defines a generous region
    in which compact DoG maxima are measured; no manual pixels or labels enter
    inference.
    """

    ys, xs = manual.as_slices()
    crop = _brightness(frame)[ys, xs]
    scale = min(1.0, maximum_side / max(crop.shape))
    width = max(24, int(round(crop.shape[1] * scale)))
    height = max(24, int(round(crop.shape[0] * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.float32)).resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    low, high = np.percentile(resized, [2.0, 99.8])
    normalized = np.clip(
        (resized - low) / max(float(high - low), 1e-5),
        0.0,
        1.0,
    )
    response = ndimage.gaussian_filter(
        normalized, 0.8
    ) - ndimage.gaussian_filter(normalized, 3.4)
    median = float(np.median(response))
    noise = float(
        1.4826 * np.median(np.abs(response - median)) + 1e-6
    )
    threshold = max(
        float(np.quantile(response, 0.975)),
        median + 3.0 * noise,
        0.008,
    )
    energy = np.maximum(response - threshold, 0.0)
    peaks = peak_local_max(
        response,
        min_distance=3,
        threshold_abs=threshold,
        exclude_border=False,
    )

    overlap_x0 = max(predicted.x, manual.x)
    overlap_y0 = max(predicted.y, manual.y)
    overlap_x1 = min(predicted.x2, manual.x2)
    overlap_y1 = min(predicted.y2, manual.y2)
    mask = np.zeros_like(energy, dtype=bool)
    if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
        x0 = int(
            np.floor((overlap_x0 - manual.x) / manual.width * width)
        )
        x1 = int(
            np.ceil((overlap_x1 - manual.x) / manual.width * width)
        )
        y0 = int(
            np.floor((overlap_y0 - manual.y) / manual.height * height)
        )
        y1 = int(
            np.ceil((overlap_y1 - manual.y) / manual.height * height)
        )
        mask[
            max(0, y0) : min(height, y1),
            max(0, x0) : min(width, x1),
        ] = True
    total_energy = float(energy.sum())
    energy_coverage = (
        float(energy[mask].sum() / total_energy)
        if total_energy > 1e-8
        else manual_area_coverage(predicted, manual)
    )
    if len(peaks):
        peak_coverage = float(np.mean(mask[tuple(peaks.T)]))
    else:
        peak_coverage = energy_coverage
    return energy_coverage, peak_coverage, int(len(peaks))


def fit_bundle(
    targets: pd.DataFrame,
    *,
    grouped: bool,
    quantiles: tuple[float, float, float, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    def fit_group(group: pd.DataFrame) -> dict[str, float]:
        return {
            name: float(group[name].quantile(quantile))
            for name, quantile in zip(BOUNDARY_NAMES, quantiles)
        }

    groups = {"global": fit_group(targets)}
    if grouped:
        for name, group in targets.groupby("orientation"):
            if len(group) >= 4:
                groups[str(name)] = fit_group(group)
    return {
        "schema_version": 1,
        "model_family": (
            "orientation_conditioned_conservative_aperture_geometry"
        ),
        "groups": groups,
        "quantiles": {
            name: float(value)
            for name, value in zip(BOUNDARY_NAMES, quantiles)
        },
        "grouped_by_orientation": bool(grouped),
        "arc_margin_analysis_pixels": float(
            config["arc_margin_analysis_pixels"]
        ),
        "right_boundary_margin_analysis_pixels": float(
            config["right_boundary_margin_analysis_pixels"]
        ),
        "left_padding_aperture_fraction": float(
            config["left_padding_aperture_fraction"]
        ),
        "right_padding_aperture_fraction": float(
            config["right_padding_aperture_fraction"]
        ),
        "top_padding_aperture_fraction": float(
            config["top_padding_aperture_fraction"]
        ),
        "bottom_padding_aperture_fraction": float(
            config["bottom_padding_aperture_fraction"]
        ),
        "minimum_width_fraction": float(
            config["minimum_width_fraction"]
        ),
    }


def load_baseline_rect(
    sample: str,
    manifest_row: pd.Series,
    config: dict[str, Any],
) -> Rect:
    payload = json.loads(
        (
            ROOT
            / config["baseline_selection_root"]
            / f"{sample}.json"
        ).read_text(encoding="utf-8")
    )
    return Rect(
        **payload["roi_predictions"]["calibrated_safe"]["rect"]
    )


def evaluate_prediction(
    *,
    sample: str,
    method: str,
    predicted: Rect,
    manual: Rect,
    analysis: ApertureAnalysis,
    manual_frame: np.ndarray,
    held_overlap: int,
) -> dict[str, Any]:
    spot_energy, peak_coverage, peak_count = spot_support_metrics(
        manual_frame, predicted, manual
    )
    return {
        "sample_id": sample,
        "method": method,
        "auto_x": predicted.x,
        "auto_y": predicted.y,
        "auto_width": predicted.width,
        "auto_height": predicted.height,
        "manual_x": manual.x,
        "manual_y": manual.y,
        "manual_width": manual.width,
        "manual_height": manual.height,
        "iou_vs_manual": rect_iou(predicted, manual),
        "manual_area_coverage": manual_area_coverage(predicted, manual),
        "spot_energy_coverage": spot_energy,
        "spot_peak_coverage": peak_coverage,
        "evaluation_spot_peak_count": peak_count,
        "circular_edge_intrusion_fraction": (
            circular_arc_intrusion_fraction(analysis, predicted)
        ),
        "left_boundary_error_px": predicted.x - manual.x,
        "right_boundary_error_px": predicted.x2 - manual.x2,
        "top_boundary_error_px": predicted.y - manual.y,
        "bottom_boundary_error_px": predicted.y2 - manual.y2,
        "fully_contains_manual_roi": bool(
            predicted.x <= manual.x
            and predicted.y <= manual.y
            and predicted.x2 >= manual.x2
            and predicted.y2 >= manual.y2
        ),
        "includes_left_reference_margin": bool(predicted.x <= manual.x),
        "includes_right_reference_boundary": bool(
            predicted.x2 >= manual.x2
        ),
        "includes_vertical_reference_envelope": bool(
            predicted.y <= manual.y and predicted.y2 >= manual.y2
        ),
        "held_video_overlap": int(held_overlap),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("method", sort=False)
        .agg(
            n_videos=("sample_id", "size"),
            median_iou=("iou_vs_manual", "median"),
            median_manual_area_coverage=(
                "manual_area_coverage", "median"
            ),
            minimum_manual_area_coverage=(
                "manual_area_coverage", "min"
            ),
            median_spot_energy_coverage=(
                "spot_energy_coverage", "median"
            ),
            minimum_spot_energy_coverage=(
                "spot_energy_coverage", "min"
            ),
            median_spot_peak_coverage=(
                "spot_peak_coverage", "median"
            ),
            circular_edge_intrusion_rate=(
                "circular_edge_intrusion_fraction",
                lambda values: float(np.mean(np.asarray(values) > 0.0)),
            ),
            median_circular_edge_intrusion=(
                "circular_edge_intrusion_fraction", "median"
            ),
            full_manual_containment_rate=(
                "fully_contains_manual_roi", "mean"
            ),
            left_reference_margin_inclusion_rate=(
                "includes_left_reference_margin", "mean"
            ),
            right_reference_boundary_inclusion_rate=(
                "includes_right_reference_boundary", "mean"
            ),
            vertical_reference_envelope_inclusion_rate=(
                "includes_vertical_reference_envelope", "mean"
            ),
            held_video_overlap=("held_video_overlap", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "median_spot_energy_coverage",
                "minimum_spot_energy_coverage",
                "median_iou",
            ],
            ascending=False,
        )
    )


def _enhanced_crop(frame: np.ndarray, rect: Rect) -> Image.Image:
    ys, xs = rect.as_slices()
    crop = Image.fromarray(frame[ys, xs]).convert("RGB")
    return ImageEnhance.Contrast(crop).enhance(1.7)


def _draw_rectangle(
    image: Image.Image,
    rect: Rect,
    *,
    color: str,
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (rect.x, rect.y, rect.x2, rect.y2),
        outline=color,
        width=width,
    )


def save_atlas(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    method: str,
    report_root: Path,
) -> None:
    selected = predictions.loc[
        predictions["method"] == method
    ].set_index("sample_id")
    baseline = predictions.loc[
        predictions["method"] == "v4_calibrated_safe"
    ].set_index("sample_id")
    pages = math.ceil(len(manifest) / 5)
    atlas_root = report_root / method
    atlas_root.mkdir(parents=True, exist_ok=True)
    for page in range(pages):
        subset = manifest.iloc[page * 5 : (page + 1) * 5]
        fig, axes = plt.subplots(
            len(subset), 3, figsize=(13.2, 3.15 * len(subset))
        )
        axes = np.atleast_2d(axes)
        for row_index, (_, record) in enumerate(subset.iterrows()):
            sample = str(record["sample_id"])
            frame = load_frame(
                ROOT / str(record["frames_dir"]),
                int(record["keyframe_index"]),
            )
            manual = Rect(
                int(record["roi_x"]),
                int(record["roi_y"]),
                int(record["roi_width"]),
                int(record["roi_height"]),
                int(record["source_width"]),
                int(record["source_height"]),
            )
            result = selected.loc[sample]
            old_result = baseline.loc[sample]
            predicted = Rect(
                int(result["auto_x"]),
                int(result["auto_y"]),
                int(result["auto_width"]),
                int(result["auto_height"]),
                manual.source_width,
                manual.source_height,
            )
            old = Rect(
                int(old_result["auto_x"]),
                int(old_result["auto_y"]),
                int(old_result["auto_width"]),
                int(old_result["auto_height"]),
                manual.source_width,
                manual.source_height,
            )
            overlay = Image.fromarray(frame).convert("RGB")
            line_width = max(3, round(min(overlay.size) / 180))
            _draw_rectangle(
                overlay, manual, color="#f2c94c", width=line_width
            )
            _draw_rectangle(
                overlay, old, color="#ef5350", width=line_width
            )
            _draw_rectangle(
                overlay, predicted, color="#00d4d8", width=line_width
            )
            axes[row_index, 0].imshow(overlay)
            axes[row_index, 0].set_title(
                f"{sample} | yellow human, red V4, cyan V7\n"
                f"spot coverage {result['spot_energy_coverage']:.2f}, "
                f"arc {100*result['circular_edge_intrusion_fraction']:.1f}%",
                fontsize=9,
            )
            axes[row_index, 1].imshow(_enhanced_crop(frame, old))
            axes[row_index, 1].set_title(
                f"V4 crop | manual coverage "
                f"{old_result['manual_area_coverage']:.2f}",
                fontsize=9,
            )
            axes[row_index, 2].imshow(_enhanced_crop(frame, predicted))
            axes[row_index, 2].set_title(
                f"V7 complete-lattice crop | manual coverage "
                f"{result['manual_area_coverage']:.2f}",
                fontsize=9,
            )
            for axis in axes[row_index]:
                axis.axis("off")
        fig.suptitle(
            f"Strict leave-one-video-out ROI comparison: {method} "
            f"(page {page + 1}/{pages})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(
            atlas_root / f"all_samples_page_{page + 1:02d}.png",
            dpi=210,
            bbox_inches="tight",
        )
        fig.savefig(
            atlas_root / f"all_samples_page_{page + 1:02d}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def save_benchmark(summary: pd.DataFrame, report_root: Path) -> None:
    order = summary["method"].tolist()
    labels = [name.replace("_", "\n") for name in order]
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.2))
    metrics = (
        ("median_spot_energy_coverage", "Median compact-spot coverage"),
        ("minimum_spot_energy_coverage", "Worst compact-spot coverage"),
        ("median_manual_area_coverage", "Median human-ROI coverage"),
        (
            "right_reference_boundary_inclusion_rate",
            "Right-boundary inclusion rate",
        ),
        ("median_iou", "Median IoU"),
    )
    colors = [
        "#ef5350" if name == "v4_calibrated_safe" else "#00a6a6"
        for name in order
    ]
    for axis, (metric, title) in zip(axes, metrics):
        axis.bar(np.arange(len(order)), summary[metric], color=colors)
        axis.set_xticks(np.arange(len(order)))
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Full-lattice ROI benchmark — strict leave-one-video-out",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(report_root / "roi_method_benchmark.png", dpi=220)
    fig.savefig(report_root / "roi_method_benchmark.pdf")
    plt.close(fig)


def save_failure_panel(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    method: str,
    report_root: Path,
) -> None:
    selected = (
        predictions.loc[predictions["method"] == method]
        .nsmallest(8, "spot_energy_coverage")
        .copy()
    )
    records = manifest.set_index("sample_id")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8))
    for axis, result in zip(axes.flat, selected.itertuples(index=False)):
        record = records.loc[str(result.sample_id)]
        frame = load_frame(
            ROOT / str(record["frames_dir"]),
            int(record["keyframe_index"]),
        )
        predicted = Rect(
            int(result.auto_x),
            int(result.auto_y),
            int(result.auto_width),
            int(result.auto_height),
            int(record["source_width"]),
            int(record["source_height"]),
        )
        axis.imshow(_enhanced_crop(frame, predicted))
        axis.set_title(
            f"{result.sample_id} | spot {result.spot_energy_coverage:.2f}\n"
            f"manual {result.manual_area_coverage:.2f}, "
            f"arc {100*result.circular_edge_intrusion_fraction:.1f}%",
            fontsize=10,
        )
        axis.axis("off")
    fig.suptitle(
        "Lowest compact-spot-coverage V7 held videos; no omitted failures",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(report_root / "lowest_spot_coverage_cases.png", dpi=220)
    fig.savefig(report_root / "lowest_spot_coverage_cases.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_manifest(config)
    output_root = (
        ROOT
        / "outputs"
        / "rheed_auto_roi_keyframe"
        / config["experiment_id"]
    )
    report_root = (
        ROOT
        / "reports"
        / "rheed_auto_roi_keyframe"
        / config["experiment_id"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    analyses: dict[str, ApertureAnalysis] = {}
    manual_frames: dict[str, np.ndarray] = {}
    target_rows = []
    manual_rects: dict[str, Rect] = {}
    for position, record in manifest.iterrows():
        sample = str(record["sample_id"])
        print(
            f"[aperture {position + 1:02d}/{len(manifest):02d}] {sample}",
            flush=True,
        )
        frames, _ = sample_frames(
            ROOT / str(record["frames_dir"]),
            maximum=int(config["roi_sample_count"]),
        )
        analysis = analyze_aperture(frames)
        analyses[sample] = analysis
        manual = Rect(
            int(record["roi_x"]),
            int(record["roi_y"]),
            int(record["roi_width"]),
            int(record["roi_height"]),
            int(record["source_width"]),
            int(record["source_height"]),
        )
        manual_rects[sample] = manual
        manual_frames[sample] = load_frame(
            ROOT / str(record["frames_dir"]),
            int(record["keyframe_index"]),
        )
        target_rows.append(
            {
                "sample_id": sample,
                "orientation": orientation_group(analysis),
                **normalized_roi_bounds(manual, analysis),
            }
        )
    targets = pd.DataFrame(target_rows)
    targets.to_csv(output_root / "normalized_roi_targets.csv", index=False)

    predictions = []
    for sample in manifest["sample_id"].astype(str):
        record = manifest.loc[manifest["sample_id"].astype(str) == sample].iloc[
            0
        ]
        manual = manual_rects[sample]
        baseline = load_baseline_rect(sample, record, config)
        predictions.append(
            evaluate_prediction(
                sample=sample,
                method="v4_calibrated_safe",
                predicted=baseline,
                manual=manual,
                analysis=analyses[sample],
                manual_frame=manual_frames[sample],
                held_overlap=0,
            )
        )
        training = targets.loc[targets["sample_id"] != sample]
        for method, settings in METHODS.items():
            bundle = fit_bundle(
                training,
                grouped=bool(settings["grouped"]),
                quantiles=tuple(settings["quantiles"]),
                config=config,
            )
            predicted = predict_full_lattice_roi(
                analyses[sample], bundle
            ).rect
            predictions.append(
                evaluate_prediction(
                    sample=sample,
                    method=method,
                    predicted=predicted,
                    manual=manual,
                    analysis=analyses[sample],
                    manual_frame=manual_frames[sample],
                    held_overlap=int(sample in set(training["sample_id"])),
                )
            )
    prediction_table = pd.DataFrame(predictions)
    prediction_table.to_csv(
        output_root / "full_lattice_roi_loo_predictions.csv", index=False
    )
    summary = summarize(prediction_table)
    summary.to_csv(output_root / "full_lattice_roi_summary.csv", index=False)

    selected_method = str(config["selected_method"])
    selected_settings = METHODS[selected_method]
    final_bundle = fit_bundle(
        targets,
        grouped=bool(selected_settings["grouped"]),
        quantiles=tuple(selected_settings["quantiles"]),
        config=config,
    )
    final_bundle.update(
        {
            "training_video_count": int(len(targets)),
            "training_sample_ids": targets["sample_id"].tolist(),
            "excluded_sample_ids": sorted(
                config["expected_excluded_sample_ids"]
            ),
            "validation_protocol": "strict_leave_one_video_out",
            "validation_method": selected_method,
            "held_video_overlap_sum": int(
                prediction_table.loc[
                    prediction_table["method"] == selected_method,
                    "held_video_overlap",
                ].sum()
            ),
            "prospective_note": (
                "Requires validation on newly acquired videos before "
                "closed-loop deployment."
            ),
        }
    )
    joblib.dump(
        final_bundle,
        output_root / "full_lattice_roi_calibration.joblib",
        compress=3,
    )
    save_benchmark(summary, report_root)
    save_atlas(
        prediction_table,
        manifest,
        method=selected_method,
        report_root=report_root,
    )
    save_failure_panel(
        prediction_table,
        manifest,
        method=selected_method,
        report_root=report_root,
    )
    selected_summary = (
        summary.loc[summary["method"] == selected_method].iloc[0].to_dict()
    )
    metadata = {
        "experiment_id": config["experiment_id"],
        "config": config,
        "retained_video_count": int(len(manifest)),
        "removelist_overlap": [],
        "selected_method": selected_method,
        "selected_summary": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in selected_summary.items()
        },
        "raw_data_policy": "read_only",
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
