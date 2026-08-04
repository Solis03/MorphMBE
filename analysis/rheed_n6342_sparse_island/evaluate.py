from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kurtosis, skew
from skimage import measure, morphology

from analysis.rheed_to_afm_full_cohort_loo.run import (
    load_config,
    load_source_tables,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _real_afm,
    _rheed_keyframe,
    _save,
    _scale_bar,
    _style,
)
from analysis.rheed_to_afm_sharp_generation.spectral import load_unit_map
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np


COUNT_FEATURES = (
    "persistent_peak_count_h025",
    "persistent_peak_count_h050",
    "persistent_peak_count_h075",
    "excursion_component_count_z100",
    "excursion_component_count_z150",
)
AREA_FEATURES = (
    "excursion_median_area_z100_px",
    "excursion_p90_area_z100_px",
    "excursion_median_area_z150_px",
)
DIRECT_FEATURES = (
    "excursion_fraction_z100",
    "excursion_fraction_z150",
    "height_skewness",
    "height_kurtosis",
    "fine_detail_rms_fraction",
)
FEATURES = (*COUNT_FEATURES, *AREA_FEATURES, *DIRECT_FEATURES)


def _component_features(mask: np.ndarray) -> tuple[int, float, float]:
    labels = measure.label(mask, connectivity=2)
    regions = [
        region
        for region in measure.regionprops(labels)
        if region.area >= 3
        and not (
            region.bbox[0] == 0
            or region.bbox[1] == 0
            or region.bbox[2] == mask.shape[0]
            or region.bbox[3] == mask.shape[1]
        )
    ]
    areas = np.asarray([region.area for region in regions], dtype=float)
    if not len(areas):
        return 0, 0.0, 0.0
    return int(len(areas)), float(np.median(areas)), float(np.quantile(areas, 0.9))


def peak_signature(array: np.ndarray) -> dict[str, float]:
    """Measure persistent peaks, bright-area topology and fine AFM texture."""

    unit = project_unit_rq_np(np.asarray(array, dtype=np.float32))
    smooth = ndimage.gaussian_filter(unit, sigma=0.8, mode="reflect")
    record: dict[str, float] = {}
    for prominence in (0.25, 0.50, 0.75):
        peaks = morphology.h_maxima(smooth, prominence)
        record[f"persistent_peak_count_h{int(prominence * 100):03d}"] = float(
            measure.label(peaks, connectivity=2).max()
        )
    for threshold in (1.0, 1.5):
        suffix = f"z{int(threshold * 100):03d}"
        mask = smooth >= threshold
        count, median_area, p90_area = _component_features(mask)
        record[f"excursion_fraction_{suffix}"] = float(np.mean(mask))
        record[f"excursion_component_count_{suffix}"] = float(count)
        record[f"excursion_median_area_{suffix}_px"] = median_area
        if threshold == 1.0:
            record[f"excursion_p90_area_{suffix}_px"] = p90_area
    record["height_skewness"] = float(skew(unit.ravel(), bias=False))
    record["height_kurtosis"] = float(
        kurtosis(unit.ravel(), fisher=False, bias=False)
    )
    fine = unit - ndimage.gaussian_filter(unit, sigma=1.5, mode="reflect")
    record["fine_detail_rms_fraction"] = float(np.std(fine))
    return record


def _generated_payload(path: Path) -> tuple[list[np.ndarray], float]:
    payload = np.load(path, allow_pickle=False)
    return (
        [
            np.asarray(array, dtype=np.float32)
            for array in payload["generated_unit_shapes"]
        ],
        float(payload["predicted_rq_nm"]),
    )


def _aggregate_signatures(signatures: list[dict[str, float]]) -> dict[str, float]:
    return {
        feature: float(np.median([row[feature] for row in signatures]))
        for feature in FEATURES
    }


def _signature_tables(
    *,
    descriptors: pd.DataFrame,
    output: Path,
    methods: list[str],
    groups: list[str],
    resolution: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    for group in groups:
        scans = descriptors.loc[descriptors["growth_run_id"] == group]
        real_signature = _aggregate_signatures(
            [
                peak_signature(load_unit_map(row, resolution))
                for _, row in scans.iterrows()
            ]
        )
        real_rows.append({"growth_run_id": group, **real_signature})
        for method in methods:
            arrays, _ = _generated_payload(
                output
                / "crossfit"
                / "generated_maps"
                / method
                / f"{group}.npz"
            )
            generated_rows.append(
                {
                    "growth_run_id": group,
                    "method": method,
                    **_aggregate_signatures(
                        [peak_signature(array) for array in arrays]
                    ),
                }
            )
    real = pd.DataFrame(real_rows)
    generated = pd.DataFrame(generated_rows)
    scales: dict[str, float] = {}
    for feature in FEATURES:
        transformed = np.log1p(real[feature]) if feature in (*COUNT_FEATURES, *AREA_FEATURES) else real[feature]
        scales[feature] = max(
            float(transformed.quantile(0.75) - transformed.quantile(0.25))
            / 1.349,
            0.05,
        )
    merged = generated.merge(real, on="growth_run_id", suffixes=("_generated", "_real"))
    error_columns = []
    for feature in FEATURES:
        generated_values = merged[f"{feature}_generated"]
        real_values = merged[f"{feature}_real"]
        if feature in (*COUNT_FEATURES, *AREA_FEATURES):
            generated_values = np.log1p(generated_values)
            real_values = np.log1p(real_values)
        column = f"absolute_error_z__{feature}"
        merged[column] = np.abs(generated_values - real_values) / scales[feature]
        error_columns.append(column)
    merged["peak_signature_mae_z"] = merged[error_columns].mean(axis=1)
    summary = (
        merged.groupby("method")
        .agg(
            growth_count=("growth_run_id", "nunique"),
            mean_peak_signature_mae_z=("peak_signature_mae_z", "mean"),
            median_peak_signature_mae_z=("peak_signature_mae_z", "median"),
        )
        .reset_index()
        .sort_values("mean_peak_signature_mae_z")
    )
    return real, merged, summary


def _plot_n6342_ablation(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    methods: list[str],
) -> None:
    group = "N6342"
    rheed = _rheed_keyframe(phase1, group)
    real = _real_afm(phase1, group)
    panels: list[tuple[str, np.ndarray]] = [("RHEED key frame", rheed)]
    values = []
    for method in methods:
        arrays, predicted_sq = _generated_payload(
            output / "crossfit" / "generated_maps" / method / f"{group}.npz"
        )
        values.append(arrays[0] * predicted_sq)
        panels.append((method.replace("_", "\n", 1), values[-1]))
    panels.append(("measured AFM", real))
    surface_values = np.concatenate([array.ravel() for array in [*values, real]])
    vmin, vmax = np.quantile(surface_values, [0.01, 0.99])
    columns = 4
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 3.1 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (title, array) in zip(axes.ravel(), panels):
        if title == "RHEED key frame":
            axis.imshow(array, cmap="gray")
        else:
            axis.imshow(array, cmap="viridis", vmin=vmin, vmax=vmax)
            _scale_bar(axis, pixels=array.shape[1])
        axis.set_title(title, fontsize=7.5)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(panels) :]:
        axis.axis("off")
    figure.suptitle(
        "N6342 renderer ablation: fewer dominant bright peaks while preserving fine AFM texture",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_dir / "Fig10_N6342_renderer_ablation")


def _plot_peak_signatures(
    *, figure_dir: Path, merged: pd.DataFrame, methods: list[str]
) -> None:
    rows = merged.loc[merged["growth_run_id"] == "N6342"].set_index("method")
    labels = ["persistent\npeaks h=0.5", "bright area\nz>1.5 Sq", "bright median\narea z>1 Sq", "kurtosis", "fine detail"]
    features = [
        "persistent_peak_count_h050",
        "excursion_fraction_z150",
        "excursion_median_area_z100_px",
        "height_kurtosis",
        "fine_detail_rms_fraction",
    ]
    figure, axes = plt.subplots(
        1,
        len(features),
        figsize=(12.0, 3.8),
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))
    short_labels = [method.split("_", maxsplit=1)[0] for method in methods]
    for axis, feature, label_text in zip(axes, features, labels):
        real_value = float(rows.iloc[0][f"{feature}_real"])
        generated = [float(rows.loc[method, f"{feature}_generated"]) for method in methods]
        axis.bar(np.arange(len(methods)), generated, color=colors)
        axis.axhline(real_value, color="#D55E00", ls="--", lw=1.5, label="measured median")
        axis.set_title(label_text, fontsize=8)
        axis.set_xticks(np.arange(len(methods)), short_labels)
        axis.tick_params(axis="x", labelrotation=55, labelsize=6)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle("N6342 multiscale peak-topology signature", fontsize=11, fontweight="bold")
    _save(figure, figure_dir / "Fig11_N6342_peak_signature")


def run(
    config_path: str | Path,
    *,
    suffix_override: str | None = None,
    evaluation_groups: list[str] | None = None,
) -> None:
    _style()
    config = load_config(config_path)
    suffix = (
        str(suffix_override)
        if suffix_override is not None
        else str(config["full_run_suffix"])
    )
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    figure_dir = report / "figures"
    tables = load_source_tables(config)
    descriptors, _ = prepare_full_cohort(tables, config)
    available_groups = sorted(
        descriptors["growth_run_id"].astype(str).unique()
    )
    groups = evaluation_groups or available_groups
    unknown_groups = sorted(set(groups) - set(available_groups))
    if unknown_groups:
        raise RuntimeError(f"unknown evaluation growths: {unknown_groups}")
    if "6081" in groups:
        raise RuntimeError("6081 entered the peak-topology audit")
    methods = [
        path.name
        for path in sorted((output / "crossfit" / "generated_maps").iterdir())
        if path.is_dir()
    ]
    display_methods = [
        str(method)
        for method in config.get("publication_ablation_methods", methods)
    ]
    missing_display_methods = sorted(set(display_methods) - set(methods))
    if missing_display_methods:
        raise RuntimeError(
            "publication_ablation_methods are absent from generated maps: "
            f"{missing_display_methods}"
        )
    real, merged, summary = _signature_tables(
        descriptors=descriptors,
        output=output,
        methods=methods,
        groups=groups,
        resolution=int(config["resolution"]),
    )
    write_csv(real, report / "measured_peak_signature_per_group.csv")
    write_csv(merged, report / "peak_signature_per_group.csv")
    write_csv(summary, report / "peak_signature_summary.csv")
    _plot_n6342_ablation(
        figure_dir=figure_dir,
        output=output,
        phase1=tables["phase1"],
        methods=display_methods,
    )
    _plot_peak_signatures(
        figure_dir=figure_dir,
        merged=merged,
        methods=display_methods,
    )
    write_json(
        {
            "cohort_count": len(groups),
            "growth_run_ids": groups,
            "excluded_6081": "6081" not in groups,
            "methods": methods,
            "publication_ablation_methods": display_methods,
            "features": list(FEATURES),
            "selection_warning": (
                "N6342 motivated method development; these are retrospective "
                "LOO morphology diagnostics, not untouched prospective evidence."
            ),
        },
        report / "peak_signature_manifest.json",
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix")
    parser.add_argument("--groups", nargs="+")
    args = parser.parse_args()
    run(
        args.config,
        suffix_override=args.suffix,
        evaluation_groups=args.groups,
    )


if __name__ == "__main__":
    main()
