"""Compare uncorrected, reselected, and keyframe-locked RHEED orientation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.extra_five_integration.build_rheed import _keyframe
from analysis.extra_five_integration.summarize import _generation_metrics
from analysis.rheed_to_afm_functional_morphology.visualization import _save
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json


M15B = "M15b_auto_r3d_angular_tta"
M10 = "M10_dense_island_spectral_pareto"
M12A = "M12a_edge_preserving_terrace"
CORRECTED = ("N6389", "N6390")


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _read_selection(root: str | Path, protocol: str) -> pd.DataFrame:
    rows = pd.read_csv(
        repo_path(root)
        / "machine_dataset_extra5"
        / "selection_comparison.csv",
        dtype={"sample_id": str},
    )
    rows["protocol"] = protocol
    if "frame_rotation_clockwise_degrees" not in rows:
        rows["frame_rotation_clockwise_degrees"] = 0
    return rows


def _plot_selection(
    selections: dict[str, pd.DataFrame],
    figure_root: Path,
) -> None:
    labels = {
        "uncorrected": "A  Original orientation",
        "rotate_reselect": "B  CW 90° + reselect",
        "rotate_locked": "C  CW 90° + locked vertex",
    }
    figure, axes = plt.subplots(
        len(CORRECTED),
        len(selections),
        figsize=(11.2, 7.0),
        constrained_layout=True,
    )
    for row_index, sample_id in enumerate(CORRECTED):
        for column_index, (protocol, table) in enumerate(selections.items()):
            row = table.loc[table["sample_id"] == sample_id].iloc[0]
            frame = _keyframe(
                repo_path(row["source_video"]),
                int(row["machine_keyframe_index"]),
                frame_rotation_clockwise_degrees=int(
                    row["frame_rotation_clockwise_degrees"]
                ),
            )
            axis = axes[row_index, column_index]
            axis.imshow(frame)
            axis.add_patch(
                Rectangle(
                    (
                        float(row["machine_roi_x"]),
                        float(row["machine_roi_y"]),
                    ),
                    float(row["machine_roi_width"]),
                    float(row["machine_roi_height"]),
                    fill=False,
                    edgecolor="#00D6B4",
                    linewidth=2.0,
                )
            )
            axis.set_title(
                f"{labels[protocol]}\n"
                f"{sample_id}, frame {int(row['machine_keyframe_index'])}",
                fontsize=9.5,
            )
            axis.set_axis_off()
    figure.suptitle(
        "Orientation correction audit: spatial rotation is separated from "
        "temporal keyframe choice",
        fontsize=12,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig5_orientation_keyframe_roi_audit")


def _embedding_isolation_audit(
    *,
    before_root: Path,
    after_root: Path,
) -> pd.DataFrame:
    registries = {
        "before": pd.read_csv(before_root / "embedding_registry.csv"),
        "after": pd.read_csv(after_root / "embedding_registry.csv"),
    }
    records: list[dict[str, Any]] = []
    for embedding_id in registries["after"]["embedding_id"].astype(str):
        payloads = {}
        for label, registry in registries.items():
            path = registry.loc[
                registry["embedding_id"].astype(str) == embedding_id,
                "path",
            ].iloc[0]
            payloads[label] = np.load(repo_path(path), allow_pickle=False)
        before_ids = list(map(str, payloads["before"]["growth_run_ids"]))
        after_ids = list(map(str, payloads["after"]["growth_run_ids"]))
        if before_ids != after_ids:
            raise RuntimeError(
                f"{embedding_id}: orientation audit growth order changed"
            )
        before = np.asarray(payloads["before"]["embeddings"], dtype=float)
        after = np.asarray(payloads["after"]["embeddings"], dtype=float)
        differences = np.linalg.norm(after - before, axis=1)
        for sample_id, difference in zip(after_ids, differences):
            records.append(
                {
                    "embedding_id": embedding_id,
                    "growth_run_id": sample_id,
                    "orientation_corrected_sample": (
                        sample_id in CORRECTED
                    ),
                    "l2_difference_from_uncorrected_v1": float(difference),
                    "expected_to_change": sample_id in CORRECTED,
                }
            )
    result = pd.DataFrame(records)
    controls = result.loc[~result["orientation_corrected_sample"]]
    if not (controls["l2_difference_from_uncorrected_v1"] == 0.0).all():
        raise RuntimeError("non-target RHEED embeddings changed")
    return result


def _afm_target_invariance_audit(
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    before = pd.read_csv(before_path, dtype={"growth_run_id": str})
    after = pd.read_csv(after_path, dtype={"growth_run_id": str})
    keys = ["growth_run_id", "afm_file_id"]
    before = before.sort_values(keys).reset_index(drop=True)
    after = after.sort_values(keys).reset_index(drop=True)
    if before[keys].to_dict("records") != after[keys].to_dict("records"):
        raise RuntimeError("AFM descriptor row identity changed")
    numeric = sorted(
        set(before.select_dtypes(include=[np.number]).columns)
        & set(after.select_dtypes(include=[np.number]).columns)
    )
    differences = np.abs(
        after[numeric].to_numpy(float) - before[numeric].to_numpy(float)
    )
    maximum = float(np.nanmax(differences)) if differences.size else 0.0
    if maximum != 0.0:
        raise RuntimeError("AFM targets changed during RHEED correction")
    return {
        "row_count": int(len(after)),
        "growth_count": int(after["growth_run_id"].nunique()),
        "numeric_column_count": int(len(numeric)),
        "maximum_absolute_numeric_difference": maximum,
        "afm_target_changed": False,
    }


def _metrics_from_predictions(
    predictions: pd.DataFrame,
    protocol: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    selected = predictions.loc[predictions["method"] == M15B]
    for target in ("Rq_nm", "FSMI_nm"):
        rows = selected.loc[selected["target"] == target]
        truth = rows["true_target"].to_numpy(float)
        predicted = rows["predicted_target"].to_numpy(float)
        linear = pearsonr(truth, predicted)
        confidence_error = spearmanr(
            rows["confidence"].to_numpy(float),
            rows["absolute_error"].to_numpy(float),
        )
        records.append(
            {
                "protocol": protocol,
                "target": target,
                "growth_count": int(len(rows)),
                "mae_nm": float(np.mean(np.abs(predicted - truth))),
                "pearson_r": float(linear.statistic),
                "pearson_p": float(linear.pvalue),
                "confidence_error_spearman": float(
                    confidence_error.statistic
                ),
                "confidence_error_p": float(confidence_error.pvalue),
            }
        )
    return records


def _plot_parameter_comparison(
    predictions: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    figure_root: Path,
) -> None:
    labels = {
        "uncorrected": "Original\norientation",
        "rotate_reselect": "CW 90°\n+ reselect",
        "rotate_locked": "CW 90°\n+ locked vertex",
    }
    colors = {
        "uncorrected": "#7A7A7A",
        "rotate_reselect": "#D55E00",
        "rotate_locked": "#009E73",
    }
    protocols = list(predictions)
    figure, axes = plt.subplots(
        1, 2, figsize=(9.3, 3.8), constrained_layout=True
    )
    x = np.arange(len(protocols))
    width = 0.34
    for offset, target, label in [
        (-width / 2, "Rq_nm", "Sq"),
        (width / 2, "FSMI_nm", "FSMI"),
    ]:
        rows = metrics.loc[metrics["target"] == target].set_index("protocol")
        axes[0].bar(
            x + offset,
            rows.loc[protocols, "mae_nm"],
            width,
            label=label,
            alpha=0.9,
        )
        axes[1].bar(
            x + offset,
            rows.loc[protocols, "pearson_r"],
            width,
            label=label,
            alpha=0.9,
        )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([labels[item] for item in protocols])
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("strict-LOO MAE (nm)")
    axes[0].set_title("Absolute prediction error")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Held-growth linear association")
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Rotation-reselection confounding and the controlled correction",
        fontsize=11.5,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig6a_orientation_protocol_metrics")

    figure, axes = plt.subplots(
        2, len(protocols), figsize=(12.0, 7.2), constrained_layout=True
    )
    confidence_points = None
    for column, protocol in enumerate(protocols):
        data = predictions[protocol]
        for row_index, (target, label) in enumerate(
            [("Rq_nm", "Sq (nm)"), ("FSMI_nm", "FSMI (nm)")]
        ):
            rows = data.loc[
                (data["method"] == M15B) & (data["target"] == target)
            ]
            axis = axes[row_index, column]
            confidence_points = axis.scatter(
                rows["true_target"],
                rows["predicted_target"],
                c=100 * rows["confidence"],
                cmap="viridis",
                vmin=0,
                vmax=100,
                edgecolor="black",
                linewidth=0.4,
                s=38,
            )
            low = min(rows["true_target"].min(), rows["predicted_target"].min())
            high = max(rows["true_target"].max(), rows["predicted_target"].max())
            axis.plot([low, high], [low, high], "--", color="black", linewidth=0.8)
            for sample in CORRECTED:
                point = rows.loc[rows["growth_run_id"] == sample]
                if len(point):
                    value = point.iloc[0]
                    axis.annotate(
                        sample,
                        (value["true_target"], value["predicted_target"]),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=7,
                    )
            metric = metrics.loc[
                (metrics["protocol"] == protocol)
                & (metrics["target"] == target)
            ].iloc[0]
            axis.set_title(
                f"{labels[protocol].replace(chr(10), ' ')}\n"
                f"MAE={metric['mae_nm']:.2f}, r={metric['pearson_r']:.2f}",
                fontsize=9.5,
            )
            axis.set_xlabel(f"measured {label}")
            axis.set_ylabel(f"LOO predicted {label}")
            axis.grid(alpha=0.16)
    if confidence_points is not None:
        colorbar = figure.colorbar(
            confidence_points,
            ax=axes,
            shrink=0.83,
            pad=0.015,
        )
        colorbar.set_label("cross-fitted confidence index (0-100)")
    figure.suptitle(
        "All 28 held-growth predictions; color encodes cross-fitted confidence",
        fontsize=11.5,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig6b_orientation_protocol_scatter")


def _measured_map(
    descriptors: Path,
    sample_id: str,
) -> tuple[np.ndarray, float, float, float, float]:
    rows = pd.read_csv(descriptors, dtype={"growth_run_id": str})
    sample_rows = rows.loc[rows["growth_run_id"] == sample_id]
    row = sample_rows.iloc[0]
    height = np.load(repo_path(row["plane_corrected_array_path"]))
    values = sample_rows["rq_nm"].to_numpy(float)
    return (
        np.asarray(height, dtype=float),
        float(row["rq_nm"]),
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    )


def _generated_map(
    root: Path,
    method: str,
    sample_id: str,
) -> tuple[np.ndarray, float]:
    payload = np.load(
        root
        / "crossfit"
        / "generated_maps"
        / method
        / f"{sample_id}.npz",
        allow_pickle=False,
    )
    predicted_sq = float(np.asarray(payload["predicted_rq_nm"]).item())
    unit = np.asarray(payload["generated_unit_shapes"])[0]
    return unit * predicted_sq, predicted_sq


def _plot_generated_comparison(
    *,
    old_root: Path,
    final_root: Path,
    descriptors: Path,
    figure_root: Path,
) -> None:
    figure, axes = plt.subplots(
        len(CORRECTED), 4, figsize=(11.8, 5.9), constrained_layout=True
    )
    for row_index, sample_id in enumerate(CORRECTED):
        (
            measured,
            measured_sq,
            sample_median_sq,
            sample_q1_sq,
            sample_q3_sq,
        ) = _measured_map(descriptors, sample_id)
        old_m12, old_sq = _generated_map(old_root, M12A, sample_id)
        final_m12, final_sq = _generated_map(final_root, M12A, sample_id)
        final_m10, final_m10_sq = _generated_map(final_root, M10, sample_id)
        maps = [measured, old_m12, final_m12, final_m10]
        titles = [
            f"{sample_id} measured AFM\n"
            f"shown scan Sq={measured_sq:.2f} nm\n"
            f"sample median [IQR]={sample_median_sq:.2f} "
            f"[{sample_q1_sq:.2f}, {sample_q3_sq:.2f}] nm",
            f"Original-orientation M12a\npredicted Sq={old_sq:.2f} nm",
            f"Corrected M12a\npredicted Sq={final_sq:.2f} nm",
            f"Corrected M10\npredicted Sq={final_m10_sq:.2f} nm",
        ]
        all_values = np.concatenate([item.ravel() for item in maps])
        limit = float(np.quantile(np.abs(all_values), 0.985))
        for column, (height, title) in enumerate(zip(maps, titles)):
            axis = axes[row_index, column]
            image = axis.imshow(
                height,
                cmap="afmhot",
                vmin=-limit,
                vmax=limit,
                extent=(0, 1, 1, 0),
            )
            axis.set_title(title, fontsize=8.8)
            axis.set_xlabel("x (µm)")
            if column == 0:
                axis.set_ylabel("y (µm)")
            else:
                axis.set_yticklabels([])
            figure.colorbar(image, ax=axis, shrink=0.78, label="height (nm)")
    figure.suptitle(
        "Generated AFM after RHEED orientation correction "
        "(strict held-growth conditioning)",
        fontsize=11.5,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig10_orientation_corrected_generated_afm")


def _write_report(
    *,
    destination: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    generation: pd.DataFrame,
    selection_rows: pd.DataFrame,
    sample_predictions: pd.DataFrame,
) -> None:
    metric = metrics.set_index(["protocol", "target"])
    image = generation.set_index(["method", "cohort"])
    selected = selection_rows.loc[
        selection_rows["sample_id"].isin(CORRECTED)
    ]
    keyframes = (
        selected.pivot(
            index="sample_id",
            columns="protocol",
            values="machine_keyframe_index",
        )
        .astype(int)
        .to_dict(orient="index")
    )
    sq_before = metric.loc[("uncorrected", "Rq_nm")]
    sq_reselect = metric.loc[("rotate_reselect", "Rq_nm")]
    sq_final = metric.loc[("rotate_locked", "Rq_nm")]
    fsmi_before = metric.loc[("uncorrected", "FSMI_nm")]
    fsmi_reselect = metric.loc[("rotate_reselect", "FSMI_nm")]
    fsmi_final = metric.loc[("rotate_locked", "FSMI_nm")]
    text = f"""# RHEED orientation-correction rerun

## Decision and data provenance

N6389 and N6390 are decoded **clockwise by 90 degrees** before every
model-visible crop, temporal clip, RHEED physics feature, embedding, live UI
frame, and generated-AFM prediction. Raw videos are read-only and were not
transcoded or overwritten. The other 26 growths have byte-identical model
embeddings relative to the prior full-28 run.

The first correction experiment rotated the videos and reran the automatic
keyframe selector. That changed both spatial orientation and temporal sample,
so it was not a controlled test. The final protocol locks the previously
target-blind V5 rotation vertex and recomputes the complete-lattice ROI in the
corrected coordinate system:

| sample | original vertex | rotate + reselect | final locked vertex |
|---|---:|---:|---:|
| N6389 | {keyframes["N6389"]["uncorrected"]} | {keyframes["N6389"]["rotate_reselect"]} | {keyframes["N6389"]["rotate_locked"]} |
| N6390 | {keyframes["N6390"]["uncorrected"]} | {keyframes["N6390"]["rotate_reselect"]} | {keyframes["N6390"]["rotate_locked"]} |

No AFM target was used to select these frames or ROIs.

## Strict held-growth scalar results

Every point below is a complete outer leave-one-growth-out prediction
(27 growths fitted, one growth held out), repeated for all 28 growths.

| protocol | Sq MAE (nm) | Sq Pearson r (p) | Sq confidence-error rho (p) | FSMI MAE (nm) | FSMI Pearson r (p) | FSMI confidence-error rho (p) |
|---|---:|---:|---:|---:|---:|---:|
| original orientation | {sq_before.mae_nm:.3f} | {sq_before.pearson_r:.3f} ({sq_before.pearson_p:.4g}) | {sq_before.confidence_error_spearman:.3f} ({sq_before.confidence_error_p:.4g}) | {fsmi_before.mae_nm:.3f} | {fsmi_before.pearson_r:.3f} ({fsmi_before.pearson_p:.4g}) | {fsmi_before.confidence_error_spearman:.3f} ({fsmi_before.confidence_error_p:.4g}) |
| CW 90 degrees + reselect | {sq_reselect.mae_nm:.3f} | {sq_reselect.pearson_r:.3f} ({sq_reselect.pearson_p:.4g}) | {sq_reselect.confidence_error_spearman:.3f} ({sq_reselect.confidence_error_p:.4g}) | {fsmi_reselect.mae_nm:.3f} | {fsmi_reselect.pearson_r:.3f} ({fsmi_reselect.pearson_p:.4g}) | {fsmi_reselect.confidence_error_spearman:.3f} ({fsmi_reselect.confidence_error_p:.4g}) |
| **CW 90 degrees + locked vertex (final)** | **{sq_final.mae_nm:.3f}** | **{sq_final.pearson_r:.3f} ({sq_final.pearson_p:.4g})** | **{sq_final.confidence_error_spearman:.3f} ({sq_final.confidence_error_p:.4g})** | **{fsmi_final.mae_nm:.3f}** | **{fsmi_final.pearson_r:.3f} ({fsmi_final.pearson_p:.4g})** | **{fsmi_final.confidence_error_spearman:.3f} ({fsmi_final.confidence_error_p:.4g})** |

The reselected run is retained as a negative ablation. Its degradation shows
that the selector, which was calibrated in the original acquisition
coordinate system, moved to a different rotation cycle when the frame was
rotated. Locking the target-blind temporal vertex isolates the requested
spatial correction and recovers most of the full-cohort association.

## Generated AFM

The final run generates four 128 x 128 AFM height-field draws for every held
growth with both M10 and M12a; no measured AFM patch or nearest-neighbor image
is available at inference.

| renderer | texture-gate pass | median sharpness ratio | median island-feature MAE (z) |
|---|---:|---:|---:|
| M10 dense-island spectral | {image.loc[(M10, "full28"), "texture_gate_pass_fraction"]:.3f} | {image.loc[(M10, "full28"), "median_sharpness_ratio"]:.3f} | {image.loc[(M10, "full28"), "median_island_feature_mae_z"]:.3f} |
| M12a edge-preserving terrace | {image.loc[(M12A, "full28"), "texture_gate_pass_fraction"]:.3f} | {image.loc[(M12A, "full28"), "median_sharpness_ratio"]:.3f} | {image.loc[(M12A, "full28"), "median_island_feature_mae_z"]:.3f} |

The figure atlas includes all 28 held growths; the dedicated orientation panel
shows measured AFM, the old-orientation M12a result, corrected M12a, and
corrected M10 for N6389/N6390. Generated maps are morphology samples
conditioned on RHEED, not pixel-registered reconstructions of a unique AFM
field of view.

## Key artifacts

- Input/ROI audit: `figures/Fig5_orientation_keyframe_roi_audit`
- All-28 parameter comparison: `figures/Fig6a_orientation_protocol_metrics`
  and `figures/Fig6b_orientation_protocol_scatter`
- Corrected generated AFM: `figures/Fig10_orientation_corrected_generated_afm`
- Complete final generator atlas: `{config["report_root"]}/{config.get("full_run_suffix", "full28_loo")}/figures`
- Per-sample scalar comparison: `orientation_corrected_sample_predictions.csv`
- Machine-readable protocol metrics: `orientation_parameter_comparison.csv`
- Non-target embedding invariance: `orientation_embedding_isolation_audit.csv`
- AFM-target invariance: `afm_target_invariance_audit.json`

## Limitations

This is strict retrospective leave-one-growth-out evaluation, not a new
prospective acquisition batch. Only two videos required orientation
correction, so the comparison cannot establish a general law about arbitrary
camera rotations. Confidence is a cross-fitted relative risk index, not a
probability of correctness. The final live replay override applies only to
archived N6389/N6390; unseen streaming samples continue to use the automatic
selector without an ID-specific temporal override.
"""
    destination.write_text(text, encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    integration_report = repo_path(config["integration_report_root"])
    figure_root = integration_report / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    v1_selection = (
        "outputs/extra_five_integration/"
        "20260729_line3_full28_v1"
    )
    v2_selection = (
        "outputs/extra_five_integration/"
        "20260729_line3_full28_orientation90_v2"
    )
    v3_selection = str(
        Path(config["phase1_manifest"]).parents[1]
    )
    selections = {
        "uncorrected": _read_selection(v1_selection, "uncorrected"),
        "rotate_reselect": _read_selection(v2_selection, "rotate_reselect"),
        "rotate_locked": _read_selection(v3_selection, "rotate_locked"),
    }
    selection_rows = pd.concat(selections.values(), ignore_index=True)
    write_csv(
        selection_rows,
        integration_report / "orientation_selection_comparison.csv",
    )
    _plot_selection(selections, figure_root)
    embedding_audit = _embedding_isolation_audit(
        before_root=repo_path(v1_selection) / "machine_dataset_full28",
        after_root=repo_path(v3_selection) / "machine_dataset_full28",
    )
    write_csv(
        embedding_audit,
        integration_report / "orientation_embedding_isolation_audit.csv",
    )
    afm_audit = _afm_target_invariance_audit(
        repo_path(
            "outputs/rheed_video_afm_story/"
            "phase3a_line3_full28_v1/afm_descriptors.csv"
        ),
        repo_path(config["afm_descriptors"]),
    )
    write_json(
        afm_audit,
        integration_report / "afm_target_invariance_audit.json",
    )

    prediction_paths = {
        "uncorrected": (
            "reports/rheed_auto_input_robustness/"
            "20260729_m15b_line3_full28_extra5_v1/"
            "m15b_strict_loo_predictions.csv"
        ),
        "rotate_reselect": (
            "reports/rheed_auto_input_robustness/"
            "20260729_m15b_line3_full28_orientation90_v2/"
            "m15b_strict_loo_predictions.csv"
        ),
        "rotate_locked": str(
            repo_path(config["external_confidence_predictions"])
        ),
    }
    predictions = {
        protocol: pd.read_csv(path, dtype={"growth_run_id": str})
        for protocol, path in prediction_paths.items()
    }
    metrics = pd.DataFrame(
        [
            record
            for protocol, rows in predictions.items()
            for record in _metrics_from_predictions(rows, protocol)
        ]
    )
    write_csv(
        metrics,
        integration_report / "orientation_parameter_comparison.csv",
    )
    _plot_parameter_comparison(predictions, metrics, figure_root)
    corrected_predictions = pd.concat(
        [
            rows.loc[
                (rows["method"] == M15B)
                & rows["growth_run_id"].isin(CORRECTED)
            ].assign(protocol=protocol)
            for protocol, rows in predictions.items()
        ],
        ignore_index=True,
    ).sort_values(["growth_run_id", "target", "protocol"])
    write_csv(
        corrected_predictions,
        integration_report / "orientation_corrected_sample_predictions.csv",
    )

    final_generation_report = (
        repo_path(config["report_root"])
        / str(config.get("full_run_suffix", "full28_loo"))
    )
    old_generation_output = repo_path(
        str(config["prior_orientation_generation_report_root"]).replace(
            "reports/", "outputs/", 1
        )
    )
    final_generation_output = (
        repo_path(config["output_root"])
        / str(config.get("full_run_suffix", "full28_loo"))
    )
    _plot_generated_comparison(
        old_root=old_generation_output,
        final_root=final_generation_output,
        descriptors=repo_path(config["afm_descriptors"]),
        figure_root=figure_root,
    )
    generation, _ = _generation_metrics(
        final_generation_report,
        set(config["extra_batch_growths"]),
    )
    write_csv(
        generation,
        integration_report / "orientation_final_generation_metrics.csv",
    )
    _write_report(
        destination=integration_report / "ORIENTATION_CORRECTION_REPORT.md",
        config=config,
        metrics=metrics,
        generation=generation,
        selection_rows=selection_rows,
        sample_predictions=corrected_predictions,
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "final_controlled_orientation_correction",
        "corrected_samples": list(CORRECTED),
        "clockwise_rotation_degrees": 90,
        "protocols": list(predictions),
        "strict_outer_loo": True,
        "afm_target_used_for_keyframe_selection": False,
        "unchanged_control_growth_count": 26,
        "maximum_control_embedding_l2_difference": float(
            embedding_audit.loc[
                ~embedding_audit["orientation_corrected_sample"],
                "l2_difference_from_uncorrected_v1",
            ].max()
        ),
        "afm_target_invariance": afm_audit,
        "best_scalar_protocol": "rotate_locked",
        "best_scalar_metrics": (
            metrics.loc[metrics["protocol"] == "rotate_locked"]
            .set_index("target")
            .to_dict(orient="index")
        ),
        "selected_live_renderer": M12A,
        "stronger_image_metric_comparator": M10,
        "deployment_bundle": (
            "outputs/rheed_realtime_ui/"
            "morphmbe_m15b_m12a_line3_full28_"
            "orientation90_keyframe_locked_live_v7.joblib"
        ),
        "default_ui_config": "configs/rheed_realtime_ui.json",
        "report": str(
            Path(config["integration_report_root"])
            / "ORIENTATION_CORRECTION_REPORT.md"
        ),
        "figures": sorted(
            str(path.relative_to(integration_report))
            for path in figure_root.glob("Fig*.png")
        ),
    }
    write_json(
        manifest,
        integration_report / "orientation_comparison_manifest.json",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "rheed_m15b_end_to_end_generation_line3_full28_"
            "orientation90_keyframe_locked_v3.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2), flush=True)


if __name__ == "__main__":
    main()
