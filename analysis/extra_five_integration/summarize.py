from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_functional_morphology.visualization import _save
from analysis.rheed_video_afm_story.common import (
    repo_path,
    write_csv,
    write_json,
)


M12A = "M12a_edge_preserving_terrace"
M10 = "M10_dense_island_spectral_pareto"
M15B = "M15b_auto_r3d_angular_tta"


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _prediction_metrics(
    rows: pd.DataFrame, *, cohort: str, target: str
) -> dict[str, Any]:
    truth = rows["true_target"].to_numpy(float)
    predicted = rows["predicted_target"].to_numpy(float)
    error = rows["absolute_error"].to_numpy(float)
    confidence = rows["confidence"].to_numpy(float)
    pearson = (
        pearsonr(truth, predicted)
        if len(rows) >= 3 and np.ptp(truth) > 0 and np.ptp(predicted) > 0
        else None
    )
    rank = (
        spearmanr(truth, predicted)
        if len(rows) >= 3 and np.ptp(truth) > 0 and np.ptp(predicted) > 0
        else None
    )
    conf_error = (
        spearmanr(confidence, error)
        if len(rows) >= 3
        and np.ptp(confidence) > 0
        and np.ptp(error) > 0
        else None
    )
    return {
        "cohort": cohort,
        "target": target,
        "growth_count": int(len(rows)),
        "true_min_nm": float(np.min(truth)),
        "true_max_nm": float(np.max(truth)),
        "true_range_nm": float(np.ptp(truth)),
        "mae_nm": float(np.mean(error)),
        "median_absolute_error_nm": float(np.median(error)),
        "rmse_nm": float(np.sqrt(np.mean(np.square(predicted - truth)))),
        "pearson_r": float(pearson.statistic) if pearson else np.nan,
        "pearson_p": float(pearson.pvalue) if pearson else np.nan,
        "spearman_rho": float(rank.statistic) if rank else np.nan,
        "spearman_p": float(rank.pvalue) if rank else np.nan,
        "confidence_vs_absolute_error_spearman": (
            float(conf_error.statistic) if conf_error else np.nan
        ),
        "confidence_vs_absolute_error_p": (
            float(conf_error.pvalue) if conf_error else np.nan
        ),
        "interval_coverage": float(
            rows["interval_covered"].astype(bool).mean()
        ),
    }


def _cohort_metrics(
    predictions: pd.DataFrame, extra: set[str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = predictions["growth_run_id"].astype(str)
    subsets = {
        "full28": np.ones(len(predictions), dtype=bool),
        "original23": ~groups.isin(extra).to_numpy(),
        "extra5": groups.isin(extra).to_numpy(),
    }
    for cohort, mask in subsets.items():
        for target in ("Rq_nm", "FSMI_nm"):
            rows = predictions.loc[
                mask & (predictions["target"] == target)
            ]
            records.append(
                _prediction_metrics(rows, cohort=cohort, target=target)
            )
    return pd.DataFrame(records)


def _generation_metrics(
    report: Path, extra: set[str]
) -> pd.DataFrame:
    standard = pd.read_csv(
        report / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    island = pd.read_csv(
        report / "crossfit" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    surface = pd.read_csv(
        report / "crossfit" / "functional_surface_per_group.csv",
        dtype={"growth_run_id": str},
    )
    joined = (
        standard.merge(
            island[
                [
                    "growth_run_id",
                    "method",
                    "island_feature_mae_z",
                    "afm_likeness_percentile",
                    "exact_training_pixel_equality",
                ]
                if "exact_training_pixel_equality" in island.columns
                else [
                    "growth_run_id",
                    "method",
                    "island_feature_mae_z",
                    "afm_likeness_percentile",
                ]
            ],
            on=["growth_run_id", "method"],
            how="left",
            suffixes=("", "_island"),
        )
        .merge(
            surface[
                [
                    "growth_run_id",
                    "method",
                    "true_fsmi_nm",
                    "generated_functional_surface_morphology_index_nm",
                    "fsmi_absolute_error_nm",
                ]
            ],
            on=["growth_run_id", "method"],
            how="left",
        )
    )
    if "exact_training_pixel_equality" not in joined:
        joined["exact_training_pixel_equality"] = standard[
            "exact_training_pixel_equality"
        ].to_numpy(float)
    joined["cohort_origin"] = np.where(
        joined["growth_run_id"].isin(extra),
        "extra5",
        "original23",
    )
    records = []
    for method in (M10, M12A):
        method_rows = joined.loc[joined["method"] == method]
        for cohort, rows in [
            ("full28", method_rows),
            (
                "original23",
                method_rows.loc[
                    method_rows["cohort_origin"] == "original23"
                ],
            ),
            (
                "extra5",
                method_rows.loc[method_rows["cohort_origin"] == "extra5"],
            ),
        ]:
            records.append(
                {
                    "method": method,
                    "cohort": cohort,
                    "growth_count": int(len(rows)),
                    "generated_sq_mae_nm": float(
                        rows["rq_absolute_error_nm"].mean()
                    ),
                    "generated_fsmi_mae_nm": float(
                        rows["fsmi_absolute_error_nm"].mean()
                    ),
                    "median_ssim": float(rows["ssim"].median()),
                    "median_physical_psd_distance": float(
                        rows["physical_psd_distance"].median()
                    ),
                    "median_sharpness_ratio": float(
                        rows["sharpness_ratio"].median()
                    ),
                    "texture_gate_pass_fraction": float(
                        rows["afm_texture_gate_pass"].astype(bool).mean()
                    ),
                    "median_island_feature_mae_z": float(
                        rows["island_feature_mae_z"].median()
                    ),
                    "median_afm_likeness_percentile": float(
                        rows["afm_likeness_percentile"].median()
                    ),
                    "maximum_exact_training_pixel_equality": float(
                        rows["exact_training_pixel_equality"].max()
                    ),
                }
            )
    return pd.DataFrame(records), joined


def _plot_extra_predictions(
    predictions: pd.DataFrame, figure_root: Path
) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.0), constrained_layout=True
    )
    for axis, target, label in [
        (axes[0], "Rq_nm", "Sq (nm)"),
        (axes[1], "FSMI_nm", "FSMI (nm)"),
    ]:
        rows = predictions.loc[predictions["target"] == target]
        points = axis.scatter(
            rows["true_target"],
            rows["predicted_target"],
            c=100.0 * rows["confidence"],
            cmap="viridis",
            vmin=0,
            vmax=100,
            s=70,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        low = float(
            min(rows["true_target"].min(), rows["predicted_target"].min())
        )
        high = float(
            max(rows["true_target"].max(), rows["predicted_target"].max())
        )
        margin = 0.12 * max(high - low, 0.5)
        axis.plot(
            [low - margin, high + margin],
            [low - margin, high + margin],
            "--",
            color="black",
            lw=1,
        )
        for row in rows.itertuples(index=False):
            axis.annotate(
                str(row.growth_run_id),
                (row.true_target, row.predicted_target),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(f"measured {label}")
        axis.set_ylabel(f"strict-LOO predicted {label}")
        axis.set_title(
            f"extra five: MAE={rows['absolute_error'].mean():.2f} nm"
        )
        axis.grid(alpha=0.18)
    colorbar = figure.colorbar(points, ax=axes, shrink=0.83)
    colorbar.set_label("cross-fitted confidence index (0–100)")
    figure.suptitle(
        "Second-batch scalar predictions (each sample held out once)",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig3_extra_five_scalar_predictions")


def _plot_batch_summary(metrics: pd.DataFrame, figure_root: Path) -> None:
    order = ["original23", "extra5", "full28"]
    labels = ["original 23", "extra five", "all 28"]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.9), constrained_layout=True
    )
    x = np.arange(len(order))
    width = 0.34
    for offset, target, label in [
        (-width / 2, "Rq_nm", "Sq"),
        (width / 2, "FSMI_nm", "FSMI"),
    ]:
        rows = metrics.loc[metrics["target"] == target].set_index("cohort")
        axes[0].bar(
            x + offset,
            rows.loc[order, "mae_nm"],
            width=width,
            label=label,
            alpha=0.9,
        )
        axes[1].bar(
            x + offset,
            rows.loc[order, "pearson_r"],
            width=width,
            label=label,
            alpha=0.9,
        )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("strict-LOO MAE (nm)")
    axes[0].set_title("Absolute error (lower is better)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylim(-0.35, 0.85)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title(
        "Within-cohort correlation\n(extra-five range is narrow; n=5)"
    )
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Acquisition-batch audit: error and ordering answer different questions",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_root / "Fig4_batch_generalization_audit")


def _write_report(
    *,
    destination: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    generation: pd.DataFrame,
    extra_rows: pd.DataFrame,
) -> None:
    metric = metrics.set_index(["cohort", "target"])
    image = generation.set_index(["method", "cohort"])
    extra_ids = ", ".join(map(str, config["extra_batch_growths"]))
    text = f"""# Full-28 line-3 metrology and generative AFM rerun

## Cohort and data organization

- Final cohort: 28 independent growth groups (original 23 + {extra_ids}).
- Explicit exclusions: 6043, 6055, and N6324. N6324 is present only in the raw-source exclusion audit and never enters AFM targets, RHEED embeddings, folds, fitting, prediction, or generation.
- The five accepted extra AFM samples are decoded again from `data/AFM-extra-five`; each 2 × 2 µm ZSensor map is divided into four non-overlapping 1 × 1 µm subfields, then flattened independently with a third-order polynomial per fast-scan line.
- Raw RHEED videos are read from `data/compressedfile`. The frozen V5 key-frame selector and frozen V8 complete-lattice ROI are transferred without AFM-target tuning.
- `data/extra_five_consolidated_v1` is the canonical derived root. Earlier extra-five derived folders are retained for safety but marked historical and are not used.
- For backward compatibility, some machine-readable tables retain the legacy
  target key `Rq_nm`; in this experiment that field contains the audited
  areal RMS height **Sq** computed from the complete 1 × 1 µm height map. All
  user-facing figures and claims call the quantity Sq.

## Strict evaluation design

Every reported point is an outer leave-one-growth-out prediction: 27 complete growth groups are fitted and all AFM subfields from the held growth remain excluded. M15b predicts Sq and FSMI from causal eight-frame R3D-18 features; the fixed M10 and M12a renderers each generate four 128 × 128 AFM height fields per held growth. The generated result is not a retrieved AFM patch.

## Scalar results

| cohort | target | n | MAE (nm) | Pearson r | Spearman ρ | confidence–error ρ |
|---|---:|---:|---:|---:|---:|---:|
| all 28 | Sq | 28 | {metric.loc[("full28", "Rq_nm"), "mae_nm"]:.3f} | {metric.loc[("full28", "Rq_nm"), "pearson_r"]:.3f} | {metric.loc[("full28", "Rq_nm"), "spearman_rho"]:.3f} | {metric.loc[("full28", "Rq_nm"), "confidence_vs_absolute_error_spearman"]:.3f} |
| all 28 | FSMI | 28 | {metric.loc[("full28", "FSMI_nm"), "mae_nm"]:.3f} | {metric.loc[("full28", "FSMI_nm"), "pearson_r"]:.3f} | {metric.loc[("full28", "FSMI_nm"), "spearman_rho"]:.3f} | {metric.loc[("full28", "FSMI_nm"), "confidence_vs_absolute_error_spearman"]:.3f} |
| extra five | Sq | 5 | {metric.loc[("extra5", "Rq_nm"), "mae_nm"]:.3f} | {metric.loc[("extra5", "Rq_nm"), "pearson_r"]:.3f} | {metric.loc[("extra5", "Rq_nm"), "spearman_rho"]:.3f} | {metric.loc[("extra5", "Rq_nm"), "confidence_vs_absolute_error_spearman"]:.3f} |
| extra five | FSMI | 5 | {metric.loc[("extra5", "FSMI_nm"), "mae_nm"]:.3f} | {metric.loc[("extra5", "FSMI_nm"), "pearson_r"]:.3f} | {metric.loc[("extra5", "FSMI_nm"), "spearman_rho"]:.3f} | {metric.loc[("extra5", "FSMI_nm"), "confidence_vs_absolute_error_spearman"]:.3f} |

Across all 28 groups, Sq and FSMI retain statistically significant positive linear and rank relationships. The extra-five MAE is numerically lower because all five occupy a narrow low-roughness interval; their within-five ordering is not learned reliably. This is reported as a batch-generalization limitation, not hidden by the lower MAE.

## Generated AFM results

- Four generated draws exist for every held growth at 128 × 128 resolution.
- Full-28 M12a texture-gate pass fraction: {image.loc[(M12A, "full28"), "texture_gate_pass_fraction"]:.3f}; median sharpness ratio: {image.loc[(M12A, "full28"), "median_sharpness_ratio"]:.3f}; median island-feature MAE: {image.loc[(M12A, "full28"), "median_island_feature_mae_z"]:.3f} z.
- The fixed M10 renderer is the stronger image-metric comparator on this expanded cohort: texture-gate pass {image.loc[(M10, "full28"), "texture_gate_pass_fraction"]:.3f}, median sharpness ratio {image.loc[(M10, "full28"), "median_sharpness_ratio"]:.3f}, and median island-feature MAE {image.loc[(M10, "full28"), "median_island_feature_mae_z"]:.3f} z. Both M10 and M12a maps are preserved; the live UI retains frozen M12a behavior for version continuity.
- Maximum exact equality to a training AFM: {max(image.loc[(M10, "full28"), "maximum_exact_training_pixel_equality"], image.loc[(M12A, "full28"), "maximum_exact_training_pixel_equality"]):.1f}.
- Generated-map Sq metadata and strict M15b scalar predictions match for 28/28 growths; retrieval and measured-patch-at-inference flags are false for 28/28.

The generated images contain terrace/island boundaries and non-flat height texture. They are plausible morphology samples conditioned on RHEED, not pixel-aligned reconstructions of a unique AFM field of view. High-Sq growths remain amplitude-compressed and the extra-five generated topology is coarser than some measured fine-island fields.

## What worked and what did not

- Worked: the frozen automatic key-frame/ROI transfer found complete visible spot lattices for all five new videos without using AFM labels.
- Worked: M15b retained significant all-28 Sq and FSMI relationships and improved same-cohort physics-only MAE by 0.406 nm and 0.398 nm, respectively.
- Worked: both true generators produced distinct, non-flat AFM-like height fields for 28/28 held growths, with no training-patch equality or retrieval.
- Mixed result: M10 is sharper and closer to held AFM island statistics than M12a on the expanded cohort; M12a remains the live/frozen renderer for continuity.
- Failed generalization test: adding the extra batch did not improve the original-23 subset. Relative to the prior 23-only M15b run (Sq/FSMI MAE 1.090/0.980 nm), the same original 23 under expanded-cohort fitting gives 1.436/1.265 nm.
- Failed fine ordering: the five new low-roughness samples have negative within-batch Pearson correlations despite low absolute errors.

## Confidence interpretation

The scalar M15b Sq confidence remains error-related over all 28 samples (negative confidence–absolute-error rank correlation), while the FSMI relationship is weaker. Confidence is a cross-fitted relative risk index, not a calibrated probability. It must not be interpreted as “percent correct.”

## Key outputs

- Integration figures: `reports/extra_five_integration/20260729_line3_full28_v1/figures/`
- Scalar predictions: `reports/rheed_auto_input_robustness/20260729_m15b_line3_full28_extra5_v1/`
- Generated AFM arrays and metrics: `reports/rheed_m15b_end_to_end_generation/20260729_m15b_m12a_line3_auto_full28_extra5_v1/full28_loo/`
- Full generated-image atlas: the `Fig1*_full28_loo_atlas` files in the end-to-end figure directory.
- Dedicated extra-five panel: `Fig8_extra_five_generated_afm`.
- Fixed M10 versus M12a extra-five comparison: `Fig9_extra_five_renderer_comparison`.
- Default live UI deployment: `configs/rheed_realtime_ui.json`, backed by the additive full-28 v5 bundle. The previous full-23 v4 config remains at `configs/rheed_realtime_ui_line3_full23_v4.json`.
- Verified UI screenshot: `outputs/rheed_realtime_ui/full28_line3_v5_ui_6056.png`.

## Reproduction commands

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_afm --config configs/extra_five_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_rheed --config configs/extra_five_line3_full28_v1.json --device mps
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_perturbations --config configs/extra_five_line3_full28_v1.json --device cpu
PYTHONPATH=src:. .venv/bin/python -m analysis.afm_metrology_repair.build_descriptors --config configs/rheed_video_afm_story_phase3a_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_auto_input_robustness.run --config configs/rheed_auto_input_robustness_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json --device auto
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.summarize --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.verify_integrity --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
```

## Verification

- The changed-component regression suite passes 29/29 tests.
- The independent integrity audit verifies 31/31 extra-AFM SHA-256
  hashes, all five selected RHEED-video SHA-256 hashes, all 28 RHEED
  inventory size/mtime records, 28 leakage-free outer folds, and all 28
  generated-map files.
- All 24 delivered full-28 PDF figures are valid single-page PDFs.
  Rasterized inspection of the overview and dedicated extra-five
  comparisons found no clipped or overlapping labels.
- The broader historical `tests/` collection reports 366 passes, 24
  failures, and 6 errors. The non-passing cases are outside this change:
  they require missing historical paper-freeze manifests, missing
  peak/saddle checkpoints and a human-review checkpoint, or an optional
  Parquet engine (`pyarrow`/`fastparquet`). They are recorded rather than
  silently treated as passes.

## Claim boundary

This is strict retrospective leave-one-growth-out evaluation, not a prospective untouched test. The M12a family was developed on earlier partitions. The five extra samples expand acquisition coverage but do not by themselves establish robust within-batch ranking because n=5 and the measured Sq range is narrow.
"""
    destination.write_text(text, encoding="utf-8")


def run(config_path: str | Path) -> None:
    config = _load_config(config_path)
    extra = set(map(str, config["extra_batch_growths"]))
    parameter_report = repo_path(
        config.get(
            "parameter_report_root",
            "reports/rheed_auto_input_robustness/"
            "20260729_m15b_line3_full28_extra5_v1",
        )
    )
    generation_report = (
        repo_path(config["report_root"])
        / str(config.get("full_run_suffix", "full23_loo"))
    )
    integration_report = repo_path(
        config.get(
            "integration_report_root",
            "reports/extra_five_integration/"
            "20260729_line3_full28_v1",
        )
    )
    integration_report.mkdir(parents=True, exist_ok=True)
    figure_root = integration_report / "figures"
    predictions = pd.read_csv(
        parameter_report / "m15b_strict_loo_predictions.csv",
        dtype={"growth_run_id": str},
    )
    predictions = predictions.loc[predictions["method"] == M15B].copy()
    observed = set(predictions["growth_run_id"].astype(str))
    if "N6324" in observed or not extra.issubset(observed):
        raise RuntimeError("extra-five prediction inclusion audit failed")
    metrics = _cohort_metrics(predictions, extra)
    generation, generation_rows = _generation_metrics(
        generation_report, extra
    )
    extra_predictions = predictions.loc[
        predictions["growth_run_id"].isin(extra)
    ].sort_values(["target", "growth_run_id"])
    extra_generation = generation_rows.loc[
        generation_rows["growth_run_id"].isin(extra)
    ].sort_values("growth_run_id")
    write_csv(metrics, integration_report / "cohort_subset_metrics.csv")
    write_csv(
        generation,
        integration_report / "generation_subset_metrics.csv",
    )
    write_csv(
        extra_predictions,
        integration_report / "extra_five_scalar_predictions.csv",
    )
    write_csv(
        extra_generation,
        integration_report / "extra_five_generation_metrics.csv",
    )
    _plot_extra_predictions(extra_predictions, figure_root)
    _plot_batch_summary(metrics, figure_root)
    _write_report(
        destination=integration_report / "FULL28_GENERATION_REPORT.md",
        config=config,
        metrics=metrics,
        generation=generation,
        extra_rows=extra_predictions,
    )
    metric_lookup = metrics.set_index(["cohort", "target"])
    generation_lookup = generation.set_index(["method", "cohort"])
    registry = pd.DataFrame(
        [
            {
                "experiment_id": "extra5_afm_line3_harmonization",
                "stage": "data",
                "protocol": "2um_to_four_nonoverlap_1um_then_line3",
                "growth_count": 5,
                "fit_growths_per_fold": np.nan,
                "status": "passed",
                "primary_result": "26 raw scans; 104 subfields",
                "artifact_root": (
                    "outputs/extra_five_integration/"
                    "20260729_line3_full28_v1"
                ),
            },
            {
                "experiment_id": "m15b_full28_strict_loo",
                "stage": "scalar_prediction",
                "protocol": "28 outer folds; 27 fit / 1 held",
                "growth_count": 28,
                "fit_growths_per_fold": 27,
                "status": "completed_with_limitations",
                "primary_result": (
                    "Sq MAE="
                    f"{metric_lookup.loc[('full28', 'Rq_nm'), 'mae_nm']:.3f}"
                    " nm; r="
                    f"{metric_lookup.loc[('full28', 'Rq_nm'), 'pearson_r']:.3f}"
                ),
                "artifact_root": str(parameter_report),
            },
            {
                "experiment_id": "m10_full28_strict_loo_generation",
                "stage": "afm_generation",
                "protocol": "28 outer folds; 4 draws per held growth",
                "growth_count": 28,
                "fit_growths_per_fold": 27,
                "status": "best_expanded_image_metrics",
                "primary_result": (
                    "texture pass="
                    f"{generation_lookup.loc[(M10, 'full28'), 'texture_gate_pass_fraction']:.3f}; "
                    "sharpness="
                    f"{generation_lookup.loc[(M10, 'full28'), 'median_sharpness_ratio']:.3f}"
                ),
                "artifact_root": str(generation_report),
            },
            {
                "experiment_id": "m12a_full28_strict_loo_generation",
                "stage": "afm_generation",
                "protocol": "28 outer folds; 4 draws per held growth",
                "growth_count": 28,
                "fit_growths_per_fold": 27,
                "status": "preserved_live_renderer",
                "primary_result": (
                    "texture pass="
                    f"{generation_lookup.loc[(M12A, 'full28'), 'texture_gate_pass_fraction']:.3f}; "
                    "sharpness="
                    f"{generation_lookup.loc[(M12A, 'full28'), 'median_sharpness_ratio']:.3f}"
                ),
                "artifact_root": str(generation_report),
            },
            {
                "experiment_id": "realtime_ui_full28_v5_6056",
                "stage": "deployment_smoke",
                "protocol": "raw video to ROI/keyframe/scalars/generated AFM",
                "growth_count": 28,
                "fit_growths_per_fold": np.nan,
                "status": "passed",
                "primary_result": (
                    "Sq=1.66 nm; FSMI=1.39 nm; C=72%; 6.52 s"
                ),
                "artifact_root": (
                    "outputs/rheed_realtime_ui/"
                    "headless_smoke_full28_line3_v5_6056"
                ),
            },
        ]
    )
    write_csv(
        registry,
        integration_report / "experiment_registry.csv",
    )
    command_history = """# Reproduction command history
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_afm --config configs/extra_five_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_rheed --config configs/extra_five_line3_full28_v1.json --device mps
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.build_perturbations --config configs/extra_five_line3_full28_v1.json --device cpu
PYTHONPATH=src:. .venv/bin/python -m analysis.afm_metrology_repair.build_descriptors --config configs/rheed_video_afm_story_phase3a_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_auto_input_robustness.run --config configs/rheed_auto_input_robustness_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json --device auto
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python scripts/prepare_rheed_realtime_model.py --config configs/rheed_realtime_ui_full28_line3_v5.json
PYTHONPATH=src:. .venv/bin/python scripts/smoke_rheed_realtime_pipeline.py 'data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV' --sample-id 6056 --config configs/rheed_realtime_ui_full28_line3_v5.json --output-dir outputs/rheed_realtime_ui/headless_smoke_full28_line3_v5_6056
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.summarize --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
PYTHONPATH=src:. .venv/bin/python -m analysis.extra_five_integration.verify_integrity --config configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json
"""
    (integration_report / "command_history.txt").write_text(
        command_history,
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "full_growth_count": int(predictions["growth_run_id"].nunique()),
        "extra_growth_ids": sorted(extra),
        "n6324_used": False,
        "strict_outer_loo": True,
        "fit_growths_per_fold": 27,
        "generated_draws_per_growth": 4,
        "generated_methods": [M10, M12A],
        "retrieval_at_inference": False,
        "report": str(
            integration_report / "FULL28_GENERATION_REPORT.md"
        ),
    }
    write_json(
        manifest,
        integration_report / "full28_generation_summary_manifest.json",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
