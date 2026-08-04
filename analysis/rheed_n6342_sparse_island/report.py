from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)


METHOD_NOTES = {
    "M10_dense_island_spectral_pareto": "preserved dense-island/spectral baseline",
    "M16b_baseline_dense_microisland_terrace": "preserved fixed dense-maxima smooth renderer",
    "M17a_fixed_sparse_peak_terrace": "fixed 12-peak sparse ablation",
    "M17b_topology_sparse_peak_terrace": "selected RHEED-conditioned q82 peak-count renderer",
    "M17c_topology_sparse_finetexture_terrace": "stronger fine-residual ablation",
    "M17d_multiscale_texture_no_peak_terrace": "no explicit peak layer",
    "M17e_broad_sparse_peak_terrace": "few broad-peak ablation",
    "M17f_broad_sparse_finetexture_terrace": "broad peaks plus stronger fine residual",
    "M17g_moderate_broad_topology_terrace": "moderately broadened topology peaks",
    "M17h_hierarchical_sparse_island_terrace": "two-level shoulder/peak hierarchy",
    "M17i_soft_hierarchical_island_terrace": "lower-weight two-level hierarchy",
}


def _method_row(frame: pd.DataFrame, method: str) -> pd.Series:
    rows = frame.loc[frame["method"] == method]
    if len(rows) != 1:
        raise RuntimeError(f"expected one row for {method}, found {len(rows)}")
    return rows.iloc[0]


def _map_integrity(
    *, output: Path, report: Path, method: str, expected_groups: list[str]
) -> pd.DataFrame:
    expected_sq = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")["predicted_target"]
    rows: list[dict[str, Any]] = []
    for group in expected_groups:
        path = output / "crossfit" / "generated_maps" / method / f"{group}.npz"
        payload = np.load(path, allow_pickle=False)
        stored_group = str(payload["growth_run_id"])
        stored_method = str(payload["method"])
        predicted_sq = float(payload["predicted_rq_nm"])
        rows.append(
            {
                "growth_run_id": group,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "stored_growth_matches": stored_group == group,
                "stored_method_matches": stored_method == method,
                "predicted_sq_matches_strict_loo": bool(
                    np.isclose(predicted_sq, float(expected_sq.loc[group]))
                ),
                "retrieval_at_inference": bool(
                    payload["retrieval_at_inference"]
                ),
                "measured_afm_patch_used_at_inference": bool(
                    payload["measured_afm_patch_used_at_inference"]
                ),
                "ensemble_draw_count": int(
                    payload["generated_unit_shapes"].shape[0]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(expected_groups):
        raise RuntimeError("selected-map count mismatch")
    if "6081" in set(frame["growth_run_id"]):
        raise RuntimeError("6081 entered selected generated maps")
    boolean_checks = (
        frame["stored_growth_matches"]
        & frame["stored_method_matches"]
        & frame["predicted_sq_matches_strict_loo"]
        & ~frame["retrieval_at_inference"]
        & ~frame["measured_afm_patch_used_at_inference"]
    )
    if not bool(boolean_checks.all()):
        raise RuntimeError("selected generated-map integrity check failed")
    return frame


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    suffix = str(config["full_run_suffix"])
    output = repo_path(config["output_root"]) / suffix
    source_report = repo_path(config["report_root"]) / suffix
    report = repo_path("reports/rheed_n6342_sparse_island")
    report.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(
        (source_report / "best_model_manifest.json").read_text(encoding="utf-8")
    )
    selected = str(source_manifest["selected_method"])
    baseline = "M16b_baseline_dense_microisland_terrace"
    groups = [str(group) for group in source_manifest["growth_run_ids"]]
    if len(groups) != 27 or "6081" in groups:
        raise RuntimeError("final cohort is not the expected 27-growth cohort")

    methods = pd.read_csv(source_report / "method_summary.csv")
    peaks = pd.read_csv(source_report / "peak_signature_summary.csv")
    peak_groups = pd.read_csv(
        source_report / "peak_signature_per_group.csv",
        dtype={"growth_run_id": str},
    )
    standard = pd.read_csv(
        source_report / "crossfit/standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    islands = pd.read_csv(
        source_report / "crossfit/island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    target_summary = pd.read_csv(source_report / "target_prediction_summary.csv")
    confidence = pd.read_csv(
        source_report / "confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    )

    registry_rows: list[dict[str, Any]] = []
    for method in sorted(methods["method"]):
        summary = _method_row(methods, method)
        peak = _method_row(peaks, method)
        npeak = peak_groups.loc[
            (peak_groups["method"] == method)
            & (peak_groups["growth_run_id"] == "N6342")
        ].iloc[0]
        nstandard = standard.loc[
            (standard["method"] == method)
            & (standard["growth_run_id"] == "N6342")
        ].iloc[0]
        nisland = islands.loc[
            (islands["method"] == method)
            & (islands["growth_run_id"] == "N6342")
        ].iloc[0]
        registry_rows.append(
            {
                "method": method,
                "role": "selected" if method == selected else "ablation/baseline",
                "hypothesis": METHOD_NOTES.get(method, "documented candidate"),
                "full27_mean_psd_log_distance": summary["mean_normalized_psd_log_distance"],
                "full27_mean_island_mae_z": summary["mean_island_feature_mae_z"],
                "full27_mean_peak_signature_mae_z": peak["mean_peak_signature_mae_z"],
                "full27_texture_gate_fraction": summary["mean_afm_texture_gate_pass"],
                "n6342_psd_log_distance": nstandard["normalized_psd_log_distance"],
                "n6342_island_mae_z": nisland["island_feature_mae_z"],
                "n6342_peak_signature_mae_z": npeak["peak_signature_mae_z"],
                "n6342_persistent_peaks_h050": npeak["persistent_peak_count_h050_generated"],
                "n6342_bright_fraction_z150": npeak["excursion_fraction_z150_generated"],
                "n6342_bright_median_area_z100_px": npeak["excursion_median_area_z100_px_generated"],
                "n6342_kurtosis": npeak["height_kurtosis_generated"],
                "n6342_fine_detail_fraction": npeak["fine_detail_rms_fraction_generated"],
            }
        )
    registry = pd.DataFrame(registry_rows)
    write_csv(registry, report / "experiment_registry.csv")

    baseline_method = _method_row(methods, baseline)
    selected_method = _method_row(methods, selected)
    baseline_peak = _method_row(peaks, baseline)
    selected_peak = _method_row(peaks, selected)
    nbaseline_peak = peak_groups.loc[
        (peak_groups["method"] == baseline)
        & (peak_groups["growth_run_id"] == "N6342")
    ].iloc[0]
    nselected_peak = peak_groups.loc[
        (peak_groups["method"] == selected)
        & (peak_groups["growth_run_id"] == "N6342")
    ].iloc[0]
    nbaseline_standard = standard.loc[
        (standard["method"] == baseline)
        & (standard["growth_run_id"] == "N6342")
    ].iloc[0]
    nselected_standard = standard.loc[
        (standard["method"] == selected)
        & (standard["growth_run_id"] == "N6342")
    ].iloc[0]
    nbaseline_island = islands.loc[
        (islands["method"] == baseline)
        & (islands["growth_run_id"] == "N6342")
    ].iloc[0]
    nselected_island = islands.loc[
        (islands["method"] == selected)
        & (islands["growth_run_id"] == "N6342")
    ].iloc[0]

    comparisons = pd.DataFrame(
        [
            ("N6342", "normalized PSD log distance", nbaseline_standard["normalized_psd_log_distance"], nselected_standard["normalized_psd_log_distance"], "lower"),
            ("N6342", "composite morphology score", nbaseline_standard["composite_score"], nselected_standard["composite_score"], "lower"),
            ("N6342", "island feature MAE (z)", nbaseline_island["island_feature_mae_z"], nselected_island["island_feature_mae_z"], "lower"),
            ("N6342", "all-feature peak-signature MAE (z)", nbaseline_peak["peak_signature_mae_z"], nselected_peak["peak_signature_mae_z"], "lower; includes low-amplitude peak count"),
            ("N6342", "bright area fraction above 1.5 Sq", nbaseline_peak["excursion_fraction_z150_generated"], nselected_peak["excursion_fraction_z150_generated"], "measured=0.054016"),
            ("N6342", "bright component median area (px)", nbaseline_peak["excursion_median_area_z100_px_generated"], nselected_peak["excursion_median_area_z100_px_generated"], "measured=26"),
            ("N6342", "height kurtosis", nbaseline_peak["height_kurtosis_generated"], nselected_peak["height_kurtosis_generated"], "measured=3.326682"),
            ("N6342", "fine-detail RMS fraction", nbaseline_peak["fine_detail_rms_fraction_generated"], nselected_peak["fine_detail_rms_fraction_generated"], "measured=0.321725"),
            ("full27", "mean normalized PSD log distance", baseline_method["mean_normalized_psd_log_distance"], selected_method["mean_normalized_psd_log_distance"], "lower"),
            ("full27", "mean island feature MAE (z)", baseline_method["mean_island_feature_mae_z"], selected_method["mean_island_feature_mae_z"], "lower"),
            ("full27", "mean peak-signature MAE (z)", baseline_peak["mean_peak_signature_mae_z"], selected_peak["mean_peak_signature_mae_z"], "lower"),
            ("full27", "mean sharpness ratio", baseline_method["mean_sharpness_ratio"], selected_method["mean_sharpness_ratio"], "closer to 1"),
            ("full27", "texture-gate pass fraction", baseline_method["mean_afm_texture_gate_pass"], selected_method["mean_afm_texture_gate_pass"], "higher"),
        ],
        columns=["scope", "metric", "M16b_baseline", "M17b_selected", "preferred_or_reference"],
    )
    comparisons["selected_minus_baseline"] = (
        comparisons["M17b_selected"] - comparisons["M16b_baseline"]
    )
    write_csv(comparisons, report / "baseline_vs_final_metrics.csv")

    map_integrity = _map_integrity(
        output=output,
        report=source_report,
        method=selected,
        expected_groups=groups,
    )
    write_csv(map_integrity, report / "selected_map_integrity.csv")

    rq_predictions = pd.read_csv(
        source_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    invariance_rows = []
    for group in groups:
        baseline_maps = np.load(
            output
            / "crossfit/generated_maps"
            / baseline
            / f"{group}.npz",
            allow_pickle=False,
        )["generated_unit_shapes"]
        selected_maps = np.load(
            output
            / "crossfit/generated_maps"
            / selected
            / f"{group}.npz",
            allow_pickle=False,
        )["generated_unit_shapes"]
        predicted_sq = float(
            rq_predictions.loc[group, "predicted_target"]
        )
        delta = float(np.max(np.abs(baseline_maps - selected_maps)))
        invariance_rows.append(
            {
                "growth_run_id": group,
                "predicted_sq_nm": predicted_sq,
                "smooth_branch_can_change": predicted_sq < 1.6,
                "max_absolute_unit_map_change": delta,
                "exactly_unchanged": delta == 0.0,
            }
        )
    invariance = pd.DataFrame(invariance_rows)
    if not bool(
        invariance.loc[
            ~invariance["smooth_branch_can_change"], "exactly_unchanged"
        ].all()
    ):
        raise RuntimeError("rough-branch invariance check failed")
    write_csv(invariance, report / "renderer_branch_invariance.csv")

    ncondition = pd.read_csv(
        source_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).loc[lambda frame: frame["growth_run_id"] == "N6342"].iloc[0]
    nconf = confidence.loc[
        confidence["growth_run_id"] == "N6342"
    ].iloc[0]
    figure_dir = source_report / "figures"
    figure_paths = sorted(
        str(path) for path in figure_dir.glob("*.png")
    )
    best_manifest = {
        "experiment_id": config["experiment_id"],
        "selected_method": selected,
        "cohort_count": len(groups),
        "growth_run_ids": groups,
        "excluded_growths": list(config["explicitly_excluded_growths"]),
        "6081_excluded_everywhere": "6081" not in groups,
        "outer_fit_growth_count": 26,
        "all_outer_fold_leakage_checks_passed": source_manifest[
            "all_outer_fold_leakage_checks_passed"
        ],
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "n6342": {
            "measured_sq_nm": float(ncondition["true_target"]),
            "predicted_sq_nm": float(ncondition["predicted_target"]),
            "absolute_error_nm": float(ncondition["absolute_error"]),
            "joint_confidence_index": float(nconf["joint_confidence_index"]),
        },
        "target_prediction_summary": target_summary.to_dict(orient="records"),
        "confidence_manifest": source_manifest["confidence"],
        "selected_generated_map_count": len(map_integrity),
        "exactly_unchanged_vs_m16b_count": int(
            invariance["exactly_unchanged"].sum()
        ),
        "rough_branch_invariance_passed": True,
        "config_path": str(repo_path(config_path)),
        "config_sha256": sha256_file(repo_path(config_path)),
        "source_report": str(source_report),
        "source_output": str(output),
        "figure_pngs": figure_paths,
        "claim_boundary": source_manifest["claim_boundary"],
    }
    write_json(best_manifest, report / "best_model_manifest.json")

    rq = target_summary.loc[target_summary["target"] == "Rq_nm"].iloc[0]
    fsmi = target_summary.loc[target_summary["target"] == "FSMI_nm"].iloc[0]
    markdown = f"""# M17b N6342 sparse-peak AFM generation report

Date: 2026-08-04
Status: completed retrospective method-development experiment

## Outcome

The selected method is **M17b topology-conditioned sparse-peak terrace**. It is a true stochastic generator: it uses RHEED-conditioned AFM descriptors, a learned spectral population prior and generated island geometry; it does not retrieve an AFM image and never uses the held growth's measured AFM patch at inference. Growth 6081 was added to `removelist.txt` before fitting and is absent from all 27 outer folds, maps and figures.

For N6342, strict leave-one-growth-out training uses the other 26 growths. The measured sample-median Sq is **{ncondition['true_target']:.3f} nm**, the prediction is **{ncondition['predicted_target']:.3f} nm** (absolute error **{ncondition['absolute_error']:.3f} nm**), and the final joint reliability index is **{nconf['joint_confidence_index']:.1f}/100**. N6342 motivated method development, so this is retrospective LOO development evidence rather than an untouched prospective-test result.

## Why the old N6342 image looked too dot-dense

M16b selected a fixed dense local-maximum field and then applied a final `tanh` compression. That made many moderate extrema look like similarly bright circular dots. M17b replaces it with a sparse peak layer whose count is predicted, inside each outer fold, from the RHEED-conditioned q82 island-component descriptor. Fine spectral residuals remain, so removing visually dominant peaks does not make the surface featureless. The new branch is smoothly gated below predicted Sq 1.6 nm; rough samples retain the prior terrace renderer.

## N6342 morphology comparison

| diagnostic | M16b | M17b | measured/reference |
| --- | ---: | ---: | ---: |
| normalized PSD log distance | {nbaseline_standard['normalized_psd_log_distance']:.3f} | **{nselected_standard['normalized_psd_log_distance']:.3f}** | lower is better |
| composite morphology score | {nbaseline_standard['composite_score']:.3f} | **{nselected_standard['composite_score']:.3f}** | lower is better |
| island-feature MAE (z) | {nbaseline_island['island_feature_mae_z']:.3f} | **{nselected_island['island_feature_mae_z']:.3f}** | lower is better |
| all-feature peak-signature MAE (z) | **{nbaseline_peak['peak_signature_mae_z']:.3f}** | {nselected_peak['peak_signature_mae_z']:.3f} | includes low-amplitude peak count |
| visually persistent peaks (h=0.5 Sq) | {nbaseline_peak['persistent_peak_count_h050_generated']:.1f} | **{nselected_peak['persistent_peak_count_h050_generated']:.1f}** | 82.0; count alone does not encode peak area/intensity |
| bright area fraction (>1.5 Sq) | {nbaseline_peak['excursion_fraction_z150_generated']:.4f} | **{nselected_peak['excursion_fraction_z150_generated']:.4f}** | 0.0540 |
| bright component median area (px) | {nbaseline_peak['excursion_median_area_z100_px_generated']:.1f} | **{nselected_peak['excursion_median_area_z100_px_generated']:.1f}** | 26.0 |
| height kurtosis | {nbaseline_peak['height_kurtosis_generated']:.3f} | **{nselected_peak['height_kurtosis_generated']:.3f}** | 3.327 |
| fine-detail RMS fraction | {nbaseline_peak['fine_detail_rms_fraction_generated']:.3f} | **{nselected_peak['fine_detail_rms_fraction_generated']:.3f}** | 0.322 |

The main improvement is therefore not a claim of pixel registration. It is a closer stochastic morphology distribution in the failure dimensions identified by the domain reviewer: fewer visually dominant peaks, less excess bright area, larger coherent bright components, corrected tail shape, much closer PSD and preserved fine texture. The N6342 all-feature peak-signature aggregate changes slightly adversely because the measured scan itself contains many low-amplitude local maxima and that audit gives their count equal weight. SSIM is not used as the main selection criterion because generated morphology is not expected to align island-for-island with a measured AFM scan.

## Full 27-growth non-inferiority audit

Relative to M16b, M17b changes the cohort-mean normalized PSD distance from **{baseline_method['mean_normalized_psd_log_distance']:.3f}** to **{selected_method['mean_normalized_psd_log_distance']:.3f}**, island-feature MAE from **{baseline_method['mean_island_feature_mae_z']:.3f}** to **{selected_method['mean_island_feature_mae_z']:.3f}**, and the new peak-signature MAE from **{baseline_peak['mean_peak_signature_mae_z']:.3f}** to **{selected_peak['mean_peak_signature_mae_z']:.3f}**. Mean sharpness ratio improves from **{baseline_method['mean_sharpness_ratio']:.3f}** to **{selected_method['mean_sharpness_ratio']:.3f}**. Texture-gate pass fraction decreases slightly from **{baseline_method['mean_afm_texture_gate_pass']:.3f}** to **{selected_method['mean_afm_texture_gate_pass']:.3f}**; this limitation is retained rather than hidden. M10 remains the strongest aggregate baseline on some population-average texture/island metrics, while M17b is selected for the N6342-specific failure mode and its improvement over the deployed M16b baseline.

The renderer branch-invariance audit confirms that all growths with predicted Sq at or above 1.6 nm are bitwise unchanged from M16b; **{int(invariance['exactly_unchanged'].sum())}/27** generated ensembles are exactly unchanged overall. The remaining changes are confined to the configured smooth/interpolation regime.

The scalar heads are unchanged by renderer selection and were retrained after excluding 6081. Across all 27 held growths, Sq has MAE **{rq['mean_absolute_error']:.3f} nm**, Pearson r **{rq['pearson_r']:.3f}**, Spearman rho **{rq['spearman_rho']:.3f}**; FSMI has MAE **{fsmi['mean_absolute_error']:.3f} nm**, Pearson r **{fsmi['pearson_r']:.3f}**, Spearman rho **{fsmi['spearman_rho']:.3f}**. `Rq_nm` remains a legacy internal column name for the audited areal Sq target.

The selected M17b joint confidence is strictly cross-fitted and combines expected FSMI and island-topology error. Confidence versus realized joint error is Spearman rho **{source_manifest['confidence']['joint_confidence_vs_realized_error_spearman']:.3f}** (p={source_manifest['confidence']['joint_confidence_vs_realized_error_pvalue']:.4g}); it is a relative reliability index, not a probability.

## Ablations and negative results

- M17a showed that fixed sparse peaks already remove the M16b dot-field artifact, but M17b makes peak count conditional on RHEED-derived morphology.
- M17c/M17d added more high-frequency texture but increased cohort PSD error and reduced texture-gate performance.
- M17e broadened peaks and has a competitive cohort composite, but overshot N6342 kurtosis.
- M17h/M17i added a two-level shoulder/peak hierarchy. The islands looked broader, but N6342 kurtosis rose to an implausible range, so these variants were rejected.
- All 11 methods remain in `experiment_registry.csv`; no failed candidate was overwritten.

## Figures and tables

- Complete fixed-order 27-growth atlas: `{figure_dir}/Fig1a_full27_loo_atlas.png` through `Fig1f_full27_loo_atlas.png` (also PDF).
- Scalar scatter: `{figure_dir}/Fig2_full27_target_scatter.png`.
- Confidence/error audit: `{figure_dir}/Fig5_confidence_audit.png`.
- Roughness-stratified renderer comparison: `{figure_dir}/Fig6_renderer_roughness_strata.png`.
- Largest-error cases: `{figure_dir}/Fig7_largest_error_cases.png`.
- Extra-five panels: `{figure_dir}/Fig8_extra_five_generated_afm.png` and `Fig9_extra_five_renderer_comparison.png`.
- N6342 focused ablation: `{figure_dir}/Fig10_N6342_renderer_ablation.png`.
- N6342 peak-topology diagnostics: `{figure_dir}/Fig11_N6342_peak_signature.png`.
- Baseline/final table: `{report}/baseline_vs_final_metrics.csv`.
- All candidates: `{report}/experiment_registry.csv`.
- Selected-map hash/integrity audit: `{report}/selected_map_integrity.csv`.
- Smooth/rough renderer invariance audit: `{report}/renderer_branch_invariance.csv`.

## Reproduction

```bash
cd {repo_path('.')}
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_auto_input_robustness.run --config configs/rheed_auto_input_robustness_line3_full27_exclude6081_v4.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_endpoint_generation.run_endpoint_ensemble \\
  --perturbation-embeddings outputs/rheed_auto_input_robustness/20260729_m15b_line3_full28_orientation90_keyframe_locked_v3/r3d_causal8_input_perturbations.npz \\
  --targets reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/expanded_afm_targets.csv \\
  --manifest outputs/extra_five_integration/20260729_line3_full28_orientation90_keyframe_locked_v3/machine_dataset_full28/modeling_manifest.csv \\
  --baseline-predictions reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/m15b_strict_loo_predictions.csv \\
  --baseline-nested reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/m15b_nested_inner_predictions.csv \\
  --data-root . --output outputs/rheed_endpoint_generation/m16_full27_exclude6081_v2 \\
  --removelist removelist.txt
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json --device mps
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_n6342_sparse_island.evaluate --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_n6342_sparse_island.report --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
```

## Limitations

The cohort contains 27 independent growths, so renderer choice and uncertainty remain small-data estimates. N6342 is no longer an untouched test because it motivated this work. Pixelwise correspondence is not identifiable from one RHEED observation, and the model produces a plausible conditional morphology realization rather than the exact AFM scan. Prospective validation on new streaky GaSb growths is the next decisive test.

See `literature_review.md` for the literature basis and claim boundaries.
"""
    (report / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(best_manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
