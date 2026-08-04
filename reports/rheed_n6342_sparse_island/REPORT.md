# M17b N6342 sparse-peak AFM generation report

Date: 2026-08-04
Status: completed retrospective method-development experiment

## Outcome

The selected method is **M17b topology-conditioned sparse-peak terrace**. It is a true stochastic generator: it uses RHEED-conditioned AFM descriptors, a learned spectral population prior and generated island geometry; it does not retrieve an AFM image and never uses the held growth's measured AFM patch at inference. Growth 6081 was added to `removelist.txt` before fitting and is absent from all 27 outer folds, maps and figures.

For N6342, strict leave-one-growth-out training uses the other 26 growths. The measured sample-median Sq is **0.804 nm**, the prediction is **0.833 nm** (absolute error **0.029 nm**), and the final joint reliability index is **71.4/100**. N6342 motivated method development, so this is retrospective LOO development evidence rather than an untouched prospective-test result.

## Why the old N6342 image looked too dot-dense

M16b selected a fixed dense local-maximum field and then applied a final `tanh` compression. That made many moderate extrema look like similarly bright circular dots. M17b replaces it with a sparse peak layer whose count is predicted, inside each outer fold, from the RHEED-conditioned q82 island-component descriptor. Fine spectral residuals remain, so removing visually dominant peaks does not make the surface featureless. The new branch is smoothly gated below predicted Sq 1.6 nm; rough samples retain the prior terrace renderer.

## N6342 morphology comparison

| diagnostic | M16b | M17b | measured/reference |
| --- | ---: | ---: | ---: |
| normalized PSD log distance | 2.832 | **0.260** | lower is better |
| composite morphology score | 10.173 | **9.729** | lower is better |
| island-feature MAE (z) | 1.010 | **0.890** | lower is better |
| all-feature peak-signature MAE (z) | **0.677** | 0.690 | includes low-amplitude peak count |
| visually persistent peaks (h=0.5 Sq) | 84.5 | **58.5** | 82.0; count alone does not encode peak area/intensity |
| bright area fraction (>1.5 Sq) | 0.0680 | **0.0509** | 0.0540 |
| bright component median area (px) | 17.0 | **19.5** | 26.0 |
| height kurtosis | 2.651 | **3.337** | 3.327 |
| fine-detail RMS fraction | 0.239 | **0.293** | 0.322 |

The main improvement is therefore not a claim of pixel registration. It is a closer stochastic morphology distribution in the failure dimensions identified by the domain reviewer: fewer visually dominant peaks, less excess bright area, larger coherent bright components, corrected tail shape, much closer PSD and preserved fine texture. The N6342 all-feature peak-signature aggregate changes slightly adversely because the measured scan itself contains many low-amplitude local maxima and that audit gives their count equal weight. SSIM is not used as the main selection criterion because generated morphology is not expected to align island-for-island with a measured AFM scan.

## Full 27-growth non-inferiority audit

Relative to M16b, M17b changes the cohort-mean normalized PSD distance from **1.454** to **0.969**, island-feature MAE from **1.738** to **1.718**, and the new peak-signature MAE from **1.457** to **1.327**. Mean sharpness ratio improves from **0.810** to **0.836**. Texture-gate pass fraction decreases slightly from **0.852** to **0.815**; this limitation is retained rather than hidden. M10 remains the strongest aggregate baseline on some population-average texture/island metrics, while M17b is selected for the N6342-specific failure mode and its improvement over the deployed M16b baseline.

The renderer branch-invariance audit confirms that all growths with predicted Sq at or above 1.6 nm are bitwise unchanged from M16b; **18/27** generated ensembles are exactly unchanged overall. The remaining changes are confined to the configured smooth/interpolation regime.

The scalar heads are unchanged by renderer selection and were retrained after excluding 6081. Across all 27 held growths, Sq has MAE **1.107 nm**, Pearson r **0.741**, Spearman rho **0.684**; FSMI has MAE **1.126 nm**, Pearson r **0.675**, Spearman rho **0.583**. `Rq_nm` remains a legacy internal column name for the audited areal Sq target.

The selected M17b joint confidence is strictly cross-fitted and combines expected FSMI and island-topology error. Confidence versus realized joint error is Spearman rho **-0.565** (p=0.002126); it is a relative reliability index, not a probability.

## Ablations and negative results

- M17a showed that fixed sparse peaks already remove the M16b dot-field artifact, but M17b makes peak count conditional on RHEED-derived morphology.
- M17c/M17d added more high-frequency texture but increased cohort PSD error and reduced texture-gate performance.
- M17e broadened peaks and has a competitive cohort composite, but overshot N6342 kurtosis.
- M17h/M17i added a two-level shoulder/peak hierarchy. The islands looked broader, but N6342 kurtosis rose to an implausible range, so these variants were rejected.
- All 11 methods remain in `experiment_registry.csv`; no failed candidate was overwritten.

## Figures and tables

- Complete fixed-order 27-growth atlas: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig1a_full27_loo_atlas.png` through `Fig1f_full27_loo_atlas.png` (also PDF).
- Scalar scatter: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig2_full27_target_scatter.png`.
- Confidence/error audit: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig5_confidence_audit.png`.
- Roughness-stratified renderer comparison: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig6_renderer_roughness_strata.png`.
- Largest-error cases: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig7_largest_error_cases.png`.
- Extra-five panels: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig8_extra_five_generated_afm.png` and `Fig9_extra_five_renderer_comparison.png`.
- N6342 focused ablation: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig10_N6342_renderer_ablation.png`.
- N6342 peak-topology diagnostics: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/Fig11_N6342_peak_signature.png`.
- Baseline/final table: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_n6342_sparse_island/baseline_vs_final_metrics.csv`.
- All candidates: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_n6342_sparse_island/experiment_registry.csv`.
- Selected-map hash/integrity audit: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_n6342_sparse_island/selected_map_integrity.csv`.
- Smooth/rough renderer invariance audit: `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804/reports/rheed_n6342_sparse_island/renderer_branch_invariance.csv`.

## Reproduction

```bash
cd /Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_auto_input_robustness.run --config configs/rheed_auto_input_robustness_line3_full27_exclude6081_v4.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_endpoint_generation.run_endpoint_ensemble \
  --perturbation-embeddings outputs/rheed_auto_input_robustness/20260729_m15b_line3_full28_orientation90_keyframe_locked_v3/r3d_causal8_input_perturbations.npz \
  --targets reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/expanded_afm_targets.csv \
  --manifest outputs/extra_five_integration/20260729_line3_full28_orientation90_keyframe_locked_v3/machine_dataset_full28/modeling_manifest.csv \
  --baseline-predictions reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/m15b_strict_loo_predictions.csv \
  --baseline-nested reports/rheed_auto_input_robustness/20260804_m15b_line3_full27_exclude6081_v4/m15b_nested_inner_predictions.csv \
  --data-root . --output outputs/rheed_endpoint_generation/m16_full27_exclude6081_v2 \
  --removelist removelist.txt
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json --device mps
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_n6342_sparse_island.evaluate --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
PYTHONPATH=. /Users/ziyi/Desktop/LAB/code/.venv/bin/python -m analysis.rheed_n6342_sparse_island.report --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json
```

## Limitations

The cohort contains 27 independent growths, so renderer choice and uncertainty remain small-data estimates. N6342 is no longer an untouched test because it motivated this work. Pixelwise correspondence is not identifiable from one RHEED observation, and the model produces a plausible conditional morphology realization rather than the exact AFM scan. Prospective validation on new streaky GaSb growths is the next decisive test.

See `literature_review.md` for the literature basis and claim boundaries.
