# M22 dense intermediate AFM: paired morphology-cohort experiment

Date: 2026-08-09
Branch: `codex/m22-dense-mid-dual-cohort-20260809`

## Outcome

M22 replaces the M21 intermediate-state random overlayer with a growth-stage
construction that (1) locates the widest remaining low substrate region,
(2) nucleates a finite oval island there, (3) expands intermediate islands
laterally to promote impingement, and (4) applies a monotone coalescence
response that raises low terraces faster than existing summits. The response
preserves height rank and narrow deep channels while changing the intermediate
height distribution from positive-skewed isolated peaks to a predominantly
high island layer with sparse valleys.

The selected method is `M22c_gap_completion_strong`. It is active only along
the predicted intermediate growth coordinate. The accepted M20 structure is
requested in the same run so that the inclusive protocol is exactly protected
at predicted Sq below 2.2 nm and at or above 7.6 nm.

## Paired protocols

Both protocols evaluate the same 27 held-out growths, use seed 173205, and read
the same frozen strict outer-LOO Sq/FSMI prediction files.

1. **Inclusive morphology fit**: all 26 non-held growths enter each AFM
   morphology fit.
2. **Exclude 6022/6101 from morphology only**: 6022 and 6101 never enter the
   condition, spectral, or island-statistic fits. There are 24 morphology
   training growths in ordinary folds and 25 when 6022 or 6101 is itself held
   out. They remain in the Sq training protocol wherever allowed by strict
   outer LOO, and all 27 remain evaluation targets.

Audit results:

- paired Sq rows are exactly identical;
- every `outer_target_used_for_training` flag is false;
- every held-growth/morphology-fit overlap flag is false;
- the exclusion protocol contains neither 6022 nor 6101 in any morphology fit;
- no measured query AFM is used at inference;
- all inclusive protected-regime arrays are pixel-identical to accepted M20.

## Sq performance (shared by both protocols)

| Metric | Full 27-growth strict outer LOO |
|---|---:|
| MAE | 0.685 nm |
| RMSE | 0.829 nm |
| Pearson r | 0.923 |
| Spearman rho | 0.786 |
| 90% interval coverage | 1.000 |

Sq is intentionally unchanged from the accepted M20/M21 amplitude head.

## Intermediate morphology results

The primary diagnostic subset contains the five groups with measured
sample-level Sq from 3.5 to 6.0 nm. Each AFM is normalized only for the same
independent Gwyddion display limits used in the atlas (0.5% and 99.5%
quantiles). A dark pixel is at or below 0.18 of that individual display range.

| Source | Dark fraction | Largest continuous dark block | Display median | Height skew | Island feature MAE-z |
|---|---:|---:|---:|---:|---:|
| Measured AFM | 0.0324 | 0.0027 | 0.602 | -0.640 | — |
| Accepted M21 | 0.1514 | 0.0360 | 0.398 | +0.375 | 1.243 |
| M22 inclusive | **0.0334** | **0.0029** | 0.558 | -0.222 | **1.142** |
| M22 exclude 6022/6101 | 0.0342 | 0.0032 | **0.563** | **-0.277** | 1.571 |

The inclusive protocol reduces the M21 dark fraction by 77.9% and the largest
dark block by 92.0%. Its intermediate island-feature MAE improves by 8.1%.
The exclusion protocol gives a slightly closer tone median/skew but worsens
island-feature MAE by 26.3% relative to M21. Therefore the inclusive protocol
is the recommended quantitative result, while the exclusion protocol is kept
as the requested morphology-cohort ablation rather than selected as the main
model.

Across all 27 growths, island-feature MAE-z is 1.261 for inclusive M22, 1.778
for exclusion M22, and 1.348 for accepted M21.

## Visual outputs

The paired atlas uses the official `Gwyddion.net` black-rust-gold-white palette
and an independent height bar for every AFM. Columns are held-out RHEED,
measured AFM, desktop standalone M17, inclusive M22, and morphology-exclusion
M22. There are six complete atlas pages, one five-growth intermediate focus
page, and the shared 27-growth Sq comparison plot.

Primary focus image:

`reports/rheed_m22_dense_mid/20260809_m22_paired_comparison/figures/gwyddion_individual_height_atlas_M17_vs_M22_dual/Focus_true_Sq_3p5_to_6p0_M17_vs_M22_dual.png`

## Reproducibility

Configurations:

- `configs/rheed_m22_dense_mid_full27_inclusive_v1.json`
- `configs/rheed_m22_dense_mid_full27_exclude_6022_6101_v1.json`

Main commands:

```text
.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m22_dense_mid_full27_inclusive_v1.json --device auto
.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m22_dense_mid_full27_exclude_6022_6101_v1.json --device auto
.venv/bin/python -m analysis.rheed_rough_island_redesign.dense_mid_evaluate --inclusive-config configs/rheed_m22_dense_mid_full27_inclusive_v1.json --excluded-config configs/rheed_m22_dense_mid_full27_exclude_6022_6101_v1.json
.venv/bin/python -m analysis.rheed_rough_island_redesign.dense_mid_dual_atlas --inclusive-config configs/rheed_m22_dense_mid_full27_inclusive_v1.json --excluded-config configs/rheed_m22_dense_mid_full27_exclude_6022_6101_v1.json --standalone-root /Users/ziyi/Desktop/MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804
```

Runtimes were 582.2 s for inclusive and 608.0 s for morphology exclusion on
Apple Silicon MPS.

## Tests and limitations

- Focused M22/full-cohort tests: 23 passed.
- Main `tests/` suite: 395 passed, 29 failed, 6 errors. The failures/errors are
  pre-existing repository-state issues: absent paper-freeze and RHEED
  peak-saddle checkpoint artifacts, frozen old removelist hashes, and missing
  parquet engine. None touches the M22 paths; all focused M22 tests pass.
- The experiment is retrospective method development over 27 growth groups,
  not a prospective untouched validation.
- Tone/topology now closely matches the requested intermediate regime, but
  exact per-sample island placement is neither identifiable from a single
  RHEED frame nor claimed.
- Raw RHEED and AFM data, the desktop standalone package, and its result folder
  were read only and were not modified.
