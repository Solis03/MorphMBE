# M23 low-Sq discrete-island AFM and 6022 full-flow exclusion

Date: 2026-08-09
Branch: `codex/m23-smooth-discrete-islands-removelist-6022-20260809`

## Outcome

M23 replaces the filament/cotton-like M22 smooth endpoint with explicit finite
round/elliptical islands. The selected `M23c_dense_discrete_smooth_islands`
renderer is active below predicted Sq 3.3 nm, transitions to the frozen M22
renderer from 3.3 to 3.8 nm, and returns the exact M22 arrays at and above
3.8 nm. Growth 6022 is now in the canonical removal list and is absent from
the Sq head, FSMI head, morphology statistics, all outer/inner fits, generated
maps, and evaluation tables.

The desktop
`MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804` package and its results
were read only for the requested baseline atlas. All M23 code and derived
results are in this repository.

## Diagnosis and model design

The old low-Sq result was not mainly an amplitude error. Its stationary
spectral/capture-zone mixture connected neighboring maxima into fibres and
left broad dark pools. The measured low-Sq AFMs instead contain a high-density
population of individually legible grains with a compact height marginal.

M23 therefore uses:

1. finite elliptical radial mounds rather than a continuous spectral field;
2. high-clearance stochastic placement, sampled among several admissible
   positions so coverage stays uniform without becoming a lattice;
3. low-order boundary harmonics and learned eccentricity to avoid perfect
   circles;
4. a stage-dependent second population of smaller islands, which subdivides
   remaining channels as the surface approaches the intermediate state;
5. a fixed rank-preserving skew-normal height marginal (`shape = -1.0`) that
   matches the compact low-Sq AFM height distribution without moving island
   boundaries or copying a measured AFM patch;
6. exact M22 rough-branch reuse at predicted Sq >= 3.8 nm.

The v1-v6 screening was intentionally iterative. v1 was too sparse, v2 left
clusters and gaps, v3 became a bright dot lattice, v4 formed close-packed
cobbles but retained excessive black tone, v5 corrected the height marginal,
and v6 added non-crystalline placement, boundary variation, and a partial
second growth layer. v6 is the selected result.

## Strict 26-growth Sq result

| Metric | M23, 6022 excluded before every fit |
|---|---:|
| Growth groups | 26 |
| Outer folds / fit groups per fold | 26 / 25 |
| MAE | 0.768 nm |
| RMSE | 0.904 nm |
| Pearson r | 0.911 |
| Spearman rho | 0.746 |
| Interval coverage | 1.000 |

Removing 6022 costs some scalar accuracy: the previous M22 27-growth result
had MAE 0.685 nm, and the same predictions after merely dropping the 6022 row
had MAE 0.699 nm. Those numbers are not a valid replacement for M23 because
their scalar model was trained with 6022 in eligible folds. M23 reports the
honest full-flow exclusion result instead.

The rough endpoint remains ordered: 6095 is 9.074 -> 8.525 nm and 6099 is
9.395 -> 8.381 nm. The largest remaining scalar errors are 6085, 6048, and
6084; their renderings inherit the scalar-head regime error. An exploratory
post-hoc isotonic calibration reduced aggregate error in a non-nested screen,
but it did not fix those samples and was rejected because a fully nested
meta-calibration was not established.

## Low-Sq AFM tone and dark-region topology

Every map is evaluated with the same independent 0.5%-99.5% height limits used
in the Gwyddion atlas. A dark pixel is at or below 0.18 of its own display
range.

| User low-Sq focus (7 growths) | Dark fraction | Largest dark component | Display median | Height skew |
|---|---:|---:|---:|---:|
| Measured AFM | 0.0417 | 0.00465 | 0.522 | -0.149 |
| Frozen M22 | 0.0586 | 0.01253 | 0.489 | +0.034 |
| M23 v6 | **0.0414** | **0.00556** | **0.529** | **-0.137** |

Across all 19 groups with predicted Sq <= 3.3 nm, M23 has dark fraction
0.0414 versus 0.0395 measured, and largest dark component 0.00580 versus
0.00451 measured. The rank-preserving marginal mapping makes the tone match;
the explicit ellipses and second layer make the geometry visibly granular
rather than filamentary.

For the five measured 3.5-6.0 nm intermediate groups, M23 dark fraction is
0.0372 versus 0.0324 measured, and largest dark component is 0.00524 versus
0.00270 measured. This is still less exact than the low-Sq tone match and is a
known target for prospective validation.

## Regression and exclusion protection

- 6022 removal audit: passed.
- 6022 rows in cohort, Sq/FSMI predictions, or generated maps: zero.
- 6022 occurrences in every outer/inner morphology fit manifest: zero.
- Outer folds: 26; held-growth/fit overlap: zero.
- `all_outer_fold_leakage_checks_passed`: true.
- Retrieval or measured query-AFM use at inference: false.
- Six samples triggered the M22 rough branch: 6028, 6062, 6063, 6085, 6095,
  and 6099.
- M22-versus-M23 equality over all four generated draws for those samples:
  exact; maximum absolute pixel difference 0.0.

## Visualization

The six-page atlas contains held-out RHEED, measured AFM, desktop standalone
M17, and M23. Every AFM has its own physical height bar and the official
Gwyddion.net black-rust-orange-gold-white palette. M19 is not displayed.

Primary artifacts:

- full atlas directory:
  `reports/rheed_m23_smooth_discrete_islands/20260809_m23_full26_v6/full26_loo/figures/gwyddion_individual_height_atlas_M17_standalone_vs_M23/`
- low-Sq focus:
  `Focus_N6342_N6358_N6382_6048_6056_6070_6078_Gwyddion_individual_height_M17_vs_M23.png`
- Sq comparison:
  `M23_Sq_measured_vs_predicted_ordered.png`
- per-group tone audit: `m23_display_metrics_per_group.csv`
- subset tone audit: `m23_display_metrics_summary.csv`
- rough freeze audit: `m23_rough_branch_protection_audit.csv`
- exclusion audit: `m23_exclusion_6022_audit.json`

## Literature basis

- RHEED/AFM correspondence between transmission spots and elongated islands:
  https://www.sciencedirect.com/science/article/abs/pii/S0022024898000712
- Random nucleation and island-growth Monte Carlo models that reproduce AFM
  film profiles and coalescence:
  https://experts.illinois.edu/en/publications/coalescence-of-ultrathin-films-by-atomic-layer-deposition-or-chem/
- AFM-observed island separation followed by lateral coherent merging to a
  flatter surface:
  https://www.nature.com/articles/s41467-023-36301-w
- Kinetic simulation of island growth, coalescence, and AFM roughness
  evolution:
  https://www.sciencedirect.com/science/article/pii/S0169433221010229
- 2024 neural implicit AFM reconstruction, supporting explicit geometry and
  consistency constraints over purely textural synthesis:
  https://www.nature.com/articles/s44172-024-00270-9

These papers motivate the nucleation/coalescence representation and explicit
geometry constraints. They do not validate this dataset-specific parameter
choice; M23 remains retrospective method development.

## Reproducibility

Final config:

`configs/rheed_m23_smooth_discrete_islands_full26_v6.json`

Main commands:

```text
PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run --config configs/rheed_m23_smooth_discrete_islands_full26_v6.json --device mps
PYTHONPATH=. .venv/bin/python -m analysis.rheed_rough_island_redesign.smooth_island_evaluate --config configs/rheed_m23_smooth_discrete_islands_full26_v6.json
PYTHONPATH=. .venv/bin/python -m analysis.rheed_rough_island_redesign.gwyddion_atlas --config configs/rheed_m23_smooth_discrete_islands_full26_v6.json --standalone-root /Users/ziyi/Desktop/MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804
```

Full 26-fold morphology runtime was 455.8 s on Apple Silicon MPS.

## Verification

The M23 implementation and all directly affected pipelines passed 58/58
targeted tests. Static checks also passed:

```text
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_rheed_auto_input_robustness.py \
  tests/test_rheed_to_afm_full_cohort_loo.py \
  tests/test_rheed_to_afm_functional_morphology.py \
  tests/test_rheed_to_afm_island_generation.py \
  tests/test_rheed_to_afm_conditional_vae.py \
  tests/test_rheed_realtime_ui.py
# 58 passed

uvx ruff check <changed Python files>
# All checks passed

git diff --check
# passed
```

The broader `pytest -q tests` run completed with 398 passed, 29 failed, and
6 errors. None of those failures execute the M23 path: they arise from
missing historical paper-freeze manifests and peak-saddle checkpoint
artifacts, frozen tests that intentionally expect the pre-M23 removelist
hash, and an unavailable optional parquet engine (`pyarrow`/`fastparquet`).
Running pytest from the repository root also finds duplicate test module
names inside historical paper-freeze code snapshots and stops during
collection. These pre-existing repository-state issues are recorded rather
than hidden or repaired by changing frozen artifacts.

## Limitations

- The explicit-island prior is visibly closer to the requested low-Sq physics,
  but it can still look too regular on some samples.
- Sq errors can place a sample in the wrong morphology regime; 6085 is the
  clearest remaining example.
- The low-Sq renderer and its fixed marginal were selected retrospectively on
  this cohort. A new untouched low/intermediate-Sq growth batch is required
  for a prospective claim.
- Exact per-sample island placement is not identifiable from the selected
  RHEED input and is not claimed.
- Raw RHEED/AFM data, publication freezes, desktop standalone code, and its
  result folders were not modified.
