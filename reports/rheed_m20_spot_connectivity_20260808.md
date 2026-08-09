# M20 RHEED spot-connectivity Sq and AFM-island redesign

## Outcome

M20 resolves the 6062/6099 physical-ordering failure in M19. It uses
target-blind RHEED spot topology both to calibrate Sq and to control generated
island count, footprint, separation, height, and substrate-valley depth. The
selected renderer is `M20b_connectivity_coupled_islands` (medium coupling).

The desktop M17 standalone package and its result folders were used only as the
frozen baseline. All implementation and derived results are in this repository.

## Diagnosis

M19's rough-tail gate was driven mainly by temporal/expert consensus. It did
not distinguish a horizontally bridged spot field from persistent, round,
isolated spots. This caused 6062 to inherit an excessively rough amplitude and
AFM surface while 6099 remained under-confident. Separate per-row AFM color
normalization also made cross-sample contrast visually ambiguous.

The most discriminating target-blind topology features were:

- component merge rate when relaxing the RHEED threshold from p97 to p90;
- persistent p90 component count;
- round-component fraction;
- p90 large-component area.

For 6062 and 6099 the resulting isolation scores are 0.382 and 0.840,
respectively. The full atlas therefore uses one symmetric physical nm color
scale across every measured and generated surface.

## Method

1. Fit a robust-scaled, distance-weighted three-neighbor log-residual
   calibrator inside each strict outer leave-one-growth-out fold.
2. Correct bridged/non-round rough-tail cases only when independent M19 rough
   support is already active.
3. For a highly isolated rough spot field, use 45% of the existing target-blind
   upper uncertainty headroom. The query AFM Sq never enters this decision.
4. Feed the same isolation score to the separated-ellipse generator. More
   isolated spots yield fewer, larger, taller, and more separated islands.
5. Couple isolation to the negative height tail before unit-Sq projection so
   isolated spots produce a deeper connected substrate, while bridged spots
   compress the negative tail.

Weak, medium, and strong topology couplings were run as ablations. Strong
coupling slightly improves aggregate island/PSD metrics but visually breaks
6062 into overly isolated, regular islands. Medium coupling preserves 6062's
coalesced low-contrast morphology while giving 6099 large separated islands and
deep trenches, so it is the selected balance.

## Sq results

| Evaluation | M19 MAE (nm) | M20 MAE (nm) | M20 RMSE (nm) |
|---|---:|---:|---:|
| All 27 growth groups | 0.966 | 0.685 | 0.829 |
| Rough 3–10 nm, 9 groups | 1.095 | 0.583 | 0.683 |

Across all 27 growth groups, M20 has Pearson `r = 0.923` and Spearman
`rho = 0.786`. Smooth-surface MAE below 1.6 nm remains 0.584 nm, unchanged from
M19.

| Growth | Measured Sq (nm) | M19 (nm) | M20 (nm) | M19 abs. error | M20 abs. error |
|---|---:|---:|---:|---:|---:|
| 6062 | 3.092 | 6.169 | 3.223 | 3.076 | 0.131 |
| 6099 | 9.395 | 7.389 | 8.459 | 2.006 | 0.936 |

For the selected physical AFM realization, the median-to-q01 negative-tail
depth is 4.75 nm for 6062 and 14.81 nm for 6099. Their q01–q99 height ranges
are 12.06 and 31.76 nm. All six pairwise direction checks in
`connectivity_audit_manifest.json` pass.

## Leakage and data-safety checks

- Outer protocol: 27 folds, fit 26 growth groups and hold out one complete
  growth group per fold.
- `all_outer_fold_leakage_checks_passed = true`.
- RHEED connectivity correction excludes the held growth's AFM target.
- Retrieval and measured AFM patch use at inference are both false.
- No raw RHEED/AFM file, removelist, publication freeze, desktop standalone
  code, or desktop standalone result was modified.
- Gate thresholds and coupling strengths were developed retrospectively on the
  27-growth cohort. Prospective 3–10 nm confirmation is still required.

## Main artifacts

- Final config: `configs/rheed_m20_spot_connectivity_islands_full27_v2.json`
- Cross-fitted Sq predictions:
  `outputs/rheed_m20_spot_connectivity/20260808_m20_connectivity_full27_v2/`
- Full derived results:
  `outputs/rheed_m20_spot_connectivity/20260808_m20_full27_v2/full27_loo/`
- Reports and audits:
  `reports/rheed_m20_spot_connectivity/20260808_m20_full27_v2/full27_loo/`
- Six-page global physical atlas and 6062/6099 focus image:
  `reports/rheed_m20_spot_connectivity/20260808_m20_full27_v2/full27_loo/figures/global_physical_atlas_M20b_connectivity_coupled_islands/`

## Literature basis

- Thin Solid Films (1998), RHEED intensity-profile changes and surface
  roughness: https://www.sciencedirect.com/science/article/abs/pii/S0040609098014035
- SrTiO3 RHEED/AFM study, spotty transmission diffraction from 3D islands:
  https://www.sciencedirect.com/science/article/abs/pii/S0921453498003384
- Si/Ge morphology study, streaky-to-spotty RHEED as a 2D-to-3D transition:
  https://www.sciencedirect.com/science/article/abs/pii/S0022024898014146
- InGaN quantum-dot growth, streaky-to-spotty RHEED and AFM dot morphology:
  https://www.sciencedirect.com/science/article/pii/S0040609013008250

## Verification

- Changed-file lint: passed.
- M20/island/renderer/full-cohort targeted tests: 24 passed.
- Physical consistency audit: all six checks passed.
- Full repository suite: 388 passed, 29 failed, 6 errors. The non-passing
  tests require pre-existing unavailable paper-freeze manifests, peak-saddle
  checkpoint outputs, an old removelist hash, or a parquet engine. None touches
  the M20 implementation or its targeted test set.
