# Full 23-growth RHEED-to-AFM leave-one-out audit

**Experiment:** `20260728_m13_full23_nested_loo_v1`

**Date:** 2026-07-28

**Branch:** `codex/rheed-afm-morphology-index-20260727`

**Status:** completed retrospective audit; **no performance improvement**

## Executive conclusion

The requested experiment was completed over all 23 approved growth groups.
Growths 6043 and 6055 were not used. Each growth was held out exactly once,
the remaining 22 growths were used to refit every target-dependent component,
and an AFM ensemble was generated before the held growth's AFM was evaluated.
All 23 fold-integrity checks passed.

The additional eight growths did **not** improve the frozen M12 method. Full
23-growth LOO gave Rq Pearson \(r=0.265\), Spearman
\(\rho=0.303\), and mean/median absolute errors of 1.910/1.322 nm. The prior
15-growth M12 audit gave \(r=0.836\), \(\rho=0.932\), and
0.664/0.281 nm. The regression also became worse when the new 23-growth
fits were evaluated only on the same 15 growth IDs, so the difference is not
explained merely by adding eight difficult points to the plotted cohort.

This is a scientifically important negative result. The earlier 15-growth
linear relationship is not robust to the broader cohort and must not be
presented as the project's final generalization claim.

## Fixed cohort and protocol

The harmonized cohort contains 23 independent growth groups and 116 AFM
scans:

`6022, 6028, 6029, 6033, 6047, 6048, 6056, 6057, 6062, 6063, 6070, 6072,
6078, 6080, 6081, 6082, 6084, 6085, 6090, 6094, 6095, 6099, 6101`.

- 6043: explicitly excluded at the user's request.
- 6055: explicitly excluded and also present in `removelist.txt`.
- The old train/validation/test labels are retained only as provenance:
  15 old-train, 3 old-validation, and 5 old-test groups.
- New outer protocol: leave one growth out, fit 22, predict one, repeat 23
  times.
- Inner refits per outer fold: Rq/FSMI target head, condition scaler and
  predictor, variance calibration, conditional spectrum, descriptor
  calibration, and island-condition model.
- Evaluation against the held AFM occurs only after the generated maps are
  produced.
- The method is stochastic conditional generation. It does not retrieve an
  AFM image or use a measured AFM patch at inference.

This protocol is retrospective cross-validation, not a prospective untouched
test. The M12 method family and its RHEED feature definitions were developed
using earlier partitions.

## Target-prediction results

| Protocol | Target | Groups | Mean MAE (nm) | Median MAE (nm) | RMSE (nm) | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior M12 strict LOO, 14 fit | Rq | 15 | 0.664 | 0.281 | 1.194 | 0.836 | 0.932 |
| M13 23-growth fits, same 15 plotted | Rq | 15 | 1.745 | 1.322 | 2.187 | 0.353 | 0.550 |
| **M13 full cohort, 22 fit** | **Rq** | **23** | **1.910** | **1.322** | **2.485** | **0.265** | **0.303** |
| Prior M12 strict LOO, 14 fit | FSMI | 15 | 0.722 | 0.417 | 1.189 | 0.765 | 0.832 |
| M13 23-growth fits, same 15 plotted | FSMI | 15 | 1.638 | 0.979 | 2.116 | 0.173 | 0.354 |
| **M13 full cohort, 22 fit** | **FSMI** | **23** | **1.748** | **0.979** | **2.352** | **0.158** | **0.237** |

The full cohort has measured Rq from 1.227 to 10.321 nm. Predictions span
0.007 to 8.270 nm, but the range is not trustworthy: 6101 is an unstable
near-zero extrapolation, while the two roughest surfaces are strongly
underpredicted:

- 6099: 10.321 nm measured, 3.679 nm predicted; error 6.642 nm.
- 6095: 9.867 nm measured, 4.551 nm predicted; error 5.316 nm.
- 6033: 2.539 nm measured, 6.180 nm predicted; error 3.641 nm.
- 6090: 4.685 nm measured, 8.270 nm predicted; error 3.585 nm.

An exploratory post-run sensitivity audit tested stronger Ridge
regularization and different fixed morphology/dynamics head weights. The
best observed Rq Pearson correlation was only about 0.37 (alpha 10,
morphology-only), and the best mean MAE was about 1.52 nm (alpha 10, equal
head weights). These checks do not rescue the conclusion and are not promoted
as a new selected model because they were inspected after the outer results.

## Generated-image results

M12a remains a true stochastic generator and its edge-preserving renderer
still produces visible island/terrace-like objects. In the full 23-growth
audit, however, inaccurate amplitude and condition predictions dominate
several samples.

| Full-23 median metric | M10 baseline | M12a edge-preserving |
|---|---:|---:|
| Rq absolute error (nm) | 1.410 | 1.410 |
| FSMI absolute error (nm) | 1.162 | **1.057** |
| PSD log distance | **0.948** | 1.118 |
| Sharpness ratio (target 1) | **0.798** | 0.714 |
| AFM texture-gate pass rate | **16/23** | 14/23 |
| Island-feature MAE (z) | **1.031** | 1.505 |
| AFM-prior Mahalanobis distance | **3.239** | 6.650 |
| Island-boundary contrast | 1.356 | **1.631** |
| Composite error | **7.845** | 7.863 |

Thus M12a preserves its clearer island-boundary advantage, but M10 is better
on most aggregate realism/texture metrics for this broader cohort. M12a
should remain a preserved visual milestone, not be claimed as the full-cohort
quantitative winner.

## Confidence audit

The 90% cross-fitted intervals have useful aggregate coverage:

- Rq: 20/23 = 86.96%.
- FSMI: 20/23 = 86.96%.
- Island-topology upper bound: 22/23 = 95.65%.

The pointwise confidence rank does **not** generalize:

- confidence versus realized joint-error rank:
  Spearman \(\rho=+0.043\), \(p=0.846\);
- predicted versus realized Rq error: \(\rho=0.290\);
- predicted versus realized FSMI error: \(\rho=0.172\);
- predicted versus realized island error: \(\rho=0.304\).

The index is not a probability and should not be used in a paper as evidence
that the model reliably “knows when it is wrong.” Figure 5 explicitly shows
this failure. Aggregate conformal intervals are the only uncertainty output
supported by the current full-cohort evidence.

## Failure diagnosis

1. **The original Rq head is not cohort-stable.** It uses 15 curated
   morphology features and one dynamics feature with only 22 fit groups per
   fold. Several coefficients and the robust scaling change substantially
   when the broader cohort is included.
2. **Rare high-roughness states remain unsupported.** The two roughest
   growths are each strongly underpredicted even when the other rough growth
   is in the fit set.
3. **The old 15-growth relationship was optimistic.** Performance falls on
   the identical 15 plotted IDs when the eight additional growths influence
   each fit.
4. **Range calibration can extrapolate catastrophically.** For 6101 the
   positive log-space model predicts an almost zero Rq, producing a flat
   generated surface. This is visible in Figure 6.
5. **Image rendering cannot repair incorrect conditioning.** The renderer
   can make sharper islands, but it cannot infer the missing height amplitude
   after the RHEED-to-condition head fails.

## Figures

- `Fig1a`–`Fig1e`: all 23 RHEED / generated AFM / measured AFM rows, ordered
  by measured Rq.
- `Fig2`: full-23 Rq and FSMI scatter with 90% intervals and old split
  provenance.
- `Fig3`: prior 15-growth versus new same-15 and all-23 metric comparison.
- `Fig4`: measured and predicted Rq over the ordered full cohort.
- `Fig5`: confidence-ranking failure and interval coverage.
- `Fig6`: five fixed roughness strata comparing M10, M12a, and measured AFM.
- `Fig7`: the four largest Rq failures, reported without cherry-picking.

All figures are available as 360-dpi PNG and vector-text PDF under
`reports/rheed_to_afm_full_cohort_loo/20260728_m13_full23_nested_loo_v1/full23_loo/figures`.

## Reproducibility and audit

- Configuration:
  `configs/rheed_to_afm_full_cohort_loo.json`.
- Runner:
  `analysis/rheed_to_afm_full_cohort_loo/run.py`.
- Visualization:
  `analysis/rheed_to_afm_full_cohort_loo/visualization.py`.
- Cohort manifest:
  `reports/rheed_to_afm_full_cohort_loo/20260728_m13_full23_nested_loo_v1/full23_loo/cohort_manifest.csv`.
- Per-fold leakage audit:
  `reports/rheed_to_afm_full_cohort_loo/20260728_m13_full23_nested_loo_v1/full23_loo/fold_integrity_audit.csv`.
- Generated arrays:
  `outputs/rheed_to_afm_full_cohort_loo/20260728_m13_full23_nested_loo_v1/full23_loo/crossfit/generated_maps`.
- Complete metrics and manifest:
  `reports/rheed_to_afm_full_cohort_loo/20260728_m13_full23_nested_loo_v1/full23_loo`.

The smoke and full runs are feasible on the local Apple Silicon machine; the
full 23-fold run took approximately 158 seconds. A CUDA handoff is not
recommended for this audit because the limitation is generalization evidence,
not runtime.

## Recommended next experiment

Do not tune another high-dimensional linear head on these same 23 outer
targets and report the resulting LOO score as independent confirmation.
Instead:

1. freeze this M13 audit as the cohort-wide falsification result;
2. replace the 16-feature amplitude head with a lower-dimensional,
   support-bounded hierarchical model that explicitly models growth regime
   and temporal RHEED uncertainty;
3. predeclare its features and hyperparameters using nested CV only;
4. acquire prospective high-Rq growths and confirm on an untouched cohort;
5. retain aggregate prediction intervals, but recalibrate pointwise
   confidence only after enough prospective errors exist.

No raw data was modified, and no remote action was performed.
