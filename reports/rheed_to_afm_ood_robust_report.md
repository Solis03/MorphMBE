# OOD-aware robust RHEED-to-AFM generation report

Date: 2026-07-28
Branch: `codex/rheed-afm-ood-robust-20260728`

## Executive result

M12a and all prior evidence were preserved. The selected M14i pipeline keeps
the stochastic M12a edge-preserving island/terrace image generator but
replaces its weak full-cohort morphology target head with a target-specific,
OOD-aware multiview head:

- Rq uses a 60:40 blend of curated RHEED physics features and causal temporal
  R3D features (M14g).
- FSMI uses target-blind RHEED-density weighted regression (M14b).
- uncertainty uses only information available without the held AFM target:
  temporal feature support, high-amplitude extrapolation and inner-fold
  residual calibration.

Every reported point is a strict outer leave-one-growth-out prediction: 22
growths are fit and the remaining growth is predicted, repeated 23 times.
There is no AFM-patch retrieval or nearest-neighbor AFM substitution at
inference. The final image is newly sampled by the M12a stochastic generator.

Compared with the frozen M12a full-23 target head, M14i reduces Rq MAE by
23.3% and FSMI MAE by 24.7%. Rq correlation rises from Pearson r = 0.265 to
0.509 and FSMI rank correlation rises from Spearman rho = 0.237 to 0.430.
The relative confidence index is meaningfully error-aware: joint confidence
versus realized joint target error has rho = -0.696.

This is the best locally defensible result, not a prospective final test.
All 23 growths have already participated in retrospective method development.
A new sealed growth cohort is required for an unbiased publication-level
confirmation.

## Data and protocol

- Cohort: 23 independent growth groups; 116 AFM scans.
- Growths 6043 and 6055 remain excluded under the previously agreed cohort.
- The canonical `removelist.txt` was not changed.
- Historical train/validation/test names are retained only as provenance.
  They do not define the current outer loop.
- Outer protocol: leave one entire growth group out; fit on 22; predict one.
- All scaling, imputation, PCA, sample weighting, target range calibration,
  residual quantiles and generator condition fitting are refit without the
  held growth.
- The generator fold audit contains 23/23 rows with
  `held_overlap_with_fit = False`.
- The supplied AFM for the held growth is used only after prediction for
  evaluation and figures.

## Experiment 1: target-blind hard exclusion

The requested 2–4 “bad sample” experiment was implemented without using AFM
target error to choose deletions. Fifteen curated RHEED physics features were
robust-scaled in leave-one-growth fashion. Three target-blind support
diagnostics were percentile-ranked and averaged:

1. three-neighbor RHEED distance;
2. maximum absolute robust feature z-score;
3. Ledoit–Wolf Mahalanobis distance.

The four most RHEED-atypical growths were 6101, 6063, 6029 and 6028. Separate
top-2, top-3 and top-4 sensitivity cohorts were evaluated with the otherwise
unchanged frozen M12a target head.

| Excluded RHEED-only OOD growths | n | Rq MAE (nm) | Rq r | Rq rho | FSMI MAE (nm) | FSMI r | FSMI rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 23 | 1.910 | 0.265 | 0.303 | 1.748 | 0.158 | 0.237 |
| 6101, 6063 | 21 | 1.960 | 0.271 | 0.218 | 1.688 | 0.210 | 0.325 |
| + 6029 | 20 | 2.321 | 0.253 | 0.218 | 1.966 | 0.211 | 0.298 |
| + 6028 | 19 | 2.041 | 0.156 | 0.191 | 1.715 | 0.135 | 0.249 |

The hard-exclusion hypothesis is rejected for Rq: none of the exclusions
improves MAE or rank correlation. FSMI has a small top-2 MAE gain, but not a
general improvement. Growth 6099, the largest high-Rq miss, is actually the
most in-domain sample by this handcrafted RHEED audit (rank 23/23). Its
failure is therefore consistent with conditional ambiguity or a missing
growth variable, not simple covariate OOD. No new IDs were added to
`removelist.txt`.

## Experiment 2: sample weighting and model improvements

The following candidates were evaluated under the same full-23 outer LOO:

- M12a: frozen alpha-1 curated/dynamic RHEED ridge head.
- M14a: stronger ridge regularization.
- M14b: target-blind k-neighbor RHEED-density sample weights.
- M14c: fold-local residual self-paced sample weights.
- M14d: causal temporal R3D embedding, standardized PCA and ridge.
- M14e/f/g: curated/temporal blends of 20:80, 40:60 and 60:40.
- M14h: nested inner-LOO support-aware selector.
- M14i: retrospective target-specific robust assembly, using M14g for Rq and
  M14b for FSMI.

The user’s downweighting hypothesis is partly supported. Target-blind density
weighting improves both targets and is selected for FSMI. Residual
self-paced weighting reduces Rq MAE relative to M12a (1.596 versus 1.910 nm)
but is inferior to density weighting and multiview fusion. This is expected:
residual weighting can suppress valid rare, high-roughness examples because
they are hard, thereby reinforcing the range-compression problem.

The fully nested per-fold method selector M14h is unstable at this sample size
(Rq MAE 1.766 nm). Fixed, low-capacity multiview fusion is more reliable.

## Final quantitative results

| Target and method | MAE (nm) | Median AE (nm) | RMSE (nm) | Pearson r (p) | Spearman rho (p) | Predicted range (nm) | True range (nm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rq, frozen M12a | 1.910 | 1.322 | 2.485 | 0.265 (0.221) | 0.303 (0.159) | 0.007–8.270 | 1.227–10.321 |
| Rq, final M14i | **1.466** | **1.191** | **2.054** | **0.509 (0.013)** | **0.499 (0.015)** | 0.650–7.057 | 1.227–10.321 |
| FSMI, frozen M12a | 1.748 | 0.979 | 2.352 | 0.158 (0.471) | 0.237 (0.276) | 0.002–7.397 | 0.838–9.324 |
| FSMI, final M14i | **1.316** | **0.771** | **2.066** | 0.281 (0.195) | **0.430 (0.041)** | 1.169–6.045 | 0.838–9.324 |

M14e has the highest isolated Rq Pearson correlation (r = 0.593), but M14g
was retained as the Rq component of M14i because it gives the more balanced
combination of MAE, linear correlation, rank correlation and confidence–error
behavior. M14i is a retrospective fixed-candidate selection on this cohort;
this choice must be frozen before evaluating any new prospective cohort.

## Confidence and selective risk

For each outer held growth and target:

1. the R3D feature-space k-neighbor distance is calibrated relative to the
   22-growth training support;
2. upward target-amplitude extrapolation is measured relative to the
   training median and IQR;
3. their sum forms a fold-local epistemic-risk score;
4. inner-LOO residuals set expected error and a 90% adaptive residual radius;
5. confidence is `exp(-risk / fold-local risk scale)`.

Confidence is a relative index, not a correctness probability.

| Target | Confidence vs absolute error rho (p) | 90% interval coverage | Mean interval width |
|---|---:|---:|---:|
| Rq | -0.601 (0.0024) | 20/23 (86.96%) | 6.72 nm |
| FSMI | -0.677 (0.00039) | 20/23 (86.96%) | 6.50 nm |

Joint confidence (the geometric mean of Rq and FSMI confidence) versus
realized joint standardized target error is rho = -0.696. Selective risk
behaves in the intended direction:

| Retained highest-confidence growths | Rq MAE (nm) | FSMI MAE (nm) |
|---:|---:|---:|
| 100.0% (23) | 1.466 | 1.316 |
| 91.3% (21) | 1.418 | 1.223 |
| 82.6% (19) | 1.266 | 0.963 |
| 73.9% (17) | 1.045 | 0.741 |
| 60.9% (14) | 0.898 | 0.762 |
| 52.2% (12) | 0.707 | 0.715 |

This supports an industrial workflow in which low-confidence predictions are
flagged for additional AFM measurement or for deliberate expansion of the
RHEED training domain. It does not justify hiding low-confidence cases from
scientific evaluation.

## Generated AFM results

The final full-23 image experiment uses:

- M14i Rq/FSMI target predictions;
- the frozen M12a edge-preserving stochastic island/terrace renderer;
- no held AFM patch, AFM nearest neighbor or image retrieval at inference;
- four stochastic draws at 128 × 128 pixels over a 1 × 1 micrometer field.

The images preserve the useful M12a milestone: they contain non-flat island,
terrace, valley and boundary structure and visibly differ across predicted
conditions. Five atlas pages show all 23 growths in fixed ascending measured
Rq order. The failure atlas deliberately includes the four largest Rq errors.

The difficult high-Rq cases remain important limitations:

- 6099: measured Rq 10.32 nm, predicted 4.64 nm, confidence 6/100.
- 6095: measured Rq 9.87 nm, predicted 4.90 nm, confidence 22/100.
- 6063: measured Rq 5.87 nm, predicted 3.01 nm, confidence 65/100.
- 6090 is overpredicted (4.69 to 7.06 nm) and receives confidence 4/100.

Thus the confidence system shows useful self-awareness in aggregate and for
the most extreme case, but it does not catch every failure. The M12a texture
gate passes 14/23 growths (60.9%). Its generated AFM-support likeness remains
shifted relative to measured AFM, especially at the highest roughness. The
images are more AFM-like than the prior smooth/cloud baselines but should not
be claimed as indistinguishable from measured topography.

## Figures and artifacts

Robust target-head figures:

`reports/rheed_to_afm_ood_robust/20260728_m14_ood_robust_multiview_v3_final/figures`

- `fig01_rheed_only_ood_audit`
- `fig02_exclusion_sensitivity`
- `fig03_method_ablation`
- `fig04_held_one_out_predictions`
- `fig05_confidence_risk_coverage`
- `fig06_training_sample_weights`

Full generated-image figures:

`reports/rheed_to_afm_ood_robust_generation/20260728_m14_target_specific_m12a_generator_v1/full23_loo/figures`

- `Fig1a`–`Fig1e`: all 23 RHEED/generated-AFM/measured-AFM comparisons
- `Fig2_full23_target_scatter`
- `Fig3_protocol_comparison`
- `Fig4_full23_rq_ordered`
- `Fig5_confidence_audit`
- `Fig6_renderer_roughness_strata`
- `Fig7_largest_error_cases`

Every figure is saved as publication-resolution PNG and vector-text PDF.
Machine-readable predictions, method tables, risk–coverage tables, fold
audits and manifests are stored alongside the figures.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_ood_robust.run \
  --config configs/rheed_to_afm_ood_robust_v3_final.json

PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_to_afm_ood_robust_generation.json \
  --mode smoke

PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_to_afm_ood_robust_generation.json \
  --mode full

PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.visualization \
  --config configs/rheed_to_afm_ood_robust_generation.json
```

The full M12a image run completed locally in approximately 137 seconds.
Apple MPS is available and CUDA is not. The next locally meaningful
experiment does not meet the project’s CUDA-handoff threshold.

## Claim boundaries and next experiment

What worked:

- target-blind density weighting;
- causal temporal R3D features;
- fixed multiview fusion;
- fold-local support/amplitude risk;
- complete, non-cherry-picked visualization;
- preservation of the genuine stochastic M12a generator.

What did not work:

- target-blind hard deletion of 2–4 RHEED OOD growths;
- residual-only self-paced weighting as the primary method;
- the fully nested per-fold candidate selector;
- complete recovery of the two highest-Rq growths.

For a defensible paper, freeze M14i now and acquire a prospective test cohort
that deliberately covers high-Rq, spotty-to-ring RHEED transitions and
replicate growth conditions. Record additional process variables such as
substrate temperature calibration, flux ratio, accumulated thickness and
growth interruption because 6099 appears RHEED-in-domain yet AFM-conditionally
atypical. After prospective confirmation, a compact conditional latent
diffusion or structure-conditioned residual diffusion model can be trained on
many growth-safe AFM crops; it should refine topology while the M14i head
continues to control global Rq/FSMI amplitude.
