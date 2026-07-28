# RHEED-to-AFM distinct morphology generation with calibrated confidence

Date: 2026-07-27
Branch: `codex/rheed-afm-distinct-confidence-20260727`

## Outcome

The selected development method, **M5**, is a genuine stochastic
RHEED-conditioned AFM generator. It does not retrieve, copy, initialize from,
or blend with a measured AFM exemplar at inference.

M5 combines:

1. a RHEED-to-morphology predictor for Rq, PSD, correlation length,
   anisotropy and height-distribution moments;
2. strictly nested variance calibration that counteracts regression to the
   mean;
3. a condition-sensitive multiscale Matérn random field;
4. the learned M2b spectral random-field prior from the preserved previous
   milestone;
5. 90% growth-group CV+/Jackknife+ intervals and a conservative confidence
   index.

The main visual failure reported by the user is materially improved. For
validation groups 6022/6056/6080, pairwise generated morphology-descriptor
distances changed from **0.062/0.464/0.451 z** for M2b to
**0.221/1.418/1.285 z** for M5. The previously collapsed 6022–6056 pair is
3.6 times farther apart, and the median pairwise separation is 2.8 times
larger. Figure 2 uses one shared physical nanometre scale and shows that 6080
now produces a visibly coarser field rather than the same fine texture.

This is a strong development result, not a new final-test claim. The
historical test partition has already been consumed by earlier work and was
not passed to fitting, selection, generation or evaluation here. A prospective
growth cohort is required for a new final claim.

## Scientific method

### RHEED condition

The fixed predictor preserved from the prior valid training-group comparison
uses:

- the selected 16-frame R3D-18 temporal embedding for unit-shape descriptors;
- a key-frame DINOv2 embedding for Rq;
- target-independent spot/streak/connection/diffuse/brightness summaries;
- one-component PLS for morphology and a small SVR head for Rq.

Only the 15 training growth groups fit scalers and models. All leave-one-group
predictions refit the entire condition path without the held growth.

### Nested variance calibration

The earlier condition path shrank toward the training mean. For every outer
fold, inner leave-one-growth-group predictions estimate

`factor = SD(true descriptor) / SD(inner-LOO prediction)`,

clipped to `[1, cap]`. The cap is selected from the strict 15-group
cross-fitted Pareto curve, never from a test group. Cap 2.0 nearly doubles
pairwise condition sensitivity from 0.485 to 0.909 z while reducing median Rq
MAE from 1.119 to 0.739 nm at the descriptor-prediction stage. The trade-off is
an increase in mean descriptor MAE from 0.825 to 0.985 z.

### M4 and M5 generation

M4 maps predicted descriptors directly to a stochastic multiscale Matérn
field:

- Rq sets physical amplitude;
- autocorrelation length sets the coarse spatial scale;
- PSD slope and predicted high/mid-band power set the spectrum;
- anisotropy sets a sampled-orientation axis ratio;
- a rank-preserving parametric height transform matches Ra, skewness and
  kurtosis;
- random phase produces multiple non-identical outputs.

M5 blends 65% of this condition-sensitive field with 35% of the learned M2b
spectral random-field prior and projects the result back to unit Rq. Both
inputs are generated. This restores plausible AFM multiscale texture without
giving up the large-scale response to the RHEED condition.

The design is informed by the mandatory Na–Yoo–Ki paper and targeted work on
continuous conditioning, contrastive condition consistency, conditional
collapse and conformal uncertainty. The detailed, linked review is
[literature_update.md](rheed_to_afm_distinct_confidence/literature_update.md).

## Quantitative results

### Strict leave-one-growth-group-out, 15 training groups

| Metric | Prior M2b | Selected M5 | Interpretation |
| --- | ---: | ---: | --- |
| AFM texture gate | 14/15 (93.3%) | 13/15 (86.7%) | Small texture-gate trade-off |
| Median Rq absolute error | 1.098 nm | **0.829 nm** | 24.5% lower |
| Median PSD log distance | 0.957 | **0.925** | 3.3% lower |
| Median sharpness ratio | 1.174 | **0.939** | M5 is closer to the target 1.0 |
| Median morphology composite | 9.572 | **8.545** | 10.7% lower |
| Median generated descriptor MAE | **0.826 z** | 0.986 z | M5 sacrifices descriptor calibration |
| Median maximum training SSIM | 0.0367 | 0.0372 | Both are far from copying a training AFM |

M5 is selected as the balanced method because it improves Rq, PSD, visual
condition separation and the composite while retaining 13/15 texture passes.
It is not claimed to dominate every metric.

### Pre-existing three-group validation

| Metric | Prior M2b | Selected M5 |
| --- | ---: | ---: |
| AFM texture gate | 2/3 | **3/3** |
| Median Rq absolute error | 1.205 nm | **0.833 nm** |
| Median PSD log distance | 0.860 | **0.666** |
| Median sharpness ratio | 1.284 | **1.065** |
| Median morphology composite | 9.604 | **8.513** |
| Median generated descriptor MAE | **0.659 z** | 0.971 z |

The validation result is supportive but not independent confirmation: these
groups were available during the broader research programme. The strict
training-group cross-validation is the primary model-development evidence.

## Confidence and “self-awareness”

Confidence is not presented as a probability of correctness.

For each query, CV+/Jackknife+ uses one model per calibration growth group.
That model excludes its corresponding calibration group, so its residual is
out-of-group. Across the strict training audit:

- nominal component coverage: 90%;
- empirical descriptor-component coverage: **93.3%**;
- empirical Rq interval coverage: **93.3%**;
- interval width versus realized descriptor error:
  Spearman **ρ = 0.536**;
- confidence versus realized error:
  Spearman **ρ = -0.536**.

The current intervals span roughly five standardized descriptor units, so all
absolute confidence scores are intentionally low; the maximum is only
**20/100**. This is more credible than stretching tiny width differences to
0–100. In Figure 7, held group 6072 is an obvious failure with error 2.12 z
and confidence 14/100, while 6029 has error 0.39 z and confidence 20/100.

There is also an important failure that is not hidden: group 6095 has measured
Rq 9.87 nm, predicted Rq 2.82 nm, and falls above its 90% interval
[1.32, 6.52] nm. Its score (18.9/100) is low in absolute terms but is not
ranked as the least confident case. Thus uncertainty is error-aware on
average, not infallible per sample.

### Why more data should help

The repeated strict held-group learning curve shows:

| Training growth groups | Median descriptor MAE |
| ---: | ---: |
| 5 | 1.013 z |
| 8 | 0.824 z |
| 11 | 0.765 z |
| 14 | **0.638 z** |

The 37% reduction from 5 to 14 groups is direct evidence—not a promise—that
additional independent growth conditions expand model capability. More scans
of the same growth help the AFM prior, but only more independent growths
improve the cross-modal condition map and its calibration.

## Methods tried and failure analysis

The complete registry is
[experiment_registry.csv](rheed_to_afm_distinct_confidence/experiment_registry.csv).

- **M4 pre-v1:** too much high-frequency mixture; 0/15 texture gates and
  approximately two orders too much high-band PSD. Rejected.
- **M4 v1:** sharp and condition-sensitive, but anisotropy was effectively
  squared, producing long nonphysical streaks. Preserved as an ablation.
- **M4 v2:** square-root axis parameterization reduced streaks, but only 8/15
  cross-fitted texture gates passed.
- **M5 v3/v4:** blending the M4 field with the learned spectral prior restored
  multiscale AFM texture (13/15 crossfit, 3/3 validation) and is selected.
- **297 handcrafted temporal RHEED geometry/quality features:** apparent
  full-data correlations became unstable or reversed when feature selection
  was repeated inside every outer fold. Rejected as a target-leakage-prone
  apparent win.
- **KRR/GPR/SVR/PLS/PCA nonlinear candidates:** no candidate produced robust
  Rq ranking and lower descriptor error at N=15. Several apparent sensitivity
  wins were near-zero-variance numerical artifacts. Rejected.

## Figures

All figures exist as high-resolution PNG and vector PDF under
[`figures/`](rheed_to_afm_distinct_confidence/20260727_m5_hybrid_v4_confidence/development/figures).

- Fig. 1a–c: all 18 strict-LOO/validation predictions sorted by measured Rq,
  with RHEED ROI, generated/real physical AFM, Rq interval and confidence.
- Fig. 2: the three validation RHEED inputs, prior M2b, M5 and measured AFM on
  one shared nanometre scale.
- Fig. 3: baseline-versus-final strict crossfit metrics.
- Fig. 4: confidence/error audit, component coverage and validation Rq
  intervals.
- Fig. 5: variance-cap ablation and data-scaling learning curve.
- Fig. 6: held-group descriptor correlations.
- Fig. 7: automatically selected confidence-aware successes and failures.

No result panel is manually cherry-picked. Atlas order is fixed by measured
Rq; failure panels use precomputed error/confidence ranks.

## Reproducibility and safety

- Config:
  [`configs/rheed_to_afm_distinct_confidence.json`](../configs/rheed_to_afm_distinct_confidence.json)
- Source package:
  [`analysis/rheed_to_afm_distinct_confidence/`](../analysis/rheed_to_afm_distinct_confidence)
- Tests:
  [`tests/test_rheed_to_afm_distinct_confidence.py`](../tests/test_rheed_to_afm_distinct_confidence.py)
- Best manifest:
  [`best_model_manifest.json`](rheed_to_afm_distinct_confidence/20260727_m5_hybrid_v4_confidence/development/best_model_manifest.json)
- Command history:
  [`run_history.md`](rheed_to_afm_distinct_confidence/run_history.md)

The removelist hash is pinned, all 11 listed samples are removed before
features/splits/fitting/evaluation, and post-filter overlap is zero. Growth
groups are the leakage boundary. The historical test partition is excluded
from the current modelling/evaluation path. Generated maps and checkpoints are
additive derived outputs. Raw RHEED and AFM files were not modified.

## Limitations and next experiment

1. Fifteen training growth groups remain too few for high confidence.
2. 6022 and 6056 are more separated than before but still share related
   normalized texture; the main visible difference is physical amplitude and
   moderate morphology statistics.
3. 6080 is clearly different but its predicted coarse morphology does not
   reproduce the measured patchy topology; its confidence remains low in
   absolute terms rather than uniquely lowest.
4. Radial PSD and second-order correlation do not determine island topology.
5. Pixel SSIM is expected to be low for stochastic, spatially unregistered AFM
   realizations and is not evidence of exact reconstruction.
6. The next decisive experiment is prospective: collect independent growth
   groups spanning the smooth, intermediate and very rough regimes, freeze M5
   and its intervals, then evaluate once.

Local compute is sufficient. The complete M5 development experiment takes
about 30 seconds; a CUDA handoff is not recommended. Larger conditional latent
diffusion should wait for materially more independent growth conditions,
because the present bottleneck is identifiability rather than GPU throughput.
