# RHEED-conditioned functional AFM morphology generation

**Final local milestone:** M12 range-calibrated, edge-preserving terrace
generator

**Date:** 2026-07-27

**Branch:** `codex/rheed-afm-morphology-index-20260727`

**Development policy:** strict leave-one-growth-out (LOO) on 15 training
growths plus the three pre-existing validation growths. The 24 historical test
rows remain closed and unused.

## Executive result

M12 addresses the two reported failures without converting the task into
retrieval:

1. a strictly nested log-range calibration restores much more of the Rq
   dynamic range;
2. a stochastic capture-zone/terrace refiner generates discrete islands,
   coalescence grooves and shoulders rather than a smooth cloud;
3. a new experimental Functional Surface Morphology Index (FSMI) measures
   height, 31.25 nm texture, curvature, bearing-height span and island
   prominence;
4. every held-growth output carries a confidence index and conformal-style
   Rq/FSMI intervals.

The selected M12a model is a genuine conditional generator. At inference it
uses RHEED features, random seeds and learned population statistics; it never
receives a measured AFM patch and never returns a nearest-neighbour AFM.

The result is materially better but not perfect. The most extreme growth 6095
has measured `Rq=9.87 nm` and is still underpredicted at `5.79 nm`. The model
assigns it `18.75/100`, the second-lowest confidence in the 15-growth LOO
cohort, and the failure is included in the main failure figure.

## Data integrity and split policy

- Training development: 15 independent growth groups.
- Pre-existing validation: 3 independent growth groups.
- Historical test: 24 rows present in the source descriptor table but never
  selected for extraction, fitting, rendering, selection or evaluation.
- Canonical exclusions: all 11 IDs in `removelist.txt`; retained overlap is
  zero.
- Leakage unit: growth run, not AFM scan or RHEED frame.
- Raw data: read only; no file under `data/` was modified.
- Inference flags written into every generated NPZ:
  `retrieval_at_inference=False` and
  `measured_afm_patch_used_at_inference=False`.

## Method

### RHEED amplitude head

The amplitude head uses 15 fixed morphology/temporal RHEED summaries plus one
small dynamic-nucleation head. The features describe spot roundness, component
area/count/eccentricity, streak connectivity, skeleton endpoints/branches,
anisotropy, diffuse-to-peak intensity and temporal change. A robust-scaled
ridge model predicts positive targets in log space.

These features were fixed before the reported M12 run, but their shortlist was
informed by the current development cohort during the preceding audit. The
correlations are therefore development evidence, not a preregistered
prospective feature test; the exact list should be frozen before collecting
the next independent cohort.

For each outer held growth:

1. fit the RHEED predictor on the other 14 growths;
2. obtain inner LOO predictions among those 14;
3. match log-space center/spread of the honest inner predictions to their
   truths, cap the spread correction at 1.20 and blend it 75% with the raw
   output;
4. build error intervals and expected-error diagnostics without using the
   outer target.

This nested calibration improves range without fitting a global correction to
the displayed held targets.

### AFM generator

The image model is staged:

`RHEED → amplitude/FSMI + morphology condition → island statistics → random
Laguerre capture zones → continuous terraces/grooves → AFM spectral texture`.

M12a uses random capture-zone centers, sizes, cell heights and a second
nucleation population. A continuous plateau transform creates terrace
interiors; signed-distance relief creates island shoulders; a low-weight
AFM-trained spectral field supplies population-level texture. The output is a
novel stochastic height map in nanometres. It is not a copied patch, a warped
training AFM or a retrieval result.

### Experimental FSMI

`FSMI = RMS(Sq, Δh31, C31, 0.25·(z90-z10), P70)`.

- `Sq`: areal RMS height;
- `Δh31`: RMS height increment over 31.25 nm;
- `C31`: half RMS second difference over 31.25 nm;
- `z90-z10`: bearing/material-ratio core-height span;
- `P70`: median q70-island prominence.

Each term is in nanometres. No cohort-fitted normalization or learned weights
are used. FSMI is an experimental descriptor, **not** an ISO/SEMI standard and
not yet a validated material-performance index. The standards/literature
basis and limitations are in
[`rheed_to_afm_functional_morphology_literature_review.md`](rheed_to_afm_functional_morphology_literature_review.md).

## Quantitative results

### Direct held-growth target prediction

| Target | Mean MAE (nm) | Median MAE (nm) | RMSE (nm) | Pearson r | Spearman rho | Predicted range (nm) | True range (nm) | Pred./true SD | 90% interval coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rq | 0.664 | 0.281 | 1.194 | 0.836 | 0.932 | 1.50–7.03 | 1.47–9.87 | 0.796 | 13/15 |
| FSMI | 0.722 | 0.417 | 1.189 | 0.765 | 0.832 | 1.26–4.92 | 1.43–8.57 | 0.686 | 14/15 |

For comparison, the uncalibrated M11 Rq predictor had predicted/true standard
deviation ratio 0.652, mean MAE 0.682 nm and RMSE 1.322 nm. Nested range
calibration raises the spread ratio to 0.796 while reducing mean MAE to
0.664 nm and RMSE to 1.194 nm.

### Baseline versus selected renderer

| Split | Method | Median Rq error (nm) | Median FSMI error (nm) | q70 area log error | Boundary contrast | Composite | Island MAE (z) | AFM-prior distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict 15-growth LOO | M10 baseline | 0.829 | 0.684 | 0.811 | 1.324 | 8.106 | 1.514 | 6.438 |
| strict 15-growth LOO | **M12a** | **0.316** | **0.438** | **0.574** | **1.584** | **7.963** | 2.166 | 10.551 |
| pre-existing validation | M10 baseline | 0.833 | 0.659 | 0.765 | 1.309 | 7.910 | 1.423 | 5.951 |
| pre-existing validation | **M12a** | **0.477** | **0.329** | **0.664** | **1.595** | **7.792** | 1.567 | 10.183 |

On strict LOO, M12a reduces median Rq error by 61.9%, FSMI error by 35.9%
and q70 island-area error by 29.3%, while raising visible boundary contrast by
19.6%. On the pre-existing validation groups, it reduces median Rq error by
42.8%, FSMI error by 50.1% and composite error by 1.5%; all 3/3 validation
growths pass the texture gate.

The adverse results are equally important: aggregate island-feature MAE and
AFM-prior Mahalanobis distance are worse for M12a, and strict-LOO texture-gate
coverage is 12/15 versus 13/15 for M10. These diagnostics say that explicit
terraces improve the requested visual topology and functional/amplitude
targets but over-sharpen some features relative to the small training
population. M11b remains the conservative alternative: validation island MAE
is 1.238 z, but its contours are visibly softer.

## Confidence and failure awareness

The confidence value is a rank index, not a probability. It is the smoothed
survival percentile of an equal-weight combination of strictly cross-fitted
expected FSMI error and expected island-topology error. Rq is not added again
because FSMI already contains amplitude.

- confidence versus realized joint-error rank:
  `Spearman rho=-0.554`, `p=0.0320`;
- Rq 90% interval coverage: `13/15=0.867`;
- FSMI 90% interval coverage: `14/15=0.933`;
- island-error 90% upper-bound coverage: `13/15=0.867`;
- lowest-confidence cases include 6048 (`12.5/100`) and the extreme
  underprediction 6095 (`18.75/100`);
- in the three pre-existing validation groups, confidence decreases
  56.25 → 43.75 → 37.5 as realized joint error increases.

With only 15 LOO points this is promising calibration evidence, not proof of a
universally calibrated probability. The paper should use “confidence index”
or “model support” rather than “probability of correctness.”

## Ablations and negative results

- M11a: strongest conservative PSD/AFM-prior behavior, but soft edges.
- M11b: soft SDF contours and best validation island MAE; visually less
  terrace-like.
- M11c: stronger SDF edges, rejected because AFM-prior distance increased.
- M12b: clearest terraces, rejected as over-sharpened.
- M12c: recovered part of the island/AFM-prior tradeoff, rejected because its
  confidence-error relation was not statistically significant
  (`rho=-0.425`, `p=0.114`).
- Earlier full-noise and weak image-space DDPM refiners are preserved in the
  M7/M8 milestone. Full-noise diffusion added granular artifacts, and weak
  diffusion did not preserve strict-LOO topology improvements.

There are no epoch curves for M12 because its learned heads are closed-form
regularized regressions and its renderer is procedural. The target scatter,
group-held-out intervals and method ablation replace an inapplicable
train-loss curve.

## Figures

All figures are saved as 360-dpi PNG and vector-text PDF under
[`development/figures`](rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/development/figures):

- Fig. 1: all three pre-existing validation RHEED/generated/measured panels;
- Fig. 2a–c: all 15 strict held-one predictions, fixed order by measured Rq;
- Fig. 3: Rq and FSMI predicted-versus-measured scatter with intervals and
  confidence;
- Fig. 4: dynamic-range recovery versus M10;
- Fig. 5: confidence versus realized error and expected/realized FSMI errors;
- Fig. 6: all renderer ablations and boundary contrast;
- Fig. 7: six component-level surface correlations;
- Fig. 8: lowest-confidence failures, including 6095.

Ten PNG and ten PDF files were generated. Fig. 1, Fig. 3 and Fig. 8 PDFs were
rendered with Poppler and visually inspected.

## Reproducibility

Primary configuration:
[`rheed_to_afm_functional_morphology_m12.json`](../configs/rheed_to_afm_functional_morphology_m12.json).

```bash
.venv/bin/python -m analysis.rheed_to_afm_functional_morphology.run \
  --config configs/rheed_to_afm_functional_morphology_m12.json

.venv/bin/python -m analysis.rheed_to_afm_functional_morphology.visualization \
  --config configs/rheed_to_afm_functional_morphology_m12.json

PYTHONPATH=. .venv/bin/pytest -q tests/test_rheed_to_afm_*.py
```

The final manifest is
[`best_model_manifest.json`](rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/development/best_model_manifest.json);
the complete ablation registry is
[`rheed_to_afm_functional_morphology_experiment_registry.csv`](rheed_to_afm_functional_morphology_experiment_registry.csv).

Final verification: 23/23 RHEED-to-AFM tests pass. The M12 smoke run takes
about 30 seconds; the complete local run takes about 75 seconds. PyTorch
2.12.0 reports MPS built/available and CUDA unavailable. Local compute is
sufficient, so the defined CUDA-handoff condition is not met.

## Claim boundary and next experiment

The strongest defensible claim is:

> On strict development-growth LOO and a pre-existing three-growth validation
> cohort, a RHEED-conditioned stochastic terrace generator produces visibly
> object-like AFM topography, substantially improves Rq/FSMI amplitude
> prediction over the frozen M10 baseline, and emits a confidence index that
> is significantly related to held-growth error.

Do not claim pixelwise reconstruction, exact island identity or prospective
generalization. The 6095 extreme shows that the training range is still
insufficient. The highest-value next step is to add independent growths that
populate the high-Rq/high-FSMI regime and record a material-property endpoint
(for example optical scatter, mobility or interface-related performance).
Only then should FSMI weights be related to material function. With roughly
50–100 independent growths, a conditional latent diffusion or
structure-conditioned diffusion refiner can be compared prospectively against
M12.
