# Sharp RHEED-to-AFM generation: research continuation report

Date: 2026-07-27

Branch: `codex/rheed-afm-sharp-generation-20260727`

Starting commit: `ee56f8e0e6fe4a4a8bddb2a1c805bc4cb7bdf7d1`

## Executive conclusion

The visibly blurred/flat AFM generation failure is solved.

The best development pipeline is a genuine, stochastic, non-retrieval
generator:

1. a hybrid RHEED encoder predicts physical Rq and eight normalized
   morphology descriptors;
2. a conditional spectral model predicts an AFM radial power spectrum and
   height distribution;
3. IAAFT synthesizes a new random AFM field;
4. a differentiable descriptor calibration step makes the field agree with
   the RHEED-predicted condition without seeing a measured AFM target;
5. an optional circular adversarial residual generator refines texture.

The previous CVAE had a median sharpness ratio of **0.44** relative to
measured AFM and passed the AFM texture gate for **0/3** validation groups.
The selected calibrated spectral generator reaches **1.28** and **2/3** on
the separate validation cohort; its optional adversarial refiner reaches
**1.23** and **3/3**. Across 15 leave-one-growth-group-out development folds,
the selected generator reaches **1.17**, passes the texture gate for
**14/15** groups, and has maximum training-image SSIM only **0.037**. The
images are therefore sharp AFM-like stochastic fields, not retrieved or
slightly perturbed training AFMs.

The RHEED-conditioning conclusion is more limited. In 15-group cross-fitting,
the RHEED-conditioned generator improves median descriptor error from
**0.849 z** (mean condition) to **0.826 z** and Rq error from **1.39 nm** to
**1.10 nm**. On the separate three-group validation cohort, however, the mean
condition is slightly better (0.631 versus 0.659 z), and the 15-group cyclic
condition control wins only 53% of comparisons. Thus:

> AFM-like non-retrieval generation is demonstrated; a small RHEED-conditioned
> group-out benefit is present, but strong conditional generalization is not
> yet established.

The previously consumed five-group test partition was not reused. These are
development/cross-validation results, not a new unseen-test claim.

## Data integrity and exclusions

- Canonical data after filtering: 116 AFM scans from 23 growth groups.
- Fixed development split: 15 train groups / 68 scans.
- Separate validation split: 3 groups / 24 scans.
- Sealed, previously consumed test split: 5 groups / 24 scans; not reused.
- Growth group is the leakage boundary.
- Train/validation/test group intersections are empty.
- `removelist.txt` SHA-256:
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
- All 11 listed IDs are filtered from AFM rows, fold tables, RHEED physics,
  phase-1 manifests, and embedding payloads before fitting or evaluation.
- Post-filter overlap is zero.
- No raw RHEED or AFM file was modified.

## Literature-informed design

The broad review from the first research phase remains at
[`rheed_to_afm_generation/literature_review.md`](rheed_to_afm_generation/literature_review.md).
It covers conditional GANs, CVAEs, VQ-VAE/VQGAN, pixel and latent diffusion,
small-data learning, microscopy/microstructure generation, RHEED/MBE physics,
AFM statistics, and generative evaluation.

The mandatory Na, Yoo, and Ki paper,
[*Prediction of surface morphology and reflection spectrum of laser-induced
periodic surface structures using deep learning*](https://doi.org/10.1016/j.jmapro.2022.11.004),
was reread in full and visually reviewed. Its conditional BigGAN-deep used
26/3/3 process conditions, 200 random crops per condition, projection
conditioning, conditional normalization, spectral normalization, hinge loss,
DiffAugment, a 1:4 generator/discriminator learning-rate ratio, and
FFT-domain early stopping. The present adversarial refiner adopts the
small-data-compatible parts of that design, while retaining growth groups as
the statistical unit. Crops or repeated AFM scans are never treated as new
independent conditions.

Additional targeted references support the design:

- [DiffAugment](https://arxiv.org/abs/2006.10738) for limited-data
  adversarial training;
- [Projected GANs](https://arxiv.org/abs/2111.01007) for data-efficient
  discriminator features;
- [StyleGAN2-ADA](https://arxiv.org/abs/2006.06676) and a
  [medical small-data assessment](https://arxiv.org/abs/2210.03786);
- [VQ-VAE](https://arxiv.org/abs/1711.00937), [DDPM](https://arxiv.org/abs/2006.11239),
  and [latent diffusion](https://arxiv.org/abs/2112.10752);
- [diffusion microstructure reconstruction](https://arxiv.org/abs/2211.10949)
  and [physics-informed microscopy diffusion](https://arxiv.org/abs/2306.02929).

The key scientific choice is to evaluate height statistics, PSD, gradients,
correlation scale, diversity, and condition controls rather than optimize
generic visual realism alone.

## Methods explored

### M1 — previous conditional VAE

Retained unchanged as the blur baseline. It is stochastic and non-retrieval,
but pixel/multiscale reconstruction averages uncertain morphology, produces
smooth interiors, and repeats a lower-border artifact.

### M2 — conditional spectral random field

For each training growth group, the model estimates a 24-bin radial log PSD
and 33 height quantiles. A regularized condition-to-spectrum regression
predicts these parameters; IAAFT then synthesizes a new field from random
phase. It does not select, copy, crop, or initialize from a measured AFM at
inference.

### M2b — descriptor-calibrated spectral generator

The generated random field is optimized for 50 steps against differentiable
Ra, PSD-band, PSD-slope, anisotropy-proxy, skewness, and kurtosis targets.
The only target is the RHEED-predicted condition. A small seed-content term
preserves stochastic diversity. This is the selected balanced method.

### M3/M3b — conditional adversarial residual refinement

A six-block circular-convolution residual generator refines the spectral seed
without any upsampling path. A projection discriminator uses spectral
normalization, hinge loss, differentiable flips/rotations/circular
translations/cutout, scientific descriptor loss, gradient-statistic loss,
feature matching, EMA, and FFT/gradient/boundary validation early stopping.
The best checkpoint is step 200; later steps improve gradients but degrade FFT
fidelity, validating the early-stop criterion.

### Hybrid RHEED condition model

Absolute amplitude and unit-Rq morphology are separated:

- physical Rq: key-frame DINO features + physics summaries + strongly
  regularized SVR;
- normalized morphology: selected 16-frame R3D features + physics summaries
  + one-component PLS.

This improves cross-fitted Rq MAE, but predictions still shrink strongly
toward the training mean. Raw Rq rank correlation across LOO folds remains
negative (−0.39); this failure is recorded in the manifest and is not used as
a success claim.

## Quantitative results

### Separate validation cohort: 3 growth groups

| Method | Rq MAE nm ↓ | Descriptor MAE z ↓ | PSD ↓ | Sharpness / real | Texture gate | Condition wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous CVAE | 1.519 | 2.702 | 0.686 | 0.441 | 0/3 | — |
| Mean-condition calibrated spectral | **1.120** | 0.631 | 0.870 | 1.238 | **3/3** | — |
| RHEED spectral | 1.205 | 1.270 | **0.542** | 1.348 | 2/3 | 1/3 |
| RHEED calibrated spectral (selected) | 1.205 | 0.659 | 0.860 | 1.284 | 2/3 | **3/3** |
| RHEED calibrated adversarial refiner | 1.205 | 0.628 | 1.031 | 1.231 | **3/3** | 2/3 |
| Mean-condition adversarial refiner | **1.120** | **0.546** | 1.113 | 1.196 | **3/3** | — |

The unconditional controls prevent an inflated interpretation: RHEED
conditioning does not beat the mean condition on this three-group cohort.

### Leave-one-training-growth-group-out: 15 growth groups

All scalers, PCA, RHEED regressors, AFM output scalers, and spectral regressors
are refit without the held-out group.

| Method | Rq MAE nm ↓ | Descriptor MAE z ↓ | PSD ↓ | Sharpness / real | Texture gate | Max train SSIM ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean-condition calibrated spectral | 1.392 | 0.849 | 0.936 | 1.222 | 14/15 | 0.036 |
| RHEED spectral | **1.098** | 1.479 | **0.350** | 1.236 | 13/15 | 0.037 |
| RHEED calibrated spectral (selected) | **1.098** | **0.826** | 0.957 | 1.174 | **14/15** | 0.037 |

Calibration trades exact radial PSD for better target descriptors and
realistic gradient magnitude. It does not collapse diversity: median
within-condition pairwise L1 is 1.118, versus 1.120 for the mean-condition
stochastic generator.

### Retrieval/memorization audit

- Inference never queries a training AFM bank.
- Exact training pixel equality: zero.
- Validation maximum training SSIM: 0.035 for the adversarial variant.
- Cross-fitted maximum training SSIM: 0.037 for the selected method.
- Cross-fitted nearest-training L1: 1.062.

These values are incompatible with nearest-neighbour retrieval or trivial
copying.

## Qualitative review and failures

Positive:

- all fixed-seed draws contain clear, multiscale AFM-like height texture;
- no flat interiors or systematic top/bottom decoder bands;
- circular generation removes edge-energy artifacts;
- ensembles remain diverse across eight uncurated draws;
- 6056 and 6080 reproduce plausible granular/island morphology scale.

Failures:

- group 6022 has large connected islands that remain under-resolved;
- descriptor calibration increases PSD log distance;
- predicted physical Rq remains strongly mean-shrunk;
- the 15-group cyclic condition win rate is 53%, near chance;
- RHEED-conditioned generation beats the mean baseline slightly in
  cross-fitting but not on the three-group validation cohort;
- 15 development groups are too few to establish a robust conditional
  generative law.

## Figures and artifacts

Publication-resolution PNG and PDF figures are in
[`20260727_sharp_v4_hybrid/development/figures`](rheed_to_afm_sharp_generation/20260727_sharp_v4_hybrid/development/figures):

1. `Fig1_real_cvae_spectral_refiner` — fixed medoid comparison;
2. `Fig2_texture_metric_summary` — baseline/final metric panels;
3. `Fig3_spatial_frequency_comparison` — AFM and log FFT;
4. `Fig4_condition_permutation_control` — negative control;
5. `Fig5_adversarial_training_curves` — training and validation curves;
6. `Fig6_all_generated_draws` — all eight fixed draws, no cherry-picking;
7. `Fig8_rheed_generated_measured_afm` — RHEED input, generated AFM, measured AFM;
8. `Fig9_automatic_failure_case` — automatically selected failure and PSD.

The 15-group cross-validation figure is in
[`training_group_cross_validation/figures`](rheed_to_afm_sharp_generation/20260727_sharp_v4_hybrid/development/training_group_cross_validation/figures).

Machine-readable outputs:

- [`experiment_registry.csv`](rheed_to_afm_sharp_generation/experiment_registry.csv);
- [`best_model_manifest.json`](rheed_to_afm_sharp_generation/best_model_manifest.json);
- [`development_manifest.json`](rheed_to_afm_sharp_generation/20260727_sharp_v4_hybrid/development/development_manifest.json);
- [`method_summary.csv`](rheed_to_afm_sharp_generation/20260727_sharp_v4_hybrid/development/validation_evaluation/method_summary.csv);
- [`cross_validation_manifest.json`](rheed_to_afm_sharp_generation/20260727_sharp_v4_hybrid/development/training_group_cross_validation/cross_validation_manifest.json);
- [`reproducibility.md`](rheed_to_afm_sharp_generation/reproducibility.md).

## Compute, limitations, and next experiment

Full adversarial training took 395 seconds on Apple MPS; all downstream
generation/cross-validation took under one minute. The local machine was
sufficient, and a CUDA handoff is not recommended for the current bottleneck.

The next important step is new independent growth conditions, especially
high-Rq and large-island regimes. With a larger group count, freeze this model
and evaluate prospectively. Only after condition signal is confirmed should a
larger latent-diffusion refiner be justified. More GPU compute cannot repair
the present identifiability limit.

## Final verification

- The 11 focused removal-list and generation tests pass.
- Python compilation passes for both AFM-generation packages.
- The broader active suite reports 316 passing and 23 failing checks. The
  failures require absent historical `rheed_peak_saddle` checkpoint artifacts
  or a parquet engine; none exercises the new sharp-generation code. Running
  pytest from the repository root without limiting collection also encounters
  duplicate test module names inside the immutable paper-freeze snapshot.
- All final PDFs open as one-page vector figures; the RHEED/generated/measured
  panel was independently rendered from PDF and visually checked.
- No data file changed after the branch start, `git diff -- data` is empty,
  and the canonical removal-list hash is unchanged.

No action is waiting for approval. A remote push or PR would require explicit
approval.
