# RHEED-to-AFM conditional generation: final research report

Date: 2026-07-27

Branch: `codex/rheed-afm-generative-20260727`

Starting commit: `aa00163f3cd560c0d0561ea979590f2de0f62551`

Compute: Apple M1 Pro, 32 GiB unified memory, PyTorch 2.12 MPS

## Executive conclusion

This work delivers a reproducible **true generative** RHEED-to-AFM pipeline,
but the held-out evidence does **not** support the stronger claim that the
current model reliably learned the RHEED-to-AFM relationship.

The selected model is a two-stage RHEED descriptor predictor plus a
conditional variational autoencoder (CVAE). At inference it receives only
RHEED-derived features, samples a learned conditional Gaussian prior, and
decodes new 128 × 128 AFM-like morphology maps. It never queries an AFM bank,
copies a source AFM, or performs nearest-neighbour retrieval.

On the one-time held-out test of five growth groups / 24 AFM scans:

- exact generated/training identities: **0**;
- generated/real diversity ratio: **0.676**;
- morphology composite: **6.248**, versus **8.508** for nearest-RHEED
  retrieval and **7.518** for the unconditional train mean;
- Rq absolute error: **0.822 nm**, versus **5.775 nm** for retrieval;
- PSD log distance: **0.396**, versus **0.424** for retrieval;
- but descriptor MAE was worse (**2.713 z**) and SSIM was lower (**0.0067**);
- the decisive condition-permutation negative control passed for only
  **1/5 test groups**, with median error increase **−0.011 z**;
- predicted-versus-measured Rq Spearman correlation was **−0.40**.

Visual audit agrees with the negative control: generated maps are diverse and
not retrieved, but they are too smooth, under-represent sharp island
morphology, and contain a repeated lower-edge decoder artifact. Therefore:

> The pipeline has moved beyond retrieval in implementation, but strong,
> scientifically meaningful RHEED-conditioned AFM generation has not yet been
> demonstrated on unseen growth groups.

This is the best locally defensible result. It is a useful research checkpoint
and a rigorous negative result, not a deployment-ready model or a claim of
solved conditional generation.

## Repository and baseline audit

The repository already contained:

- manually selected RHEED key frames and temporal windows;
- 116 one-micrometre AFM scans from 23 growth groups;
- AFM descriptors, plane-corrected unit-Rq maps, DINOv2/R3D-18 embeddings,
  and explicit RHEED physics summaries;
- AFM autoencoders, VQ models, latent diffusion, temporal encoders, retrieval,
  quilting, texture, and spectral-synthesis methods;
- strict group-aware outer folds and extensive reports.

The prior strict fixed-method benchmark selected nearest-neighbour retrieval:
retrieval A3 had median visual composite 0.211, while VQ F1 had 0.343 and
diffusion G4 had 0.355 (lower is better). See the existing
[`fixed_method_family_summary.md`](rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7b_fixed_method_atlases/fixed_method_family_summary.md).
Those metrics use a different benchmark definition and are not numerically
mixed with the new fixed-split results. They establish the historical
bottleneck: prior nominal generators did not beat retrieval.

No existing baseline was removed or overwritten.

## Data partition and leakage control

The split was fixed from the pre-existing group folds before model selection:

| Partition | Growth groups | AFM scans | Group IDs |
| --- | ---: | ---: | --- |
| Train | 15 | 68 | 6028, 6029, 6047, 6048, 6057, 6062, 6063, 6070, 6072, 6078, 6082, 6084, 6085, 6090, 6095 |
| Validation | 3 | 24 | 6022, 6056, 6080 |
| Test | 5 | 24 | 6033, 6081, 6094, 6099, 6101 |

Controls:

- growth group is the leakage boundary;
- train, validation, and test group intersections are empty;
- PCA, feature scalers, descriptor scalers, and ridge fits use training groups;
- temporal representation, ridge regularization, CVAE epoch, and CVAE variant
  are selected on training/validation evidence only;
- the selected checkpoint, predictor, config, and split hashes were frozen;
- test evaluation refuses to run without the frozen manifest and refuses to
  overwrite an existing test manifest;
- the test partition was evaluated exactly once;
- group-level medians and 2,000 growth-group bootstrap resamples are reported;
- representative AFMs are descriptor-space medoids, not hand-picked examples.

The detailed audit is
[`split_integrity_audit.json`](rheed_to_afm_generation/20260727_cvae_film_tradeoff/split_integrity_audit.json).

## Literature-informed method

The full review is in
[`literature_review.md`](rheed_to_afm_generation/literature_review.md) and
covers conditional GANs, CVAEs, VQ models, pixel and latent diffusion,
small-data augmentation, microscopy generation, microstructure statistics,
RHEED/MBE physics, multimodal RHEED–AFM work, and generative evaluation.

The mandatory Na et al. paper was read fully and visually inspected. Its most
important influence was to treat frequency/morphology statistics and
downstream physics as first-class evidence rather than relying on a
photorealism metric. Its conditional BigGAN strategy was not copied directly
because the effective dataset here has only 23 independent growth conditions.

### RHEED condition path

Three pre-existing temporal representations were compared on validation only:

| RHEED representation | Validation descriptor MAE (z) | Validation Rq MAE (nm) | Selection score |
| --- | ---: | ---: | ---: |
| DINOv2 centered 8-frame window | **0.929** | **1.232** | **0.797** |
| DINOv2 key frame | 0.935 | 1.943 | 0.859 |
| R3D-18 selected 16-frame window | 1.018 | 2.312 | 0.952 |

The centered eight-frame DINOv2 representation was selected. Eight
train-fitted PCA components, explaining 76.0% of training embedding variance,
were concatenated with five RHEED summaries: spot, streak, connection,
diffuse, and temporal brightness drift. A multi-output ridge model predicts:

1. log Rq;
2. unit-Rq Ra;
3. mid-frequency PSD fraction;
4. high-frequency PSD fraction;
5. PSD slope;
6. log autocorrelation length;
7. log anisotropy ratio;
8. height skewness;
9. height kurtosis.

Ridge alpha 100 was selected by leave-one-training-group-out error.

### AFM generator

The CVAE has:

- a convolutional AFM encoder for \(q(z \mid x,c)\);
- a learned diagonal Gaussian conditional prior \(p(z \mid c)\);
- 16 stochastic latent dimensions;
- a 128 × 128 decoder with condition concatenation and four FiLM stages;
- an exact output projection to zero mean and unit Rq;
- physical height obtained by multiplying by RHEED-predicted Rq, never the
  measured test Rq.

The AFM reconstruction objective combines pixel L1, gradients, multiscale
structure, radial PSD, and height quantiles. KL warm-up aligns the posterior
with the conditional prior. Growth-balanced sampling prevents groups with more
AFM scans from dominating. Circular translation augmentation preserves height
histograms and Fourier power under the approximately stationary morphology
assumption.

Two explicit anti-collapse terms were added after validation diagnostics:

- different prior draws under one condition must produce distinguishable
  morphology;
- the same prior mean decoded under different conditions must change.

The model is stochastic: eight fixed-seed, uncurated samples are generated per
held-out condition.

## Experiment loop and model selection

All runs used the same train/validation/test partition. Test data were sealed
until v5 was frozen.

| Variant | Key change | Validation composite ↓ | SSIM ↑ | PSD ↓ | Diversity ratio | Correct vs permuted | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | bottleneck condition | 6.508 | 0.0155 | 0.692 | 0.146 | 2/3 | diversity collapse |
| v2 | multi-scale FiLM | 6.388 | 0.0060 | 0.674 | 0.148 | 2/3 | diversity collapse |
| v3 | strong diversity hinge | **6.365** | **0.0346** | **0.573** | 0.599 | 1/3 | condition gate failed |
| v4 | strong condition + diversity | 7.306 | 0.0031 | 0.669 | 0.766 | 2/3 | passed gates, weak morphology |
| v5 | intermediate condition trade-off | 6.500 | 0.0112 | 0.686 | **0.875** | **3/3** | selected |

The predeclared final selection rule required:

1. zero exact training-image identities;
2. diversity ratio at least 0.5;
3. correct condition better than a cyclicly permuted condition for at least
   2/3 validation groups;
4. lowest morphology composite among passing variants.

v5 is the lowest-composite passing variant. v3 was not selected despite its
better morphology score because a generator that ignores its condition does
not answer the scientific question.

The full machine-readable registry is
[`experiment_registry.csv`](rheed_to_afm_generation/experiment_registry.csv).

## One-time held-out test results

Group-level medians are shown below. Brackets are 95% group-bootstrap
intervals; with five groups these intervals are necessarily wide and should
not be read as strong significance evidence.

| Method | Rq error (nm) ↓ | PSD log distance ↓ | Descriptor MAE (z) ↓ | SSIM ↑ | Diversity ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unconditional train mean | 1.000 [0.089, 7.229] | 0.847 [0.451, 1.060] | **1.471** [0.265, 2.170] | **0.038** [−0.005, 0.190] | 0.000 |
| Nearest-RHEED retrieval | 5.775 [0.056, 8.450] | 0.424 [0.167, 1.067] | 1.708 [0.846, 2.888] | 0.018 [0.009, 0.048] | 0.000 |
| RHEED-conditioned CVAE | **0.822** [0.463, 6.763] | **0.396** [0.173, 1.014] | 2.713 [2.388, 3.847] | 0.007 [−0.083, 0.149] | **0.676** [0.469, 0.777] |

Additional medians:

| Metric | Train mean | Retrieval | CVAE |
| --- | ---: | ---: | ---: |
| Unit-Rq L1 ↓ | 0.998 | 1.158 | **0.958** |
| Correlation-length relative error ↓ | 3.667 | **0.250** | 0.400 |
| Height-quantile error ↓ | 0.290 | **0.254** | 0.491 |
| Physical-height Wasserstein (nm) ↓ | **0.301** | 0.429 | 1.126 |
| Morphology composite ↓ | 7.518 | 8.508 | **6.248** |
| Nearest-training L1 ↑ | 0.525 | 0.000 | **0.568** |
| Maximum training SSIM ↓ | 0.151 | 1.000 | **0.176** |
| Exact training identity count | 0 | 5 | **0** |

The morphology composite is the repository’s fixed weighted sum of PSD
distance (0.30), correlation-length relative error (0.25), gradient MAE
(0.20), height-quantile error (0.15), and SSIM penalty (0.10). It does not
include Rq. The CVAE composite advantage is descriptive, not statistically
decisive: all five-group bootstrap intervals overlap.

### Conditioning negative control

For each test group, the same latent-noise seeds were decoded with the correct
predicted condition and with the next test group’s condition. Generated
descriptor error should be lower for the correct condition.

| Target group | Wrong-condition source | Correct error (z) | Wrong error (z) | Wrong − correct |
| --- | --- | ---: | ---: | ---: |
| 6033 | 6081 | 2.713 | 2.704 | −0.008 |
| 6081 | 6094 | 2.517 | 2.572 | +0.055 |
| 6094 | 6099 | 2.388 | 2.371 | −0.017 |
| 6099 | 6101 | 3.847 | 3.836 | −0.011 |
| 6101 | 6033 | 3.136 | 2.922 | −0.214 |

Only group 6081 behaves as required. This failure overrides an optimistic
reading of the aggregate composite.

## Qualitative scientific review

Positive observations:

- samples are newly decoded and vary with latent draws;
- no sample is an exact or near-exact training AFM copy;
- physical Rq is often closer than retrieval;
- some coarse spectral/correlation scales are plausible;
- generated ensembles expose uncertainty instead of pretending that one
  RHEED condition determines a unique spatial realization.

Failure modes:

- sharp islands, trenches, and step/terrace boundaries are smoothed;
- a repeated dark semicircular feature appears at the lower decoder boundary;
- top/bottom edge bands recur across different groups and latent draws;
- group 6099’s high roughness is strongly underpredicted;
- generated descriptors change too little or in the wrong direction when
  conditions are permuted;
- the roughness predictor regresses toward roughly 3–4 nm and has negative
  rank correlation on the test cohort;
- SSIM is not meaningfully better than zero, although exact pixel alignment is
  not expected for stochastic, unregistered AFM morphology.

The fixed medoid panels and all ensemble draws are uncurated. No favourable
samples were selected after seeing test outcomes.

## Figures

Publication-resolution PNG and vector PDF versions are in
[`reports/rheed_to_afm_generation/20260727_cvae_film_tradeoff/test_figures`](rheed_to_afm_generation/20260727_cvae_film_tradeoff/test_figures):

1. `baseline_vs_final_metric_table` — group medians and bootstrap intervals;
2. `Fig2_baseline_vs_final_metrics` — baseline comparison;
3. `Fig3_rheed_generated_ground_truth` — RHEED window, generated AFM,
   measured AFM, and retrieval for every test group;
4. `Fig4_real_vs_generated_ensembles` — four uncurated draws per group;
5. `Fig5_failure_cases` — predefined largest composite errors;
6. `Fig6_training_validation_curves` — optimization and validation selection;
7. `Fig7_descriptor_correlations` — measured versus RHEED-predicted
   morphology descriptors;
8. `Fig8_temporal_window_ablation` — temporal input comparison;
9. `Fig9_condition_swap_control` — the decisive negative control.

All final PNGs were visually inspected. The table and descriptor figure were
revised after visual QA to remove clipping and scientific-notation overlap.

## Reproducibility and artifacts

Core code:

- [`analysis/rheed_to_afm_generation/model.py`](../analysis/rheed_to_afm_generation/model.py)
- [`analysis/rheed_to_afm_generation/data.py`](../analysis/rheed_to_afm_generation/data.py)
- [`analysis/rheed_to_afm_generation/training.py`](../analysis/rheed_to_afm_generation/training.py)
- [`analysis/rheed_to_afm_generation/evaluation.py`](../analysis/rheed_to_afm_generation/evaluation.py)
- [`analysis/rheed_to_afm_generation/visualization.py`](../analysis/rheed_to_afm_generation/visualization.py)
- [`analysis/rheed_to_afm_generation/run.py`](../analysis/rheed_to_afm_generation/run.py)

Frozen inputs and results:

- selected config:
  [`configs/rheed_to_afm_generation.json`](../configs/rheed_to_afm_generation.json);
- archived experiment configs:
  [`configs/experiments`](../configs/experiments);
- artifact manifest:
  [`artifact_manifest.json`](rheed_to_afm_generation/artifacts/artifact_manifest.json);
- frozen CVAE checkpoint:
  [`selected_conditional_vae.pt`](rheed_to_afm_generation/artifacts/selected_conditional_vae.pt);
- frozen RHEED predictor:
  [`selected_rheed_descriptor_predictor.joblib`](rheed_to_afm_generation/artifacts/selected_rheed_descriptor_predictor.joblib);
- final metric CSVs:
  [`final_metrics`](rheed_to_afm_generation/final_metrics);
- exact command/runbook:
  [`reproducibility.md`](rheed_to_afm_generation/reproducibility.md).

The artifact manifest includes SHA-256 hashes and an explicit warning that the
test conditioning control failed.

## Verification

- focused unit tests: 4 passed;
- complete repository test suite: 285 tests ran; 264 passed, 1 failed, and 20
  errored. The four new generative-pipeline tests passed. All non-passing checks
  are outside this work: `test_rheed_peak_saddle.py` expects missing ignored
  `outputs/rheed_peak_saddle/...` artifacts (19 errors and the single failure),
  while `test_rheed_single_frame_manual.py` has a pre-existing macOS
  `/private/var` versus `/var` temporary-path mismatch;
- split-integrity audit: passed;
- generated/training exact-identity audit: passed;
- test overwrite guard: active;
- all relevant training/evaluation manifests inspected;
- final figures visually inspected;
- no file under `data/` had a modification time after task start;
- `git status` showed no tracked raw-data change.

## Limitations and unresolved risks

1. Only 23 independent growth groups exist; 116 AFM scans do not provide 116
   independent process conditions.
2. Validation has three groups and test has five, so model ranking and
   bootstrap intervals are unstable.
3. RHEED and AFM are not pixel-registered; pixel metrics measure texture
   similarity, not a deterministic point correspondence.
4. The decoder’s repeated border artifact demonstrates an architectural
   failure not adequately penalized by the current losses.
5. The learned condition bridge extrapolates poorly to the test descriptor
   distribution.
6. Radial PSD discards orientation and topology; a model can score tolerably
   while missing islands and steps.
7. The diversity ratio measures spread, not realism. Diverse artifacts can
   inflate it.
8. The one-time test is now consumed. Future architecture changes require a
   new prospective cohort or nested cross-validation; this test must not be
   reused for model selection.

## Recommended next research step

The immediate bottleneck is not CUDA throughput. Each full CVAE run took
approximately 1–2 minutes on MPS, well below the 30-minute handoff threshold.
A CUDA handoff is therefore **not recommended yet**.

The next defensible study should:

1. build an AFM-only translation-equivariant generator with reflection or
   circular padding and explicit border-artifact scoring;
2. require AFM-only fidelity/coverage gates before adding RHEED;
3. replace output-distance hinges with differentiable ensemble descriptor
   matching and condition-ranking losses;
4. run multi-seed nested group cross-validation;
5. collect additional independent growth groups, especially high-Rq islands
   and low-Rq step/terrace regimes;
6. reserve the new prospective groups as the only final test;
7. after those gates pass, test latent diffusion or diffusion refinement with
   classifier-free RHEED/descriptor guidance.

An NVIDIA machine becomes justified for the final latent-diffusion,
multi-seed, multi-fold experiment, but scaling the current artifact-prone
decoder would make the wrong model fail faster rather than solve the
scientific problem.

## Final claim boundary

Supported:

- a genuine, stochastic, non-retrieval RHEED-to-AFM generation pipeline exists;
- it is leakage-aware, reproducible, hashed, and evaluated on held-out growth
  groups;
- it improves some aggregate roughness and spectral metrics over retrieval;
- it generates non-identical ensembles.

Not supported:

- reliable conditioning on unseen RHEED;
- faithful AFM island/step morphology;
- superiority across all metrics;
- statistical significance with five test groups;
- readiness for prospective scientific inference.
