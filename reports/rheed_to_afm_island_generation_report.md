# RHEED-conditioned AFM island-generation report

Date: 2026-07-27

Status: selected development milestone; prospective confirmation still needed

## Executive result

The previous M5 milestone was preserved unchanged at commit
`d358781ffafccfcd77f7ab1f305081b432b0e5a7`. Its main failure was correctly
identified: it produced sharp stochastic spectral fields, but their topology
looked like cloud or cotton texture rather than AFM islands, coalesced mounds,
terraces and valleys.

The selected continuation is **M10: a RHEED-conditioned dense multiscale
island generator**. It predicts morphology descriptors from RHEED, realizes a
new population of weighted Laguerre capture zones at multiple scales, blends
65% island structure with 35% generated spectral texture, and rescales the
height field using the predicted Rq. It is a true stochastic generator:

- no measured AFM, AFM patch or nearest-neighbour image is supplied at
  inference;
- retrieval is not part of generation;
- measured AFM is used only for training targets and post-generation scoring;
- median maximum SSIM to any training AFM is 0.0366 in strict cross-fitting,
  and exact training-pixel equality is zero.

Visually, M10 contains discrete and coalesced island boundaries rather than
only stationary cloud texture. Quantitatively it improves AFM-support
distance, island area/count errors, RHEED-condition consistency and the
overall morphology composite on strict grouped development evidence.

This is a meaningful milestone, not a final prospective test claim. The old
historical test cohort was already consumed and remained closed throughout
this work.

## Data integrity and split policy

- All 11 entries in `removelist.txt` are excluded before AFM/RHEED joins,
  folds, training, evaluation and figures.
- Removal-list SHA-256:
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
- Model selection uses 15 strict leave-one-growth-group-out folds.
- Three pre-existing validation groups (6022, 6056 and 6080) are reported
  separately.
- No historical-test sample is opened for fitting, tuning, selection or
  visualization.
- Raw RHEED and AFM data are unchanged.

The valid claim boundary is therefore: **strict grouped development evidence
plus pre-existing validation**. A prospectively frozen new growth cohort is
required for confirmatory publication language.

## Literature and physical rationale

The mandatory Na, Yoo and Ki paper, *Prediction of surface morphology and
reflection spectrum of laser-induced periodic surface structures using deep
learning* ([DOI](https://doi.org/10.1016/j.jmapro.2022.11.004)), was read in
full. Its most transferable ideas were stochastic conditioning, random-crop
expansion, explicit Fourier-domain validation and evaluation of
domain-specific morphology rather than generic image similarity.

The island representation is additionally motivated by:

- point-island nucleation/capture models
  ([Han et al.](https://pubmed.ncbi.nlm.nih.gov/28799390/));
- phase-field accounts of epitaxial island density, coarsening and mound
  formation ([Liu and Metiu](https://pubmed.ncbi.nlm.nih.gov/14995452/));
- RHEED links between coalescing clusters, step advancement and step density
  ([Shitara et al.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.46.6825));
- reported early GaSb-lattice island formation and later trapezoidal
  evolution in GaSb/GaAs MBE
  ([J-STAGE record](https://www.jstage.jst.go.jp/article/jvsj1958/23/7/23_7_326/_article/-char/en));
- III-V RHEED/AFM observations linking streaks/corrugation and
  spotty/chevron patterns with nanoscale objects
  ([Azadmand et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4404429/)).

The implementation does not claim atomistic fidelity. Weighted Laguerre
cells are an interpretable approximation to nucleation capture zones and
coalescence, chosen because it matches the small-data regime better than an
unconstrained end-to-end image generator.

## Methods tried

| ID | Method | Outcome |
|---|---|---|
| M5 | conditional Matérn/spectral random field | preserved cloud-texture baseline |
| M6a | isolated superellipse islands/valleys | rejected: too large and sparse |
| M6b | multiscale weighted Laguerre terraces | retained structural component |
| M6c | M6b plus learned spectral texture | improved compromise |
| M7 | residual DDPM trained on real-AFM crops | full noise was granular; weak SDEdit retained structure |
| M8 | 50:50 weak-diffusion/M6c blend | validation looked promising but strict LOO object gains failed |
| M9a | edge enhancement | frequency/edge gains, too little visible topology change |
| M9b | quantized terraces | visibly sharper, but composite/island errors worsened |
| M10 | dense multiscale islands + spectral texture | selected development model |

M7 is the requested “real AFM second generation / diffusion” experiment.
Every AFM crop used by a fold comes only from its training growth groups.
Although weak diffusion produced locally convincing texture, it was rejected
because the improvement did not survive the stricter 15-growth LOO analysis.
This negative result is preserved in the registry and Fig. 5.

## M10 architecture

1. A temporal RHEED condition model uses the manually selected key-frame
   windows and existing leakage-safe descriptors.
2. Fold-local ridge models predict an AFM morphology vector, including
   island counts and areas at q55/q70/q82, solidity, eccentricity, boundary
   gradient ratios, valley population, gradient/laplacian texture and flat
   fraction.
3. A stochastic marked population of coarse and fine nuclei is drawn.
4. Weighted Laguerre capture zones produce island/mound/valley support and
   coalescence boundaries; the dense variant uses a sixfold seed factor.
5. The generated island field is mixed with an independently generated M5
   spectral field at a preselected 0.65:0.35 ratio.
6. The field is normalized and scaled to the RHEED-predicted physical Rq.

Different RHEED inputs therefore change predicted physical descriptors and
the sampled island population. Random draws provide one-to-many morphology
uncertainty; they do not retrieve an existing AFM.

## Quantitative results

All values below are medians; lower is better except sharpness ratio (target
1) and texture-pass fraction.

### Strict 15-growth leave-one-group-out

| Metric | M5 cloud | M10 dense islands | Change |
|---|---:|---:|---:|
| Rq MAE (nm) | 0.829 | 0.829 | unchanged |
| RHEED-condition descriptor MAE (z) | 0.986 | 0.876 | 11.1% better |
| PSD log distance | 0.925 | 0.860 | 7.1% better |
| morphology composite | 8.545 | 8.106 | 5.1% better |
| sharpness ratio | 0.939 | 0.808 | farther below target; trade-off |
| texture pass | 13/15 | 13/15 | unchanged |
| full island-feature MAE (z) | 1.492 | 1.514 | 1.5% worse; trade-off |
| AFM-support distance | 7.985 | 6.438 | 19.4% better |
| q70 median-area log error | 1.112 | 0.811 | 27.1% better |
| q70 count log error | 0.611 | 0.540 | 11.5% better |
| max training SSIM | 0.0372 | 0.0366 | no copying signal |

The slight worsening of the aggregate island MAE is not hidden: M10 improves
the specific count/area quantities and AFM support while some of the 16
topology components worsen. This is why the result is presented as a Pareto
improvement rather than universal dominance.

### Pre-existing validation cohort

| Metric | M5 cloud | M10 dense islands | Change |
|---|---:|---:|---:|
| Rq MAE (nm) | 0.833 | 0.833 | unchanged |
| RHEED-condition descriptor MAE (z) | 0.940 | 0.786 | 16.4% better |
| morphology composite | 8.513 | 7.910 | 7.1% better |
| full island-feature MAE (z) | 1.726 | 1.423 | 17.5% better |
| AFM-support distance | 7.991 | 5.951 | 25.5% better |
| q70 median-area log error | 1.138 | 0.765 | 32.7% better |
| PSD log distance | 0.627 | 0.879 | worse; explicit trade-off |
| texture pass | 3/3 | 3/3 | unchanged |

The validation set is only three growths; it supports qualitative
consistency, not precise population-level inference.

## Confidence and “self-awareness”

Two uncertainty views are kept separate:

1. the prior RHEED-condition confidence from M5;
2. a new **morphology confidence index** calibrated specifically against
   M10 island-feature error.

The morphology calibrator uses only inference-time diagnostics—distance from
the training AFM morphology support and maximum training-image SSIM—and is
nested within each outer growth fold. Higher index means lower expected
island error. It is explicitly **not a probability of correctness**.

- Cross-fitted predicted-error versus realized-error Spearman
  ρ = +0.589, p = 0.0208; equivalently confidence versus error
  ρ = -0.589.
- Predicted island-error MAE is 0.322 z.
- The 90% conformal upper bound covers 14/15 held growths (93.3%).
- Validation confidence indices are 50/100 for 6022 and 68.75/100 for 6056
  and 6080.

Figure 7 selects success and failure cases algorithmically from strict
cross-fitted error/confidence ranks. Growth 6062 is a visibly poor case with
44/100 morphology confidence and 2.26 z realized island error; this is the
desired, limited form of model “self-awareness.” With only 15 independent
training growths, the confidence score is necessarily coarse.

## Qualitative assessment

M10 visibly replaces much of M5's fine cotton-like texture with connected
objects and valleys at several length scales. The q70 contour audit shows
boundary-gradient ratios substantially closer to measured AFM for the
illustrated validation growth. The 18-sample atlas is sorted by measured Rq
and keeps sample order fixed across RHEED, generated and measured columns.

Remaining visible failures:

- large coalesced terraces in 6022 are still under-resolved;
- high-roughness 6080 is under-predicted in amplitude (2.50 versus 4.40 nm);
- the generator matches morphology distributions, not the spatial location
  of a paired AFM field, so pixel SSIM is expectedly low;
- some M10 surfaces retain residual spectral softness between island
  boundaries;
- the small data set cannot support a high-capacity conditional diffusion
  model without unstable texture or conditional under-utilization.

## Figures and artifacts

Publication-ready PNG and vector PDF files are under:

`reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/figures`

- Fig. 1: RHEED + M5 + M10 + measured AFM for all three validation growths.
- Fig. 2a-c: 18-sample Rq-ordered expanded atlas with confidence.
- Fig. 3: strict grouped baseline-versus-final metrics.
- Fig. 4: island-boundary/topology audit.
- Fig. 5: structure/diffusion ablations and DDPM learning curve.
- Fig. 6: confidence calibration and validation uncertainty.
- Fig. 7: predefined success/failure audit.
- Fig. 8: physical and island-descriptor correlations.

Key machine-readable outputs:

- `selected/baseline_vs_final_metrics.csv`
- `selected/experiment_registry.csv`
- `selected/best_model_manifest.json`
- `selected/confidence/morphology_confidence_manifest.json`
- `selected/reproducibility_runbook.md`
- generated maps in
  `outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected`

## Compute and next step

The machine is an Apple M1 Pro with 32 GiB unified memory. PyTorch MPS is
available; CUDA is not. The M10 full grouped run is locally feasible
(approximately two minutes), and the 15-fold diffusion experiment completed
in approximately 14 minutes. Neither exceeds the CUDA handoff rule, so a CUDA
handoff is not recommended for reproducing this milestone.

For the next scientifically decisive step, freeze a prospective cohort of
new growth runs before any model update. With substantially more independent
growth groups, revisit structure-conditioned latent diffusion or a
projection-conditioned GAN, while keeping the object descriptors, fold-local
calibration and confidence audit introduced here. More compute alone will not
replace independent growth diversity.

## Verification record

- Package compilation: passed.
- Focused island-generation/diffusion/confidence tests: 9/9 passed.
- All RHEED-to-AFM tests, including CVAE and sharp-generation baselines:
  19/19 passed.
- Main `tests/` suite: 326 passed and 22 failed. The 20
  `rheed_peak_saddle` failures require absent historical checkpoint artifacts;
  the remaining two require an unavailable parquet engine. They are outside
  this package and pre-exist this milestone.
- Bare repository-root `pytest` additionally encounters three collection
  mismatches because archived paper-freeze snapshots contain duplicate test
  module names. Explicit `pytest tests` avoids that archive collection.
- Removal-list audit: 11 canonical exclusions; zero overlap in retained
  descriptor, fold, physics and phase-1 tables.
- Selected safety manifests: historical test false, retrieval false, measured
  AFM patch at inference false.
- Figure audit: ten PNG plus ten PDF figures; Fig. 1 and Fig. 7 PDFs were
  independently rendered with Poppler and visually checked.
- `git diff -- data` is empty.

## Conclusion

M10 is the best locally defensible result. It is genuinely generative,
conditioned on RHEED, visibly more AFM-like than the preserved cloud baseline,
and improves several object-level and aggregate metrics under leakage-safe
grouped evaluation. It does not yet justify a claim of pixel-accurate surface
reconstruction or prospective generalization. The strongest paper framing is
therefore: **RHEED-conditioned stochastic prediction of AFM morphology
distributions with explicit island topology and calibrated, group-aware
uncertainty**.
