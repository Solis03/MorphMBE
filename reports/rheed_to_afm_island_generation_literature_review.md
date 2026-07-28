# Literature review: object-aware and diffusion-refined RHEED-to-AFM generation

Date: 2026-07-27

## Research question

The immediate failure mode is not simply blur. The M5 generator can match
roughness and frequency statistics while producing cloud-like fields that do
not contain the islands, coalesced mounds, terraces, valleys, and sharp step
boundaries visible in real AFM. The next model must therefore represent the
surface as a population of spatial objects and must separately learn the
experimental texture carried by real AFM.

## Mandatory reference

Na, Yoo, and Ki, *Prediction of surface morphology and reflection spectrum of
laser-induced periodic surface structures using deep learning*, Journal of
Manufacturing Processes 84 (2022) 1274-1283,
[doi:10.1016/j.jmapro.2022.11.004](https://doi.org/10.1016/j.jmapro.2022.11.004).
The local full text was read and pages 3-8 were visually inspected.

The paper used only 32 process conditions (26 train, 3 validation, 3 test),
but expanded each training condition to 200 random 256 x 256 crops. Its
morphology model is a stochastic conditional GAN: a 128-dimensional random
vector and process conditions are concatenated, followed by six
nearest-neighbor upsampling residual blocks. The discriminator uses
differentiable translation/cutout augmentation and projection conditioning.
Both networks use self-attention and spectral normalization, and are trained
with hinge loss. Importantly, early stopping is based on validation error in
the 2D Fourier domain, while the final scientific evaluation measures pattern
period and width rather than relying on generic image similarity.

Consequences for this project:

1. Do not collapse RHEED to one deterministic, pixel-regressed image.
2. Preserve stochastic draws for one-to-many morphology uncertainty.
3. Train an AFM-realism mechanism on random crops, but split by growth group
   before cropping.
4. Evaluate explicit island count, area, boundary, valley, and spectral
   statistics.
5. Because this dataset is substantially smaller and paired RHEED-to-AFM
   alignment is not pixelwise, a full conditional GAN is less defensible than
   a staged object generator plus an AFM-only texture prior.

## Epitaxial-growth and RHEED basis

- Point-island models describe nucleation, diffusion-limited or
  attachment-limited capture, and the island size/spatial distribution
  ([Han et al., 2016](https://pubmed.ncbi.nlm.nih.gov/28799390/)).
- A phase-field epitaxy model reproduces island density, size distributions,
  mound structures, coarsening, and roughening
  ([Liu and Metiu, 2004](https://pubmed.ncbi.nlm.nih.gov/14995452/)).
- RHEED measurements and solid-on-solid simulation connect the transition
  from coalescing 2D clusters to step advancement with surface step density
  ([Shitara et al., 1992](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.46.6825)).
- In early GaSb-on-GaAs MBE, GaSb-lattice islands were reported first and
  evolved toward trapezoidal shapes as thickness increased
  ([J-STAGE record](https://www.jstage.jst.go.jp/article/jvsj1958/23/7/23_7_326/_article/-char/en)).
- Direct RHEED/AFM correlation on a III-V surface shows broad streaks with
  corrugation and spotty/chevron patterns with nanoscale droplets; roughness
  and object geometry change together
  ([Azadmand et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4404429/)).
- A modern RHEED geometry review notes that protruding 3D islands produce
  transmission-like diffraction features
  ([Jo et al., 2022](https://www.sciencedirect.com/science/article/pii/S0968432822000828)).

These papers support, but do not prove for this dataset, the working physical
hypothesis:

> RHEED contains information about roughness, step density, island
> size/spacing, and growth mode. A generator that explicitly predicts those
> population statistics is better aligned with the measurement physics than a
> stationary Gaussian texture field.

The implementation therefore predicts a growth-group-level island descriptor
vector from RHEED and then stochastically realizes a capture-zone/terrace
surface. The Laguerre construction is an interpretable geometric
approximation, not a claim of atomistic simulation.

## AI and microscopy basis

- Object-centric compositional generation separates object presence,
  location, size, shape, and appearance before assembling a scene
  ([Yuan et al., ICML 2019](https://proceedings.mlr.press/v97/yuan19b.html)).
  This directly motivates representing AFM as a population of islands instead
  of one texture field.
- MorphoDiff conditions a diffusion model on perturbation embeddings to
  generate microscopy morphology and validates both image quality and
  biological signals
  ([MorphoDiff](https://pmc.ncbi.nlm.nih.gov/articles/PMC11702702/)).
- Conditional diffusion is established for image-to-image reconstruction,
  denoising, and inpainting in scientific/medical images
  ([Med-cDiff](https://pmc.ncbi.nlm.nih.gov/articles/PMC10669033/)).
- STEMDiff explicitly separates a structural label map from learned
  experimental image/noise appearance, using conditional diffusion to retain
  both
  ([STEMDiff](https://pmc.ncbi.nlm.nih.gov/articles/PMC12591186/)).
- A recent microscopy diffusion review distinguishes unconditional,
  text-conditioned, and image-conditioned synthesis and emphasizes
  structure-aware conditioning
  ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309395/)).

These works motivate the M7/M8 design: first create a RHEED-conditioned island
structure map, then use a residual diffusion model trained only on real AFM
crops to add step-edge and instrument-scale texture. Weak SDEdit-like
refinement is used because full-noise sampling was empirically unstable in the
small-data regime.

## Candidate families tested

| ID | Family | Scientific role | Inference uses a measured AFM? |
|---|---|---|---|
| M5 | Matérn/spectral hybrid | preserved non-retrieval baseline | No |
| M6a | superellipse island/valley primitives | explicit isolated islands | No |
| M6b | multiscale weighted Laguerre terraces | nucleation/capture/coalescence approximation | No |
| M6c | M6b plus learned spectral prior | object structure plus fine frequency content | No |
| M7 | structure-guided residual DDPM | real-AFM texture learned from training crops | No |
| M8 | M7/M6c Pareto blend | retain object topology, AFM texture, and condition signal | No |
| M9 | edge-enhanced / terrace-quantized renderers | test whether sharper boundaries alone solve realism | No |
| M10 | dense multiscale Laguerre islands + spectral prior | selected object-population generator | No |

M7 learns from real AFM during training, but no measured AFM image, patch, or
nearest neighbor is supplied at inference. All AFM-derived training crops obey
the growth-group split.

Strict grouped evaluation rejected M7/M8 as the final model: weak refinement
looked plausible on the three validation growths, but its improvement did not
survive 15-growth leave-one-group-out evaluation. M9 improved selected
frequency/edge statistics but changed the visible topology too little.
Increasing the predicted capture-zone population and selecting a 65:35 blend
of dense island structure and stochastic spectral texture (M10) gave the most
defensible development trade-off. It is intentionally described as a
physics-informed stochastic compositor, not as atomistic growth simulation or
as a diffusion success.

## Claim boundary

The historical test cohort was already consumed by earlier project work and
remains closed. Current model selection is supported by 15 strict
training-growth leave-one-group-out folds and the three pre-existing
validation growth groups. This is development evidence, not a new untouched
test claim. A prospectively frozen growth cohort is required for the next
confirmatory paper claim.
