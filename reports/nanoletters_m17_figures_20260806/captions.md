# Manuscript-ready figure captions

**Figure 1. Overview of the AutoRHEED framework.** (a) In situ RHEED video is
converted into an automatically localized, orientation-locked 16-frame clip by
causal clear-moment selection. The displayed frames are real frames 1661--1676
from Sample 23. (b) A hybrid physics--AI inference path combines an R3D-18
temporal embedding with streak-endpoint descriptors. Strictly cross-fitted
heads predict surface roughness (Sq), the functional surface morphology index
(FSMI), morphology conditions, and reliability; these quantities condition the
nonretrieval M17b stochastic AFM generator. Measured AFM is unavailable to the
model at inference. (c) One physically scaled M17b realization for Sample 23
is shown with the predicted Sq (0.83 nm) and cross-fitted reliability index
(71/100). The measured AFM is revealed only
after prediction for evaluation. Generated and measured AFM images share a
linear Gwyddion Gold false-color scale defined by the pooled 1st--99th
percentiles and made symmetric about zero; the colorbar is in nanometers. AFM
scan width, 1.0 µm; scale bars, 250 nm.

**Figure 2. Physics-guided stochastic AFM generation and leakage-controlled
validation.** (a) Sixteen real RHEED frames from Sample 23 are processed by an
RHEED representation stack comprising a 1536-dimensional DINOv2 key-frame
descriptor, a 512-dimensional R3D-18 16-frame descriptor, and a causal
eight-frame R3D descriptor augmented by six streak features. A hybrid
PCA/physics condition head and a gated three-expert endpoint ensemble predict
the nine-dimensional condition vector and Sq, respectively; the latter sets
the physical amplitude coordinate. Every scaler, dimensionality reduction,
and prediction head excludes the held growth from fitting. (b) The M17b
generator predicts a conditional spectral prior by inner-LOO-selected ridge
regression and 35-step IAAFT synthesis, while a separate ridge model generates
coarse/fine Laguerre island topology. A roughness-dependent regime blend
(0.8--1.6 nm) adds 4--24 sparse peaks before unit-Sq normalization and physical
scaling. Four genuine stochastic realizations (independent seeds) are shown
for Sample 23 at 128 × 128 pixels; no retrieval or measured AFM patch is used
at inference. All AFM images use the linear Gwyddion Gold scale; scale bars,
250 nm. (c) In each of 27 outer
leave-one-growth-out folds, one complete growth group is held out, the remaining
26 growths are fitted, and the held sample is predicted before comparison with
AFM. Growth group is the leakage boundary; one operator-invalid growth was
excluded before fitting.

**Figure 3. Selected cross-validated predictions across the measured roughness
range.** (a) Strict outer leave-one-growth-out examples selected for Sq
agreement span smooth Sample 23 (predicted/measured Sq, 0.83/0.80 nm),
intermediate Sample 04 (2.46/2.05 nm), and rough Sample 20 (8.76/9.07 nm).
Each row contains a real automatically selected RHEED key frame, one genuine
M17b stochastic AFM realization, a measured AFM scan shown only for evaluation,
and the normalized radially averaged power spectral density (PSD). Generated
maps represent conditional morphology distributions and are not expected to be
pixel registered to the measured scan. Within each row, generated and measured
AFM images share a linear Gwyddion Gold false-color scale defined by their
pooled 1st--99th percentiles and made symmetric about zero; colorbars are in
nanometers. AFM scan width, 1.0 µm; scale bars, 250 nm. (b) Cohort-wide
outer-LOO Sq agreement for all 27 growths (Pearson r = 0.74; MAE = 1.11 nm).
(c) The cross-fitted reliability index is inversely associated with the
realized joint error rank (Spearman ρ = -0.57, p = 0.002). Public sample labels
are anonymized. Sample 23 is retrospective method-development evidence, not a
prospectively untouched test; the original identifier is retained only in the
internal provenance record.
