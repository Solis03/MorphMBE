# Literature and standards basis for functional RHEED-to-AFM generation

## Scope

This review was used to design the M11/M12 continuation. It asks two separate
questions: which inductive bias can produce AFM-like island/terrace morphology
from very little paired data, and which surface descriptor can complement
`Rq` without pretending that a single scalar is an industry-standard
definition of surface function?

## Mandatory paper

Na et al., *Prediction of surface morphology and reflection spectrum of
laser-induced periodic surface structures using deep learning*, Journal of
Materials Processing Technology (2023),
[doi:10.1016/j.jmatprotec.2022.117853](https://doi.org/10.1016/j.jmatprotec.2022.117853),
was read in full from the supplied PDF and visually inspected after rendering.
Its most transferable idea is staged, multi-output learning: predict a
low-dimensional morphology representation and reconstruct an image while
validating physical outputs, rather than judging the image alone. The present
work adopts that principle but replaces the paper's comparatively large
supervised image mapping with a small-data, group-cross-fitted RHEED head and
a stochastic island/capture-zone generator.

## Surface metrology and semiconductor practice

- [ISO 25178-2:2021](https://www.iso.org/standard/74591.html) defines areal
  texture terms, parameters and material-ratio concepts. It supports reporting
  a vector of height, spatial, hybrid and functional parameters; it does not
  prescribe a universal scalar that replaces `Sq`.
- [SEMI M40](https://store-us.semi.org/products/m04000-semi-m40-guide-for-measurement-of-roughness-of-planar-surfaces-on-polished-wafers)
  is an AFM-focused guide for roughness measurement on polished wafers. The
  important lesson for this project is explicit scan scale, filtering and
  metrology provenance.
- SEMI's 2025 SNARF activity describes an
  [AFM guide for silicon-wafer roughness](https://downloads.semi.org/web/wstdsbal.nsf/0e0afa4c4969bea688256efd0062a27c/d43454e3d032f55088258d64000bfc6e%21OpenDocument),
  reinforcing that AFM sampling/filtering choices are part of the result.
- [SEMI MF1048](https://store-us.semi.org/products/mf104800-semi-mf1048-test-method-for-measuring-the-reflective-total-integrated-scatter)
  and NIST's discussion of
  [roughness regimes measurable by light scattering](https://www.nist.gov/publications/regimes-surface-roughness-measurable-light-scattering)
  motivate frequency-aware descriptions. NIST's
  [SCATMECH PSD documentation](https://pages.nist.gov/SCATMECH/docs/psd.htm)
  also makes explicit that a power spectral density represents scale content
  that one RMS height cannot.

The standards review therefore did **not** find a defensible universal
industry-standard scalar replacement for `Rq`. A paper should report `Rq/Sq`
and selected ISO-style parameters/PSD beside any new task-specific index.

## Multiscale and functional descriptors

Jacobs, Junge and Pastewka,
[Quantitative characterization of surface topography using spectral analysis](https://arxiv.org/abs/1607.03040),
show that surfaces with the same RMS height can have different slopes,
curvatures and functional behavior; PSD moments recover height-, slope- and
curvature-related information. Sanner et al.,
[Scale-dependent roughness parameters for topography analysis](https://arxiv.org/abs/2106.16103),
develop scale-dependent slope and curvature descriptions and emphasize
artifact-aware bandwidth selection. These papers directly motivate declaring
the 31.25 nm analysis scale in this experiment.

A practical ISO-parameter implementation and an AFM parameter study were also
reviewed:
[Surfalize](https://www.mdpi.com/2079-4991/14/13/1076) and
[AFM surface-parameter analysis](https://www.mdpi.com/2076-3417/15/12/6573).
A broader review of the relationship between surface texture and function is
available in
[Surface topography: metrology and properties](https://pmc.ncbi.nlm.nih.gov/articles/PMC8472325/).

## RHEED, MBE and island-growth basis

The procedural prior is not merely an image effect. Early
[GaSb epitaxial island observations](https://www.jstage.jst.go.jp/article/jvsj1958/23/7/23_7_326/_article/-char/en)
support a nucleation/coalescence view. RHEED intensity and spot/streak
structure are sensitive to surface order and step density, as illustrated by
[the RHEED step-density analysis](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.46.6825).
Joint III-V RHEED/AFM work also demonstrates that
[diffraction evolution and ex-situ morphology carry complementary growth
information](https://pmc.ncbi.nlm.nih.gov/articles/PMC4404429/).

This motivates the M12 stochastic marked-capture-zone model: RHEED features
predict amplitude and island statistics; random Laguerre capture zones,
terraces, grooves and fine texture generate a novel surface. No measured AFM
patch or nearest-neighbour AFM is provided at inference.

## Experimental FSMI

The proposed **Functional Surface Morphology Index (FSMI)** is explicitly an
experimental research descriptor, not an ISO or SEMI standard:

`FSMI = RMS(Sq, Δh31, C31, 0.25·(z90-z10), P70)`.

All five inputs are height-equivalent and expressed in nanometres:

1. `Sq`: areal RMS height;
2. `Δh31`: RMS height increment over the declared 31.25 nm lateral scale;
3. `C31`: half the RMS second height difference at 31.25 nm;
4. `0.25·(z90-z10)`: bearing/material-ratio core-height equivalent;
5. `P70`: median prominence of q70 islands.

The equal-unit RMS construction avoids cohort-fitted standardization and
target-dependent weights. It is sensitive to frequency, curvature,
bearing-height span and object relief that `Rq` alone discards. It should be
reported with its five components, scan size, pixel spacing and filtering;
prospective correlation with a real material-property endpoint is required
before calling it a material-performance index.

## Method decision

Full image diffusion was retained as an earlier negative experiment: with
only 15 independent training growths it added granular texture without
reliable conditional topology. M12 instead uses a small-data staged model:

1. physically interpretable RHEED morphology/temporal features;
2. strictly nested positive-target regression with log-range calibration;
3. RHEED-conditioned island statistics;
4. stochastic Laguerre nucleation/capture zones;
5. edge-preserving continuous terraces and signed-distance island shoulders;
6. a low-weight AFM spectral prior and fine texture;
7. group-held-out Rq, FSMI, PSD, island, AFM-prior and confidence evaluation.

This is the locally defensible choice for the present data regime. A future
conditional latent diffusion model becomes attractive after substantially
more independent growth groups are collected.
