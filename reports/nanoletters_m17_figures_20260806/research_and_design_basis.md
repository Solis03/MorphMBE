# Research and design basis

## Nano Letters / ACS production constraints

The package follows the current Nano Letters author guidance available on
2026-08-06:

- Nano Letters Letters normally contain 3--6 figures in total.
- Maximum double-column width is 504 pt (7.0 in); maximum single-column width
  is 240 pt (3.33 in).
- Recommended raster resolution is 1200 dpi for line art, 600 dpi for grayscale,
  and 300 dpi for color. The supplied color composites are conservatively
  rendered at 600 dpi, and PDF keeps diagram elements and text vectorized.
- Lettering should be at least 4.5 pt in a sans-serif face and rules at least
  0.5 pt. This package uses Arial/Helvetica-compatible lettering at 5.3--7.2 pt
  and 0.65 pt axes/rules.
- Captions are placed below figures in the manuscript and must be independently
  understandable. Standalone English captions are supplied in `captions.md`.
- Quantitative plots use redundant shape/line-style encodings and an
  Okabe--Ito-derived palette so interpretation does not depend on color alone.

Primary guidance:

- [Nano Letters author guidelines](https://researcher-resources.acs.org/publish/author_guidelines?coden=nalefd)
- [Nano Letters author-guidelines PDF, updated 2026-07-03](https://researcher-resources.acs.org/publish/author_guidelines/pdf?coden=nalefd)

## AFM false color

An AFM height map is scalar metrology, so it is represented by a continuous
false-color gradient plus an explicit nanometer colorbar rather than by a
single orange hex value. Gwyddion documents that false-color maps map height
values to colors and supports fixed/full/adaptive mapping. The official
Gwyddion 2.71 `Gold` gradient was extracted from the distributed `data/gradients/Gold`
resource and reproduced exactly as piecewise-linear RGB control points:

| Position | RGB (0--1) | Hex |
| ---: | --- | --- |
| 0.000000 | (0.000000, 0.000000, 0.000000) | `#000000` |
| 0.333333 | (0.345098, 0.109804, 0.000000) | `#581C00` |
| 0.666667 | (0.737255, 0.501961, 0.000000) | `#BC8000` |
| 1.000000 | (0.988235, 0.988235, 0.501961) | `#FCFC80` |

Sources:

- [Gwyddion color-map documentation](https://gwyddion.net/documentation/user-guide-en/color-map.html)
- [Gwyddion resource-format documentation](https://gwyddion.net/documentation/user-guide-en/resources.html)
- [Official Gwyddion source downloads](https://gwyddion.net/download.php)

For fair pairwise viewing, each generated/measured AFM pair uses the same linear
normalization. Limits are the pooled 1st and 99th percentiles, made symmetric
about zero. This suppresses isolated display outliers without changing stored
height data. Every AFM panel has a nanometer colorbar (directly or shared within
the panel) and a 250 nm scale bar. A recent Nano Letters AFM example likewise
reports scale bars and z ranges in nanometers:
[Kumar et al., Nano Letters 2023](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.2c04299).

## Figure plan and sample selection

The three main-text figures consume only half of the journal's 3--6 figure
budget and deliberately separate workflow, architecture/validation, and
results:

1. Figure 1: compact end-to-end AutoRHEED overview, distinct from the supplied
   presentation screenshots and built around real pipeline assets.
2. Figure 2: model internals, true stochastic generation, and growth-level LOO
   validation.
3. Figure 3: three selected examples plus cohort-wide statistical context.

Public IDs are assigned by ascending internal growth identifier and padded to
two digits because the cohort contains 27 samples. This resolves the user's
`001` versus `01` wording in favor of the explicitly illustrated `01, 02, ...`
format. The private mapping remains outside manuscript-facing graphics.

The qualitative rows were chosen to cover the measured Sq range while favoring
low relative Sq error: Sample 23/N6342 (3.6%), Sample 04 (19.8%), and Sample 20
(3.5%). Selection is labeled and accompanied by all-27 parity and reliability
plots; it is not presented as an exhaustive qualitative survey. In particular,
the rough Sample 20 has low scalar Sq error but low predicted reliability, which
is visibly retained instead of hidden. This preserves the distinction between
scalar agreement and full stochastic morphology agreement.

## Scientific claim boundary

- The selected model is `M17b_topology_sparse_peak_terrace`.
- `retrieval_at_inference = false`.
- `measured_afm_patch_used_at_inference = false`.
- Every displayed prediction is from an outer fold that excludes the displayed
  growth group from model and AFM-texture fitting.
- The generated AFM is a stochastic conditional morphology sample, not an
  island-to-island or pixel-to-pixel reconstruction.
- N6342 motivated M17 renderer development and is therefore retrospective
  method-development evidence, not a prospectively untouched-test result.
