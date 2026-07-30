# Audited Sq metrology and endpoint-aware RHEED-to-AFM generation

Date: 2026-07-29

Models: **M16 endpoint-aware Sq** + **M16b micro-island/terrace generator**

Evaluation: retrospective strict leave-one-growth-out (LOO), 28 growths.

## Outcome

The current AFM roughness target is numerically correct. The repository and
local Gwyddion 2.71 independently reproduce the same third-order, row-levelled
areal RMS height to below \(4\times10^{-9}\) nm over 214 checked fields. The
quantity is labelled **Sq** because it is calculated over a two-dimensional
height field; historical filenames and NanoScope profile labels may still say
Rq.

The endpoint-aware scalar model materially improves both extremes without
discarding difficult samples:

| strict 28-LOO model | Sq MAE | Pearson r | smooth MAE, true Sq < 1.2 nm | rough MAE, true Sq ≥ 5 nm |
|---|---:|---:|---:|---:|
| M15b automatic R3D baseline | 1.321 nm | 0.622 | 1.259 nm | 2.927 nm |
| **M16 endpoint-aware R3D** | **1.070 nm** | **0.738** | **0.759 nm** | **1.233 nm** |

Selected strict held-growth predictions are:

| growth | measured sample-median Sq | M16 predicted Sq | interpretation |
|---|---:|---:|---|
| 6101 | 0.481 nm | **0.567 nm** | correct very-smooth endpoint |
| N6342 | 0.804 nm | **0.954 nm** | below 1 nm |
| N6358 | 0.979 nm | **0.800 nm** | correct smooth regime |
| N6382 | 1.113 nm | **1.067 nm** | near identity |
| 6095 | 9.074 nm | **7.691 nm** | high spotty endpoint retained |
| 6099 | 9.395 nm | **6.500 nm** | high endpoint, still underestimated |

The updated live deployment was separately smoke-tested on raw video. It
returned 0.481 nm for 6101 and 0.895 nm for N6342, with generated-map Sq
matching the scalar conditioning to approximately \(10^{-6}\) nm.

## Metrology decision

For a height field \(z_{ij}\), the code evaluates

\[
S_q =
\sqrt{\frac{1}{MN}\sum_{i,j}(z_{ij}-\bar z)^2}.
\]

It uses the complete field (`ddof=0`). Before the RMS calculation, a cubic
polynomial is fitted independently to every fast-scan row and subtracted.
Raw headers confirm that only the physical `ZSensor`/height channel is used.
They also document the instrument's real-time line plane fit and no offline
plane fit. The repository correction is therefore an explicit additional
offline metrology definition, not an undocumented assumption.

The direct local Gwyddion audit used Gwyddion's own NanoScope importer,
`gwy_data_field_row_level_poly`, and `gwy_data_field_get_rms`. It is independent
of the repository's decoder and polynomial implementation. The complete audit,
including order 0/1/2/3 sensitivity and NanoScope-export QC, is in
`reports/afm_metrology_reaudit/REPORT.md`.
The final source-integrity rerun checked 180 raw AFM files and their 180
decoded ZSensor arrays; all 360 SHA-256 values still match
`reports/rheed_m16_endpoint_smooth_generation/source_integrity/`.

The formula agrees with Gwyddion's
[statistical-quantities documentation](https://gwyddion.net/documentation/user-guide-en/statistical-analysis.html).
Row polynomial levelling is documented by the
[Gwyddion processing API](https://gwyddion.net/documentation/libgwyprocess/libgwyprocess-correct.php).
The limitation that background removal can suppress real long-wavelength
morphology follows Nečas *et al.*,
[doi:10.1088/1361-6501/ab8993](https://doi.org/10.1088/1361-6501/ab8993).

## M16 scalar method

M16 combines three target-blind RHEED paths:

1. a causal-eight-frame R3D-18 embedding with PCA-5 and ridge regression;
2. the same embedding with PCA-8 and ridge regression; and
3. a low-dimensional expert augmented by a local diffraction-maximum
   horizontal-elongation statistic.

Broad phosphor illumination is removed with a difference of Gaussians before
measuring each local maximum. Highly elongated horizontal maxima activate the
smooth/streak expert. Conversely, rough extrapolation is allowed only when both
temporal heads exceed the training upper-quartile roughness threshold. All
gates are fitted using the 27 training growths of each outer fold.

Confidence remains target-blind for the query. It combines prediction-amplitude
support, pre-existing angular-TTA/head risk, and independent streak support.
Across strict LOO predictions, confidence versus absolute Sq error has
Spearman \(\rho=-0.469\); nominal 90% intervals cover 25/28 growths (89.3%).

## M16b image generator

M16b is a genuine generator. It does not retrieve a measured AFM image or copy
a held-out patch. A RHEED-conditioned island-statistics model and conditional
spectral prior generate new stochastic fields. In the very-smooth regime, the
spectral field is filtered at the smallest resolved island scale and augmented
with rounded local micro-islands. Between 0.8 and 1.6 nm, it interpolates into
the frozen edge-preserving terrace renderer; higher-Sq samples keep the
terrace/island path.

The first smooth renderer, M16a, is preserved as a failed ablation because it
amplified band-pass residuals and looked pixel-like. M16b reduces mean gradient
relative error from 0.463 to 0.263, Laplacian relative error from 0.897 to
0.450, and high-frequency PSD relative error from 12.35 to 0.82. The AFM
texture gate rises from 60.7% to 82.1%.

## Figures

The main 28-growth RHEED/generated/measured atlas and all supporting plots are
under:

`reports/rheed_m16_end_to_end_generation/20260729_m16_m16b_line3_full28_smooth_endpoint_v2/full28_loo/figures/`

Key files:

- `Fig1a_full28_loo_atlas.png` through `Fig1f_full28_loo_atlas.png`: all 28
  samples in fixed measured-Sq order, including failures;
- `Fig2_full28_target_scatter.png`: measured versus predicted descriptors;
- `Fig5_confidence_audit.png`: confidence/error behavior;
- `Fig6_renderer_roughness_strata.png`: roughness-stratified images;
- `Fig7_largest_failures.png`: largest errors;
- `Fig9_extra_five_renderer_comparison.png`: second-batch comparison.

Scalar endpoint figures are under
`reports/rheed_endpoint_generation/m16_full28_v1/figures/`.
Both PNG and vector PDF are provided.
The accepted, rejected, historical, and deployment runs are indexed in
`reports/rheed_m16_endpoint_smooth_generation/experiment_registry.csv`.

## Limitations and claim boundary

- This is strict per-sample LOO computation, but the same 28-growth cohort was
  used retrospectively during method development. It is not an untouched
  prospective test.
- 6081 remains a scientifically important failure: its RHEED looks spotty, yet
  its measured Sq is only 0.94 nm; M16 predicts 4.28 nm and assigns low
  confidence. It is retained in every atlas and metric.
- 6101's scalar is accurate, but its uncommon large-terrace AFM layout is not
  reconstructed exactly. Confidence is low, and the report does not imply
  pixel-registered reconstruction.
- FSMI still uses the M15b temporal head and remains compressed at the upper
  endpoint (MAE 1.168 nm, Pearson r 0.630).
- A future publication claim should be confirmed on a prospectively frozen
  growth batch and with an archived instrument-side flattening recipe.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_endpoint_generation.run_endpoint_ensemble --help

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_m16_end_to_end_generation_line3_full28_smooth_v2.json \
  --device mps

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.visualization \
  --config configs/rheed_m16_end_to_end_generation_line3_full28_smooth_v2.json

PYTHONPATH=src:. .venv/bin/python scripts/prepare_rheed_realtime_model.py \
  --config configs/rheed_realtime_ui_m16_full28_line3_orientation90_keyframe_locked_v8.json \
  --force
```

The active UI command remains:

```bash
PYTHONPATH=src:. .venv/bin/python -m rheed2morph.realtime.cli \
  --config configs/rheed_realtime_ui.json
```
