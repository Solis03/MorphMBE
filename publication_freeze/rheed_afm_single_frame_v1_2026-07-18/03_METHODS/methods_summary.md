# Frozen Method Summary

## Rq Definition

`Rq = sqrt(mean((h - mean(h))^2))` for AFM height map `h`, in nm.

## Target

Final target is `T4_second_order_trimmed_mean`: per-scan second-order AFM Rq values are trimmed by dropping min and max when scan count is at least four, then averaged.

## Quantitative Route

One manually selected RHEED keyframe is encoded by frozen DINOv2 ViT-S/14. q50 is the median of five selected Ridge-family strict OOF member predictions in nm space.

## q10/q50/q90

q50 is the selected ensemble median prediction. q10/q90 are fold-training absolute-error quantile amplitude bands around q50, not a validated 80% prediction interval.

## Visual A3 Route

A3 uses predicted Rq and predicted AFM descriptors to retrieve a representative historical AFM morphology from a strict heldout-excluded bank. Each heldout sample has 22 candidate source groups. The chosen map is amplitude-rescaled to q50 Rq. True heldout AFM/descriptors are evaluation-only.

## Descriptor Order

1. `rq_nm`
2. `ra_nm`
3. `robust_height_range_nm`
4. `psd_low_fraction`
5. `psd_mid_fraction`
6. `psd_high_fraction`
7. `psd_slope`
8. `correlation_length_nm`
9. `anisotropy`
10. `height_skewness`
11. `height_kurtosis`

Note: `rq_nm` is part of the 11-D descriptor vector, and A3 also adds a small explicit Rq penalty.
