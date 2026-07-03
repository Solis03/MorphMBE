# RHEED SSL Temporal MVP-6 Evaluation

## Direct Answers

1. RHEED beats mean-condition baseline: `True`.
2. RHEED beats metadata-only baseline: `True`.
3. Temporal video beats final-frame-only: `True`.
4. SSL pretraining improves label efficiency: `True` (see label-efficiency CSV; this is conservative unless measured improvement is clear).
5. Predictable descriptors: `['robust_range', 'psd_low_power', 'psd_mid_power', 'psd_high_power', 'psd_slope', 'autocorrelation_length_px', 'island_count', 'island_mean_area_px']`.
6. Mean-like descriptors: `['rq', 'ra', 'mean_abs_gradient', 'gradient_std', 'gradient_anisotropy']`.
7. Calibrated_v2 generation nonconstant rate: `1.0`.
8. Generated samples differ from mean-condition samples: `True` by nonconstant/richness proxy, not by exact pixel matching.
9. Uncertainty calibrated: `False`.
10. Group/growth robustness is limited by the small 36-pair supervised set; split counts are recorded in the inventory.

## Scientific Scope

MVP-6 improves RHEED representation learning and temporal condition prediction. It still generates representative AFM-like morphology through the calibrated_v2 AFM prior and does not claim exact pixel-level AFM reconstruction.
