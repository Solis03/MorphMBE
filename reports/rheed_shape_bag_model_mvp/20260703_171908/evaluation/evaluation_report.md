# MVP-9 Shape-Bag Model Evaluation

MVP-9 evaluates RHEED shape-bag inputs for AFM morphology descriptor/prototype prediction and representative calibrated_v2 AFM generation. It does not claim exact pixel-level AFM reconstruction.

## Summary

Matched supervised samples: `36`
Best MVP-9 ablation: `stable_features_mlp` with descriptor MSE `1.0005608797073364`
Train-fold mean baseline MSE: `1.0856924057006836`
Shuffled-label control MSE: `5.666114807128906`
Trustworthiness decision: `passes bounded checks`

## Evaluation Questions

1. Shape-bag model beats mean baseline: `True`
2. MVP-6 handcrafted comparison is in `mvp6_vs_mvp9_comparison.csv`.
3. Consensus-vs-stable comparison is in `ablation_metrics_shape_bag.csv`.
4. Frame-bag attention comparison is included when the full suite is run.
5. Exposure-invariance diagnostics are in `exposure_stability_report.md`.
6. Predictable descriptors in this run: `['mean_abs_gradient']`
7. Mean-like descriptors in this run: `['rq', 'ra', 'robust_range', 'psd_slope', 'autocorrelation_length_px', 'gradient_anisotropy', 'island_count', 'gradient_std', 'psd_mid_power', 'psd_high_power']`
8. Brightness-only shortcut MSE: `1.6972459554672241`
9. Shuffled-label beats mean: `False`
10. Generation summary: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/generation_summary_shape_bag.json`

## Limitations

Small validation folds can make negative controls look deceptively strong. Raw 240-feature diagnostics are not the default production input because MVP-8 found thresholded shape/count features remain exposure-sensitive.
