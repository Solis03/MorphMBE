# Second-Order AFM Controlled Rerun Report

- Variant: `afm_second_order_y2_v1`
- Second-order output count: 260
- Valid primary 1 x 1 um scan count: 116
- Primary growth groups: 23
- Mapping complete: True
- Per-sample Rq change range: -1.823 to 0.000734 nm
- First/second Rq Spearman: 0.92
- Rq rank reorder count: 21
- Representative scan changed count: 10
- Second-order high-confidence metrics: `{'model_id': 'R4_auto_iso_dino_residual', 'N': 16, 'MAE': 2.086943190957678, 'median_AE': 1.611720229581427, 'RMSE': 2.847982506196332, 'R2': -0.5740448432715513, 'Spearman': -0.2176470588235294, 'Kendall_tau': -0.1833333333333333, 'pairwise_concordance': 0.4083333333333333, 'low_high_balanced_accuracy': 0.3, 'high_rq_sensitivity': 0.2, 'high_rq_specificity': 0.5454545454545454, 'coverage': 0.6956521739130435, 'abstained_count': 7}`
- Validation passed: True
- Dashboard: `reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase4b_visualization/results_dashboard.html`
- Comparison figures: `['reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_A_afm_preprocessing_effect.png', 'reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_B_sample_level_rq_targets.png', 'reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_C_rq_model_performance.png', 'reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_D_oof_predictions.png', 'reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_E_s1_s4_descriptor_metrics.png', 'reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/comparison/comparison_F_visual_output_shift.png']`

## Interpretation

Second-order AFM preprocessing changes the target definition and downstream model behavior. The run does not by itself prove scientific superiority over first-order preprocessing.

Can claim: controlled AFM-preprocessing ablation with fixed RHEED inputs, fixed cohort, fixed removelist, fixed model families, and second-order target-dependent retraining.

Cannot claim: exact AFM reconstruction or that second-order correction is scientifically superior without additional QC/repeatability evidence.
