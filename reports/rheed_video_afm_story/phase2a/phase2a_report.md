# Phase 2A Report

- Primary cohort N: 23
- FPS range: 30-30
- 16-frame duration range: 0.5-0.5 s
- All-primary variants: causal_4, causal_8, centered_4, centered_8, fixed_time_16_1s, fixed_time_8_1s, keyframe_1, selected_16
- Loaded encoders: ['resnet18', 'r3d_18', 'dino_vits14']
- Frozen thresholds: q33=2.551, q67=3.899
- Best regression: {'embedding_id': 'dino_vits14__keyframe_1__raw_luminance', 'encoder': 'dino_vits14', 'clip_variant': 'keyframe_1', 'preprocessing': 'raw_luminance', 'head': 'PLSRegression', 'N': 23, 'MAE_nm': 1.4990284365554971, 'median_AE_nm': 1.2924703333233922, 'RMSE_nm': 1.9648938488205012, 'R2': 0.28967045386232704, 'Spearman': 0.37055335968379455, 'Kendall_tau': 0.25691699604743085, 'pairwise_concordance': 0.6284584980237155}
- Best ranking: {'embedding_id': 'r3d_18__selected_16__raw_luminance', 'encoder': 'r3d_18', 'clip_variant': 'selected_16', 'preprocessing': 'raw_luminance', 'N': 23, 'Spearman': 0.15611673551866287, 'Kendall_tau': 0.09949174844301832, 'pairwise_concordance': 0.5474308300395256, 'rank_MAE': 0.23770290611180792, 'rank_derived_Rq_MAE_nm': 1.6200815591674995}
- Best extreme: {'embedding_id': 'r3d_18__selected_16__raw_luminance', 'task': 'extreme_binary', 'N': 16, 'balanced_accuracy': 0.625, 'macro_F1': 0.625, 'AUROC': 0.46875, 'AUPRC': 0.5120535714285714, 'high_sensitivity': 0.625, 'high_specificity': 0.625, 'confusion_matrix': '[[5, 3], [3, 5]]'}
- Best metadata/control: {'control_model': 'M2_rheed_embedding_only', 'N': 23, 'MAE_nm': 1.581804546224523, 'median_AE_nm': 1.489383660456154, 'RMSE_nm': 1.9418830565992855, 'R2': 0.30621031564691836, 'Spearman': 0.2816205533596838, 'Kendall_tau': 0.2015810276679842, 'pairwise_concordance': 0.6007905138339921}
- Support distribution: {'high': 23}
- Prediction interval coverage: {'coverage_80': 0.782608695652174, 'coverage_90': 0.8695652173913043, 'mean_90_width': 6.110922704933838, 'median_90_width': 6.095162754476098}
- Go decisions: {'Go-A': False, 'Go-B': False, 'Go-C': False, 'Go-D': False}

## Notes

- No AFM decoder was trained and no AFM was generated.
- High-support labels use target-blind domain/support quantities only.
- UMAP/PCA visualization coordinates are descriptive only and are not predictive features.
