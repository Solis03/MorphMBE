# Phase 5B Report

Regime-aware maximal-training cross-fitted prediction on the second-order AFM target.

## Split Audit
- Current R4 fold membership is maximal N-1 LOOCV: False.
- Each fold actual training group counts: [22].
- Split bug found: True.
- Current second-order R4 target-alignment bug found: True.
- 6095 fold contains 6099: True.
- 6099 fold contains 6095: True.

## Case Studies
- 6095: top fused neighbors ["6094", "6063", "6081", "6056", "6090"]; high-regime support 7; new prediction 4.552; new source 6094; old source 6101.
- 6099: top fused neighbors ["6085", "6081", "6095", "6056", "6048"]; high-regime support 7; new prediction 2.826; new source 6085; old source 6081.

## Metrics
- Recorded current R4 all-sample metrics: {'N': 23, 'MAE': 2.095930999072786, 'median_AE': 1.3832835270060804, 'RMSE': 2.859877331770969, 'R2': -0.42058545954546167, 'Spearman': -0.12055335968379448, 'Kendall': -0.05928853754940712, 'pairwise_concordance': 0.47035573122529645, 'low_high_balanced_accuracy': 0.5, 'high_rq_sensitivity': 0.5, 'high_rq_specificity': 0.5333333333333333, 'model_id': 'L0_current_R4_recorded', 'subset': 'all_samples', 'coverage': 1.0}.
- Reconstructed current R4 all-sample metrics: {'N': 23, 'MAE': 1.4162713890290355, 'median_AE': 1.0200499784967267, 'RMSE': 2.0405274087453917, 'R2': 0.27680166159527575, 'Spearman': 0.37055335968379455, 'Kendall': 0.23320158102766797, 'pairwise_concordance': 0.616600790513834, 'low_high_balanced_accuracy': 0.75, 'high_rq_sensitivity': 0.75, 'high_rq_specificity': 0.8, 'model_id': 'L0_current_R4_reconstructed', 'subset': 'all_samples', 'coverage': 1.0}.
- L6 cross-fitted bootstrap all-sample metrics: {'N': 23, 'MAE': 2.524004277975663, 'median_AE': 1.798634469509125, 'RMSE': 3.249286832894604, 'R2': -0.8337860119688236, 'Spearman': -0.23589764262983018, 'Kendall': -0.1672805744364568, 'pairwise_concordance': 0.4209486166007905, 'low_high_balanced_accuracy': 0.4375, 'high_rq_sensitivity': 0.375, 'high_rq_specificity': 0.6, 'model_id': 'L6_cross_fitted_bootstrap_median', 'subset': 'all_samples', 'coverage': 1.0}.
- High-support metrics: {'N': 5, 'MAE': 2.3196822881698607, 'median_AE': 2.555906593799591, 'RMSE': 2.8653382814412187, 'R2': -8.899203862171872, 'Spearman': -0.7905694150420948, 'Kendall': -0.6708203932499368, 'pairwise_concordance': 0.2, 'low_high_balanced_accuracy': 0.25, 'high_rq_sensitivity': 0.5, 'high_rq_specificity': 0.3333333333333333, 'model_id': 'L6_cross_fitted_bootstrap_median', 'subset': 'high_support', 'coverage': 0.21739130434782608}.
- Regime macro-F1: 0.24264705882352944.
- Abstained sample IDs: ['6029', '6047', '6056', '6057', '6062', '6070', '6078', '6081', '6084', '6090', '6094', '6095', '6099', '6101'].

## Retrieval
- Old retrieval regime agreement: 0.21739130434782608.
- Regime-gated retrieval agreement: 1.0.

## Controls And Deployment
- First-order control: [{'N': 23, 'MAE': 2.322070132945851, 'median_AE': 1.4849228389159184, 'RMSE': 3.0382857848973512, 'R2': -0.6983959158550057, 'Spearman': 0.06169303878397316, 'Kendall': 0.024659431554869705, 'pairwise_concordance': 0.5118577075098815, 'low_high_balanced_accuracy': 0.5625, 'high_rq_sensitivity': 0.5, 'high_rq_specificity': 0.7333333333333333, 'target_variant': 'first_order_control', 'model_id': 'L3_regime_gated_kNN_same_params'}, {'N': 23, 'MAE': 1.6374729088992606, 'median_AE': 1.376578444672611, 'RMSE': 2.0455099213426653, 'R2': 0.23018765031965582, 'Spearman': 0.28952569169960474, 'Kendall': 0.20948616600790515, 'pairwise_concordance': 0.6047430830039525, 'low_high_balanced_accuracy': 0.5, 'high_rq_sensitivity': 0.5, 'high_rq_specificity': 0.6666666666666666, 'target_variant': 'first_order_control', 'model_id': 'current_R4_first_order'}, {'N': 23, 'MAE': 2.3823641823685686, 'median_AE': 1.7973990440368652, 'RMSE': 3.0728762653965616, 'R2': -0.6400711921787032, 'Spearman': -0.07018612759491237, 'Kendall': -0.05354329962222025, 'pairwise_concordance': 0.4743083003952569, 'low_high_balanced_accuracy': 0.4375, 'high_rq_sensitivity': 0.375, 'high_rq_specificity': 0.6, 'target_variant': 'second_order_y2', 'model_id': 'L3_regime_gated_kNN'}].
- Deployment model: outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/deployment_model.
- Full-cohort calibration is explicitly isolated as in-sample only and is not mixed into OOF metrics.
- Go decisions: {'Go-SPLIT': True, 'Go-LOCAL': False, 'Go-SUPPORT': False, 'Go-RETRIEVAL': True, 'Go-DEPLOY': True}.
- Raw/old hash validation: {'removelist_hash_ok': True, 'hashes': {'outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/targets/second_order_modeling_manifest.csv': '9005bf2655673aebce91370066b7bfdd6993bed054a7995a9ed07c10ce86b34a', 'outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/targets/second_order_sample_targets.csv': 'd866cc3eb81dda506616e115f2f7554b28bc75673645fc02de1bc9bd06a855b2', 'outputs/rheed_video_afm_story/phase2a/embedding_registry.csv': '116a85d17f58d6ba87c95843aebbaabe47d0c132853c2cd43684d7185de0995c', 'outputs/rheed_video_afm_story/phase2a/embeddings/dino_vits14__keyframe_1__raw_luminance.npz': 'd80ac6eaa6a6feeeba83ec1abf226e07650f997fe772448280485fecc0fabca0', 'outputs/rheed_video_afm_story/phase2a/embeddings/r3d_18__selected_16__raw_luminance.npz': '4fdb65982e8a7c08674bc96383af47ca9ad4297587e46b5372495b7429356475', 'outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase4a/rheed_physics_features.csv': '73e00798602f37584bf52b272fc6b32355ea1da7113104b7feb078abc64034be', 'removelist.txt': '8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b', 'data/processed_afm/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy': '1700a83202069e38876a966fa78432025797b8da102903dbac3256a209370234', 'data/afm_second_order/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy': 'a702e45eae8d4690cf9b586fcbc100a1ad44885814deb98841aff47e89f3550a'}}.

Cannot claim: exact reconstruction, all-data test prediction, or independent-test performance.
