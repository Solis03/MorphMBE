# Phase 1 RHEED Video to AFM Report

## Cohort

- Frozen selected samples: 27
- Modeling-eligible samples after removelist: 25
- Primary 1 x 1 um cohort: 23
- Exploratory best-available cohort: 25
- Removelist conflicts excluded from modeling: ['6023', '6087']
- Rq range/median: 1.417 to 10.321 nm, median 3.126 nm

## Baseline Metrics

| cohort | model_name | N | MAE_nm | median_absolute_error_nm | RMSE_nm | R2 | Spearman_rho | Kendall_tau | pairwise_concordance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_1um | B0_training_fold_median | 23 | 1.59066 | 1.14054 | 2.44515 | -0.100004 | -0.884652 | -0.751809 | 0.217391 |
| primary_1um | B1_keyframe_ElasticNet | 23 | 1.65191 | 1.23499 | 2.46534 | -0.118244 | -1 | -1 | 0 |
| primary_1um | B2_temporal_ElasticNet | 23 | 1.6886 | 1.23499 | 2.50103 | -0.150855 | -1 | -1 | 0 |
| primary_1um | B1_keyframe_Ridge | 23 | 1.72608 | 1.30411 | 2.51877 | -0.167237 | -0.768775 | -0.581028 | 0.209486 |
| primary_1um | B1_keyframe_PLSRegression | 23 | 1.8614 | 1.21335 | 2.61782 | -0.260839 | -0.534585 | -0.399209 | 0.300395 |
| primary_1um | B2_temporal_PLSRegression | 23 | 2.00083 | 1.42801 | 2.79792 | -0.440301 | -0.850791 | -0.660079 | 0.16996 |
| primary_1um | B3_knn_temporal_retrieval | 23 | 2.0267 | 1.55757 | 2.81748 | -0.46051 | -0.547431 | -0.359684 | 0.320158 |
| primary_1um | B2_temporal_Ridge | 23 | 2.08086 | 1.43412 | 2.90233 | -0.549803 | -0.722332 | -0.525692 | 0.237154 |

## Leakage Audit

- KNN neighbor rows: 135
- KNN neighbor leakage-free: True

## Key Risks

- Selected sample(s) 6023, 6087 are present in the canonical removelist and were excluded from modeling.
- Growth/video stage and material are inferred from local metadata/file names where explicit fields are absent.
- The primary cohort is small; OOF metrics should be treated as screening signals, not definitive model evidence.
- Best-available exploratory scans are reported separately and not mixed into the primary baseline.

## Outputs

- `outputs/rheed_video_afm_story/phase1/modeling_manifest.csv`
- `outputs/rheed_video_afm_story/phase1/modeling_manifest.parquet`
- `outputs/rheed_video_afm_story/phase1/afm_scan_audit.csv`
- `outputs/rheed_video_afm_story/phase1/rheed_quality_metrics.csv`
- `outputs/rheed_video_afm_story/phase1/oof_predictions.csv`
- `outputs/rheed_video_afm_story/phase1/baseline_metrics.csv`
- `outputs/rheed_video_afm_story/phase1/baseline_neighbor_audit.csv`
- `reports/rheed_video_afm_story/phase1/repo_audit.md`
- `reports/rheed_video_afm_story/phase1/phase1_report.md`
