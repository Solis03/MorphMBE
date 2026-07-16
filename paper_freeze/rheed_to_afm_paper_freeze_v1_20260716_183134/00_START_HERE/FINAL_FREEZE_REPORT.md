# Final Freeze Report

- Freeze ID: RHEED_AFM_PAPER_FREEZE_V1_20260716_183134
- Strict quantitative model: top5_median_cross_fitted_ensemble
- Strict metrics: MAE 1.260098, RMSE 1.839310, Spearman 0.428854
- Strict visual method: A3 RHEED-conditioned representative AFM retrieval
- Full-cohort quantitative deployment: 12_FULL_COHORT_DEPLOYMENT/quantitative_model
- Full-cohort visual method: A3_full_cohort
- Training groups: 23
- AFM scans: 116
- Removelist enforcement: passed
- Main figures: 08_PAPER_FIGURES/main
- Supplementary figures: 08_PAPER_FIGURES/supplementary
- Tables: 09_PAPER_TABLES
- Source data: 10_FIGURE_SOURCE_DATA
- Methods draft: 11_SUPPLEMENTARY_MATERIALS/paper_text
- Model diagrams: 04_METHODS/diagrams
- Model card: 12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_card.md
- Claims: 16_CLAIMS_AND_LIMITATIONS/claims_and_limitations.md
- Unseen template: 13_UNSEEN_INFERENCE/unseen_manifest_template.csv
- Unseen command: `python 13_UNSEEN_INFERENCE/predict_unseen_batch.py --bundle-root <freeze_root> --manifest <unseen_manifest.csv> --output-root <prospective_run_dir> --freeze-id RHEED_AFM_PAPER_FREEZE_V1_20260716_183134`
- Freeze predictions: `python 14_PROSPECTIVE_REGISTRY/freeze_predictions.py --prediction-root <prospective_run_dir> --registry 14_PROSPECTIVE_REGISTRY/prospective_prediction_registry.jsonl --freeze-id RHEED_AFM_PAPER_FREEZE_V1_20260716_183134`
- Reveal tool: 14_PROSPECTIVE_REGISTRY/reveal_and_evaluate_afm.py
- Validation: 15_REPRODUCIBILITY/freeze_validation.json
- Archives: {'tar_gz': 'paper_freeze/rheed_to_afm_paper_freeze_v1_20260716_183134.tar.gz', 'tar_gz_sha256': 'paper_freeze/rheed_to_afm_paper_freeze_v1_20260716_183134.tar.gz.sha256', 'zip': 'paper_freeze/rheed_to_afm_paper_freeze_v1_20260716_183134.zip', 'zip_sha256': 'paper_freeze/rheed_to_afm_paper_freeze_v1_20260716_183134.zip.sha256', 'submission_assets_zip': 'paper_freeze/rheed_to_afm_paper_freeze_v1_20260716_183134_submission_assets.zip'}
- Package size MiB: 266.85

Confirmed: unseen samples were not used for training; no original data were modified; no Phase1-7A outputs were overwritten; full-cohort model is not an independent test result; strict benchmark and deployment artifacts are separated.

