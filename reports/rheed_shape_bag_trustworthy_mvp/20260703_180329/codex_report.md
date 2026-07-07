# MVP-10 Descriptor-Wise Trustworthy Shape-Bag Validation Report

Generated: 2026-07-03T18:03:29+00:00

MVP-10 evaluates whether RHEED shape-bag features provide descriptor-wise trustworthy AFM morphology prediction under strict validation. It does not claim exact pixel-level AFM reconstruction.

## Git Status

Before MVP-10 implementation/run:

```text
?? reports/rheed_frame_selection_mvp/
?? reports/rheed_shape_bag_input_mvp/
?? reports/rheed_shape_bag_model_mvp/
?? src/rheed2morph/generative/sample_shape_bag_calibrated_v2.py
?? src/rheed2morph/rheed/build_shape_bag_inputs.py
?? src/rheed2morph/rheed/build_shape_bag_supervised_dataset.py
?? src/rheed2morph/rheed/evaluate_shape_bag_model.py
?? src/rheed2morph/rheed/exposure_invariance_training.py
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/models/
?? src/rheed2morph/rheed/predict_shape_bag_conditions.py
?? src/rheed2morph/rheed/pretrain_shape_bag_invariance.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset.py
?? src/rheed2morph/rheed/run_shape_bag_ablation.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? src/rheed2morph/rheed/shape_preprocessing.py
?? src/rheed2morph/rheed/spot_streak_geometry.py
?? src/rheed2morph/rheed/train_shape_bag_morphology_predictor.py
?? tests/test_rheed_frame_selection.py
?? tests/test_rheed_shape_bag_input.py
?? tests/test_rheed_shape_bag_model.py
```

After MVP-10 implementation/run:

```text
?? reports/rheed_frame_selection_mvp/
?? reports/rheed_shape_bag_input_mvp/
?? reports/rheed_shape_bag_model_mvp/
?? reports/rheed_shape_bag_trustworthy_mvp/
?? src/rheed2morph/generative/sample_shape_bag_calibrated_v2.py
?? src/rheed2morph/generative/sample_shape_bag_oof_calibrated_v2.py
?? src/rheed2morph/rheed/build_shape_bag_inputs.py
?? src/rheed2morph/rheed/build_shape_bag_supervised_dataset.py
?? src/rheed2morph/rheed/compare_manual_vs_auto_shape_bags.py
?? src/rheed2morph/rheed/evaluate_shape_bag_model.py
?? src/rheed2morph/rheed/export_shape_bag_oof_predictions.py
?? src/rheed2morph/rheed/exposure_invariance_training.py
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/generate_shape_bag_evidence_package.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/models/
?? src/rheed2morph/rheed/predict_shape_bag_conditions.py
?? src/rheed2morph/rheed/pretrain_shape_bag_invariance.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset.py
?? src/rheed2morph/rheed/run_shape_bag_ablation.py
?? src/rheed2morph/rheed/run_shape_bag_strict_descriptor_cv.py
?? src/rheed2morph/rheed/select_production_shape_bag_model.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? src/rheed2morph/rheed/shape_bag_feature_importance.py
?? src/rheed2morph/rheed/shape_bag_negative_controls.py
?? src/rheed2morph/rheed/shape_bag_trustworthy_utils.py
?? src/rheed2morph/rheed/shape_preprocessing.py
?? src/rheed2morph/rheed/spot_streak_geometry.py
?? src/rheed2morph/rheed/train_shape_bag_morphology_predictor.py
?? tests/test_rheed_frame_selection.py
?? tests/test_rheed_shape_bag_input.py
?? tests/test_rheed_shape_bag_model.py
?? tests/test_rheed_shape_bag_trustworthy.py
```

Pre-existing MVP-8/MVP-9 untracked files were left in place. MVP-10 additions are the trustworthy validation scripts, OOF calibrated_v2 wrapper, test file, and `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/`.

## Environment

| item | value |
| --- | --- |
| python | 3.12.3 |
| torch | 2.12.0+cu130 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5090 |

## Files Created Or Modified

- `src/rheed2morph/rheed/shape_bag_trustworthy_utils.py`
- `src/rheed2morph/rheed/run_shape_bag_strict_descriptor_cv.py`
- `src/rheed2morph/rheed/shape_bag_negative_controls.py`
- `src/rheed2morph/rheed/shape_bag_feature_importance.py`
- `src/rheed2morph/rheed/compare_manual_vs_auto_shape_bags.py`
- `src/rheed2morph/rheed/select_production_shape_bag_model.py`
- `src/rheed2morph/rheed/export_shape_bag_oof_predictions.py`
- `src/rheed2morph/rheed/generate_shape_bag_evidence_package.py`
- `src/rheed2morph/generative/sample_shape_bag_oof_calibrated_v2.py`
- `tests/test_rheed_shape_bag_trustworthy.py`
- `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/`

## Commands Run

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.run_shape_bag_strict_descriptor_cv --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv --fold-mode original_mvp9 --models mean,ridge,stable_features_mlp --feature-sets stable36,brightness_only_diagnostic --bootstrap 20 --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.shape_bag_negative_controls --cv-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/negative_controls --n-permutations 10 --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.shape_bag_feature_importance --cv-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/feature_importance --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.compare_manual_vs_auto_shape_bags --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/manual_vs_auto --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_production_shape_bag_model --cv-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv --negative-control-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/negative_controls --feature-importance-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/feature_importance --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_model_selection

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.export_shape_bag_oof_predictions --cv-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv --production-selection reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_model_selection/selected_model_config.json --descriptor-policy reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_model_selection/unsupported_descriptor_policy.json --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_predictions

PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_shape_bag_oof_calibrated_v2 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --predicted-condition-table reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_predictions/predicted_condition_table_oof_production.csv --shape-bag-index reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv --num-samples-per-condition 2 --keep-top-k 2 --ddim-steps 10 --guidance-scale 1.5 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 2 --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/trustworthy_calibrated_v2_generation --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.generate_shape_bag_evidence_package --mvp10-root reports/rheed_shape_bag_trustworthy_mvp/20260703_180329 --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/evidence_package

PYTHONPATH=src .venv/bin/python -m compileall -q src/rheed2morph/rheed/shape_bag_trustworthy_utils.py src/rheed2morph/rheed/run_shape_bag_strict_descriptor_cv.py src/rheed2morph/rheed/shape_bag_negative_controls.py src/rheed2morph/rheed/shape_bag_feature_importance.py src/rheed2morph/rheed/compare_manual_vs_auto_shape_bags.py src/rheed2morph/rheed/select_production_shape_bag_model.py src/rheed2morph/rheed/export_shape_bag_oof_predictions.py src/rheed2morph/rheed/generate_shape_bag_evidence_package.py src/rheed2morph/generative/sample_shape_bag_oof_calibrated_v2.py tests/test_rheed_shape_bag_trustworthy.py
rg -n -i "knn|nearestneighbors|nearest_neighbors|kneighbors" src/rheed2morph/rheed/shape_bag_trustworthy_utils.py src/rheed2morph/rheed/run_shape_bag_strict_descriptor_cv.py src/rheed2morph/rheed/shape_bag_negative_controls.py src/rheed2morph/rheed/shape_bag_feature_importance.py src/rheed2morph/rheed/compare_manual_vs_auto_shape_bags.py src/rheed2morph/rheed/select_production_shape_bag_model.py src/rheed2morph/rheed/export_shape_bag_oof_predictions.py src/rheed2morph/rheed/generate_shape_bag_evidence_package.py src/rheed2morph/generative/sample_shape_bag_oof_calibrated_v2.py || true
PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_trustworthy.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The run is the requested smoke-scale MVP-10 pass, not the full 200-permutation/repeated-CV pass.

## MVP-9 Summary

- MVP-9 matched supervised samples: 36
- MVP-9 best bounded ablation: `stable_features_mlp`
- MVP-9 best descriptor MSE: 1.0005608797073364
- MVP-9 mean baseline MSE: 1.0856924057006836
- MVP-9 negative controls were attempted and did not beat mean in that bounded ablation.

MVP-10 re-evaluated this signal descriptor-wise in raw descriptor space with train-fold-only scaling/imputation and stricter control reporting.

## Descriptor-Wise CV

- CV root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv/`
- Fold design: `original_mvp9`
- Train rows: original MVP-9 train split
- Validation rows: original MVP-9 val/test rows
- Samples: 36
- Held-out rows: 11
- Descriptors evaluated: 13
- Models: `mean`, `ridge`, `stable_features_mlp`
- Feature sets: `stable36`, `brightness_only_diagnostic`
- Bootstrap: 20

Outcome:

- SUPPORTED descriptors: none
- WEAK descriptors: none
- NOT_SUPPORTED descriptors: all 13
- UNRELIABLE descriptors after negative controls: production policy rejects all RHEED descriptor predictions

Every best real model row in `descriptor_predictability_table.csv` failed to beat the train-fold mean baseline on the held-out fold. Example outcomes:

- `rq`: best MSE 15.7774 vs mean 13.7630, R2 -0.1464
- `ra`: best MSE 10.5609 vs mean 9.4561, R2 -0.1168
- `robust_range`: best MSE 314.6325 vs mean 264.2867, R2 -0.1905
- `mean_abs_gradient`: best MSE 2.5716 vs mean 0.3955, R2 -5.5013
- `psd_slope`: best MSE 0.4861 vs mean 0.0845, R2 -4.7520

The MVP-9 apparent aggregate improvement does not survive this descriptor-wise raw-target smoke validation.

## Negative Controls

- Negative-control root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/negative_controls/`
- Controls attempted:
  - shuffled labels within train folds
  - shuffled labels globally
  - shuffled shape bags across groups
  - random Gaussian features
  - brightness-only diagnostic
  - exposure-only diagnostic
  - raw 240 feature diagnostic
  - forbidden ID/path diagnostic

Summary:

- `negative_controls_pass`: false
- `not_trustworthy_flags`: 23

Several controls beat the weak real stable-feature models even when they did not beat the mean baseline. This indicates the real models are not scientifically trustworthy under this strict descriptor-wise setup.

## Feature Importance

- Feature-importance root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/feature_importance/`
- Permutation importance rows: 1080
- Bar/elongation correlation rows: 4

Elongated/bar-like exploratory correlations were weak:

- `std_bar_like_score` vs `island_mean_area_px`: Spearman 0.114
- `std_bar_like_score` vs `island_count`: Spearman 0.081
- `weighted_median_mean_aspect_ratio` vs `island_mean_area_px`: Spearman 0.087
- `std_mean_aspect_ratio` vs `island_mean_area_px`: Spearman 0.071

Because no descriptor passed strict CV, the feature-importance results are exploratory only and do not support a production claim.

## Manual-vs-Auto

- Manual-vs-auto root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/manual_vs_auto/`
- Manual rows: 0
- Paired manual rows: 0
- Status: `pending_manual_selection`

The tool wrote `prioritized_manual_review_samples.csv` and `manual_review_guide.md`. No quantitative manual-vs-auto comparison was possible.

## Production Model Selection

- Selection root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_model_selection/`
- Selected descriptors: none
- Selected feature set: none
- Scientifically trustworthy RHEED descriptor model: no
- Unsupported descriptor policy: fill train mean

All descriptors are marked unsupported for production. `predicted_condition_table_oof_production.csv` contains policy-adjusted condition values, but every descriptor entry is filled by train mean rather than predicted by RHEED.

## Trustworthy Generation

- Generation root: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/trustworthy_calibrated_v2_generation/`
- Grid: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/trustworthy_calibrated_v2_generation/trustworthy_shape_bag_calibrated_v2_grid.png`
- Summary: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/trustworthy_calibrated_v2_generation/trustworthy_generation_summary.json`
- Tiny real diffusion run: 2 conditions, 2 samples per condition, 10 DDIM steps
- Generated nonconstant rate: 1.0
- Mean generated normalized std: 0.6313895523548126
- Descriptor policy counts: predicted by RHEED 0, filled by train mean 143

The generation artifact is representative calibrated_v2 output from policy-adjusted mean-filled descriptors. It is not predictive evidence of RHEED-controlled AFM morphology.

## Claim Support Matrix

Primary matrix: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/evidence_package/claim_support_matrix.csv`

Key entries:

- End-to-end pipeline works: SUPPORTED
- AFM prior generates representative AFM: SUPPORTED
- Height calibration improves roughness/range: SUPPORTED
- RHEED shape-bag features beat mean baseline: NOT_SUPPORTED
- RHEED shape-bag beats brightness-only: UNRELIABLE
- Negative controls pass: UNRELIABLE
- Elongated/bar-like RHEED features correlate with AFM descriptors: WEAK
- Generated AFM differs meaningfully from mean-condition: WEAK
- Exact pixel-level AFM reconstruction is possible: NOT_SUPPORTED

## Output Locations

- Strict CV: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv/cv_metrics_summary.csv`
- Descriptor predictability: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv/descriptor_predictability_table.csv`
- Negative-control report: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/negative_controls/negative_control_report.md`
- Feature importance: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/feature_importance/feature_importance_summary.csv`
- Production selection: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_model_selection/production_model_selection_report.md`
- Production OOF table: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/production_predictions/predicted_condition_table_oof_production.csv`
- Trustworthy generation summary: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/trustworthy_calibrated_v2_generation/trustworthy_generation_summary.json`
- Evidence report: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/evidence_package/evidence_package_report.md`
- Claim matrix: `reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/evidence_package/claim_support_matrix.csv`

## Verification

- Compile check: passed
- New-path retrieval API source scan: no hits
- `PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_trustworthy.py`: 3 tests OK
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`: 97 tests OK

Warnings were constant-correlation/convergence warnings from small synthetic or smoke folds; they did not fail tests.

## Known Limitations

- This was a smoke run with one original MVP-9 held-out split, not repeated group CV.
- Bootstrap was 20, not 1000.
- Negative-control permutations were 10, not 200.
- Manual frame selections were unavailable for paired samples.
- The production policy currently selects no RHEED descriptors.
- Generation uses policy-filled descriptors and should be viewed only as representative calibrated_v2 output.

## Recommended Next Command

Run repeated group CV after improving data volume/manual selection:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.run_shape_bag_strict_descriptor_cv \
  --mvp8-root reports/rheed_shape_bag_input_mvp/20260703_165110 \
  --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 \
  --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json \
  --out reports/rheed_shape_bag_trustworthy_mvp/20260703_180329/strict_descriptor_cv_repeated_group \
  --fold-mode repeated_group_kfold --n-splits 5 --n-repeats 50 \
  --models mean,median,ridge,elasticnet,random_forest,gradient_boosting \
  --feature-sets stable36,stable36_plus_consensus_summary,brightness_only_diagnostic,raw240_diagnostic \
  --bootstrap 1000 --seed 42
```
