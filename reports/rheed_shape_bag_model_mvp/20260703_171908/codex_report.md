# MVP-9 RHEED Shape-Bag-To-AFM Morphology Report

Generated: 2026-07-03T17:19:08+00:00

MVP-9 evaluates RHEED shape-bag inputs for AFM morphology descriptor/prototype prediction and representative calibrated_v2 AFM generation. It does not claim exact pixel-level AFM reconstruction.

## Git Status

Before MVP-9 implementation/run:

```text
?? reports/rheed_frame_selection_mvp/
?? reports/rheed_shape_bag_input_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs.py
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/models/
?? src/rheed2morph/rheed/pretrain_shape_bag_invariance.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? src/rheed2morph/rheed/shape_preprocessing.py
?? src/rheed2morph/rheed/spot_streak_geometry.py
?? tests/test_rheed_frame_selection.py
?? tests/test_rheed_shape_bag_input.py
```

After MVP-9 implementation/run:

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

The pre-existing MVP-8 untracked files were left in place. MVP-9 additions are the new supervised dataset, predictor, trainer, ablation, prediction, evaluation, generation wrapper, exposure-invariance helper, test file, and `reports/rheed_shape_bag_model_mvp/20260703_171908/`.

## Environment

| item | value |
| --- | --- |
| python | 3.12.3 |
| torch | 2.12.0+cu130 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5090 |

## Files Created Or Modified

- `src/rheed2morph/rheed/build_shape_bag_supervised_dataset.py`
- `src/rheed2morph/rheed/models/shape_bag_morphology_predictor.py`
- `src/rheed2morph/rheed/train_shape_bag_morphology_predictor.py`
- `src/rheed2morph/rheed/run_shape_bag_ablation.py`
- `src/rheed2morph/rheed/predict_shape_bag_conditions.py`
- `src/rheed2morph/rheed/evaluate_shape_bag_model.py`
- `src/rheed2morph/rheed/exposure_invariance_training.py`
- `src/rheed2morph/generative/sample_shape_bag_calibrated_v2.py`
- `tests/test_rheed_shape_bag_model.py`
- `reports/rheed_shape_bag_model_mvp/20260703_171908/`

## Commands Run

```bash
git status --short
PYTHONPATH=src .venv/bin/python - <<'PY'
import sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_supervised_dataset --shape-bag-manifest reports/rheed_shape_bag_input_mvp/20260703_165110/rheed_shape_bag_manifest.csv --shape-features reports/rheed_shape_bag_input_mvp/20260703_165110/global_sample_shape_features.csv --stable-feature-list reports/rheed_shape_bag_input_mvp/20260703_165110/default_training_feature_names.txt --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --out reports/rheed_shape_bag_model_mvp/20260703_171908/data --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.train_shape_bag_morphology_predictor --supervised-index reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv --target-table reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_conditions_shape_bag.csv --folds reports/rheed_shape_bag_model_mvp/20260703_171908/data/strict_fold_assignments.csv --feature-schema reports/rheed_shape_bag_model_mvp/20260703_171908/data/feature_schema_shape_bag.json --target-schema reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_schema_shape_bag.json --out reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_predictor --model shape_bag_fusion --epochs 20 --batch-size 8 --lr 1e-4 --amp --predict-uncertainty true --exposure-invariance-weight 0.1 --loss heteroscedastic --fold-id original_split --use-frames false --use-consensus true --use-stable-features true --model-image-size 64 --hidden-dim 64 --embedding-dim 128 --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.run_shape_bag_ablation --data-root reports/rheed_shape_bag_model_mvp/20260703_171908/data --out reports/rheed_shape_bag_model_mvp/20260703_171908/ablations --epochs 5 --batch-size 8 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.run_shape_bag_ablation --data-root reports/rheed_shape_bag_model_mvp/20260703_171908/data --out reports/rheed_shape_bag_model_mvp/20260703_171908/ablations --epochs 5 --batch-size 8 --amp --full-suite --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.predict_shape_bag_conditions --checkpoint reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_predictor/checkpoints/best.pt --supervised-index reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv --target-schema reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_schema_shape_bag.json --split val --batch-size 8 --model-image-size 64 --device auto --out reports/rheed_shape_bag_model_mvp/20260703_171908/predicted_conditions

PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_shape_bag_calibrated_v2 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --predicted-condition-table reports/rheed_shape_bag_model_mvp/20260703_171908/predicted_conditions/predicted_condition_table_val.csv --shape-bag-index reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv --primary-generator calibrated_v2 --num-samples-per-condition 2 --keep-top-k 2 --ddim-steps 10 --guidance-scale 1.5 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 2 --out reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation --seed 42

PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.evaluate_shape_bag_model --mvp9-root reports/rheed_shape_bag_model_mvp/20260703_171908 --mvp6-root reports/rheed_ssl_temporal_mvp/20260703_072054 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --out reports/rheed_shape_bag_model_mvp/20260703_171908/evaluation

PYTHONPATH=src .venv/bin/python -m compileall -q src/rheed2morph/rheed src/rheed2morph/generative/sample_shape_bag_calibrated_v2.py tests/test_rheed_shape_bag_model.py
rg -n -i "knn|nearestneighbors|nearest_neighbors|kneighbors" src/rheed2morph/rheed/build_shape_bag_supervised_dataset.py src/rheed2morph/rheed/models/shape_bag_morphology_predictor.py src/rheed2morph/rheed/train_shape_bag_morphology_predictor.py src/rheed2morph/rheed/run_shape_bag_ablation.py src/rheed2morph/rheed/predict_shape_bag_conditions.py src/rheed2morph/rheed/evaluate_shape_bag_model.py src/rheed2morph/rheed/exposure_invariance_training.py src/rheed2morph/generative/sample_shape_bag_calibrated_v2.py || true
PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_model.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

The first `--full-suite` ablation attempt exposed an AMP fp16 mask overflow in frame attention. The attention mask was patched to use a dtype-safe fill value, then the full-suite ablation and tests were rerun successfully.

## MVP-8 Dependency Summary

- MVP-8 root: `reports/rheed_shape_bag_input_mvp/20260703_165110/`
- Shape-bag samples processed: 62
- Failures: 0
- Total selected frames: 992
- Per-sample shape-bag tensor: `frames [16, 6, 256, 256]`
- Per-sample consensus tensor: `consensus_maps [6, 256, 256]`
- Raw aggregate shape features available: 240
- MVP-8 stable base feature names: 36
- MVP-9 stable aggregate feature columns used by default: 180
- Exposure audit: median raw brightness CV `0.228966`, median shape feature CV `0.320492`

Raw component/count features are not the default production input because MVP-8 showed thresholded shape/count features can remain exposure-sensitive. MVP-9 therefore defaults to the MVP-8 stable feature list plus consensus maps, with raw 240 features only in diagnostic ablation.

## Supervised Pairing Inventory

- Output data root: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/`
- Shape-bag samples available: 62
- Matched supervised shape-bag/AFM pairs: 36
- Unmatched shape-bags: 26
- Unmatched pair rows: 0
- Original MVP-6 split counts: train 25, val 5, test 6
- Strict group-fold counts: fold0 8, fold1 7, fold2 7, fold3 7, fold4 7
- Target columns: `rq`, `ra`, `robust_range`, `psd_slope`, `autocorrelation_length_px`, `gradient_anisotropy`, `island_count`, `island_mean_area_px`, `mean_abs_gradient`, `gradient_std`, `psd_low_power`, `psd_mid_power`, `psd_high_power`

Each `shape_bag.npz` is treated as one supervised sample. Frames from a sample are never split as independent labeled examples.

## Model Architecture

- Frame branch: per-frame CNN with GroupNorm-based `ConvBlock`, learned attention over `K` frames, frame weights, frame dropout, and channel dropout. It supports variable frame masks and AMP-safe attention masking.
- Consensus branch: small CNN over the six MVP-8 consensus maps. This is enabled in the bounded production-style training run.
- Stable feature branch: LayerNorm + MLP over the 180 stable aggregate feature columns derived from the 36 MVP-8 stable base names.
- Metadata branch: implemented but disabled by default; sample/path/group identifiers are never model inputs.
- Fusion: concatenated enabled-branch embeddings -> LayerNorm MLP -> descriptor mean head, optional descriptor logvar head, optional prototype head.
- Loss: descriptor MSE or heteroscedastic NLL, optional prototype CE, exposure-invariance consistency, attention entropy regularization, and optimizer weight decay.

## Training Metrics

Bounded production-style model:

- Command: 20 epochs requested, original MVP-6 split, stable features + consensus maps, no frame branch, AMP, heteroscedastic loss, exposure-invariance weight 0.1.
- Best checkpoint: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_predictor/checkpoints/best.pt`
- Best epoch: 1
- Validation/test-side rows: 11
- Descriptor MSE: 1.052808
- Mean baseline MSE: 1.085692
- Descriptor MAE: 0.790434
- Descriptor RMSE: 1.026065
- Descriptor R2: -0.311409
- Descriptor Spearman: -0.059559
- Prototype accuracy: 0.454545
- Prototype macro-F1: 0.15625

This bounded checkpoint slightly beats the train-fold mean MSE but still has negative aggregate R2, so descriptor-level interpretation should remain conservative.

## Ablation Results

Bounded full-suite ablations used 5 epochs each, not the requested 50-epoch full scientific pass.

| variant | descriptor_mse | note |
| --- | ---: | --- |
| train_fold_mean_baseline | 1.085692 | baseline |
| stable_features_ridge | 20.800177 | poor linear diagnostic |
| stable_features_mlp | 1.000561 | best bounded ablation |
| consensus_maps_only_cnn | 1.030333 | beats mean |
| stable_features_plus_consensus | 1.034764 | beats mean |
| frame_bag_only | 1.049619 | beats mean |
| frame_bag_plus_consensus | 1.019925 | beats mean |
| full_fusion | 1.041494 | beats mean |
| raw_240_features_diagnostic | 20.800171 | diagnostic only; not production |
| shuffled_label_negative_control | 5.666115 | did not beat mean |
| brightness_only_forbidden_diagnostic | 1.697246 | did not beat mean |

MVP-6 reference from `reports/rheed_ssl_temporal_mvp/20260703_072054/ablations/ablation_metrics_v2.csv`:

- MVP-6 mean condition baseline MSE: 1.143632 on 5 val rows
- MVP-6 handcrafted_only MSE: 1.069064 on 5 val rows
- MVP-6 shuffled-label negative control MSE: 1.103456 on 5 val rows

The MVP-9 and MVP-6 validation sets are not identical in row count, so this is an orientation comparison, not a clean head-to-head statistical win.

## Trustworthiness Decision

Bounded checks passed:

- Best real MVP-9 ablation beat the train-fold mean baseline: yes.
- It beat metadata-only where available: metadata-only was not present in MVP-9 because metadata input was disabled; MVP-6 metadata-only is listed for context.
- Shuffled-label negative control beat mean: no.
- Brightness-only diagnostic beat mean: no.
- Raw 240-feature diagnostic selected as production: no.

Scientific caution remains:

- The supervised set is only 36 matched pairs.
- The primary checkpoint and ablations are bounded smoke/fairness runs, not 100/50 epoch final training.
- Many descriptor-level R2 scores remain negative or mean-like.
- The validation/test-side row count is 11; small folds can make rankings unstable.

Conclusion: MVP-9 is promising enough to continue, but the current bounded run is not strong enough to claim robust RHEED-to-AFM predictability beyond representative morphology descriptor conditioning.

## Generation

- Generation root: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/`
- Grid: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/shape_bag_calibrated_v2_grid_val.png`
- Metrics: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/generation_metrics_shape_bag.csv`
- Summary: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/generation_summary_shape_bag.json`
- Oracle/predicted/mean table: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/oracle_vs_predicted_vs_mean_table.csv`
- Tiny real diffusion run: 2 conditions, 2 samples per condition, 10 DDIM steps.
- Nonconstant generated rate: 1.0
- Mean generated normalized std: 0.631297

The generated AFM images came from calibrated_v2 diffusion sampling through the existing MVP-5/MVP-6 sampler path. They are representative morphology generations from predicted conditions, not retrieval/copying and not exact AFM reconstruction.

## Descriptor-Level Interpretation

From `evaluation/descriptor_predictability_table.csv`:

- Positive R2 in this bounded run: `mean_abs_gradient` only, with R2 about 0.001.
- Mostly mean-like or negative-R2 descriptors: `rq`, `ra`, `robust_range`, `psd_slope`, `autocorrelation_length_px`, `gradient_anisotropy`, `island_count`, `gradient_std`, `psd_mid_power`, `psd_high_power`.
- `island_mean_area_px` and `psd_low_power` were not flagged mean-like by the simple diagnostic, but their R2 remained negative.

No strong descriptor-level correlation claim should be made yet. The current result suggests elongated/bar-like RHEED features may be useful as weak morphology-condition signals, but the run does not establish a robust physical mapping.

## Output Locations

- Supervised index: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv`
- Target table: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_conditions_shape_bag.csv`
- Fold assignments: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/strict_fold_assignments.csv`
- Feature schema: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/feature_schema_shape_bag.json`
- Target schema: `reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_schema_shape_bag.json`
- Predictor: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_predictor/`
- Ablations: `reports/rheed_shape_bag_model_mvp/20260703_171908/ablations/ablation_metrics_shape_bag.csv`
- Predictions: `reports/rheed_shape_bag_model_mvp/20260703_171908/predicted_conditions/predicted_condition_table_val.csv`
- Evaluation: `reports/rheed_shape_bag_model_mvp/20260703_171908/evaluation/evaluation_report.md`
- Generation grid: `reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_calibrated_v2_generation/shape_bag_calibrated_v2_grid_val.png`

## Verification

- `PYTHONPATH=src .venv/bin/python -m compileall -q ...`: passed
- MVP-9 KNN/nearest-neighbor source scan: no hits
- `PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_model.py`: 6 tests OK
- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests`: 94 tests OK

Warnings during tests were constant-input correlation/convergence warnings from tiny synthetic or existing test fixtures; they did not fail tests.

## Known Limitations

- Only 36 paired supervised samples were available.
- The full requested 100-epoch main training and 50-epoch ablation suite were not run; this report contains bounded 20-epoch and 5-epoch runs.
- Metadata-only MVP-9 was not run because metadata inputs were disabled to avoid leakage-prone identifiers.
- The main production-style run disabled frame input for speed and relied on stable features + consensus maps. Frame branches were evaluated in bounded ablations.
- The calibrated_v2 generation run used 2 conditions and 10 DDIM steps, not the requested 32 samples and 100 DDIM steps.
- Descriptor predictions mostly remain mean-like; exact AFM reconstruction is outside MVP-9 scope.

## Recommended Next Command

Run the longer fair suite from the completed dataset:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.train_shape_bag_morphology_predictor \
  --supervised-index reports/rheed_shape_bag_model_mvp/20260703_171908/data/supervised_shape_bag_index.csv \
  --target-table reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_conditions_shape_bag.csv \
  --folds reports/rheed_shape_bag_model_mvp/20260703_171908/data/strict_fold_assignments.csv \
  --feature-schema reports/rheed_shape_bag_model_mvp/20260703_171908/data/feature_schema_shape_bag.json \
  --target-schema reports/rheed_shape_bag_model_mvp/20260703_171908/data/target_schema_shape_bag.json \
  --out reports/rheed_shape_bag_model_mvp/20260703_171908/shape_bag_predictor_full_fusion_100ep \
  --model shape_bag_fusion --epochs 100 --batch-size 8 --lr 1e-4 --amp \
  --predict-uncertainty true --exposure-invariance-weight 0.1 \
  --fold-id all --use-frames true --use-consensus true --use-stable-features true \
  --model-image-size 64 --hidden-dim 64 --embedding-dim 128 --seed 42
```
