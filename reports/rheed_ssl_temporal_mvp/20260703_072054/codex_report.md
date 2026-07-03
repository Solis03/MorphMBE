# RHEED SSL Temporal MVP-6 Report

Run root: `reports/rheed_ssl_temporal_mvp/20260703_072054`

## Scope Statement

MVP-6 adds RHEED self-supervised frame/temporal representation learning, a small-data temporal morphology condition predictor, ablations, label-efficiency checks, and calibrated_v2 AFM-prior sampling. It does not use a new retrieval path and does not claim exact pixel-level AFM reconstruction.

MVP-6 improves RHEED representation learning and temporal condition prediction. It still generates representative AFM-like morphology through the calibrated_v2 AFM prior and does not claim exact pixel-level AFM reconstruction.

## Git Status

Before MVP-6 implementation, the worktree already had untracked prior MVP source/artifacts:

```text
?? reports/afm_condition_control_v3/
?? reports/afm_prior_v2/
?? reports/afm_prior_v4_height_calibrated/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_afm_prior_v4_height_calibration.py
?? tests/test_generative_condition_control_v3.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

After MVP-6:

```text
?? reports/afm_condition_control_v3/
?? reports/afm_prior_v2/
?? reports/afm_prior_v4_height_calibrated/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? reports/rheed_ssl_temporal_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_afm_prior_v4_height_calibration.py
?? tests/test_generative_condition_control_v3.py
?? tests/test_generative_rheed_conditioned_diffusion.py
?? tests/test_generative_rheed_ssl_temporal.py
```

## Files Created

- `src/rheed2morph/generative/prepare_rheed_ssl_dataset.py`
- `src/rheed2morph/generative/rheed_ssl_augmentations.py`
- `src/rheed2morph/generative/models/rheed_mae.py`
- `src/rheed2morph/generative/models/rheed_temporal_encoder.py`
- `src/rheed2morph/generative/pretrain_rheed_frame_mae.py`
- `src/rheed2morph/generative/pretrain_rheed_temporal_ssl.py`
- `src/rheed2morph/generative/extract_rheed_ssl_embeddings.py`
- `src/rheed2morph/generative/train_rheed_morphology_encoder_v2.py`
- `src/rheed2morph/generative/run_rheed_morphology_ablation_v2.py`
- `src/rheed2morph/generative/predict_rheed_conditions_v2.py`
- `src/rheed2morph/generative/sample_rheed_conditioned_calibrated_v2.py`
- `src/rheed2morph/generative/evaluate_rheed_ssl_temporal.py`
- `tests/test_generative_rheed_ssl_temporal.py`

## Environment

- Python: `3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`

## Prior Artifact Summary

MVP-2 input status:

- RHEED records found: `62`
- Matched RHEED/AFM condition pairs: `36`
- Unmatched RHEED records: `26`
- Split counts: train `25`, val `5`, test `6`
- Best old RHEED encoder val MSE: `1.190013`
- Old mean-condition val MSE: `1.227069`
- Old predicted tables include `pred_cond_*`, physical `pred_*`, `prototype_id`, true condition columns, RHEED paths, cached tensors, and AFM paths.

MVP-5 dependency:

- Primary production generator decision: `calibrated_v2`
- Autoencoder: `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt`
- V2 diffusion: `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt`
- V3 diffusion control: `reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt`
- Condition schema: `reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json`
- Calibrated_v2 method: weighted least-squares height scaling using Rq/Ra/robust range with mode `weighted_rq_ra_range`.
- V3 condition descriptors used by MVP-6: `rq`, `ra`, `robust_range`, `mean_abs_gradient`, `gradient_std`, `gradient_anisotropy`, `psd_low_power`, `psd_mid_power`, `psd_high_power`, `psd_slope`, `autocorrelation_length_px`, `island_count`, `island_mean_area_px`.

## RHEED Data Inventory

Final data root: `reports/rheed_ssl_temporal_mvp/20260703_072054/data`

- Cached videos: `62`
- Cached frame rows: `992`
- Paired videos: `36`
- Unpaired videos used for SSL: `26`
- Video read failures: `0`
- MVP-2 split preserved: `true`
- Split counts: train `25`, val `5`, test `6`, unpaired `26`
- Video/cache source: `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/raw_crop_video_manifest.csv`
- Sequence tensor shape: `[T,1,H,W]`

Preprocessing:

- Frames: `16`
- Final fraction: `0.25`
- Image size: `224`
- Sampling: uniform over final fraction
- Normalization: percentile 1-99 to `[0,1]`
- Augmentations: mild crop/translation, brightness/contrast/gamma, noise, blur, patch mask; rotations/flips disabled unless explicitly enabled.

## SSL Pretraining

Frame MAE:

- Model: compact CNN masked frame autoencoder
- Frames used: `992`
- Epochs run: `10`
- Mask ratio: `0.60`
- Best reconstruction loss: `0.006074`
- Reconstruction grid: `reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/mae_reconstruction_grid.png`
- Embedding PCA: `reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/embedding_pca.png`

Optional temporal SSL:

- Objective: binary temporal intensity trend prediction with GRU temporal pooling
- Epochs run: `5`
- Best loss: `0.497122`
- Embedding PCA: `reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_temporal_ssl/temporal_embedding_umap_or_pca.png`
- Note: the first temporal SSL attempt failed because the quick time-bin target produced frame indices instead of binary labels. The target was fixed and the command was rerun successfully.

## Supervised Encoder

Main temporal model:

- Architecture: small CNN frame encoder initialized from frame MAE, attention temporal pooling, handcrafted MLP, metadata MLP, fusion MLP, descriptor head, prototype head, uncertainty head
- Epochs run: `40`
- Val MSE: `1.151810`
- Val MAE: `0.785838`
- Mean baseline MSE in that run: `1.143632`
- Test MSE: `1.037624`
- Result: main temporal visual+handcrafted+metadata model did not beat mean-condition baseline.

Best validated checkpoint for final predictions:

- Variant: `handcrafted_only`
- Checkpoint: `reports/rheed_ssl_temporal_mvp/20260703_072054/ablations/handcrafted_only/checkpoints/best.pt`
- Val MSE: `1.069064`
- Val MAE: `0.772288`
- Test MSE: `0.957480`
- Test MAE: `0.771408`

## Ablations

Metrics file: `reports/rheed_ssl_temporal_mvp/20260703_072054/ablations/ablation_metrics_v2.csv`

| Variant | Val MSE | Beats Mean |
| --- | ---: | --- |
| mean-condition baseline | 1.143632 | - |
| metadata-only | 1.109478 | true |
| handcrafted-only | 1.069064 | true |
| final-frame visual-only | 1.155434 | false |
| temporal attention visual+handcrafted | 1.115933 | true |
| temporal attention visual+handcrafted+metadata | 1.113310 | true |
| shuffled-label negative control | 1.103456 | true |

Interpretation:

- RHEED-derived features beat the mean-condition baseline, but the best model is handcrafted-only rather than visual-temporal.
- Temporal visual+handcrafted beats final-frame visual-only (`1.115933 < 1.155434`).
- Metadata-only also beats mean, so metadata/source-frame information is a real confounder.
- The shuffled-label negative control also beats mean. This is a leakage/low-sample warning; the small 36-pair set is not enough for a strong causal claim.

Label efficiency:

- File: `reports/rheed_ssl_temporal_mvp/20260703_072054/ablations/label_efficiency_metrics.csv`
- MAE-initialized temporal models were slightly better than random-init temporal models at 25/50/75/100 percent labels, but the margins are small.
- Handcrafted-only stayed strongest across label fractions.

## Prediction And Generation

Predicted condition tables:

- Val: `reports/rheed_ssl_temporal_mvp/20260703_072054/predicted_conditions_v2/predicted_condition_table_val.csv`
- Test: `reports/rheed_ssl_temporal_mvp/20260703_072054/predicted_conditions_v2/predicted_condition_table_test.csv`

Descriptor predictability on val:

- Structured: `psd_low_power`, `psd_mid_power`, `psd_high_power`, `psd_slope`, `island_mean_area_px`
- Weak: `robust_range`, `autocorrelation_length_px`, `island_count`
- Mean-like: `rq`, `ra`, `mean_abs_gradient`, `gradient_std`, `gradient_anisotropy`

Calibrated_v2 generation:

- Output: `reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_conditioned_calibrated_v2_samples`
- Grid: `rheed_conditioned_calibrated_v2_grid_val.png`
- Candidates: `16` per condition
- DDIM steps: `50`
- Conditions sampled: `4`
- Nonconstant rate: `1.000`
- Generated normalized std mean: `0.632806`
- No v3 predicted/oracle descriptors were filled by the adapter; v2 compatibility filled descriptors outside the v3 schema with train means.

Top-1 calibrated roughness errors over the four generated val conditions:

| Mode | Rq MAE | Ra MAE | Robust range MAE |
| --- | ---: | ---: | ---: |
| MVP-6 predicted calibrated_v2 | 1.435993 | 1.679287 | 1.183870 |
| Oracle calibrated_v2 | 2.060020 | 2.234301 | 1.655131 |
| Mean-condition calibrated_v2 | 1.643070 | 1.839051 | 1.356639 |

This small generation comparison favors predicted conditions over mean for roughness metrics, but the sample count is only four validation conditions.

## Evaluation

Evaluation folder: `reports/rheed_ssl_temporal_mvp/20260703_072054/evaluation`

Direct answers:

1. RHEED beats mean-condition baseline: `true`, by best handcrafted-only validation MSE.
2. RHEED beats metadata-only baseline: `true`, but only modestly.
3. Temporal video beats final-frame-only: `true`.
4. SSL pretraining improves label efficiency: `true`, but by small margins.
5. Predictable descriptors: primarily PSD/texture and island-area descriptors.
6. Mean-like descriptors: roughness `rq`/`ra` and gradient descriptors remain weak.
7. Predicted-condition calibrated_v2 generation is physically plausible by nonconstant/richness proxy.
8. Generated samples differ from mean-condition samples by nonconstant/richness and descriptor metrics, but this is not an exact AFM reconstruction claim.
9. Uncertainty calibration is weak: validation error vs predicted variance Pearson was `-0.323`.
10. Robustness under group split remains limited by only 5 validation and 6 test paired samples.

## Known Limitations

- The supervised set is only 36 paired RHEED/AFM rows.
- Shuffled-label negative control beating mean indicates leakage risk or a very weak baseline under tiny validation size.
- Metadata-only beating mean means source/frame-count metadata must be audited before claiming visual causality.
- The best predictor uses handcrafted RHEED summaries, not the learned visual-temporal branch.
- Rq/Ra remain mean-like in descriptor-level prediction.
- Sampling uses calibrated_v2 as a representative morphology prior; it is not pixel-level AFM reconstruction.

## Exact Commands Run

Key inspection and environment:

```bash
git status --short
.venv/bin/python --version
PYTHONPATH=src .venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Smoke:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_rheed_ssl_dataset --out reports/rheed_ssl_temporal_mvp/20260703_072054/smoke --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --frames 8 --image-size 64 --limit 16 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.pretrain_rheed_frame_mae --frame-index reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/data/rheed_ssl_frame_index.csv --out reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/rheed_frame_mae --image-size 64 --patch-size 8 --embed-dim 128 --depth 2 --decoder-depth 1 --mask-ratio 0.60 --epochs 1 --batch-size 16 --lr 1e-4 --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_morphology_encoder_v2 --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --out reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/rheed_morphology_encoder_v2 --frames 8 --image-size 64 --frame-encoder small_cnn --frame-mae-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/rheed_frame_mae/checkpoints/best.pt --temporal-pooling attention --use-handcrafted true --use-metadata true --predict-uncertainty true --epochs 1 --batch-size 4 --lr 1e-4 --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.run_rheed_morphology_ablation_v2 --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --frame-mae-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/rheed_frame_mae/checkpoints/best.pt --out reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/ablations --epochs 1 --label-efficiency-epochs 1 --batch-size 4 --lr 1e-4 --frames 8 --image-size 64 --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.predict_rheed_conditions_v2 --checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/rheed_morphology_encoder_v2/checkpoints/best.pt --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --out reports/rheed_ssl_temporal_mvp/20260703_072054/smoke/predicted_conditions_v2 --batch-size 4
```

Final MVP-6:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_rheed_ssl_dataset --out reports/rheed_ssl_temporal_mvp/20260703_072054 --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --frames 16 --image-size 224 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.pretrain_rheed_frame_mae --frame-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_ssl_frame_index.csv --out reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae --image-size 224 --patch-size 16 --embed-dim 256 --depth 6 --decoder-depth 3 --mask-ratio 0.60 --epochs 10 --batch-size 64 --lr 1e-4 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.pretrain_rheed_temporal_ssl --video-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_ssl_video_index.csv --frame-encoder-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/checkpoints/best.pt --out reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_temporal_ssl --frames 16 --image-size 224 --epochs 5 --batch-size 16 --lr 1e-4 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_morphology_encoder_v2 --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --out reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_morphology_encoder_v2 --frames 16 --image-size 224 --frame-encoder small_cnn --frame-mae-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/checkpoints/best.pt --temporal-pooling attention --use-handcrafted true --use-metadata true --predict-uncertainty true --epochs 40 --batch-size 8 --lr 1e-4 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.run_rheed_morphology_ablation_v2 --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --frame-mae-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/checkpoints/best.pt --out reports/rheed_ssl_temporal_mvp/20260703_072054/ablations --epochs 20 --label-efficiency-epochs 5 --batch-size 8 --lr 1e-4 --frames 16 --image-size 224 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.predict_rheed_conditions_v2 --checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/ablations/handcrafted_only/checkpoints/best.pt --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --out reports/rheed_ssl_temporal_mvp/20260703_072054/predicted_conditions_v2 --batch-size 8
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_rheed_conditioned_calibrated_v2 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --predicted-condition-table reports/rheed_ssl_temporal_mvp/20260703_072054/predicted_conditions_v2/predicted_condition_table_val.csv --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --primary-generator calibrated_v2 --split val --num-samples-per-condition 16 --keep-top-k 4 --ddim-steps 50 --guidance-scale 1.5 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 4 --out reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_conditioned_calibrated_v2_samples
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_rheed_ssl_temporal --mvp6-root reports/rheed_ssl_temporal_mvp/20260703_072054 --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --out reports/rheed_ssl_temporal_mvp/20260703_072054/evaluation
```

Tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_rheed_ssl_temporal.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
rg -n "kneighbors|nearestneighbors|nearest_neighbors" src/rheed2morph/generative/prepare_rheed_ssl_dataset.py src/rheed2morph/generative/rheed_ssl_augmentations.py src/rheed2morph/generative/models/rheed_mae.py src/rheed2morph/generative/models/rheed_temporal_encoder.py src/rheed2morph/generative/pretrain_rheed_frame_mae.py src/rheed2morph/generative/pretrain_rheed_temporal_ssl.py src/rheed2morph/generative/extract_rheed_ssl_embeddings.py src/rheed2morph/generative/train_rheed_morphology_encoder_v2.py src/rheed2morph/generative/run_rheed_morphology_ablation_v2.py src/rheed2morph/generative/predict_rheed_conditions_v2.py src/rheed2morph/generative/sample_rheed_conditioned_calibrated_v2.py src/rheed2morph/generative/evaluate_rheed_ssl_temporal.py tests/test_generative_rheed_ssl_temporal.py
```

Test results:

- `tests/test_generative_rheed_ssl_temporal.py`: `4` tests OK.
- Full suite: `67` tests OK.
- Retrieval-neighbor scan: no matches after excluding no files; the scan returned exit code `1` because no prohibited terms were found.

## Recommended Next Command

For a longer run, continue from the same prepared data and run the full visual ablation suite:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.run_rheed_morphology_ablation_v2 --paired-index reports/rheed_ssl_temporal_mvp/20260703_072054/data/rheed_supervised_pair_index.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp5-root reports/afm_prior_v4_height_calibrated/20260703_064826 --frame-mae-checkpoint reports/rheed_ssl_temporal_mvp/20260703_072054/rheed_frame_mae/checkpoints/best.pt --out reports/rheed_ssl_temporal_mvp/20260703_072054/ablations_full_suite --epochs 50 --label-efficiency-epochs 10 --batch-size 8 --lr 1e-4 --frames 16 --image-size 224 --amp --full-suite --seed 42
```
