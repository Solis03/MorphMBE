# RHEED-Conditioned Latent Diffusion MVP-2 Report

Run root: `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816`

## Scope Statement

This MVP evaluates a two-stage RHEED-conditioned latent diffusion pipeline: RHEED predicts AFM descriptor/prototype conditions, and the existing AFM latent diffusion model generates AFM-like morphology. It does not claim exact pixel-level AFM reconstruction from RHEED.

No KNN retrieval is used in the new MVP-2 path. The old baseline code was left intact.

## Environment And Git Status

- `pwd`: `/home/wangziyi/MorphMBE/MorphMBE`
- Python: `Python 3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`

Git status before MVP-2 implementation:

```text
?? reports/conditional_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
```

Git status after MVP-2 implementation:

```text
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

## Files Created Or Modified

New MVP-2 source files:

- `src/rheed2morph/generative/rheed_video.py`
- `src/rheed2morph/generative/rheed_features.py`
- `src/rheed2morph/generative/prepare_rheed_condition_dataset.py`
- `src/rheed2morph/generative/models/rheed_condition_encoder.py`
- `src/rheed2morph/generative/train_rheed_condition_encoder.py`
- `src/rheed2morph/generative/predict_rheed_conditions.py`
- `src/rheed2morph/generative/sample_rheed_conditioned_diffusion.py`
- `src/rheed2morph/generative/evaluate_rheed_conditioned_generation.py`
- `tests/test_generative_rheed_conditioned_diffusion.py`

New artifact/report tree:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/`

## MVP-1 Dependency Summary

MVP-1 report read:

- `reports/conditional_latent_diffusion_mvp/20260703_041331/codex_report.md`

MVP-1 checkpoints used for final MVP-2 generation:

- AFM autoencoder: `reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt`
- Latent diffusion: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt`

Condition table used:

- `reports/conditional_latent_diffusion_mvp/20260703_041331/latents_5epoch/condition_table.csv`

Columns in that condition table:

```text
row_id, sample_id, group_id, split, network_input_path, descriptor_height_path, prototype_id,
height_mean, cond_height_mean, height_std, cond_height_std, rq, cond_rq, ra, cond_ra,
peak_to_valley, cond_peak_to_valley, p01, cond_p01, p05, cond_p05, p50, cond_p50,
p95, cond_p95, p99, cond_p99, robust_range, cond_robust_range, skewness, cond_skewness,
kurtosis, cond_kurtosis, mean_abs_gradient, cond_mean_abs_gradient, gradient_std,
cond_gradient_std, gradient_orientation_entropy, cond_gradient_orientation_entropy,
gradient_anisotropy, cond_gradient_anisotropy, psd_low_power, cond_psd_low_power,
psd_mid_power, cond_psd_mid_power, psd_high_power, cond_psd_high_power, psd_slope,
cond_psd_slope, autocorrelation_length_px, cond_autocorrelation_length_px,
island_coverage, cond_island_coverage, island_count, cond_island_count,
island_mean_area_px, cond_island_mean_area_px
```

Diffusion condition columns:

```text
cond_height_mean, cond_height_std, cond_rq, cond_ra, cond_peak_to_valley,
cond_p01, cond_p05, cond_p50, cond_p95, cond_p99, cond_robust_range,
cond_skewness, cond_kurtosis, cond_mean_abs_gradient, cond_gradient_std,
cond_gradient_orientation_entropy, cond_gradient_anisotropy, cond_psd_low_power,
cond_psd_mid_power, cond_psd_high_power, cond_psd_slope,
cond_autocorrelation_length_px, cond_island_coverage, cond_island_count,
cond_island_mean_area_px
```

Descriptor format:

- Raw descriptor columns are physical/plane-corrected AFM descriptor values.
- `cond_*` columns are standardized train-set descriptor conditions.
- Prototype labels exist: `prototype_count=6`.
- The MVP-1 diffusion checkpoint expects `condition_dim=31`: 25 standardized `cond_*` descriptor values plus a 6-way one-hot `prototype_id`.
- The sampler reads predicted `cond_*` columns and `prototype_id`; oracle comparison reads `true_cond_*` and `true_prototype_id`.

## Exact Commands Run

Environment and schema inspection:

```bash
git status --short
pwd && .venv/bin/python --version && .venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
sed -n '1,260p' reports/conditional_latent_diffusion_mvp/20260703_041331/codex_report.md
sed -n '1,5p' reports/conditional_latent_diffusion_mvp/20260703_041331/latents/condition_table.csv
sed -n '1,180p' reports/conditional_latent_diffusion_mvp/20260703_041331/latents/latent_stats.json
sed -n '1,160p' reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/config.json
find data reports -type f -iname '*raw_crop*.mp4' | sed -n '1,240p'
```

Tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_rheed_conditioned_diffusion.py
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_afm_latent_diffusion.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Initial 16-pair smoke:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_rheed_condition_dataset --mvp1-root reports/conditional_latent_diffusion_mvp/20260703_041331 --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data --scan-size-filter 1um --frames 8 --image-size 224 --limit 16 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_condition_encoder --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/condition_schema.json --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder --frames 8 --image-size 224 --visual-backbone small_cnn --temporal-pooling attention --epochs 1 --batch-size 4 --amp --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.predict_rheed_conditions --checkpoint reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder/checkpoints/best.pt --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/condition_schema.json --split val --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_rheed_conditioned_diffusion --diffusion-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --predicted-condition-table reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions/predicted_condition_table_val.csv --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/paired_rheed_condition_index.csv --split val --num-samples-per-condition 4 --ddim-steps 10 --guidance-scale 1.5 --max-conditions 4 --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_rheed_conditioned_generation --predicted-condition-table reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions/predicted_condition_table_val.csv --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data/condition_schema.json --generation-metrics reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples/generation_metrics.csv --split val --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/evaluation
```

Bounded 36-pair run and ablations:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_rheed_condition_dataset --mvp1-root reports/conditional_latent_diffusion_mvp/20260703_041331 --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64 --scan-size-filter 1um --frames 8 --image-size 224 --limit 64 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_condition_encoder --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/condition_schema.json --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder_10epoch_visual_handcrafted --frames 8 --image-size 224 --visual-backbone small_cnn --temporal-pooling attention --epochs 10 --batch-size 8 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_condition_encoder --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/condition_schema.json --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder_10epoch_handcrafted --frames 8 --image-size 224 --visual-backbone small_cnn --temporal-pooling mean --epochs 10 --batch-size 8 --amp --use-visual false --use-handcrafted true --use-metadata false --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.predict_rheed_conditions --checkpoint reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder_10epoch_visual_handcrafted/checkpoints/best.pt --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/condition_schema.json --split val --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_rheed_conditioned_diffusion --diffusion-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --predicted-condition-table reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted/predicted_condition_table_val.csv --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/paired_rheed_condition_index.csv --split val --num-samples-per-condition 4 --ddim-steps 50 --guidance-scale 1.5 --max-conditions 4 --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples_10epoch_visual_handcrafted
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_rheed_conditioned_generation --predicted-condition-table reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted/predicted_condition_table_val.csv --paired-index reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/paired_rheed_condition_index.csv --condition-schema reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/condition_schema.json --generation-metrics reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples_10epoch_visual_handcrafted/generation_metrics.csv --split val --out reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/evaluation_10epoch_visual_handcrafted
```

## RHEED-AFM Pairing Inventory

Final paired data directory: `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64`

- RHEED records found: `62`
- Matched RHEED-AFM condition pairs: `36`
- Unmatched RHEED records: `26`
- Unmatched AFM condition rows: `0`
- Split counts by sample: train `25`, val `5`, test `6`
- Split counts by group: train `25`, val `5`, test `6`
- Sample key matching rule: RHEED numeric token joined to MVP-1 condition `sample_id`/`group_id`/`growth_id`, preferring explicit columns when present.
- Debug files:
  - `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/unmatched_rheed.csv`
  - `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/unmatched_afm_conditions.csv`
  - `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/pair_grid.png`

## RHEED Preprocessing Summary

- Frames used: `8`
- Sampling: uniform over final portion
- Final fraction: `0.25`
- Image size: `224`
- Color handling: grayscale
- Normalization: percentile clip 1st-99th, normalize to `[0, 1]`
- Cached tensors: `36`
- Cached tensor shape: `[T, 1, H, W]`
- Video read failures: `0`

## Handcrafted Feature Summary

Feature table:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/data_limit64/rheed_handcrafted_features.csv`

Feature columns:

```text
mean_intensity_mean, mean_intensity_std, std_intensity_mean, std_intensity_std,
p01_intensity_mean, p01_intensity_std, p05_intensity_mean, p05_intensity_std,
p50_intensity_mean, p50_intensity_std, p95_intensity_mean, p95_intensity_std,
p99_intensity_mean, p99_intensity_std, saturated_fraction_mean, saturated_fraction_std,
dark_fraction_mean, dark_fraction_std, laplacian_sharpness_mean, laplacian_sharpness_std,
gradient_mean_mean, gradient_mean_std, gradient_std_mean, gradient_std_std,
horizontal_projection_entropy_mean, horizontal_projection_entropy_std,
vertical_projection_entropy_mean, vertical_projection_entropy_std,
horizontal_peak_location_mean, horizontal_peak_location_std,
horizontal_peak_width_mean, horizontal_peak_width_std, vertical_peak_location_mean,
vertical_peak_location_std, vertical_peak_width_mean, vertical_peak_width_std,
intensity_center_of_mass_x_mean, intensity_center_of_mass_x_std,
intensity_center_of_mass_y_mean, intensity_center_of_mass_y_std,
fft_low_power_mean, fft_low_power_std, fft_mid_power_mean, fft_mid_power_std,
fft_high_power_mean, fft_high_power_std, fft_anisotropy_mean, fft_anisotropy_std,
temporal_mean_abs_frame_difference, temporal_std_frame_mean_intensity,
temporal_std_frame_sharpness
```

Imputation counts: all handcrafted feature columns had `0` imputed values in the final 36-pair run.

## RHEED Condition Encoder Metrics

Initial 16-pair smoke, visual+handcrafted, 1 quick epoch:

- Val descriptor MSE: `0.754134`
- Val descriptor MAE: `0.644621`
- Val descriptor R2: `-0.611277`
- Val descriptor Spearman: `0.750000`
- Prototype accuracy: `0.500000`
- Prototype macro-F1: `0.111111`
- Mean-condition baseline MSE: `0.792075`
- Result: smoke model beat mean-condition MSE on a 2-row validation split.

Final bounded 36-pair visual+handcrafted 10-epoch run:

- Training directory: `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder_10epoch_visual_handcrafted`
- Last epoch val descriptor MSE: `1.502854`
- Best checkpoint selected at epoch 5 and used for prediction/sampling.
- Best-checkpoint val prediction MSE: `1.190013`
- Best-checkpoint val prediction MAE: `0.775115`
- Best-checkpoint val prediction R2: `-0.227278`
- Best-checkpoint val prediction Spearman: `-0.031716`
- Best-checkpoint prototype accuracy: `0.400000`
- Best-checkpoint prototype macro-F1: `0.111111`
- Mean-condition baseline MSE: `1.227069`
- Result: best visual+handcrafted checkpoint slightly beat mean-condition MSE, but the final epoch overfit/regressed.

Final bounded handcrafted-only 10-epoch ablation:

- Training directory: `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_condition_encoder_10epoch_handcrafted`
- Last epoch val descriptor MSE: `1.388928`
- Last epoch val descriptor MAE: `0.829076`
- Last epoch prototype accuracy: `0.000000`
- Mean-condition baseline MSE: `1.227069`
- Result: final handcrafted-only model did not beat mean-condition baseline.

Ablation outputs:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/ablation_metrics.csv`
- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/ablation_summary.json`
- Per-run scatter/confusion outputs:
  - `rheed_condition_encoder_10epoch_visual_handcrafted/descriptor_scatter_top_targets.png`
  - `rheed_condition_encoder_10epoch_visual_handcrafted/prototype_confusion.png`
  - `rheed_condition_encoder_10epoch_handcrafted/descriptor_scatter_top_targets.png`
  - `rheed_condition_encoder_10epoch_handcrafted/prototype_confusion.png`

## Generation Metrics

Final sample output:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples_10epoch_visual_handcrafted`

Sample grid:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples_10epoch_visual_handcrafted/rheed_conditioned_sample_grid_val.png`

Generation metrics:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/rheed_conditioned_samples_10epoch_visual_handcrafted/generation_metrics.csv`

Evaluation summary:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/evaluation_10epoch_visual_handcrafted/generation_summary.json`

Generation summary:

- Mean-condition generation std mean: `0.218609`
- Oracle-conditioned generation std mean: `0.209523`
- RHEED-predicted-conditioned generation std mean: `0.212343`
- Generated image nonconstant rate: `1.000`
- Generated image std min across all modes: `0.174709`
- Predicted-conditioned generated outputs nonconstant: `true`
- Oracle-conditioned generated outputs nonconstant: `true`
- Mean-conditioned generated outputs nonconstant: `true`

Failure case grid:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/evaluation_10epoch_visual_handcrafted/failure_cases_grid.png`

## Visual Inspection Notes

- The final sample grid renders successfully and the generated AFM-like maps are visibly nonconstant.
- RHEED-predicted generations show morphology variation across samples and stochastic draws, but they are not yet convincingly separated from the mean-condition baseline by the scalar generation statistics.
- Oracle-conditioned samples are useful as an upper-bound comparison, but in this short run they do not visually dominate the predicted-condition samples. The fixed AE decoder itself is still smooth/blobby in reconstructions, which limits visual conclusions.
- Some RHEED final frames are low-contrast or nearly blank after final-window sampling, so future runs should consider frame-quality filtering or temporal windows matched to the growth event.

## Test Results

Focused MVP-2 test:

```text
Ran 7 tests in 2.541s
OK
```

Focused MVP-1 generative test:

```text
Ran 7 tests in 1.477s
OK
```

Full test discovery:

```text
Ran 37 tests in 2.618s
OK
```

The full suite emitted existing sklearn warnings from older tests and `ConstantInputWarning` during small-split Spearman calculations, but no tests failed.

## Known Limitations And Failures

- No command failed during MVP-2 implementation or smoke runs.
- The validation splits are tiny. Spearman can be undefined for constant targets/predictions; this produced warnings but not failures.
- The best 10-epoch visual+handcrafted checkpoint only slightly beat the mean-condition descriptor MSE. This is not strong evidence of RHEED-to-morphology predictive power yet.
- Final epoch metrics were worse than the selected best checkpoint, suggesting overfit/instability.
- Prototype accuracy remains weak.
- The generated outputs are realistic enough to validate data flow, but not final scientific-quality morphology.
- Generated descriptor metrics are computed on normalized decoded AFM-like images; the condition descriptors are standardized from physical AFM descriptors, so absolute descriptor agreement remains approximate.

## Acceptance Criteria Check

- Unit tests pass: yes.
- Existing tests pass: yes.
- `paired_rheed_condition_index.csv` exists: yes.
- At least one RHEED-conditioned sample grid exists: yes.
- `ablation_metrics.csv` or equivalent baseline comparison exists: yes.
- `generation_summary.json` exists: yes.
- New MVP-2 path does not use KNN: yes, covered by test scan.
- Final report exists: yes.

## Recommended Longer RTX 5090 Run

Use a new timestamped output root and train longer with early stopping/model selection:

```bash
RUN=reports/rheed_conditioned_latent_diffusion_mvp/$(date -u +%Y%m%d_%H%M%S)
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_rheed_condition_dataset --mvp1-root reports/conditional_latent_diffusion_mvp/20260703_041331 --out "$RUN/data" --scan-size-filter 1um --frames 16 --image-size 224 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_condition_encoder --paired-index "$RUN/data/paired_rheed_condition_index.csv" --condition-schema "$RUN/data/condition_schema.json" --out "$RUN/rheed_condition_encoder_visual_handcrafted" --frames 16 --image-size 224 --visual-backbone small_cnn --temporal-pooling attention --epochs 100 --batch-size 16 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_rheed_condition_encoder --paired-index "$RUN/data/paired_rheed_condition_index.csv" --condition-schema "$RUN/data/condition_schema.json" --out "$RUN/rheed_condition_encoder_handcrafted" --frames 16 --image-size 224 --use-visual false --use-handcrafted true --use-metadata false --epochs 100 --batch-size 16 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.predict_rheed_conditions --checkpoint "$RUN/rheed_condition_encoder_visual_handcrafted/checkpoints/best.pt" --paired-index "$RUN/data/paired_rheed_condition_index.csv" --condition-schema "$RUN/data/condition_schema.json" --split val --out "$RUN/predicted_conditions"
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_rheed_conditioned_diffusion --diffusion-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --predicted-condition-table "$RUN/predicted_conditions/predicted_condition_table_val.csv" --paired-index "$RUN/data/paired_rheed_condition_index.csv" --split val --num-samples-per-condition 4 --ddim-steps 50 --guidance-scale 1.5 --out "$RUN/rheed_conditioned_samples"
```
