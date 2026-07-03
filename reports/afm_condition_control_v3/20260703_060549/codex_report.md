# AFM Prior V3 MVP-4 Condition-Control Report

Run root: `reports/afm_condition_control_v3/20260703_060549`

## Scope Statement

MVP-4 targets AFM prior condition control and descriptor calibration. It reuses the MVP-3 AFM autoencoder v2 and does not retrain the RHEED encoder.

MVP-4 improves condition control and descriptor calibration of the AFM latent diffusion prior. In this run, that improvement is partial and strongest for PSD slope, autocorrelation length, island count, and sampling-time reranking. It does not improve Rq, Ra, or robust range yet. It does not by itself prove strong RHEED-to-AFM predictive accuracy.

## Environment And Git Status

- `pwd`: `/home/wangziyi/MorphMBE/MorphMBE`
- Python: `3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`

Git status before MVP-4 implementation:

```text
?? reports/afm_prior_v2/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

Git status after MVP-4 implementation and run:

```text
?? reports/afm_condition_control_v3/
?? reports/afm_prior_v2/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_condition_control_v3.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

## Files Created Or Updated

MVP-4 source files:

- `src/rheed2morph/generative/condition_control_v3_utils.py`
- `src/rheed2morph/generative/analyze_condition_sensitivity_v2.py`
- `src/rheed2morph/generative/train_latent_descriptor_regressor.py`
- `src/rheed2morph/generative/train_afm_latent_diffusion_v3.py`
- `src/rheed2morph/generative/sample_afm_prior_v3.py`
- `src/rheed2morph/generative/descriptor_guided_sampling.py`
- `src/rheed2morph/generative/evaluate_condition_control_v3.py`
- `src/rheed2morph/generative/compare_v2_v3_condition_control.py`
- `src/rheed2morph/generative/rerun_rheed_conditioned_with_v3_prior.py`

MVP-4 test:

- `tests/test_generative_condition_control_v3.py`

New artifact tree:

- `reports/afm_condition_control_v3/20260703_060549/`

## MVP-3 Dependency Summary

MVP-3 source report:

- `reports/afm_prior_v2/20260703_052537/codex_report.md`

Checkpoints reused:

- AE v2 checkpoint: `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt`
- Diffusion v2 checkpoint: `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt`

MVP-3 latent and condition schema:

- Latent shape: `[16, 16, 16]`
- Train/val/test latent counts: `115 / 31 / 22`
- V2 descriptor columns: `32`
- Prototype one-hot count: `4`
- V2 condition dimension: `36`
- V2 condition vectors used standardized `cond_*` descriptor columns plus prototype one-hot.
- V2 diffusion used EMA and sampled from `ema_last.pt`.

V2 descriptor columns:

```text
height_mean, height_std, rq, ra, peak_to_valley, p01, p05, p50, p95, p99,
robust_range, skewness, kurtosis, mean_abs_gradient, gradient_std,
gradient_orientation_entropy, gradient_anisotropy, psd_low_power,
psd_mid_power, psd_high_power, psd_slope, autocorrelation_length_px,
island_coverage, island_count, island_mean_area_px, height_min, height_max,
slope_p50, slope_p95, slope_p99, psd_peak_frequency, island_mean_height
```

MVP-3 generation quality summary:

- `samples_v2/generation_summary_v2.json`: generated nonconstant rate `1.000`, generated std mean `0.631862`, generated std min `0.615581`.
- `latent_diffusion_v2/metrics.json`: train loss `0.416426`, val loss `0.401060`, final sample pixel std `0.638744`.
- MVP-3 report caveat: generated descriptors were recomputed on decoded normalized tensors, while requested descriptors are physical nm descriptors. Absolute descriptor errors are therefore not physically calibrated.

MVP-3 RHEED condition adapter:

- Mapped the 25 MVP-1/MVP-2 descriptors shared with MVP-3.
- Filled v2-only descriptors from MVP-3 train means: `height_min`, `height_max`, `slope_p50`, `slope_p95`, `slope_p99`, `psd_peak_frequency`, `island_mean_height`.
- Used a zero prototype vector because MVP-1/MVP-2 and MVP-3 prototype IDs were not semantically aligned.

## Exact Commands Run

Focused v3 tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_condition_control_v3.py
```

Smoke MVP-4 run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.analyze_condition_sensitivity_v2 --mvp3-root reports/afm_prior_v2/20260703_052537 --diffusion-checkpoint reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --condition-schema reports/afm_prior_v2/20260703_052537/latents_v2/condition_schema_v2.json --out reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity_smoke --split val --num-base-conditions 2 --num-samples-per-condition 2 --ddim-steps 10 --guidance-scales 0.0,1.5 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.condition_control_v3_utils --condition-table-v2 reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --descriptors reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv --prototypes reports/afm_prior_v2/20260703_052537/data/morphology_prototypes_v2.csv --sensitivity-summary reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity_smoke/v2_condition_sensitivity_summary.json --out reports/afm_condition_control_v3/20260703_060549/condition_schema_v3
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_latent_descriptor_regressor --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --out reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor_smoke --epochs 2 --batch-size 32 --lr 1e-3 --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v3 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor_smoke/checkpoints/best.pt --out reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3_smoke --epochs 1 --batch-size 32 --lr 1e-4 --timesteps 100 --prediction-target epsilon --beta-schedule cosine --cond-dropout 0.10 --aux-cond-loss-weight 0.05 --sample-every 1 --quick --amp --ema --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v3 --diffusion-checkpoint reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3_smoke/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor_smoke/checkpoints/best.pt --split val --num-samples-per-condition 2 --keep-top-k 1 --ddim-steps 10 --guidance-scale 1.5 --descriptor-guidance-weight 0.05 --rerank true --max-conditions 2 --out reports/afm_condition_control_v3/20260703_060549/samples_v3_smoke
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_condition_control_v3 --samples-dir reports/afm_condition_control_v3/20260703_060549/samples_v3_smoke --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp3-v2-sensitivity reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity_smoke --out reports/afm_condition_control_v3/20260703_060549/evaluation_v3_smoke
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.compare_v2_v3_condition_control --v2-sensitivity reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity_smoke --v3-evaluation reports/afm_condition_control_v3/20260703_060549/evaluation_v3_smoke --out reports/afm_condition_control_v3/20260703_060549/v2_vs_v3_comparison_smoke
```

Full MVP-4 run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.analyze_condition_sensitivity_v2 --mvp3-root reports/afm_prior_v2/20260703_052537 --diffusion-checkpoint reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --condition-schema reports/afm_prior_v2/20260703_052537/latents_v2/condition_schema_v2.json --out reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity --split val --num-base-conditions 8 --num-samples-per-condition 8 --ddim-steps 100 --guidance-scales 0.0,0.5,1.0,1.5,2.0,3.0 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.condition_control_v3_utils --condition-table-v2 reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --descriptors reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv --prototypes reports/afm_prior_v2/20260703_052537/data/morphology_prototypes_v2.csv --sensitivity-summary reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity/v2_condition_sensitivity_summary.json --out reports/afm_condition_control_v3/20260703_060549/condition_schema_v3
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_latent_descriptor_regressor --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --out reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor --epochs 200 --batch-size 64 --lr 1e-3 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v3 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor/checkpoints/best.pt --out reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3 --epochs 300 --batch-size 64 --lr 1e-4 --timesteps 1000 --prediction-target v --beta-schedule cosine --cond-dropout 0.10 --aux-cond-loss-weight 0.10 --prototype-balance true --sample-every 50 --amp --ema --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v3 --diffusion-checkpoint reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor/checkpoints/best.pt --split val --num-samples-per-condition 16 --keep-top-k 4 --ddim-steps 100 --guidance-scale 2.0 --descriptor-guidance-weight 0.1 --rerank true --max-conditions 4 --out reports/afm_condition_control_v3/20260703_060549/samples_v3
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_condition_control_v3 --samples-dir reports/afm_condition_control_v3/20260703_060549/samples_v3 --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --mvp3-v2-sensitivity reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity --out reports/afm_condition_control_v3/20260703_060549/evaluation_v3
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.compare_v2_v3_condition_control --v2-sensitivity reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity --v3-evaluation reports/afm_condition_control_v3/20260703_060549/evaluation_v3 --out reports/afm_condition_control_v3/20260703_060549/v2_vs_v3_comparison
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.rerun_rheed_conditioned_with_v3_prior --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --mvp3-root reports/afm_prior_v2/20260703_052537 --v3-root reports/afm_condition_control_v3/20260703_060549 --v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --v3-autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --v3-condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor/checkpoints/best.pt --split val --num-samples-per-condition 16 --keep-top-k 4 --ddim-steps 100 --guidance-scale 2.0 --descriptor-guidance-weight 0.1 --rerank true --out reports/afm_condition_control_v3/20260703_060549/rheed_conditioned_v3_prior
```

Final tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_condition_control_v3.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
rg -n "kneighbors|nearestneighbors|nearest_neighbors" src/rheed2morph/generative/condition_control_v3_utils.py src/rheed2morph/generative/analyze_condition_sensitivity_v2.py src/rheed2morph/generative/train_latent_descriptor_regressor.py src/rheed2morph/generative/train_afm_latent_diffusion_v3.py src/rheed2morph/generative/sample_afm_prior_v3.py src/rheed2morph/generative/descriptor_guided_sampling.py src/rheed2morph/generative/evaluate_condition_control_v3.py src/rheed2morph/generative/compare_v2_v3_condition_control.py src/rheed2morph/generative/rerun_rheed_conditioned_with_v3_prior.py || true
```

## V2 Condition Sensitivity Diagnosis

Output:

- `reports/afm_condition_control_v3/20260703_060549/v2_condition_sensitivity`

Summary:

- Metric rows: `19204`
- Generated nonconstant rate: `1.000`
- Generated std mean: `0.631734`
- Guidance scales tested: `0.0`, `0.5`, `1.0`, `1.5`, `2.0`, `3.0`

Best v2 sweep metrics by descriptor:

| Descriptor | Best guidance | Pearson | MAE | Monotonicity |
| --- | ---: | ---: | ---: | ---: |
| `rq` | 0.5 | -0.0981 | 4.5255 | 0.0157 |
| `ra` | 3.0 | -0.0845 | 3.4868 | -0.0157 |
| `robust_range` | 0.0 | 0.0895 | 21.2573 | 0.0031 |
| `psd_low_power` | 1.5 | -0.1009 | 4.5173 | -0.0345 |
| `psd_mid_power` | 0.5 | 0.0829 | 5.0052 | 0.0345 |
| `psd_high_power` | 1.5 | 0.0847 | 5.0841 | 0.0282 |
| `psd_slope` | 2.0 | 0.1104 | 0.6207 | -0.0345 |
| `autocorrelation_length_px` | 2.0 | -0.0818 | 19.2998 | -0.0116 |
| `gradient_anisotropy` | 0.0 | -0.0831 | 0.1531 | -0.0533 |
| `island_count` | 2.0 | -0.1344 | 149.8908 | -0.0405 |

Interpretation:

- MVP-3 v2 samples were visually nonconstant but weakly controlled.
- Absolute correlations stayed near zero for all swept descriptors.
- Guidance scale changed statistics slightly but did not produce reliable monotonic descriptor control.
- Weak descriptors included Rq/Ra/range, PSD powers, autocorrelation length, anisotropy, and island count.

Sweep artifacts:

- `v2_condition_sensitivity/requested_vs_generated_scatter_v2.png`
- `v2_condition_sensitivity/monotonicity_curves_v2.png`
- `v2_condition_sensitivity/condition_sweep_rq.png`
- `v2_condition_sensitivity/condition_sweep_psd_slope.png`
- `v2_condition_sensitivity/condition_sweep_autocorr.png`
- `v2_condition_sensitivity/condition_sweep_anisotropy.png`
- `v2_condition_sensitivity/condition_sweep_prototype.png`

## V3 Condition Schema

Output:

- `reports/afm_condition_control_v3/20260703_060549/condition_schema_v3`

Selected descriptor columns:

```text
rq, ra, robust_range, mean_abs_gradient, gradient_std, gradient_anisotropy,
psd_low_power, psd_mid_power, psd_high_power, psd_slope,
autocorrelation_length_px, island_count, island_mean_area_px
```

Schema summary:

- Descriptor count: `13`
- Prototype count: `4`
- Condition dimension: `17`
- Standardization: train-set mean/std.
- Imputation counts: all `0`.
- V2 descriptors dropped: the high-count v2 schema descriptors not in the selected robust subset, including percentiles, central moments, height min/max, slope percentiles, PSD peak frequency, and island mean height.

Artifacts:

- `condition_schema_v3/condition_schema_v3.json`
- `condition_schema_v3/condition_table_v3.csv`
- `condition_schema_v3/condition_selection_report.md`
- `condition_schema_v3/condition_selection_metrics.csv`

## Latent Descriptor Regressor

Output:

- `reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor`

Architecture:

- Small ConvNet over standardized spatial latents `[B, 16, 16, 16]`.
- Continuous head predicts standardized v3 descriptor vector.
- Prototype head predicts 4 prototype classes.

Final metrics:

- Train loss: `0.000414`
- Val loss: `3.267454`
- Val descriptor MSE: `0.546081`
- Mean-condition baseline MSE: `0.835891`
- Beats mean-condition baseline: `True`
- Val prototype accuracy: `0.580645`

Artifacts:

- `latent_descriptor_regressor/checkpoints/best.pt`
- `latent_descriptor_regressor/checkpoints/last.pt`
- `latent_descriptor_regressor/metrics.json`
- `latent_descriptor_regressor/descriptor_regression_scatter.png`
- `latent_descriptor_regressor/prototype_confusion.png`
- `latent_descriptor_regressor/training_curves.png`
- `latent_descriptor_regressor/latent_descriptor_regressor_report.md`

## Diffusion V3

Output:

- `reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3`

Architecture and training:

- Conditional latent U-Net over `[B, 16, 16, 16]`.
- Condition embedding dimension: `256`.
- FiLM-style condition scale/shift modulation in every residual block.
- Prediction target: `v`.
- Timesteps: `1000`.
- Cosine beta schedule.
- Epochs: `300`.
- Batch size: `64`.
- Descriptor mask probability: `0.10`.
- Whole-condition dropout: `0.10`.
- Auxiliary latent descriptor consistency loss weight: `0.10`.
- Prototype-balanced sampler: enabled.
- AMP: enabled.
- EMA: enabled.

Final metrics:

- Train loss: `0.849369`
- Val loss: `0.913360`
- Train denoising loss: `0.674502`
- Val denoising loss: `0.861777`
- Train aux condition loss: `1.748665`
- Val aux condition loss: `0.515838`
- Final sample pixel std: `0.373891`
- Final sample latent std: `0.997041`
- Generated nonconstant: `True`

Sampling implementation note:

- `descriptor_guided_sampling.py` was corrected so DDIM sampling uses `GaussianDiffusionV2.predict_epsilon(...)` for both epsilon and v-prediction models.

Artifacts:

- `latent_diffusion_v3/checkpoints/best.pt`
- `latent_diffusion_v3/checkpoints/last.pt`
- `latent_diffusion_v3/checkpoints/ema_last.pt`
- `latent_diffusion_v3/training_curves.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch50.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch100.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch150.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch200.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch250.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epoch300.png`
- `latent_diffusion_v3/sample_grid_v3_oracle_val_epochfinal.png`

## V3 Sampling And Reranking

Output:

- `reports/afm_condition_control_v3/20260703_060549/samples_v3`

Sampling configuration:

- Split: `val`
- Conditions sampled: `4`
- Samples per condition: `16`
- Keep top K: `4`
- DDIM steps: `100`
- CFG guidance scale: `2.0`
- Descriptor guidance weight: `0.1`
- Reranking: enabled

Generation summary:

- Generated metric rows: `144`
- Generated nonconstant rate: `1.000`
- Generated std mean: `0.376785`
- Generated std min: `0.356079`

Descriptor error by mode:

- Plain descriptor error: `2.127632`
- Guided descriptor error: `2.124628`
- Reranked descriptor error: `2.083974`
- Reranking all-score mean: `2.124628`
- Reranking top-1 score mean: `2.069293`

Interpretation:

- Descriptor guidance alone produced only a tiny improvement over plain sampling.
- Candidate reranking produced a small but real descriptor-error improvement.
- No sample collapse was observed, but v3 samples are less contrast-rich than MVP-3 v2 samples: v3 generated std mean `0.376785` vs v2 generated std mean about `0.631862`.

Sample artifacts:

- `samples_v3/afm_prior_v3_oracle_grid_val.png`
- `samples_v3/afm_prior_v3_condition_sweep_rq.png`
- `samples_v3/afm_prior_v3_condition_sweep_psd.png`
- `samples_v3/afm_prior_v3_condition_sweep_autocorr.png`
- `samples_v3/afm_prior_v3_condition_sweep_anisotropy.png`
- `samples_v3/afm_prior_v3_prototype_grid.png`
- `samples_v3/afm_prior_v3_random_grid.png`
- `samples_v3/generated_candidates_v3.npz`
- `samples_v3/generation_metrics_v3.csv`
- `samples_v3/generation_summary_v3.json`
- `samples_v3/reranking_metrics_v3.csv`
- `samples_v3/condition_sweep_metrics_v3.csv`

## V2 vs V3 Comparison

Output:

- `reports/afm_condition_control_v3/20260703_060549/v2_vs_v3_comparison`

Comparison summary:

- Comparable descriptors: `10`
- Comparable key descriptors: `8`
- Improved abs Pearson descriptors: `autocorrelation_length_px`, `island_count`, `psd_high_power`, `psd_low_power`, `psd_mid_power`, `psd_slope`
- Improved MAE descriptors: `autocorrelation_length_px`, `gradient_anisotropy`, `island_count`, `psd_slope`

Selected descriptor comparison:

| Descriptor | V2 abs Pearson | V3 abs Pearson | V2 MAE | V3 MAE | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `rq` | 0.0981 | 0.0231 | 4.5255 | 5.1552 | worse |
| `ra` | 0.0845 | 0.0376 | 3.4868 | 3.9646 | worse |
| `robust_range` | 0.0895 | 0.0434 | 21.2573 | 24.7633 | worse |
| `psd_low_power` | 0.1009 | 0.2021 | 4.5173 | 5.9473 | mixed |
| `psd_slope` | 0.1104 | 0.1399 | 0.6207 | 0.2717 | improved |
| `autocorrelation_length_px` | 0.0818 | 0.3397 | 19.2998 | 5.3681 | improved |
| `island_count` | 0.1344 | 0.2590 | 149.8908 | 17.3403 | improved |
| `gradient_anisotropy` | 0.0831 | 0.0047 | 0.1531 | 0.0466 | mixed |

Evaluation artifacts:

- `evaluation_v3/condition_control_summary_v3.json`
- `evaluation_v3/condition_control_metrics_v3.csv`
- `evaluation_v3/requested_vs_generated_scatter_v3.png`
- `evaluation_v3/condition_sweep_summary_v3.png`
- `evaluation_v3/v2_vs_v3_descriptor_control.png`
- `evaluation_v3/v2_vs_v3_visual_comparison_grid.png`
- `evaluation_v3/failure_cases_grid_v3.png`
- `evaluation_v3/nearest_real_diagnostic_grid_v3.png`
- `v2_vs_v3_comparison/v2_vs_v3_condition_control_summary.json`
- `v2_vs_v3_comparison/v2_vs_v3_condition_control_metrics.csv`
- `v2_vs_v3_comparison/v2_vs_v3_condition_control_report.md`

## RHEED-Conditioned V3 Prior

Output:

- `reports/afm_condition_control_v3/20260703_060549/rheed_conditioned_v3_prior`

Predicted condition table:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted/predicted_condition_table_val.csv`

Condition adapter result:

- Exact descriptor-name mapping was used.
- `--fill-missing-with-train-mean` was not enabled.
- Predicted/oracle mapping filled descriptors: `[]`
- Mean-condition baseline filled all v3 descriptors from train means by design.

Generated std by mode:

| Mode | Count | Mean std | Min std | Max std |
| --- | ---: | ---: | ---: | ---: |
| `mvp3_v2_predicted` | 4 | 0.634541 | 0.625831 | 0.642497 |
| `v3_predicted_plain` | 64 | 0.369885 | 0.347993 | 0.389490 |
| `v3_predicted_guided` | 64 | 0.371291 | 0.356018 | 0.399628 |
| `v3_predicted_reranked` | 16 | 0.373991 | 0.362600 | 0.399628 |
| `v3_oracle_reranked` | 16 | 0.379638 | 0.351040 | 0.407987 |
| `v3_mean_condition` | 4 | 0.370506 | 0.368084 | 0.373497 |

Summary:

- Generated nonconstant rate: `1.000`
- V3 prior changes the output contrast/richness relative to MVP-3 v2, but predicted, oracle, and mean-conditioned v3 modes are still close.
- This remains two-stage RHEED-conditioned generation. It does not demonstrate strong RHEED predictive accuracy.

Artifacts:

- `rheed_conditioned_v3_prior/rheed_conditioned_v3_prior_grid.png`
- `rheed_conditioned_v3_prior/rheed_conditioned_v3_metrics.csv`
- `rheed_conditioned_v3_prior/condition_adapter_report.md`
- `rheed_conditioned_v3_prior/rheed_conditioned_v3_summary.json`

## Test Results

Focused v3 tests:

```text
Ran 9 tests in 0.900s
OK
```

Full test discovery:

```text
Ran 55 tests in 9.968s
OK
```

The full suite emitted existing NumPy/sklearn warnings in tiny synthetic tests, but no failures.

KNN marker scan:

- No matches for `kneighbors`, `nearestneighbors`, or `nearest_neighbors` in the v3 condition-control path.

## Acceptance Check

- Unit tests pass: yes.
- Existing tests pass: yes.
- V2 condition sensitivity diagnostic exists: yes, `v2_condition_sensitivity/v2_condition_sensitivity_summary.json`.
- `condition_schema_v3.json` exists: yes.
- Latent descriptor regressor checkpoint exists: yes, `latent_descriptor_regressor/checkpoints/best.pt`.
- Diffusion v3 checkpoint exists: yes, `latent_diffusion_v3/checkpoints/ema_last.pt`.
- At least one condition sweep grid exists: yes, multiple v2 and v3 sweep grids exist.
- `generation_summary_v3.json` exists: yes.
- `condition_control_summary_v3.json` exists: yes.
- V2 vs V3 comparison attempted and documented: yes.
- RHEED-conditioned v3 prior comparison attempted and documented: yes.
- New v3 path does not use KNN: yes.
- No exact AFM reconstruction claim is made: yes.

## Known Limitations

- Condition control remains weak for Rq, Ra, and robust range.
- The largest v3 gains are partial and mostly appear in PSD slope, autocorrelation length, island count, and reranking.
- V3 samples are noncollapsed but less visually rich/contrasty than MVP-3 v2 in generated std.
- Descriptor errors still compare requested physical descriptors to generated normalized decoded maps, so absolute calibration is imperfect.
- Descriptor guidance and reranking help more than the diffusion model's intrinsic conditioning.
- The RHEED-conditioned comparison reuses MVP-2 predicted descriptors and does not retrain or validate a stronger RHEED encoder.
- The `nearest_real_diagnostic_grid_v3.png` artifact is labeled diagnostic only; no nearest-neighbor retrieval is used for generation.

## Recommended Next Command

Smallest next fix: continue v3 diffusion with a lower auxiliary condition weight and a longer schedule, then resample with the same evaluator. This should test whether the weaker Rq/Ra/range control is due to undertraining/over-regularized aux loss or the normalized-height calibration mismatch.

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v3 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor/checkpoints/best.pt --out reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3_continue_aux002 --epochs 600 --batch-size 64 --lr 5e-5 --timesteps 1000 --prediction-target v --beta-schedule cosine --cond-dropout 0.15 --aux-cond-loss-weight 0.02 --prototype-balance true --sample-every 50 --amp --ema --resume reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/last.pt --seed 42
```
