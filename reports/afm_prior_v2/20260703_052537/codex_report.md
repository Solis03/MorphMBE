# AFM Prior V2 MVP-3 Report

Run root: `reports/afm_prior_v2/20260703_052537`

## Scope Statement

MVP-3 improves and evaluates the AFM generative prior. It does not by itself prove strong RHEED-to-AFM predictive accuracy.

The new path does not use neighbor retrieval in generation. Old MVP-1 and MVP-2 code and artifacts were not deleted or overwritten.

## Environment And Git Status

- `pwd`: `/home/wangziyi/MorphMBE/MorphMBE`
- Python: `Python 3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`

Git status before MVP-3 implementation:

```text
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

Git status after MVP-3 implementation and run:

```text
?? reports/afm_prior_v2/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

## Files Created Or Modified

New MVP-3 source files:

- `src/rheed2morph/generative/afm_prior_v2_utils.py`
- `src/rheed2morph/generative/models/afm_autoencoder_v2.py`
- `src/rheed2morph/generative/diffusion_v2.py`
- `src/rheed2morph/generative/prepare_afm_prior_v2_dataset.py`
- `src/rheed2morph/generative/train_afm_autoencoder_v2.py`
- `src/rheed2morph/generative/export_afm_latents_v2.py`
- `src/rheed2morph/generative/train_afm_latent_diffusion_v2.py`
- `src/rheed2morph/generative/sample_afm_prior_v2.py`
- `src/rheed2morph/generative/evaluate_afm_prior_v2.py`
- `src/rheed2morph/generative/compare_mvp1_mvp3_generation.py`

New test:

- `tests/test_generative_afm_prior_v2.py`

New artifact tree:

- `reports/afm_prior_v2/20260703_052537/`

## Dependency Summary

MVP-1 report read:

- `reports/conditional_latent_diffusion_mvp/20260703_041331/codex_report.md`

MVP-2 report read:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/codex_report.md`

MVP-1 used only `36` AFM files in its final 5-epoch run because its data prep started from `data/manifests/manifest_1um_one_to_one.csv`, which selected one representative one-to-one RHEED/AFM sample per group rather than all AFM height maps.

MVP-2 inherited the same 36 AFM condition rows because it paired RHEED inputs to MVP-1 condition rows. The AFM prior was therefore effectively limited to paired RHEED-AFM samples.

MVP-1 condition schema:

- Latent shape: `[8, 16, 16]`
- Descriptor condition columns: `25` standardized `cond_*` columns
- Prototype count: `6`
- Condition dimension: `31`
- Condition table columns: `row_id`, `sample_id`, `group_id`, `split`, `network_input_path`, `descriptor_height_path`, `prototype_id`, plus 25 raw descriptor columns and 25 standardized `cond_*` columns.

All processed AFM height maps are principally under:

- `data/processed_afm/<sample_id>/<afm_file_id>/<afm_file_id>_height.npy`
- `data/plane_corrected_afm/<sample_id>/<afm_file_id>/<afm_file_id>_plane_corrected.npy`
- Metadata summaries: `data/processed_afm/afm_summary.csv`, `data/afm_descriptor_reconstruction_large/large_afm_manifest.csv`, and `data/manifests/afm_candidate_table_complete.csv`

No `network_input.npy` files were found or needed for the full v2 data run.

## Exact Commands Run

Focused tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_afm_prior_v2.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Smoke run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_afm_prior_v2_dataset --out reports/afm_prior_v2/20260703_052537/data_smoke --scan-size-filter 1um --image-size 128 --limit 32 --min-files-required 20 --strict true --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_autoencoder_v2 --data-index reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_descriptors.csv --out reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2_smoke --image-size 128 --latent-channels 8 --latent-size 16 --epochs 1 --batch-size 8 --amp --ema --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.export_afm_latents_v2 --checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2_smoke/checkpoints/best.pt --data-index reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_descriptors.csv --prototypes reports/afm_prior_v2/20260703_052537/data_smoke/morphology_prototypes_v2.csv --out reports/afm_prior_v2/20260703_052537/latents_v2_smoke
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v2 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2_smoke --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2_smoke/checkpoints/best.pt --out reports/afm_prior_v2/20260703_052537/latent_diffusion_v2_smoke --epochs 1 --batch-size 16 --lr 1e-4 --timesteps 100 --amp --ema --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v2 --diffusion-checkpoint reports/afm_prior_v2/20260703_052537/latent_diffusion_v2_smoke/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2_smoke/checkpoints/best.pt --condition-table reports/afm_prior_v2/20260703_052537/latents_v2_smoke/condition_table_v2.csv --split val --num-samples-per-condition 4 --ddim-steps 10 --guidance-scale 1.5 --max-conditions 2 --out reports/afm_prior_v2/20260703_052537/samples_v2_smoke
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_afm_prior_v2 --samples-dir reports/afm_prior_v2/20260703_052537/samples_v2_smoke --real-index reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data_smoke/afm_prior_v2_descriptors.csv --out reports/afm_prior_v2/20260703_052537/evaluation_v2_smoke
```

Full MVP-3 run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_afm_prior_v2_dataset --out reports/afm_prior_v2/20260703_052537/data --scan-size-filter 1um --image-size 128 --min-files-required 60 --strict true --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_autoencoder_v2 --data-index reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv --out reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2 --image-size 128 --latent-channels 16 --latent-size 16 --epochs 100 --batch-size 32 --lr 2e-4 --weight-decay 1e-4 --early-stop-patience 20 --amp --ema --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.export_afm_latents_v2 --checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --data-index reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv --prototypes reports/afm_prior_v2/20260703_052537/data/morphology_prototypes_v2.csv --out reports/afm_prior_v2/20260703_052537/latents_v2
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v2 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --out reports/afm_prior_v2/20260703_052537/latent_diffusion_v2 --epochs 200 --batch-size 64 --lr 1e-4 --weight-decay 1e-4 --timesteps 1000 --beta-schedule cosine --prediction-target epsilon --cond-dropout 0.15 --sample-every 50 --amp --ema --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v2 --diffusion-checkpoint reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --split val --num-samples-per-condition 8 --ddim-steps 100 --guidance-scale 1.5 --max-conditions 4 --out reports/afm_prior_v2/20260703_052537/samples_v2
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v2 --diffusion-checkpoint reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_prior_v2/20260703_052537/latents_v2/condition_table_v2.csv --split test --num-samples-per-condition 8 --ddim-steps 100 --guidance-scale 1.5 --max-conditions 4 --out reports/afm_prior_v2/20260703_052537/samples_v2_test
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_afm_prior_v2 --samples-dir reports/afm_prior_v2/20260703_052537/samples_v2 --real-index reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_index.csv --descriptors reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv --out reports/afm_prior_v2/20260703_052537/evaluation_v2
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.compare_mvp1_mvp3_generation --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --mvp1-diffusion reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt --mvp1-autoencoder reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --mvp3-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --mvp3-autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --mvp3-condition-schema reports/afm_prior_v2/20260703_052537/latents_v2/condition_schema_v2.json --split val --num-samples-per-condition 4 --ddim-steps 100 --guidance-scale 1.5 --max-conditions 4 --out reports/afm_prior_v2/20260703_052537/mvp2_rheed_conditioned_with_v2_prior
```

## AFM Data Discovery

Full data directory:

- `reports/afm_prior_v2/20260703_052537/data`

Discovery result:

- Candidate path count before filter: `780`
- Candidate count after `1um` scan filter: `336`
- Deduplicated source AFM files indexed: `168`
- Increase over MVP-1 final 36-file prior run: `+132`
- Groups: `36`
- Physical height maps: `168`
- Network input fallbacks: `0`
- PNG fallbacks: `0`
- Load failures: `0`

Split counts:

- Train: `115` files, `25` groups
- Val: `31` files, `5` groups
- Test: `22` files, `6` groups

Scan sizes accepted by the `1um` filter:

- `0.891`, `1.0`, `1.016` um

The split is group-based. Files from the same `group_id` do not cross train/val/test.

## Descriptor And Prototype Summary

Descriptor columns:

```text
height_mean, height_std, rq, ra, peak_to_valley, p01, p05, p50, p95, p99,
robust_range, skewness, kurtosis, mean_abs_gradient, gradient_std,
gradient_orientation_entropy, gradient_anisotropy, psd_low_power,
psd_mid_power, psd_high_power, psd_slope, autocorrelation_length_px,
island_coverage, island_count, island_mean_area_px, height_min, height_max,
slope_p50, slope_p95, slope_p99, psd_peak_frequency, island_mean_height
```

Descriptor imputation counts: all `0`.

Prototype clustering:

- Candidate K values: `4`, `6`, `8`
- Selected K: `4`
- Silhouette scores: K=4 `0.2607`, K=6 `0.2280`, K=8 `0.2180`
- Cluster counts: `{0: 27, 1: 55, 2: 5, 3: 81}`

Descriptor/prototype artifacts:

- `reports/afm_prior_v2/20260703_052537/data/afm_prior_v2_descriptors.csv`
- `reports/afm_prior_v2/20260703_052537/data/descriptor_scaler_v2.json`
- `reports/afm_prior_v2/20260703_052537/data/morphology_prototypes_v2.csv`
- `reports/afm_prior_v2/20260703_052537/data/descriptor_histograms_train_val.png`
- `reports/afm_prior_v2/20260703_052537/data/descriptor_correlation_matrix.png`
- `reports/afm_prior_v2/20260703_052537/data/prototype_examples_grid.png`

## Autoencoder V2

Autoencoder output:

- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2`

Architecture:

- Input: `[B, 1, 128, 128]`
- ResNet-style encoder/decoder with GroupNorm, SiLU, small dropout
- Downsample path: `128 -> 64 -> 32 -> 16`
- Latent: `[B, 16, 16, 16]`
- Decoder output: `[B, 1, 128, 128]` in `[-1, 1]`
- EMA enabled

Loss:

```text
1.0 * L1
+ 0.25 * gradient_l1
+ 0.10 * log_psd_l1
+ 0.10 * roughness_consistency
+ 0.05 * histogram_loss
+ 0.05 * multiscale_l1
```

Final metrics:

- Epochs: `100`
- Train loss: `0.109116`
- Val loss: `0.127982`
- Val L1: `0.099514`
- Val gradient L1: `0.076015`
- Val PSD L1: `0.020399`
- Val histogram loss: `0.003129`
- Val roughness error: `0.033872`
- Original pixel std on recon grid: `0.410562`
- Reconstructed pixel std: `0.268022`
- Collapse warning: `false`

Comparison to MVP-1:

- MVP-1 5-epoch AE val L1 was `0.386358` on the old 36-file representative set.
- MVP-3 AE val L1 is `0.099514` on the broader 168-file v2 set. This is not a perfectly controlled comparison, but it is a large practical reconstruction improvement.
- Visually, MVP-3 reconstructions preserve coarse morphology and PSD structure better, but still smooth high-contrast island edges.

AE artifacts:

- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/last.pt`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/ema_best.pt`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/recon_grid_val.png`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/recon_grid_test.png`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/descriptor_reconstruction_scatter.png`
- `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/latent_stats_preview.json`

## Latent Export V2

Latent output:

- `reports/afm_prior_v2/20260703_052537/latents_v2`

Latent schema:

- Latent shape: `[16, 16, 16]`
- Train latent count: `115`
- Val latent count: `31`
- Test latent count: `22`
- Latents standardized using train-set latent mean/std
- Descriptor conditions: `32`
- Prototype one-hot count: `4`
- Condition dimension: `36`

Files:

- `latents_train.npz`
- `latents_val.npz`
- `latents_test.npz`
- `latent_standardization_v2.npz`
- `latent_stats_v2.json`
- `condition_table_v2.csv`
- `condition_schema_v2.json`

## Latent Diffusion V2

Diffusion output:

- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2`

Architecture and training:

- Conditional latent U-Net over `[B, 16, 16, 16]`
- Timestep embedding plus descriptor/prototype condition MLP
- Cosine beta schedule
- Epsilon prediction target
- Timesteps: `1000`
- Epochs: `200`
- Batch size: `64`
- Condition dropout: `0.15`
- AMP: enabled
- EMA: enabled

Final metrics:

- Train denoising loss: `0.416426`
- Val denoising loss: `0.401060`
- EMA val loss at final epoch: `0.922755`
- Final training sample std: `0.638744`
- Generated nonconstant: `true`

Note: the raw model val loss improved more than the EMA validation loss in this short run. Sampling used `ema_last.pt` because the EMA samples were visually stable/nonconstant, but the EMA lag suggests future runs should tune EMA decay or warmup.

Diffusion artifacts:

- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/best.pt`
- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/last.pt`
- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt`
- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/training_curves.png`
- `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/sample_grid_oracle_val_epochfinal.png`

## Sampling And AFM Prior Evaluation

Sample output:

- `reports/afm_prior_v2/20260703_052537/samples_v2`

Sampling configuration:

- Split: `val`
- Conditions sampled: `4`
- Samples per condition: `8`
- DDIM steps: `100`
- Guidance scale: `1.5`

Generation summary:

- Generated samples: `72`
- Generated std mean: `0.631862`
- Generated std min: `0.615581`
- Nonconstant rate: `1.000`

Sample grids:

- `reports/afm_prior_v2/20260703_052537/samples_v2/afm_prior_v2_oracle_grid_val.png`
- `reports/afm_prior_v2/20260703_052537/samples_v2/afm_prior_v2_prototype_grid.png`
- `reports/afm_prior_v2/20260703_052537/samples_v2/afm_prior_v2_random_grid.png`
- Test split sample directory: `reports/afm_prior_v2/20260703_052537/samples_v2_test`

Evaluation output:

- `reports/afm_prior_v2/20260703_052537/evaluation_v2`

Evaluation metrics:

- Generated nonconstant rate: `1.000`
- Mean absolute real-vs-generated descriptor mean delta: `13.043643`
- Descriptor two-sample diagnostic accuracy: `0.995833`
- Pairwise generated descriptor distance mean: `26.305845`
- Near duplicate rate: `0.0`

Interpretation:

- MVP-3 generates visually rich, nonconstant AFM-like texture.
- The generated distribution is still easy to distinguish from real AFM in descriptor space.
- Descriptor consistency is weak in physical units because generated images are decoded normalized tensors, not calibrated nm height maps.
- Samples appear over-contrasty relative to AE reconstructions and predicted/oracle/mean conditions are not visually separated strongly.

Evaluation artifacts:

- `descriptor_distribution_comparison.png`
- `psd_distribution_comparison.png`
- `real_vs_generated_umap_or_pca.png`
- `nearest_real_generated_grid.png`
- `failure_cases_grid.png`

Closest-real visualization is diagnostic only. Generated samples come from diffusion sampling, not retrieval.

## RHEED-Conditioned Regeneration With V2 Prior

Comparison output:

- `reports/afm_prior_v2/20260703_052537/mvp2_rheed_conditioned_with_v2_prior`

MVP-2 predicted table reused:

- `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted/predicted_condition_table_val.csv`

Condition adapter:

- Shared descriptors mapped: `25`
- V2-only descriptors filled with MVP-3 train mean: `height_min`, `height_max`, `slope_p50`, `slope_p95`, `slope_p99`, `psd_peak_frequency`, `island_mean_height`
- Prototype policy: zero prototype vector, because MVP-1 and MVP-3 prototype ids are not semantically aligned.
- Adapter report: `reports/afm_prior_v2/20260703_052537/mvp2_rheed_conditioned_with_v2_prior/condition_adapter_report.md`

Comparison grid:

- `reports/afm_prior_v2/20260703_052537/mvp2_rheed_conditioned_with_v2_prior/mvp1_vs_mvp3_rheed_conditioned_grid.png`

Generated std by mode:

- MVP-1 predicted-conditioned: mean `0.198070`
- MVP-3 predicted-conditioned: mean `0.637856`
- MVP-3 oracle-conditioned: mean `0.636142`
- MVP-3 mean-conditioned: mean `0.633124`

Interpretation:

- MVP-3 visually improves texture richness and contrast over the MVP-1 5-epoch prior.
- MVP-3 predicted, oracle, and mean-condition samples are too similar, so this run does not show strong descriptor/prototype controllability.
- This task did not retrain the RHEED encoder.

## Test Results

Focused MVP-3 tests:

```text
Ran 9 tests in 8.353s
OK
```

Full test discovery:

```text
Ran 46 tests in 9.903s
OK
```

The full suite emitted existing NumPy/sklearn warnings in small synthetic or older tests, but no failures.

## Acceptance Check

- Unit tests pass: yes.
- Existing tests pass: yes.
- Dataset discovery found substantially more than 36 AFM files: yes, `168`.
- Improved AE recon grid exists: yes, `afm_autoencoder_v2/recon_grid_val.png`.
- V2 diffusion sample grid exists: yes, `samples_v2/afm_prior_v2_oracle_grid_val.png`.
- `generation_summary_v2.json` exists: yes, under `samples_v2/` and `evaluation_v2/`.
- MVP-1 vs MVP-3 comparison attempted: yes, succeeded.
- New v2 path avoids retrieval generation: yes.
- No exact AFM reconstruction claim is made: yes.

## Known Limitations

- Generated descriptors are computed on normalized decoded images, while requested descriptors are physical nm descriptors. Absolute descriptor errors are therefore not physically calibrated yet.
- The v2 prior improves visual richness but overshoots contrast/roughness and remains descriptor-distribution mismatched.
- The condition adapter can only map the 25 MVP-1/MVP-2 descriptors shared with MVP-3. Seven v2 descriptors are filled from train means for RHEED-conditioned comparison.
- MVP-3 prototypes are new v2 clusters and are not aligned with MVP-1 prototype ids.
- The EMA diffusion checkpoint produced stable visual samples, but EMA validation loss lagged raw validation loss.
- No human scientific validation of morphology classes was performed.

## Recommended Next Command

Smallest next fix: calibrate generated normalized maps back to physical height units or train the AE/diffusion with an explicit physical descriptor consistency term on decoded samples. After that, run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v2 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --out reports/afm_prior_v2/20260703_052537/latent_diffusion_v2_continue --epochs 400 --batch-size 64 --lr 5e-5 --timesteps 1000 --beta-schedule cosine --prediction-target epsilon --cond-dropout 0.25 --sample-every 50 --amp --ema --seed 42
```

Use the generated grids and descriptor distribution plots to decide whether lower learning rate plus stronger condition dropout improves controllability without further increasing contrast.
