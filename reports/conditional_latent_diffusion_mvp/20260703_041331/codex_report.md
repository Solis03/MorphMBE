# Conditional Latent Diffusion MVP-1 Report

Run root: `reports/conditional_latent_diffusion_mvp/20260703_041331`

## Scope Statement

This MVP validates AFM descriptor-conditioned latent diffusion only. It does not yet claim RHEED-conditioned AFM prediction.

The old RHEED-to-AFM latent/KNN baseline was not deleted or rewritten. The new implementation lives under `src/rheed2morph/generative/` and uses AFM-derived descriptor/prototype oracle conditions.

## Environment And Initial State

- `pwd`: `/home/wangziyi/MorphMBE/MorphMBE`
- Git status before implementation: clean (`git status --short` returned no rows)
- Python: `Python 3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`
- Quick data search:
  - `rg --files -g '*manifest*' -g '*.csv' -g '*.npy' -g '*.npz'` found existing manifests, processed AFM summaries, prior latent arrays, and one-to-one outputs.
  - `find data -iname '*manifest*.csv' | wc -l`: `71`
  - `find data -name '*.npy' | wc -l`: `2318`
  - `find data -name 'network_input.npy' -o -name 'processed_zsensor_nm.npy' -o -name '*plane_corrected.npy' | wc -l`: `260`

## Files Created Or Modified

New source package:

- `src/rheed2morph/generative/__init__.py`
- `src/rheed2morph/generative/common.py`
- `src/rheed2morph/generative/afm_descriptors.py`
- `src/rheed2morph/generative/prepare_afm_latent_dataset.py`
- `src/rheed2morph/generative/models/__init__.py`
- `src/rheed2morph/generative/models/afm_autoencoder.py`
- `src/rheed2morph/generative/losses.py`
- `src/rheed2morph/generative/train_afm_autoencoder.py`
- `src/rheed2morph/generative/export_afm_latents.py`
- `src/rheed2morph/generative/models/latent_unet.py`
- `src/rheed2morph/generative/diffusion.py`
- `src/rheed2morph/generative/train_afm_latent_diffusion.py`
- `src/rheed2morph/generative/sample_afm_latent_diffusion.py`
- `src/rheed2morph/generative/visualization.py`

New test:

- `tests/test_generative_afm_latent_diffusion.py`

New report/artifact tree:

- `reports/conditional_latent_diffusion_mvp/20260703_041331/`

Git status after implementation:

```text
?? reports/conditional_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
```

## Exact Commands Run

Environment capture:

```bash
pwd
git status --short
.venv/bin/python --version
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
rg --files -g '*manifest*' -g '*.csv' -g '*.npy' -g '*.npz' | sed -n '1,240p'
find . -name 'network_input.npy' -o -name 'processed_zsensor_nm.npy' -o -name '*afm*.npy' | sed -n '1,240p'
```

Tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_afm_latent_diffusion.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Initial 32-sample smoke run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_afm_latent_dataset --out reports/conditional_latent_diffusion_mvp/20260703_041331/data --scan-size-filter 1um --image-size 128 --limit 32 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_autoencoder --data-index reports/conditional_latent_diffusion_mvp/20260703_041331/data/data_index.csv --out reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder --image-size 128 --latent-channels 8 --epochs 1 --batch-size 8 --amp --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.export_afm_latents --checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder/checkpoints/best.pt --data-index reports/conditional_latent_diffusion_mvp/20260703_041331/data/data_index.csv --descriptors reports/conditional_latent_diffusion_mvp/20260703_041331/data/afm_descriptors.csv --out reports/conditional_latent_diffusion_mvp/20260703_041331/latents
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion --latents-dir reports/conditional_latent_diffusion_mvp/20260703_041331/latents --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder/checkpoints/best.pt --out reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion --epochs 1 --batch-size 16 --timesteps 100 --amp --quick --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_latent_diffusion --diffusion-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion/checkpoints/last.pt --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder/checkpoints/best.pt --condition-table reports/conditional_latent_diffusion_mvp/20260703_041331/latents/condition_table.csv --split val --num-samples-per-condition 4 --ddim-steps 10 --guidance-scale 1.5 --max-conditions 4 --out reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion/samples
```

Bounded 5-epoch GPU run:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_afm_latent_dataset --out reports/conditional_latent_diffusion_mvp/20260703_041331/data_limit64 --scan-size-filter 1um --image-size 128 --limit 64 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_autoencoder --data-index reports/conditional_latent_diffusion_mvp/20260703_041331/data_limit64/data_index.csv --out reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch --image-size 128 --latent-channels 8 --epochs 5 --batch-size 16 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.export_afm_latents --checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --data-index reports/conditional_latent_diffusion_mvp/20260703_041331/data_limit64/data_index.csv --descriptors reports/conditional_latent_diffusion_mvp/20260703_041331/data_limit64/afm_descriptors.csv --out reports/conditional_latent_diffusion_mvp/20260703_041331/latents_5epoch
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion --latents-dir reports/conditional_latent_diffusion_mvp/20260703_041331/latents_5epoch --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --out reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch --epochs 5 --batch-size 32 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_latent_diffusion --diffusion-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/checkpoints/last.pt --autoencoder-checkpoint reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/checkpoints/best.pt --condition-table reports/conditional_latent_diffusion_mvp/20260703_041331/latents_5epoch/condition_table.csv --split val --num-samples-per-condition 4 --ddim-steps 50 --guidance-scale 1.5 --max-conditions 4 --out reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/samples
```

## Data Inventory

Primary 5-epoch run data directory: `reports/conditional_latent_diffusion_mvp/20260703_041331/data_limit64`

- Manifest: `data/manifests/manifest_1um_one_to_one.csv`
- Scan size filter: `1um`
- AFM files indexed: `36`
- Sample groups: `36`
- Split counts by sample: train `25`, val `5`, test `6`
- Split counts by group: train `25`, val `5`, test `6`
- `group_id` determination: `column:group_id` for all 36 rows
- Physical height maps: `36`
- Existing network inputs used as primary source: `0`
- PNG fallbacks: `0`
- Skipped rows: `0`

Initial smoke data directory: `reports/conditional_latent_diffusion_mvp/20260703_041331/data`

- AFM files indexed: `32`
- Groups: `32`
- Split counts by sample/group: train `22`, val `5`, test `5`
- Physical height maps: `32`; network inputs `0`; PNG fallbacks `0`

## Descriptor Summary

Descriptor columns:

`height_mean`, `height_std`, `rq`, `ra`, `peak_to_valley`, `p01`, `p05`, `p50`, `p95`, `p99`, `robust_range`, `skewness`, `kurtosis`, `mean_abs_gradient`, `gradient_std`, `gradient_orientation_entropy`, `gradient_anisotropy`, `psd_low_power`, `psd_mid_power`, `psd_high_power`, `psd_slope`, `autocorrelation_length_px`, `island_coverage`, `island_count`, `island_mean_area_px`

- NaN/imputation counts: all descriptor columns had `0` imputed values in both the 32-sample smoke and 36-sample 5-epoch runs.
- 32-sample prototype clustering: `K=5`, counts `{0: 8, 1: 12, 2: 1, 3: 6, 4: 5}`
- 36-sample prototype clustering: `K=6`, counts `{0: 12, 1: 14, 2: 5, 3: 1, 4: 2, 5: 2}`

## Autoencoder Metrics

Initial smoke autoencoder:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder`
- Checkpoints: `checkpoints/best.pt`, `checkpoints/last.pt`
- Train loss: `0.600891`
- Val loss: `0.568937`
- Val L1: `0.511319`
- Val gradient L1: `0.098659`
- Val PSD L1: `0.058050`
- Val roughness error: `0.300509`
- Reconstruction grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder/recon_grid_val.png`
- Target pixel std: `0.428717`
- Reconstructed pixel std: `0.231140`
- Collapse warning: `false`

5-epoch autoencoder:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch`
- Checkpoints: `checkpoints/best.pt`, `checkpoints/last.pt`
- Final train loss: `0.428873`
- Final val loss: `0.451893`
- Final val L1: `0.386358`
- Final val gradient L1: `0.109475`
- Final val PSD L1: `0.066653`
- Final val roughness error: `0.348339`
- Reconstruction grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/afm_autoencoder_5epoch/recon_grid_val.png`
- Target pixel std: `0.453662`
- Reconstructed pixel std: `0.231658`
- Collapse warning: `false`

## Latent Export

Initial smoke export:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/latents`
- Latent shape: `[8, 16, 16]`
- Train latent count: `22`
- Files: `latents_train.npz`, `latents_val.npz`, `latents_test.npz`, `latent_standardization.npz`, `latent_stats.json`, `condition_table.csv`

5-epoch export:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/latents_5epoch`
- Latent shape: `[8, 16, 16]`
- Train latent count: `25`

## Latent Diffusion Metrics

Initial smoke diffusion:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion`
- Schedule used for speed: `100` timesteps
- Final train denoising loss: `1.180721`
- Final val denoising loss: `1.122460`
- Training sample grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion/sample_grid_val.png`
- Standalone sample grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion/samples/sample_grid_val.png`
- Generation metrics: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion/samples/generation_metrics.csv`
- Generated outputs nonconstant: `true`
- Generated std mean/min from standalone sampler: `0.146137` / `0.139449`

5-epoch diffusion:

- Output: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch`
- Schedule: default `1000` timesteps
- Final train denoising loss: `1.045098`
- Final val denoising loss: `1.042429`
- Training sample grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/sample_grid_val.png`
- Standalone sample grid: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/samples/sample_grid_val.png`
- Generation metrics: `reports/conditional_latent_diffusion_mvp/20260703_041331/latent_diffusion_5epoch/samples/generation_metrics.csv`
- Generated outputs nonconstant: `true`
- Generated std mean/min from standalone sampler: `0.205949` / `0.174708`

## Test Results

Focused generative test:

```text
Ran 7 tests in 1.402s
OK
```

Full unittest discovery:

```text
Ran 30 tests in 1.236s
OK
```

The full suite emitted existing sklearn convergence/runtime warnings in older tests, but no failures.

## Acceptance Criteria Check

- Unit tests pass: yes.
- At least one `recon_grid_val.png` exists: yes, both quick and 5-epoch autoencoder runs.
- At least one diffusion sample grid exists: yes, both quick and 5-epoch diffusion runs.
- New path does not use nearest-neighbor retrieval: yes. The new test scans `src/rheed2morph/generative/` for nearest-neighbor code markers.
- Final report exists: yes, this file.
- Generated samples come from diffusion sampling and fixed AFM decoder: yes.

## Known Limitations And Failures

- No command failed during implementation or smoke/5-epoch runs.
- The 5-epoch run is still a data-flow and first-artifact validation, not final model quality.
- The decoder is trained on robust normalized AFM tensors in `[-1, 1]`. Generated descriptor metrics are computed on decoded normalized images, while oracle condition descriptors are extracted from physical/plane-corrected height arrays. Absolute roughness consistency should therefore not be overinterpreted until a physical-height calibration path is added.
- Prototype labels are descriptor clusters, not physically validated morphology classes.
- No RHEED condition encoder is implemented in this MVP.

## Recommended Longer RTX 5090 Run

For a longer all-available 1um run, use a fresh timestamped root and remove the smoke limits:

```bash
RUN=reports/conditional_latent_diffusion_mvp/$(date -u +%Y%m%d_%H%M%S)
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.prepare_afm_latent_dataset --out "$RUN/data" --scan-size-filter 1um --image-size 128 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_autoencoder --data-index "$RUN/data/data_index.csv" --out "$RUN/afm_autoencoder" --image-size 128 --latent-channels 8 --epochs 50 --batch-size 32 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.export_afm_latents --checkpoint "$RUN/afm_autoencoder/checkpoints/best.pt" --data-index "$RUN/data/data_index.csv" --descriptors "$RUN/data/afm_descriptors.csv" --out "$RUN/latents"
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion --latents-dir "$RUN/latents" --autoencoder-checkpoint "$RUN/afm_autoencoder/checkpoints/best.pt" --out "$RUN/latent_diffusion" --epochs 100 --batch-size 64 --timesteps 1000 --amp --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_latent_diffusion --diffusion-checkpoint "$RUN/latent_diffusion/checkpoints/last.pt" --autoencoder-checkpoint "$RUN/afm_autoencoder/checkpoints/best.pt" --condition-table "$RUN/latents/condition_table.csv" --split val --num-samples-per-condition 4 --ddim-steps 50 --guidance-scale 1.5 --out "$RUN/latent_diffusion/samples"
```
