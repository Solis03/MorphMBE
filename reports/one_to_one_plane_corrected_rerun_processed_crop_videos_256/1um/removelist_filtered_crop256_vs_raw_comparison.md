# Removelist-filtered Crop256 RHEED Experiment

## 过滤规则

- Source manifest: `data/manifests/manifest_1um_one_to_one.csv`
- Filtered manifest: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/manifest_1um_one_to_one_removelist_filtered.csv`
- `removelist.txt` ids: `6018, 6061, 6104, 6066, 6087, 6023, 6068`
- Removed from 1um manifest: `6023, 6066, 6068, 6087`
- Not present in 1um manifest: `6018, 6061, 6104`
- Rows: `36` -> `32`

## AFM Latent Target

这次没有复用旧的 36-sample AFM latents，而是基于过滤后的 32-sample manifest 重新训练 AFM autoencoder。

- Train rows / groups: `25` / `25`
- Val rows / groups: `7` / `7`
- Best epoch: `27`
- Best val loss: `0.477923`
- Reconstruction MAE: `0.730571`
- Reconstruction std ratio: `0.3821`
- Quality warning: `none`

## Latent Benchmark

| Variant | Source | Model | Train/Test | Learned MSE | Learned cosine | Nearest distance | Nearest cosine | Retrieved MSE | Top-k hit | Beats mean latent? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw_filtered | removelist-filtered raw video + 8 frames + mean/std | `ridge` | 24/8 | `0.396026` | `0.702746` | `1.748233` | `0.676758` | `0.415456` | `0.375000` | no |
| crop256_filtered | removelist-filtered ROI shadow-right v2 raw-crop videos 256 + 64 frames + temporal stats | `knn` | 24/8 | `0.754014` | `0.455213` | `1.882114` | `0.456461` | `0.699739` | `0.250000` | no |

Mean-latent baseline for this filtered split:

- MSE / cosine: `0.364814` / `0.775990`

## Descriptor Diagnostic

Crop256 descriptor MVP:

- Embedded samples: `32`
- Best model: `knn {"n_neighbors": 5}`
- Learned descriptor mean MAE / RMSE / R^2: `0.9036 / 1.2667 / -0.4624`

Raw descriptor MVP:

- Embedded samples: `32`
- Best model: `knn {"n_neighbors": 7}`
- Learned descriptor mean MAE / RMSE / R^2: `0.8553 / 1.2068 / -0.2266`

## 结论

在剔除 `removelist.txt` 指定样本并重训 AFM latent target 后，本次 crop256 ROI 视频结果没有提升。

- Raw filtered learned MSE / cosine: `0.396026` / `0.702746`
- Crop256 filtered learned MSE / cosine: `0.754014` / `0.455213`
- Mean-latent baseline MSE / cosine: `0.364814` / `0.775990`

也就是说，旧的 crop256 改善结论依赖了本应移除的样本；过滤后 crop256 不仅没有超过 raw baseline，也没有超过 mean-latent sanity baseline。

## 产物位置

- Filtered manifest: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/manifest_1um_one_to_one_removelist_filtered.csv`
- Crop256 metrics: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/metrics.json`
- Raw filtered metrics: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/raw_baseline_rheed_to_afm_latent/metrics.json`
- Comparison CSV: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/removelist_filtered_crop256_vs_raw_metrics.csv`
- Standard raw-vs-crop report: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/processed_vs_raw_comparison.md`
