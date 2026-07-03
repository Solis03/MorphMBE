# Crop256 RHEED vs Previous Results

## 实验口径

- Manifest: `data/manifests/manifest_1um_one_to_one.csv`
- AFM latent target: `reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/afm_latents.npy`
- AFM latent index: `reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/afm_latent_index.csv`
- Split: `GroupShuffleSplit(random_state=42)`，train/test 为 `27/9`
- Latent predictor candidates: `ridge / knn / mlp`
- 本次新输入: `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256`
- 本次 embedding: `64` 帧采样 + ResNet50 frozen encoder + `mean/std/delta_mean/delta_std` temporal stats，输出维度 `8192`

## 数据覆盖

- Manifest samples: `36`
- New crop256 dataset dirs: `62`
- Mapped samples: `36`
- Embedded samples: `36`
- Mapping failed: `0`
- Skipped samples: `0`

## Latent Benchmark

| Variant | Source | Model | Learned latent MSE | Learned cosine | Nearest latent distance | Nearest latent cosine | Retrieved latent MSE | Top-k hit rate | Beats mean latent? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | raw video + 8 frames + mean/std | `knn` | `0.451245` | `0.686244` | `1.370465` | `0.678092` | `0.503228` | `0.333333` | no |
| old processed npz | processed clean_frames + 64 frames + temporal stats | `knn` | `0.697626` | `0.510644` | `1.876462` | `0.520817` | `0.727294` | `0.000000` | no |
| new crop256 | ROI shadow-right v2 raw-crop videos 256 + 64 frames + temporal stats | `knn` | `0.357216` | `0.729136` | `1.793520` | `0.720454` | `0.369349` | `0.111111` | no |

Mean-latent baseline MSE is unchanged across these controlled runs:

- `0.326974`

## Descriptor Diagnostic

本次 crop256 embedding 的 descriptor MVP 诊断结果:

- Embedded samples: `36`
- Joined rows: `36`
- Train/test rows: `27/9`
- Best descriptor model: `knn {"n_neighbors": 7}`
- Learned descriptor mean MAE / RMSE / R^2: `0.7757 / 1.0485 / -1.0500`
- Nearest-neighbor descriptor mean MAE / RMSE / R^2: `0.8088 / 1.2338 / -3.1783`

## 结论

本次 `rheed_roi_shadow_right_v2_main_raw_crop_videos_256` 相对旧 raw baseline 在两个主 latent 指标上有提升:

- Learned latent MSE: `0.451245 -> 0.357216`
- Learned latent cosine: `0.686244 -> 0.729136`

它也显著好于上一轮 `processed_npz` 结果:

- Learned latent MSE: `0.697626 -> 0.357216`
- Learned latent cosine: `0.510644 -> 0.729136`

但本次结果仍不能视为已经解决跨模态预测问题，因为:

- Learned latent MSE `0.357216` 仍高于 mean-latent baseline `0.326974`
- Top-k retrieval hit rate `0.111111` 低于旧 raw baseline `0.333333`
- AFM autoencoder 仍有既有 warning: `AFM latent space may not yet be morphology-preserving`

因此最稳妥的判断是:

> 新的 crop256 ROI 视频确实让当前 pipeline 的 headline latent regression 指标相对旧 raw 和旧 processed npz 都有改善，但还没有超过 mean-latent sanity baseline；它是一个积极信号，不是最终成功证据。

## 产物位置

- New embeddings: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/descriptor_data/sample_embeddings.npy`
- New embedding index: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/descriptor_data/sample_embedding_index.csv`
- New latent metrics: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/metrics.json`
- New nearest latent grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/nearest_latent_grid.png`
- New generated AFM grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/generated_afm_grid.png`
- Raw-vs-new standard comparison: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/processed_vs_raw_comparison.md`
