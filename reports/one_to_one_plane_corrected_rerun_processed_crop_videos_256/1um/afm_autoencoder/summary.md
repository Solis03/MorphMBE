# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/manifest_1um_one_to_one_removelist_filtered.csv`
- Model type: `residual`
- Normalization: `per_image_zscore`
- Pixel loss / edge weight: `smooth_l1` / `0.200`
- Train rows / groups: `25` / `25`
- Val rows / groups: `7` / `7`
- Best epoch: `27`
- Best val reconstruction loss: `0.477923`
- Final val reconstruction loss: `0.477923`
- Reconstruction MAE: `0.730571`
- Reconstruction std ratio: `0.3821`

## Interpretation

- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.
