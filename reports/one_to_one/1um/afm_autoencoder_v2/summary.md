# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_1um_one_to_one.csv`
- Model type: `residual`
- Normalization: `per_image_zscore`
- Pixel loss / edge weight: `smooth_l1` / `0.200`
- Train rows / groups: `29` / `29`
- Val rows / groups: `8` / `8`
- Best epoch: `10`
- Best val reconstruction loss: `0.474034`
- Final val reconstruction loss: `0.474034`
- Reconstruction MAE: `0.753009`
- Reconstruction std ratio: `0.3626`

## Interpretation

- Warning: AFM latent space may not yet be morphology-preserving; RHEED-to-latent metrics should not be overinterpreted.
