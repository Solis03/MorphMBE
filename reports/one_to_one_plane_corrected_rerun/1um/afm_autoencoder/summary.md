# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_1um_one_to_one.csv`
- Model type: `residual`
- Normalization: `per_image_zscore`
- Pixel loss / edge weight: `smooth_l1` / `0.200`
- Train rows / groups: `28` / `28`
- Val rows / groups: `8` / `8`
- Best epoch: `20`
- Best val reconstruction loss: `0.479466`
- Final val reconstruction loss: `0.479466`
- Reconstruction MAE: `0.734201`
- Reconstruction std ratio: `0.3041`

## Interpretation

- Warning: AFM latent space may not yet be morphology-preserving; RHEED-to-latent metrics should not be overinterpreted.
