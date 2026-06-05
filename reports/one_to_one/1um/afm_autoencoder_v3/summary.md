# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_1um_one_to_one.csv`
- Model type: `baseline`
- Normalization: `per_image_zscore`
- Pixel loss / edge weight: `smooth_l1` / `0.100`
- Train rows / groups: `29` / `29`
- Val rows / groups: `8` / `8`
- Best epoch: `26`
- Best val reconstruction loss: `0.426915`
- Final val reconstruction loss: `0.426915`
- Reconstruction MAE: `0.746048`
- Reconstruction std ratio: `0.3934`

## Interpretation

- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.
