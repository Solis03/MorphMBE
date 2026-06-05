# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_1um_one_to_one.csv`
- Normalization: `per_image_zscore`
- Train rows / groups: `29` / `29`
- Val rows / groups: `8` / `8`
- Best epoch: `51`
- Best val reconstruction loss: `0.873106`
- Final val reconstruction loss: `0.873106`
- Reconstruction MAE: `0.743780`
- Reconstruction std ratio: `0.4233`

## Interpretation

- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.
