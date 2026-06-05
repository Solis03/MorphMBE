# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_all_size_representative_one_to_one.csv`
- Model type: `residual`
- Normalization: `per_image_zscore`
- Pixel loss / edge weight: `smooth_l1` / `0.200`
- Train rows / groups: `32` / `32`
- Val rows / groups: `8` / `8`
- Best epoch: `24`
- Best val reconstruction loss: `0.475022`
- Final val reconstruction loss: `0.475022`
- Reconstruction MAE: `0.725234`
- Reconstruction std ratio: `0.4550`

## Interpretation

- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.
