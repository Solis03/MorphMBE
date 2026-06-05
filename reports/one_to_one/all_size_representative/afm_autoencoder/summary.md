# AFM Autoencoder MVP

- Manifest: `/home/wangziyi/MorphMBE/MorphMBE/data/manifests/manifest_all_size_representative_one_to_one.csv`
- Normalization: `per_image_zscore`
- Train rows / groups: `32` / `32`
- Val rows / groups: `8` / `8`
- Best epoch: `31`
- Best val reconstruction loss: `0.891828`
- Final val reconstruction loss: `0.891828`
- Reconstruction MAE: `0.747905`
- Reconstruction std ratio: `0.3709`

## Interpretation

- Reconstruction does not trigger the collapse heuristic, but qualitative grid inspection is still required.
