# RHEED-to-AFM Latent MVP

- Selected model: `knn`
- Embedding source: `removelist-filtered ROI shadow-right v2 raw-crop videos 256 + 64 frames + temporal stats`
- Learned latent MSE / cosine: `0.754014` / `0.455213`
- Mean-latent baseline MSE / cosine: `0.364814` / `0.775990`
- Random-train baseline MSE / cosine: `1.023955` / `0.512525`

## Interpretation

- AFM autoencoder does not trigger the collapse warning; qualitative retrieval inspection remains essential.
- Use nearest_latent_grid.png as the main scientific decision aid.
