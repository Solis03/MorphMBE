# RHEED-to-AFM Latent MVP

- Selected model: `knn`
- Embedding source: `processed clean_frames + 64 frames + temporal stats`
- Learned latent MSE / cosine: `0.697626` / `0.510644`
- Mean-latent baseline MSE / cosine: `0.326974` / `0.657868`
- Random-train baseline MSE / cosine: `1.203384` / `0.621998`

## Interpretation

- Warning: AFM latent space may not yet be morphology-preserving; RHEED-to-latent metrics should not be overinterpreted.
- Use nearest_latent_grid.png as the main scientific decision aid.
