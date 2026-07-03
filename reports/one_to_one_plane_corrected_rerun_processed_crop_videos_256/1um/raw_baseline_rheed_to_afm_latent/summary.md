# RHEED-to-AFM Latent MVP

- Selected model: `ridge`
- Embedding source: `removelist-filtered raw video + 8 frames + mean/std`
- Learned latent MSE / cosine: `0.396026` / `0.702746`
- Mean-latent baseline MSE / cosine: `0.364814` / `0.775990`
- Random-train baseline MSE / cosine: `1.023955` / `0.512525`

## Interpretation

- AFM autoencoder does not trigger the collapse warning; qualitative retrieval inspection remains essential.
- Use nearest_latent_grid.png as the main scientific decision aid.
