# RHEED-to-AFM Descriptor MVP

- Encoder backend: `torchvision`
- Embedded samples: `32`
- Skipped samples: `0`
- Joined rows: `32`
- Training rows: `24`
- Test rows: `8`
- Best model: `knn` `{"n_neighbors": 7}`

## Holdout Metrics

- Learned model mean MAE / RMSE / R^2: `0.8553` / `1.2068` / `-0.2266`
- Nearest-neighbor baseline mean MAE / RMSE / R^2: `0.9795` / `1.3789` / `-0.9401`
