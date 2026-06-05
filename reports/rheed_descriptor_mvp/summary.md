# RHEED-to-AFM Descriptor MVP

- Encoder backend: `torchvision`
- Embedded samples: `41`
- Skipped samples: `0`
- Joined rows: `166`
- Training rows: `124`
- Test rows: `42`
- Best model: `knn` `{"n_neighbors": 7}`

## Holdout Metrics

- Learned model mean MAE / RMSE / R^2: `0.7577` / `0.9755` / `-0.8206`
- Nearest-neighbor baseline mean MAE / RMSE / R^2: `0.8040` / `1.0760` / `-1.1392`
