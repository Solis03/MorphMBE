# RHEED-to-AFM Descriptor MVP

- Encoder backend: `torchvision`
- Embedded samples: `36`
- Skipped samples: `0`
- Joined rows: `36`
- Training rows: `27`
- Test rows: `9`
- Best model: `knn` `{"n_neighbors": 3}`

## Holdout Metrics

- Learned model mean MAE / RMSE / R^2: `0.7815` / `1.0611` / `-1.1133`
- Nearest-neighbor baseline mean MAE / RMSE / R^2: `0.8969` / `1.1576` / `-1.3280`
