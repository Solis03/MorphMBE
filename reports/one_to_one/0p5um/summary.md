# RHEED-to-AFM Descriptor MVP

- Encoder backend: `torchvision`
- Embedded samples: `3`
- Skipped samples: `0`
- Joined rows: `3`
- Training rows: `2`
- Test rows: `1`
- Best model: `ridge` `{"alphas": [0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0, 316.22776601683796, 1000.0]}`

## Holdout Metrics

- Learned model mean MAE / RMSE / R^2: `1.0921` / `1.0921` / `nan`
- Nearest-neighbor baseline mean MAE / RMSE / R^2: `0.2871` / `0.2871` / `nan`
