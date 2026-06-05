# 1um Autoencoder Iteration Note

| Variant | Best val loss | Recon MAE | Recon std ratio | Warning | Qualitative note |
| --- | ---: | ---: | ---: | --- | --- |
| v1_baseline_mse | 0.873106 | 0.743780 | 0.4233 | False | Original formal baseline. Partially preserves morphology but remains oversmoothed. |
| v2_residual_smoothl1_edge0.2 | 0.474034 | 0.753009 | 0.3626 | True | Residual model introduced visible decoder artifacts/checkerboard banding; rejected despite lower training-style loss. |
| v3_baseline_smoothl1_edge0.1 | 0.426915 | 0.746048 | 0.3934 | False | Best current compromise. Keeps baseline decoder behavior while slightly improving texture realism with smooth_l1 + edge loss. |

## Takeaway

- Increasing model complexity was not helpful under the current small-data regime.
- The safer next direction is to keep the simpler decoder and continue tuning loss/normalization/augmentation, not to deepen the network.
- `afm_autoencoder_v3` is the preferred starting point for the next 1um-only iteration, but it still does not make RHEED-to-latent beat the mean-latent baseline.
