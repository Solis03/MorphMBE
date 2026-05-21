# Large AFM Descriptor Reconstruction Summary

## Goal

Test whether using more AFM data improves descriptor-to-image MLP reconstruction without changing model architecture.

## Dataset

- Total valid AFM files: 263
- Unique sample_id count: 43
- 1um scans: 169
- Non-1um scans: 94
- Scan size distribution: `{'0.0394x0.0394': 1, '0.0721x0.0721': 1, '0.076x0.076': 1, '0.0859x0.0859': 1, '0.0938x0.0938': 2, '0.0963x0.0963': 1, '0.0977x0.0977': 1, '0.102x0.102': 2, '0.141x0.141': 1, '0.168x0.168': 1, '0.187x0.17': 1, '0.1x0.1': 1, '0.201x0.201': 2, '0.203x0.203': 1, '0.207x0.207': 1, '0.219x0.219': 1, '0.238x0.238': 1, '0.263x0.263': 1, '0.27x0.27': 1, '0.301x0.301': 3, '0.305x0.305': 1, '0.309x0.309': 1, '0.332x0.332': 2, '0.395x0.395': 1, '0.488x0.488': 1, '0.496x0.496': 1, '0.498x0.498': 1, '0.49x0.49': 1, '0.508x0.508': 1, '0.5x0.5': 32, '0.664x0.664': 1, '0.688x0.688': 1, '0.781x0.738': 1, '0.801x0.801': 1, '0.891x0.891': 1, '0.8x0.8': 1, '1.02x1.02': 3, '1.33x1.33': 1, '1.5x1.5': 1, '1.64x1.64': 1, '1x0.5': 2, '1x1': 166, '2x2': 10, '5x5': 6}`
- Original resolution distribution: `{'104x104': 3, '128x128': 2, '131x144': 1, '164x164': 1, '176x176': 1, '189x200': 1, '256x256': 193, '256x512': 2, '512x512': 59}`

## Feature Changes

Scan size, pixel size, area, log area, aspect ratio, and `is_1um_scan` were added as descriptors so resized 64x64 images remain physically conditioned.

## Model

The MLP architecture was intentionally kept unchanged: descriptor input -> 128 -> 256 -> 512 -> image pixels.

## Metrics

| Experiment | MSE | MAE | SSIM | Pearson |
|---|---:|---:|---:|---:|
| previous 1um-only MLP 5-fold | 0.2875288862 | 0.4035621831 | 0.07139394494 | 0.1829716765 |
| insample_mlp | 0.000966797 | 0.0224818 | 0.971335 | 0.997433 |
| random5fold_mlp | 0.274451 | 0.396286 | 0.0552223 | 0.161582 |
| group5fold_mlp | 0.366393 | 0.446338 | 0.0230681 | 0.112474 |
| mean_baseline_group5fold | 0.208066 | 0.370138 | 0.0075257 | 0.1816 |
| nearest_neighbor_group5fold | 0.321306 | 0.441256 | 0.0344851 | 0.0979852 |
| train_1um_test_non1um | 1.38313 | 0.799926 | 0.00612897 | 0.0278096 |
| train_non1um_test_1um | 0.225415 | 0.377007 | 0.0153291 | 0.093006 |

## Interpretation

The in-sample result tests capacity and is expected to overfit. GroupKFold is more trustworthy than random row-level CV because scans from the same sample remain in the same fold. Mixing scan sizes gives more rows but also makes the resized image target less physically uniform; the scan-size-conditioned features help but do not make different scan areas directly equivalent. Errors by scan size should be read from `metrics_by_scan_size.png`; higher errors at larger areas would suggest the model is learning common texture patterns more than scan-size-transferable morphology.

## Limitations

- Different scan sizes are not directly equivalent after resizing.
- Image normalization may remove absolute height scale.
- Descriptor-to-image reconstruction is one-to-many.
- A direct MLP pixel decoder may fail on rare morphology types.
- GroupKFold is more trustworthy than random row-level CV.

## Next Steps

- Train separate models per scan-size group.
- Use descriptor-to-PCA latent instead of direct pixels.
- Use a convolutional decoder or autoencoder latent.
- Condition the decoder on scan size.
- Use morphology clustering before reconstruction.
