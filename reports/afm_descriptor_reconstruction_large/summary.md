# Large AFM Descriptor Reconstruction Summary

## Goal

Test whether using more AFM data improves descriptor-to-image MLP reconstruction without changing model architecture.

## Dataset

- Total valid AFM files: 250
- Unique sample_id count: 40
- 1um scans: 166
- Non-1um scans: 84
- Scan size distribution: `{'0.0394x0.0394': 1, '0.0721x0.0721': 1, '0.076x0.076': 1, '0.0859x0.0859': 1, '0.0938x0.0938': 1, '0.0963x0.0963': 1, '0.0977x0.0977': 1, '0.102x0.102': 2, '0.168x0.168': 1, '0.187x0.17': 1, '0.1x0.1': 1, '0.201x0.201': 2, '0.203x0.203': 1, '0.207x0.207': 1, '0.219x0.219': 1, '0.238x0.238': 1, '0.263x0.263': 1, '0.27x0.27': 1, '0.301x0.301': 3, '0.305x0.305': 1, '0.309x0.309': 1, '0.332x0.332': 2, '0.395x0.395': 1, '0.488x0.488': 1, '0.496x0.496': 1, '0.498x0.498': 1, '0.508x0.508': 1, '0.5x0.5': 32, '0.664x0.664': 1, '0.801x0.801': 1, '0.891x0.891': 1, '0.8x0.8': 1, '1.02x1.02': 3, '1.33x1.33': 1, '1.64x1.64': 1, '1x1': 163, '2x2': 8, '5x5': 6}`
- Original resolution distribution: `{'104x104': 3, '128x128': 2, '131x144': 1, '164x164': 1, '256x256': 188, '512x512': 55}`

## Feature Changes

Scan size, pixel size, area, log area, aspect ratio, and `is_1um_scan` were added as descriptors so resized 64x64 images remain physically conditioned.

## Model

The MLP architecture was intentionally kept unchanged: descriptor input -> 128 -> 256 -> 512 -> image pixels.

## Metrics

| Experiment | MSE | MAE | SSIM | Pearson |
|---|---:|---:|---:|---:|
| previous 1um-only MLP 5-fold | 0.2863661285 | 0.4011227477 | 0.07122692654 | 0.1881195379 |
| insample_mlp | 0.000570646 | 0.0179733 | 0.97921 | 0.998454 |
| random5fold_mlp | 0.335528 | 0.427907 | 0.040606 | 0.135064 |
| group5fold_mlp | 0.433545 | 0.465485 | 0.0205669 | 0.107226 |
| mean_baseline_group5fold | 0.205092 | 0.366979 | 0.00807403 | 0.184142 |
| nearest_neighbor_group5fold | 0.322827 | 0.444937 | 0.026648 | 0.105042 |
| train_1um_test_non1um | 1.53945 | 0.817752 | 0.00591726 | 0.0358344 |
| train_non1um_test_1um | 0.217984 | 0.371096 | 0.0181495 | 0.1031 |

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
