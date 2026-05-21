# MLP Descriptor Decoder Baseline

This MLP decoder is an exploratory small-data baseline. The PCA decoder is the more stable reference.

The model maps selected AFM descriptors directly to normalized plane-corrected ZSensor AFM height maps. It should not be interpreted as exact reconstruction.

## Metrics

- insample: MSE=0.000737733, MAE=0.0183829, SSIM=0.986335, Pearson=0.998707
- 5-fold_cv: MSE=0.287529, MAE=0.403562, SSIM=0.0713939, Pearson=0.182972

## PCA Reference

- best PCA LOOCV row: k=8, MSE=0.2159521145, MAE=0.3309649212, SSIM=0.08523758601, Pearson=0.2178610543
