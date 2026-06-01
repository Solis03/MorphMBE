# MLP Descriptor Decoder Baseline

This MLP decoder is an exploratory small-data baseline. The PCA decoder is the more stable reference.

The model maps selected AFM descriptors directly to normalized plane-corrected ZSensor AFM height maps. It should not be interpreted as exact reconstruction.

## Metrics

- insample: MSE=0.00109243, MAE=0.0242829, SSIM=0.978123, Pearson=0.997228
- 5-fold_cv: MSE=0.286366, MAE=0.401123, SSIM=0.0712269, Pearson=0.18812

## PCA Reference

- best PCA LOOCV row: k=8, MSE=0.1674179605, MAE=0.3193132348, SSIM=0.08371481485, Pearson=0.2160688719
