# MLP Descriptor Decoder Baseline

This PyTorch MLP decoder is an exploratory baseline. The PCA decoder is the more stable reference.

The model maps selected AFM descriptors directly to normalized plane-corrected ZSensor AFM height maps. It should not be interpreted as exact reconstruction.

## Metrics

- insample: MSE=0.000863557, MAE=0.0215935, SSIM=0.978908, Pearson=0.997985
- 5-fold_cv: MSE=0.279006, MAE=0.40003, SSIM=0.0714864, Pearson=0.182623

## PCA Reference

- best PCA LOOCV row: k=8, MSE=0.1670414351, MAE=0.3191190192, SSIM=0.08338554581, Pearson=0.2156345599
