# Processed vs Raw RHEED Comparison

## Metrics

| Variant | Source | Model | Learned latent MSE | Learned cosine | Nearest latent distance | Nearest latent cosine | Retrieved latent MSE | Top-k hit rate | Beats mean latent? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | raw video + 8 frames + mean/std | knn | 0.451245 | 0.686244 | 1.370465 | 0.678092 | 0.503228 | 0.333333 | no |
| processed | ROI shadow-right v2 raw-crop videos 256 + 64 frames + temporal stats | knn | 0.357216 | 0.729136 | 1.793520 | 0.720454 | 0.369349 | 0.111111 | no |

## Visual Checks

- Raw nearest-latent grid: `reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/nearest_latent_grid.png`
- Processed nearest-latent grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/nearest_latent_grid.png`
- Raw generated-AFM grid: `reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/generated_afm_grid.png`
- Processed generated-AFM grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/generated_afm_grid.png`

## Conclusion

- Processed RHEED improves on the raw baseline on both learned latent MSE and cosine similarity.
- Processed RHEED still does not beat the mean-latent dummy baseline on latent MSE.
- If processed RHEED still underperforms the dummy baseline, the likely bottleneck remains AFM latent target quality or cross-modal supervision mismatch rather than basic input cleaning alone.
