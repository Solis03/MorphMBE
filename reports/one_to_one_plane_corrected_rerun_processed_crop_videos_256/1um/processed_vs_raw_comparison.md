# Processed vs Raw RHEED Comparison

## Metrics

| Variant | Source | Model | Learned latent MSE | Learned cosine | Nearest latent distance | Nearest latent cosine | Retrieved latent MSE | Top-k hit rate | Beats mean latent? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | removelist-filtered raw video + 8 frames + mean/std | ridge | 0.396026 | 0.702746 | 1.748233 | 0.676758 | 0.415456 | 0.375000 | no |
| processed | removelist-filtered ROI shadow-right v2 raw-crop videos 256 + 64 frames + temporal stats | knn | 0.754014 | 0.455213 | 1.882114 | 0.456461 | 0.699739 | 0.250000 | no |

## Visual Checks

- Raw nearest-latent grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/raw_baseline_rheed_to_afm_latent/nearest_latent_grid.png`
- Processed nearest-latent grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/nearest_latent_grid.png`
- Raw generated-AFM grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/raw_baseline_rheed_to_afm_latent/generated_afm_grid.png`
- Processed generated-AFM grid: `reports/one_to_one_plane_corrected_rerun_processed_crop_videos_256/1um/rheed_to_afm_latent/generated_afm_grid.png`

## Conclusion

- Processed RHEED does not beat the raw baseline on the main learned latent metrics.
- Processed RHEED still does not beat the mean-latent dummy baseline on latent MSE.
- If processed RHEED still underperforms the dummy baseline, the likely bottleneck remains AFM latent target quality or cross-modal supervision mismatch rather than basic input cleaning alone.
