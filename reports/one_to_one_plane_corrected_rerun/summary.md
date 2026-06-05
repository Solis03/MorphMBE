# Plane-Corrected Manifest Rerun Summary

This rerun uses the corrected one-to-one manifests under `data/manifests/` after fixing:

- manifest builder output-path resolution
- candidate ranking so `plane_corrected` AFM is preferred over processed AFM

## Subsets rerun

- `1um`: 36 pairs / 36 groups
- `all_size_representative`: 40 pairs / 40 groups

## Main outcomes

- `1um`
  - Autoencoder best val loss: `0.479466`
  - Reconstruction warning triggered: `yes`
  - Selected latent model: `knn`
  - Learned latent MSE / cosine: `0.451245` / `0.686244`
  - Mean-latent baseline MSE / cosine: `0.326974` / `0.657868`
  - Interpretation: latent target still looks too blurred / partially collapsed for strong scientific claims, and learned retrieval does not beat the mean-latent baseline.

- `all_size_representative`
  - Autoencoder best val loss: `0.475022`
  - Reconstruction warning triggered: `no`
  - Selected latent model: `ridge`
  - Learned latent MSE / cosine: `2.811252` / `0.273156`
  - Mean-latent baseline MSE / cosine: `1.448372` / `0.527218`
  - Interpretation: reconstruction looks better than `1um`, but learned retrieval still underperforms the mean-latent baseline.

## Qualitative notes

- `1um` reconstructions preserve coarse morphology only weakly and often blur distinct island boundaries.
- `all_size_representative` reconstructions preserve texture class somewhat better, but decoded predictions remain noticeably smoothed.
- In both subsets, nearest retrieved AFM prototypes are sometimes plausible, but the learned RHEED-to-latent mapping still does not outperform a simple latent prior.

## Bottom line

The plane-corrected rerun fixes the data-integrity issue, but it does not change the scientific conclusion:

- AFM autoencoder quality is still the main bottleneck, especially for `1um`.
- Current RHEED-to-latent results remain preliminary.
- Next work should prioritize improving the AFM latent space before claiming cross-modal success.
