# Manual RHEED-AFM Pair Visualization

## Folder structure

The manual selection root was inspected recursively. In this checkout, sample folders are direct children of `data/manual_selection` and contain `RHEED` and `AFM` subfolders.

## Manual RHEED rule

Only image files with a basename starting with `select` are used. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`. Videos, hidden files, metadata files, and temporary files are ignored.

Multiple selections are resolved deterministically by exact stem `select`, then `select_final*`, then `select_best*`, then lexicographic filename order. The warning is recorded as `multiple_manual_selections`.

## Sample matching

Manual folders are matched by exact normalized four-digit sample ID to AFM sample IDs from plane-corrected AFM metadata. Broad fuzzy matching is not used.

## AFM representative scan

Valid physical plane-corrected height maps are selected without using RHEED appearance. The rule prefers 1.0 um scans and chooses the scan whose Rq is closest to the sample median Rq within that subset. If no 1.0 um scan exists, the dominant valid scan size is used and the median-closeness rule is applied within that size.

## Rq definition

The displayed roughness is RMS roughness, `Rq = sqrt(mean((z - mean(z))^2))`, in nanometers. Descriptor-table Rq is used when available; otherwise Rq is recomputed from the physical height map.

## Rendering

RHEED panels show the manual screenshot with minimal display transformation and preserved aspect ratio. AFM panels are rendered from physical height arrays in nanometers with viridis, equal spatial aspect, a physical color bar, and a lateral scale bar.

Native AFM scale uses per-scan 1st to 99th percentile height limits. Common scale uses pooled 1st to 99th percentile limits over selected 1.0 um scans.

## Counts

- sample folders inspected: 41
- included samples: 30
- skipped samples: 11
- skipped `missing_manual_selection`: 11
- common AFM scale: -14.67 to 14.55 nm

## Outputs

- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_native.png`
- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_native.pdf`
- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_common_scale.png`
- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_common_scale.pdf`
- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_sample_id.png`
- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_sample_id.pdf`
- `reports/rheed_roughness/manual_pair_visualization/index.html`
- `reports/rheed_roughness/manual_pair_visualization/skipped_samples.md`
- `outputs/rheed_roughness/manual_pair_visualization/manual_selection_audit.csv`
- `outputs/rheed_roughness/manual_pair_visualization/afm_selection_audit.csv`
- `outputs/rheed_roughness/manual_pair_visualization/manual_pair_figure_manifest.csv`

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.visualize_manual_pairs --config configs/rheed_roughness.yaml --manual-selection-root data/manual_selection
```

The optional morphology-index sorted figure was not generated unless a reliable existing morphology table was available.