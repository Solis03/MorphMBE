# MorphMBE M17 Nano Letters figure package

This directory contains a three-figure, manuscript-oriented package for the
retrospective MorphMBE M17b study. The public figures use anonymized two-digit
labels (`Sample 01`--`Sample 27`); the private mapping is retained only in
`sample_id_mapping_internal.csv` for provenance.

## Deliverables

- `figures/Figure_1_AutoRHEED_overview.{pdf,png,tiff}`: acquisition-to-AFM
  overview using real Sample 23 RHEED and AFM data.
- `figures/Figure_2_model_and_validation.{pdf,png,tiff}`: model architecture,
  four genuine stochastic M17b realizations, and the outer leave-one-growth-out
  protocol.
- `figures/Figure_3_selected_results.{pdf,png,tiff}`: selected smooth,
  intermediate, and rough examples plus cohort-wide validation.
- `captions.md`: manuscript-ready English captions.
- `research_and_design_basis.md`: journal, color-map, accessibility, and
  selection decisions.
- `figure_provenance_internal.csv`: panel-level internal traceability.
- `selected_case_metrics.csv`: numeric values used for the three displayed
  samples.
- `build_manifest.json`: output dimensions, resolution, source commit, and AFM
  color mapping.
- `source_snapshots/`: checksum-locked derived inputs needed to rebuild the
  figures without reading or modifying raw data.

PDF is the preferred submission/editing format because labels and paths remain
vector. The PNG and LZW-compressed TIFF versions are 600 dpi at 7.0 in width.

## Rebuild and verify

Run from the repository root with the repository environment:

```bash
/Users/ziyi/Desktop/LAB/code/.venv/bin/python scripts/make_nanoletters_m17_figures.py --verify-only
/Users/ziyi/Desktop/LAB/code/.venv/bin/python scripts/make_nanoletters_m17_figures.py
/Users/ziyi/Desktop/LAB/code/.venv/bin/python -m pytest -q tests/test_nanoletters_m17_figures.py
```

The first command checks every displayed source hash, public/internal sample
identity, selected model, and the inference-time nonretrieval/no-measured-AFM
flags. The test also audits the 27 growth-level outer folds.

## Claim boundary

All displayed predictions are strict outer leave-one-growth-out predictions:
the displayed growth is absent from fitting in its outer fold. The model does
not retrieve a training image and does not read a measured AFM patch at
inference. Generated AFM panels are stochastic conditional morphology
realizations, not pixel-registered reconstructions of the measured scan.
N6342 (public Sample 23) motivated renderer development, so it is retrospective
method-development evidence rather than a prospectively untouched test.
