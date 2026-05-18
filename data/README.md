# Data Directory

This directory stores lightweight data documentation and manifests.

Large raw and generated data are intentionally excluded from git:

- `data/raw/`
- `data/pair/`
- `data/processed/`
- `data/processed_afm/`

Use `data/manifests/` for small CSV/JSON files that describe available samples,
splits, labels, or provenance.
