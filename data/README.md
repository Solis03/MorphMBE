# Data availability and layout

Raw RHEED videos and AFM measurements are not distributed in this repository.
They remain read-only research records and are excluded from Git. Authorized
users can stage them without changing tracked files:

```text
data/raw/raw_RHEED/<growth-id>/<video>
data/raw/raw_AFM/<growth-id>/<scan>
```

Sample IDs, physical units, orientation corrections, and growth-group split
metadata must be preserved. A growth group is the minimum leakage boundary:
frames or scans from a held growth must not enter fitting, calibration, model
selection, or retrieval for that fold. Missing measurements must never be
imputed or fabricated.

The frozen aggregate and per-growth tables needed to audit the published M22
claims are versioned under `results/m22/`. Derived local outputs belong under
`data/processed/` or `artifacts/`; both are ignored by Git.
