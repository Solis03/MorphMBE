# Canonical extra-five derived dataset

This directory is the only extra-five derived-data root used by the full-28
experiments.

- Included growths: N6342, N6358, N6382, N6389, N6390.
- Excluded growth: N6324.
- Raw AFM source: `data/AFM-extra-five` (read only).
- Raw RHEED source: `data/compressedfile` (read only).
- AFM harmonization: split each decoded 2 × 2 µm ZSensor map into four
  non-overlapping 1 × 1 µm subfields, then independently flatten every
  fast-scan line with polynomial orders 0, 1, 2, and 3.
- Selected modeling target: third-order per-line result; sample Sq is the
  arithmetic median in nm across deduplicated 1 × 1 µm subfields, followed by
  the log transform used by the predictor.

The manifests in this directory and under
`reports/extra_five_integration/20260729_line3_full28_v1` record source hashes,
exclusion decisions, and the relationship to earlier derived folders.
