# Independent AFM Sq/Rq re-audit with Gwyddion 2.71

Date: 2026-07-29

Scope: the metrology-audited full-28 dataset; raw AFM files are read-only.

## Decision

No formula, unit-conversion, channel-selection, or polynomial-evaluation error
was found in the current metrology pipeline. The training target remains the
**areal RMS height Sq in nm** of the ZSensor map after a third-order polynomial
is independently fitted and subtracted from every fast-scan row.

The word **Rq** is retained only when quoting NanoScope's own profile-oriented
label or a historical filename. For a two-dimensional height map, **Sq** is the
correct ISO 25178 areal name. This terminology distinction changes the label,
not the numerical RMS definition used by the code.

## Formula and implementation

For a discrete \(M\times N\) height field \(z_{ij}\), the code evaluates

\[
S_q =
\sqrt{\frac{1}{MN}
\sum_{i=1}^{M}\sum_{j=1}^{N}
\left(z_{ij}-\bar z\right)^2},
\qquad
\bar z=\frac{1}{MN}\sum_{i,j}z_{ij}.
\]

The divisor is \(MN\) (`ddof=0`), as required for the RMS of the complete
measured field. Before this calculation, each scan row is fitted by ordinary
least squares to \(1,x,x^2,x^3\), and the fitted cubic background is
subtracted. The operation is row-wise, not a single two-dimensional plane fit.

The definition agrees with Gwyddion's [statistical quantities
documentation](https://gwyddion.net/documentation/user-guide-en/statistical-analysis.html).
Gwyddion also documents row polynomial levelling as a scan-line correction in
its [process API](https://gwyddion.net/documentation/libgwyprocess/libgwyprocess-correct.php).
The background-order sensitivity remains a metrology choice: higher-order
subtraction can remove genuine long-wavelength morphology, a known source of
roughness bias discussed by Nečas *et al.*,
[doi:10.1088/1361-6501/ab8993](https://doi.org/10.1088/1361-6501/ab8993).

## Independent Gwyddion reproduction

The locally installed `/opt/homebrew/bin/gwyddion` is version 2.71. The audit
does not reimplement Gwyddion mathematics. It calls Gwyddion's own:

- NanoScope file importer;
- `gwy_data_field_row_level_poly` for row orders 0, 1, 2, and 3; and
- `gwy_data_field_get_rms` for the final RMS.

The bridge uses Gwyddion's non-GUI batch initializer, so it exercises the same
installed importer and numerical library without a manual screenshot or
copy/paste step.

| independently checked fields | rows | line-3 mean absolute delta | line-3 maximum absolute delta |
|---|---:|---:|---:|
| original cohort, complete 1 µm ZSensor maps | 110 | \(4.72\times10^{-10}\) nm | \(3.64\times10^{-9}\) nm |
| extra-five, non-overlapping 1 µm ZSensor subfields | 104 | \(2.04\times10^{-10}\) nm | \(7.32\times10^{-10}\) nm |
| **total** | **214** | — | **< \(4\times10^{-9}\) nm** |

All 214 comparisons also agree at orders 0, 1, and 2. The largest difference
over any order is \(7.41\times10^{-9}\) nm, numerical round-off far below any
physical measurement resolution.

Representative Gwyddion line-3 ZSensor results are:

| growth / raw scan | Gwyddion Sq |
|---|---:|
| 6101 / `N6101_1um.002` | 0.567806 nm |
| 6081 / `N6081_1um.000` | 0.941725 nm |
| 6095 / `N6095_1um.003` | 8.149551 nm |
| 6099 / `N6099_1um.000` | 9.320814 nm |
| N6342 / `N6342_2um_4.000`, complete 2 µm scan | 0.860652 nm |

The extra-five raw headers declare `Frame direction: Down`. Gwyddion
normalizes that acquisition orientation on import, while the repository
decoder retains stored row order. Therefore a top/bottom subfield is vertically
mirrored between the two array conventions. Sq is invariant to this mirror,
and the audit explicitly maps the physical quadrants before comparison.

## Raw-header and channel audit

The raw NanoScope headers identify the height channel as `ZSensor`/`Height
Sensor`, with a physical Z scale and line direction `Retrace`. They also record
`Realtime Planefit: Line`, `Offline Planefit: None`, pixel dimensions, and
scan direction. Consequently:

1. the acquisition already contains the instrument's real-time line
   correction;
2. the documented cubic correction is an additional offline background
   definition used consistently for all derived targets; and
3. non-height channels such as `Peak Force Error`, stiffness, adhesion, and
   dissipation are excluded even when Gwyddion imports them.

For example, the N6342 file contains ZSensor, Height, Deformation, and several
voltage channels. Only the channel titled `ZSensor` enters the audited target.

## NanoScope exported-number QC

The earlier independent NanoScope record comparison remains valid:

| scope | matched scans | line-3 MAE | Pearson r | within 0.2 nm |
|---|---:|---:|---:|---:|
| all filename/software records | 77 | 0.0606 nm | 0.9970 | 94.8% |
| active original-23 primary 1 µm, hash-deduplicated | 42 | **0.0224 nm** | **0.9998** | **100%** |

NanoScope records are QC only. They are not copied into the model target.

## Aggregation and provenance safeguards

- Exact derived height arrays are deduplicated by SHA-256.
- `6094/N6081_1um_000` remains excluded because its explicit sample name
  conflicts with the containing growth and cannot be resolved locally.
- Every scan is converted to nm and assigned a scan Sq first.
- A sample target is the arithmetic median of its valid scan Sq values in nm;
  the IQR is retained as within-sample heterogeneity.
- The natural logarithm is taken only after physical-space aggregation.
- Figures distinguish `displayed scan Sq` from `sample median Sq ± IQR`.

## Interpretation boundary

The numerical computation is now independently reproduced to machine
precision, but this does not make cubic line levelling the only possible
physical background definition. Long-wavelength waviness and true morphology
cannot always be separated from scanner bow using image data alone. The
order-0/1/2/3 sensitivity table and the NanoScope QC must therefore remain with
the publication artifacts. Prospective work should archive the exact
NanoScope analysis recipe and a reference calibration surface.

## Reproducibility

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.afm_metrology_reaudit.gwyddion_crosscheck --help

PYTHONPATH=. .venv/bin/python \
  -m analysis.afm_metrology_reaudit.verify_extra_subfields \
  --scan-table outputs/extra_five_integration/20260729_line3_full28_v1/combined_primary_1um_scans.csv \
  --path-root . \
  --output-csv reports/afm_metrology_reaudit/gwyddion_vs_pipeline_extra104_subfields.csv
```

Primary evidence:

- `gwyddion_vs_pipeline_base110.csv`
- `gwyddion_vs_pipeline_extra104_subfields.csv`
- `gwyddion_representative_crosscheck.csv`
- `analysis/afm_metrology_reaudit/gwyddion_crosscheck.py`
- `analysis/afm_metrology_reaudit/verify_extra_subfields.py`
