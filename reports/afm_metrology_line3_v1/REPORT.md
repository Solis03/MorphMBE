# AFM metrology repair: third-order scan-line flattening

## Decision

The previous global first-order areal plane correction is superseded for model-target construction. The selected correction fits and subtracts a cubic polynomial independently from every fast-scan line. The output is an areal height map, so the recomputed RMS height is called **Sq**; NanoScope filename values retain their original **Rq** label as an independent software-export QC record.

## Evidence

- Source decoded ZSensor maps processed: 180.
- Independent labelled NanoScope TIF records matched: 77.
- Selected line-3 QC MAE: 0.0606 nm.
- Selected line-3 QC Pearson r: 0.9970.
- Selected line-3 within 0.2 nm: 94.8%.
- Exact duplicate scan rows identified: 14 (7 hash groups).
- Corrected modeling growths: 23.

## Provenance adjudication

The local repository cannot replace a lab notebook or acquisition log. Therefore `6094/N6081_1um_000` is conservatively excluded from the corrected target, while legacy N69/N74 names are retained but flagged. These rows still require human confirmation before a paper freeze.

- 6070/N69_Edg_000: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_Edg_001: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_Edg_002: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_center_003: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_center_200nm_004: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_center_200nm_005: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_center_200nm_006: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6070/N69_center_200nm_007: include_flagged; all AFM files use legacy N69 naming while paired growth is 6070; exclusion would remove the whole sample
- 6078/N74_ctr_004: include_flagged; legacy short ID conflicts with containing sample but corrected Sq agrees with the N78 scan family
- 6078/N74_ctr_005: include_flagged; legacy short ID conflicts with containing sample but corrected Sq agrees with the N78 scan family
- 6094/N6081_1um_000: exclude; explicit four-digit sample ID conflicts with containing sample 6094; conservative exclusion prevents unresolved provenance from entering targets

## Target definition

For each growth, exact selected-array hashes are deduplicated, unresolved excluded provenance is removed, and Sq is aggregated as an arithmetic median in nm across primary 1 × 1 µm scans. Only then is the sample median transformed with the natural log for model fitting. IQR is retained as within-sample uncertainty.

## Files

- Derived maps: `data/afm_metrology_line3_v1`
- Audit tables: `outputs/afm_metrology_line3_v1`
- Figures: `reports/afm_metrology_line3_v1/figures`
- Raw AFM files and decoded source arrays were read only.
