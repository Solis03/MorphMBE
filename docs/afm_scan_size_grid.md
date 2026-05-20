# AFM Scan Size Grid Figures

Use this optional visualization step to create overview figures for AFM scans
with a selected square scan size. It does not modify extraction or plane
correction outputs.

Create two overview figures for 1 x 1 um scans:

```bash
uv run python scripts/afm_scan_size_grid.py
```

Outputs are written to `reports/figures/afm_scan_size_grids/`:

- `processed_afm_scan_size_1um_grid.png`
- `plane_corrected_afm_scan_size_1um_grid.png`

Each sample id appears at most once. If a sample has multiple matching 1 x 1 um
AFM scans, the script uses the first match in sorted metadata-path order. If a
sample has no matching scan, it is skipped.

Each subplot is drawn from the height `.npy` array, labeled with an index,
sample id, and AFM file id, and includes its own height colorbar in nm.

Quick test:

```bash
uv run python scripts/afm_scan_size_grid.py --limit 6
```
