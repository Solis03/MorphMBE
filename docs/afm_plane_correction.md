# AFM Plane Correction

AFM height maps can include a low-frequency sample tilt or scanner background.
Fitted-plane subtraction removes a first-order plane from each height map while
preserving the original height units. This is useful as an optional preprocessing
step before comparing morphology across scans.

Run plane correction on all processed AFM height maps:

```bash
uv run python scripts/afm_plane_correct.py
```

The script reads `data/processed_afm/` and writes a mirrored folder tree under
`data/plane_corrected_afm/`. Each output folder contains:

- `*_plane_corrected.npy`
- `*_plane_corrected_render.png`
- `*_fitted_plane.npy`
- `*_plane_corrected_metadata.json`

Quick test on the first five height maps:

```bash
uv run python scripts/afm_plane_correct.py --limit 5 --overwrite
```

The fitted model is `z = a*x + b*y + c` using normalized image coordinates
`x,y` in `[-1, 1]`. NaN and infinite pixels are ignored during fitting.

The default paths are resolved from the repository root, so the command can be
run from the project checkout without needing a `raw_data` folder.
