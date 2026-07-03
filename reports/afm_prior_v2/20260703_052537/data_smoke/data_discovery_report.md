# AFM Prior V2 Data Discovery Report

- Scan filter: `1um`
- Source AFM files indexed: `32`
- Indexed training rows after patch policy: `32`
- Groups: `10`
- Physical height maps: `32`
- Network inputs: `0`
- PNG fallback rows: `0`
- Increase versus MVP-1 36-file run: `-4`
- Patch mode: `none`

## Split Counts

- Files by split: `{'train': 25, 'val': 5, 'test': 2}`
- Rows by split: `{'train': 25, 'val': 5, 'test': 2}`
- Groups by split: `{'train': 7, 'val': 2, 'test': 1}`

## Scan Sizes Seen In Raw Candidates

- `0.072` um: `3`
- `0.076` um: `3`
- `0.086` um: `3`
- `0.094` um: `3`
- `0.096` um: `3`
- `0.098` um: `3`
- `0.100` um: `6`
- `0.102` um: `6`
- `0.150` um: `3`
- `0.168` um: `3`
- `0.179` um: `3`
- `0.201` um: `6`
- `0.203` um: `3`
- `0.207` um: `3`
- `0.219` um: `3`
- `0.238` um: `3`
- `0.263` um: `3`
- `0.270` um: `3`
- `0.301` um: `9`
- `0.305` um: `3`
- `0.309` um: `3`
- `0.332` um: `6`
- `0.395` um: `3`
- `0.488` um: `3`
- `0.496` um: `3`
- `0.498` um: `3`
- `0.500` um: `105`
- `0.508` um: `3`
- `0.664` um: `3`
- `0.800` um: `9`
- `0.801` um: `3`
- `0.891` um: `3`
- `1.000` um: `492`
- `1.016` um: `9`
- `1.328` um: `3`
- `1.641` um: `3`
- `2.000` um: `24`
- `39.429` um: `3`
- `5.000` um: `18`

## Path Conventions Observed

- Physical processed height maps: `data/processed_afm/<sample_id>/<afm_file_id>/<afm_file_id>_height.npy`
- Plane-corrected physical height maps: `data/plane_corrected_afm/<sample_id>/<afm_file_id>/<afm_file_id>_plane_corrected.npy`
- Prepared model input tensors: `reports/afm_prior_v2/<timestamp>/data/standardized_tensors/<row_id>.npy`
- No `network_input.npy` files were required when physical height maps were available.
