# AFM Second-Order Background Fitting

This branch creates a separate AFM background-subtraction dataset for ablation
against the existing first-order plane-corrected AFM data.

## Input Provenance

The input root for this repository is:

```bash
data/processed_afm
```

The script only selects files matching:

```text
data/processed_afm/<sample_id>/<afm_file_id>/<afm_file_id>_height.npy
```

Each selected file must have a sibling metadata file whose `primary_channel` or
`source_channel` is `ZSensor`. In the current checkout, these arrays are
physical AFM height maps exported in nm, with metadata fields such as
`height_unit_exported: nm`, `sample_id`, `afm_file_id`, `scan_size_um`, and the
original raw AFM file reference.

The script explicitly does not consume:

- `data/plane_corrected_afm/` first-order outputs
- network-normalized arrays in `network_inputs/`
- descriptor, reconstruction, latent, RHEED, cache, or QC arrays
- the new `data/afm_second_order/` output tree

## Model

The default model is `y2`:

```text
z_bg(x,y) = c0 + c1*x + c2*y + c3*y^2
```

The optional model is `full2d`:

```text
z_bg(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
```

Coordinates are normalized to `[-1, 1]`, with axis 0 as vertical `y` rows and
axis 1 as horizontal `x` columns. The corrected output is:

```text
corrected = original - fitted_background
```

No smoothing, resizing, interpolation, percentile clipping, normalization,
line flattening, or post-subtraction mean/median shift is applied.

## Running

Dry-run:

```bash
.venv/bin/python scripts/fit_afm_second_order.py \
  --input-dir data/processed_afm \
  --output-dir data/afm_second_order \
  --model y2 \
  --dry-run
```

Process a small deterministic prefix:

```bash
.venv/bin/python scripts/fit_afm_second_order.py \
  --input-dir data/processed_afm \
  --output-dir data/afm_second_order \
  --model y2 \
  --limit 5 \
  --save-background \
  --qc-count 5 \
  --verbose
```

Full run:

```bash
.venv/bin/python scripts/fit_afm_second_order.py \
  --input-dir data/processed_afm \
  --output-dir data/afm_second_order \
  --model y2 \
  --save-background \
  --qc-count 16
```

## Outputs

Outputs are written only under:

```text
data/afm_second_order/
```

The corrected `.npy` files mirror the input root relative paths. Fitted
backgrounds are stored under `_backgrounds/`, per-file metadata under
`_metadata/`, and QC under `_qc/`.

The processing manifest is written as both:

```text
data/afm_second_order/processing_manifest.csv
data/afm_second_order/processing_manifest.jsonl
```

Existing corrected, background, or metadata files are not overwritten. If an
output already exists, that input is recorded as `exists_skipped`.
