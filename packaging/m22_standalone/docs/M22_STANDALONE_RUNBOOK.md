# M22 standalone runbook

## 1. UI

```bash
./scripts/run_m22_standalone.sh run-ui
```

Wait for `M20 + M22c | READY`, select a sample/video, then click **Analyze and
replay**. Every accepted causal clear-moment event adds one Sq/FSMI/confidence
point and one newly generated AFM map.

## 2. Command-line inference

```bash
./scripts/run_m22_standalone.sh predict-video VIDEO_PATH SAMPLE_ID [OUTPUT_DIR]
```

Example:

```bash
./scripts/run_m22_standalone.sh predict-video \
  "data/raw/raw_RHEED/N6063/rampdown to 300C.MOV" 6063 \
  reproduced_outputs/cli_6063
```

## 3. Validation

```bash
./scripts/run_m22_standalone.sh validate
./scripts/run_m22_standalone.sh list-visualizations
./scripts/run_m22_standalone.sh verify-checksums
./scripts/run_m22_standalone.sh test
./scripts/run_m22_standalone.sh smoke-model-6063
```

The archived 6063 smoke is expected to identify the
`MorphMBE-M20-SpotConnectivitySq + M22c-DenseMidGapCompletion...` model and to
produce generated Sq equal to predicted Sq within `1e-4 nm`.

## 4. Rebuild derived model cache

```bash
./scripts/run_m22_standalone.sh prepare-model
```

This overwrites only the derived v10 deployment bundle. It does not modify raw
RHEED/AFM data or frozen M22 atlas results.

## 5. Re-run the retrospective experiment

```bash
./scripts/run_m22_standalone.sh reproduce-m22-inclusive
./scripts/run_m22_standalone.sh reproduce-m22-exclusion
```

These are complete 27-fold runs and write derived outputs. Duplicate the
standalone or change output roots before re-running if archival preservation is
required.

## Portability

The bundled environment targets macOS Apple Silicon. If `.venv` is not usable
after transfer, install `uv` and run:

```bash
uv sync --frozen --extra test
```

The launcher forces offline Hugging Face and Torch caches inside the archive.
