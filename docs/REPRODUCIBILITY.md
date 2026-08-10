# Reproducibility

## Environment

The frozen development runtime was macOS Apple Silicon, Python 3.12, PyTorch
2.12, and Apple MPS. The lock file is authoritative.

```bash
uv sync --frozen --group dev
uv run python -c "import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.backends.mps.is_available())"
```

For deterministic cache placement:

```bash
export TORCH_HOME="$PWD/tmp/torch"
export HF_HOME="$PWD/tmp/huggingface"
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
```

Do not set offline mode until the official R3D-18 weights have been downloaded.
The repository does not version the 127 MB torchvision checkpoint.

## Release integrity

```bash
uv run python scripts/validate_release.py
uv run pytest -q
uv run ruff check .
```

The validator checks SHA-256 hashes of frozen assets, M22 model identity,
27-growth membership, exclusion of 6081, non-retrieval inference, strict
outer-LOO fold integrity, exact published metrics, the frozen 6063 result, and
the absence of large or historical tracked artifacts.

## End-to-end prediction

```bash
uv run morphmbe-predict VIDEO_PATH \
  --sample-id SAMPLE_ID \
  --config configs/morphmbe_m22_realtime.json \
  --output-dir artifacts/predictions/SAMPLE_ID
```

Deterministic seed is `integer_sample_id * 1,000,003 + event_frame * 97` after
removing an optional N/n prefix. The selected event is the retained event with
the maximum selector score. The command records all source-pixel ROIs, retained
events, model identity, physical units, confidence, and the exact arrays used by
the predictor.

For the frozen 6063 regression case, equivalence requires:

- 813 decoded frames and 20 retained events;
- selected frame 189;
- reference-run predicted Sq 5.0612720509 nm, with absolute tolerance
  `0.002 nm` across repeated CPU/MPS executions;
- predicted FSMI 3.8718539184 nm;
- generated height-map standard deviation within `1e-4 nm` of predicted Sq;
- identical selected-16, physics, causal-view, key-frame, ROI, and event data
  relative to the frozen baseline under the same runtime and weight cache;
- AFM-map correlation above `0.999999` and SSIM above `0.99999`.

Torchvision R3D-18 convolution produces small floating-point variation on both
CPU and Apple MPS even with fixed input and seed; repeated untouched-standalone
runs varied by approximately `0.0012 nm` in Sq. Wall-clock times and the
`inference_seconds`/`total_seconds` JSON fields are also not expected to be
identical. See `RELEASE_VERIFICATION.md` for the measured release comparison.

## Reproducing the retrospective experiment

`configs/morphmbe_m22.json` contains the exact M22 method selection, seed,
renderer, cohort, target variant, and source-table paths. Complete regeneration
requires the private raw/derived research data excluded from Git. With those
paths staged, run:

```bash
uv run python -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/morphmbe_m22.json --device auto
```

Never point this command at a publication freeze or raw-data directory as an
output. Use a fresh path under `artifacts/` by changing the output/report roots
in a copied, untracked configuration.

## Publication figures

```bash
uv run python scripts/build_nanoletters_m22_figure_package.py \
  --config configs/morphmbe_m22.json \
  --output artifacts/nanoletters_m22
uv run python scripts/validate_nanoletters_m22_figure_package.py \
  artifacts/nanoletters_m22
```

The builder refuses a nonempty output directory. SVG text remains editable;
CSV data and exact Python source are copied into the package. Internal growth
IDs are separated from manuscript-facing public sample labels.
