# RHEED2Morph

RHEED2Morph is a research pipeline for preparing paired RHEED and AFM data for
morphology prediction. The current code focuses on reproducible data staging:
pairing samples, extracting AFM height maps, and writing summary metadata that
can support later modeling experiments.

This repository is an initial feasibility demo. Any generated model output or
analysis from the current pipeline should be treated as exploratory, not as a
final scientific result.

## Current Canonical Baseline

The current publication-oriented baseline is frozen in two immutable packages:

- `publication_freeze/rheed_afm_single_frame_v1_2026-07-18`
- `publication_freeze/prospective_unseen_single_frame_v1`

The retrospective freeze contains the 23 strict historical growth groups and
the selected single-frame RHEED-to-AFM result. The prospective package contains
five unseen RHEED samples, four currently matched AFM truth samples, the N6390
missing-AFM state, and the N6324 AFM-side mismatch record.

Supporting cleanup and provenance notes:

- `docs/repository_map.md`
- `docs/data_provenance.md`
- `docs/current_scientific_baseline.md`
- `docs/legacy_experiments_index.md`


## AFM Overview Figures

The pipeline also renders two 1 x 1 um AFM overview figures. Each sample appears
at most once in each grid, and each subplot includes its own height colorbar in
nm.

Processed AFM height maps:

<img src="./reports/figures/afm_scan_size_grids/processed_afm_scan_size_1um_grid.png" alt="Processed AFM 1 x 1 um overview" width="50%">

Plane-corrected AFM height maps:

<img src="./reports/figures/afm_scan_size_grids/plane_corrected_afm_scan_size_1um_grid.png" alt="Plane-corrected AFM 1 x 1 um overview" width="50%">

## Repository Structure

```text
.
├── data/
│   ├── README.md
│   └── manifests/
│       └── README.md
├── notebooks/
├── outputs/
├── reports/
├── scripts/
│   ├── batch_extract_afm_by_sample.py
│   ├── inspect_afm_raw.py
│   └── make_pairs.py
├── src/
│   └── rheed2morph/
│       ├── afm/
│       ├── pairing/
│       ├── pipeline/
│       ├── rheed/
│       └── utils/
├── tests/
├── run_pipeline.py
├── pyproject.toml
└── uv.lock
```

Large raw and generated files are intentionally excluded from git. Keep raw
inputs under `data/raw/`, generated pairs under `data/pair/`, processed arrays
under `data/processed_afm/`, and experiment artifacts under `outputs/`.

## Setup

```bash
uv sync
```

If you only need to run a single command without activating an environment, use
`uv run ...` as shown below.

## Basic Commands

Create paired AFM/RHEED folders:

```bash
uv run python scripts/make_pairs.py \
  --afm_root data/raw/raw_AFM \
  --rheed_root data/raw/raw_RHEED \
  --pair_root data/pair
```

Inspect one AFM raw file:

```bash
uv run python scripts/inspect_afm_raw.py \
  --input data/pair/6022/AFM/N6022_Ctr_000 \
  --output_dir data/processed_afm
```

Batch extract AFM height maps:

```bash
uv run python scripts/batch_extract_afm_by_sample.py \
  --pair_root data/pair \
  --output_root data/processed_afm
```

Run the full pipeline, including AFM descriptor-to-image reconstruction:

```bash
uv run python run_pipeline.py
```

Run only the reconstruction experiments from existing `data/plane_corrected_afm/`:

```bash
uv run python run_pipeline.py recon
```

Run the frozen-encoder RHEED-to-AFM descriptor MVP baseline:

```bash
uv run python scripts/rheed_to_afm_descriptor_mvp.py
```

Build clean one-to-one manifests by AFM scan size:

```bash
uv run python scripts/build_one_to_one_manifests.py \
  --out-dir data/manifests \
  --target-sizes 1.0 0.5 5.0 \
  --size-tolerance 0.05
```

Run descriptor MVP on each one-to-one manifest and write a comparison summary:

```bash
uv run python scripts/run_one_to_one_experiments.py --device cuda
```

Prepare full-frame PNG libraries from RHEED MP4/MOV videos for manual keyframe
selection:

```bash
uv run python scripts/prepare_rheed_keyframe_selection.py \
  --input-root data/pair \
  --output-root data/rheed_keyframe_selection
```

Run only one sample:

```bash
uv run python scripts/prepare_rheed_keyframe_selection.py \
  --input-root data/pair \
  --output-root data/rheed_keyframe_selection \
  --sample-id 6022
```

The output layout is:

```text
data/rheed_keyframe_selection/
  <sample_id>/
    metadata.json
    videos/
      <video_id>/
        frames/
          0.png
          1.png
          ...
```

Each sample has one `metadata.json`. For every video, manually edit only the
`selection` fields: `keyframe_index` is a 0-based PNG index, and
`clip_frame_count` is the total number of frames in the later clip, including
the keyframe itself. The exported images are lossless PNGs at the decoded
video's original width, height, and RGB color pixels. This step does not apply
grayscale conversion, crop, resize, exposure adjustment, brightness
normalization, background subtraction, ROI extraction, or geometric alignment.
Later model datasets can read continuous clips directly from the PNG folders
without decoding MP4 again. The original MP4 files remain the highest-level
source data and must be kept permanently; PNGs preserve decoded frames
losslessly but cannot recover information already lost in MP4 encoding.

Review keyframes and draw manual source-pixel ROIs:

```bash
uv run python tools/manual_rheed_roi_reviewer.py \
  --root data/rheed_keyframe_selection
```

The reviewer restores saved selections from each sample's `metadata.json`.
Dragging a rectangle may use a scaled display, but saved ROI coordinates are
written in original PNG pixel coordinates under `selection.roi`. `Save and Next`
preserves all other metadata and advances to the next incomplete video. Optional
filters include `--sample-id`, `--video-id`, and `--start-from-first`.

Preview the pipeline without modifying files:

```bash
uv run python run_pipeline.py --dry-run
```

## Tests

The smoke tests use the Python standard library test runner:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests
```

These tests verify package imports, expected data manifest layout, and basic
pipeline path configuration.
