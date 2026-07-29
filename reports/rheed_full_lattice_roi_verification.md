# V7 full-lattice ROI verification

Date: 2026-07-28

## Scientific integrity

- Retained annotated videos: 25.
- Excluded before fitting/evaluation: 6023 and 6087, matching
  `removelist.txt`.
- Outer protocol: strict leave-one-video-out, 24 train / 1 held.
- Held-video overlap summed across the selected method: 0.
- Manual ROI and DoG spot support are evaluation-only.
- V5 keyframe scores are computed from the frozen tracking ROI before V7 is
  applied.
- Raw sources are opened read-only; outputs are additive under
  `outputs/` and `reports/`.

## Automated checks

The targeted selector suite covers ROI geometry, source loading, temporal
tracking, V4/V5 feature extraction and manual-review helpers: 51/51 tests
pass in 4.16 s in the project environment. The V7 tests assert that a
synthetic sparse right-hand point family is retained, circular arc intrusion
is zero, and the export ROI cannot accidentally replace the frozen tracking
ROI.

Artifact integrity checks validate:

- 25 selected-method held predictions;
- zero held/removelist overlap;
- complete fitted-bundle provenance for all 25 retained samples;
- numerical agreement between CSV and metadata summaries;
- readable PNG/PDF comparison figures;
- readable JSON/PNG outputs for real 6063 and 6048 inference.

Seven PNG files were decoded and verified with Pillow. Seven PDF files passed
PDF header/EOF integrity checks. Python compilation, `uv lock --check`, and
`git diff --check` also pass.

## Manual visual review

All five fixed-order atlas pages (25/25 videos) and the lowest-coverage
eight-sample panel were inspected at full resolution. The final crop contains
the visible point family and its right transition, provides vertical margin,
and does not contain the circular eyepiece edge. In particular, previously
severely clipped samples 6048, 6056 and 6085 are corrected.

## Runtime smoke tests

```text
6063 MOV: 813 frames, selected f189, 28.71 s, arc intrusion 0
6048 PNGs: 371 frames, selected f191, 25.08 s, arc intrusion 0
```

The 6048 automatic frame is eight frames from the human reference (199) and
visually retains the full right-hand point family. CUDA is not required for
this task.

## Remaining boundary

Repository-wide tests contain known failures caused by unrelated absent
historical freeze/checkpoint/parquet artifacts. The task-scoped tests and
artifact checks are the acceptance boundary for V7; this continuation does
not alter those unrelated components.
