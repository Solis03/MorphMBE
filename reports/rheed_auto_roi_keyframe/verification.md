# Verification record

Date: 2026-07-28

## Passed checks

- `PYTHONPATH=src .venv/bin/python -m pytest
  tests/test_automatic_rheed_roi_keyframe.py
  tests/test_manual_rheed_roi_reviewer.py
  tests/test_rheed_frame_selection.py
  tests/test_rheed_frame_selection_v2.py
  tests/test_rheed_keyframe_selection.py
  tests/test_rheed_single_frame_manual.py -q`
  - 48 passed.
- Python compile check passed for the automatic selector, CLI, experiment
  scripts and tests.
- 167 generated PNG files opened and passed Pillow verification.
- 56 generated PDF files have valid PDF headers, EOF markers and nontrivial
  size. Primary V2 deterministic comparison atlases and canonical V4 Ridge
  benchmark, confidence and held-video atlas PNGs were also visually inspected.
- V2 trajectory/ROI row-integrity checks passed:
  - 81 ROI predictions = 27 videos x 3 ROI methods;
  - 162 heuristic keyframe predictions = 27 videos x 6 methods;
  - 724 labelled physical vertex candidates in the noncanonical diagnostic.
- Canonical V4 checks passed:
  - 25 strict leave-one-video-out Ridge predictions;
  - samples 6023 and 6087 have zero rows;
  - zero held-video overlap;
  - 642 labelled physical vertex candidates.
- The fitted V4 `ridge_phase_ranker.joblib` loads successfully and records 25
  training videos.
- Every V1/V2/V3 run audit reports unchanged raw source video size and
  modification time.
- Direct source-video inference on
  `data/pair/6022/RHEED/After GaSb.MP4` completed successfully in 12.9 seconds
  for 810 frames and wrote JSON, overlay and ROI crop outputs.
- `git diff -- data removelist.txt` is empty.

## Repository-wide test boundary

`PYTHONPATH=src .venv/bin/python -m pytest tests -q` produced:

- 334 passed;
- 24 failed;
- 6 errors.

The failures are outside this task:

- the latest paper-freeze directory is missing ignored
  `FREEZE_MANIFEST.json` and related runtime artifacts;
- the unrelated `rheed_peak_saddle` workflow lacks required human-checkpoint
  and ignored output files;
- two historical phase-4 tests require `pyarrow` or `fastparquet`, neither of
  which is installed in the existing environment.

No placeholder checkpoint, freeze manifest or raw/derived scientific record
was fabricated to make those tests pass.

Running plain `pytest` from the repository root also collects duplicated test
module names inside a historical code snapshot and stops on import-file
mismatch. Scoping to `tests/` avoids that unrelated discovery problem.

## Independent review notes

- The final V4 supervised metrics use strict video-level leave-one-out; candidate
  labels from the held video never enter its fold model.
- The final model and its validation exclude every `removelist.txt` overlap.
  Earlier 27-video experiments are labelled noncanonical diagnostics.
- ROI and keyframe development is retrospective. The final retained-data model
  is correctly described as awaiting prospective validation.
- Repeated rotation cycles mean absolute frame difference is not treated as
  the primary scientific endpoint.
- Failure cases 6048, 6063 and 6056 are included in tables and figures.
- Confidence is described as expected human-frame similarity, not a
  correctness probability.
- V1 and rejected V3 artifacts remain separate and were not overwritten.
