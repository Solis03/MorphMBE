# Deep spot-visibility selector verification

Date: 2026-07-28

## Passed task checks

- Targeted selector and adjacent manual-frame tests:
  `49 passed in 4.59s`.
- Python compilation passed for the automatic selector, spot-visibility
  module, CLI and V5/V6 experiment program.
- Complete original-video inference passed for
  `data/pair/6063/RHEED/rampdown to 300C.MOV`:
  - 813 frames;
  - 30.1 seconds on Apple M1 Pro MPS;
  - automatic frame 189 versus human frame 186;
  - ROI safe-pixel fraction 0.9980;
  - deep confidence 0.602.
- V5 strict leave-one-video-out integrity:
  - 25 held predictions;
  - 24 training videos per fold;
  - 642 candidate rows;
  - zero held-video overlap;
  - zero rows for removelist overlaps 6023 and 6087.
- V6 strict leave-one-video-out integrity passes the same split checks.
- Both fitted bundles load and record 25 training videos, 642 candidates,
  zero held overlap and the pinned foundation-model revisions.
- V5 contains 48 PNG and 48 PDF result figures; V6 contains 38 PNG and
  38 PDF figures. Every PNG passes Pillow verification and every PDF has a
  valid header, EOF marker and nontrivial size.
- Across all automatic-selector experiments, 253 PNG and 142 PDF artifacts
  are present.
- V5 full-sample, confidence and failure figures were visually inspected.
- `git diff -- data removelist.txt` is empty.

## Repository-wide boundary

`PYTHONPATH=src .venv/bin/python -m pytest tests -q` produced:

- 335 passed;
- 24 failed;
- 6 errors.

No failure originates in the automatic ROI/keyframe selector. The unchanged
pre-existing failures are:

- ignored paper-freeze manifests and related runtime artifacts are absent;
- the unrelated `rheed_peak_saddle` workflow is awaiting its required human
  checkpoints and generated receipts;
- two historical tests require `pyarrow` or `fastparquet`, neither installed
  in the existing environment.

No scientific checkpoint, freeze receipt, parquet file or raw datum was
fabricated to make those tests pass.

## Independent review

- DINOv2 is frozen; scaling, PCA and all fitted heads are trained inside each
  strict video-level fold.
- Image feature extraction uses no AFM target and no held-video label.
- The diffuse-shadow proxy compares selected and human spot mass only for
  evaluation. It is not an input label.
- V5 selection was retrospective and is correctly described as awaiting a
  new prospective video cohort.
- DINOv2-Base V6 is reported as a rejected negative result rather than being
  hidden.
- Remaining failures 6022, 6048, 6056, 6062 and 6080 are retained in
  all-sample/failure figures.
- Confidence is expected similarity, not a correctness probability.
- The error relation is reported for the raw negative-margin reliability
  signal. A leave-one-prediction-out isotonic calibration audit is
  nonsignificant (rho -0.130, p 0.537), so the absolute calibrated confidence
  is explicitly marked for prospective recalibration rather than presented
  as independently validated.
