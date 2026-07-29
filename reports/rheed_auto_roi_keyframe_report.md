# Automatic RHEED ROI and rotation-phase keyframe selection

Date: 2026-07-28
Branch: `codex/rheed-auto-roi-keyframe-20260728`

> Continuation note: the V4 result frozen in this report remains the canonical
> pre-image-content baseline. It has been superseded for default inference by
> the removelist-compliant DINOv2/spot-visibility V5 model documented in
> [the deep visibility report](rheed_deep_visibility_keyframe_report.md).

## Outcome

An end-to-end, read-only tool now accepts complete MOV/MP4/AVI-style RHEED
videos, estimates a useful ROI without the circular eyepiece border, tracks
periodic diffraction motion, selects a right-most rotation vertex and reports
a validated confidence score. The recommended method is:

> `calibrated_safe` ROI + dual physical trajectories + supervised Ridge
> phase-candidate ranker, with all `removelist.txt` samples excluded.

The final ranker was evaluated by strict leave-one-video-out: each of 25 folds
trained on candidates from 24 annotated videos and selected a frame from the
one excluded video. Samples 6023 and 6087 overlap `removelist.txt` and have
zero participation in final fitting, evaluation or confidence calibration.
The total held-video overlap is zero.

## Data and protocol

The source annotation inventory contains 27 videos in
`outputs/rheed_video_afm_story/phase1/modeling_manifest.csv` that have a human
keyframe and ROI. The final V4 cohort contains the 25 videos with no
`removelist.txt` overlap. The two excluded RHEED annotations are 6023 and
6087.

V1–V3 were broad RHEED-only method-development diagnostics over all available
annotations. A final compliance audit caught the two removelist overlaps;
those broad results remain labelled noncanonical and are not the delivered
model. Raw RHEED files were read only. Every full-video run records source
size and modification time before and after processing, and all sources
passed the unchanged audit.

Multiple rotation periods make absolute frame difference a secondary metric:
another period can contain an equally valid physical vertex. The evaluation
therefore reports:

- nearest detected vertex to the human frame;
- periodic phase error;
- normalized cross-correlation (NCC), SSIM and gradient NCC between human and
  machine RHEED patterns;
- machine/human clarity ratio;
- full, fixed-order qualitative atlases and explicit failure pages.

All model choices were developed retrospectively on this annotated cohort.
Strict leave-one-video-out estimates cross-video generalization, but the
exported all-data model still requires a new prospective video cohort.

## Methods tested

### ROI models

1. `aperture_inscribed`: the largest border-safe rectangle.
2. `activity_safe`: maximizes high-pass diffraction activity density.
3. `calibrated_safe`: balances aperture safety, activity and the scale of
   human ROIs.

All three achieve a median safe-pixel fraction above 99.8%. The largest
inscribed ROI has the best human ROI overlap, while calibrated-safe covers
more diffraction activity. A controlled V3 experiment showed that the larger
ROI worsens keyframe tracking because it permits switching between unrelated
bright features; it was rejected.

| ROI method | Median IoU | Human-area coverage | Safe fraction | Activity coverage |
|---|---:|---:|---:|---:|
| aperture_inscribed | 0.602 | 0.816 | 0.9981 | 0.272 |
| activity_safe | 0.487 | 0.537 | 0.9982 | 0.313 |
| calibrated_safe | 0.570 | 0.798 | 0.9981 | 0.326 |

IoU is not the sole objective: a machine ROI may safely contain useful
diffraction information outside the narrower human rectangle.

### Keyframe models

The physical model follows the operator's description. A compact brightest
feature moves upward while its horizontal coordinate rises and then falls;
the desired frame is a right-most local vertex. A second tracker follows the
85th-percentile horizontal front of the whole diffraction family. Candidate
features include local clarity, raw mean intensity, raw high-pass contrast,
vertex prominence, pre/post horizontal reversal, upward displacement and
cross-tracker agreement.

The following alternatives were preserved:

- quality-only baseline;
- vertex plus clarity;
- physical vertex scoring;
- absolute-visibility-gated front and compact trackers;
- Ridge, random forest, ExtraTrees and gradient-boosting regressors;
- within-video rank regression;
- top-quintile classification;
- mean, nearest and PCA human-frame templates.

Mean/PCA templates failed because RHEED morphology differs materially across
growth states. On the final 25-video cohort, the compact and auditable Ridge
ranker gave the best balance of median NCC, mean NCC, SSIM, frame difference
and calibrated confidence. A Ridge/gradient rank ensemble was tested and
rejected.

## Results

| Model | Evaluation | Median NCC | Mean NCC | Median SSIM | Median gradient NCC | Median absolute frame difference |
|---|---|---:|---:|---:|---:|---:|
| quality-only | all 27 | 0.505 | — | 0.358 | 0.200 | 111 |
| V1 physics vertex | all 27 | 0.602 | — | 0.433 | 0.277 | 106 |
| V2 compact visibility | all 27 | 0.669 | — | 0.442 | 0.294 | 46 |
| V2 Ridge ranker (noncanonical 27-video diagnostic) | strict video LOO | 0.714 | 0.680 | 0.513 | 0.373 | 74 |
| V2 gradient ranker (noncanonical 27-video diagnostic) | strict video LOO | 0.715 | 0.687 | 0.516 | 0.373 | 24 |
| V3 larger-ROI compact visibility | all 27 | 0.568 | — | 0.379 | 0.256 | 76 |
| **V4 Ridge ranker (removelist compliant)** | **strict video LOO** | **0.714** | **0.670** | **0.482** | **0.362** | **46** |

For V2, the human keyframe is a median 2 frames from the nearest automatically
detected physical vertex. This directly supports the proposed parabolic
trajectory mechanism. The harder problem is selecting the correct clear
vertex among several tracker maxima and repeated cycles.

The final Ridge ranker's held-video predicted similarity is inversely related
to realized error: Spearman rho is -0.548 (p = 0.0046). The exported confidence
is an isotonic expected composite similarity, not a correctness probability.
Cases 6048, 6063 and 6056 remain visible failures and are not omitted. In an
operational workflow, low-confidence cases should trigger review.

## Visual evidence

Primary figures:

- [method benchmark](rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2/method_benchmark.pdf)
- [canonical V4 supervised ranker benchmark](rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4/supervised_phase_ranker/phase_ranker_benchmark.pdf)
- [canonical V4 confidence validation](rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4/supervised_phase_ranker/confidence_validation.pdf)
- [V2 failure cases](rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2/failure_cases/lowest_similarity_cases.pdf)

Complete, non-cherry-picked atlases:

- ROI models:
  `reports/rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2/roi_models/`
- six deterministic keyframe models:
  `reports/rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2/keyframe_models/`
- canonical strict-LOO supervised model:
  `reports/rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4/supervised_phase_ranker/ridge/`
- noncanonical 27-video gradient diagnostic:
  `reports/rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2/supervised_phase_ranker/gradient_boosting/`

V1 and rejected V3 have separate directories and were not overwritten. The
experiment registry is
`reports/rheed_auto_roi_keyframe/experiment_registry.csv`.

## Reproducibility

Recommended inference:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_rheed_roi_keyframe.py \
  "path/to/video.MOV" \
  --output-dir "outputs/automatic_selection"
```

Reproduce the selected experiment and ranker:

```bash
.venv/bin/python analysis/rheed_auto_roi_keyframe/run_experiment.py \
  --config configs/rheed_auto_roi_keyframe_v2.json

.venv/bin/python analysis/rheed_auto_roi_keyframe/train_phase_ranker.py \
  --config configs/rheed_auto_roi_keyframe_v4.json \
  --rebuild-candidates
```

The frozen fitted ranker and metadata are under:

`outputs/rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4/supervised_phase_ranker/`

On the Apple Silicon M1 Pro, the original 810-frame, 27-second 6022 MP4 was
processed directly in 12.9 seconds (about 62.7 frames/s). The extracted PNG
version took 15.6 seconds in a separate smoke run. Accuracy, not speed, was
the present objective.

## Limitations and next step

- The final all-data ranker is not yet prospectively validated.
- Some videos contain bright-feature identity changes that neither tracker
  fully resolves.
- Confidence supports risk-aware review but does not guarantee correctness.
- The implementation is two-pass offline. A production streaming version
  should freeze the ROI from the first seconds, buffer about nine frames to
  confirm a vertex and emit the first clear candidate above a validated
  confidence threshold.
- Before closed-loop MBE use, acquire a new video-only prospective set,
  freeze the current model, measure selection latency and have an operator
  blind-review all predictions.

The local M1 Pro was sufficient; no CUDA handoff is recommended.
