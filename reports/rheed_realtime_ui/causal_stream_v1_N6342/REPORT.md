# Causal multi-event RHEED UI repair

Date: 2026-07-29

## Root cause

The prior desktop UI reused the single-input paper-evaluation protocol:
`analyze_replay` decoded the complete recording before playback and
`best_visible_cycle` retained one globally best rotation vertex. That behavior
is reproducible for leave-one-growth-out experiments, but it is not a valid
simulation of a live camera stream and yields only one timeline prediction.

## Corrected live protocol

- The first 48 arrived frames are reserved for causal aperture and ROI
  initialization.
- After warm-up, every frame is processed once in acquisition order.
- A physical trajectory vertex at frame `k` is confirmable at `k+4`; no later
  frame is used by the clear-moment detector.
- One prediction is submitted at `k+8`, when the frozen selected-16 model
  context is complete.
- Every accepted clear moment is queued. The prediction queue is unbounded, so
  an event is not discarded merely because M12a is still generating the
  preceding AFM image.
- Offline paper experiments retain their separate single-best-frame protocol.
- N6389/N6390 clockwise orientation correction remains active in the live
  stream. Their archived single-keyframe locks are deliberately disabled in
  multi-event mode.

## Clear-moment model

The online ranker uses absolute trajectory and image-visibility descriptors;
it does not use whole-video percentile ranks or future candidates. It was fit
on 642 physical candidates from 25 annotated videos.

Strict leave-one-video-out validation:

| Metric | Result |
|---|---:|
| MAE of human-pattern similarity | 0.0950 |
| Pearson r | 0.7089 |
| Spearman rho | 0.7117 |
| AUC for similarity >= 0.50 | 0.8600 |
| Held-video overlap | 0 |

The `0.40` event threshold is intentionally a coverage-oriented operating
point. Low-confidence morphology predictions remain visibly marked in the UI.
Prospective validation on a newly connected industrial camera is still
required.

## Full N6342 replay audit

- Source frames decoded: 2201
- Full-video pre-analysis: false
- Accepted clear moments: 13
- Frames:
  `129, 221, 312, 495, 678, 862, 1045, 1228, 1321, 1595, 1962, 2054, 2145`
- Model-input ROI from the causal warm-up:
  `x=301, y=120, width=649, height=899`
- Raw video was read-only.

The preliminary Qt integration smoke completed predictions at frames 129, 221
and 312 and displayed all three points on the Sq timeline before its deliberately
short screenshot was captured:

`outputs/rheed_realtime_ui/causal_stream_v1_ui_N6342_three_predictions_final.png`

That smoke was not a complete-video event-count validation. The final Qt test
therefore added an explicit completion barrier and ran all 2201 frames. The UI
may display `DRAINING x/13` after video decoding ends, but it cannot display
`COMPLETE` or start another video until all four counts agree:

| End-to-end count | N6342 result |
|---|---:|
| Clear moments detected | 13 |
| Prediction triggers accepted by inference queue | 13 |
| M15b + M12a predictions completed | 13 |
| Sq scatter points displayed | 13 |
| Generated AFM archives written | 13 |

The final frames are exactly
`129, 221, 312, 495, 678, 862, 1045, 1228, 1321, 1595, 1962, 2054, 2145`.
The generated-map Sq matches its conditioning Sq for every event; the maximum
absolute numerical discrepancy is `1.41e-6 nm`. Inference took 6.40--7.79 s
per event on this M1 Pro run. The output is a sequence of prospective estimates
from one video, not 13 independent AFM ground-truth measurements; low confidence
is therefore retained visibly rather than interpreted as accuracy.

Final complete-session artifacts:

- UI with `COMPLETE 13/13`, 13 timeline points, and final generated AFM:
  `outputs/rheed_realtime_ui/causal_stream_v1_ui_N6342_thirteen_predictions_final.png`
- Shared-height-scale atlas of all 13 generated AFM maps:
  `N6342_all_13_generated_afm.png` and
  `N6342_all_13_generated_afm.pdf`
- Complete per-event scalar/confidence/provenance table:
  `ui_thirteen_prediction_timeline.csv`

Machine-readable audit artifacts:

- `accepted_online_events.csv`
- `audit_manifest.json`
- `ui_three_prediction_summary.csv`
- `ui_thirteen_prediction_timeline.csv`
