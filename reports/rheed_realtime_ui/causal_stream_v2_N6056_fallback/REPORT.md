# N6056 zero-event repair

Date: 2026-07-29

## Failure

The 488-frame, 30 fps N6056 recording completed with
`detected=triggered=completed=scatter=0`. The previous UI incorrectly treated
the numerical equality of four zeros as success and displayed `COMPLETE 0/0`.

The video did contain useful RHEED patterns. The primary causal detector
proposed nine geometric vertices, but none passed all gates:

- its highest regression score was 0.343 versus the fixed 0.40 threshold;
- the 48-frame warm-up tracking ROI ended at source x=1044 and omitted the
  rightmost part of the diffraction lattice;
- the full model-input ROI extended to x=1308 and recovered a candidate at
  frame 157, four frames from the human reference frame 161.

This is an input-domain/ROI calibration failure, not an empty video.

## Repair

The strict tracking path remains unchanged. A second causal path examines the
already validated full-lattice/model-input ROI only while the strict detector
has never accepted an event.

A fallback candidate must satisfy all of the following absolute gates:

| Gate | Value |
|---|---:|
| clear-moment model score | >= 0.30 |
| visibility proxy | >= 1.30 |
| shadow fraction | <= 0.20 |
| visible spot count | >= 8 |
| clarity | >= 8.0 |

The fallback waits until `k+8`, so a nearby strict event can suppress it and
the selected-16 context is already complete. Accepted fallback events are at
least 3.0 seconds apart. Once the primary path succeeds, fallback is disabled
for that stream. This preserves the validated primary detector rather than
globally lowering its threshold.

The UI no longer treats four zeros as success. A genuine zero-event stream is
reported as `NO CLEAR MOMENT` with an explicit error log.

## End-to-end results

| Count | Result |
|---|---:|
| decoded frames | 488 |
| strict accepted events | 0 |
| full-lattice fallback events | 2 |
| inference triggers accepted | 2 |
| M15b + M12a predictions completed | 2 |
| Sq scatter points | 2 |
| generated AFM archives | 2 |

| Event frame | Human relation | Sq (nm) | FSMI (nm) | model confidence |
|---:|---|---:|---:|---:|
| 157 | 4 frames from human frame 161 | 2.288 | 1.908 | 59.8% |
| 314 | second causal clear moment | 1.709 | 1.379 | 80.2% |

The two generated maps match their conditioning Sq within `3.18e-6 nm`.
Inference took 7.02 and 7.13 seconds on the M1 Pro.

## Regression protection

The complete N6342 replay was rerun with the fallback enabled. It retained
exactly the previously validated 13 strict events:

`129, 221, 312, 495, 678, 862, 1045, 1228, 1321, 1595, 1962, 2054, 2145`

N6342 used zero fallback events, so the prior 13/13 result is unchanged.

## Artifacts

- `audit_manifest.json`: N6056 causal detector audit.
- `ui_two_prediction_timeline.csv`: per-event scalar, confidence and generated
  artifact provenance.
- `N6056_rheed_to_generated_afm_events.{png,pdf}`: detected RHEED crops beside
  generated AFM.
- `N6056_all_2_generated_afm.{png,pdf}`: shared-height-scale AFM comparison.
- `outputs/rheed_realtime_ui/causal_stream_v2_ui_N6056_two_predictions_final.png`:
  actual Qt UI with `COMPLETE 2/2`.

The raw RHEED video remained read-only. Its SHA-256 is
`8e36f1a697af4986a0f004de8e46be1181f32f0c7eb13ab19bd378f93907c0e6`.
