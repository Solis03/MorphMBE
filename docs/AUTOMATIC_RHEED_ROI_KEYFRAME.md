# Automatic RHEED ROI and keyframe selection

This tool reads a complete RHEED video, detects a border-safe rectangular
region of interest, tracks the periodic diffraction motion, and selects the
most plausible clear rotation vertex. It never writes to the input video.

## Recommended command

Run from the repository root with the existing environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_rheed_roi_keyframe.py \
  "data/pair/6022/RHEED/After GaSb.MP4" \
  --output-dir "outputs/my_automatic_selection"
```

The default command uses:

- `calibrated_safe` aperture ROI;
- a compact brightest-feature trajectory and a whole-pattern diffraction
  front trajectory;
- physical right-most vertex candidates;
- the frozen supervised phase ranker at
  `outputs/rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4/supervised_phase_ranker/ridge_phase_ranker.joblib`.

The output directory contains:

- `automatic_selection.json`: source provenance, ROI coordinates, every
  heuristic prediction, the selected frame, and confidence;
- `selected_frame_with_roi.png`: full selected frame with ROI overlay;
- `selected_roi.png`: contrast-enhanced selected crop.

The default selected method is `supervised_phase_ranker`. To use a
deterministic method without the fitted ranker:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_rheed_roi_keyframe.py \
  "input.MOV" \
  --output-dir "outputs/physics_only" \
  --phase-ranker "" \
  --selected-method compact_visibility
```

Supported video suffixes are AVI, M4V, MKV, MOV, MP4, MPEG and MPG. A directory
of lossless numeric PNG frames is also accepted.

## Method

ROI detection estimates the illuminated phosphor aperture from a temporal
median, erodes it to a safe interior, aggregates high-pass diffraction
activity over sampled frames, and searches for a fixed-aspect rectangle with
less than 0.2% unsafe pixels. The selected `calibrated_safe` method balances
aperture safety, useful diffraction coverage and the scale of existing human
ROIs.

Two trajectories are extracted from every frame:

1. the compact, highest-response diffraction feature corresponding most
   closely to the operator's “brightest point”;
2. the 85th-percentile horizontal front of the complete diffraction family,
   which is less likely to jump between vertical streaks.

Local horizontal maxima are candidate vertices. Candidate features encode
clarity, raw visibility, horizontal prominence, the before/after reversal,
upward motion and agreement between the two trackers. The final Ridge ranker
was trained on candidates from the 25 annotated videos that do not occur in
`removelist.txt`, only after strict leave-one-video-out evaluation. Its
confidence is an isotonic
estimate of expected human-frame composite similarity, not a probability that
the frame is “correct.”

## Validation and limitations

The strict leave-one-video-out result has 24 annotated training videos and one
completely excluded held video in each fold. Median human–machine pattern NCC
is 0.714 and median SSIM is 0.482 over all 25 held predictions. The confidence
score is significantly related to realized error (Spearman rho = -0.548,
p = 0.0046).

This is retrospective development evidence. The final exported ranker is
refit on all 25 retained annotations and must be prospectively tested on newly
acquired videos. Samples 6048, 6063 and 6056 remain important failure cases.
For scientific acquisition, low confidence should trigger manual review and
the reviewed frame should be added to the next training freeze.

## Streaming adaptation

The current implementation is an accurate two-pass offline reference. On the
M1 Pro, the 810-frame, 27-second 6022 MP4 was processed in 12.9 seconds
(approximately 62.7 decoded/scored frames per second). A causal version can:

1. estimate and freeze the aperture ROI during the first one or two seconds;
2. update both trajectory features per frame;
3. buffer approximately nine frames to confirm a local vertex (about 0.27 s
   at 30 fps);
4. score every confirmed clear vertex and emit the first one above a chosen
   confidence threshold.

That causal state machine is the recommended production follow-up. It should
be benchmarked against new videos before connecting it to growth-control
logic.
