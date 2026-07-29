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
- 27 explicit multi-scale spot, haze and frequency descriptors;
- the pinned 22-million-parameter DINOv2-S/14 visual foundation model;
- a fold-validated Ridge + pairwise ranker + ExtraTrees ensemble with a
  low-visibility candidate gate;
- the fitted ensemble at
  `outputs/rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/dinov2_spot_visibility_ranker.joblib`.

The output directory contains:

- `automatic_selection.json`: source provenance, ROI coordinates, every
  heuristic prediction, the selected frame, and confidence;
- `selected_frame_with_roi.png`: full selected frame with ROI overlay;
- `selected_roi.png`: contrast-enhanced selected crop.

The default selected method is `deep_visibility_ranker`. The original
removelist-compliant V4 Ridge model is preserved and is still reported as
`supervised_phase_ranker`. To use a
deterministic method without the fitted ranker:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_rheed_roi_keyframe.py \
  "input.MOV" \
  --output-dir "outputs/physics_only" \
  --phase-ranker "" \
  --deep-visibility-ranker "" \
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

Local horizontal maxima are candidate vertices. The V4 features encode
clarity, raw visibility, horizontal prominence, the before/after reversal,
upward motion and agreement between the two trackers.

V5 then analyzes the actual candidate crops. Multi-scale
difference-of-Gaussian responses count compact spots and measure peak
prominence, spot-energy concentration, vertical point-family organization,
high-to-low-frequency energy and low-frequency haze. A frozen DINOv2-S/14
encoder adds a 1,152-dimensional representation formed from its CLS token and
the mean and standard deviation of its patch tokens. PCA, regression,
pairwise ranking and tree heads are fitted only on training videos inside
each fold.

The final score blends fold-fitted DINOv2 Ridge, pairwise RankIQA-style,
visual ExtraTrees and explicit visibility ranks. Candidates below the 25th
within-video visibility percentile are rejected when other candidates exist.
All fitting uses the 25 annotated videos with no `removelist.txt` overlap.
The reported confidence is an isotonic estimate of expected human-frame
similarity derived from the strict-LOO selection-margin reliability signal;
it is not a probability that a frame is “correct.”

## Validation and limitations

The strict leave-one-video-out result has 24 annotated training videos and one
completely excluded held video in each fold. V5 median human–machine pattern
NCC is 0.820, mean NCC is 0.730, median SSIM is 0.559 and median gradient NCC
is 0.583 over all 25 held predictions. Median absolute frame difference is
3 frames. The predefined diffuse-shadow proxy falls from 16% for V4 to 4%
for V5. The reliability confidence is significantly related to realized error
(Spearman rho = -0.459, p = 0.021).

This is retrospective development evidence. The final exported ranker is
refit on all 25 retained annotations and must be prospectively tested on newly
acquired videos. The user-identified 6063 failure changes from V4 frame 461
(NCC 0.279) to V5 frame 188 (NCC 0.784), two frames from the human choice.
Samples 6056 and 6080 remain important phase-selection failures. Sample 6022
is the one remaining diffuse-shadow-proxy failure.
For scientific acquisition, low confidence should trigger manual review and
the reviewed frame should be added to the next training freeze.

## Streaming adaptation

The deterministic implementation is a two-pass offline reference. Deep V5
adds one candidate-frame pass and frozen foundation-model inference. The
DINOv2-S extraction benchmark processed 598 unique evaluation images in
35.5 seconds on MPS (16.9 images/s). The complete V5 CLI processed the
813-frame 6063 MOV in 30.1 seconds and selected frame 189, three frames from
the human choice. A causal version can:

1. estimate and freeze the aperture ROI during the first one or two seconds;
2. update both trajectory features per frame;
3. buffer approximately nine frames to confirm a local vertex (about 0.27 s
   at 30 fps);
4. score every confirmed clear vertex and emit the first one above a chosen
   confidence threshold.

That causal state machine is the recommended production follow-up. It should
be benchmarked against new videos before connecting it to growth-control
logic.
