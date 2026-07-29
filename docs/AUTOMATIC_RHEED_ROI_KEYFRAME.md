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

- the frozen `calibrated_safe` aperture ROI for trajectory extraction and
  V5 keyframe scoring;
- a post-selection V7 `full_lattice` ROI for the exported crop;
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

- `automatic_selection.json`: source provenance, the exported full-lattice
  ROI, the smaller tracking ROI, every heuristic prediction, the selected
  frame, and confidence;
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

## ROI method

ROI detection first estimates the illuminated phosphor aperture from a
temporal median. The frozen `calibrated_safe` tracking rectangle then
aggregates high-pass diffraction activity over sampled frames and searches
for a fixed-aspect safe rectangle. This geometry is retained internally
because the V5 keyframe ranker was trained with it.

V7 predicts the exported crop only after keyframe scoring. It calibrates
left, right, top and bottom independently relative to the detected aperture,
uses separate landscape/portrait statistics, and deliberately includes the
right light/shadow transition. Conservative 5th/95th-percentile bounds and
extra top/bottom padding retain vertical motion around the selected frame.
The left edge is constrained row by row to remain inside the circular
eyepiece arc. Thus the crop can include the complete sparse point family
without inheriting the fixed aspect ratio that caused V4 to clip its right,
top or bottom.

The fitted V7 calibration is:

`outputs/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7/full_lattice_roi_calibration.joblib`

To reproduce the previous smaller exported crop while leaving keyframe
selection unchanged, pass `--full-lattice-roi-calibration ""`.

## Keyframe method

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

The V7 ROI was separately evaluated by strict leave-one-video-out
calibration over the same 25 removelist-compliant videos. Relative to the V4
tracking crop, median compact-spot energy coverage rises from 0.501 to 1.000,
worst-case coverage from 0.063 to 0.997, median human-ROI area coverage from
0.798 to 0.975, right reference-boundary inclusion from 4% to 100%, and the
circular-edge intrusion rate falls from 72% to 0%. The vertical reference
envelope is included for 24/25 videos. Compact-spot coverage is an
evaluation-only difference-of-Gaussians measure computed inside the manual
reference ROI; manual pixels are never used during inference. The complete
atlas, including the eight lowest-coverage cases, is under
`reports/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7`.

V7 intentionally trades some IoU for content completeness: empty margin in a
manual rectangle is not treated as more important than retaining all
diffraction spots while avoiding the eyepiece arc. Like V5, this remains a
retrospective result and needs prospective validation on new camera
geometries. If the camera, eyepiece magnification, or screen orientation
changes materially, collect several reviewed ROIs and refit the small
calibration bundle.

## Streaming adaptation

The deterministic implementation is a two-pass offline reference. Deep V5
adds one candidate-frame pass and frozen foundation-model inference. The
DINOv2-S extraction benchmark processed 598 unique evaluation images in
35.5 seconds on MPS (16.9 images/s). The final V5+V7 CLI processed the
813-frame 6063 MOV in 28.7 seconds and selected frame 189, three frames from
the human choice. It processed the 371-frame 6048 PNG sequence in 25.1
seconds and selected frame 191 versus human frame 199. A causal version can:

1. estimate and freeze the aperture ROI during the first one or two seconds;
2. update both trajectory features per frame;
3. buffer approximately nine frames to confirm a local vertex (about 0.27 s
   at 30 fps);
4. score every confirmed clear vertex and emit the first one above a chosen
   confidence threshold.

That causal state machine is the recommended production follow-up. It should
be benchmarked against new videos before connecting it to growth-control
logic.
