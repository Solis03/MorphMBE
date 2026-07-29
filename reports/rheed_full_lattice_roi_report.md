# Complete-lattice RHEED ROI — V7 report

## Outcome

The V7 ROI corrects the systematic crop failure identified in the V4/V5
automatic selector. It retains the full visible diffraction point family,
including the rightmost spots and light/shadow transition, leaves vertical
motion margin, and excludes the circular eyepiece edge. The validated V5
DINOv2-S keyframe selector is preserved unchanged.

V7 is now the default **export ROI**. The smaller `calibrated_safe` rectangle
remains the internal **tracking ROI**, so the keyframe model sees exactly the
geometry on which it was developed.

## Method

The old ROI maximized activity density inside a fixed-aspect safe rectangle.
That objective could improve density by discarding sparse spots at the
right/top/bottom. V7 replaces that export objective with four independent
boundaries normalized to the detected aperture:

1. fit left, right, top and bottom boundaries from reviewed ROIs;
2. fit landscape and portrait camera geometries separately;
3. use conservative 5th/95th-percentile boundaries plus aperture-relative
   padding;
4. force inclusion of the right aperture/screen transition;
5. move the left edge inside the maximum row-wise circular arc position;
6. apply the result only after V5 has selected the keyframe.

The method is deliberately small-data and geometry-aware. It does not require
a large segmentation network to relearn the stable camera/eyepiece boundary
from 25 labels.

## Leakage-safe evaluation

The canonical cohort contains 25 annotated videos after excluding samples
6023 and 6087 because they overlap `removelist.txt`. In each outer fold, the
four boundary statistics are fitted on 24 videos and applied to the one
completely held video. The summed held-video overlap is zero.

Compact-spot coverage is measured only for evaluation. A multi-scale
difference-of-Gaussians response identifies compact diffraction energy
inside the manual reference ROI, and the fraction retained by the automatic
ROI is reported. Manual pixels, compact-spot masks and held-video boundaries
are not inputs during inference.

## Results

| Metric | V4 tracking ROI | V7 full-lattice ROI |
|---|---:|---:|
| Videos | 25 | 25 |
| Median compact-spot energy coverage | 0.501 | **1.000** |
| Worst compact-spot energy coverage | 0.063 | **0.997** |
| Median compact-spot peak coverage | 0.655 | **1.000** |
| Median manual-ROI area coverage | 0.798 | **0.975** |
| Worst manual-ROI area coverage | 0.310 | **0.758** |
| Right reference-boundary inclusion | 1/25 (4%) | **25/25 (100%)** |
| Vertical reference-envelope inclusion | 9/25 (36%) | **24/25 (96%)** |
| Circular-edge intrusion | 18/25 (72%) | **0/25 (0%)** |
| Median IoU with manual ROI | 0.553 | 0.628 |

The selected q05/q95 method is slightly larger than the q10/q90 alternative:
its median IoU is lower (0.628 versus 0.668), but it improves vertical
reference-envelope inclusion from 23/25 to 24/25. This is the appropriate
trade-off for the user's explicit requirement that the complete moving
point family be retained.

The left manual-margin inclusion rate is only 10/25 because the automatic
left edge is intentionally shifted right when the manual rectangle includes
empty space outside the circular aperture. This does not indicate lost point
content: the worst retained compact-spot energy is 99.65%, while circular-arc
intrusion is zero.

## Real inference

- 6063 original MOV: 813 frames, 28.7 s on Apple MPS, V5 frame 189 versus
  human frame 186, V7 ROI `(474, 54, 588, 984)`, arc intrusion 0.
- 6048 PNG frame sequence: 371 frames, 25.1 s on Apple MPS, V5 frame 191
  versus human frame 199, V7 ROI `(774, 54, 486, 984)`, arc intrusion 0.

Both final overlays were visually inspected. The rightmost vertical point
family and right light/shadow boundary are present, top/bottom margin is
available, and no circular eyepiece arc enters the rectangle.

## Artifacts and reproduction

- Configuration:
  `configs/rheed_auto_roi_keyframe_v7.json`
- Training/evaluation:
  `analysis/rheed_auto_roi_keyframe/train_full_lattice_roi.py`
- Runtime implementation:
  `src/rheed2morph/rheed/lattice_roi.py`
- Calibration:
  `outputs/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7/full_lattice_roi_calibration.joblib`
- Fold predictions and summary:
  `outputs/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7`
- Full 25-video atlas and failure panel:
  `reports/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7`

Reproduce the evaluation:

```bash
PYTHONPATH=src .venv/bin/python \
  analysis/rheed_auto_roi_keyframe/train_full_lattice_roi.py
```

Run the final selector:

```bash
PYTHONPATH=src .venv/bin/python scripts/select_rheed_roi_keyframe.py \
  "input.MOV" --output-dir "outputs/my_selection"
```

## Limitations

This is strict retrospective held-video evidence, not a prospective camera
validation. The 25 labels cover the current camera and eyepiece geometries;
a material magnification, resolution, aperture placement or orientation
change should trigger recalibration. The compact-spot metric measures visual
content retention, not crystallographic correctness. Before closed-loop
growth control, prospectively test complete videos and route low-confidence
keyframes or unusual apertures to manual review.
