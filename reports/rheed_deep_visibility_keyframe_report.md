# Deep spot-visibility RHEED keyframe selection

Date: 2026-07-28
Branch: `codex/rheed-keyframe-deep-visibility-20260728`

## Outcome

The V4 phase tracker was preserved and extended with an image-content-aware
candidate ranker. The selected V5 model is:

> calibrated-safe ROI + dual physical vertex proposals + multi-scale spot/haze
> descriptors + frozen DINOv2-S/14 + Ridge/pairwise/ExtraTrees rank fusion +
> 25th-percentile visibility gate.

The user-identified 6063 failure changes from V4 frame 461 (pattern NCC
0.279) to V5 frame 188 (NCC 0.784), while the human frame is 186. The V5 frame
shows the same compact vertical spot family instead of the diffuse blue shadow
selected by V4.

## Data and leakage control

- The input inventory contains 27 manually annotated videos.
- Samples 6023 and 6087 overlap `removelist.txt` and are excluded before
  feature extraction, model fitting, confidence calibration and evaluation.
- V5 contains 25 videos, 642 physical candidate rows and 598 unique candidate
  or manual images.
- Every reported supervised metric is strict leave-one-video-out. Each fold
  fits scaling, PCA and all ranking heads using candidates from 24 videos;
  the 25th video's labels never enter that fold.
- Frozen DINOv2 inference on the held images is analogous to applying a fixed
  ImageNet-scale feature transform. It does not fit to held labels.
- Raw RHEED data are read only.

This is retrospective method-development evidence. Because V5 was selected
after comparing alternatives on this cohort, a new prospective video cohort
is still required before an unbiased publication claim or closed-loop use.

## Scientific and AI rationale

The physical vertex generator remains necessary: a visually sharp frame at
the wrong rotation phase is not the requested keyframe. V4 nevertheless
allowed smooth high-intensity haze to dominate because its scalar clarity
feature did not explicitly require a family of compact diffraction maxima.

V5 adds 27 image descriptors inside the automatic ROI:

- multi-scale Difference-of-Gaussian peak count, mass and signal-to-noise;
- connected spot count, area and compactness;
- spot-energy concentration;
- vertical span, column alignment and principal-axis verticality;
- Laplacian and gradient sharpness;
- high-to-low-frequency energy;
- low-frequency haze dominance;
- raw shadow, saturation and dynamic-range measures.

A frozen DINOv2-S/14 encoder contributes a 1,152-dimensional vector made from
its CLS token plus patch-token mean and standard deviation. Fold-local PCA
prevents a 22-million-parameter foundation model from being directly
fine-tuned on only 25 videos.

The final score combines:

| Component | Weight |
|---|---:|
| DINOv2 + physical/spot-feature Ridge rank | 0.38 |
| within-video pairwise ranker | 0.27 |
| interpretable visual ExtraTrees rank | 0.25 |
| explicit spot-visibility rank | 0.10 |

Candidates below the 25th within-video visibility percentile are rejected
when alternatives exist. This is deliberately a weak gate: a 40th-percentile
gate rejected the correct held frame for 6082.

The design is informed by:

- [DINOv2](https://arxiv.org/abs/2304.07193), for frozen self-supervised
  transferable visual features;
- [MUSIQ](https://openaccess.thecvf.com/content/ICCV2021/papers/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.pdf),
  for multi-scale image-quality representations;
- [RankIQA](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_RankIQA_Learning_From_ICCV_2017_paper.pdf),
  for learning relative quality from small datasets;
- [NIST's deep-learning RHEED phase-mapping work](https://www.nist.gov/publications/application-machine-learning-reflection-high-energy-electron-diffraction-images),
  which supports explicit learned spot/streak representations for automated
  RHEED interpretation.

## Strict held-video results

| Method | Median NCC | Mean NCC | Median SSIM | Median gradient NCC | Median frame difference | Diffuse-shadow proxy rate |
|---|---:|---:|---:|---:|---:|---:|
| V4 Ridge | 0.714 | 0.670 | 0.482 | 0.362 | 46 | 16% |
| V5 visual Ridge | 0.694 | 0.639 | 0.476 | 0.369 | 150 | 4% |
| V5 DINOv2-S Ridge | 0.720 | 0.691 | 0.504 | 0.402 | 70 | 12% |
| V5 DINOv2-S pairwise | 0.670 | 0.662 | 0.504 | 0.385 | 12 | 4% |
| V5 DINOv2-S hybrid, 40% gate | 0.784 | 0.710 | 0.504 | 0.502 | 8 | 8% |
| **V5 DINOv2-S tree hybrid, 25% gate** | **0.820** | **0.730** | **0.559** | **0.583** | **3** | **4%** |
| V6 DINOv2-Base tree hybrid, 25% gate | 0.701 | 0.658 | 0.476 | 0.342 | 148 | 4% |

The diffuse-shadow proxy is an evaluation-only flag: selected
Difference-of-Gaussian top-eight peak mass below 60% of the corresponding
human frame. It is not used as a hidden training label. V5 leaves one such
case, 6022, versus four for V4.

The 86-million-parameter DINOv2-Base ablation is a negative result. It is
larger but substantially worse than the 22-million-parameter small model,
consistent with irrelevant natural-image variation overwhelming a
very-small-domain ranking task.

## Confidence and failures

The selected ensemble's raw top-two margin is an empirical extrapolation
indicator: unusually dominant winners were less reliable, not more reliable.
The negative margin is significantly related to realized error across the 25
strict-LOO selections (Spearman rho = -0.459, p = 0.021). It is monotonically
calibrated to expected composite similarity for deployment. That numerical
calibration is a development fit to the 25 LOO outcomes: a second
leave-one-prediction-out calibration audit is not significant (rho = -0.130,
p = 0.537), so its absolute value requires prospective recalibration. The
rank signal is supported; the score must not be described as a probability of
correctness.

V5's remaining lowest-similarity examples are retained:

- 6056: NCC 0.183;
- 6080: NCC 0.287;
- 6062: NCC 0.508;
- 6048: NCC 0.491.

The first three primarily show a clear diffraction family at the wrong
rotation phase rather than the original shadow/haze failure. Sample 6022 is
the remaining diffuse proxy failure. These cases should trigger manual review
until prospective data improve phase coverage.

## Visual evidence

- [V5 benchmark](rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/deep_visibility_benchmark.pdf)
- [V5 confidence audit](rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/confidence_validation.pdf)
- [V5 retained failure panel](rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/lowest_similarity_cases.pdf)
- [V5 final all-sample atlas](rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/v5_dinov2_tree_hybrid_gate25/)
- [V5 per-model atlases](rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/)
- [Rejected V6-Base artifacts](rheed_auto_roi_keyframe/20260728_dinov2_base_spot_visibility_v6/)

Every model atlas uses fixed sample order and shows:

1. the full human frame with automatic ROI in cyan and human ROI in yellow;
2. the human keyframe cropped to the automatic ROI;
3. the strict-held machine keyframe with NCC and spot-mass ratio.

## Reproducibility

Reproduce V5:

```bash
PYTHONPATH=src HF_HOME="$PWD/tmp/huggingface" \
  .venv/bin/python \
  analysis/rheed_auto_roi_keyframe/train_deep_visibility_ranker.py \
  --config configs/rheed_auto_roi_keyframe_v5.json \
  --rebuild-features \
  --device mps
```

Run inference:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/select_rheed_roi_keyframe.py \
  "path/to/video.MOV" \
  --output-dir "outputs/automatic_selection"
```

The first run downloads the pinned
`facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`
weights into `tmp/huggingface`; no paid service is used. The fitted ensemble
is:

`outputs/rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/dinov2_spot_visibility_ranker.joblib`

On the M1 Pro, extraction of all 598 unique V5 evaluation images took
35.5 seconds (16.9 images/s). The complete CLI processed the original
813-frame 6063 MOV in 30.1 seconds and selected frame 189, versus human frame
186. Local MPS was sufficient; CUDA is not required for this experiment.
