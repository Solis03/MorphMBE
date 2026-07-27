# Exact algorithm specification for the drawing

This document is the factual source of truth for the figure. The drawing tool
must not simplify the method into an end-to-end image generator.

## 1. Data division

- Begin with 23 historical labeled samples.
- Remove N6022 and N6099 globally, leaving 21 historical samples.
- Add N6358 and N6382 to training: 23 training rows total.
- Predict N6342, N6389, and N6390 prospectively. None of these three labels or
  AFMs enters the prospective fit or the 23-group retrieval bank.
- N6324 is ignored.
- The post-hoc quantitative LOO evaluation contains 26 retained labeled
  samples; each fold trains on the other 25.
- The held-one-out AFM experiment also uses 26 folds and removes the target's
  entire AFM group, leaving 25 source groups in that fold.

## 2. RHEED preprocessing and tensor shapes

1. Select one physical RHEED keyframe manually.
2. Apply the recorded manual ROI.
3. Convert RGB to luminance.
4. Bilinearly resize while preserving aspect ratio and zero-pad to
   224 × 224.
5. Convert uint8 to [0,1], duplicate the luminance image into three channels,
   and normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard
   deviation `[0.229, 0.224, 0.225]`.
6. DINO input shape: `[1,3,224,224]`.

Use the N6390 files in `01_rheed_inputs/` as the worked example.

## 3. Frozen DINOv2 ViT-S/14 encoder

Weight identifier: `facebookresearch/dinov2:dinov2_vits14`.

- Patch projection: `Conv2d(3,384,kernel=14,stride=14,bias=True)`.
- 224/14 = 16 patches along each axis; 256 patch tokens total.
- Patch tensor: `[1,256,384]`.
- Prepend one CLS token and add interpolated positional encoding.
- Runtime token tensor: `[1,257,384]`.
- No register tokens.
- 12 Transformer blocks; hidden dimension 384; 6 attention heads; head
  dimension 64.
- Each block is:
  `LayerNorm -> multi-head self-attention (QKV 384→1152, output 384) ->
  residual + LayerScale -> LayerNorm -> MLP 384→1536→384 with GELU ->
  residual + LayerScale`.
- LayerNorm epsilon is 1e-6; LayerScale starts at 1.0; dropout and stochastic
  depth are both zero.
- Final LayerNorm; classification head is Identity; use the 384-D CLS output.
- Exact parameter count: 22,056,576, all frozen in this experiment.

The pretrained positional parameter has shape `[1,1370,384]` because the
upstream model uses a 37 × 37 base grid at 518 px. At runtime it is interpolated
to the 16 × 16 grid. Draw the runtime sequence as 257 tokens, not 1370.

## 4. Actual 1536-D feature used by regression

The 384-D frame embeddings are transformed by the frozen Phase2A temporal
aggregate:

`feature = concat(mean, standard deviation, first-last delta, linear slope)`.

This yields 4 × 384 = 1536 dimensions. Because this experiment uses one frame,
the exact feature is `[CLS_384, 0_384, 0_384, 0_384]`. This block must remain in
the diagram; do not connect the 384-D CLS directly to Ridge.

## 5. Five-member quantitative ensemble

Each member independently fits:

`StandardScaler().fit(X_train)` followed by
`Ridge(alpha=1.0, fit_intercept=True)`.

- Input: 1536 values.
- Scalar output: Rq in nm.
- Ridge parameters per member: 1536 coefficients + 1 intercept = 1,537.
- Five-member Ridge total: 7,685 fitted parameters.
- StandardScaler stores 1,536 means and 1,536 scales per member: 15,360 fitted
  statistics across five members.
- Members 1, 2, and 5 use target T4; members 3 and 4 use target T6.
- The final point prediction is the median of the five scalar outputs.
- q10/q90 in this prospective package are descriptive member quantiles, not a
  calibrated prediction interval.

Parameter accounting:

- Frozen DINO: 22,056,576.
- Fitted Ridge coefficients/intercepts: 7,685.
- Encoder plus Ridge values: 22,064,261.
- Including fitted scaler means/scales: 22,079,621 numeric state values.
- No end-to-end gradient fine-tuning is performed.

## 6. A3 representative AFM retrieval

The prospective retrieval bank contains 23 sample groups and 118 AFM maps.
Only quality-passing maps are ranked.

The 11 descriptor coordinates are, in order:

1. Rq
2. Ra
3. robust height range
4. PSD low-frequency fraction
5. PSD mid-frequency fraction
6. PSD high-frequency fraction
7. PSD slope
8. correlation length
9. anisotropy
10. height skewness
11. height kurtosis

Construct the condition vector by setting the Rq coordinate to the predicted
ensemble-median Rq and setting the other ten coordinates to the corresponding
bank medians. For candidate map `i`,

`score_i = sqrt(sum_j(((x_ij-c_j)/max(std_j,1e-6))^2))
           + 0.05*abs(Rq_i-Rq_pred)`.

Sort by score, sample ID, and AFM file ID. Select rank 1. This procedure has no
learned neural parameters.

The selected physical source map is mean-centered and projected to unit Rq:

`Z_unit = (Z_source - mean(Z_source)) / (Rq(Z_source) + 1e-6)`.

The predicted physical map is:

`Z_pred = Rq_pred × Z_unit`.

This is representative historical morphology retrieval plus amplitude scaling,
not pixel-by-pixel reconstruction and not a generative decoder.

For the N6390 worked example:

- predicted Rq = 2.250672 nm;
- selected source = sample 6028, `N6028_500_nm_006`;
- source Rq = 2.319988 nm;
- post-hoc sample T4 ground truth = 2.297728 nm;
- displayed ground-truth scan 1 Rq = 2.397455 nm.

Use `04_afm_candidate_bank/N6390_A3_top5_candidate_bank_montage.png` for the
candidate-bank visual and
`03_afm_retrieval_outputs/N6390_retrieved_heightbar_rq.png` for the result.

## 7. AFM ground-truth processing and leakage boundary

The five extra samples were measured as 2 µm × 2 µm, 512 × 512 AFMs. Crop the
upper-left 256 × 256 quarter first, corresponding to 1 µm × 1 µm, then apply
the unchanged robust second-order `y²` background correction. Compute Rq from
the corrected physical height array.

Ground-truth AFM images belong in a visually separated post-hoc evaluation
panel. Use dashed arrows labeled “evaluation only.” Never draw ground-truth AFM
as an input to RHEED encoding, Ridge prediction, or prospective A3 selection.

All displayed AFMs—ground truth, bank candidates, selected source, and
retrieved output—must retain their physical height bar in nm and their Rq label.
