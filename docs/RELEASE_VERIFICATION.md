# M22 release verification

The minimized release was compared with the untouched
`MorphMBE_M22_DenseMid_UI_Standalone_20260810` using its N6063 rampdown video,
deployment bundle, DINOv2 cache, and torchvision R3D-18 weights. Both runs used
sample ID 6063 and event seed `6063 * 1,000,003 + 189 * 97`.

## Exact control-path agreement

The following values were identical:

- 813 decoded frames and 20 retained events;
- all event frame indices, scores, qualities, and tracker labels;
- selected frame 189;
- model-input, physics, tracking, and audit ROIs in source pixels;
- selected-16 array: shape `(16, 224, 224)`, maximum absolute difference 0;
- physics selected-16 array: shape `(16, 224, 224)`, maximum difference 0;
- 11 causal eight-frame views: shape `(11, 8, 224, 224)`, maximum difference 0;
- causal view names and RGB key-frame pixels;
- FSMI 3.8718539184 nm;
- model, key-frame, and combined confidence, including combined confidence
  0.4010375719;
- non-retrieval inference flag.

## Floating-point output comparison

| Quantity | Untouched standalone | Minimized release | Absolute difference |
|---|---:|---:|---:|
| Predicted Sq (nm) | 5.0612720509 | 5.0609938375 | 0.0002782134 |
| Generated Sq (nm) | 5.0612668991 | 5.0609884262 | 0.0002784729 |
| AFM height-map RMSE (nm) | — | — | 0.0003499573 |
| AFM height-map maximum difference (nm) | — | — | 0.0013933182 |
| AFM height-map correlation | — | — | 0.9999999991 |
| AFM height-map SSIM | — | — | 0.9999999801 |

The Sq difference is below the variation of repeated identical-backend runs:
three repeated MPS predictions from the same release object spanned
5.060054-5.061296 nm, and repeated untouched-standalone CPU predictions spanned
5.060842-5.061268 nm. Thus the measured old/new difference is inside observed
torchvision convolution floating-point nondeterminism and is not attributable
to changed model logic.

## Additional checks

- Deployment bundle SHA-256 is unchanged:
  `f15a89f5e82833046624665a64b533e7b6f06267b2f43960f4721ac36b1e63d5`.
- The minimized and standalone environments resolve the same NumPy 2.4.5,
  SciPy 1.17.1, scikit-learn 1.8.0, PyTorch 2.12.0, and torchvision 0.27.0.
- Strict outer-LOO tables contain 27 unique held growths, 26 fit growths per
  fold, no held/fitted overlap, and no held Sq target used for training.
- The complete public test suite and release-integrity validator pass.
- Raw RHEED and AFM records and the desktop standalone were read only.

Timing fields are intentionally excluded because they measure wall-clock load,
selection, and inference duration rather than model output.
