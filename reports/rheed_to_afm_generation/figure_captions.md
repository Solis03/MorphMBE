# Figure captions

All panels use fixed growth-group ordering. AFM axes are in micrometres and
physical roughness is in nanometres. Generated examples are deterministic
fixed-seed draws or descriptor-space medoids from eight uncurated draws; none
were selected by visual inspection.

**Metric table.** Group-level medians and 95% bootstrap intervals over the
five held-out growth groups. Diversity is the median generated pairwise L1
distance divided by the corresponding real-AFM pairwise distance.

**Figure 2.** Held-out comparison of the unconditional train mean,
nearest-RHEED retrieval, and the frozen RHEED-conditioned CVAE. The model
improves Rq and PSD medians but is worse on descriptor MAE and SSIM.

**Figure 3.** For every held-out growth group: manually selected RHEED temporal
window, generated CVAE medoid, measured AFM medoid, and nearest-RHEED
retrieval. Generated physical scale uses RHEED-predicted Rq; the measured test
Rq is not supplied to generation.

**Figure 4.** Measured AFM medoid and four uncurated stochastic CVAE draws for
each held-out condition. The panel reveals nonzero diversity and recurrent
decoder boundary artifacts.

**Figure 5.** Failure cases ranked by a predefined morphology composite, with
measured AFM, generated medoid, and physical-height residual. The systematic
loss of sharp islands and the lower-edge artifact are visible.

**Figure 6.** Training reconstruction loss and validation-only prior-generation
metrics. The dashed line marks epoch 65, frozen before test evaluation.

**Figure 7.** Growth-group measured versus RHEED-predicted Rq, high-frequency
PSD fraction, autocorrelation length, and skewness. The one-to-one dashed line
shows strong regression to the training mean on the held-out cohort.

**Figure 8.** Validation-only temporal-input ablation. Centered eight-frame
DINOv2 features outperform key-frame-only DINOv2 and selected-16-frame R3D-18
features.

**Figure 9.** Condition-permutation negative control. Lower descriptor error
is better; the correct condition wins for only one of five held-out groups,
which invalidates a strong conditioning claim.
