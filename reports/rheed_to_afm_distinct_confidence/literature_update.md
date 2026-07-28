# Targeted literature update: condition sensitivity and confidence

Review date: 2026-07-27

## Why the earlier generator collapsed

The earlier M2b spectral generator was a genuine random-field generator, but
the small-data RHEED regressor and the second condition-to-spectrum regression
both shrank toward the training mean. Different RHEED observations could
therefore reach almost the same AFM condition. This is a known conditional
generation failure mode rather than evidence that the three inputs are
physically equivalent.

Relevant literature and the design consequence:

- [Continuous Conditional GAN](https://arxiv.org/abs/2011.07466) replaces
  brittle discrete labels with hard/soft vicinal losses for continuous
  conditions. It supports treating Rq, correlation length, PSD and height
  moments as a continuous morphology condition.
- [ContraGAN](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f490c742cd8318b8ee6dca10af2a163f-Abstract.html)
  adds data-to-data and data-to-condition contrastive objectives. This
  motivates an explicit condition-sensitivity audit instead of assuming that
  a concatenated condition is used.
- [Collapse by Conditioning](https://arxiv.org/abs/2201.06578) shows that
  strong conditional injection can itself cause mode collapse under limited
  data. This is why the present work does not simply increase a GAN condition
  loss or guidance scale.
- [ControlNet++](https://arxiv.org/abs/2404.07987) uses consistency feedback:
  the generated result is decoded back to the condition. The analogous local
  design is to measure generated AFM descriptors and require them to respond
  to the RHEED-predicted descriptor vector.
- [D2C for few-shot conditional generation](https://proceedings.neurips.cc/paper/2021/hash/682e0e796084e163c5ca053dd8573b0c-Abstract.html)
  supports separating a learned data representation from a lightweight
  conditional model when labeled conditions are scarce.

## Mandatory paper

Na, Yoo and Ki, *Prediction of surface morphology and reflection spectrum of
laser-induced periodic surface structures using deep learning*, Journal of
Materials Processing Technology (2022),
[DOI 10.1016/j.jmapro.2022.11.004](https://doi.org/10.1016/j.jmapro.2022.11.004),
was reread from the supplied PDF and visually inspected.

The useful transferable ideas are:

- genuine stochastic condition-to-morphology generation rather than exemplar
  retrieval;
- conditioning at multiple generator/discriminator stages;
- small-data augmentation and spectral normalization;
- Fourier-domain morphology validation;
- evaluating whether generated morphology supports a downstream physical
  property, not merely whether it looks realistic.

The local data have only 15 independent training growth groups, so the present
method adopts the paper's frequency/statistics validation and conditional
generation principles but not a large end-to-end BigGAN. Repeated AFM scans
and crops remain observations within a growth group, never extra independent
conditions.

## Scientific random-field design

[Surface PSD as a quantitative roughness characterization](https://arxiv.org/abs/1607.03040)
supports using the full spatial-frequency distribution rather than Rq alone.
AFM examples combining PSD and correlation analysis
([example study](https://arxiv.org/abs/2305.19795)) support the use of
correlation length and anisotropy. The resulting M4 generator is a stochastic
multiscale Matérn field whose Rq, PSD slope, coarse/fine balance, correlation
scale, anisotropy, skewness and kurtosis are explicit functions of the
RHEED-predicted condition.

M5 mixes that condition-sensitive large-scale field with the learned M2b
spectral random-field prior. Both inputs are newly generated fields. No
measured AFM exemplar, nearest neighbour, patch bank or retrieval source is
used at inference.

## Confidence and small-data credibility

The uncertainty design follows group-level conformal prediction:

- [Conformalized unconditional quantile regression](https://proceedings.mlr.press/v206/alaa23a.html)
  motivates distribution-free calibration around flexible point predictors.
- [Mondrian conformal regression](https://proceedings.mlr.press/v128/bostrom20a.html)
  motivates respecting structured subpopulations rather than mixing
  exchangeability units carelessly.
- [Conformal regression with a reject option](https://proceedings.mlr.press/v230/johansson24a.html)
  supports using uncertainty to identify predictions that should not be
  trusted automatically.

Every calibration residual and every query prediction is produced by a model
that excludes the corresponding growth group. The reported 90% component
interval is therefore a growth-group CV+/Jackknife+ interval. The displayed
confidence index is explicitly not a probability. It penalizes intervals
spanning several training standard deviations and only uses a small rank
adjustment; because all current intervals are wide, no current prediction
receives a high absolute score.

## Claim boundary

The literature supports the architecture, but it cannot create information
absent from 15 independent training growths. Current evidence can support:

- genuine non-retrieval conditional generation;
- improved morphology-condition separation;
- cross-fitted morphology/Rq/PSD comparisons;
- empirically audited but wide uncertainty intervals;
- a learning curve showing that additional independent growth groups reduce
  error.

It cannot yet support exact local AFM reconstruction, uniformly reliable
rough-regime prediction, or a new final held-out-test claim.
