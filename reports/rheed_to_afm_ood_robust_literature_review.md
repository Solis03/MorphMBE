# OOD-aware small-sample RHEED-to-AFM modeling: literature review

Date: 2026-07-28

## Scope

This review asks whether a small conditional scientific-imaging model can
identify training examples that are rare, unreliable, or weakly supported,
reduce their influence without erasing legitimate rare physics, and report
lower confidence for poorly supported test cases. It also revisits the
mandatory laser-induced periodic surface structure (LIPSS) paper as a
reference for morphology-aware conditional generation.

## Mandatory reference

Na et al., *Prediction of surface morphology and reflection spectrum of
laser-induced periodic surface structures using deep learning*, Journal of
Manufacturing Processes 84 (2022) 1274–1283,
DOI 10.1016/j.jmapro.2022.11.004, was read from the supplied local PDF:
`/Users/ziyi/Desktop/1-s2.0-S1526612522007757-main.pdf`.

The paper uses a conditional BigGAN-deep model with:

- projection discrimination and conditional batch normalization;
- self-attention, spectral normalization and hinge losses;
- differentiable augmentation for limited data;
- frequency-domain validation and early stopping;
- quantitative morphology validation through ripple period, ripple width and
  two-dimensional Fourier spectra.

The especially transferable ideas are to condition a genuine generator,
preserve spatial-frequency content, and evaluate physical morphology
descriptors rather than image loss alone. Its data regime is nevertheless
different from this project: 32 process conditions lie on a deliberately
sampled, comparatively continuous process grid, and each training condition
is expanded to 200 image crops. Twenty-three heterogeneous growth groups do
not provide an equally dense condition manifold. Directly replacing M12a
with a large BigGAN would therefore have high collapse and memorization risk.
The present experiment instead retains the already validated stochastic M12a
island/terrace generator and strengthens its RHEED-conditioned morphology
head.

## Robust learning and sample weighting

Curriculum and self-paced learning preferentially fit examples with small
current loss, while MentorNet and Co-teaching use learned or peer-model
curricula to reduce the effect of corrupt labels:

- [Self-Paced Learning for Latent Variable Models (NeurIPS 2010)](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)
- [MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels (ICML 2018)](https://proceedings.mlr.press/v80/jiang18c)
- [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels (NeurIPS 2018)](https://papers.nips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)
- [Iterative Learning With Open-Set Noisy Labels (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Iterative_Learning_With_CVPR_2018_paper.html)

These methods motivate the residual self-paced ablation, but their implicit
assumption—large loss often means bad supervision—is unsafe here. A rare
high-Rq growth can be scientifically valid and difficult precisely because
the data set undersamples that regime. Downweighting it because its AFM target
is hard to predict can worsen range compression. This is observed in the
M14c experiment.

Covariate-density and importance-weighting methods instead ask whether an
input lies in a supported portion of RHEED feature space:

- [Robust Covariate Shift Regression (AISTATS 2016)](https://proceedings.mlr.press/v51/chen16d.html)
- [Robust Importance Weighting for Covariate Shift (AISTATS 2020)](https://proceedings.mlr.press/v108/li20b.html)
- [MAPLE: Model Agnostic Supervised Local Explanations (ICML 2022)](https://proceedings.mlr.press/v162/zhou22d.html)

This motivates a target-blind weight from leave-one-growth RHEED k-nearest
neighbor support. It is safer than residual weighting because it never looks
at the AFM target when deciding whether an observation is unusual. It still
cannot detect conditional shift: two similar RHEED representations may map
to different AFM outcomes because a missing growth variable is not encoded.

Distributionally robust optimization (DRO) and hidden-group robustness
provide an important counterpoint. Instead of discarding minority examples,
they optimize worst-case or latent-group performance:

- [Distributionally Robust Optimization: A Review (JMLR 2018)](https://jmlr.org/beta/papers/v19/17-295.html)
- [Data Geometry in Distributionally Robust Optimization (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/da535999561b932f56efdd559498282e-Abstract-Conference.html)
- [No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)

With only 23 groups, a full learned group-DRO system is not identifiable.
The practical lesson is nevertheless decisive: hard exclusion must be an
explicit sensitivity analysis, never the default scientific conclusion, and
coverage of rare roughness regimes must be reported.

## OOD detection, uncertainty and selective prediction

Feature-space Mahalanobis distance and ensembles are established practical
signals of epistemic uncertainty:

- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)
- [Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles (NeurIPS 2017)](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)

Selective regression formalizes the option to abstain or flag a prediction,
but selective systems can magnify subgroup disparities if “difficult” is
confused with “unimportant”:

- [Selective Regression Under Fairness Constraints (ICML 2022)](https://proceedings.mlr.press/v162/shah22a.html)

This project therefore reports the complete 23-growth result and a
risk–coverage curve. Low-confidence filtering is an operational mode, not a
way to hide failures.

Conformal prediction supplies finite-sample marginal coverage guarantees
under exchangeability, and importance-weighted conformal prediction extends
the idea to known covariate shift:

- [Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06019)
- [Conformal prediction for small-data molecular property prediction under covariate shift](https://arxiv.org/abs/2310.12033)

The M14 intervals use fold-local residual quantiles and achieved 20/23
(86.96%) empirical coverage for both Rq and FSMI at nominal 90%. With only 23
outer points and retrospective method development, the reported confidence
must remain a relative ranking index, not a calibrated probability of
correctness.

## Design adopted here

The literature supports a guarded hybrid:

1. use target-blind robust RHEED support diagnostics for exclusion audits and
   sample weights;
2. compare them against target-dependent residual self-paced weighting;
3. add a causal temporal R3D view so confidence and prediction do not depend
   only on hand-engineered key-frame features;
4. blend curated physics features and temporal features;
5. estimate risk from fold-local temporal density and predicted
   high-amplitude extrapolation;
6. report all held-growth predictions, empirical interval coverage,
   confidence–error correlation and risk–coverage behavior;
7. preserve valid rare samples unless independent physical evidence—not model
   error—shows that they are invalid measurements.

The experiments validate this direction: density weighting and multiview
fusion improve the full-cohort targets, while hard exclusion and
target-residual weighting are weaker. The remaining high-Rq failures indicate
missing condition coverage or unobserved growth variables rather than a
justification for silently deleting those growths.
