# Literature update: sparse persistent peaks for N6342

Date: 2026-08-04

## Failure being addressed

The M16b smooth-regime renderer generates the correct order of areal RMS
height, but its fixed-density local-maximum layer and final `tanh` compression
turn many moderate extrema into similarly bright, rounded plateaus.  For
N6342, the measured AFM instead contains fine continuous texture with a
smaller population of visually dominant protrusions.  This is a topology and
height-distribution mismatch, not simply a blur or Sq problem.

N6342 was examined during method development.  Its new leave-one-growth-out
result must therefore be described as retrospective development evidence,
not as an untouched prospective test.

## Literature basis

1. Na, Yoo and Ki, *Prediction of surface morphology and reflection spectrum
   of laser-induced periodic surface structures using deep learning*,
   Journal of Manufacturing Processes 84 (2022),
   [doi:10.1016/j.jmapro.2022.11.004](https://doi.org/10.1016/j.jmapro.2022.11.004).
   The supplied full text was previously read page by page.  Its transferable
   principle is to condition a stochastic image generator on process/morphology
   variables and validate Fourier-domain and downstream physical quantities,
   rather than relying on pixel loss alone.

2. Portilla and Simoncelli, *A Parametric Texture Model Based on Joint
   Statistics of Complex Wavelet Coefficients*, IJCV 40 (2000),
   [author project page](https://www.cns.nyu.edu/~lcv/texture/).  Their model
   demonstrates that spectral power alone does not determine perceptually
   convincing texture; cross-scale and higher-order statistics matter.  This
   supports retaining fine multiscale AFM residuals while separately
   controlling persistent peaks.

3. Shaham, Dekel and Michaeli, *SinGAN: Learning a Generative Model from a
   Single Natural Image*, ICCV 2019,
   [CVF paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.html).
   Its multiscale internal-patch result supports exploiting repeated texture
   observations within AFM scans, while keeping the independent growth—not a
   crop—as the cross-validation unit.

4. Clough et al., *A Topological Loss Function for Deep-Learning Based Image
   Segmentation Using Persistent Homology*, TPAMI 2020,
   [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC9721526/).
   It shows that connected components across thresholds can be constrained
   explicitly.  The present experiment adopts the auditable small-data
   analogue: counts of persistent maxima and bright excursion components
   across several prominence/height thresholds.

5. Azadmand et al., *Reliable synthesis of self-running Ga droplets on GaAs
   (001) in MBE using RHEED patterns*, Scientific Reports 2015,
   [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4404429/), directly
   correlates streaky RHEED with corrugation and spotty/chevron RHEED with
   nanoscale droplets.  Related GaSb-family work reports bright streaky RHEED
   as evidence of good two-dimensional growth
   ([Scientific Reports 2023](https://www.nature.com/articles/s41598-023-29169-9)).
   These observations support a streak-sensitive smooth-surface prior, while
   not claiming that RHEED uniquely determines a pixel-registered AFM map.

6. The physics-informed SimuScan AFM synthesis work combines representative
   geometry with AFM-specific imaging artifacts and flattening distortions
   ([Nature Communications 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13125452/)).
   It reinforces separating morphology primitives from fine measurement
   appearance rather than asking one small end-to-end network to learn both.

## Implemented hypotheses

| ID | Hypothesis | Persistent peak control | Fine texture |
| --- | --- | --- | --- |
| M16b | preserved dense-microisland baseline | fixed dense maxima | low-pass, tanh-clipped |
| M17a | fixed sparse peaks | exactly 12 candidate maxima | 10% spectral residual |
| M17b | topology-conditioned sparse peaks | RHEED-predicted q82 component count | 10% spectral residual |
| M17c | topology-conditioned sparse peaks plus stronger fine texture | lower count scale and wider spacing | 16% spectral residual |
| M17d | multiscale texture alone | no explicit added peaks | 18% spectral residual |
| M17e | broad sparse peaks | fewer, wider Gaussian peaks | 10% spectral residual |
| M17f | broad sparse peaks plus fine texture | moderate count and width | 14% spectral residual |
| M17g | moderate broad topology | intermediate count/width | 11% spectral residual |
| M17h | hierarchical islands | sparse high peaks plus broad shoulder population | 12% spectral residual |
| M17i | soft hierarchical islands | lower-weight peaks and shoulders | 14% spectral residual |

All candidates use generated spectral/geometry fields.  No held AFM patch,
nearest-neighbour AFM, sample-ID rule, or retrieval operation is used at
inference.  The non-smooth terrace branch remains the frozen M12a/M16b branch,
which limits adverse changes to rough samples.

## Selection rule

The selected candidate must improve the N6342 broad-bright-region mismatch
and full-cohort peak-signature error while preserving:

- strict 27-growth LOO Sq/FSMI predictions;
- full-cohort PSD and island-topology metrics;
- AFM texture gate pass rate;
- generated diversity and non-identity audits;
- rough-sample morphology, because the new branch is gated to the low-Sq
  regime.

This is a Pareto/non-inferiority decision, not selection by one attractive
image.

## Final decision

M17b was selected after the N6342 development fold and then rerun under the
complete 27-growth outer LOO protocol.  It corrects the M16b excess-bright-
area/tail-shape failure while retaining the best N6342 PSD and island balance.
The broad and hierarchical variants make some peaks look larger, but overshoot
N6342 height kurtosis and do not improve the full-cohort audit.  M10 remains a
strong population-average texture baseline; M17b is the best response to the
specific deployed-M16b N6342 failure mode, not a claim that it dominates every
metric for every growth.
