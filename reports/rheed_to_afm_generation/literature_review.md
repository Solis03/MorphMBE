# Literature review: generative RHEED-to-AFM morphology modeling

Review date: 2026-07-27
Scope: conditional generation, small-data learning, scientific/microscopy
imaging, microstructure statistics, RHEED/MBE physics, AFM morphology, temporal
conditioning, and generative-model evaluation.

## Design conclusion

The dataset has only 23 independent growth groups and 116 one-micrometre AFM
scans. The AFM scans within a growth group are repeated spatial observations,
not independent process conditions. A large pixel-space diffusion model or GAN
trained end-to-end from RHEED is therefore poorly matched to the effective
sample size, even though such models are attractive on much larger datasets.

The most defensible local experiment is staged:

1. learn a compact AFM generative representation;
2. predict physically interpretable AFM descriptors from a temporal RHEED
   representation using growth-group-level regularization;
3. condition a stochastic AFM decoder on those predicted descriptors;
4. select on group-held-out validation data using morphology fidelity,
   diversity, conditioning controls, and identity audits together;
5. evaluate a frozen model once on an untouched growth-group test split.

This follows the computational economy of latent diffusion and VQ models
without assuming that this dataset can support a large latent denoiser. It also
follows the mandatory laser-induced periodic surface structure paper by
validating morphology in Fourier/statistical space rather than relying on
pixel similarity alone.

## Mandatory reference

Na et al., *Prediction of surface morphology and reflection spectrum of
laser-induced periodic surface structures using deep learning*, Journal of
Materials Processing Technology (2022),
[DOI 10.1016/j.jmapro.2022.11.004](https://doi.org/10.1016/j.jmapro.2022.11.004).
The supplied file
`/Users/ziyi/Desktop/1-s2.0-S1526612522007757-main.pdf` was read in full and
visually inspected page by page.

Useful elements:

- a conditional BigGAN-deep generator maps process conditions to surface
  morphology;
- projection conditioning and conditional batch normalization make the
  experimental condition enter multiple generator/discriminator stages;
- DiffAugment addresses limited data;
- image realism is not the only endpoint: the paper compares Fourier-domain
  period and ripple statistics and then evaluates a downstream reflection
  spectrum;
- the model generates a morphology rather than retrieving an observed surface.

Important caveat for this project: that work has 32 process conditions
(26/3/3 train/validation/test) and creates 200 crops per condition. Crops add
texture observations but not independent process conditions. Here, growth
groups must remain the leakage boundary and uncertainty must be summarized at
the group level.

## Generative modeling foundations

| Topic | Evidence and implication |
| --- | --- |
| Conditional image translation | [pix2pix](https://arxiv.org/abs/1611.07004) established conditional adversarial image-to-image translation. It assumes aligned source/target images; RHEED and AFM are different modalities with no pixel registration, so a direct L1 paired translator is not physically well posed here. |
| Variational latent modeling | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) provides a tractable stochastic latent generator. A conditional prior permits multiple AFM realizations for one RHEED condition and is materially different from deterministic regression or retrieval. |
| Conditional VAE | [Learning Structured Output Representation using Deep Conditional Generative Models](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) motivates a learned conditional latent distribution for structured output. This is the closest computational match to the local experiment. |
| Discrete AFM latents | [VQ-VAE](https://arxiv.org/abs/1711.00937) and [VQGAN](https://arxiv.org/abs/2012.09841) show how discrete or perceptually trained latents can preserve texture. Existing repository VQ experiments remain relevant baselines, but the very small number of growth conditions makes codebook use and conditional generalization difficult. |
| Diffusion | [DDPM](https://arxiv.org/abs/2006.11239) is a high-quality stochastic generator, while [latent diffusion](https://arxiv.org/abs/2112.10752) reduces computation by denoising in a pretrained autoencoder latent space and supports flexible conditioning. The repository already contains diffusion experiments; their strict fixed-method morphology score remains worse than retrieval. |
| Small-data GANs | [DiffAugment](https://arxiv.org/abs/2006.10738) and [StyleGAN2-ADA](https://arxiv.org/abs/2006.06676) directly address discriminator memorization under limited data. They are useful if an adversarial refinement stage is revisited, but augmentations must preserve AFM height and PSD statistics. |
| Conditional trade-offs | [Consistency-diversity-realism Pareto fronts](https://arxiv.org/abs/2406.10429) argues that conditional generators must be evaluated on all three axes. This directly motivated the validation gates for morphology, diversity, and a condition-permutation negative control. |

## Scientific and microscopy generation

| Work | Relevance |
| --- | --- |
| [Microstructure reconstruction using diffusion-based generative models](https://arxiv.org/abs/2211.10949) | Evaluates diffusion across several material microstructures with FID, precision/recall, and conventional statistical descriptors. It supports descriptor-space evaluation rather than a single generic image metric. |
| [Microscopy image reconstruction with a physics-informed DDPM](https://arxiv.org/abs/2306.02929) | Embeds the microscopy image-formation model in conditioning/reverse dynamics and explicitly studies hallucination and artifact reduction. It motivates treating decoder artifacts as scientific failures even when aggregate metrics improve. |
| [Stable diffusion for inverse design of microstructures](https://arxiv.org/abs/2409.19133) | Demonstrates property-conditioned microstructure generation, but uses 576,000 synthetic images—orders of magnitude more data than are available here. |
| [3D multiphase heterogeneous microstructure generation using conditional latent diffusion](https://pubs.rsc.org/en-us/content/articlelanding/2025/dd/d5dd00159e) | Conditions on statistical/topological targets such as volume fraction and tortuosity, supporting the use of Rq, PSD fractions/slope, correlation length, anisotropy, skewness, and kurtosis as the bridge between modalities. |
| [Deep-learning electron-microscopy image synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6119234/) | Shows the practical appeal of adversarial synthesis in microscopy, while reinforcing the need to separate plausible texture from faithful scientific reconstruction. |
| [Two-point statistics for microstructure reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10904791/) | Supports spectral/spatial-correlation evaluation. Radially averaged PSD is useful but cannot capture all topology. |
| [Higher-order correlations in heterogeneous media](https://pmc.ncbi.nlm.nih.gov/articles/PMC9889385/) | Motivates reporting histogram/quantile and morphology descriptors in addition to PSD; matching a two-point statistic alone is not enough. |

## RHEED, MBE, and AFM physics

RHEED uses grazing-incidence high-energy electrons and is especially sensitive
to the topmost surface. Streaks generally indicate flatter, two-dimensional
surfaces; spotty/transmission-like features are associated with three-
dimensional islands or roughness; diffuse scattering and intensity evolution
carry disorder, coverage, and growth-mode information. Specular and
off-specular intensity oscillations encode layer completion and growth
dynamics, so a temporal window is scientifically better motivated than a
single key frame when enough signal is available.

The project therefore retains both pretrained visual embeddings and explicit
summaries of spot, streak, connection, diffuse, and brightness-drift behavior.
The explicit summaries are not claimed to be a full forward scattering model;
they are a low-dimensional inductive bias.

Relevant sources:

- [Machine-learning-assisted analysis of transition-metal-dichalcogenide
  thin-film growth](https://link.springer.com/article/10.1186/s40580-023-00359-5)
  shows that full RHEED sequences contain surface crystallinity, morphology,
  growth-rate, strain, disorder, and reconstruction information. It also shows
  that low-variance PCA components can contain physically important
  oscillations, warning against using variance alone as relevance.
- [Application of machine learning to RHEED images for automated structural
  phase mapping](https://www.nist.gov/publications/application-machine-learning-reflection-high-energy-electron-diffraction-images)
  supports learned RHEED representations but addresses phase mapping rather
  than AFM generation.
- [Multi-modal machine learning analysis of GaSe MBE growth
  conditions](https://arxiv.org/abs/2606.13900) is unusually close to this
  project because it combines RHEED and AFM modalities. It reinforces the need
  for growth-level multimodal validation.
- [RHEED pattern classification for chalcogenide films and
  nanostructures](https://arxiv.org/abs/2602.18243) demonstrates the value of
  convolutional RHEED features, but classification success does not establish
  morphology generation.
- Sullivan et al., *RHEED intensity variations during the initial stages of
  epitaxial growth* and related surface-roughness work,
  [DOI 10.1016/j.jcrysgro.2016.12.082](https://doi.org/10.1016/j.jcrysgro.2016.12.082),
  motivate the roughness/oscillation link.
- Braun, *Applied RHEED: Reflection High-Energy Electron Diffraction During
  Crystal Growth*, especially the treatment of intensity oscillations,
  [DOI 10.1017/CBO9780511735097.020](https://doi.org/10.1017/CBO9780511735097.020),
  provides the physical basis for temporal monitoring.

## Evaluation implications

Generic FID is not reliable with five test growth groups and domain-mismatched
ImageNet features. The evaluation therefore uses group-level:

- physical Rq absolute error in nanometres;
- unit-Rq L1 and SSIM, with the caution that AFM and RHEED are not
  pixel-registered and exact AFM realization is stochastic;
- normalized radial PSD log distance;
- autocorrelation-length relative error;
- height-quantile and physical-height Wasserstein errors;
- nine-descriptor standardized MAE;
- generated/real within-condition pairwise-distance ratio;
- nearest-training L1, maximum training SSIM, and exact-pixel identity;
- a correct-condition versus cyclicly permuted-condition negative control;
- fixed-seed uncurated ensembles and group-level bootstrap intervals.

[Generative precision and recall](https://arxiv.org/abs/1806.00035) and its
[revisited formulation](https://arxiv.org/abs/1905.05441) motivate separating
quality from coverage, but feature-space PR is also unstable at this sample
size. The diversity ratio and identity audit are transparent small-sample
proxies, not replacements for a larger-cohort precision/recall study.

## What the literature says to try next

The local result shows that decoder capacity and boundary behavior, not raw
compute, are the immediate bottlenecks. Before scaling to a large diffusion
model:

1. learn a translation-equivariant AFM prior with circular/reflection padding
   and explicit border-artifact penalties;
2. validate AFM-only sample fidelity and coverage before attaching RHEED;
3. use a richer condition-matching loss that estimates descriptors
   differentiably from generated ensembles;
4. acquire more independent growth conditions, especially the rough and
   step/terrace regimes;
5. perform nested group cross-validation with multiple training seeds;
6. only then test a latent diffusion or diffusion-refinement model, possibly
   with classifier-free guidance on the RHEED/descriptor condition.

The broad literature supports a true generator, but it does not remove the
identifiability limit imposed by 23 independent growth groups.
