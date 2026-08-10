# Model card: MorphMBE M22

## Intended use

MorphMBE M22 is a research model for estimating surface roughness and producing
a plausible 1 micrometer by 1 micrometer AFM morphology realization from an
in-situ RHEED video acquired during the studied MBE process. Outputs support
retrospective materials-science analysis and hypothesis generation.

It is not validated for autonomous process control, acceptance/rejection of
manufactured material, safety-critical decisions, or extrapolation to unrelated
materials, instruments, geometries, or growth protocols.

## Inputs and outputs

Input: one RHEED video plus a sample identifier used for deterministic seeding
and documented orientation overrides. Automatic localization produces a
16-frame model clip; no AFM measurement is an input.

Outputs:

- predicted areal root-mean-square roughness, Sq, in nanometers;
- predicted functional surface morphology index, FSMI, in nanometers;
- model, key-frame, and combined confidence values;
- a stochastic, physically scaled AFM height map and its unit-Sq shape.

The AFM map represents a conditional morphology distribution. It is not
pixel-registered to a measured scan and should not be interpreted as a
reconstruction of exact island positions.

## Frozen model

- Model ID: `MorphMBE-M20-SpotConnectivitySq +
  M22c-DenseMidGapCompletion-line3-metrology-live-v10`
- Sq head: M20 target-blind spot-connectivity calibration of the rough tail.
- Image generator: `M22c_gap_completion_strong`.
- Training/evaluation cohort: 27 growth groups; growth 6081 excluded.
- Inference boundary: no measured query AFM, AFM retrieval, or nearest-image
  copying.

## Evaluation

Metrics are strict outer leave-one-growth-out estimates. Each fold excludes the
complete held growth from fitting and uses 26 growths.

| Target | n | MAE (nm) | RMSE (nm) | Pearson r | Spearman rho | Interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| Sq | 27 | 0.6853 | 0.8291 | 0.9234 | 0.7863 | 1.0000 |
| FSMI | 27 | 1.1263 | 1.4477 | 0.6748 | 0.5830 | 0.8889 |

The selected M22c intermediate-regime morphology reduced mean dark-pixel
fraction from 0.1514 for M21 to 0.0334, close to the measured 0.0324, on the
five samples with measured Sq from 3.5 to 6.0 nm. See
`results/m22/display_tone_summary.csv` and `docs/METHOD_DEVELOPMENT.md`.

## Limitations and risks

- All reported evidence is retrospective and includes method selection on the
  studied cohort; an untouched prospective cohort is required for external
  validation.
- Twenty-seven growth groups are insufficient to establish broad domain
  generalization or calibrated rare-event behavior.
- Confidence reflects support within the frozen cohort, not a guarantee of
  correctness.
- AFM generation is stochastic and morphology-level; exact island placement is
  not identifiable from the RHEED input.
- Performance may degrade under instrument drift, changed exposure, crop,
  material, substrate, growth recipe, scan size, tip response, or AFM
  preprocessing.
- The deployment bundle depends on pretrained R3D-18 representations. Official
  weights must be available locally or downloaded at first use.

## Reproducibility and provenance

Frozen per-growth predictions, interval columns, confidence values, fold
membership, and morphology audits are under `results/m22/`. Asset hashes are in
`assets/manifest.sha256`. Run `scripts/validate_release.py` before interpreting
results. Raw research data are not distributed; see `data/README.md`.
