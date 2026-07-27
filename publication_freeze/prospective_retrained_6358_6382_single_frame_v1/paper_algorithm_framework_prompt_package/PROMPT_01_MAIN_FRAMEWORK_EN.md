# Copy everything below into ChatGPT's scientific drawing tool

Act as a senior scientific illustrator working for a high-impact materials
science journal. Create a beautiful, publication-grade, vector-style methods
schematic titled:

**“Single-frame RHEED-to-Rq prediction and representative AFM retrieval”**

Before drawing, read the attached `ALGORITHM_SPECIFICATION.md`,
`MODEL_PARAMETER_AUDIT.json`, and `ASSET_INDEX.md`. These files are
authoritative. Do not invent layers, dimensions, training samples, model
parameters, AFM processing operations, or numerical results.

## Deliverable

Create one wide, two-column-journal main figure in a 16:9 landscape format,
preferably 6000 × 3375 px or vector-equivalent. Keep all text editable if the
tool supports SVG/PDF. The figure should look like a polished fusion of a Nature
Methods architecture diagram and an Advanced Materials graphical abstract:
clean white or very pale warm-gray background, generous whitespace, precise
alignment, thin elegant arrows, subtle depth, and restrained visual polish.
Avoid cartoonish icons, dark backgrounds, heavy gradients, neon effects,
photorealistic laboratory decorations, and generic AI imagery.

Use a left-to-right flow with five clearly labeled zones:

**A. Cohort and physical inputs → B. Frozen RHEED encoder → C. Quantitative Rq
ensemble → D. A3 representative AFM retrieval → E. Post-hoc evaluation**

Use the following coordinated palette:

- RHEED/preprocessing: deep emerald and teal (`#0B6E4F`, `#1F9D8A`);
- frozen DINO encoder: indigo and blue (`#4257C9`, `#6D7FE5`);
- quantitative regression: amber and warm gold (`#E69F00`, `#F4C95D`);
- AFM retrieval: violet/teal accents that complement the supplied viridis maps
  (`#6A4C93`, `#2A9D8F`);
- evaluation only: neutral graphite and coral (`#3A3A3A`, `#E76F51`);
- exclusion or forbidden leakage: muted red dashed stroke (`#C94C4C`).

The actual RHEED and AFM image pixels must come from the attached files. Place
them as image panels with subtle borders. Do not repaint, regenerate, denoise,
stylize, recolor, crop away scientific content, or fabricate microscopy
textures. Preserve every AFM height bar and Rq label.

## A. Cohort and physical inputs

At the upper left, show a compact cohort card:

- “23 original historical samples”
- cross out only N6022 and N6099
- “21 retained historical”
- add N6358 and N6382
- final training cohort: **n = 23**
- prospective test: **N6342, N6389, N6390 (n = 3)**
- N6324 shown in small gray text as “ignored”

The arithmetic should be visually obvious:

`23 historical − {N6022,N6099} + {N6358,N6382} = 23 training samples`.

Do not imply that the three prospective test AFMs entered training.

Below the cohort card, use N6390 as the worked example. Place these files in
sequence:

1. `01_rheed_inputs/raw_keyframes/N6390_frame_000137.png`
   labeled “selected raw RHEED keyframe, 1376 × 1100 RGB”;
2. `01_rheed_inputs/roi_keyframes/N6390_frame_000137_roi.png`
   labeled “recorded manual ROI”;
3. `01_rheed_inputs/model_ready/N6390_model_ready_224x224_luminance.png`
   labeled “luminance → aspect-preserving resize → zero pad, 224 × 224”;
4. `01_rheed_inputs/model_ready/N6390_model_ready_patch14_grid.png`
   as a small optional overlay/inset labeled “16 × 16 patches, patch size 14”.

Between ROI and model-ready input, use a small preprocessing ribbon:

`uint8 / 255 → duplicate luminance to 3 channels → ImageNet normalization`

and print the exact normalization below in tiny but legible type:

`mean = [0.485, 0.456, 0.406]`

`std = [0.229, 0.224, 0.225]`

## B. Frozen RHEED encoder

Draw a refined DINOv2 ViT-S/14 module—not a generic black box. Show:

`[1,3,224,224]`
→ `Conv2d 14 × 14 / stride 14, 3 → 384`
→ `256 patch tokens`
→ `+ 1 CLS token`
→ `[1,257,384]`
→ a stacked Transformer block icon marked **×12**
→ final LayerNorm
→ `CLS [1,384]`.

Inside one expanded Transformer block, show:

`LN → 6-head self-attention (head dim 64; QKV 384→1152; output 384)
→ residual + LayerScale`

then

`LN → MLP 384→1536→384, GELU → residual + LayerScale`.

Place a prominent but elegant badge:

**“DINOv2 ViT-S/14 · 12 blocks · 22,056,576 parameters · 100% frozen”**

Also state:

`patch 14 | hidden 384 | 6 heads | MLP ratio 4 | no register tokens`

Do not draw a trainable classification head; the head is Identity.

After the 384-D CLS output, include a distinct feature-assembly block:

`concat[temporal mean, std, first−last delta, linear slope]`

with tensor transition:

`[1,384] → [1,1536]`.

For this single-frame experiment add a small mathematical note:

`T = 1: [CLS384, 0₃₈₄, 0₃₈₄, 0₃₈₄]`.

This 1536-D block is mandatory. Never connect the 384-D CLS directly to the
Ridge regressors.

## C. Quantitative Rq ensemble

Draw five thin parallel branches, each receiving the same 1536-D feature.
Every branch is:

`StandardScaler (1536 means + 1536 scales)`
→ `Ridge, α = 1.0`
→ one scalar Rq value in nm.

Label the five members exactly:

1. `model_01_trial_0004 · target T4`
2. `model_02_trial_0012 · target T4`
3. `model_03_trial_0006 · target T6`
4. `model_04_trial_0014 · target T6`
5. `model_05_trial_0028 · target T4`

Place an exact parameter callout:

**“1,536 coefficients + 1 intercept = 1,537 fitted Ridge parameters/member;
7,685 across five members.”**

In a smaller note:

“Scaler state: 3,072 fitted statistics/member; 15,360 total.”

Merge the five outputs into a gold median operator labeled:

`median in Rq space → predicted Rq q50`.

For the N6390 worked example, print:

**`Rq_pred = 2.250672 nm`**

If q10/q90 are shown, label them “descriptive five-member spread” and explicitly
state “not a calibrated prediction interval.”

Add a compact global accounting badge:

`22,056,576 frozen encoder parameters + 7,685 fitted Ridge parameters
= 22,064,261 model parameter values`

and below:

`No end-to-end gradient fine-tuning`.

## D. A3 representative AFM retrieval

Split a curved arrow from the predicted q50 Rq into a lower visual-retrieval
lane. Show a bank card labeled:

**“quality-passing AFM bank · 23 sample groups · 118 maps”**

Embed
`04_afm_candidate_bank/N6390_A3_top5_candidate_bank_montage.png`
or, if the montage becomes too small, use the five individual rank images from
`04_afm_candidate_bank/`.

Place an 11-coordinate descriptor ribbon beside the bank:

`Rq, Ra, robust height range, PSD low/mid/high fractions, PSD slope,
correlation length, anisotropy, height skewness, height kurtosis`.

Make the conditioning rule explicit:

`c_Rq = predicted Rq; other 10 coordinates = bank medians`.

Typeset the exact A3 score in a clean equation box:

`d_i = sqrt[ Σ_j ((x_ij − c_j) / max(s_j,10⁻⁶))² ]
       + 0.05 |Rq_i − Rq_pred|`

Then show:

`minimum score`
→ selected rank-1 source
→ use
`04_afm_candidate_bank/rank1_6028_N6028_500_nm_006_selected_heightbar_Rq.png`.

Label it:

`selected source: sample 6028 / N6028_500_nm_006`

`source Rq = 2.319988 nm`.

Next show a simple physical operation:

`mean center → normalize to unit Rq → multiply by Rq_pred`.

Typeset:

`Z_unit = (Z_source − mean(Z_source)) / (Rq(Z_source)+10⁻⁶)`

`Z_pred = Rq_pred × Z_unit`.

Finish this lane with
`03_afm_retrieval_outputs/N6390_retrieved_heightbar_rq.png`,
labeled:

**“Representative retrieved AFM · Rq = 2.25 nm · 1 × 1 µm”**

Add a conspicuous but tasteful note:

**“Representative historical morphology retrieval—not pixel reconstruction,
not a generative decoder.”**

The A3 block has **0 learned neural parameters**.

## E. Post-hoc evaluation and leakage boundary

At the far right or lower right, place
`02_afm_ground_truth_selected/N6390_GT1_top_left_quarter_heightbar_Rq.png`.

Label it:

`N6390 experimental AFM, ground truth 1`

`2 µm × 2 µm raw scan → upper-left quarter → 1 µm × 1 µm`

`robust second-order y² correction`

`displayed GT Rq = 2.397455 nm; sample-level T4 = 2.297728 nm`.

Connect prediction and ground truth only with a thin dashed graphite/coral line
labeled **“post-hoc evaluation only.”** Place a vertical dashed leakage boundary
before the ground-truth panel. No arrow may travel from the ground-truth AFM
back into DINO, Ridge, or A3 selection.

Include a small evaluation inset:

- Prospective test: train 23 → predict 3;
- Quantitative LOO: 26 folds, train 25 per fold;
- AFM held-one-out: target group removed, 25 AFM source groups per fold.

## Required visual hierarchy and quality checks

- The primary eye path must be RHEED → frozen encoder → 1536-D feature →
  five Ridge members → median Rq → A3 retrieval → representative AFM.
- Use numbered panel labels A–E and short subtitles.
- Use clean arrowheads and avoid crossing arrows.
- Keep equations sharp and legible at journal-column scale.
- Use real attached scientific images as rectangular inset panels; do not turn
  them into icons.
- Preserve AFM viridis colors, height bars, “nm,” Rq labels, and 1 × 1 µm
  meaning.
- Correctly distinguish “frozen parameters,” “fitted Ridge parameters,” and
  “fitted scaler statistics.”
- Do not depict a CNN, U-Net, diffusion model, autoencoder, decoder, or
  end-to-end RHEED-to-AFM neural generator.
- Do not show N6022 or N6099 inside any retained cohort or bank.
- Do not show N6342/N6389/N6390 AFMs as model inputs.
- Add a tiny footer: “Algorithms fixed; only data division changed in this
  exclusion sensitivity experiment.”

Before returning the final figure, perform a text and arrow audit against
`ALGORITHM_SPECIFICATION.md`. Return both a high-resolution PNG preview and
SVG/PDF if supported.
