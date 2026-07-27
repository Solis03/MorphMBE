# Supplementary architecture prompt

Using the same colors, typography, and visual language as the main figure,
create a separate publication-grade supplementary architecture plate titled:

**“Exact tensor and parameter architecture of the frozen single-frame
RHEED-to-Rq model”**

Read `ALGORITHM_SPECIFICATION.md` and `MODEL_PARAMETER_AUDIT.json` first.
Produce a wide vector diagram with three horizontal lanes: preprocessing,
DINOv2 ViT-S/14, and five-member regression. This plate prioritizes exact
dimensions and parameter counts over the AFM retrieval story.

Use
`01_rheed_inputs/model_ready/N6390_model_ready_patch14_grid.png`
as the real input illustration.

Show this exact tensor ledger:

`RGB RHEED [1376,1100,3]`
→ manual ROI
→ luminance/preserve aspect/zero pad `[224,224]`
→ duplicate channels and ImageNet normalize `[1,3,224,224]`
→ patch Conv2d 14/14 `[1,256,384]`
→ prepend CLS `[1,257,384]`
→ Transformer block ×12 `[1,257,384]`
→ final LayerNorm/CLS `[1,384]`
→ temporal aggregate `[1,1536]`
→ five independently standardized inputs `[5,1536]`
→ five scalar Rq outputs `[5]`
→ median `[1]`.

Expand one Transformer block with two residual sub-blocks:

1. `LN(384, eps 1e−6) → QKV Linear 384→1152 →
   6-head attention, 64 dimensions/head → output Linear 384→384 →
   LayerScale(384, init 1.0) → residual`.
2. `LN(384, eps 1e−6) → Linear 384→1536 → GELU →
   Linear 1536→384 → LayerScale(384, init 1.0) → residual`.

State that dropout = 0, stochastic depth = 0, register tokens = 0, and the head
is Identity.

Add a parameter ledger table:

- patch embedding: 226,176;
- CLS token: 384;
- positional embedding: 526,080;
- mask token: 384, present in the backbone but unused during inference;
- one Transformer block: 1,775,232;
- 12 blocks: 21,302,784;
- final LayerNorm: 768;
- exact DINO total: **22,056,576 frozen parameters**;
- Ridge per member: 1,537;
- five Ridges: **7,685 fitted parameters**;
- five StandardScalers: **15,360 fitted mean/scale statistics**.

Make the positional-embedding distinction explicit:

“Pretrained parameter `[1,1370,384]` from the 37 × 37 base grid is interpolated
to the runtime 16 × 16 patch grid; runtime token count is 257.”

Show the temporal aggregation equation:

`f = concat(mean_t h_t, std_t h_t, h_T−h_1, slope_t h_t)`.

Then state:

`T=1 ⇒ f=[CLS384,0₃₈₄,0₃₈₄,0₃₈₄]`.

Draw the five Ridge members with exact names and T4/T6 targets from
`MODEL_PARAMETER_AUDIT.json`. End with a median diamond and the N6390 example
`Rq_pred = 2.250672 nm`.

Design style: bright white background, extremely clean thin-line vector
graphics, indigo transformer stack, amber regression branches, precise
monospaced tensor labels, restrained shadows, no decorative neural-network
clip art. Return PNG plus editable SVG/PDF if supported.
