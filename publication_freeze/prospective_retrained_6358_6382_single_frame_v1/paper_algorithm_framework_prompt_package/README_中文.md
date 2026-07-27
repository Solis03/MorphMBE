# 论文算法框架绘图素材包

这个文件夹可以整体压缩后上传给 ChatGPT 的 AI 科学绘图工具。

建议分三次生成图，以保证文字和结构准确：

1. 先提交 `PROMPT_01_MAIN_FRAMEWORK_EN.md`，生成论文主图；
2. 再提交 `PROMPT_02_DETAILED_ARCHITECTURE_EN.md`，生成精确到层和参数量的
   supplementary architecture；
3. 最后提交 `PROMPT_03_VALIDATION_DESIGN_EN.md`，生成数据划分和无泄漏验证图。

英文 prompt 是为了提高科学绘图模型对复杂版式和公式的遵循率。事实依据在
`ALGORITHM_SPECIFICATION.md`，模型参数审计在
`MODEL_PARAMETER_AUDIT.json`，图片索引在 `ASSET_INDEX.md`。

## 上传时应保留的目录

- `01_rheed_inputs/`：五个 extra 样本的 raw、ROI 和 224 × 224 模型输入；
- `02_afm_ground_truth_selected/`：带 height bar 和 Rq 的选定实验 AFM；
- `03_afm_retrieval_outputs/`：三个 prospective 输出；
- `04_afm_candidate_bank/`：N6390 的 A3 前五候选和选中 source；
- `05_reference_only/`：仅供绘图工具理解既有数据关系，不应直接拼进新图；
- `support/`：素材再生成和检查脚本。

## 最重要的科学限制

- 不能画成端到端 RHEED-to-AFM 生成网络。
- DINOv2 是 22,056,576 参数、完全冻结的 ViT-S/14。
- 回归输入是 1536 维 temporal aggregate，不是直接的 384 维 CLS。
- 五个 Ridge 每个 1,537 个参数，总共 7,685 个拟合参数。
- A3 是固定的 descriptor ranking 与物理幅值缩放，神经网络参数为 0。
- Ground truth AFM 只能出现在 post-hoc evaluation 区域，不能画成训练外的
  prospective 输入。
- 所有 AFM 必须保留 height bar、单位 nm 和 Rq。
- N6022、N6099 在整个新实验中排除；N6324 忽略。

建议让绘图工具先返回低分辨率草图，人工核对全部文字、数字、箭头方向和
ground-truth leakage boundary 后，再要求输出最终 PNG、SVG 和 PDF。
