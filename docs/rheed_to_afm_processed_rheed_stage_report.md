# 使用处理后 RHEED 数据后的 RHEED-to-AFM 当前阶段实验报告

## 1. 报告目的与一句话结论

本文档是当前项目阶段的正式总结，目标是把下面三件事讲清楚：

1. 这个项目到底在解决什么科学与工程问题；
2. 为什么此前已经做了多轮数据清洗、pairing 修复和 latent 实验；
3. 在**已经接入处理后 RHEED 数据**之后，当前阶段的真实结论是什么。

一句话结论是：

> 即使把 RHEED 输入从原始视频抽帧切换为经过 ROI、稳定化、归一化和时序采样后的 `processed RHEED`，当前 `1um one-to-one + plane-corrected AFM` 基准上的 `RHEED -> AFM latent` 结果仍然没有优于旧 raw baseline，更没有优于 `mean-latent` dummy baseline。

因此，当前阶段最重要的判断不是“预处理还不够彻底”，而是：

- 仅仅把输入视频清洗得更干净，并不能自动带来跨模态预测成功；
- 当前主要瓶颈依旧更像是：
  - `AFM latent target` 本身还不够稳定、还不够 morphology-preserving；
  - 当前 RHEED 表征方式仍然太弱，尤其缺少真正贴合该物理场景的时序表征学习。

---

## 2. 项目背景与问题本质

### 2.1 项目要回答的核心问题

本项目的核心科学问题是：

> 能否从原位 `in-situ RHEED` 生长视频中提取与表面 morphology 有关的表征，并进一步预测或检索出对应的 `ex-situ AFM` 表面形貌？

这是一个典型的**跨模态表征学习 / 检索 / 预测**问题，而不是普通的单模态图像分类任务。

两种模态的本质不同：

- `RHEED` 记录的是生长过程中的电子衍射动态；
- `AFM` 记录的是生长结束后的表面几何形貌。

它们都与表面结构有关，但不是同一个物理观测，因此问题难点不在“图像是否清晰”，而在：

- 输入模态与输出模态不一致；
- 两者之间的映射未必简单、一一对应、线性可学；
- 小样本下很难依赖大模型自己“学出来”。

### 2.2 为什么采用 small-data MVP 路线

当前 paired 数据并不大，真正干净、可信的一对一子集更小：

| 子集 | pairs | groups | 用途 |
| --- | ---: | ---: | --- |
| `1um` | 36 | 36 | 当前主 benchmark |
| `0.5um` | 15 | 15 | exploratory |
| `5um` | 6 | 6 | smoke only |
| `all_size_representative` | 40 | 40 | 多尺度代表性集合 |

在这样的规模下，直接训练大模型、扩散模型、GAN 或复杂视频模型，通常会遇到：

- 参数规模远大于样本规模；
- 极高的过拟合风险；
- 实验失败后很难判断到底是数据问题、pairing 问题、latent 问题，还是模型本身过强或过弱。

所以本项目采用了一个更稳妥的 MVP 路线：

1. 先做 frozen-feature / descriptor baseline，回答“RHEED 里是否有信号”；
2. 再做 `RHEED -> AFM latent`，回答“能否在低维形貌空间里做跨模态预测或检索”；
3. 先把路线排错，再决定是否需要更复杂的表征学习方案。

---

## 3. 历史实验阶段回顾

### 3.1 阶段一：RHEED-to-AFM descriptor baseline

最早的实验不是直接预测 AFM 图像，而是让模型先预测 AFM handcrafted descriptors。  
这条路线的作用主要是诊断：

- 如果 descriptor 都完全预测不了，说明当前 RHEED 表征几乎没有可用信号；
- 如果 descriptor 有部分信号，说明跨模态相关性可能存在，但还不能证明真正的 morphology prediction 成功。

这一阶段的结论是：

- descriptor regression 只能算诊断性 baseline；
- 它能帮助判断“有没有信号”，但不能替代真正的 AFM morphology prediction。

### 3.2 阶段二：one-to-one manifest 与 plane-corrected rerun

随后项目发现了两个核心数据工程问题：

1. 原始 pairing 存在 `one-to-many` 歧义：
   - 一个 RHEED 视频可能对应多个 AFM；
   - 这会直接污染监督信号，让模型学成“平均值”。
2. 旧 manifest 中混入了未 plane-correct 的 AFM：
   - 监督目标本身不一致；
   - 会让实验结论不可靠。

因此后续做了两件关键修复：

- 构建干净的 one-to-one manifests；
- 在 AFM 侧统一优先使用 `plane_corrected_afm`。

修复后的可信结果写入：

- [one_to_one_plane_corrected_rerun](../reports/one_to_one_plane_corrected_rerun)

这一轮最关键的 `1um` 结果是：

- AFM autoencoder best val loss: `0.479466`
- Reconstruction warning triggered: `yes`
- Selected latent model: `knn`
- Learned latent MSE / cosine: `0.451245` / `0.686244`
- Mean-latent baseline MSE / cosine: `0.326974` / `0.657868`

也就是说，即便监督集已经比早期版本干净很多，`RHEED -> AFM latent` 依旧**没有优于 mean-latent dummy baseline**。

### 3.3 阶段三：为什么还要接入处理后的 RHEED 数据

在上述结果不理想之后，一个自然的怀疑是：

> 也许问题不只在 pairing 或 AFM latent，还在于 RHEED 输入本身太“原始”，包含了过多无关噪声。

旧 raw RHEED pipeline 的特点是：

- 从 canonical raw video 中解码整段视频；
- 只均匀抽样 `8` 帧；
- 用 ImageNet 预训练的 ResNet50 对每帧做 embedding；
- 对 sample 级特征只做 `mean/std` 聚合。

这套流程有几个明显局限：

- 视频时间信息利用很弱，只看 8 帧；
- RHEED 原始视频里有背景、漂移、亮度变化、非衍射区域；
- 聚合方式过于简单，几乎没显式建模时序变化。

因此，接入处理后 RHEED 数据的出发点是：

> 在保持下游 latent benchmark 口径不变的前提下，仅替换 RHEED 输入表示，观察更干净、更稳定、时间采样更充分的输入，是否能提升跨模态结果。

---

## 4. 当前阶段试验的出发点与假设

### 4.1 当前试验想验证什么

本轮试验并不是重新定义整个任务，而是在一个严格受控的框架下验证下面这个假设：

> 如果把 RHEED 输入换成经过 ROI 截取、稳定化、归一化、有效区域掩膜和更密集时间采样后的 processed 表示，那么 `RHEED -> AFM latent` 的效果应该优于旧 raw baseline。

### 4.2 本轮试验中哪些量保持不变

为了让比较尽可能公平，本轮试验刻意固定了以下内容：

- manifest 固定为：`data/manifests/manifest_1um_one_to_one.csv`
- benchmark 固定为：`1um`
- 监督目标固定为：旧 rerun 中已经训练好的 AFM autoencoder latent
- AFM latent 文件固定为：
  - `reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/afm_latents.npy`
  - `reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/afm_latent_index.csv`
- latent 评估脚本固定为：`scripts/rheed_to_afm_latent_mvp.py`
- holdout 逻辑固定为：`GroupShuffleSplit(random_state=42)`
- candidate regressors 固定为：`ridge / knn / mlp`
- dummy baselines 固定为：
  - `train_mean_latent`
  - `random_train_latent`

因此，这一轮比较的核心含义非常明确：

> 在监督集、AFM latent target、split 和下游回归器都不变的情况下，仅改变 RHEED 输入表示，看看结果会不会更好。

---

## 5. 处理后 RHEED 数据接口与本轮使用方式

### 5.1 数据根目录

本轮正式训练使用的是：

- `data/raw_RHEED_selected_test_512`

根据 [DATA_INTERFACE.md](../data/DATA_INTERFACE.md)：

- 该根目录包含 `62` 个成功处理的样本目录；
- 名字中的 `_512` 表示**最多采样 512 帧**，不是 `512 x 512` 分辨率；
- 每个样本主入口文件是：
  - `tensors/model_input.npz`

### 5.2 `model_input.npz` 中实际使用了什么

当前集成到训练链路中的字段是：

- `clean_frames`
  - `float32`
  - shape: `[T, H, W]`
  - 范围 `[0, 1]`
- `valid_mask`
  - `bool`
  - shape: `[H, W]`

本轮没有使用：

- `raw_gray_frames`
- `spot_masks`
- `streak_masks`
- `frame_features.csv`
- `video_features.json`

这些文件仍然很有价值，但在本轮里只把它们视为未来可扩展信息，而不是当前主输入。

### 5.3 sample 对齐规则

manifest 中的样本 ID 是纯数字，例如 `6022`、`6063`。  
而 processed 目录名可能是：

- `N6022 - Copy`
- `N6063`

本轮集成时采用的对齐规则是：

1. 从 `manifest_1um_one_to_one.csv` 中读取 `sample_id`
2. 在 `data/raw_RHEED_selected_test_512/` 下查找**唯一包含该数字 sample id** 的目录
3. 读取该目录下的 `tensors/model_input.npz`

实际对齐结果是：

- requested samples: `36`
- mapped samples: `36`
- mapping failed: `0`
- embedded samples: `36`
- skipped samples: `0`

因此，本轮实验没有因为 processed 数据目录不匹配而损失样本。

---

## 6. 当前阶段实验流程

### 6.1 旧 raw baseline 的 RHEED 表征流程

历史 raw baseline 的流程是：

1. 在 `data/pair/<sample>/RHEED/` 中选择 canonical video
2. 解码整段 raw video
3. 均匀采样 `8` 帧
4. 把每帧送入 ImageNet 预训练 ResNet50
5. 对 frame embeddings 做 `mean/std` 聚合
6. 得到 sample-level RHEED embedding

这是此前 `1um plane-corrected rerun` 的对比对象。

### 6.2 新 processed RHEED 的表征流程

本轮新增的 processed 输入流程是：

1. 从 `data/raw_RHEED_selected_test_512/<sample>/tensors/model_input.npz` 读取 `clean_frames`
2. 读取同一文件中的 `valid_mask`
3. 用 `valid_mask` 将无效背景区域置零
4. 从最多 `512` 帧中均匀采样 `64` 帧
5. 将每帧灰度图复制成 3 通道
6. resize 到 ResNet50 所需的 `224 x 224`
7. 用同一个 ImageNet 预训练 ResNet50 提取 per-frame embeddings
8. 对 sample-level embedding 使用更强一些的时序统计聚合：
   - frame embedding `mean`
   - frame embedding `std`
   - 相邻帧 embedding 差分的 `mean`
   - 相邻帧 embedding 差分的 `std`

这一步的含义是：

- 相比 raw baseline，不再只用 8 帧；
- 不再直接输入原始背景和不稳定区域；
- 试图显式编码“时间变化趋势”，而不是只编码静态平均图样。

### 6.3 下游 latent 评估流程

RHEED embedding 生成完之后，后续流程与旧 rerun 保持一致：

1. 读取固定的 AFM latents
2. 按 group 做 train/test split
3. 在训练集上做 grouped model selection
4. 在测试集上评估：
   - learned latent MSE
   - learned latent cosine similarity
   - nearest-neighbor latent distance
   - nearest-neighbor cosine similarity
   - retrieved latent MSE
   - top-k retrieval hit rate
5. 与两个 baseline 对比：
   - mean-latent
   - random-train-latent

### 6.4 本轮主要脚本与产物位置

本轮最关键的代码与报告产物包括：

| 位置 | 作用 |
| --- | --- |
| `src/rheed2morph/rheed/mvp.py` | 新增 `processed_npz` 输入模式、sample 映射和 temporal aggregation |
| `scripts/rheed_to_afm_latent_mvp.py` | 复用 latent benchmark，并支持 processed RHEED 可视化 |
| `scripts/compare_processed_rheed_benchmark.py` | 汇总 raw vs processed 对比 |
| `reports/one_to_one_plane_corrected_rerun_processed/1um/descriptor_data/` | processed RHEED embeddings 与 descriptor 诊断结果 |
| `reports/one_to_one_plane_corrected_rerun_processed/1um/rheed_to_afm_latent/` | processed latent benchmark 结果 |
| `reports/one_to_one_plane_corrected_rerun_processed/1um/processed_vs_raw_comparison.md` | raw vs processed 结论汇总 |

---

## 7. 实验细节与运行配置

### 7.1 processed RHEED embedding 配置

本轮 processed embedding 的关键配置为：

- encoder backend: `torchvision`
- encoder: ImageNet pretrained `ResNet50`
- input mode: `processed_npz`
- processed root: `data/raw_RHEED_selected_test_512`
- frame key: `clean_frames`
- processed sample map: `manifest_sample_id_to_dataset_dir`
- processed max frames: `64`
- image size at encoder input: `224`

需要注意一个实现细节：

- 旧 CLI 中仍保留 raw 模式的 `frame_count` 参数；
- 但在 `processed_npz` 模式下，真正起作用的是 `processed_max_frames=64`；
- 因此看到 summary 里有 `frame_count_requested=8` 不代表 processed 实验只用了 8 帧。

### 7.2 本轮输出目录

本轮独立产物写入：

- `reports/one_to_one_plane_corrected_rerun_processed/1um`

这样做的目的，是避免覆盖旧 raw baseline，并保证对比结果可回溯。

### 7.3 descriptor 诊断性结果

在 processed embeddings 上，descriptor MVP 的 summary 是：

- Embedded samples: `36`
- Joined rows: `36`
- Train rows: `27`
- Test rows: `9`
- Best model: `knn` with `n_neighbors=7`
- Learned model mean MAE / RMSE / R^2:
  - `0.7240 / 0.9644 / -0.5968`
- Nearest-neighbor baseline mean MAE / RMSE / R^2:
  - `1.0945 / 1.6185 / -4.9045`

这个结果说明：

- processed RHEED embeddings 并不是完全没有信息；
- 它们对 handcrafted descriptor 仍然可以产生一定可学习信号；
- 但这并不自动等价于 “可以成功预测 AFM latent morphology”。

---

## 8. 当前阶段核心实验结果

### 8.1 raw vs processed 的主结果对比

当前最核心的比较如下：

| Variant | Source | Model | Learned latent MSE | Learned cosine | Nearest latent distance | Nearest latent cosine | Retrieved latent MSE | Top-k hit rate | Beats mean latent? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | raw video + 8 frames + mean/std | `knn` | `0.451245` | `0.686244` | `1.370465` | `0.678092` | `0.503228` | `0.333333` | no |
| processed | processed clean_frames + 64 frames + temporal stats | `knn` | `0.697626` | `0.510644` | `1.876462` | `0.520817` | `0.727294` | `0.000000` | no |

### 8.2 如何读取这张表

这张表里最值得看的是三类量：

1. `learned latent MSE`
   - 越低越好
2. `learned latent cosine similarity`
   - 越高越好
3. `beats mean latent?`
   - 这是最重要的 sanity check
   - 如果连 mean-latent 都打不过，就不能说模型学到了稳定的跨模态映射

从这些指标看：

- processed 的 MSE **更差**
- processed 的 cosine **更差**
- processed 的 nearest-neighbor retrieval 指标也**更差**
- processed 的 top-k hit rate 从 `0.333333` 降到了 `0.0`
- processed 仍然**没有优于 mean-latent baseline**

### 8.3 为什么 `mean-latent` 指标在 raw 与 processed 中完全一样

这是一个重要但容易忽略的细节。

`mean-latent baseline` 使用的是：

- 同一个 AFM latent target
- 同一个 train/test split
- 同一个训练集 latent 平均值

因为本轮只改了 RHEED 输入，没改 AFM latent 和 split，所以：

- raw 与 processed 的 `mean-latent baseline` 完全相同是正常现象；
- 这恰恰说明本轮比较是公平、受控的。

### 8.4 定性产物

本轮同时生成了 raw 与 processed 的可视化结果，主要应检查：

- `nearest_latent_grid.png`
- `generated_afm_grid.png`

对应位置为：

- raw:
  - `reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/`
- processed:
  - `reports/one_to_one_plane_corrected_rerun_processed/1um/rheed_to_afm_latent/`

这些图的作用不是证明“模型成功了”，而是帮助判断：

- 检索出的 AFM prototype 是否在 morphology 上有肉眼可接受的一致性；
- decoded prediction 是否只是更平滑、更平均的 blob；
- processed RHEED 的清洗是否确实带来了更可解释的检索行为。

当前从 summary 和指标看，答案仍然偏负面。

---

## 9. 当前结果说明了什么

### 9.1 结论一：更干净的输入，不等于更强的跨模态预测

本轮试验最直接的启示是：

> 即使对 RHEED 做了 ROI、稳定化、归一化、有效区域掩膜和更密集的时间采样，最终的 `RHEED -> AFM latent` 结果仍然没有自动改善。

也就是说，问题不是简单的“原始视频太脏”。

### 9.2 结论二：RHEED preprocessing 本身并未解决 representation mismatch

虽然 processed 数据让输入更整洁，但当前 encoder 仍然是：

- ImageNet 预训练的 ResNet50

这意味着：

- encoder 并不是为衍射图样设计的；
- 它擅长自然图像纹理与物体边缘，不一定擅长物理衍射 pattern；
- 即使输入变干净，encoder 也未必会抽取出真正与 morphology 强相关的特征。

### 9.3 结论三：AFM latent target 仍然是主要瓶颈之一

本轮 processed 实验复用了旧 rerun 的 AFM autoencoder，而旧 rerun 自己已经给出 warning：

> AFM latent space may not yet be morphology-preserving

这意味着：

- 即使 RHEED 侧表征真的变好；
- 下游监督目标本身若仍然模糊、平滑、部分塌缩；
- 模型也很难学到稳定而有意义的跨模态映射。

换句话说：

> 输入改善并不能弥补目标空间本身不稳定的问题。

### 9.4 结论四：更多帧与简单 temporal stats 仍然不够

本轮已经把时间维从 raw baseline 的 `8` 帧提升到 processed 的 `64` 帧，并加入了：

- frame mean/std
- delta mean/std

但结果依然更差，说明：

- 简单增加帧数并不足以解决问题；
- 简单的一阶统计和差分统计，还不足以捕获真正关键的 growth dynamics；
- 如果未来要继续深挖 RHEED 时序，可能需要更针对性的时序 encoder 或 self-supervised pretraining。

### 9.5 结论五：当前项目已进入“验证瓶颈来源”的阶段

这轮实验非常重要，因为它把一个常见但模糊的猜想变成了一个明确结论：

> “只要把 RHEED 预处理得更好，结果就会明显提升” 这个猜想，在当前 benchmark 上并没有成立。

因此，当前项目不再处于“先把视频清洗好再说”的阶段，而是进入了：

- 明确识别瓶颈；
- 区分“输入问题”、“目标问题”和“表征学习问题”；
- 决定下一步资源该投入到哪里。

---

## 10. 当前阶段的总判断

截至现在，项目已经完成了几轮关键排错：

1. 证明 descriptor baseline 只能做诊断，不足以代表 morphology prediction 成功；
2. 证明 one-to-many pairing 和 plane-correction 不一致会污染监督信号；
3. 构建了更干净的 one-to-one、plane-corrected benchmark；
4. 验证了 processed RHEED 输入并没有自动提升 `RHEED -> AFM latent` 结果。

因此，当前阶段最合理、最诚实的结论是：

> 本项目已经完成了“工程链路打通”和“关键假设排错”，但尚未完成“跨模态 morphology prediction 的有效性验证”。

更具体地说：

- 现在不能声称“processed RHEED 已经带来了更好的 AFM 预测”；
- 现在也不能声称“当前 latent benchmark 已经证明了 RHEED 中存在稳定可用的 morphology signal”；
- 但可以明确声称：
  - 数据清洗方向已经被实证检验过；
  - 当前失败不是因为最基础的数据接口没打通；
  - 下一阶段应该把重点放在更高层次的表示学习与目标构造上。

---

## 11. 建议的下一步分析与实验方向

当前最值得优先推进的方向，我建议按下面顺序排序。

### 11.1 第一优先级：提升 AFM latent target 质量

这是当前最重要的工作。

理由是：

- 当前 autoencoder 已经触发 quality warning；
- 如果 latent 不够 morphology-preserving，后续任何 RHEED 侧建模都会被天花板限制。

可做方向包括：

- 改进 AFM autoencoder 结构与损失；
- 更强地约束 morphology 保真度，而不是只追求像素重建；
- 做更系统的 latent 可解释性与 collapse 检查。

### 11.2 第二优先级：做 processed 输入的更细致消融

虽然当前 processed 总结果更差，但还不能说明“所有 processed 信息都无用”。

后续可做的消融包括：

- `clean_frames` vs `raw_gray_frames`
- 是否使用 `valid_mask`
- `64` 帧 vs 更短或更长的时间采样
- 只做空间清洗，不做强归一化
- 直接使用 `frame_features.csv` / `video_features.json` 做轻量表格 baseline

### 11.3 第三优先级：构建更适合 RHEED 的表征学习方法

如果 AFM latent 问题先得到缓解，那么下一步应考虑：

- 专门针对衍射 pattern 的自监督 encoder；
- 更明确的时间序列建模；
- 不再依赖单纯的 ImageNet ResNet50 迁移。

### 11.4 第四优先级：继续把“负结果”变成结构化结论

当前这个阶段虽然没有得到更好的 prediction performance，但它不是“无结果”。

相反，它已经回答了一个非常重要的问题：

> 处理后 RHEED 输入本身，并不是当前项目成功与否的决定性瓶颈。

这类结论对于后续资源分配和研究方向判断是非常有价值的。

---

## 12. 最终结论

如果用一句更完整的话总结当前项目阶段，那么最准确的表述是：

> 在完成 one-to-one pairing 清洗、plane-corrected AFM 统一化、以及 processed RHEED 输入接入之后，当前 `1um` benchmark 上的 `RHEED -> AFM latent` 结果依然没有优于简单的 mean-latent dummy baseline；processed RHEED 相比 raw baseline 甚至进一步退化。这说明当前项目的核心瓶颈更可能位于 AFM latent target 质量与跨模态表征学习能力，而不是基础视频预处理本身。

这就是当前阶段最重要、也最值得保留下来的研究结论。
