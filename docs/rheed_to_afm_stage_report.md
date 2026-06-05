# RHEED-to-AFM Morphology Prediction 项目阶段性实验报告

## 1. 报告目的与一句话结论

本报告面向第一次接触该项目的教授或合作者，系统回顾目前已经完成的三组实验：

1. `reports/rheed_descriptor_mvp`
2. `reports/one_to_one`
3. `reports/one_to_one_plane_corrected_rerun`

项目的核心科学问题是：

> 能否从原位 `in-situ RHEED` 视频中提取与表面 morphology 相关的表征，并进一步预测或检索出与 `ex-situ AFM` 对应的表面形貌？

截至当前阶段，最重要的结论是：

- `descriptor regression` 已经证明只能作为诊断性 baseline，不能作为最终技术路线。
- `one-to-many` 配对歧义和 `plane-corrected` 数据使用不一致，确实会污染监督信号，因此数据工程清洗是必要的。
- 即使修正到更干净的 `one-to-one`、`plane-corrected` 数据集，当前 `RHEED -> AFM latent` 结果仍然**没有稳定优于简单 dummy baseline**。
- 当前真正的主要瓶颈，不是回归器本身，而是：
  - `AFM latent target` 还不够稳定、足够 morphology-preserving；
  - `RHEED representation` 仍然较弱，尤其缺少针对该物理场景的自监督/时序表征学习。

因此，当前阶段最合理的判断是：

> 本项目已经完成了“路线排错”和“工程打通”，但尚未完成“跨模态预测有效性验证”。下一阶段的主任务应当是优先提升 AFM latent 空间质量，再评估是否需要构建更强的 RHEED self-supervised encoder。

---

## 2. 项目背景与第一性原理

### 2.1 问题本质

RHEED 视频记录的是生长过程中的电子衍射动态信息；AFM 图像记录的是生长结束后的表面形貌。二者都与表面结构有关，但不是同一种观测：

- `RHEED` 更接近“生长过程中的衍射信号”
- `AFM` 更接近“最终表面几何形貌”

所以这个问题本质上不是普通图像分类，而是一个**跨模态 representation learning / retrieval / prediction** 问题：

- 输入模态：RHEED 视频
- 输出模态：AFM 形貌图

在理想情况下，模型应该学到一类“形貌相关的共同表征”，从而让 RHEED 端的信息足以预测 AFM 端的 morphology。

### 2.2 为什么不能一开始就上大模型或 diffusion

当前 paired data 规模只有约 `100` 个 sample 左右，真实 clean `one-to-one` 子集更小：

- `1um`: 36 pairs
- `0.5um`: 15 pairs
- `5um`: 6 pairs
- `all_size_representative`: 40 pairs

在这个数据量下，直接训练大模型、GAN 或 diffusion 的主要问题不是“算力不够”，而是：

- 模型参数量远大于有效样本数
- 过拟合风险极高
- 很难把实验失败归因到“数据问题”还是“模型太大”

因此，本阶段采用的是**small-data MVP 路线**：

1. 先做 frozen-feature baseline
2. 再做 image latent MVP
3. 先回答“有没有信号”，再回答“怎么做更强”

这条路线在方法论上是合理的。

---

## 3. 代码与实验组织结构

本阶段最关键的脚本与模块如下：

| 位置 | 作用 |
| --- | --- |
| `scripts/rheed_to_afm_descriptor_mvp.py` | descriptor baseline 入口 |
| `src/rheed2morph/rheed/mvp.py` | RHEED 视频处理、ResNet50 embedding、descriptor baseline 主逻辑 |
| `scripts/build_one_to_one_manifests.py` | 扫描 AFM 文件、生成 one-to-one manifests |
| `src/rheed2morph/manifests/afm_candidates.py` | 构建 AFM candidate table |
| `src/rheed2morph/manifests/one_to_one.py` | 根据规则筛选 one-to-one pairing |
| `scripts/train_afm_autoencoder_mvp.py` | 训练 AFM autoencoder，得到 latent target |
| `src/rheed2morph/afm/mvp.py` | AFM 数据读取、预处理、AE 结构、可视化与 checkpoint |
| `scripts/rheed_to_afm_latent_mvp.py` | 训练 `RHEED -> AFM latent`，做 retrieval 和 baseline 对比 |

报告中所有方法与结论都基于上述代码回溯，而不是仅基于口头总结。

---

## 4. 数据与数据工程

### 4.1 原始数据组织

项目数据的关键目录包括：

- `data/pair/`
  - 每个 sample 下有 RHEED 视频
- `data/processed_afm/`
  - AFM 处理结果，包含 height map、render 图等
- `data/plane_corrected_afm/`
  - 做过 fitted-plane / plane correction 的 AFM 结果
- `data/afm_descriptor_reconstruction/`
  - 预先计算好的 AFM descriptor 表

### 4.2 关键数据难点：one-to-many pairing

原始数据中，一个 RHEED 视频常常对应多个 AFM 图像。  
这意味着监督学习里会出现：

> 同一个输入 `x`，对应多个输出 `y`

这会直接导致：

- 监督信号模糊
- 回归目标自相矛盾
- 模型容易学到“平均值”
- 定量指标和定性可视化都被污染

这也是为什么早期 descriptor baseline 会出现明显的 `prediction collapse`。

### 4.3 one-to-one manifest 的构建原则

`scripts/build_one_to_one_manifests.py` 和 `src/rheed2morph/manifests/*.py` 做了两层工作。

第一层：扫描全部 AFM 候选文件，写出完整 candidate table：

- 扫描 `processed_afm/` 和 `plane_corrected_afm/`
- 为每个 AFM 文件记录：
  - `sample_id`
  - `group_id`
  - `afm_path`
  - `rheed_path`
  - `scan_size_um`
  - `channel_name`
  - 是否 `plane_corrected`
  - 是否 `rendered image`
  - 是否 `physical height map`

第二层：对每个 sample 选出一个最合适的 AFM，形成 clean one-to-one pairing。

筛选逻辑的核心原则是：

1. `plane_corrected` 优先
2. `physical height map` 优先
3. 非 rendered 图优先
4. `ZSensor` 优先
5. 分辨率更高的候选更优
6. `scan_size_um` 优先从 metadata 读取，文件名只做 fallback

这一步的目标不是追求“所有数据都保留”，而是追求：

> 每个 sample 只保留一个监督目标，尽量减少 target ambiguity。

### 4.4 scan size 归一化

代码对 scan size 做了归一化处理。例如：

- metadata 中若记录为 `500`，会根据单位规则归一化为 `0.5 um`
- 支持从 metadata 或文件名解析 `0.5um`, `500nm`, `1um`, `5um` 等形式

然后构建以下子集：

- `manifest_1um_one_to_one.csv`
- `manifest_0p5um_one_to_one.csv`
- `manifest_5um_one_to_one.csv`
- `manifest_all_size_representative_one_to_one.csv`

当前修正后的真实规模为：

| 子集 | pairs | groups | 说明 |
| --- | ---: | ---: | --- |
| `1um` | 36 | 36 | 当前主 benchmark |
| `0.5um` | 15 | 15 | exploratory |
| `5um` | 6 | 6 | smoke only |
| `all_size_representative` | 40 | 40 | 多尺度代表性集合 |

### 4.5 数据完整性问题与修复

这一阶段发现了两个非常重要的数据完整性问题。

#### 问题 A：`1um` 与 `all_size_representative` 的旧 descriptor 报告高度疑似重复

见：

- [report_integrity_check.md](../reports/one_to_one/report_integrity_check.md)

明确证据包括：

- `joined_dataset.csv` 完全相同
- `metrics_summary.json` 完全相同
- `test_predictions.csv` 完全相同

因此，旧的 `one_to_one` descriptor 结果**不能**被当作 `1um vs all_size_representative` 的严格比较依据。

#### 问题 B：旧 manifest 中混入了未做 plane correction 的 AFM

典型案例是 `sample 6100`：

- 旧 manifest 曾指向 `processed_afm/..._height.npy`
- 但该 sample 实际存在 `plane_corrected_afm/..._plane_corrected.npy`

后续已经修正：

- manifest builder 输出路径解析错误
- candidate ranking 中 `plane_corrected` 优先级问题

修复后的 rerun 结果统一写入：

- [one_to_one_plane_corrected_rerun](../reports/one_to_one_plane_corrected_rerun)

这一点非常关键，因为它说明：

> 当前报告中的“最新可信结果”应优先参考 `plane_corrected rerun`，而不是更早的旧 one_to_one 结果。

---

## 5. 实验一：RHEED-to-AFM Descriptor MVP

对应目录：

- [reports/rheed_descriptor_mvp](../reports/rheed_descriptor_mvp)

### 5.1 这条路线想回答什么

这是项目最早的诊断性 MVP，目标不是直接生成 AFM 图像，而是先问：

> RHEED 视频中是否含有足以预测 AFM handcrafted descriptor 的稳定信息？

如果答案是“是”，说明至少存在一些跨模态相关性；  
如果答案是“否”，就说明：

- RHEED 表征太弱
- 或 descriptor 太压缩，无法承载 morphology
- 或数据配对本身有问题

### 5.2 从 RHEED 视频到数值特征：代码如何处理

这部分实现位于 `src/rheed2morph/rheed/mvp.py`。

#### 第一步：选择 canonical RHEED 视频

每个 sample 的 `RHEED/` 文件夹中可能有多个视频。代码会：

1. 过滤出可见、可解码的视频文件
2. 用 `imageio_ffmpeg.count_frames_and_secs` 读取帧数和时长
3. 若文件名含 `main`，优先选它
4. 否则选“可解码且时长最长”的视频

这么做的原因是：

- 避免手工指定视频
- 尽量选信息量最大的那段生长过程

#### 第二步：解码整段视频，再均匀抽帧

代码会先解码视频全部帧，再均匀采样 `8` 帧：

- 默认 `frame_count = 8`
- 采样方式是 `linspace` 均匀取样

这里的思想很朴素：  
在小数据条件下，不试图建复杂时序模型，而先保留视频从头到尾的代表性截面。

#### 第三步：每帧变成标准化图像 tensor

每一帧会被：

1. 转成 RGB
2. resize 到 `224 x 224`
3. 按 ImageNet mean/std 标准化

这样做是因为后面使用的是 ImageNet 预训练 ResNet50，它期待这种输入分布。

#### 第四步：用预训练 ResNet50 提特征

代码优先尝试：

- `torchvision.models.resnet50(weights=DEFAULT)`

如果环境不支持，再 fallback 到：

- `torch.hub` 下载 ResNet50

注意这里**没有微调**，而是把最后的分类头去掉，只保留卷积 backbone：

- 每帧输出一个 `2048` 维特征

#### 第五步：视频级聚合

对 8 帧特征做：

- 按维度求 `mean`
- 按维度求 `std`

然后拼接成一个 `4096` 维视频 embedding。

这一步很重要，因为它保留了两类信息：

- 平均外观
- 随时间的变化幅度

这是一个非常轻量但合理的时序汇总方案。

### 5.3 为什么先用预训练 ResNet50

对当前阶段而言，这是一个合理选择：

1. `ResNet50` 是成熟、稳定、广泛验证的图像 backbone
2. ImageNet 预训练能提供通用纹理、边缘、频率结构特征
3. 在小样本场景下，冻结 encoder 能显著降低过拟合风险
4. 它给出了一个“如果连通用视觉特征都不行，那问题多半不在回归头”的诊断基线

当然，它也有明显局限：

- RHEED 与自然图像差异很大
- 它没有专门建模 RHEED 的时序动态
- 它不是在材料衍射数据上预训练的

所以未来完全可以替换成更合适的 encoder，例如：

- RHEED domain-specific self-supervised encoder
- 时序 CNN / transformer
- contrastive video representation model

但在当前阶段，`ResNet50 frozen encoder` 是一个很好的“起点基线”。

### 5.4 AFM descriptor 目标是什么

descriptor 表来自：

- `data/afm_descriptor_reconstruction/selected_descriptors/selected_descriptor_table.csv`

表中共有 `30` 列，其中 3 列是 ID 信息，因此实际数值 descriptor 约为 `27` 个。  
这些 descriptor 大体分为几类：

- 高度统计量
  - `median_height`, `std_height`, `peak_to_valley`
- 分位数统计
  - `p01`, `p25`, `p75`, `p95`
- 粗糙度分布形状
  - `Rsk`, `Rku`
- 梯度/方向性
  - `grad_mean`, `orientation_entropy`
- 频域能量
  - `low_freq_power`, `mid_freq_power`, `high_freq_power`, `radial_psd_slope`
- 自相关长度与各向异性
  - `autocorr_length_x`, `autocorr_length_y`, `anisotropy_ratio`
- 形貌连通域统计
  - `coverage_fraction`, `connected_component_count`, `component_density`, `mean_component_area`

这条路线的优点是：

- 目标维度小
- 训练简单
- 结果容易快速诊断

它的缺点也很明显：

- 把复杂形貌压缩成少量 hand-crafted 数字
- descriptor 可能丢失空间结构
- 即使数字误差不大，也不代表 morphology 真的被学到

### 5.5 descriptor 回归模型与评估方法

数据 join 后，代码会：

1. 按 `group_id` 做 group-aware holdout split
2. 在训练集内再用 `GroupKFold` 做模型选择
3. 候选模型包括：
   - `RidgeCV`
   - `PLSRegression`
   - `KNeighborsRegressor`
4. 输出 learned model，并和一个最近邻 baseline 做对比

指标包括：

- 每个 descriptor 的 `MAE`
- `RMSE`
- `R^2`

并导出：

- `predicted_vs_true_scatter.png`
- `nearest_neighbor_qualitative_grid.png`
- `metrics_summary.json`

### 5.6 descriptor baseline 结果

#### 全量早期 baseline

见：

- [summary.md](../reports/rheed_descriptor_mvp/summary.md)
- [predicted_vs_true_scatter.png](../reports/rheed_descriptor_mvp/predicted_vs_true_scatter.png)
- [nearest_neighbor_qualitative_grid.png](../reports/rheed_descriptor_mvp/nearest_neighbor_qualitative_grid.png)

关键数字：

- Embedded samples: `41`
- Joined rows: `166`
- Best model: `knn (n_neighbors = 7)`
- Learned mean `MAE / RMSE / R^2 = 0.7577 / 0.9755 / -0.8206`

这个结果说明：

- 数值上略好于最近邻 baseline
- 但 `R^2` 仍显著为负
- 说明模型比“预测均值附近”并没有表现出可靠的解释能力

#### one-to-one 子集上的 descriptor baseline

旧 `one_to_one` 目录里最常被引用的是 `1um`：

- [1um summary](../reports/one_to_one/1um/summary.md)

结果为：

- `1um`: `MAE 0.7815`, `RMSE 1.0611`, `R^2 -1.1133`

`0.5um` 更差：

- `MAE 1.0921`, `R^2 = nan`

旧 `all_size_representative` descriptor 结果由于报告完整性问题，不适合独立比较。

### 5.7 对 descriptor 路线的结论

descriptor MVP 的价值主要体现在“诊断”：

1. 它证明原始 one-to-many pairing 会污染监督信号
2. 它证明即使清洗 pairing，`RHEED -> handcrafted AFM descriptors` 仍然很弱
3. 它说明最终路线不应停留在 descriptor regression

从第一性原理看，这个失败并不意外：

- 输入是高维、复杂、时变的衍射视频
- 输出却被压缩成几十个手工数字

这会把真正关键的 morphology 空间信息丢掉。  
所以 descriptor 路线适合作为“故障诊断仪”，不适合作为“最终模型路线”。

---

## 6. 实验二：one-to-one 正式 image-latent MVP

对应目录：

- [reports/one_to_one](../reports/one_to_one)

### 6.1 为什么从 descriptor 转向 image latent

如果最终目标是“AFM-like morphology prediction”，那么输出空间本质上应该保留：

- island / grain 结构
- 空间分布
- 纹理方向性
- 连通性和尺度信息

hand-crafted descriptor 很难完整承载这些结构。  
因此更自然的思路是：

1. 先用 autoencoder 把 AFM 图像压缩进 latent space
2. 再学习 `RHEED embedding -> AFM latent`

这样做的直觉是：

- latent 比原图低维，更容易学习
- latent 比 descriptor 更保留图像结构
- 如果 autoencoder 足够好，latent 就是一个更自然的 morphology target

### 6.2 AFM autoencoder 的输入与预处理

这部分由 `scripts/train_afm_autoencoder_mvp.py` 和 `src/rheed2morph/afm/mvp.py` 实现。

数据处理流程是：

1. 从 manifest 读取 `afm_path`
2. 加载 `npy/png/csv/txt` 形式的 AFM height map
3. 双线性 resize 到 `128 x 128`
4. 默认做 `per_image_zscore`

为什么默认使用 per-image normalization？

- 不同 AFM 的绝对高度零点和量纲范围可能不同
- 当前目标是先学习 morphology，而不是绝对高度标定
- 每张图单独标准化，能把模型注意力更多放在空间纹理结构上

脚本也支持：

- `per_image_minmax`
- `global_zscore`

但当前正式实验主要使用 `per_image_zscore`。

### 6.3 AFM autoencoder 结构

项目里实现了两种 AE：

#### A. baseline ConvAutoencoder

- 编码器：3 层 stride-2 Conv
- 通道大致从 `1 -> 16 -> 32 -> 64`
- flatten 后线性映射到 latent
- decoder 用转置卷积逐步还原

#### B. residual ConvAutoencoder

- 更宽的通道数
- 中间加入 residual block
- 解码器也更复杂

默认正式实验更多使用 `residual` 版本，但后续迭代发现：

> 在当前 small-data regime 下，更复杂的 decoder 不一定更好。

### 6.4 AE 的损失函数与训练策略

脚本支持两种 pixel loss：

- `MSE`
- `Smooth L1`

并额外加入了 `edge loss`：

- 比较重建图与原图在 x/y 方向的梯度差异

总损失形式可理解为：

`pixel loss + edge_weight * edge loss`

这样设计的原因是：

- 单纯 MSE 容易过度平滑
- morphology 任务里，边界/纹理梯度很重要

训练策略包括：

- group-aware train/val split
- early stopping
  - monitor `val reconstruction loss`
  - 默认 `patience = 15`
  - 默认 `min_delta = 1e-5`

输出包括：

- `autoencoder_checkpoint.pt`
- `metrics.json`
- `training_history.csv`
- `training_curve.png`
- `recon_grid.png`
- `afm_latents.npy`
- `afm_latent_index.csv`
- `latent_pca.png`

### 6.5 如何判断 AE 是否“有资格”作为跨模态 target

这一步是整个项目最关键的科学判断之一。

当前代码不会只看 loss，而是同时看：

- `reconstruction_mae`
- `reconstruction_std_ratio`
- `recon_grid.png`

脚本内置了一个 warning heuristic。若出现以下情况，会写入明确 warning：

- reconstruction 过平滑
- latent 输出塌缩
- 重建方差太小

对应警告语句为：

> AFM latent space may not yet be morphology-preserving; RHEED-to-latent metrics should not be overinterpreted.

这个规则在方法论上非常重要，因为它防止出现一种常见误判：

> latent 指标看起来不错，但其实 latent 空间本身没有学到真实 morphology。

### 6.6 RHEED-to-AFM latent 模型

`scripts/rheed_to_afm_latent_mvp.py` 使用：

- 现成的 RHEED embedding
- AE 产生的 AFM latent

构造 `x = RHEED embedding`, `y = AFM latent`。

流程包括：

1. 根据 manifest 与 latent index 做 join
2. group-aware train/test split
3. 训练候选模型
   - `RidgeCV`
   - `KNeighborsRegressor`
   - `MLPRegressor(hidden_layer_sizes=(128,), early_stopping=True)`
4. 通过 `GroupKFold` 选择最优模型
5. 与 dummy baselines 对比：
   - `train_mean_latent`
   - `random_train_latent`

### 6.7 latent 实验为什么要特别看 qualitative retrieval

这个阶段，单纯看 `latent MSE` 并不够。  
真正重要的是：预测出来的 latent，在 decoder 或近邻检索后，是否能对应到“形貌上像”的 AFM。

因此脚本会固定输出：

- `nearest_latent_grid.png`
  - `RHEED thumbnail`
  - `True AFM`
  - `nearest predicted AFM prototype`
  - `decoded predicted AFM`（若有 decoder）
- `generated_afm_grid.png`

这比单一标量更接近真实科学问题：

> 模型找回来的 morphology，是否真的和 ground truth 在形貌上相似？

### 6.8 `one_to_one` 正式实验结果

总表见：

- [comparison_summary.md](../reports/one_to_one/comparison_summary.md)
- [comparison_metrics.csv](../reports/one_to_one/comparison_metrics.csv)
- [latent_experiment_interpretation.md](../reports/one_to_one/latent_experiment_interpretation.md)

关键结果可以概括为：

| 子集 | AE best val loss | AE 判断 | latent model | learned vs mean-latent |
| --- | ---: | --- | --- | --- |
| `1um` | 0.8731 | 部分保留 morphology，但偏平滑 | `knn` | 未优于 mean baseline |
| `all_size_representative` | 0.8918 | 部分保留 morphology，但异质性更强 | `ridge` | 未优于 mean baseline |
| `0.5um` | 1.0091 | warning 触发，latent 不稳定 | `knn` | 未优于 mean baseline |
| `5um` | 1.0161 | smoke only | `knn` | 数值优于 mean，但样本太小，无统计意义 |

`1um` 与 `all_size_representative` 的旧 formal latent summary：

- [1um latent summary](../reports/one_to_one/1um/rheed_to_afm_latent/summary.md)
- [all_size latent summary](../reports/one_to_one/all_size_representative/rheed_to_afm_latent/summary.md)

从结果上看：

- 模型一般能优于随机 baseline
- 但在最关键的比较上，**未能优于 `train_mean_latent` baseline**

这意味着当前 learned mapping 还没有证明：

> RHEED 端提供了超出“latent 平均先验”的稳定 morphology 信息。

### 6.9 `1um` autoencoder 进一步迭代

后续在 `1um` 上做了额外 AE 结构/损失试验：

- [autoencoder_iteration_note.md](../reports/one_to_one/1um/autoencoder_iteration_note.md)

三版结果如下：

| 版本 | 主要设置 | 结论 |
| --- | --- | --- |
| `v1_baseline_mse` | formal baseline | 可用，但偏平滑 |
| `v2_residual_smoothl1_edge0.2` | 更复杂 residual AE | 虽然 loss 更低，但出现明显 decoder artifact / checkerboard |
| `v3_baseline_smoothl1_edge0.1` | 简单 decoder + 更温和 edge loss | 当前最稳妥的 compromise |

这组结果非常有启发：

> 在当前小样本条件下，模型更复杂不等于更好；更强的 decoder 反而更容易产生伪纹理和视觉 artifact。

对应 `v3` 的 latent rerun：

- [rheed_to_afm_latent_v3 summary](../reports/one_to_one/1um/rheed_to_afm_latent_v3/summary.md)

结果仍然表明：

- learned model 还是没打败 `mean-latent baseline`

所以问题不只是“换个回归器”就能解决。

---

## 7. 实验三：plane-corrected rerun

对应目录：

- [reports/one_to_one_plane_corrected_rerun](../reports/one_to_one_plane_corrected_rerun)

### 7.1 rerun 的动机

在代码回溯中确认：

- 旧 manifest 曾混入 `processed_afm`
- `1um` 中甚至包含了本不该属于该子集的样本
- 因此旧 `one_to_one` 结果虽然有参考价值，但不应被当作最终可信版本

因此对两个最关键子集重新运行：

- `1um`
- `all_size_representative`

且明确要求：

- 使用修正后的 manifest
- 使用 `plane_corrected` 数据
- 不覆盖旧报告

### 7.2 rerun 结果

汇总见：

- [plane-corrected rerun summary](../reports/one_to_one_plane_corrected_rerun/summary.md)

#### `1um` rerun

- [AE summary](../reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/summary.md)
- [latent summary](../reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/summary.md)
- [AE recon grid](../reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/recon_grid.png)
- [latent retrieval grid](../reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/nearest_latent_grid.png)

关键数字：

- AE best val loss: `0.4795`
- Reconstruction warning: `yes`
- Selected latent model: `knn`
- Learned latent MSE / cosine: `0.4512 / 0.6862`
- Mean-latent baseline: `0.3270 / 0.6579`

解释：

- 数值看上去比旧 run “更好看”，但仍未优于 `mean-latent baseline`
- AE 仍触发 morphology warning，说明 target latent 仍不够稳

#### `all_size_representative` rerun

- [AE summary](../reports/one_to_one_plane_corrected_rerun/all_size_representative/afm_autoencoder/summary.md)
- [latent summary](../reports/one_to_one_plane_corrected_rerun/all_size_representative/rheed_to_afm_latent/summary.md)
- [AE recon grid](../reports/one_to_one_plane_corrected_rerun/all_size_representative/afm_autoencoder/recon_grid.png)
- [latent retrieval grid](../reports/one_to_one_plane_corrected_rerun/all_size_representative/rheed_to_afm_latent/nearest_latent_grid.png)

关键数字：

- AE best val loss: `0.4750`
- Reconstruction warning: `no`
- Selected latent model: `ridge`
- Learned latent MSE / cosine: `2.8113 / 0.2732`
- Mean-latent baseline: `1.4484 / 0.5272`

解释：

- reconstruction 比 `1um rerun` 更稳定一些
- 但 cross-modal retrieval 依然明显不如 mean baseline

### 7.3 rerun 的科学意义

这次 rerun 的价值非常大，因为它帮助我们分离了两个问题：

1. 数据有没有用错？
2. 即使用对了，模型到底行不行？

答案是：

- 数据确实曾经有问题
- 但修正数据后，主结论没有反转

也就是说，当前困境**不只是数据用错**，而是：

- 数据小
- AFM latent target 还弱
- RHEED 表征还弱

这比单纯“修 bug 后就成功”更接近真实科研情况。

---

## 8. 三组实验的总体对比

### 8.1 方法论层面的演进

三组实验其实对应了三层越来越接近最终目标的方法论：

| 阶段 | 输出目标 | 目的 |
| --- | --- | --- |
| `rheed_descriptor_mvp` | AFM handcrafted descriptors | 快速验证是否存在弱相关信号 |
| `one_to_one` | AFM image latent | 验证 image-level / latent-level 路线是否可行 |
| `one_to_one_plane_corrected_rerun` | 同上，但数据更干净 | 排除数据完整性问题后，确认结论是否稳定 |

### 8.2 当前最可信的结论

如果只保留最可信、最保守的判断，应当是：

1. `descriptor regression` 弱，不能作为最终路线
2. `one-to-one manifest` 是必要的数据工程
3. `plane_corrected` 数据应作为主数据源
4. 当前 `AFM autoencoder` 只能部分保留 morphology
5. 当前 `RHEED -> latent` 仍未证明优于简单 latent prior

---

## 9. 当前实验困境的原因分析

这一部分是整份报告最重要的解释性内容。

### 9.1 数据角度的问题

#### 9.1.1 样本量极小

这是最大的现实约束。

以最有价值的主 benchmark `1um` 为例，当前只有 `36` 个 clean pair。  
对机器学习来说，这意味着：

- 划分 train/val/test 后，每个 split 都非常小
- 不同随机划分会显著影响结果
- 复杂模型极易过拟合

换句话说，目前不是“模型不会学”，而是“可供学习的稳定统计规律非常少”。

#### 9.1.2 配对歧义真实存在

RHEED 是 in-situ 动态观测，AFM 是 ex-situ 终态观测。  
即使同一个 sample 的配对被人工定义为一对一，也仍然可能存在：

- 生长后处理差异
- AFM 扫描区域选择差异
- 局部 morphology 非均匀性

所以从物理上说，这个任务的标注本来就带噪声，不是一个完美监督任务。

#### 9.1.3 多尺度异质性

`all_size_representative` 里混合了不同 scan size。  
这会引入两个层面的变化：

- morphology 尺度本身不同
- 统计分布也不同

从模型角度看，这等于要求一个很小的数据集同时学多个尺度域。  
这会放大 smoothing 和 collapse。

#### 9.1.4 metadata 与文件名并不总一致

例如此前 `6100` 就出现了：

- 文件名像 `1um`
- metadata 却记录成 `2.0 um`

这类问题说明：

> 数据 QA 不是附属工作，而是实验结论可信度的一部分。

### 9.2 算法角度的问题

#### 9.2.1 RHEED encoder 仍然过于通用

当前 RHEED 表征来自：

- ImageNet 预训练 ResNet50
- 对 8 帧做 mean/std pooling

这是一种合理 baseline，但它不是为 RHEED 设计的。  
主要局限包括：

- 不理解衍射条纹的物理意义
- 不建模长时序动态
- 对 domain-specific pattern 缺少适应性

所以即使有真实 morphology signal，也可能没有被当前 encoder 充分提取出来。

#### 9.2.2 时序建模过弱

当前视频聚合方式是：

- 均匀抽 8 帧
- 每帧提特征
- 最后做 mean/std pooling

这非常稳定，但也非常“粗”。  
它保留了静态纹理统计，却几乎没有显式建模：

- 哪些变化发生在前期/后期
- 哪些短时动态与最终 morphology 更相关

因此，有可能真正关键的信息在时序维度上被平均掉了。

#### 9.2.3 AFM autoencoder 仍然偏平滑

这是当前最核心的技术瓶颈。

autoencoder 若重建出来的是“平均纹理图”，那它的 latent 学到的就是：

- 大致纹理类别
- 低频结构

而不是细致 morphology。

于是后续 `RHEED -> latent` 就会出现一种错觉：

- latent MSE 看起来不差
- cosine 甚至可能偏高
- 但其实模型只是在预测“平均 AFM”

这也是为什么当前代码专门加入了 `quality_warning` 和 qualitative grid。

#### 9.2.4 小样本下复杂 decoder 反而带来 artifact

`1um` 的 `v2 residual` 版本已经给出直接证据：

- loss 下降
- 但图像质量变差
- 出现 checkerboard / banding artifact

这说明在当前阶段：

> 更复杂的生成器不等于更真实的 morphology。

#### 9.2.5 当前评价指标还没有触达最终科学问题

`latent MSE`、`cosine similarity` 都是必要指标，但并不是最终问题本身。  
最终想回答的是：

> 模型能否根据 RHEED 给出 morphology 上像真的 AFM？

而当前 latent space 还不够可靠，因此这些标量指标只能被“谨慎参考”，不能被“强解释”。

---

## 10. 当前阶段的综合结论

如果把这一阶段工作压缩成几句话，可以这样向教授汇报：

1. 我们已经验证了 small-data 条件下直接做大模型生成不现实，因此采用了分阶段 MVP 路线。
2. 第一阶段 descriptor baseline 表明：修复配对歧义是必要的，但 handcrafted descriptor 空间不足以承载最终 morphology 目标。
3. 第二阶段 image-latent pipeline 已经完全打通，包括 AFM autoencoder、RHEED-to-latent、检索式 qualitative grid 与 dummy baseline 对比。
4. 第三阶段通过 plane-corrected rerun 修复了数据完整性问题，并确认：即使用对数据，当前 learned cross-modal mapping 仍未稳定优于 mean-latent baseline。
5. 因此当前最主要的技术瓶颈，是 AFM latent 目标质量与 RHEED representation 质量，而不是简单回归头没有调好。

---

## 11. 下一阶段建议

### 11.1 优先级一：先把 AFM latent target 做扎实

这是最重要的下一步。  
建议策略：

1. 继续以 `1um` 作为主 benchmark
2. 保持简单 decoder，不盲目加深网络
3. 系统比较：
   - `latent_dim`
   - normalization
   - `pixel_loss`
   - `edge_loss_weight`
   - mild augmentation
4. 以 qualitative reconstruction 为第一判断标准

成功标准不是 loss 更低，而是：

> reconstruction 是否真的保留 grain / island morphology，而不是只变得更平滑。

### 11.2 优先级二：当 AFM target 过关后，再评估 RHEED encoder

只有在 AFM latent 已经比较可信时，才有资格判断 RHEED 端到底弱不弱。  
届时的判断标准应是：

- learned model 是否稳定优于 `train_mean_latent`
- nearest retrieval 是否在 morphology 上更像 ground truth

如果那时仍然失败，下一步就应明确转向：

> 构建 RHEED self-supervised encoder

### 11.3 优先级三：继续做数据质量治理

虽然当前主要瓶颈已不只是数据，但数据治理仍然重要：

- 持续核查 `plane_corrected` 使用一致性
- 继续审查 metadata / filename 冲突
- 明确哪些 sample 适合作为主 benchmark
- 对极小子集只做 smoke/qualitative，不做统计性结论

### 11.4 暂不建议的方向

在当前阶段，不建议优先投入：

- diffusion
- 大规模 finetuning
- 继续在 descriptor regression 上做精调

原因很简单：

- 数据规模还不足以支撑这些方向的有效判断
- 它们会增加复杂度，却不一定增加科学可解释性

---

## 12. 建议汇报时重点展示的图表

建议组会或汇报时优先展示以下图：

### 12.1 descriptor 诊断结果

- [descriptor scatter](../reports/rheed_descriptor_mvp/predicted_vs_true_scatter.png)
- [descriptor qualitative grid](../reports/rheed_descriptor_mvp/nearest_neighbor_qualitative_grid.png)

这两张图用于说明：

- descriptor 路线确实弱
- 但它作为 baseline 已完成使命

### 12.2 旧 one_to_one 的 AE 与 latent grid

- [1um recon grid](../reports/one_to_one/1um/afm_autoencoder/recon_grid.png)
- [1um latent retrieval grid](../reports/one_to_one/1um/rheed_to_afm_latent/nearest_latent_grid.png)

用于说明：

- image-latent 路线已跑通
- 但当前只到“部分有希望”，还谈不上“成功”

### 12.3 plane-corrected rerun 的关键图

- [1um rerun recon](../reports/one_to_one_plane_corrected_rerun/1um/afm_autoencoder/recon_grid.png)
- [1um rerun retrieval](../reports/one_to_one_plane_corrected_rerun/1um/rheed_to_afm_latent/nearest_latent_grid.png)
- [all-size rerun recon](../reports/one_to_one_plane_corrected_rerun/all_size_representative/afm_autoencoder/recon_grid.png)
- [all-size rerun retrieval](../reports/one_to_one_plane_corrected_rerun/all_size_representative/rheed_to_afm_latent/nearest_latent_grid.png)

用于说明：

- 数据修正后，结论没有翻转
- 当前真正的 bottleneck 已经很清楚

---

## 13. 最后的判断

到当前阶段，这个项目已经完成了非常重要的一步：

> 它把“有没有信号”和“为什么现在还没成功”这两个问题拆开了。

我们现在知道：

- 不是简单代码没跑通
- 不是单纯 manifest 没修好
- 也不是随便换个回归器就能解决

当前困境来自更根本的地方：

- 数据量小
- 配对噪声高
- 输出 morphology 目标复杂
- RHEED 表征与 AFM latent 都还不够物理上贴切

这其实是一个正常且有价值的科研阶段。  
因为它告诉我们，下一步应当把资源投向哪里，而不是在错误方向上继续消耗时间。

当前最合理、最保守、也最具有科学价值的下一步路线是：

> 先把 AFM latent target 做到足够 morphology-preserving，再判断是否需要更强的 RHEED self-supervised representation learning。
