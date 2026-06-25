# 当前 RHEED-to-AFM Pipeline 模型结构说明

## 1. 说明范围

这份文档只描述**当前真正参与模型前向传播的 pipeline**，重点回答下面几个问题：

1. 当前输入 tensor 进入 pipeline 之后，会经过哪些模型模块；
2. 每个模块的 layer 结构、输入输出维度、参数量分别是多少；
3. 哪些模块是 trainable 的，哪些模块只是 frozen feature extractor 或 non-parametric predictor；
4. 如果要画结构图，应该如何拆成几个清晰的子模块。

本文**不重点讨论**：

- `clean_frames` / `valid_mask` 是如何从原始 RHEED 视频生成的；
- AFM plane correction 是如何实现的；
- manifest 是如何构建的。

本文重点讨论的是：

> 一个处理后的 RHEED tensor 进入当前系统之后，模型是如何把它映射到 AFM latent，再映射到 AFM 图像可视化的。

---

## 2. 当前主 pipeline 的总览

当前主 pipeline 可以拆成 4 段：

1. `Processed RHEED tensor -> ResNet50 frame encoder`
2. `Frame embeddings -> sample-level RHEED embedding`
3. `RHEED embedding -> AFM latent regressor`
4. `Predicted AFM latent -> AFM decoder`

如果用一句话概括当前主链路：

> `clean_frames [T,H,W] -> 64帧采样 -> ResNet50 -> 64个2048维帧特征 -> 8192维样本特征 -> KNN -> 64维AFM latent -> residual decoder -> 1x128x128 AFM图`

需要特别注意：

- 当前 `RHEED -> AFM latent` 的**主预测器不是神经网络**，而是 `KNeighborsRegressor`
- 因此当前推理链中真正有大量参数的只有：
  - 前端 frozen `ResNet50`
  - 后端 `AFM residual autoencoder`
- 中间的 latent predictor `KNN` 没有可训练参数

---

## 3. 数据流总表

下面这张表是当前最适合拿去画总结构图的版本。

| Stage | 输入 shape | 输出 shape | 模块 | 参数量 | 说明 |
| --- | --- | --- | --- | ---: | --- |
| A1 | `[T,H,W]` | `[64,H,W]` | 时间均匀采样 | `0` | 当前 processed 模式最多取 64 帧 |
| A2 | `[64,H,W]` | `[64,3,224,224]` | 灰度复制到3通道 + resize + ImageNet normalize | `0` | 非学习模块 |
| B | `[64,3,224,224]` | `[64,2048]` | truncated ResNet50 encoder | `23,508,032` | frozen |
| C | `[64,2048]` | `[8192]` | temporal stats aggregation | `0` | `mean/std/delta_mean/delta_std` |
| D | `[8192]` | `[64]` | KNN regressor (`k=3`) | `0` | 当前主 latent predictor |
| E | `[64]` | `[1,128,128]` | AFM residual decoder | `2,681,889` | 只用于定性可视化 |

如果画完整系统，还应补一条离线目标生成支线：

| Offline Stage | 输入 shape | 输出 shape | 模块 | 参数量 | 说明 |
| --- | --- | --- | --- | ---: | --- |
| F | `[1,128,128]` | `[64]` | AFM residual encoder | `2,577,408` | 用于生成训练目标 latent，不是当前 RHEED 推理时的前向输入 |

完整 AFM autoencoder 总参数量是：

- `5,259,297`

它由两部分组成：

- encoder: `2,577,408`
- decoder: `2,681,889`

---

## 4. 当前 RHEED 主干：Processed tensor 到 sample embedding

### 4.1 输入张量

当前主输入来自：

- `data/raw_RHEED_selected_test_512/<sample>/tensors/model_input.npz`

当前真正送进模型的是：

- `clean_frames`
- shape: `[T,H,W]`
- dtype: `float32`
- value range: `[0,1]`

在 processed 模式下：

- 先用 `valid_mask` 把无效区域置零
- 再把时间维采样到 `64` 帧

因此进入视觉编码器之前，单个样本的数据形状是：

- `[64,H,W]`

这里的 `H,W` 在数据集里不是全局固定的，但进入 ResNet50 前会统一 resize。

### 4.2 前处理到 ResNet 输入

每一帧做如下变换：

1. 灰度帧 `[H,W]`
2. 复制成 3 通道 `[H,W,3]`
3. resize 到 `[224,224,3]`
4. 转成 CHW tensor `[3,224,224]`
5. 做 ImageNet 标准化

对 64 帧堆叠后，送入 ResNet50 的 batch 形状为：

- `[64,3,224,224]`

### 4.3 ResNet50 encoder

当前代码使用的是：

- `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`

然后截断为：

- `torch.nn.Sequential(*list(model.children())[:-1])`

也就是说：

- 保留 `conv1` 到 `avgpool`
- 去掉最终分类层 `fc`

参数量：

- ResNet50 full: `25,557,032`
- 去掉分类头后的 truncated encoder: `23,508,032`
- 被去掉的 `fc` 参数量: `2,049,000`

### 4.4 ResNet50 的 stage 结构与维度

当前 encoder 的 stage 级结构可以写成：

1. `conv1`
   - `7x7 conv`, stride `2`, channels `3 -> 64`
   - 输出：`[64,112,112]`
2. `bn1`
   - 输出：`[64,112,112]`
3. `relu`
   - 输出：`[64,112,112]`
4. `maxpool`
   - `3x3`, stride `2`
   - 输出：`[64,56,56]`
5. `layer1`
   - `3` 个 bottleneck blocks
   - 输出：`[256,56,56]`
6. `layer2`
   - `4` 个 bottleneck blocks
   - 输出：`[512,28,28]`
7. `layer3`
   - `6` 个 bottleneck blocks
   - 输出：`[1024,14,14]`
8. `layer4`
   - `3` 个 bottleneck blocks
   - 输出：`[2048,7,7]`
9. `avgpool`
   - global average pooling
   - 输出：`[2048,1,1]`

对单帧 flatten 后：

- 单帧 feature 维度 = `2048`

因此 64 帧全部编码后，得到：

- `[64,2048]`

### 4.5 当前 sample-level aggregation

当前 processed 模式不是直接对 64 帧做平均，而是做 `temporal_stats` 聚合：

1. frame embedding mean
   - `2048`
2. frame embedding std
   - `2048`
3. 相邻帧 embedding 差分 mean
   - `2048`
4. 相邻帧 embedding 差分 std
   - `2048`

最终拼接得到：

- sample-level RHEED embedding = `8192`

所以当前从 RHEED 到 latent predictor 的真正输入向量维度是：

- `8192`

对比旧 raw baseline：

- raw baseline 使用 `8` 帧 + `mean/std`
- 因此旧 raw baseline 的样本向量维度是 `4096`
- 当前 processed pipeline 使用 `64` 帧 + `mean/std/delta_mean/delta_std`
- 因此当前 processed pipeline 的样本向量维度是 `8192`

---

## 5. 当前 latent predictor：KNN regressor

### 5.1 当前选中的模型

在当前 processed `1um` benchmark 中，最终被选中的 latent predictor 是：

- `KNeighborsRegressor`
- `n_neighbors = 3`

这是因为当前 `scripts/rheed_to_afm_latent_mvp.py` 中的候选集固定为：

- `ridge`
- `knn`
- `mlp`

其中 `knn` 的参数在代码里固定为：

- `KNeighborsRegressor(n_neighbors=3)`

### 5.2 KNN 在当前 pipeline 中的角色

它的作用是：

> 把 `8192` 维 RHEED embedding 映射成 `64` 维 AFM latent

但它不是神经网络，所以：

- trainable parameter count = `0`
- 没有层数可以展开成传统深度网络图
- 核心操作是：
  - 对输入 embedding 做 `StandardScaler`
  - 在训练集 embedding 空间中找最近的 `3` 个邻居
  - 对对应的 `3` 个目标 latent 做均值回归

### 5.3 当前训练/测试张量维度

在当前 processed `1um` 实验中：

- train rows: `27`
- test rows: `9`

因此：

- `x_train`: `[27,8192]`
- `y_train`: `[27,64]`
- `x_test`: `[9,8192]`
- `y_test`: `[9,64]`

这说明当前 KNN 实际上是在一个非常小的样本 regime 中工作：

- 只用 `27` 个训练样本
- 每个样本的输入维度却高达 `8192`

这一点本身也解释了为什么模型很容易不稳定。

### 5.4 当前 KNN 输出

KNN 输出的是：

- predicted latent shape: `[64]`

注意这里是**连续 latent 回归**，不是类别标签。

所以当前主 quantitative branch 实际上是：

`8192-d RHEED embedding -> 64-d predicted AFM latent`

---

## 6. 当前 AFM latent 空间与 autoencoder 结构

当前 RHEED pipeline 预测的不是 AFM image 本身，而是 AFM autoencoder 的 latent。

也就是说，当前系统的监督目标是：

- `64` 维 AFM latent vector

这个 latent 来自一个已经训练好的 AFM residual autoencoder。

### 6.1 当前 AFM autoencoder checkpoint 配置

当前 `1um` rerun 使用的 AFM autoencoder 配置是：

- model type: `residual`
- image size: `128`
- latent dim: `64`
- normalize mode: `per_image_zscore`
- pixel loss: `smooth_l1`
- edge loss weight: `0.2`

因此，AFM 侧输入输出的标准形状是：

- input AFM image: `[1,128,128]`
- latent: `[64]`
- reconstructed AFM image: `[1,128,128]`

### 6.2 AFM residual autoencoder 总参数量

总参数量：

- `5,259,297`

其中：

- encoder: `2,577,408`
- decoder: `2,681,889`

### 6.3 AFM residual encoder 结构

输入：

- `[1,128,128]`

编码器结构：

1. `Conv2d(1 -> 32, kernel=3, stride=2, padding=1)`
   - 输出：`[32,64,64]`
2. `LeakyReLU(0.1)`
   - 输出：`[32,64,64]`
3. `ResidualBlock(32)`
   - 两个 `3x3 conv`
   - 输出：`[32,64,64]`
4. `Conv2d(32 -> 64, kernel=3, stride=2, padding=1)`
   - 输出：`[64,32,32]`
5. `LeakyReLU(0.1)`
   - 输出：`[64,32,32]`
6. `ResidualBlock(64)`
   - 输出：`[64,32,32]`
7. `Conv2d(64 -> 128, kernel=3, stride=2, padding=1)`
   - 输出：`[128,16,16]`
8. `LeakyReLU(0.1)`
   - 输出：`[128,16,16]`
9. `ResidualBlock(128)`
   - 输出：`[128,16,16]`
10. `Flatten`
   - 输出：`128 * 16 * 16 = 32768`
11. `Linear(32768 -> 64)`
   - 输出 latent：`[64]`

因此 encoder 的核心压缩路径是：

`1x128x128 -> 32x64x64 -> 64x32x32 -> 128x16x16 -> 32768 -> 64`

### 6.4 ResidualBlock 的内部结构

每个 `ResidualBlock(C)` 的结构是：

1. `Conv2d(C -> C, kernel=3, padding=1)`
2. `LeakyReLU(0.1)`
3. `Conv2d(C -> C, kernel=3, padding=1)`
4. 残差相加
5. `LeakyReLU(0.1)`

也就是说，它不是 bottleneck block，而是一个比较简单的两层残差块。

### 6.5 AFM residual decoder 结构

当前 RHEED 推理完成后，如果要把预测的 latent 变回 AFM 图像做可视化，就走 decoder。

decoder 输入：

- latent `[64]`

结构如下：

1. `Linear(64 -> 32768)`
2. reshape
   - `[128,16,16]`
3. `ResidualBlock(128)`
   - 输出：`[128,16,16]`
4. `ConvTranspose2d(128 -> 64, kernel=4, stride=2, padding=1)`
   - 输出：`[64,32,32]`
5. `LeakyReLU(0.1)`
6. `ResidualBlock(64)`
   - 输出：`[64,32,32]`
7. `ConvTranspose2d(64 -> 32, kernel=4, stride=2, padding=1)`
   - 输出：`[32,64,64]`
8. `LeakyReLU(0.1)`
9. `ResidualBlock(32)`
   - 输出：`[32,64,64]`
10. `ConvTranspose2d(32 -> 1, kernel=4, stride=2, padding=1)`
   - 输出：`[1,128,128]`

因此 decoder 的扩张路径是：

`64 -> 32768 -> 128x16x16 -> 64x32x32 -> 32x64x64 -> 1x128x128`

---

## 7. 当前“推理 pipeline”和“训练 pipeline”的区别

这是画图时最容易混淆的一点。

### 7.1 当前在线推理主链

如果你画“一个 processed RHEED tensor 如何变成预测结果”，主链应该是：

1. `clean_frames [64,H,W]`
2. `-> [64,3,224,224]`
3. `-> truncated ResNet50`
4. `-> [64,2048]`
5. `-> temporal aggregation`
6. `-> [8192]`
7. `-> StandardScaler`
8. `-> KNN regressor (k=3)`
9. `-> predicted AFM latent [64]`
10. `->`
    - quantitative evaluation in latent space
    - optional decode to `[1,128,128]` AFM image

### 7.2 当前 AFM latent 目标生成支线

如果你画“训练目标 latent 是怎么来的”，那是另一条离线支线：

1. `AFM image [1,128,128]`
2. `-> residual encoder`
3. `-> latent [64]`

这条支线不是当前 RHEED tensor 推理时的主前向输入，但它决定了 RHEED 回归器到底在学什么。

### 7.3 为什么这两条线最好画成两块

因为当前系统本质上不是一个 end-to-end 联合训练网络，而是：

1. 先单独训练 AFM autoencoder，定义 latent 空间
2. 再把 RHEED 映射到这个 latent 空间

所以最适合的画法通常是：

- 左边：`RHEED branch`
- 右边：`AFM autoencoder branch`
- 中间：共享的 `64-d latent space`

---

## 8. 当前 pipeline 中哪些模块可训练

| 模块 | 当前是否训练 | 参数量 | 备注 |
| --- | --- | ---: | --- |
| ResNet50 truncated encoder | 否 | `23,508,032` | frozen feature extractor |
| temporal aggregation | 否 | `0` | 统计聚合 |
| KNN regressor | 否 | `0` | non-parametric |
| AFM residual encoder | 已离线训练完成 | `2,577,408` | 当前主推理时不更新 |
| AFM residual decoder | 已离线训练完成 | `2,681,889` | 当前只用于可视化解码 |

因此，从“当前 processed RHEED 推理”这个角度看：

- 实时主链里没有任何正在训练的深度网络参数
- 前端是 frozen CNN
- 中间是 non-parametric KNN
- 后端 decoder 只是拿来把 predicted latent 变成图像做 qualitative inspection

---

## 9. 你可以直接给 AI 画图工具的结构描述

如果你要喂给 AI 画图工具，我建议直接用下面这一段：

```text
Draw a two-branch scientific model diagram for the current RHEED-to-AFM pipeline.

Main inference branch:
Input is a processed RHEED tensor clean_frames with shape [64, H, W].
Each frame is masked by valid_mask, replicated to 3 channels, resized to 224x224, and normalized with ImageNet statistics.
The 64-frame batch with shape [64, 3, 224, 224] is passed through a frozen truncated ResNet50 encoder (23,508,032 parameters).
Show the ResNet stages explicitly:
conv1 7x7 s2 -> 64x112x112,
bn1/relu,
maxpool -> 64x56x56,
layer1 with 3 bottleneck blocks -> 256x56x56,
layer2 with 4 bottleneck blocks -> 512x28x28,
layer3 with 6 bottleneck blocks -> 1024x14x14,
layer4 with 3 bottleneck blocks -> 2048x7x7,
global average pool -> 2048.
This produces 64 frame embeddings of shape [64, 2048].
Aggregate them with temporal statistics: frame mean, frame std, delta mean, delta std.
Concatenate into one sample-level RHEED embedding of dimension 8192.
Pass the 8192-d vector into a KNN regressor with k=3 and zero trainable parameters.
The output is a predicted AFM latent vector of dimension 64.

AFM latent branch:
Show a separate AFM residual autoencoder latent space.
AFM images are 1x128x128.
The residual encoder has 2,577,408 parameters:
Conv2d 1->32 stride 2 -> 32x64x64,
ResidualBlock(32),
Conv2d 32->64 stride 2 -> 64x32x32,
ResidualBlock(64),
Conv2d 64->128 stride 2 -> 128x16x16,
ResidualBlock(128),
flatten 32768,
Linear 32768->64 latent.
Show the shared 64-d latent space in the center.
The decoder has 2,681,889 parameters:
Linear 64->32768,
reshape to 128x16x16,
ResidualBlock(128),
ConvTranspose2d 128->64 -> 64x32x32,
ResidualBlock(64),
ConvTranspose2d 64->32 -> 32x64x64,
ResidualBlock(32),
ConvTranspose2d 32->1 -> 1x128x128.

Emphasize that the current system is not end-to-end trainable:
front-end ResNet50 is frozen,
middle latent predictor is non-parametric KNN,
AFM autoencoder was trained offline and defines the 64-d target latent space.
```

---

## 10. 最关键的结构结论

如果你只保留最核心的结构信息，那么当前 pipeline 可以压缩成下面这 5 行：

1. `processed RHEED [64,H,W] -> [64,3,224,224]`
2. `frozen ResNet50 -> [64,2048]`
3. `temporal aggregation -> [8192]`
4. `KNN(k=3) -> AFM latent [64]`
5. `residual decoder -> AFM image [1,128,128]`

而当前真正决定系统能力上限的关键事实是：

- 前端不是专门为 RHEED 训练的 encoder；
- 中间不是 learnable deep regressor，而是小样本 KNN；
- 目标空间来自一个存在 quality warning 的 AFM autoencoder latent。

这三点一起定义了当前 pipeline 的结构边界。
