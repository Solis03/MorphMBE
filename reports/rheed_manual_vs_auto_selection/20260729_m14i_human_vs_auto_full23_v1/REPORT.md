# M14i/M12a：人工与自动 RHEED 输入的 Full23 配对比较

实验 ID：`20260729_m14i_human_vs_auto_full23_v1`

日期：2026-07-29

固定队列：23 个 growth（排除 6043 和 6055）

冻结方法：MorphMBE-M14i-Full23-OODAware-v1 + M12a edge-preserving terrace

## 结论

本实验建立了一个与原人工数据逐样品并列的机器选择数据集，并使用
Standalone 中完全相同的 M14i 目标头方法和 M12a 生成器进行比较。

最主要的结论是：

1. 自动 ROI/keyframe **没有破坏 Rq 的总体可预测性**。严格 LOO 下，
   `auto→auto` 的 Pearson `r=0.536`，`human→auto` 的
   `r=0.554`；两者的 Rq 误差与人工基准没有统计显著差异。
2. 自动输入的 **FSMI 和 confidence 迁移尚未验证成功**。
   `auto→auto` FSMI 的 Pearson `r=-0.130`，机器域 confidence 与实际
   误差的排序关系也不显著。因此不能把人工域的 FSMI/confidence 结论
   原样用于实时 UI。
3. M12a 生成器在机器域仍生成清楚的岛屿/台阶型 AFM，而非平面；
   机器域的图像纹理指标与人工域基本不变。但它仍继承 M14i 对高 Rq
   的动态范围压缩。
4. 6056 并不是当前自动流程的失败样品。机器选择 frame 160、人工
   frame 161；严格机器域 LOO 的 Rq 为 3.277 nm，真值 3.225 nm；
   同一 all-23 部署权重下自动输入预测 3.048 nm，与修复后的 UI 一致。

## 输入数据协议

23 个样品均从原人工 manifest 所指向的**同一原始视频、同一生长阶段**
重新解码。没有修改 raw RHEED 或 AFM。

- 人工版：原始人工 keyframe、人工 ROI、`k-7..k+8`。
- 机器版：V5 DINOv2-S 选择最佳可见旋转周期，V8 预测完整点阵模型
  输入 ROI，随后固定保存 `k-7..k+8`。
- keyframe 在 `selected_16` 中固定为 index 7。
- 每个机器样品保存 `keyframe_1`、`causal_8`、`selected_16`、
  RHEED 物理特征、R3D-18/DINOv2 嵌入以及完整来源 metadata。

机器选择相对人工选择：

| 指标 | 结果 |
|---|---:|
| 样品数 | 23 |
| 周期相位残差中位数 | 2.0 frames |
| ROI IoU 中位数 | 0.753 |
| 人工 ROI 被机器 ROI 覆盖的中位数 | 0.996 |
| 机器 keyframe quality 中位数 | 0.717 |

相位残差按估计旋转周期折叠；相隔完整周期的同相位帧不会被错误计作大
frame error。

## 三个严格 LOO 协议

### 1. human→human

冻结的原 M14i Full23 结果。每次用 22 个 growth 的人工输入训练，预测
第 23 个 growth 的人工输入。

### 2. auto→auto

使用相同方法、固定超参数和相同 23 growth；每次用 22 个机器输入训练，
预测第 23 个机器输入。这是未来全自动 pipeline 最公平的回顾性估计。

### 3. human→auto

每次只用 22 个 growth 的人工输入训练和做 inner calibration，再预测
第 23 个 growth 的机器输入。它专门量化 Standalone/人工训练域到 UI/
机器推理域的输入偏移。

所有三个协议均满足：

- held growth 的 AFM target 不参与外层训练；
- confidence 和 interval 的 inner calibration 也不使用 held target；
- 每个 growth 恰好 held once；
- 6043、6055 均未进入；
- 无检索式 AFM patch 输入。

## M14i 定量结果

| Target | 协议 | MAE (nm) | RMSE (nm) | Pearson r | Spearman ρ | 90% interval coverage |
|---|---|---:|---:|---:|---:|---:|
| Rq | human→human | 1.466 | 2.054 | 0.509 | 0.499 | 0.870 |
| Rq | auto→auto | 1.536 | 2.122 | 0.536 | 0.469 | 0.826 |
| Rq | human→auto | 1.480 | 2.001 | 0.554 | 0.574 | 0.957 |
| FSMI | human→human | 1.316 | 2.066 | 0.281 | 0.430 | 0.870 |
| FSMI | auto→auto | 1.625 | 2.293 | -0.130 | -0.208 | 0.870 |
| FSMI | human→auto | 1.516 | 2.174 | 0.204 | 0.221 | 0.870 |

### 配对误差统计

相对 `human→human`：

| Target | 比较 | 平均 MAE 变化 (nm) | paired bootstrap 95% CI (nm) | Wilcoxon p |
|---|---|---:|---:|---:|
| Rq | auto→auto | +0.071 | [-0.108, 0.255] | 0.520 |
| Rq | human→auto | +0.014 | [-0.270, 0.286] | 0.800 |
| FSMI | auto→auto | +0.309 | [-0.035, 0.661] | 0.111 |
| FSMI | human→auto | +0.200 | [-0.092, 0.491] | 0.119 |

23 个样品下没有任何误差变化达到显著性。应表述为“Rq 性能在自动输入下
大体保持”，不能声称自动输入提升了模型。FSMI 的相关性下降仍是重要
警告，即使 paired MAE 检验没有显著。

## Confidence 审计

人工冻结结果中 confidence 与 absolute error 有良好负相关：

- Rq：Spearman `ρ=-0.601`, `p=0.0024`
- FSMI：Spearman `ρ=-0.677`, `p=0.00039`

机器域中该关系未保留：

- auto→auto Rq：`ρ=-0.090`, `p=0.684`
- auto→auto FSMI：`ρ=-0.207`, `p=0.344`
- M12a joint confidence vs realized error：`ρ=0.121`, `p=0.582`

所以当前自动 pipeline 的 confidence 只能作为相对 support/uncertainty
指示，不能作为已经验证的误差排序器。若用于论文或 UI，必须标注这一
限制；建议在新采集的 prospective 自动输入数据上重新校准。

## M12a 机器域生成结果

对机器数据完成了完整 23-fold LOO 图像生成。每个 outer fold 均重新拟合
22 growth 的条件描述符、岛屿统计和非检索式频谱先验，然后生成 held
growth 的 AFM 高度场。

| 图像/形貌指标（M12a 中位数） | 人工域 | 机器域 |
|---|---:|---:|
| condition descriptor MAE (z) | 0.995 | 0.972 |
| sharpness ratio | 0.714 | 0.713 |
| AFM texture-gate pass fraction | 0.609 | 0.609 |
| island feature MAE (z) | 1.502 | 1.533 |
| AFM-likeness percentile | 8.696 | 8.696 |
| physical PSD distance | 1.119 | 1.104 |
| generated FSMI absolute error (nm) | 0.873 | 0.943 |

因此，自动输入造成的主要问题不是 M12a 纹理生成器“变平”，而是：

- Rq 头仍压缩高粗糙度范围；
- FSMI 的手工物理特征关系发生域偏移；
- confidence 校准没有迁移。

## 同权重配对敏感性（不是 held-out）

另做了一项诊断：加载 UI 当前同一个 all-23 部署 bundle，对每个样品
分别输入 human clip 和 auto clip，使用相同随机种子生成。

- Rq 输入替换导致的中位绝对变化：0.333 nm
- FSMI 输入替换导致的中位绝对变化：0.480 nm
- 标准化生成图 L1 中位数：1.102

这项结果只用于证明 UI/Standalone 偏差确实可以由输入选择造成。因为
all-23 部署 bundle 已见过每个 growth 的人工训练行，它**不是 held-out
泛化证据**。

最大 Rq 输入敏感样品包括 6029、6101、6048、6022。6029 同时有较大的
周期相位残差（13.5 frames）和较低的人工 ROI 覆盖（0.874），属于明确
的自动选择改进对象。6101 即使相位和几何覆盖良好仍有大偏移，说明仅靠
frame delta/ROI IoU 不能完全描述曝光、亮斑分布和时序内容的域差。

## 6056 核查

| 项目 | 结果 |
|---|---:|
| 人工 / 机器 keyframe | 161 / 160 |
| 相位残差 | 1 frame |
| ROI IoU / 人工 ROI 覆盖 | 0.805 / 0.925 |
| 真 Rq | 3.225 nm |
| human→human strict LOO | 3.255 nm |
| auto→auto strict LOO | 3.277 nm |
| human→auto strict LOO | 3.088 nm |
| 同权重 all-23 自动输入 | 3.048 nm |
| 同权重自动输入模型 confidence | 90.4% |

这解释了之前 UI 的异常：旧 UI 的 frame 203/错误 compact ROI 才产生
1.23 nm 和 4% confidence；修复后使用 frame 160 与完整点阵 ROI，结果
回到受支持范围。

## 结果与数据路径

- 机器数据集：
  `outputs/rheed_manual_vs_auto_selection/20260729_m14i_human_vs_auto_full23_v1/machine_dataset/`
- 三协议指标：
  `protocol_metrics.csv`
- 所有逐样品预测：
  `paired_target_predictions.csv`
- paired bootstrap/Wilcoxon：
  `paired_error_statistics.csv`
- 同权重输入敏感性：
  `same_weights_deployment_sensitivity.csv`
- 自动域完整 LOO 生成：
  `auto_input_generation/full23_loo/`
- 主图：
  `figures/`
- 自动域完整生成图：
  `auto_input_generation/full23_loo/figures/`

重点图：

1. `Fig1_target_protocol_comparison`：三协议 Rq/FSMI scatter。
2. `Fig2_rq_ordered_protocol_comparison`：按真实 Rq 排序的 23 样品。
3. `Fig3_input_shift_diagnostics`：相位/ROI 与预测偏移。
4. `Fig4_model_input_atlas_page_01..04`：全部 23 个人工/机器模型输入。
5. `Fig5_same_weights_deployment_atlas_page_01..05`：同权重输入和生成图。
6. `auto_input_generation/.../Fig1a..e_full23_loo_atlas`：机器域严格 LOO
   RHEED→生成 AFM→真实 AFM 全样品图册。

## 复现

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_manual_vs_auto_selection.dataset \
  --config configs/rheed_manual_vs_auto_selection.json \
  --device mps

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_manual_vs_auto_selection.comparison \
  --config configs/rheed_manual_vs_auto_selection.json

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_manual_vs_auto_selection.statistics \
  --config configs/rheed_manual_vs_auto_selection.json

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_manual_vs_auto_generation.json \
  --device mps

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.visualization \
  --config configs/rheed_manual_vs_auto_generation.json

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_manual_vs_auto_selection.deployment_pair \
  --config configs/rheed_manual_vs_auto_selection.json \
  --device cpu
```

## 科学边界与下一步

- 这是 retrospective LOO；M14i/M12a 和自动 selector 都已在历史数据上
  开发，不是 prospective untouched test。
- 只有 23 growth，配对置信区间较宽。
- 自动 selector 的标注训练与 AFM LOO 是不同问题：AFM held target 没有
  泄漏，但自动选择器本身使用过历史 RHEED 人工标注。
- 当前最合理的实时版本是保留自动 ROI/keyframe，并将 Rq 作为主要输出；
  FSMI 与 confidence 在 prospective 自动输入数据上重新校准前应明确标注
  “experimental”。
- 下一轮方法改进应优先做 paired human/auto augmentation、输入域不变
  表征或 fold-local CORAL/feature normalization，并预先冻结评估方案；
  不应根据这 23 个结果反复挑选后再把同一 LOO 当作新测试证据。
