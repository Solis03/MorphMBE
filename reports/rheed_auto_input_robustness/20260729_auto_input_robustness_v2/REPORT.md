# M15b 自动 RHEED 输入：FSMI 与角覆盖置信度稳健性研究

实验编号：`20260729_auto_input_robustness_v2`
队列：23 个允许 growth，排除 `6043` 与 `6055`
最终标量模型：`MorphMBE-M15b-AutoR3D-AngularTTA`
图像生成器：冻结的非检索式 `M12a-RangeTerrace`

## 结论

本轮定位并修复了两个相互独立的问题：

1. **FSMI 下降不是帧数不一致，也不是机器关键帧本身失效。**人工和机器缓存
   都严格使用 `keyframe_1=k`、`causal_8=k-7..k`、
   `selected_16=k-7..k+8`。主要失效来自 V8 完整点阵 ROI 改变了手工
   connected-component、skeleton、difference-area 等特征的统计域。
2. **相邻帧数相同，不代表物理相位覆盖相同。**视频估计旋转周期约为
   25–40 帧；固定 causal-8 在慢周期视频中覆盖的转角更小。旧 confidence
   既看不到这种角覆盖差异，也看不到关键帧/ROI 小扰动下的局部预测跳变。

最终 M15b 使用自动输入域 causal-8 R3D 作为 Rq 和 FSMI 的点预测头，并使用：

- 11-view 关键帧/ROI test-time perturbation centrality；
- RHEED 轨迹估计的 rotation-period/angular-coverage risk；
- temporal R3D 与 physics 头的严格嵌套极端 disagreement 诊断；
- 完全不读取 query AFM 的经验风险、误差与区间校准。

在 23 个 outer AFM-target held folds 上：

| Target | MAE (nm) | RMSE (nm) | Pearson r | Spearman ρ | confidence–\|error\| ρ | p | 90% interval coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rq | 1.212 | 1.577 | 0.757 | 0.561 | **-0.538** | **0.0081** | 0.870 |
| FSMI | 1.036 | 1.419 | 0.748 | 0.552 | **-0.710** | **0.00015** | 0.870 |

这是对之前 automatic-input FSMI `MAE=1.625 nm, r=-0.130` 的明显修复。23 个
样品全部保留，失败样品没有从队列中删除。

## 1. 验证边界

对每个 outer held growth：

- 该 growth 的 AFM Rq/FSMI 不进入点预测训练；
- 该 growth 的 AFM target 不进入 confidence 或 interval 校准；
- scalar head 在其余 22 个 growth 上拟合；
- calibration 使用 inner leave-one-growth-out 结果；
- growth ID 是泄漏边界；
- `removelist.txt` 对应样品不进入队列；
- raw RHEED/AFM 与桌面 Standalone 均只读。

自动 ROI/关键帧选择器是本实验之前冻结的无 AFM-target 预处理器。它自身的
关键帧能力已有 25-video leave-one-video-out 验证；本报告的“strict LOO”
特指 AFM scalar target 边界，而不是把整个预处理器在 23 个 outer fold 内
重新训练。发表时应明确这一区别。M15b 的角覆盖组合和 95th-percentile
conflict 诊断是本队列上的探索性方法选择，仍需要未来前瞻性 growth 作最终
确认。

UI 中历史 23 样品使用 all-23 deployment refit，仅用于工程演示；它不是新的
held-out 证据。本报告中的 CSV/图才是论文性能证据。

## 2. 帧数与时序审计

人工和机器两套 23-growth 缓存逐文件检查结果：

- `keyframe_1`：1 帧，索引 `k`；
- `causal_8`：8 帧，索引 `k-7..k`；
- `selected_16`：16 帧，索引 `k-7..k+8`；
- 23/23 clip 内帧索引连续；
- 23/23 的关键帧位于 selected-16 零基下标 7；
- UI 在收到 `k+8` 后才发起推理。

因此没有发现前后相邻帧数不一致、人工/机器 off-by-one，或未来帧偷看。

但审计也发现固定 8 帧在不同视频中并不覆盖固定旋转角：估计周期从约 25 帧到
40 帧，causal-8 对应约 18%–28% 的一个旋转周期。这个物理相位覆盖差异成为
M15b confidence 的独立、target-blind 输入。

## 3. FSMI 为什么下降

### 3.1 frame × ROI 因子实验

使用相同的 physics-only M14b 算法，把关键帧来源和 ROI 来源独立交换：

| Target | Frame | ROI | MAE (nm) | Pearson r | Spearman ρ |
|---|---|---|---:|---:|---:|
| FSMI | human | human | 1.316 | 0.281 | 0.430 |
| FSMI | machine | V8 machine | 1.625 | -0.130 | -0.208 |
| FSMI | machine | human | **1.225** | **0.544** | **0.475** |
| FSMI | human | V8 machine | 13.476 | -0.239 | -0.139 |
| Rq | human | human | 1.496 | 0.372 | 0.474 |
| Rq | machine | V8 machine | 1.793 | -0.065 | -0.101 |
| Rq | machine | human | **1.366** | **0.630** | **0.660** |
| Rq | human | V8 machine | 23.222 | -0.233 | -0.275 |

机器关键帧配合人工 ROI 反而优于 human/human；一旦换成 V8 ROI，physics
head 排序崩溃，甚至产生严重外推。说明：

- V5/V8 的关键帧位置保留了足够形貌信息；
- 完整点阵 V8 ROI 对 R3D/图像生成是合理的；
- 但基于图像百分位阈值的手工物理特征不具备 crop-invariance；
- 背景占比、resize-and-pad 比例、右侧明暗交界与上下空白都会改变二值连通域、
  skeleton、largest-component 和 temporal-difference 的数值。

完整因子表见
[`frame_roi_factorial_audit.csv`](frame_roi_factorial_audit.csv)。

### 3.2 Q50 专用 physics ROI

为避免破坏用户已确认的完整点阵 ROI，增加一个只服务于手工物理特征的
orientation-conditioned median-geometry ROI：

- R3D/M12a 继续使用 V8 完整点阵 ROI；
- physics feature 使用 Q50 ROI；
- ROI 拟合不使用 AFM target；
- 严格 leave-one-video-out ROI 校准；
- held ROI annotation 不进入该 fold；
- 对人工 ROI 的中位 IoU 为 0.768，中位 coverage 为 0.895。

Q50 physics-only FSMI 改善到 `MAE=1.359 nm, r=0.455`，Rq 改善到
`MAE=1.540 nm, r=0.557`。这验证了根因，但仍弱于时序 R3D，所以 Q50 被保留
为独立诊断头和 disagreement 证据，不控制最终点预测。

### 3.3 最终点预测头

候选包括 physics-only、physics/R3D 混合和 causal R3D。对两个 target 的每个
outer fold，都只用其余 growth 的 inner CV 选择候选。结果是：

- Rq：23/23 folds 选择 `M14d_r3d_causal_temporal`；
- FSMI：23/23 folds 选择 `M14d_r3d_causal_temporal`。

这说明自动输入并没有丢失 FSMI 信息；问题是原先沿用人工域固定 mapping，
错误地把 FSMI 交给了对 ROI 域漂移敏感的 physics head。

| Target | Model | MAE (nm) | Pearson r | Spearman ρ |
|---|---|---:|---:|---:|
| Rq | old M14b physics | 1.793 | -0.065 | -0.101 |
| Rq | old M14g mixed | 1.536 | 0.536 | 0.469 |
| Rq | **M15b causal R3D** | **1.212** | **0.757** | **0.561** |
| FSMI | old M14b physics | 1.625 | -0.130 | -0.208 |
| FSMI | old M14g mixed | 1.349 | 0.505 | 0.397 |
| FSMI | **M15b causal R3D** | **1.036** | **0.748** | **0.552** |

## 4. Confidence 为什么失效、如何修复

### 4.1 旧方法的盲点

旧 confidence 主要包含：

- query R3D embedding 相对训练样品的 density/OOD；
- 目标幅值是否向训练范围上方外推。

它没有直接测试：

- 自动关键帧若偏移 1–2 帧，预测是否变化；
- 自动 ROI 若偏移或缩放少量，预测是否变化；
- temporal 与 physics 表征是否给出互相矛盾的判断。

因此它对真实误差的排序较弱：

| Target | old confidence–\|error\| ρ | p |
|---|---:|---:|
| Rq | -0.150 | 0.496 |
| FSMI | -0.137 | 0.533 |

### 4.2 11-view target-blind TTA

对同一确认关键帧构造 11 个 causal-8 视图：

1. base：`k-7..k`；
2. 结束在 `k-2, k-1, k+1, k+2` 的相邻时序窗口；
3. ROI 左/右/上/下平移 3%；
4. ROI 紧缩/放宽 6%。

这些扰动固定、可解释、在读取 query AFM 之前生成。每个 inner/outer fold 都
使用其自己的训练集重拟合 R3D 与 range calibration。

简单 TTA variance 几乎不相关，因为 6057 是“所有视图稳定地错”。更有效的
量是 base prediction 到 11-view median 的距离：它检查正式输入是否恰好落在
局部预测跃迁的一侧。

### 4.3 旋转角覆盖风险

每个 outer fold 仅用 RHEED 亮斑轨迹估计周期，并相对于其 22 个训练 growth
计算 smoothed empirical period risk。最终输入风险为：

```text
angular_TTA_risk = sqrt(TTA_centrality_risk * rotation_period_risk)
confidence = 1 - angular_TTA_risk
```

这里用风险几何平均是为了要求两个因素共同提供证据：长周期本身不必然失败，
局部 TTA 波动本身也不必然失败；“相位覆盖不足 + 对选择敏感”同时出现时才
明显降低 confidence。两个风险及其 reference 都在每个 outer fold 内重新
计算，不读取 held AFM。

### 4.4 Multi-head disagreement 的负结果与保留用途

最初的 post-hoc 版本曾让 6057 的 confidence 降至 2.2%，但独立审查发现其
全局参考诊断行可能间接使用 outer-held target。该版本已移入
`superseded_posthoc_confidence/`，明确禁止作为最终证据。

完全严格嵌套后，6057 的 head agreement 仍很低（Rq 10.9%、FSMI 6.5%），但
它不是每个 inner reference 中唯一超过 95th percentile 的样品。强行把阈值
调高虽然会压低 6057，也会错误惩罚 6063/6101 等预测准确样品，并使 Rq
confidence–error 关系变差。因此最终 confidence 不为单个失败例事后调参；
head agreement 单独显示为 conflict alert，只在真正超过严格 inner 95th
percentile 时 veto。

confidence 是经验可靠性指数，不是正确概率。

### 4.5 置信度消融

| Target | Method | confidence–\|error\| ρ | p | AURC (nm) | risk at 50% coverage (nm) |
|---|---|---:|---:|---:|---:|
| Rq | old density/amplitude | -0.150 | 0.496 | 1.122 | 1.144 |
| Rq | TTA variance only | -0.019 | 0.932 | 1.231 | 1.077 |
| Rq | M15a TTA centrality | -0.271 | 0.210 | 0.890 | 0.939 |
| Rq | period only | -0.216 | 0.323 | 1.031 | 1.153 |
| Rq | **M15b angular × TTA** | **-0.538** | **0.0081** | **0.817** | **0.753** |
| FSMI | old density/amplitude | -0.137 | 0.533 | 0.927 | 0.845 |
| FSMI | TTA variance only | -0.056 | 0.799 | 0.996 | 0.930 |
| FSMI | M15a TTA centrality | -0.399 | 0.059 | 0.715 | 0.729 |
| FSMI | period only | -0.235 | 0.280 | 0.840 | 1.026 |
| FSMI | **M15b angular × TTA** | **-0.710** | **0.00015** | **0.602** | **0.666** |

低 AURC 表示先保留高 confidence 样品时，累计误差更低。

代表性 held folds：

| Growth | Target | True | Predicted | Absolute error | Confidence |
|---|---|---:|---:|---:|---:|
| 6056 | Rq | 3.225 | 2.608 | 0.617 | 69.0% |
| 6056 | FSMI | 2.664 | 2.233 | 0.431 | 64.9% |
| 6057 | Rq | 5.337 | 1.422 | 3.915 | 48.8% |
| 6057 | FSMI | 4.205 | 1.081 | 3.124 | 56.1% |
| 6099 | Rq | 10.321 | 6.956 | 3.365 | **37.9%** |
| 6099 | FSMI | 9.324 | 5.225 | 4.099 | **39.3%** |

6099 在两个 target 上同时得到低 confidence。6057 的最终 reliability 处于中低
区间，而不是人为压到接近 0；它同时显示 10.9%/6.5% 的 head-agreement
警报。这个呈现比为一个已知失败例调阈值更严格，也清楚告诉用户：总体
confidence 有显著误差排序能力，但并不能识别每一个失败。

## 5. 文献依据

本轮方法选择与以下工作一致：

- [Deep Ensembles](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)：
  不同模型/表征之间的分歧可作为 epistemic uncertainty 信号。
- [Test-time augmentation uncertainty for image segmentation](https://arxiv.org/abs/1807.07356)：
  对合理输入扰动的预测分布可反映输入和预处理不确定性。
- [TTA with post-hoc calibration](https://ojs.aaai.org/index.php/AAAI/article/view/26735)：
  TTA dispersion 需要结合 calibration，不能直接当作正确概率。
- [Conformal prediction under covariate shift](https://arxiv.org/abs/1904.06019)：
  分布漂移下应显式考虑 calibration 和覆盖率，而非只报告点误差。
- [Deep Evidential Regression](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html)：
  回归的不确定性应区分数据噪声与模型知识不足。

由于本项目只有 23 个 growth，本轮没有引入高参数 evidential neural head；
选择了无需额外 target 参数、可审计的 rotation-angular coverage + TTA，
并把 representation conflict 保留为独立告警。

## 6. 代码、数据与图

核心代码：

- `analysis/rheed_auto_input_robustness/perturbation.py`
- `analysis/rheed_auto_input_robustness/confidence.py`
- `analysis/rheed_auto_input_robustness/physics_roi.py`
- `analysis/rheed_auto_input_robustness/run.py`
- `src/rheed2morph/realtime/clips.py`
- `src/rheed2morph/realtime/model.py`
- `src/rheed2morph/realtime/selector.py`
- `src/rheed2morph/realtime/workers.py`

配置：

- `configs/rheed_auto_input_robustness.json`
- `configs/rheed_realtime_ui.json`

派生实验输入与拟合 artifact：

- `outputs/rheed_auto_input_robustness/20260729_auto_input_robustness_v2/`
- `outputs/rheed_realtime_ui/morphmbe_m15b_m12a_auto_live_v3.joblib`
- `outputs/rheed_realtime_ui/morphmbe_m15b_m12a_auto_live_v3_manifest.json`

主要结果表：

- [`m15b_strict_loo_predictions.csv`](m15b_strict_loo_predictions.csv)
- [`m15b_metrics.csv`](m15b_metrics.csv)
- [`m15a_tta_centrality_ablation_predictions.csv`](m15a_tta_centrality_ablation_predictions.csv)
- [`baseline_vs_final_metrics.csv`](baseline_vs_final_metrics.csv)
- [`confidence_method_ablation.csv`](confidence_method_ablation.csv)
- [`physics_roi_target_metrics.csv`](physics_roi_target_metrics.csv)

论文图：

- [`Fig1_m15b_target_predictions.pdf`](figures/Fig1_m15b_target_predictions.pdf)
- [`Fig2_confidence_vs_error.pdf`](figures/Fig2_confidence_vs_error.pdf)
- [`Fig3_risk_coverage.pdf`](figures/Fig3_risk_coverage.pdf)
- [`Fig4_target_head_ablation.pdf`](figures/Fig4_target_head_ablation.pdf)
- [`Fig5_physics_roi_ablation.pdf`](figures/Fig5_physics_roi_ablation.pdf)
- [`Fig6_all23_ordered_predictions.pdf`](figures/Fig6_all23_ordered_predictions.pdf)

## 7. 原始视频端到端 smoke

用 `data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV` 只读运行
与 ReplayWorker 相同的 18-frame/TTA 路径：

- 自动关键帧：160；人工参考：161；
- V8 R3D/生成 ROI：`(594,114,648,888)`；
- Q50 physics diagnostic ROI：`(666,174,534,798)`；
- Rq：2.687 nm；
- FSMI：2.324 nm；
- Rq/FSMI reliability confidence：均为 61.5%；
- M12a 生成图实测 Rq：2.687 nm；
- retrieval at inference：false；
- 11-view M1 Pro 推理：7.14 s；
- selector + inference：28.95 s。

结果与图位于
`outputs/rheed_realtime_ui/20260729_6056_m15b_auto_robustness_v3/`。

这里是 all-23 deployment refit，不是 6056 held-out 证据；6056 的严格 held
结果仍以第 4.5 节的 2.608/2.233 nm 与 69.0%/64.9% 为准。

## 8. 局限与下一步

1. 23 个 growth 仍然很少；显著相关不等于 confidence 已成为真实概率。
2. angular-coverage 几何风险与 95th-percentile conflict 诊断是当前队列上的
   探索性设计，应预注册并在新 growth 上前瞻验证。
3. 高 Rq/FSMI 端只有少量样品，6099 仍明显低估；低 confidence 正确暴露了
   风险，但没有消除点预测误差。
4. Q50 physics ROI 证明 ROI 域偏移可修复，但手工特征仍不够 crop-invariant。
   后续可训练显式 ROI-invariant descriptor encoder，或在训练时加入 crop
   augmentation 与 consistency loss。
5. M15b 改善的是自动输入标量端与可靠性。AFM 图像端仍是已冻结的 M12a
   真生成模型，没有读取最近邻 AFM 或训练 patch；本轮没有把历史 UI replay
   当作新的 held-out 图像证据。
6. 当前 M1 Pro 单事件要计算 11 个 R3D view，实测约 6.3–7.2 秒。仍远低于
   CUDA handoff 的 30 分钟阈值；实时工业部署可通过 view batching、embedding
   cache 或轻量 student encoder 优化。

当前本地机器足以完成验证，无需 CUDA handoff。
