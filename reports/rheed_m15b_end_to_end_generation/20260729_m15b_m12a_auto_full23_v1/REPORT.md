# M15b 自动 RHEED 视频输入 → M12a 生成 AFM 端到端验证

实验编号：`20260729_m15b_m12a_auto_full23_v1`

## 结论

本版本已经把改进后的自动 ROI/关键帧输入、M15b Rq/FSMI 预测和 M12a
非检索式 AFM 生成器连接成一条完整链路。严格 23-fold leave-one-growth-out
（LOO）中，每个 held growth 的标量目标、形貌条件和生成器拟合均不使用该
growth；每折用其余 22 个 growth 拟合。

与此前的 M14i 自动输入版本相比，M15b 明显恢复了 Rq 和 FSMI 的排序与幅度
关系。生成图保持 M12a 的岛屿/台阶纹理，并严格缩放到 held-fold 预测的 Rq。
这是一种条件随机形貌生成，不是像素配准的 AFM 重建，也不是最近邻检索。

## 模型

完整模型名：

`MorphMBE-M15b-AutoR3D-AngularTTA + M12a-RangeTerrace`

- 自动输入：V5 DINOv2-S 选择可见旋转周期，V8 完整点阵 ROI 在周期内细化
  顶点。
- 标量头：causal R3D，输入 `k-7..k`，预测 Rq 和 FSMI。
- 可靠性：关键帧偏移、ROI 平移/尺度 TTA centrality 与旋转周期覆盖风险，
  外加严格 inner-fold 表征冲突诊断。
- 形貌条件：selected-16 R3D，输入 `k-7..k+8`。
- AFM 生成：M12a 条件岛屿统计、非检索式频谱先验、随机 Laguerre capture
  zones 和 edge-preserving terrace renderer。
- 输出：每个 growth 四个独立随机 AFM 高度场，`128×128`，横向视场
  `1×1 µm²`，高度单位 nm。

推理时不读取最近邻 AFM，不复制训练 AFM patch。生成器会产生多个共享预测
条件、但微观岛屿排布不同的随机 realization。

## 严格验证协议

- 队列：23 个确定可用的 growth；排除 6043、6055 以及 `removelist.txt`
  所列样本。
- 外层：每次 held out 一个完整 growth，训练集为另外 22 个 growth。
- 标量：直接读取 M15b 严格 outer-LOO 预测；置信度校准不使用 held target。
- 生成：每个 held growth 的条件模型、岛屿模型与频谱模型只在其余 22 个
  growth 上拟合。
- 完整性审计：23/23 的 held growth 与生成器 fit 集合交集为 0；所有 map
  元数据与 M15b 严格预测逐项一致。
- 禁止项审计：`retrieval_at_inference=false`，
  `measured_afm_patch_used_at_inference=false`。

## 主要结果

| 协议 | Rq MAE (nm) | Rq Pearson r | FSMI MAE (nm) | FSMI Pearson r |
|---|---:|---:|---:|---:|
| 旧 M14i 自动输入 | 1.536 | 0.536 | 1.625 | -0.130 |
| **M15b 自动输入（本版本）** | **1.212** | **0.757** | **1.036** | **0.748** |

M15b 相对旧自动输入版本将 Rq MAE 降低 0.325 nm，将 FSMI MAE 降低
0.588 nm。两项目标的经验 90% 区间覆盖率均为 20/23（86.96%）。

把预测目标落实为生成图并重新测量：

| 指标 | 旧 M14i+M12a | 本版本 M15b+M12a |
|---|---:|---:|
| 生成图 Rq MAE (nm) | 1.510 | **1.197** |
| 生成图 FSMI MAE (nm) | 1.350 | **1.064** |
| median sharpness ratio | 0.713 | 0.711 |
| texture-gate pass fraction | 0.609 | 0.565 |
| mean island-feature MAE (z) | 1.668 | 1.639 |

因此，本次改进的主要收益是 RHEED 条件到物理幅度和 FSMI 的映射，而不是
更换渲染器。纹理清晰度基本保持，但 texture-gate 通过率没有提高，不能声称
形貌纹理已经全面优于旧版本。

M15b 的联合置信度与 Rq/FSMI 联合实际误差 Spearman
`rho=-0.646`（`p≈0.001`）：低置信度样品总体更容易出现较大误差。置信度
仍是相对可靠性指数，不是“预测正确的概率”。

## 可视化

所有图均同时保存 PNG 和 PDF：

- `full23_loo/figures/Fig0_m15b_end_to_end_overview.*`：四个固定 Rq
  分层样品加一个最大非重复失败样品；每行依次为自动 RHEED、生成 AFM、
  实测 AFM。
- `full23_loo/figures/Fig1a...Fig1e_full23_loo_atlas.*`：完整 23 growth
  图谱，严格按实测 Rq 固定排序，没有挑选隐藏。
- `full23_loo/figures/Fig2_full23_target_scatter.*`：Rq/FSMI 预测—实测
  散点图。
- `full23_loo/figures/Fig4_full23_rq_ordered.*`：按真实 Rq 排序的幅度趋势。
- `full23_loo/figures/Fig5_confidence_audit.*`：置信度—实际误差与区间覆盖。
- `full23_loo/figures/Fig6_renderer_roughness_strata.*`：不同粗糙度层级下的
  随机生成差异。
- `full23_loo/figures/Fig7_largest_error_cases.*`：最大误差案例，未隐藏
  6057、6099 等失败。

## UI 集成与真实视频验证

UI 配置直接加载：

`outputs/rheed_realtime_ui/morphmbe_m15b_m12a_auto_live_v3.joblib`

模型 ID 为：

`MorphMBE-M15b-AutoR3D-AngularTTA + M12a-RangeTerrace-live-v3`

6056 原始 MOV 的实际端到端回放验证：

- 自动关键帧：160（人工参考 161）；
- 完整点阵 ROI：`x=594, y=114, width=648, height=888`；
- 预测 Rq：2.6873 nm；
- 预测 FSMI：2.3243 nm；
- 模型置信度：61.5%；
- 生成图重新测量 Rq：2.6873 nm；
- 模型推理：约 7.02 s；
- 视频筛选加推理总时间：约 28.75 s；
- retrieval：false。

验证目录：

`outputs/rheed_realtime_ui/20260729_m15b_m12a_end_to_end_ui_verification_6056`

其中 `prediction.npz` 保存实际 16 帧模型输入和生成高度图，
`rheed_to_generated_afm_panel.*` 保存三联图，`ui_offscreen.png` 是实际
UI 运行截图。

## 局限与论文表述边界

1. 23 折 LOO 对每个 growth 都严格 held out，但 M12a 方法家族在更早的
   数据划分上开发过；它是回顾性严格验证，不等同于从未接触的新批次前瞻测试。
2. 生成 AFM 是与 RHEED 条件一致的随机形貌 realization，不应表述为逐像素
   还原真实 AFM。
3. 23 个 growth 仍然很少；高 Rq 尾部存在明显回归到均值，尤其 6099 和
   6057 等失败必须在论文中保留。
4. AFM-likeness 与 texture gate 显示渲染器仍有改进空间。本版本有力支持
   “物理幅度/FSMI 映射改善”，不支持“所有纹理指标全面提升”。
5. UI 缓存使用全部 23 growth 重拟合，只用于部署演示；UI 对历史样本的输出
   不能替代外层 LOO 证据。首个真正新 growth 应在 AFM 测量前预登记并保存
   session，作为前瞻验证。

## 复现

完整命令见 `COMMANDS.md`。结构化结果见：

- `full23_loo/end_to_end_manifest.json`
- `full23_loo/end_to_end_integrity_audit.csv`
- `full23_loo/baseline_vs_m15b_end_to_end.csv`
- `full23_loo/target_prediction_summary.csv`
- `full23_loo/method_summary.csv`
- `full23_loo/visualization_manifest.json`

关键成果的 SHA-256 见 `ARTIFACTS.sha256`。
