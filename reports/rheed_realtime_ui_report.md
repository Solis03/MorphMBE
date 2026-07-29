# RHEED→AFM 模拟实时监测界面：实现与验证报告

日期：2026-07-29
分支：`codex/rheed-realtime-morphology-ui-20260729`

## 结论

已经在当前仓库完成一套可运行的 PySide6 桌面界面。它从只读 raw RHEED 视频
开始，依次执行自动完整点阵 ROI、DINOv2-S 旋转顶点筛选、严格的
causal-8/selected-16 时序构造、M14i Rq/FSMI 预测、误差相关置信度和 M12a
岛屿/台阶 AFM 生成，并实时显示视频、AFM、指标卡、置信度着色粗糙度曲线和
管线日志。

最终方法仍是真生成：

- `retrieval_at_inference = false`；
- `measured_afm_patch_at_inference = false`；
- 每个事件由随机 capture zones、岛屿统计和条件频谱生成新的 128×128
  单位-Rq 形貌，再按预测 Rq 转为 nm 高度场。

## 模型与数据协议

部署拟合核对并复用桌面 standalone 中冻结的：

- `MorphMBE-M14i-Full23-OODAware-v1`：
  - Rq = M14g 60% 可解释 RHEED + 40% causal R3D；
  - FSMI = M14b RHEED-density weighted；
- `MorphMBE-M12a-Strict15-RangeTerrace-v1`：
  - selected-16 R3D 形貌条件；
  - M12a edge-preserving terrace renderer。

部署缓存在 23 个允许 growth 上全拟合。`removelist.txt` 的编号不会进入模型
训练队列或 UI 视频目录。历史样品回放是部署工程演示，不是新的 held-out
评估；论文性能应继续引用已经冻结的 LOO 结果。

## 6063 端到端证据

输入：

```text
data/raw/raw_RHEED/N6063/rampdown to 300C.MOV
```

结果：

| 项目 | 结果 |
|---|---:|
| 原始帧数 | 813 |
| 自动旋转周期 | 27 frames |
| V5 全局 smoke 关键帧 | 189 |
| 人工参考关键帧 | 约 186 |
| tracking ROI | `(414, 258, 450, 696)` |
| 完整点阵 ROI | `(474, 54, 588, 984)` |
| 质量门控后可用事件 | 20 |
| Rq | 5.2082 nm |
| 生成图实测 Rq | 5.2081 nm |
| FSMI | 3.7834 nm |
| 综合 confidence | 48.7% |
| 单事件推理 | 3.94 s |
| ROI→关键帧→AFM 总 smoke | 31.58 s |

生成图具有清楚的随机岛屿、台阶、沟槽与高度起伏，而不是检索图或光滑平面。

## 失败分析与修正

真实 UI 回放发现，某些次优旋转周期虽通过亮斑门控，但 R3D 表征超出 23
样品训练支持，标量头会外推到接近 0 nm。该事件的 confidence 约 5%，说明
不确定性方向正确；但直接以接近零的幅值画 AFM 会产生误导性的平面。

最终实现加入显式支持范围约束：

- 原始外推保存在 `unconstrained_*`；
- 显示和生成幅值约束到训练目标最小/最大值；
- `*_support_clipped=true`；
- confidence 再乘 0.5；
- 区间仍随 OOD 风险扩大。

6063 frame 266 的复查：

| 项目 | 修正前 | 修正后 |
|---|---:|---:|
| unconstrained Rq | 2.1e-9 nm | 保留 |
| 用于生成的 Rq | 2.1e-9 nm | 1.2270 nm |
| 综合 confidence | 5.2% | 3.3% |
| 生成图 Rq | 近似 0 | 1.2270 nm |

因此界面仍清楚显示“模型没把握”，同时不再把无支持外推渲染成没有 AFM
纹理的平面。

## 性能与实时性

- M1 Pro，PyTorch 2.12，MPS 可用；
- R3D-18 在本机 CPU 比 MPS 略快，并可逐位复现冻结 embedding；
- 全 23 部署缓存构建约 11.3 秒；
- 6063 V5/V7 全视频分析约 27.6 秒；
- 单事件双 R3D + M14i + M12a 约 3.8–4.3 秒；
- 默认把视频时长放慢到 1.67×；
- 视频和模型分别在线程运行，推理队列最大一个待处理事件；
- 队列满时明确跳过周期，避免回放延迟无限增长。

本地机器足以完成当前模拟实时任务，不满足 CUDA handoff 条件。

## 验证

- realtime + automatic-selector tests：13/13 pass；
- M14i/M12a/full-cohort/island adjacent tests：16/16 pass；
- `uv sync` 后安装式命令 `rheed2morph-realtime-ui --help` 正常；
- Qt offscreen 初始界面和 6063 实时预测界面均完成 1540×980 渲染；
- `data/raw` 当日修改文件数为 0；
- desktop standalone 未写入任何本任务产物。

## 文件

- UI：[ui.py](../src/rheed2morph/realtime/ui.py)
- 后台调度：[workers.py](../src/rheed2morph/realtime/workers.py)
- M14i/M12a 部署：[model.py](../src/rheed2morph/realtime/model.py)
- V5/V7 replay 分析：[selector.py](../src/rheed2morph/realtime/selector.py)
- 时序构造：[clips.py](../src/rheed2morph/realtime/clips.py)
- 配置：[rheed_realtime_ui.json](../configs/rheed_realtime_ui.json)
- 用户说明：[RHEED_REALTIME_MORPHOLOGY_UI.md](../docs/RHEED_REALTIME_MORPHOLOGY_UI.md)
- 启动脚本：[run_rheed_realtime_ui.py](../scripts/run_rheed_realtime_ui.py)
- headless smoke：[smoke_rheed_realtime_pipeline.py](../scripts/smoke_rheed_realtime_pipeline.py)

## 图

- 完整 UI：`reports/rheed_realtime_ui/figures/ui_6063_live_prediction.png`
- RHEED/ROI/生成 AFM：
  `reports/rheed_realtime_ui/figures/rheed_to_generated_afm_panel.png`
- 同一三联图 PDF：
  `reports/rheed_realtime_ui/figures/rheed_to_generated_afm_panel.pdf`

## 限制与下一步

1. 当前 playback 前的 V5 分析使用完整视频，是模拟实时，不是完全因果的相机
   选择器。
2. 真正工业相机接入需实现在线滑动轨迹、相机 SDK 适配和硬件时间戳。
3. 约 4 秒/预测适合低频工艺监测；若要求每个约 0.9 秒旋转周期都预测，应采用
   R3D 蒸馏/缓存增量特征或 CUDA 服务。
4. 首个前瞻新样品应在 AFM 测量前登记，作为真正的 prospective validation。
