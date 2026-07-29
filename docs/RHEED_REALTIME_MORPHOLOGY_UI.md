# MorphMBE RHEED→AFM 模拟实时界面

## 1. 用途

该程序把已经冻结的自动 RHEED ROI/关键帧工具与两套论文模型连接成一条
可运行的“模拟实时”链路：

1. 从 `data/raw/raw_RHEED` 只读发现原始视频；
2. 在下拉菜单中选择样品编号和视频；
3. V5 DINOv2-S 在内部 tracking ROI 中定位旋转周期，V8 在完整点阵区域
   细化顶点并给出 R3D/生成模型输入 ROI；
4. 在顶点之后等待 8 帧，构造与冻结模型一致的时序输入；
5. M15b 自动输入域 causal R3D 输出 Rq、FSMI、经验预测区间和误差相关
   置信度；
6. M12a 随机岛屿/台阶生成器输出新的 AFM 高度场；
7. UI 实时更新 AFM 图、指标卡、置信度和粗糙度时间曲线；
8. 每次预测自动写入派生 session 目录，原始视频从不改写。

桌面冻结目录
`/Users/ziyi/Desktop/MorphMBE_RHEED_AFM_Standalone` 只用于核对模型身份和
冻结参数；程序、缓存和 session 全部位于当前仓库。

## 2. 运行

在仓库根目录执行：

```bash
uv sync
PYTHONPATH=src:. .venv/bin/python scripts/run_rheed_realtime_ui.py
```

也可以在 `uv sync` 后使用：

```bash
uv run rheed2morph-realtime-ui
```

首次运行若没有部署缓存，程序会自动用冻结的方法和 23 个允许样品完成一次
全队列部署拟合。也可预先显式生成：

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/prepare_rheed_realtime_model.py --force
```

部署缓存位于：

```text
outputs/rheed_realtime_ui/morphmbe_m15b_m12a_auto_live_v3.joblib
```

这是可重建的派生缓存，不是新的论文验证结果，也不会覆盖两个论文模型冻结。

## 3. 时序输入语义

设自动检测的顶点为帧 `k`：

- `keyframe_1`：帧 `k`；
- `causal_8`：`k-7 ... k`，用于 M15b 的 Rq 和 FSMI 时序分支；
- `selected_16`：`k-7 ... k+8`，用于 M12a 的形貌条件分支；
- 预测触发时刻为收到 `k+8` 后，因此没有偷看尚未进入模拟流的图像。

R3D 和 M12a 的全部帧都使用青色的 **V8 R3D / 生成模型 ROI** 转为亮度图，
再保持纵横比缩放、零填充到 `224×224`。粉色虚线 **Q50 物理特征 ROI**
只用于诊断性手工物理特征；M15b 最终标量头也是 R3D，因此它不会改变最终
图像输入。内部 V5 tracking ROI 只负责亮斑轨迹，V7 框只作审计；二者均不
进入 M15b/M12a。每一种矩形的角色都会写入 session provenance。

V8 的四边界来自 25 个 removelist-compliant 视频的 20%/80% 稳健分位校准。
严格 leave-one-video-out 中，held-video overlap 为 0，点阵峰覆盖率中位数
为 100%，目镜圆边侵入率为 0。它在“完整点阵覆盖”和“匹配冻结模型的人工
裁剪分布”之间做了专门的部署折中。

关键帧也不再直接等同于内部 V5 候选。程序先选最可信的旋转周期，再在一个
周期的局部邻域内按完整点阵质量细化：奖励高点状能量、纵向列对齐和低水平
展宽，排除“少数亮点很亮、但阵列不完整”的帧。6056 从 V5 候选 146 细化到
160，人工参考为 161。

为检验“前后帧数量不一致”这个假设，对人工和机器两套 23 样品缓存逐项审计：
两者都严格使用 `keyframe_1=k`、`causal_8=k-7..k` 和
`selected_16=k-7..k+8`，关键帧固定为 selected-16 的下标 7，且全部连续。
因此自动输入性能差异不是帧数或 off-by-one 错误。

## 4. 模型

标量端使用
`MorphMBE-M15b-AutoR3D-AngularTTA + M12a-RangeTerrace-live-v3`：

- Rq：自动输入域 `M14d_r3d_causal_temporal`；
- FSMI：自动输入域 `M14d_r3d_causal_temporal`；
- 在 23 个外层 held fold 内，两项目标的内层选择都 23/23 次选择该时序头；
- 论文验证使用严格 23-fold LOO；UI 部署缓存则在全部 23 个允许样品上重拟合。

图像端使用 `MorphMBE-M12a-Strict15-RangeTerrace-v1` 的
`M12a_edge_preserving_terrace` 生成器：

- R3D-18 selected-16 预测 AFM 形貌条件；
- 非检索式频谱先验；
- 随机 Laguerre capture zones；
- 岛屿、台阶、沟槽和细纹理混合；
- 最终高度场严格缩放到预测 Rq。

推理时不读取最近邻 AFM、不复制训练 AFM patch。每个事件使用不同随机种子，
因此输出是条件生成结果而非检索结果。

## 5. 置信度与支持范围

界面主卡片和时间曲线中的 confidence 是 M15b 的“误差相关可靠性指数”，
不是正确概率：

- 围绕同一自动关键帧建立 11 个 target-blind 输入视图：`k-2...k+2`
  五个 causal-8 结束位置、ROI 上下左右各 3% 平移和 ±6% 尺度；
- 主风险是 base 预测偏离 11 个视图中位数的程度，而不是简单 TTA 方差；
- 固定 8 帧在不同旋转周期下覆盖不同转角，因此加入从 RHEED 轨迹估计的
  period/angular-coverage empirical risk，并与 TTA centrality 风险取几何平均；
- 当 causal R3D 与独立物理头的 disagreement 进入训练参照最极端 5% 时，
  启用 veto，取 TTA 与 head-agreement confidence 的较小值；
- 主 confidence 是 Rq/FSMI 置信度的几何平均，因此与冻结 LOO 图表中的
  模型置信度定义一致；
- V5/V8 给出独立的输入质量；
- 指标卡小字另列包含输入质量的保守综合 confidence；
- 点的颜色从红到绿表示低到高置信度；
- 指标卡同时显示预计绝对误差和经验 90% 区间。

若标量头把新输入外推到 23 样品训练目标范围以外，程序会：

1. 保留 `unconstrained_value`；
2. 将用于显示和图像幅值的数值约束到训练支持边界；
3. 设置 `support_clipped=true`；
4. 再降低置信度。

这样低支持事件仍显示有 AFM 纹理的保守边界结果，而不会把接近 0 nm 的无支持
外推伪装成一张可信的平面 AFM。该 confidence 是小样本严格 LOO 下验证的排序
指数，不应解释为“85% 概率正确”；95th-percentile veto 仍需要未来前瞻性
样品再验证。

## 6. 实时调度

默认 `1.67× 时长` 对应把 15 秒视频约放慢到 25 秒。视频解码和 AFM 预测运行
在不同后台线程：

- 视频线程持续回放并维持 18 帧环形缓存；
- 推理线程按事件处理 base + 10 个 TTA causal-8 视图，当前 M1 Pro 实测约
  6.3–7.2 秒；
- 推理队列最多保留一个待处理事件；
- 队列满时跳过该旋转周期并在日志中明确记录，防止越积越慢；
- 当前 replay 只保留全视频中最可信的周期，并在完整点阵 ROI 内局部细化；
  阴影周期不再覆盖已经得到的高支持结果。

## 7. Session 输出

每次回放在以下目录新建一个只含派生结果的 session：

```text
outputs/rheed_realtime_ui/sessions/
  YYYYMMDD_HHMMSS_<sample>_<video>/
    session.json
    prediction_timeline.csv
    generated_afm/
      event_XXXXXX.npz
```

`session.json` 明确保存 `model_input_roi`、
`physics_feature_roi_not_generator_input`、
`internal_tracking_roi_not_model_input` 和
`conservative_audit_roi_not_model_input` 四种角色。CSV 保存时间、Rq/FSMI、
预计误差、区间、TTA/head/模型/综合置信度、推理延迟、支持约束标记和生成
文件路径。NPZ
保存真实 nm 高度场和无量纲单位-Rq 形貌。

## 8. 当前边界

- 当前版本是完整的 raw-video 模拟实时界面。为了复用已经严格验证的 V5
  “全视频中选清晰周期”能力，播放前有一次约 20–30 秒的分析通道。
- 工业相机选项已在 UI 中预留，但真正的相机部署还需把 V5 候选排序改成因果
  滑动窗口，并接入相机 SDK/帧时间戳。
- 23 样品部署缓存使用全部允许样品拟合。回放这些历史样品用于工程演示，不是
  新的 held-out 性能证据；标量能力应引用 M15b 严格 LOO 报告，图像生成能力
  则仍引用冻结 M12a 报告。
- 自动 ROI 会改变按百分位阈值计算的 connected-component/skeleton 特征域。
  Q50 专用物理 ROI 缓解了该问题，但最终 M15b 选择 R3D 时序头，避免让不稳健
  的手工特征控制 FSMI 点预测。
- 新样品第一次进入系统时才是前瞻性部署测试，应保留 session 并在看到 AFM
  之前登记样品编号。

## 9. 可重复 smoke test

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/smoke_rheed_realtime_pipeline.py \
  "data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV" \
  --sample-id 6056 \
  --output-dir outputs/rheed_realtime_ui/20260729_6056_m15b_auto_robustness_v3
```

该命令保存自动 ROI、全部保留事件、选定关键帧、多帧模型输入、生成 AFM、
标量、置信度、运行时间以及 PNG/PDF 三联图。

当前 M15b+M12a 端到端验证使用：

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/smoke_rheed_realtime_pipeline.py \
  "data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV" \
  --sample-id 6056 \
  --output-dir \
  outputs/rheed_realtime_ui/20260729_m15b_m12a_end_to_end_ui_verification_6056
```

该次运行自动选择帧 160，模型输入为 `k-7..k+8` 的 16 帧完整点阵 ROI；
输出 Rq 2.687 nm、FSMI 2.324 nm、模型置信度 61.5%，生成高度场重新测得
Rq 2.687 nm。界面截图、三联图、输入 clip 和生成高度场均保存在上述目录。

论文性能证据与部署演示必须分开解释。严格 23-fold 自动输入端到端结果、
全部 23 个样品的 RHEED→生成 AFM→真实 AFM 图谱及失败案例位于：

```text
reports/rheed_m15b_end_to_end_generation/
  20260729_m15b_m12a_auto_full23_v1/full23_loo/
```
