# 最终方法学澄清

## 结论

当前可作为论文 strict OOF 证据的主线不是 `AFM -> AFM decoder`，也不是 `RHEED -> neural decoder -> AFM pixels`。最终应表述为两条相连但不同的路线：

1. **RHEED -> Rq 定量预测**：RHEED keyframe PNG 经过手工 ROI、灰度化、224 x 224 resize/pad、ImageNet normalization，进入冻结 DINOv2 ViT-S/14。Phase2A 缓存的单帧 CLS 为 384 维，但实际回归输入是 `mean/std/delta/slope` temporal aggregate，因此是 1536 维。freeze 的 full-cohort 定量模型是 `full_cohort_top5_median_ridge_ensemble`，由 5 个 ridge member 组成，在 Rq nm 空间取 median。

2. **RHEED-conditioned representative AFM retrieval**：strict visual 路线使用 Phase7A/Phase7B 的固定 `A3` 检索。输入是预测 Rq、预测 AFM descriptors 和 prototype probabilities；candidate bank 是历史 AFM representative maps。strict OOF 时每个 held-out growth group 被排除，所以每折有 22 个 source group。A3 选出一个历史 AFM source morphology，将它投影到 unit-Rq，再用 q10/q50/q90 Rq 重新缩放成三张 representative AFM。

## 必须避免的误述

- 不要说 final visual result 是 Phase3A AFM autoencoder。
- 不要说 final visual result 是 RHEED 直接 decoder 出 AFM pixels。
- 不要把 Phase7A 的 per-sample mixed-best atlas 当成固定方法图。
- main architecture figure 应画 fixed A3 retrieval。
- strict OOF 和 full-cohort future deployment 必须分开。

## q10/q50/q90

strict OOF 中，q50 是 Phase6A strict OOF 的 predicted Rq。q10/q90 来自该 fold training samples 的 absolute-error 分布：`q10 = max(0.001, pred - err90)`，`q90 = max(q10+0.001, pred + err90)`。三张 AFM 图共享同一个 selected source morphology，只改变 Rq amplitude。

## 部署包差异

freeze 的 registry 写的是 A3 full-cohort retrieval，但 `13_UNSEEN_INFERENCE/predict_unseen_batch.py` 当前实际使用 deterministic placeholder embedding，并按 `abs(source_rq - pred)` 选 source scan。这不是 strict OOF 的完整 DINO -> descriptor -> A3 路线，应标为 technical smoke implementation。
