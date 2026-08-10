# MorphMBE M20 + M22c dense-intermediate AFM standalone

Frozen on **2026-08-10** from M22 research commit `94f20d0` and the dedicated
M22 UI packaging branch. The active scalar model is the M20 target-blind
spot-connectivity Sq upgrade; the active AFM generator is
`M22c_gap_completion_strong`. The desktop M17 standalone was used read-only as
the runtime/data template and was not modified.

## 打开 UI

```bash
cd /Users/ziyi/Desktop/LAB/code/standalone/MorphMBE_M22_DenseMid_UI_Standalone_20260810
./scripts/run_m22_standalone.sh run-ui
```

UI 顶部徽标必须显示 **M20 + M22c | READY**。

## 命令行预测一段 RHEED 视频

```bash
./scripts/run_m22_standalone.sh predict-video \
  "data/raw/raw_RHEED/N6063/rampdown to 300C.MOV" \
  6063 \
  reproduced_outputs/my_6063_prediction
```

结果包含 `result.json`、`prediction.npz`、PNG 和 PDF 对比图。

## 验证与结果索引

```bash
./scripts/run_m22_standalone.sh validate
./scripts/run_m22_standalone.sh list-visualizations
./scripts/run_m22_standalone.sh verify-checksums
./scripts/run_m22_standalone.sh test
./scripts/run_m22_standalone.sh smoke-model-6063
```

详细命令见 `docs/M22_STANDALONE_RUNBOOK.md`，逐页可视化说明见
`docs/M22_VISUALIZATION_INDEX.md`。

## Active model boundary

- Sq: M20 spot-connectivity calibrated rough-tail head.
- AFM: M22c dense-intermediate largest-gap completion and coalescence.
- Cohort: 27 growths; 6081 excluded.
- AFM metrology: sample-median areal Sq after order-3 independent fast-scan
  line flattening.
- No measured query AFM, AFM retrieval, or nearest-image copying at inference.
- M22 is retrospective method-development evidence, not prospective validation.

The selected strict outer-LOO M22 result has Sq MAE `0.685 nm`, RMSE
`0.829 nm`, and Pearson `r=0.923`. In the measured Sq 3.5-6.0 nm subset, the
generated dark fraction is `0.0334` versus `0.0324` measured.
