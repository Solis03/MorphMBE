"""Run the manual single-frame RHEED to AFM Rq experiment."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage, stats

from analysis.rheed_roughness.run import display_path, read_config, safe_float
from analysis.rheed_single_frame.connectivity_features import (
    extract_feature_table,
    extract_features_for_image,
    nuisance_feature_names,
    physics_feature_names,
)
from analysis.rheed_single_frame.data import DatasetBundle, build_dataset, make_paths, write_csv_rows, write_dataset_outputs
from analysis.rheed_single_frame.embeddings import embedding_feature_names, extract_frozen_embeddings
from analysis.rheed_single_frame.models import (
    build_model_specs,
    confidence_calibration_rows,
    evaluate_fixed_models,
    evaluate_nested_selector,
    feature_importance_from_ridge,
    final_model_comparison,
    influence_analysis,
    make_model_table,
    permutation_tests,
    sensitivity_without_6023,
    write_model_outputs,
)
from analysis.rheed_single_frame.preprocessing import PreprocessedImage, preprocess_pairs
from analysis.rheed_single_frame.removelist import audit_to_json, load_removelist_audit, write_json
from analysis.rheed_single_frame.visualization import generate_html_report, make_all_figures


warnings.filterwarnings("ignore", category=FutureWarning, module=r".*skimage.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"analysis\.rheed_single_frame\.connectivity_features")


def _smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg["outputs_dir"] = str(cfg["outputs_dir"]) + "_smoke"
    cfg["reports_dir"] = str(cfg["reports_dir"]) + "_smoke"
    cfg.setdefault("models", {})["permutation_resamples"] = 100
    cfg.setdefault("models", {})["bootstrap_ensembles"] = 8
    cfg.setdefault("models", {})["max_inner_splits"] = 3
    return cfg


def _limit_bundle_for_smoke(bundle: DatasetBundle, limit: int = 8) -> DatasetBundle:
    keep_ids = {pair.sample_id for pair in bundle.pairs[:limit]}
    return DatasetBundle(
        paths=bundle.paths,
        selections=bundle.selections,
        pairs=tuple(pair for pair in bundle.pairs if pair.sample_id in keep_ids),
        dataset_rows=tuple(row for row in bundle.dataset_rows if row.get("sample_id") in keep_ids or row.get("removelist_status") == "excluded"),
        target_rows=tuple(row for row in bundle.target_rows if row.get("sample_id") in keep_ids),
        manifest_rows=tuple(row for row in bundle.manifest_rows if row.get("sample_id") in keep_ids),
        skipped_rows=bundle.skipped_rows,
        excluded_rows=bundle.excluded_rows,
        common_scale=bundle.common_scale,
        afm_available_fields=bundle.afm_available_fields,
    )


def _numeric_feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    skip = {
        "sample_id",
        "sample_group_id",
        "growth_run_id",
        "manual_rheed_path",
        "selected_afm_scan_id",
        "selected_height_map_path",
        "rq_true_nm",
        "log_rq_true",
        "embedding_model",
        "embedding_status",
        "component_overlay_path",
        "horizontal_closing_overlay_path",
        "skeleton_overlay_path",
    }
    names: list[str] = []
    if not rows:
        return names
    for key in rows[0]:
        if key in skip:
            continue
        value = rows[0].get(key)
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        names.append(key)
    return sorted(set(names))


def print_initial_audit(
    bundle: DatasetBundle,
    removelist_payload: dict[str, Any],
    feature_plan: Sequence[str],
    model_specs: Sequence[Any],
    embedding_summary: dict[str, Any],
) -> None:
    removed = sorted(row["sample_id"] for row in bundle.excluded_rows)
    discovered = [f"{pair.sample_id}:{display_path(pair.manual_rheed_path, bundle.paths.repo_root)}" for pair in bundle.pairs]
    benchmark_table = [(spec.name, spec.family, len(spec.feature_names)) for spec in model_specs]
    print("Single-frame manual RHEED experiment audit:")
    print(f"1. Canonical removelist: {removelist_payload['absolute_path']}")
    print(f"   sha256={removelist_payload['sha256']}")
    print(f"   sample IDs={', '.join(removelist_payload['parsed_sample_ids'])}")
    print(f"2. Manually selected images discovered for modeling ({len(discovered)}):")
    for item in discovered:
        print(f"   - {item}")
    print(f"3. Samples removed before processing: {', '.join(removed) if removed else 'none'}")
    print(f"4. Final independent sample count: {len(bundle.pairs)}")
    print(f"5. Available AFM/Rq fields: {', '.join(bundle.afm_available_fields)}")
    print("6. Existing functions reused: discover_manual_rheed_images, select_representative_afm_scan, load_height_nm, render_afm_height_panel.")
    print(f"7. Pre-registered connectivity features: {', '.join(feature_plan)}")
    print("8. Model benchmark table:")
    for name, family, n_features in benchmark_table:
        print(f"   - {name} [{family}], features={n_features}")
    print("9. CV design: outer leave-one-growth-run/sample-out; inner group-aware K-fold with one-standard-error model choice.")
    print(
        "10. Confidence method: outer-training residual conformal 90% PI, bootstrap ensemble spread, "
        f"domain support distance, perturbation sensitivity. Embedding status={embedding_summary.get('status', '')}."
    )


def _apply_perturbation(image: np.ndarray, mask: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32).copy()
    if name.startswith("brightness_"):
        out = out * float(name.split("_", 1)[1])
    elif name.startswith("gamma_"):
        out = np.clip(out, 0, 1) ** float(name.split("_", 1)[1])
    elif name.startswith("translate_x_"):
        out = ndimage.shift(out, shift=(0, float(name.rsplit("_", 1)[1])), order=1, mode="nearest")
    elif name.startswith("translate_y_"):
        out = ndimage.shift(out, shift=(float(name.rsplit("_", 1)[1]), 0), order=1, mode="nearest")
    elif name.startswith("blur_"):
        out = ndimage.gaussian_filter(out, sigma=float(name.split("_", 1)[1]))
    elif name.startswith("noise_"):
        out = out + rng.normal(0.0, float(name.split("_", 1)[1]), size=out.shape)
    elif name.startswith("crop_jitter_"):
        shift = float(name.rsplit("_", 1)[1])
        out = ndimage.shift(out, shift=(shift, -shift), order=1, mode="nearest")
    out = np.clip(out, 0.0, 1.0)
    return np.where(mask, out, 0.0).astype(np.float32)


def perturbation_names(config: dict[str, Any]) -> list[str]:
    p = config.get("perturbations", {})
    names = []
    names.extend(f"brightness_{value:g}" for value in p.get("brightness_scales", [0.8, 1.2]))
    names.extend(f"gamma_{value:g}" for value in p.get("gammas", [0.85, 1.15]))
    names.extend(f"translate_x_{value:g}" for value in p.get("translations_px", [-4, 4]))
    names.extend(f"translate_y_{value:g}" for value in p.get("translations_px", [-4, 4]))
    names.extend(f"blur_{value:g}" for value in p.get("blur_sigmas", [0.7]))
    names.extend(f"noise_{value:g}" for value in p.get("noise_sigmas", [0.015]))
    names.extend(f"crop_jitter_{value:g}" for value in p.get("crop_jitter_px", [-4, 4]))
    return names


def build_perturbed_feature_rows(
    images: Sequence[PreprocessedImage],
    paths: Any,
    config: dict[str, Any],
    original_model_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(int(config.get("random_seed", 0)) + 101)
    original_by_id = {str(row["sample_id"]): row for row in original_model_rows}
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for item in images:
        rows_by_sample[item.sample_id] = []
        for name in perturbation_names(config):
            pert = _apply_perturbation(item.gray_padded, item.valid_mask, name, rng)
            if item.valid_mask.any():
                vals = pert[item.valid_mask]
                lo, hi = np.percentile(vals, [1, 99])
                norm = np.zeros_like(pert)
                norm[item.valid_mask] = np.clip((vals - lo) / max(float(hi - lo), 1e-8), 0, 1)
            else:
                norm = pert
            pert_item = PreprocessedImage(
                sample_id=f"{item.sample_id}__{name}",
                manual_rheed_path=item.manual_rheed_path,
                original_rgb=item.original_rgb,
                cropped_gray=pert,
                gray_padded=pert,
                normalized=norm,
                valid_mask=item.valid_mask,
                audit_row=item.audit_row,
            )
            result = extract_features_for_image(pert_item, paths, config)
            row = {**result.features}
            row["sample_id"] = item.sample_id
            row["perturbation"] = name
            original = original_by_id.get(item.sample_id, {})
            for key, value in original.items():
                if key.startswith("embedding_") and key not in row:
                    row[key] = value
            rows_by_sample[item.sample_id].append(row)
    return rows_by_sample


def write_report_texts(
    paths: Any,
    predictions: Sequence[dict[str, Any]],
    model_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    permutation_payload: dict[str, Any],
    removelist_payload: dict[str, Any],
    embedding_summary: dict[str, Any],
) -> None:
    best = sorted(model_rows, key=lambda row: safe_float(row.get("mae_log"), math.inf))[0] if model_rows else {}
    feature_by_id = {str(row["sample_id"]): row for row in feature_rows}
    rq = np.asarray([safe_float(row["rq_true_nm"]) for row in predictions], dtype=float)
    h = np.asarray([safe_float(feature_by_id.get(str(row["sample_id"]), {}).get("horizontal_connectivity_score"), math.nan) for row in predictions])
    iso = np.asarray([safe_float(feature_by_id.get(str(row["sample_id"]), {}).get("isolation_score"), math.nan) for row in predictions])
    err = np.asarray([safe_float(row["absolute_error_nm"]) for row in predictions], dtype=float)
    conf = np.asarray([safe_float(row["confidence_score"]) for row in predictions], dtype=float)
    h_rho = stats.spearmanr(h[np.isfinite(h)], rq[np.isfinite(h)]).statistic if np.isfinite(h).sum() >= 3 else math.nan
    iso_rho = stats.spearmanr(iso[np.isfinite(iso)], rq[np.isfinite(iso)]).statistic if np.isfinite(iso).sum() >= 3 else math.nan
    iso_conf = stats.spearmanr(iso[np.isfinite(iso)], conf[np.isfinite(iso)]).statistic if np.isfinite(iso).sum() >= 3 else math.nan
    iso_err = stats.spearmanr(iso[np.isfinite(iso)], err[np.isfinite(iso)]).statistic if np.isfinite(iso).sum() >= 3 else math.nan
    cover90 = next((row.get("empirical_coverage") for row in calibration_rows if safe_float(row.get("interval_nominal")) == 0.9), math.nan)
    largest_errors = sorted(predictions, key=lambda row: safe_float(row["absolute_error_nm"]), reverse=True)[:5]
    lines = [
        "# Single-frame manual RHEED to AFM Rq results",
        "",
        f"- Included independent samples: {len(predictions)}",
        f"- Canonical removelist hash: `{removelist_payload['sha256']}`",
        f"- Best OOF model row by log MAE: `{best.get('model_name', '')}`",
        f"- Nested selector Spearman (nm): `{permutation_payload.get('nested_spearman_nm', math.nan):.4g}`; permutation p `{permutation_payload.get('nested_spearman_permutation_p', math.nan):.4g}`",
        f"- Nested MAE improvement vs median baseline (nm): `{permutation_payload.get('mae_improvement_vs_median_nm', math.nan):.4g}`; permutation p `{permutation_payload.get('mae_improvement_permutation_p', math.nan):.4g}`",
        f"- 90% prediction interval empirical coverage: `{safe_float(cover90, math.nan):.3g}`",
        "",
        "## Scientific answers",
        "",
        f"1. Horizontal connectivity versus Rq: Spearman rho `{safe_float(h_rho, math.nan):.3g}` on the included non-removelist samples.",
        f"2. Isolation score versus Rq: Spearman rho `{safe_float(iso_rho, math.nan):.3g}`.",
        "3. Model comparison is in `outputs/rheed_single_frame_manual/model_comparison.csv`; the nested row is selected only from inner folds.",
        f"4. Strongly isolated patterns are not hard-coded as confident. Isolation versus confidence rho is `{safe_float(iso_conf, math.nan):.3g}` and isolation versus absolute error rho is `{safe_float(iso_err, math.nan):.3g}`.",
        "5. Sample 6023 is not analyzed because it is in the canonical removelist.",
        "",
        "## Largest errors",
        "",
    ]
    lines.extend(
        f"- sample {row['sample_id']}: true `{safe_float(row['rq_true_nm']):.3f}` nm, predicted `{safe_float(row['rq_pred_nm']):.3f}` nm, error `{safe_float(row['absolute_error_nm']):.3f}` nm"
        for row in largest_errors
    )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "This is a small-data hypothesis-testing regression experiment. The result should be read as an out-of-fold association test, not as causal evidence or a universal RHEED-to-roughness law.",
        ]
    )
    (paths.reports_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    methods_text = f"""# rheed_single_frame_manual 方法说明

本文档说明当前 `reports/rheed_single_frame_manual/` 和 `outputs/rheed_single_frame_manual/` 的生成代码、底层算法、特征含义和数据流。对应入口是 `analysis/rheed_single_frame/run.py`，配置是 `configs/rheed_single_frame_manual.yaml`。

## 1. 一句话概览

这条实验链路用每个样品一张人工挑选的 `select*` RHEED 截图，先通过固定图像处理把亮斑和亮条变成多组二值 mask，再从二值 mask 中提取连通域、水平邻接图、水平闭运算、骨架方向性、纹理频域和采集质量特征，最后用小样本外层留一交叉验证预测 AFM 的 RMS 粗糙度 `Rq`。

这里的“连接性”和“独立性”不是人工标注训练出来的类别，而是代码中明确定义的一组可复查几何统计：

- `horizontal_connectivity_score`：亮区域是否沿水平方向形成长条、可桥接、骨架水平、组件有水平邻居。
- `isolation_score`：亮区域是否更像彼此分离的紧凑斑点，而不是一个大连通结构或水平排列结构。
- `horizontal_run_length`：二值 mask 每一行连续亮像素段长度的 90 分位数。

## 2. 当前代码和框架

主入口：

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml
```

可清理重跑：

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml --clean
```

主要代码文件：

| 文件 | 作用 |
| --- | --- |
| `configs/rheed_single_frame_manual.yaml` | 输入目录、输出目录、RHEED 图像尺寸、阈值、模型和扰动参数。该文件是 JSON-compatible YAML。 |
| `analysis/rheed_single_frame/run.py` | 总控：读配置、建数据集、预处理、提特征、提 ResNet embedding、交叉验证、写输出和报告。 |
| `analysis/rheed_single_frame/data.py` | 匹配人工 RHEED 图片和 AFM plane-corrected height map，生成 target 和 manifest。 |
| `analysis/rheed_single_frame/preprocessing.py` | RHEED 单图加载、灰度化、等比例缩放、padding、强度归一化和预处理审计图。 |
| `analysis/rheed_single_frame/connectivity_features.py` | 当前最核心的水平连接性、孤立性、纹理、nuisance 和旧 morphology 特征。 |
| `analysis/rheed_single_frame/embeddings.py` | 可选的本地 frozen ResNet50 embedding，用作非解释性视觉基线。 |
| `analysis/rheed_single_frame/models.py` | 小样本模型、nested CV、one-standard-error 选择、置信区间和扰动稳定性。 |
| `analysis/rheed_single_frame/visualization.py` | 生成预测图、特征 overlay、HTML 报告。 |
| `analysis/rheed_single_frame/removelist.py` | 解析 canonical `removelist.txt`，并在各阶段阻止被移除样品进入流程。 |

使用到的主要 Python 框架和库：

- `numpy`：数组、统计量、FFT、特征矩阵。
- `scipy.ndimage` 和 `scipy.spatial`：图像平移/模糊、邻接距离、连通辅助。
- `scikit-image`：Otsu/local threshold、连通域 `regionprops`、形态学 closing/skeleton、Canny、HOG。
- `Pillow`：读取和保存 RHEED 图片。
- `matplotlib`：生成审计图和报告图。
- `scikit-learn`：`Pipeline`、`SimpleImputer`、`RobustScaler`、`PCA`、`Ridge`、`ElasticNet`、`PLSRegression`、`SVR`、`GaussianProcessRegressor`、`ExtraTreesRegressor`、`LeaveOneGroupOut`、`GroupKFold`。
- `torch`/`torchvision`：只用于 frozen ResNet50 embedding；当前状态见文末。

## 3. 数据流总览

| 阶段 | 函数 | 输入 | 输出 |
| --- | --- | --- | --- |
| 读配置和路径 | `run.run`, `data.make_paths` | `configs/rheed_single_frame_manual.yaml` | `outputs/rheed_single_frame_manual/`, `reports/rheed_single_frame_manual/` |
| removelist 审计 | `load_removelist_audit` | `removelist.txt` | `outputs/rheed_single_frame_manual/removelist_audit.json` |
| 样品配对 | `build_dataset` | `data/manual_selection`, `data/plane_corrected_afm` | `dataset_manifest.csv`, `selected_afm_targets.csv`, `dataset_audit.csv` |
| RHEED 预处理 | `preprocess_pairs` | 每个样品的 `select*` 图片 | `preprocessed/*_normalized.png`, `image_preprocessing_audit.csv` |
| 解释性特征 | `extract_feature_table` | normalized RHEED 和 padding mask | `physics_features.csv`, `threshold_feature_details.csv`, `component_features.csv` |
| Frozen embedding | `extract_frozen_embeddings` | normalized RHEED | `frozen_embeddings.csv`, `frozen_embeddings.parquet` |
| 建模表 | `make_model_table` | target、physics features、embedding | 内存中的 sample-level 表 |
| 固定模型评估 | `evaluate_fixed_models` | model table | `fixed_model_oof_predictions.csv`, `model_comparison.csv` |
| Nested 选择器 | `evaluate_nested_selector` | model table, model specs | `nested_selector_oof_predictions.csv`, `hyperparameter_selection.csv` |
| 不确定性/稳定性 | `confidence_calibration_rows`, perturbation flow | OOF prediction 和扰动图像特征 | `confidence_calibration.csv`, `perturbation_stability.csv` |
| 图和报告 | `make_all_figures`, `generate_html_report`, `write_report_texts` | CSV 结果和 feature overlays | `reports/rheed_single_frame_manual/*.png`, `index.html`, `results.md`, `methods.md` |

当前主结果中的独立样品数来自 `nested_selector_oof_predictions.csv`，一行对应一个外层 held-out 样品。当前这批输出为 26 个非 removelist 样品。

## 4. 样品构建和 AFM target

`data.py` 中的 `build_dataset` 做三件事。

第一，读取 `removelist.txt`。`removelist.py` 只解析每个非注释行开头的数字样品 ID，例如 `6023`。这些 ID 在 AFM candidate 加载、RHEED 预处理、特征写出、CV 和预测写出阶段都会用 `assert_no_removed_samples` 再检查一次。也就是说，被移除样品不是最后过滤，而是在进入数据链路前就被阻止。

第二，从 `data/manual_selection/<sample_id>/RHEED/` 下发现人工挑选的 `select*` RHEED 图片。当前实验约定每个纳入样品只使用一张人工挑选截图，不使用时间序列。

第三，从 `data/plane_corrected_afm/*/*_plane_corrected.npy` 读取 AFM candidate，并对每个样品选择 representative scan。配置中 `primary_scan_size_um=1.0`，`primary_scan_size_tolerance_um=0.10`，所以优先选择 1 um 附近的物理有效 AFM scan，并调用已有的 `select_representative_afm_scan` 选择代表该样品的 scan。

AFM target 是 `Rq`，单位 nm。优先从 `data/afm_descriptor_reconstruction/afm_descriptors.csv` 的 `Rq` 读取；如果 descriptor 缺失，则从 height map 重新计算。计算公式是：

```text
Rq = sqrt(mean((z - mean(z))^2))
```

其中 `z` 是 plane-corrected AFM height map，并按 metadata 的高度单位转换到 nm。target 表写入：

- `outputs/rheed_single_frame_manual/selected_afm_targets.csv`
- `outputs/rheed_single_frame_manual/dataset_manifest.csv`
- `outputs/rheed_single_frame_manual/dataset_audit.csv`
- `outputs/rheed_single_frame_manual/skipped_samples.csv`
- `outputs/rheed_single_frame_manual/excluded_by_removelist.csv`

## 5. RHEED 图片预处理

预处理在 `preprocessing.py::preprocess_one` 中完成。输入是一张人工选择的 RHEED 图片，输出是固定尺寸的灰度图、归一化图和 padding mask。

具体步骤：

1. `PIL.Image.open` 读取图片。如果是 RGB/RGBA，只保留前三个通道。
2. `frame_to_gray_float32` 将图片变成 `float32` 灰度图，值域约为 0 到 1。
3. 当前 `roi = gray`，也就是不裁剪原图；`roi_rule` 记录为 `no_crop_original_manual_selection_aspect_preserving_padding`。
4. `resize_with_padding` 把图像等比例缩放到最长边不超过 256，再居中填充到 `256 x 256`。图像内容区域为 `valid_mask=True`，padding 区域为 `False`。
5. 只在 `valid_mask` 区域内取 1 和 99 分位数做 robust intensity normalization：

```text
normalized = clip((gray_padded - p01) / (p99 - p01), 0, 1)
```

padding 区域不参与分位数估计，也不会参与后续阈值、连通域和纹理统计。

预处理产物：

- `outputs/rheed_single_frame_manual/preprocessed/<sample>_gray_padded.png`
- `outputs/rheed_single_frame_manual/preprocessed/<sample>_normalized.png`
- `outputs/rheed_single_frame_manual/preprocessed/<sample>_padding_mask.png`
- `outputs/rheed_single_frame_manual/image_preprocessing_audit.csv`
- `reports/rheed_single_frame_manual/preprocessing_audit/<sample>_preprocessing_contact.png`

`image_preprocessing_audit.csv` 还记录 `mean_intensity`、`dynamic_range`、`saturation_fraction`、`underexposure_fraction`、`background_gradient`、`sharpness` 等采集质量变量。这些变量后面会作为 nuisance features 单独建模，避免把曝光/尺寸差异误当成物理结构。

## 6. 从灰度图到二值亮区 mask

连接性分析从 `connectivity_features.py::_binary_thresholds` 开始。对每张 normalized RHEED 图，代码生成一组候选二值 mask，而不是只相信一个阈值。

当前阈值集合：

- `otsu`：在 valid pixels 上用 Otsu 自动阈值。
- `p75`, `p85`, `p90`, `p95`：在 valid pixels 上按配置的 `[75, 85, 90, 95]` 分位数阈值。
- `adaptive_local`：`skimage.filters.threshold_local`，Gaussian local threshold，block size 为 31；如果配置给偶数会调整为奇数，且不小于 7。

对每个阈值，二值 mask 定义为：

```text
B_t = (normalized_image > threshold_t) AND valid_mask
```

然后做两个很小的清理：

- `remove_small_objects(binary, max_size=3)`
- `remove_small_holes(binary, max_size=3)`

这里的 `max_size=3` 只去掉小于等于 3 像素的碎点或小洞，不会重塑大的 RHEED 结构。

每个阈值下的详细结果写入 `outputs/rheed_single_frame_manual/threshold_feature_details.csv`。最终进入模型的 `physics_features.csv` 是对各阈值同名特征取 median，同时额外写入 `<feature>_threshold_std` 表示该特征对阈值选择的稳定性。

## 7. 连通域：亮斑/亮条如何被识别

对每个二值 mask，`_component_table` 使用：

```python
labels = skimage.measure.label(mask, connectivity=2)
```

`connectivity=2` 表示 8 邻域连通：水平、垂直、对角相邻的亮像素都属于同一连通域。每个连通域再用 `skimage.measure.regionprops` 计算几何属性。面积小于 4 像素的连通域被忽略。

写入 `component_features.csv` 的主要字段包括：

- `area_px`：连通域像素面积。
- `centroid_x`, `centroid_y`：质心位置，单位是 256 图上的像素坐标。
- `bbox_width_px`, `bbox_height_px`：外接框尺寸。
- `aspect_ratio = bbox_width_px / bbox_height_px`。
- `equivalent_diameter_px`：等面积圆直径。
- `eccentricity`：椭圆离心率，越接近 1 越细长。
- `solidity`：面积和凸包面积的比例，越高越实心。
- `circularity = 4*pi*area/perimeter^2`：越接近 1 越圆。
- `orientation_rad`：`regionprops` 给出的主轴方向。
- `mean_intensity`：该连通域在 normalized 图上的平均强度。

这些连通域是后面“水平邻接图”和“独立性/孤立性”计算的节点。

## 8. 水平连接性特征如何计算

### 8.1 行方向连续亮段长度

`horizontal_run_lengths(mask)` 逐行扫描二值 mask，找到每一行中连续的 `True` 片段。例如某一行中 `False False True True True False True True` 会产生长度 3 和 2。

每个 run length 都除以图像宽度 256 归一化，然后统计：

- `horizontal_run_length_mean`
- `horizontal_run_length_median`
- `horizontal_run_length_max`
- `horizontal_run_length_q90`
- `horizontal_run_length_std`

最终暴露的简写特征 `horizontal_run_length` 等于 `horizontal_run_length_q90`。它表示“比较长的水平亮段通常有多长”，对偶发的一两个极长段不如 max 敏感。

### 8.2 水平邻接图

`_graph_features` 把每个连通域视为一个图节点。节点之间是否有“水平邻居”不是看它们是否已经接触，而是看两个 bright components 是否大致在同一高度、并在合理的水平距离内。

设图像高宽为 `H, W`，连通域等效直径的中位数为 `d`。两个连通域 `i, j` 的质心距离为：

```text
dx = abs(x_i - x_j)
dy = abs(y_i - y_j)
```

代码中的水平边条件是：

```text
y_tol = max(0.045 * H, 1.5 * d)
x_min = 0.25 * d
x_max = max(0.22 * W, 6.0 * d)

horizontal_edge(i, j) =
    dy <= y_tol
    AND dx >= x_min
    AND dx <= x_max
```

这个定义的物理含义是：两个亮斑/亮条虽然没有连成同一个连通域，但如果它们高度接近、横向间距合理，就认为它们可能属于同一条水平 RHEED streak 或水平排列结构。

由这个图计算：

- `horizontal_neighbor_fraction`：有至少一个水平邻居的连通域比例。
- `horizontal_connectivity_graph_density`：实际水平边数 / 所有可能边数。
- `largest_graph_component_fraction`：水平邻接图里最大 connected group 占所有节点的比例。
- `graph_component_count`：水平邻接图被分成多少个 group。
- `horizontal_nearest_neighbor_gap`：水平边对应的 `dx` 中位数；没有水平边时取图像宽度。
- `vertical_nearest_neighbor_gap`：水平边对应的 `dy` 中位数；没有水平边时取图像高度。
- `horizontal_gap_normalized_by_diameter`：水平 gap 除以 `d`。
- `isolated_component_fraction = 1 - horizontal_neighbor_fraction`。
- `fraction_without_horizontally_aligned_neighbor = 1 - horizontal_neighbor_fraction`。

代码用一个简单 union-find 合并有水平边的节点，以得到 `largest_graph_component_fraction` 和 `graph_component_count`。

### 8.3 水平形态学闭运算

`_threshold_features` 对二值 mask 做水平和垂直 closing。当前实现实际使用的长度硬编码为 `(5, 11, 21)`，与配置文件中的 `horizontal_closing_lengths` 数值一致。

水平 closing 使用矩形结构元：

```text
horizontal footprint: 1 x length
vertical footprint: length x 1
```

对每个 length：

```text
horizontal_closing_gain =
    (largest_component_area(horizontal_closed_mask) - largest_component_area(original_mask))
    / valid_area
```

再对 5、11、21 三个长度取 median。这个特征回答的问题是：如果允许沿水平方向桥接小间隙，最大连通结构会增加多少？如果增加很多，说明原图中的亮结构具有“差一点连起来的水平连续性”。

相关字段：

- `horizontal_closing_gain`
- `vertical_closing_gain`
- `horizontal_to_vertical_closing_gain`
- `gaps_closed_by_horizontal_structuring_elements`

### 8.4 骨架方向性

代码用 `skimage.morphology.skeletonize(mask)` 把亮区变成一像素宽骨架，然后统计骨架上的水平相邻像素对和垂直相邻像素对：

```text
horizontal_skeleton_fraction =
    horizontal_links / max(horizontal_links + vertical_links, 1)
```

这个特征衡量亮结构的“中心线”更偏水平还是垂直。另有：

- `horizontal_branch_count`：骨架上 3x3 邻域内邻居数大于等于 3 的分叉点数。
- `average_width_horizontal_structures = mask.sum() / max(skeleton.sum(), 1)`：可理解为亮结构平均厚度的近似。

### 8.5 细长水平组件占比

代码还筛出一类 elongated components：

```text
aspect_ratio >= 1.7
AND abs(orientation_rad) > 45 degrees
```

然后计算：

```text
fraction_bright_pixels_in_horizontal_components =
    elongated_components_area_sum / total_bright_pixels
```

注意这里严格按当前代码的 `regionprops.orientation` 条件执行；它是一个实现层面的方向判据，不是额外人工标注。

### 8.6 综合水平连接性分数

最终的 `horizontal_connectivity_score` 在 `_composite_scores` 中定义为六个 0 到 1 子项的平均值：

```text
horizontal_connectivity_score = mean([
    clip(horizontal_neighbor_fraction, 0, 1),
    clip(largest_graph_component_fraction, 0, 1),
    clip(horizontal_skeleton_fraction, 0, 1),
    clip(fraction_bright_pixels_in_horizontal_components, 0, 1),
    clip(horizontal_closing_gain * 20, 0, 1),
    clip(horizontal_run_length_q90 * 4, 0, 1),
])
```

这些子项分别对应：

- 连通域之间是否有水平邻居。
- 水平邻接图是否形成大 group。
- 二值结构骨架是否偏水平。
- 亮像素是否集中在细长水平组件中。
- 水平 closing 能否显著桥接间隙。
- 每行连续亮段是否较长。

因此，高 `horizontal_connectivity_score` 表示“在多种阈值下都更像水平连续 streak/条带结构”；低分表示水平连通证据弱，可能是碎散斑点、孤立亮区或非水平结构。

## 9. 独立性/孤立性如何计算

当前代码中的“独立性”主要对应 `isolation_score` 和几个组件级字段。它不是样品之间的统计独立性，而是 RHEED 图中亮组件是否彼此孤立。

先定义紧凑组件比例：

```text
compact component =
    circularity > 0.45
    AND solidity > 0.55
    AND aspect_ratio < 1.6

compact_component_fraction = compact_count / component_count
```

再定义几个子项：

```text
isolated_compact =
    compact_component_fraction * isolated_component_fraction

fragmented =
    clip(graph_component_count / component_count, 0, 1)
    * clip(component_count / 6, 0, 1)

small_largest =
    clip(1 - largest_component_fraction * 8, 0, 1)

no_aligned_compact =
    compact_component_fraction * fraction_without_horizontally_aligned_neighbor
```

最终：

```text
isolation_score = mean([
    isolated_compact,
    compact_component_fraction,
    small_largest,
    fragmented,
    no_aligned_compact,
])
```

高 `isolation_score` 一般表示：

- 亮组件更圆、更紧凑。
- 很多组件没有水平邻居。
- 最大连通域没有占据太大面积。
- 水平邻接图被分成多个 group。

低 `isolation_score` 一般表示：

- 亮区形成大块或长条。
- 组件之间沿水平方向排列或可桥接。
- 不是“彼此独立的小亮斑”形态。

## 10. 纹理、方向和频域特征

除了显式连通域，`_texture_features` 还从 normalized 图中计算连续灰度纹理：

- 图像梯度：`np.gradient` 得到 `gx`, `gy`。
- Structure tensor：对 `gx*gx`、`gy*gy`、`gx*gy` 做 `sigma=2` Gaussian smoothing，计算 `structure_tensor_anisotropy` 和 `dominant_orientation_deg`。
- 梯度能量：`horizontal_gradient_energy`、`vertical_gradient_energy`、`horizontal_vs_vertical_gradient_energy`。
- FFT：对减去均值后的图像做 2D FFT，计算中心水平频带与垂直频带的功率比 `horizontal_vs_vertical_fft_power`，以及 `fft_low_frequency_power`、`fft_mid_high_frequency_power`。
- 边缘密度：`skimage.feature.canny` 得到 `edge_density`。
- HOG：`skimage.feature.hog`，9 个方向 bin，cell size 为 `32 x 32`，输出 `hog_orientation_bin_0` 到 `hog_orientation_bin_8`。

这些特征补充的是灰度方向性和频率结构，不依赖单个阈值 mask。

## 11. Nuisance 特征：曝光、尺寸和位置

`_nuisance_features` 把不应直接解释为材料形貌的采集/图像质量变量单独列出，方便和 physics features 分开建模。

包括：

- `mean_intensity`, `median_intensity`, `intensity_std`, `dynamic_range`
- `saturation_fraction`, `underexposure_fraction`
- `background_gradient`
- `sharpness`
- `image_height`, `image_width`, `valid_roi_fraction`
- `pattern_centroid_x`, `pattern_centroid_y`
- `bright_pixel_centroid_x`, `bright_pixel_centroid_y`

质心用 normalized 图中高于 70 分位的亮度作为权重计算，并归一化到 0 到 1。模型里有单独的 `nuisance_ridge`，用于检查曝光、尺寸、位置等变量本身是否已经能解释 target；这有助于防止把采集偏差误解为物理形貌信号。

## 12. 旧 morphology score 和新连接性特征的关系

`_existing_morphology_features` 复用已有 RHEED shape pipeline：

1. `preprocess_frame_for_shape` 生成 `soft_spot_streak_mask`、`log_bgsub` 和 artifact mask。
2. `extract_components_and_frame_features` 从 soft mask 中识别 `round_spot`、`elongated_spot`、`horizontal_bar`、`vertical_streak`、`diffuse_blob` 等旧组件类型。
3. `compute_morphology_scores` 计算旧的 spotty-to-streaky 指标。

旧指标公式是：

```text
spottiness = round_spot_count + 0.60 * elongated_spot_count + 0.25 * diffuse_blob_count
streakiness = horizontal_bar_count + vertical_streak_count + 0.60 * elongated_spot_count + bar_like_score * total_component_count
morphology_index = spottiness / (spottiness + streakiness)
```

输出字段：

- `existing_morphology_index`
- `morphology_index`
- `raw_spottiness`
- `raw_streakiness`
- `existing_detector_confidence`

这部分是旧方法的对照特征，不等同于本次新增的 `horizontal_connectivity_score`。模型比较中 `morphology_linear`、`morphology_ridge`、`morphology_isotonic_increasing` 使用的是这个旧 morphology score。

## 13. Frozen ResNet50 embedding

`embeddings.py` 会检查本地是否有 ImageNet ResNet50 checkpoint：

- `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`
- `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`

如果存在，就构造 `torchvision.models.resnet50(weights=None)`，加载本地权重，去掉最后的分类层，只保留 backbone。每张 normalized RHEED 图会复制成 3 通道，并用 ImageNet mean/std 标准化，然后得到 2048 维 embedding：

```text
embedding_0000 ... embedding_2047
```

这些 embedding 写入：

- `outputs/rheed_single_frame_manual/frozen_embeddings.csv`
- `outputs/rheed_single_frame_manual/frozen_embeddings.parquet`

它们是非解释性视觉基线，不用于解释“连接性”或“独立性”的物理定义。

当前 frozen embedding 状态：`{embedding_summary.get('status', '')}`，checkpoint：`{embedding_summary.get('checkpoint', '')}`。

## 14. 建模和交叉验证

模型输入表由 `models.py::make_model_table` 在内存中合并：

- `selected_afm_targets.csv` 的 `rq_nm`
- `physics_features.csv` 的解释性和 nuisance 特征
- `frozen_embeddings.csv` 的 embedding 特征

target 不是直接拟合 `Rq`，而是：

```text
log_rq_true = log10(rq_nm)
```

预测后再转换回：

```text
rq_pred_nm = 10 ** log_rq_pred
```

模型候选由 `build_model_specs` 生成，包含：

- `median_baseline`：训练 fold 的 `log10(Rq)` 中位数。
- `morphology_*`：旧 morphology score 的线性、ridge、isotonic 模型。
- `nuisance_ridge`：只用 nuisance 特征。
- `connectivity_interpretable_ridge`：只用预注册的 7 个解释性连接性特征：`horizontal_connectivity_score`、`horizontal_closing_gain`、`horizontal_run_length`、`isolation_score`、`isolated_component_fraction`、`horizontal_neighbor_fraction`、`largest_component_fraction`。
- `physics_*`：使用全部 physics 特征的 Ridge、ElasticNet、PLS、SVR、Gaussian Process、ExtraTrees。
- `frozen_resnet50_*`：使用 ResNet50 embedding 的 PCA/Ridge 或 PLS。
- `hybrid_*`：physics + embedding。

大多数 sklearn pipeline 的前处理是：

```text
SimpleImputer(strategy="median") -> RobustScaler -> optional PCA -> regressor
```

交叉验证有两层：

- 外层：`LeaveOneGroupOut`。当前 `growth_run_id` 等于 `sample_id`，所以实际相当于 leave-one-sample-out；每个样品的预测都来自没见过该样品的模型。
- 内层：`GroupKFold`，最多 5 折，用于在训练 fold 内选择模型。

Nested selector 使用 one-standard-error 规则：

1. 对每个候选模型计算 inner CV 的 mean absolute error in log space。
2. 找到 inner MAE 最小的模型。
3. 允许所有 inner MAE 不超过 `best + best_standard_error` 的模型进入候选。
4. 在这些模型中选择 `simplicity_rank` 最低的模型；如果复杂度相同，再选 inner MAE 更低的模型。

这样做的目的是在 26 个样品的小数据设置下优先选择简单模型，降低过拟合。

主要建模输出：

- `outputs/rheed_single_frame_manual/fixed_model_oof_predictions.csv`
- `outputs/rheed_single_frame_manual/nested_selector_oof_predictions.csv`
- `outputs/rheed_single_frame_manual/model_comparison.csv`
- `outputs/rheed_single_frame_manual/hyperparameter_selection.csv`
- `outputs/rheed_single_frame_manual/feature_importance.csv`
- `outputs/rheed_single_frame_manual/influence_analysis.csv`

## 15. 置信区间、confidence 和扰动稳定性

`nested_selector_oof_predictions.csv` 中的预测区间是 conformal-style interval。具体来说，内层 CV 得到训练 fold 中被选模型的 absolute residuals in log space，`conformal_q` 按 nominal 0.90 取分位数 `q`，再对 held-out 样品写：

```text
lower_log = pred_log - q
upper_log = pred_log + q
prediction_interval_lower_nm = 10 ** lower_log
prediction_interval_upper_nm = 10 ** upper_log
```

`confidence_score` 是 0 到 100 的诊断分数，不是概率。它取四个分数的平均：

- 区间宽度分数：预测区间相对数据集 Rq IQR 越窄越高。
- domain support score：held-out 样品在 selected feature space 中离训练样品越近越高。
- bootstrap ensemble score：bootstrap 预测标准差越小越高。
- perturbation sensitivity score：图像扰动后预测越稳定越高。

扰动来自 `run.py::perturbation_names`，当前包括：

- brightness scale: `0.8`, `0.9`, `1.1`, `1.2`
- gamma: `0.85`, `1.15`
- translate x/y: `-4`, `4` pixels
- Gaussian blur sigma: `0.7`
- Gaussian noise sigma: `0.015`
- crop jitter proxy: `-4`, `4`

每个扰动图像会重新走一次特征提取，然后用该外层 fold 选中的模型计算预测变化，写入 `outputs/rheed_single_frame_manual/perturbation_stability.csv`。

## 16. 可视化和人工审计

每个样品会保存三类 feature overlay：

- `reports/rheed_single_frame_manual/feature_overlays/<sample>_components.png`：normalized 图上叠加连通域边界和质心。
- `reports/rheed_single_frame_manual/feature_overlays/<sample>_horizontal_closing.png`：用于审计水平 closing 的二值图。
- `reports/rheed_single_frame_manual/feature_overlays/<sample>_skeleton.png`：normalized 图上叠加骨架结构。

`_save_overlay_figures` 为每个样品选择一个“最适合展示”的阈值：最大化 `horizontal_neighbor_fraction + bright_pixel_fraction` 的 threshold row。注意 overlay 的阈值只用于可视化；模型输入的主特征仍然是多个阈值的 median ensemble。

总览图和 HTML 报告包括：

- `reports/rheed_single_frame_manual/index.html`
- `single_frame_oof_predictions_by_true_rq_common_scale.png`
- `connectivity_feature_overlays_by_rq.png`
- `connectivity_feature_overlays_by_connectivity.png`
- `connectivity_feature_overlays_by_isolation.png`
- `figures/model_comparison.png`
- `figures/feature_hypothesis_plots.png`
- `figures/perturbation_stability.png`

## 17. 如何追踪一个样品

以样品 ID 为单位，可以按以下路径追踪：

1. 在 `outputs/rheed_single_frame_manual/dataset_manifest.csv` 找到 `sample_id`、`manual_rheed_path`、`selected_height_map_path` 和 `rq_nm`。
2. 在 `outputs/rheed_single_frame_manual/image_preprocessing_audit.csv` 查看原始尺寸、padding、normalized 图路径和质量指标。
3. 在 `outputs/rheed_single_frame_manual/threshold_feature_details.csv` 查看该样品每个阈值下的连接性特征。
4. 在 `outputs/rheed_single_frame_manual/component_features.csv` 查看每个阈值下每个亮斑/亮条连通域的几何属性。
5. 在 `outputs/rheed_single_frame_manual/physics_features.csv` 查看跨阈值聚合后的最终特征。
6. 在 `reports/rheed_single_frame_manual/feature_overlays/` 查看对应的 components、horizontal closing 和 skeleton 图。
7. 在 `outputs/rheed_single_frame_manual/nested_selector_oof_predictions.csv` 查看该样品的 held-out 预测、预测区间、confidence、选中模型和误差。

## 18. 解释边界和注意事项

- 当前连接性/孤立性特征是目标无关的图像几何规则，不使用 AFM `Rq` 来调阈值或定义 mask。
- `horizontal_connectivity_score` 是 operational definition：它衡量的是 RHEED 截图中亮结构的水平连续证据，不等于某个唯一物理机制。
- `isolation_score` 是图像亮组件的孤立性，不是统计学上样品之间的独立性。
- 由于每个样品只用一张人工挑选截图，结果不包含 RHEED 时间演化信息。
- 多阈值 median ensemble 降低了单一阈值的偶然性，但如果原图曝光异常或背景极强，分割仍可能失败，应查看 overlay 和 preprocessing audit。
- 当前 `horizontal_closing_lengths` 在配置中写为 `[5, 11, 21]`，实现中对应闭运算长度也固定使用 `(5, 11, 21)`。
- 小样本 CV 的主要价值是外层 out-of-fold 关联测试和方法审计，不应被解读为已经稳定部署的通用 RHEED-to-AFM 预测器。
"""
    (paths.reports_dir / "methods.md").write_text(methods_text.strip() + "\n", encoding="utf-8")
    limitations = [
        "# Limitations",
        "",
        "- The independent sample count is small after removelist filtering and missing manual-image rules.",
        "- The manually selected frame is useful for this first test, but does not represent temporal evolution.",
        "- The feature extractor is visually auditable, but segmentation can still fail on unusual intensity distributions; QC flags and overlays should be inspected.",
        "- Nuisance-only performance should be checked before interpreting morphology as physical signal.",
        "- 6023 is excluded by the current canonical removelist, so no primary or secondary 6023 sensitivity result is scientifically valid in this run.",
    ]
    (paths.reports_dir / "limitations.md").write_text("\n".join(limitations), encoding="utf-8")
    readme = [
        "# Single-frame Manual RHEED Experiment",
        "",
        "Reproduce full run:",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml",
        "```",
        "",
        "Smoke run:",
        "",
        "```bash",
        "PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_single_frame.run --config configs/rheed_single_frame_manual.yaml --smoke",
        "```",
        "",
        "Primary outputs are under `outputs/rheed_single_frame_manual/`; figures and the static report are under `reports/rheed_single_frame_manual/`.",
    ]
    (paths.reports_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def run(config_path: Path, *, smoke: bool = False, clean: bool = False) -> dict[str, Any]:
    config = read_config(config_path)
    if smoke:
        config = _smoke_config(config)
    paths = make_paths(config)
    if clean:
        shutil.rmtree(paths.outputs_dir, ignore_errors=True)
        shutil.rmtree(paths.reports_dir, ignore_errors=True)
        paths = make_paths(config)
    removelist = load_removelist_audit(paths.repo_root, config.get("removelist_path"))
    removelist_payload = audit_to_json(removelist)
    write_json(paths.outputs_dir / "removelist_audit.json", removelist_payload)
    bundle = build_dataset(config, paths, removelist)
    if smoke:
        bundle = _limit_bundle_for_smoke(bundle)
    write_dataset_outputs(bundle, removelist)
    images = preprocess_pairs(bundle.pairs, paths, config, removelist)
    feature_results = extract_feature_table(images, paths, config, removelist)
    feature_rows = [result.features for result in feature_results]
    embedding_rows, embedding_summary = extract_frozen_embeddings(images, paths, removelist)
    model_table = make_model_table(bundle.target_rows, feature_rows, embedding_rows)
    phys_names = physics_feature_names(feature_rows)
    nuisance_names = nuisance_feature_names(feature_rows)
    emb_names = embedding_feature_names(embedding_rows)
    all_feature_names = _numeric_feature_names(model_table)
    specs = build_model_specs(all_feature_names, phys_names, nuisance_names, emb_names, config)
    print_initial_audit(bundle, removelist_payload, [
        "horizontal run length",
        "horizontal autocorrelation",
        "horizontal/vertical closing gain",
        "horizontal neighbor graph",
        "isolated component fraction",
        "structure tensor/FFT/HOG",
    ], specs, embedding_summary)
    perturbation_feature_rows = build_perturbed_feature_rows(images, paths, config, model_table)
    fixed_predictions, fixed_comparison = evaluate_fixed_models(model_table, specs, removelist)
    nested_predictions, hyper_rows, perturbation_rows = evaluate_nested_selector(
        model_table,
        specs,
        config,
        removelist,
        perturbation_rows_by_sample=perturbation_feature_rows,
    )
    confidence_rows = confidence_calibration_rows(nested_predictions)
    model_comparison = final_model_comparison(fixed_comparison, nested_predictions)
    influence_rows = influence_analysis(nested_predictions)
    sensitivity_rows = sensitivity_without_6023(nested_predictions, removelist)
    importance_features = [name for name in phys_names + nuisance_names if name in all_feature_names]
    importance_rows = feature_importance_from_ridge(model_table, importance_features, removelist)
    permutation_payload = permutation_tests(nested_predictions, fixed_predictions, config)
    write_json(paths.outputs_dir / "permutation_tests.json", permutation_payload)
    write_model_outputs(
        paths,
        fixed_predictions,
        nested_predictions,
        model_comparison,
        hyper_rows,
        confidence_rows,
        perturbation_rows,
        importance_rows,
        influence_rows,
        sensitivity_rows,
        removelist,
    )
    make_all_figures(
        nested_predictions,
        feature_rows,
        model_comparison,
        importance_rows,
        influence_rows,
        sensitivity_rows,
        perturbation_rows,
        bundle.pairs,
        paths,
        removelist,
        bundle.common_scale,
    )
    generate_html_report(nested_predictions, feature_rows, model_comparison, bundle.skipped_rows, bundle.excluded_rows, paths)
    write_report_texts(
        paths,
        nested_predictions,
        model_comparison,
        feature_rows,
        confidence_rows,
        permutation_payload,
        removelist_payload,
        embedding_summary,
    )
    summary = {
        "included_samples": len(nested_predictions),
        "outputs_dir": display_path(paths.outputs_dir, paths.repo_root),
        "reports_dir": display_path(paths.reports_dir, paths.repo_root),
        "best_model_by_mae_log": min(model_comparison, key=lambda row: safe_float(row.get("mae_log"), math.inf)).get("model_name") if model_comparison else "",
        "nested_spearman_nm": permutation_payload.get("nested_spearman_nm"),
        "nested_spearman_permutation_p": permutation_payload.get("nested_spearman_permutation_p"),
    }
    print("Final single-frame experiment summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rheed_single_frame_manual.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete this experiment's output/report directory before running.")
    args = parser.parse_args(argv)
    run(args.config, smoke=args.smoke, clean=args.clean)


if __name__ == "__main__":
    main()
