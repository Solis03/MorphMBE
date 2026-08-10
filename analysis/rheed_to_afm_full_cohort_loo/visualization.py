from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_functional_morphology.run import M10
from analysis.rheed_to_afm_functional_morphology.visualization import (
    REAL_COLOR,
    SELECTED_COLOR,
    _comparison_figure,
    _generated,
    _real_afm,
    _real_afm_label,
    _rheed_keyframe,
    _save,
    _style,
    _surface_panel,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    write_json,
)

from .run import load_config, load_source_tables

SOURCE_COLORS = {
    "original_23_batch": "#0072B2",
    "extra_five_batch": "#E69F00",
}
SOURCE_MARKERS = {"original_23_batch": "o", "extra_five_batch": "D"}
SOURCE_LABELS = {
    "original_23_batch": "original 23-sample batch",
    "extra_five_batch": "second-batch extra five",
}


def _method_label(method: str) -> str:
    labels = {
        "M17c_topology_sparse_finetexture_terrace": (
            "M17c topology-conditioned sparse-peak texture"
        ),
        "M17b_topology_sparse_peak_terrace": ("M17b topology-conditioned sparse peaks"),
        "M16b_regime_adaptive_microisland_terrace": (
            "M16b regime-adaptive micro-island terrace"
        ),
    }
    return labels.get(method, method.replace("_", " "))


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"growth_run_id": str})


def _external_target_confidence(
    *,
    path: str | Path,
    fallback: pd.DataFrame,
    method: str = "M14i_target_specific_robust",
    fsmi_path: str | Path | None = None,
    fsmi_method: str | None = None,
) -> pd.DataFrame:
    predictions = pd.read_csv(repo_path(path), dtype={"growth_run_id": str})
    required = {
        "growth_run_id",
        "target",
        "method",
        "confidence",
        "absolute_error",
        "predicted_absolute_error",
        "interval_covered",
        "outer_target_used_for_training",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(
            f"external confidence columns missing from {path}: {missing}"
        )
    if predictions["outer_target_used_for_training"].astype(bool).any():
        raise RuntimeError(f"external confidence reports outer target leakage: {path}")
    selected = predictions.loc[predictions["method"] == str(method)]
    if selected.empty:
        available = sorted(predictions["method"].astype(str).unique())
        raise RuntimeError(
            f"external confidence method {method!r} is unavailable; found {available}"
        )
    rq = selected.loc[selected["target"] == "Rq_nm"].set_index("growth_run_id")
    fsmi_source = selected
    if fsmi_path is not None:
        fsmi_predictions = pd.read_csv(
            repo_path(fsmi_path), dtype={"growth_run_id": str}
        )
        fsmi_selected_method = str(fsmi_method or method)
        fsmi_source = fsmi_predictions.loc[
            fsmi_predictions["method"] == fsmi_selected_method
        ]
        if fsmi_source.empty:
            available = sorted(fsmi_predictions["method"].astype(str).unique())
            raise RuntimeError(
                f"external FSMI confidence method "
                f"{fsmi_selected_method!r} is unavailable; found {available}"
            )
    fsmi = fsmi_source.loc[fsmi_source["target"] == "FSMI_nm"].set_index(
        "growth_run_id"
    )
    expected = int(fallback["growth_run_id"].astype(str).nunique())
    if set(rq.index) != set(fsmi.index) or len(rq) != expected:
        raise RuntimeError(
            "external target confidence does not cover the full cohort: "
            f"expected {expected}, found Sq={len(rq)}, FSMI={len(fsmi)}"
        )
    result = fallback.set_index("growth_run_id").loc[rq.index].copy()
    result["joint_confidence_index"] = 100.0 * np.sqrt(
        rq["confidence"] * fsmi["confidence"]
    )
    result["realized_rq_absolute_error_nm"] = rq["absolute_error"]
    result["predicted_rq_absolute_error_nm"] = rq["predicted_absolute_error"]
    result["realized_fsmi_absolute_error_nm"] = fsmi["absolute_error"]
    result["predicted_fsmi_absolute_error_nm"] = fsmi["predicted_absolute_error"]
    result["rq_interval_covered"] = rq["interval_covered"].astype(bool)
    result["fsmi_interval_covered"] = fsmi["interval_covered"].astype(bool)
    result["realized_joint_error_index"] = 0.5 * (
        rq["absolute_error"].rank(pct=True) + fsmi["absolute_error"].rank(pct=True)
    )
    result["confidence_basis"] = "geometric mean of strict-LOO Sq and FSMI confidence"
    return result.reset_index()


def _source_scatter(
    axis: plt.Axes,
    table: pd.DataFrame,
    confidence: pd.Series,
) -> mpl.collections.PathCollection:
    last: mpl.collections.PathCollection | None = None
    for source, rows in table.groupby("cohort_origin"):
        last = axis.scatter(
            rows["true_target"],
            rows["predicted_target"],
            c=rows["growth_run_id"].map(confidence),
            cmap="viridis",
            vmin=0,
            vmax=100,
            marker=SOURCE_MARKERS.get(str(source), "o"),
            s=52,
            edgecolor=SOURCE_COLORS.get(str(source), "black"),
            linewidth=1.2,
            label=SOURCE_LABELS.get(str(source), str(source)),
            zorder=3,
        )
    if last is None:
        raise RuntimeError("empty prediction table")
    return last


def plot_full_atlas(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
    cohort_count: int,
) -> list[str]:
    groups = list(rq.sort_values("true_target")["growth_run_id"].astype(str))
    pages = int(math.ceil(len(groups) / 5))
    stems: list[str] = []
    for page, start in enumerate(range(0, len(groups), 5), start=1):
        subset = groups[start : start + 5]
        figure = _comparison_figure(
            groups=subset,
            split="crossfit",
            output=output,
            phase1=phase1,
            rq_predictions=rq,
            fsmi_predictions=fsmi,
            confidence=confidence,
            method=method,
            title=(
                f"Full {cohort_count}-growth retrospective LOO: "
                "RHEED → generated AFM "
                f"→ measured AFM ({page}/{pages}; C is relative, not a probability)"
            ),
        )
        stem = f"Fig1{chr(96 + page)}_full{cohort_count}_loo_atlas"
        _save(figure, figure_dir / stem)
        stems.append(stem)
    return stems


def plot_target_scatter(
    *,
    figure_dir: Path,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    source: pd.DataFrame,
    cohort_count: int,
) -> None:
    conf = confidence.set_index("growth_run_id")["joint_confidence_index"]
    source_map = source.set_index("growth_run_id")["cohort_origin"]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    for axis, values, label in (
        (axes[0], rq.copy(), "Sq (nm)"),
        (axes[1], fsmi.copy(), "FSMI (nm)"),
    ):
        values["cohort_origin"] = values["growth_run_id"].map(source_map)
        scatter = _source_scatter(axis, values, conf)
        lower = (values["predicted_target"] - values["interval_lower"]).to_numpy(float)
        upper = (values["interval_upper"] - values["predicted_target"]).to_numpy(float)
        axis.errorbar(
            values["true_target"],
            values["predicted_target"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#888888",
            alpha=0.38,
            lw=0.7,
            zorder=1,
        )
        lo = float(min(values["true_target"].min(), values["interval_lower"].min()))
        hi = float(max(values["true_target"].max(), values["interval_upper"].max()))
        axis.plot([lo, hi], [lo, hi], "--", color="black", lw=1)
        axis.set_xlim(max(0.0, lo - 0.25), hi + 0.25)
        axis.set_ylim(max(0.0, lo - 0.25), hi + 0.25)
        axis.set_xlabel(f"measured {label}")
        axis.set_ylabel(f"LOO-predicted {label}")
        pearson = pearsonr(values["true_target"], values["predicted_target"])
        rank = spearmanr(values["true_target"], values["predicted_target"])
        mae = float(values["absolute_error"].mean())
        axis.set_title(
            f"{label}: r={pearson.statistic:.2f}, "
            f"ρ={rank.statistic:.2f}, MAE={mae:.2f} nm"
        )
        for _, row in values.nlargest(4, "absolute_error").iterrows():
            axis.annotate(
                str(row["growth_run_id"]),
                (row["true_target"], row["predicted_target"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.legend(frameon=False, fontsize=7, loc="upper left")
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.82)
    colorbar.set_label("cross-fitted confidence index (0–100)")
    figure.suptitle(
        f"Every point is held out once ({cohort_count - 1} growths fit, one predicted)",
        fontsize=11,
        fontweight="bold",
    )
    _save(
        figure,
        figure_dir / f"Fig2_full{cohort_count}_target_scatter",
    )


def plot_protocol_comparison(
    *,
    figure_dir: Path,
    comparison: pd.DataFrame,
    current_method_label: str = "M13",
    cohort_count: int = 23,
) -> None:
    fit_count = cohort_count - 1
    labels = {
        "prior_M12_strict15_train14": "M12: 15-growth LOO\n(14 fit)",
        f"current_full{cohort_count}_train{fit_count}_same15": (
            f"{current_method_label}: same 15 points\n({fit_count} fit)"
        ),
        f"current_full{cohort_count}_train{fit_count}_all{cohort_count}": (
            f"{current_method_label}: all {cohort_count} points\n({fit_count} fit)"
        ),
    }
    metrics = [
        ("mean_absolute_error", "mean MAE (nm)", False),
        ("rmse", "RMSE (nm)", False),
        ("pearson_r", "Pearson r", True),
        ("spearman_rho", "Spearman ρ", True),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.1, 6.0), constrained_layout=True)
    colors = ["#56B4E9", "#E69F00", "#D55E00"]
    protocols = list(labels)
    x = np.arange(2)
    width = 0.24
    for axis, (column, title, higher) in zip(axes.ravel(), metrics, strict=False):
        for index, protocol in enumerate(protocols):
            rows = (
                comparison.loc[comparison["protocol"] == protocol]
                .set_index("target")
                .reindex(["Rq_nm", "FSMI_nm"])
            )
            axis.bar(
                x + (index - 1) * width,
                rows[column],
                width=width,
                color=colors[index],
                label=labels[protocol],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(["Sq", "FSMI"])
        axis.set_title(f"{title} ({'higher' if higher else 'lower'} is better)")
        if higher:
            axis.axhline(0, color="black", lw=0.6)
            axis.set_ylim(-0.05, 1.0)
        else:
            axis.set_ylabel("nm")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
    )
    figure.suptitle(
        "Retrospective protocol comparison (different cohort and fit size)",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_dir / "Fig3_protocol_comparison")


def plot_rq_order(
    *,
    figure_dir: Path,
    rq: pd.DataFrame,
    confidence: pd.DataFrame,
    cohort_count: int,
) -> None:
    conf = confidence.set_index("growth_run_id")["joint_confidence_index"]
    ordered = rq.sort_values("true_target").reset_index(drop=True)
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10.0, 4.2), constrained_layout=True)
    axis.plot(
        x,
        ordered["true_target"],
        "o-",
        color=REAL_COLOR,
        label="measured Sq",
        lw=1.7,
    )
    line = axis.scatter(
        x,
        ordered["predicted_target"],
        c=ordered["growth_run_id"].map(conf),
        cmap="viridis",
        vmin=0,
        vmax=100,
        edgecolor="black",
        linewidth=0.5,
        s=52,
        label="LOO-predicted Sq",
        zorder=3,
    )
    axis.plot(
        x,
        ordered["predicted_target"],
        color=SELECTED_COLOR,
        lw=1.1,
        alpha=0.8,
    )
    axis.vlines(
        x,
        ordered["true_target"],
        ordered["predicted_target"],
        color="#888888",
        lw=0.8,
        alpha=0.6,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["growth_run_id"], rotation=55, ha="right", fontsize=7)
    axis.set_xlabel("held-out growth (ordered by measured Sq)")
    axis.set_ylabel("Sq (nm)")
    axis.set_title(
        "Full-cohort LOO exposes high-Sq underprediction and unstable "
        "low-end extrapolation"
    )
    axis.legend(frameon=False, loc="upper left")
    colorbar = figure.colorbar(line, ax=axis, pad=0.02)
    colorbar.set_label("confidence index")
    _save(figure, figure_dir / f"Fig4_full{cohort_count}_rq_ordered")


def plot_confidence_audit(*, figure_dir: Path, confidence: pd.DataFrame) -> None:
    rho = spearmanr(
        confidence["joint_confidence_index"],
        confidence["realized_joint_error_index"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    points = axes[0].scatter(
        confidence["joint_confidence_index"],
        confidence["realized_joint_error_index"],
        c=confidence["realized_rq_absolute_error_nm"],
        cmap="magma",
        s=50,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in confidence.nlargest(5, "realized_joint_error_index").iterrows():
        axes[0].annotate(
            str(row["growth_run_id"]),
            (
                row["joint_confidence_index"],
                row["realized_joint_error_index"],
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_xlabel("reported confidence index (0–100)")
    axes[0].set_ylabel("realized joint error rank")
    axes[0].set_title(
        (
            "Confidence is error-related"
            if rho.statistic < 0 and rho.pvalue < 0.05
            else "Confidence ordering is not validated"
        )
        + f": Spearman ρ={rho.statistic:.2f}, p={rho.pvalue:.3f}"
    )
    colorbar = figure.colorbar(points, ax=axes[0], pad=0.02)
    colorbar.set_label("realized Sq error (nm)")

    coverage = [
        confidence["rq_interval_covered"].astype(bool).mean(),
        confidence["fsmi_interval_covered"].astype(bool).mean(),
        (
            confidence["realized_island_error_z"]
            <= confidence["island_error_90_upper_z"]
        ).mean(),
    ]
    axes[1].bar(
        ["Sq", "FSMI", "island topology"],
        coverage,
        color=[SELECTED_COLOR, "#009E73", "#E69F00"],
    )
    axes[1].axhline(0.90, color="black", ls="--", lw=1, label="nominal 90%")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("outer-LOO empirical coverage")
    axes[1].set_title("Strict-LOO empirical interval coverage")
    axes[1].legend(frameon=False, fontsize=8)
    _save(figure, figure_dir / "Fig5_confidence_audit")


def plot_renderer_strata(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    selected_method: str,
) -> None:
    ordered = rq.sort_values("true_target").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, 5).round().astype(int)
    groups = list(ordered.iloc[positions]["growth_run_id"].astype(str))
    lookup = ordered.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups),
        4,
        figsize=(10.0, 2.45 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, group in enumerate(groups):
        rheed = _rheed_keyframe(phase1, group)
        baseline, _ = _generated(output, split="crossfit", method=M10, group=group)
        selected, _ = _generated(
            output,
            split="crossfit",
            method=selected_method,
            group=group,
        )
        real = _real_afm(phase1, group)
        real_label = _real_afm_label(phase1, group)
        scale = np.concatenate([baseline.ravel(), selected.ravel(), real.ravel()])
        vmin, vmax = np.quantile(scale, [0.01, 0.99])
        axes[row_index, 0].imshow(rheed, cmap="gray")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        axes[row_index, 0].set_title(
            f"{group} RHEED\nsample median Sq={lookup.loc[group, 'true_target']:.2f} nm"
        )
        _surface_panel(
            axes[row_index, 1],
            baseline,
            vmin=float(vmin),
            vmax=float(vmax),
            title=f"M10 baseline\npredicted Sq="
            f"{lookup.loc[group, 'predicted_target']:.2f} nm",
        )
        image = _surface_panel(
            axes[row_index, 2],
            selected,
            vmin=float(vmin),
            vmax=float(vmax),
            title=_method_label(selected_method),
        )
        _surface_panel(
            axes[row_index, 3],
            real,
            vmin=float(vmin),
            vmax=float(vmax),
            title=(
                "measured AFM\n"
                f"displayed scan Sq="
                f"{real_label['displayed_scan_sq_nm']:.2f} nm\n"
                f"sample median Sq="
                f"{real_label['sample_median_sq_nm']:.2f} ± "
                f"{real_label['sample_sq_iqr_nm']:.2f} nm (IQR)"
            ),
        )
        cb = figure.colorbar(
            image,
            ax=axes[row_index, 1:],
            fraction=0.025,
            pad=0.01,
        )
        cb.set_label("height (nm)")
    figure.suptitle(
        "Fixed areal-roughness strata: renderer comparison under the same "
        "held-one conditioning",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_dir / "Fig6_renderer_roughness_strata")


def plot_largest_failures(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
) -> None:
    groups = list(rq.nlargest(4, "absolute_error")["growth_run_id"].astype(str))
    figure = _comparison_figure(
        groups=groups,
        split="crossfit",
        output=output,
        phase1=phase1,
        rq_predictions=rq,
        fsmi_predictions=fsmi,
        confidence=confidence,
        method=method,
        title=(
            "Four largest Sq failures in full-cohort LOO "
            "(reported without cherry-picking)"
        ),
    )
    _save(figure, figure_dir / "Fig7_largest_error_cases")


def plot_highlighted_growths(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
    groups: list[str],
) -> None:
    if not groups:
        return
    available = set(rq["growth_run_id"].astype(str))
    missing = sorted(set(groups) - available)
    if missing:
        raise RuntimeError(
            f"highlighted growths are absent from predictions: {missing}"
        )
    figure = _comparison_figure(
        groups=groups,
        split="crossfit",
        output=output,
        phase1=phase1,
        rq_predictions=rq,
        fsmi_predictions=fsmi,
        confidence=confidence,
        method=method,
        title=(
            "Second-batch extra-five strict LOO: automatic RHEED → "
            "generated AFM → measured 1 µm AFM subfield"
        ),
    )
    _save(figure, figure_dir / "Fig8_extra_five_generated_afm")


def plot_highlighted_renderer_comparison(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    groups: list[str],
    selected_method: str,
) -> None:
    if not groups:
        return
    lookup = rq.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups),
        4,
        figsize=(11.0, 2.75 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, group in enumerate(groups):
        rheed = _rheed_keyframe(phase1, group)
        m10, _ = _generated(output, split="crossfit", method=M10, group=group)
        selected, _ = _generated(
            output,
            split="crossfit",
            method=selected_method,
            group=group,
        )
        real = _real_afm(phase1, group)
        real_label = _real_afm_label(phase1, group)
        scale = np.concatenate([m10.ravel(), selected.ravel(), real.ravel()])
        vmin, vmax = np.quantile(scale, [0.01, 0.99])
        axes[row_index, 0].imshow(rheed, cmap="gray")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        axes[row_index, 0].set_title(
            f"{group} RHEED\nmeasured median Sq="
            f"{lookup.loc[group, 'true_target']:.2f} nm",
            fontsize=8.2,
        )
        _surface_panel(
            axes[row_index, 1],
            m10,
            vmin=float(vmin),
            vmax=float(vmax),
            title=(
                "M10 dense-island spectral\n"
                f"predicted Sq={lookup.loc[group, 'predicted_target']:.2f} nm"
            ),
        )
        image = _surface_panel(
            axes[row_index, 2],
            selected,
            vmin=float(vmin),
            vmax=float(vmax),
            title=_method_label(selected_method),
        )
        _surface_panel(
            axes[row_index, 3],
            real,
            vmin=float(vmin),
            vmax=float(vmax),
            title=(
                "measured AFM\n"
                f"scan Sq={real_label['displayed_scan_sq_nm']:.2f} nm; "
                f"median={real_label['sample_median_sq_nm']:.2f} ± "
                f"{real_label['sample_sq_iqr_nm']:.2f} nm IQR"
            ),
        )
        colorbar = figure.colorbar(
            image,
            ax=axes[row_index, 1:],
            fraction=0.025,
            pad=0.01,
        )
        colorbar.set_label("height (nm)")
    figure.suptitle(
        "Second-batch extra-five renderer comparison under identical "
        "strict-LOO scalar conditioning",
        fontsize=11,
        fontweight="bold",
    )
    _save(figure, figure_dir / "Fig9_extra_five_renderer_comparison")


def run(config: dict[str, Any]) -> None:
    _style()
    suffix = str(config.get("full_run_suffix", "full23_loo"))
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    figure_dir = report / "figures"
    tables = load_source_tables(config)
    phase1 = tables["phase1"].copy()
    rq = _read(report / "rq_crossfit_predictions.csv")
    fsmi = _read(report / "fsmi_crossfit_predictions.csv")
    confidence = _read(report / "confidence_crossfit.csv")
    if config.get("external_confidence_predictions"):
        confidence = _external_target_confidence(
            path=config["external_confidence_predictions"],
            fallback=confidence,
            method=str(
                config.get(
                    "external_confidence_method",
                    "M14i_target_specific_robust",
                )
            ),
            fsmi_path=config.get("external_fsmi_confidence_predictions"),
            fsmi_method=config.get("external_fsmi_confidence_method"),
        )
    cohort = _read(report / "cohort_manifest.csv")
    cohort_count = int(cohort["growth_run_id"].nunique())
    if "cohort_origin" not in cohort.columns:
        extra_batch = set(map(str, config.get("extra_batch_growths", [])))
        cohort["cohort_origin"] = np.where(
            cohort["growth_run_id"].astype(str).isin(extra_batch),
            "extra_five_batch",
            "original_23_batch",
        )
    comparison = pd.read_csv(report / "comparison_to_prior15_targets.csv")
    selected = str(config["selected_method"])

    stems = plot_full_atlas(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        method=selected,
        cohort_count=cohort_count,
    )
    plot_target_scatter(
        figure_dir=figure_dir,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        source=cohort,
        cohort_count=cohort_count,
    )
    plot_protocol_comparison(
        figure_dir=figure_dir,
        comparison=comparison,
        current_method_label=str(config.get("target_prediction_method", "M13")).split(
            "_", maxsplit=1
        )[0],
        cohort_count=cohort_count,
    )
    plot_rq_order(
        figure_dir=figure_dir,
        rq=rq,
        confidence=confidence,
        cohort_count=cohort_count,
    )
    plot_confidence_audit(
        figure_dir=figure_dir,
        confidence=confidence,
    )
    plot_renderer_strata(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        selected_method=selected,
    )
    plot_largest_failures(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        method=selected,
    )
    plot_highlighted_growths(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        method=selected,
        groups=list(map(str, config.get("extra_batch_growths", []))),
    )
    plot_highlighted_renderer_comparison(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        groups=list(map(str, config.get("extra_batch_growths", []))),
        selected_method=selected,
    )
    write_json(
        {
            "experiment_id": config["experiment_id"],
            "figure_directory": str(figure_dir),
            "atlas_stems": stems,
            "outer_growth_count": int(rq["growth_run_id"].nunique()),
            "fit_growths_per_fold": cohort_count - 1,
            "ordering": "ascending measured Sq across atlas pages",
            "png_count": len(list(figure_dir.glob("*.png"))),
            "pdf_count": len(list(figure_dir.glob("*.pdf"))),
            "height_units": "nm",
            "scan_size_nm": float(config["scan_size_nm"]),
            "confidence_warning": (
                "Confidence is a relative index, not a calibrated probability."
            ),
            "confidence_vs_realized_target_error_spearman": float(
                spearmanr(
                    confidence["joint_confidence_index"],
                    confidence["realized_joint_error_index"],
                ).statistic
            ),
        },
        report / "visualization_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_full_cohort_loo.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
