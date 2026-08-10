"""Build the manuscript-ready Nano Letters M20+M22c figure package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from PIL import Image

from analysis.rheed_rough_island_redesign.gwyddion_atlas import (
    _height_ticks,
    gwyddion_net_colormap,
    individual_height_limits,
)
from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _phase_row,
    _real_afm,
    _real_afm_label,
    _rheed_keyframe,
)
from analysis.rheed_video_afm_story.afm_descriptors import radial_psd
from analysis.rheed_video_afm_story.common import repo_path

MODEL = "M20 spot-connectivity Sq + M22c gap-completion AFM"
METHOD = "M22c_gap_completion_strong"
CONFIG_PATH = Path("configs/morphmbe_m22.json")
VALIDATION_SCRIPT = Path("scripts/validate_nanoletters_m22_figure_package.py")
DEFAULT_OUTPUT = Path("artifacts/nanoletters_m22")
PUBLIC_ID = {
    "6022": "01",
    "6028": "02",
    "6029": "03",
    "6033": "04",
    "6047": "05",
    "6048": "06",
    "6056": "07",
    "6057": "08",
    "6062": "09",
    "6063": "10",
    "6070": "11",
    "6072": "12",
    "6078": "13",
    "6080": "14",
    "6082": "15",
    "6084": "16",
    "6085": "17",
    "6090": "18",
    "6094": "19",
    "6095": "20",
    "6099": "21",
    "6101": "22",
    "N6342": "23",
    "N6358": "24",
    "N6382": "25",
    "N6389": "26",
    "N6390": "27",
}

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GOLD = "#E69F00"
INK = "#202124"
MUTED = "#667085"
GRID = "#D8DEE6"
PALE_BLUE = "#EAF3F8"
PALE_GREEN = "#EAF6F2"
PALE_MAGENTA = "#FAEEF5"
PALE_GOLD = "#FFF6E3"


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.1,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
        "pdf": stem.with_suffix(".pdf"),
        "svg": stem.with_suffix(".svg"),
    }
    figure.savefig(outputs["png"], dpi=600, facecolor="white")
    figure.savefig(
        outputs["tiff"],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(outputs["pdf"], facecolor="white")
    figure.savefig(outputs["svg"], facecolor="white")
    plt.close(figure)
    return {kind: str(path) for kind, path in outputs.items()}


def _panel_label(figure: plt.Figure, label: str, x: float, y: float) -> None:
    figure.text(
        x,
        y,
        label,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )


def _diagram_axis(figure: plt.Figure) -> plt.Axes:
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return axis


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    edge: str,
    fill: str,
    fontsize: float = 7.0,
    weight: str = "normal",
) -> patches.Rectangle:
    x, y = xy
    width, height = size
    rectangle = patches.Rectangle(
        (x, y),
        width,
        height,
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.0,
        zorder=2,
    )
    axis.add_patch(rectangle)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.08,
        zorder=3,
    )
    return rectangle


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    width: float = 1.0,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": width,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
        },
        zorder=1,
    )


def _clip_strip(phase1: pd.DataFrame, group: str) -> np.ndarray:
    row = _phase_row(phase1, group)
    paths = json.loads(str(row["clip_frame_paths"]))
    if not paths:
        payload = np.load(repo_path(str(row["clip_cache_path"])), allow_pickle=False)
        frames = np.asarray(payload["frames_uint8"])
    else:
        frames = np.stack(
            [np.asarray(Image.open(repo_path(path)).convert("L")) for path in paths]
        )
    chosen = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    panels: list[np.ndarray] = []
    separator = np.zeros((frames.shape[1], max(frames.shape[2] // 16, 3)))
    for index, frame_index in enumerate(chosen):
        panels.append(frames[frame_index])
        if index < len(chosen) - 1:
            panels.append(separator)
    return np.concatenate(panels, axis=1)


def _generated_map(output: Path, group: str, draw: int = 0) -> tuple[np.ndarray, float]:
    path = output / "crossfit/generated_maps" / METHOD / f"{group}.npz"
    with np.load(path, allow_pickle=False) as payload:
        unit = np.asarray(payload["generated_unit_shapes"][draw], dtype=float)
        predicted_sq = float(payload["predicted_rq_nm"])
        retrieval = bool(payload["retrieval_at_inference"])
        measured_used = bool(payload["measured_afm_patch_used_at_inference"])
    if retrieval or measured_used:
        raise RuntimeError(f"inference boundary violated for {group}")
    return unit * predicted_sq, predicted_sq


def _outlined_scale_bar(axis: plt.Axes, pixels: int) -> None:
    width = 0.25 * pixels
    y = 0.91 * pixels
    x0 = 0.67 * pixels
    axis.plot([x0, x0 + width], [y, y], color="black", lw=3.5)
    axis.plot([x0, x0 + width], [y, y], color="white", lw=2.0)
    axis.text(
        x0 + width / 2,
        y - 0.035 * pixels,
        "250 nm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=5.3,
        path_effects=[
            path_effects.Stroke(linewidth=1.6, foreground="black"),
            path_effects.Normal(),
        ],
    )


def _surface(
    figure: plt.Figure,
    position: list[float],
    array: np.ndarray,
    *,
    title: str | None = None,
    title_size: float = 6.5,
    colorbar: bool = True,
    compact_bar: bool = False,
) -> plt.Axes:
    axis = figure.add_axes(position)
    low, high = individual_height_limits(array)
    image = axis.imshow(
        array,
        cmap=gwyddion_net_colormap(),
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    if title:
        axis.set_title(title, fontsize=title_size, pad=2.0, linespacing=1.04)
    axis.set_xticks([])
    axis.set_yticks([])
    _outlined_scale_bar(axis, array.shape[1])
    if colorbar:
        gap = 0.006 if compact_bar else 0.008
        width = 0.008 if compact_bar else 0.010
        cax = figure.add_axes(
            [position[0] + position[2] + gap, position[1], width, position[3]]
        )
        bar = figure.colorbar(image, cax=cax)
        bar.set_ticks(_height_ticks(low, high))
        bar.ax.tick_params(labelsize=4.7 if compact_bar else 5.5, length=1.8)
        if not compact_bar:
            bar.set_label("height (nm)", fontsize=5.8, labelpad=1.5)
    return axis


def _rheed_panel(
    figure: plt.Figure,
    position: list[float],
    array: np.ndarray,
    *,
    title: str | None = None,
    overlay: str | None = None,
) -> plt.Axes:
    axis = figure.add_axes(position)
    axis.imshow(array, cmap="gray", interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, fontsize=6.5, pad=2.0)
    if overlay:
        axis.text(
            0.02,
            0.98,
            overlay,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.7,
            color="white",
            fontweight="bold",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.58, "pad": 1.6},
        )
    return axis


def _normalized_psd(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency, power = radial_psd(array)
    normalized = power / max(float(np.sum(power)), 1e-12)
    return frequency, normalized


def _psd_panel(
    axis: plt.Axes,
    generated: np.ndarray,
    measured: np.ndarray,
    *,
    legend: bool,
) -> None:
    generated_frequency, generated_power = _normalized_psd(generated)
    measured_frequency, measured_power = _normalized_psd(measured)
    axis.plot(generated_frequency, generated_power, color=BLUE, label="generated")
    axis.plot(
        measured_frequency,
        measured_power,
        color=VERMILLION,
        linestyle="--",
        label="measured",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(color=GRID, lw=0.45, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if legend:
        axis.legend(frameon=False, loc="lower left", handlelength=2.2)


def _load_data(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = repo_path(config["output_root"]) / str(config["full_run_suffix"])
    report = repo_path(config["report_root"]) / str(config["full_run_suffix"])
    phase1 = pd.read_csv(
        repo_path(config["phase1_manifest"]), dtype={"growth_run_id": str}
    )
    sq = pd.read_csv(
        report / "rq_crossfit_predictions.csv", dtype={"growth_run_id": str}
    )
    confidence = pd.read_csv(
        report / "confidence_crossfit.csv", dtype={"growth_run_id": str}
    )
    manifest = json.loads(
        (report / "best_model_manifest.json").read_text(encoding="utf-8")
    )
    groups = set(sq["growth_run_id"].astype(str))
    if groups != set(PUBLIC_ID):
        raise RuntimeError("public mapping does not match the M22 27-growth cohort")
    if str(manifest["selected_method"]) != METHOD:
        raise RuntimeError("the frozen M22c method is not selected")
    if bool(manifest["retrieval_at_inference"]):
        raise RuntimeError("retrieval is active in the selected manifest")
    if bool(sq["outer_target_used_for_training"].astype(bool).any()):
        raise RuntimeError("outer target leakage detected")
    return {
        "config": config,
        "config_path": config_path,
        "output": output,
        "report": report,
        "phase1": phase1,
        "sq": sq,
        "confidence": confidence,
        "manifest": manifest,
    }


def _figure_1(data: dict[str, Any], stem: Path) -> dict[str, str]:
    phase1 = data["phase1"]
    output = data["output"]
    confidence = data["confidence"].set_index("growth_run_id")
    group = "N6342"
    public = PUBLIC_ID[group]
    generated, predicted_sq = _generated_map(output, group)
    measured = _real_afm(phase1, group)
    clip = _clip_strip(phase1, group)

    figure = plt.figure(figsize=(7.0, 4.45))
    canvas = _diagram_axis(figure)

    _panel_label(figure, "a", 0.015, 0.985)
    figure.text(
        0.04,
        0.955,
        "In-situ acquisition and causal event selection",
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _box(
        canvas,
        (0.045, 0.785),
        (0.10, 0.085),
        "MBE growth\nepilayer",
        edge=INK,
        fill="#DCEBE4",
        weight="bold",
    )
    _arrow(canvas, (0.148, 0.827), (0.205, 0.827))
    clip_axis = figure.add_axes([0.205, 0.77, 0.225, 0.115])
    clip_axis.imshow(clip, cmap="gray")
    clip_axis.set_xticks([])
    clip_axis.set_yticks([])
    clip_axis.set_title("RHEED stream", fontsize=8.0, fontweight="bold", pad=2)
    clip_axis.text(
        0.5,
        -0.15,
        f"real 16-frame clip | Sample {public}",
        transform=clip_axis.transAxes,
        ha="center",
        color=MUTED,
        fontsize=6.2,
    )
    _arrow(canvas, (0.438, 0.827), (0.485, 0.827))
    _box(
        canvas,
        (0.485, 0.775),
        (0.135, 0.105),
        "automatic ROI +\nclear-moment\ndetection",
        edge=GREEN,
        fill=PALE_GREEN,
        weight="bold",
    )
    canvas.text(
        0.552,
        0.758,
        "causal, bounded latency",
        ha="center",
        va="top",
        fontsize=6.2,
        color=GREEN,
    )
    _arrow(canvas, (0.624, 0.827), (0.67, 0.827))
    accepted_axis = figure.add_axes([0.67, 0.77, 0.18, 0.115])
    accepted_axis.imshow(clip, cmap="gray")
    accepted_axis.set_xticks([])
    accepted_axis.set_yticks([])
    accepted_axis.set_title("orientation-locked input", fontsize=7.2, pad=2)
    _arrow(canvas, (0.855, 0.827), (0.895, 0.827))
    _box(
        canvas,
        (0.895, 0.79),
        (0.075, 0.075),
        "model\ninput",
        edge=BLUE,
        fill=PALE_BLUE,
        weight="bold",
    )

    _panel_label(figure, "b", 0.015, 0.70)
    figure.text(
        0.04,
        0.67,
        "Hybrid physics-AI inference",
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _box(
        canvas,
        (0.055, 0.51),
        (0.13, 0.095),
        "16-frame\nRHEED clip",
        edge=BLUE,
        fill=PALE_BLUE,
        fontsize=7.5,
    )
    _arrow(canvas, (0.185, 0.57), (0.255, 0.57), color=BLUE)
    _arrow(canvas, (0.185, 0.545), (0.255, 0.49), color=GREEN)
    _box(
        canvas,
        (0.255, 0.555),
        (0.16, 0.08),
        "R3D-18 temporal\nembedding",
        edge=BLUE,
        fill=PALE_BLUE,
        weight="bold",
    )
    _box(
        canvas,
        (0.255, 0.445),
        (0.16, 0.08),
        "endpoint streak +\nspot-connectivity features",
        edge=GREEN,
        fill=PALE_GREEN,
        weight="bold",
    )
    _arrow(canvas, (0.415, 0.595), (0.50, 0.56), color=BLUE)
    _arrow(canvas, (0.415, 0.485), (0.50, 0.53), color=GREEN)
    _box(
        canvas,
        (0.50, 0.49),
        (0.18, 0.11),
        "cross-fitted heads\nM20 Sq | condition z\nFSMI | reliability",
        edge=MAGENTA,
        fill=PALE_MAGENTA,
        weight="bold",
    )
    _arrow(canvas, (0.68, 0.545), (0.75, 0.545))
    _box(
        canvas,
        (0.75, 0.49),
        (0.19, 0.11),
        "M22c non-retrieval\nlayered-island +\ngap-completion generator",
        edge=GOLD,
        fill=PALE_GOLD,
        weight="bold",
    )
    canvas.text(
        0.60,
        0.455,
        "Measured AFM is unavailable to the model at inference",
        ha="center",
        va="top",
        fontsize=6.8,
        color=VERMILLION,
        fontweight="bold",
    )

    _panel_label(figure, "c", 0.015, 0.405)
    figure.text(
        0.04,
        0.375,
        "Quantitative output",
        fontsize=9.4,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _surface(
        figure,
        [0.075, 0.075, 0.15, 0.22],
        generated,
        title=f"M22c generated AFM | Sample {public}\npredicted Sq {predicted_sq:.2f} nm",
    )
    canvas.text(0.39, 0.24, "Sq", ha="center", color=MUTED, fontsize=7.0)
    canvas.text(
        0.39,
        0.185,
        f"{predicted_sq:.2f} nm",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    canvas.text(
        0.58, 0.24, "cross-fitted reliability", ha="center", color=MUTED, fontsize=7.0
    )
    canvas.text(
        0.58,
        0.185,
        f"{float(confidence.loc[group, 'joint_confidence_index']):.0f} / 100",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    canvas.plot([0.29, 0.72], [0.15, 0.15], color=GRID, lw=0.8)
    canvas.text(
        0.505,
        0.125,
        "one physically scaled stochastic realization",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    _surface(
        figure,
        [0.79, 0.075, 0.15, 0.22],
        measured,
        title="measured AFM | evaluation only",
    )
    canvas.text(
        0.865,
        0.055,
        "not an inference input",
        ha="center",
        fontsize=6.4,
        color=VERMILLION,
    )
    figure.subplots_adjust(0, 0, 1, 1)
    return _save_figure(figure, stem)


def _figure_2(data: dict[str, Any], stem: Path) -> dict[str, str]:
    phase1 = data["phase1"]
    output = data["output"]
    group = "N6342"
    public = PUBLIC_ID[group]
    clip = _clip_strip(phase1, group)
    draws = [_generated_map(output, group, draw)[0] for draw in range(4)]
    predicted_sq = _generated_map(output, group)[1]

    figure = plt.figure(figsize=(7.0, 6.15))
    canvas = _diagram_axis(figure)
    _panel_label(figure, "a", 0.015, 0.99)
    figure.text(
        0.045,
        0.965,
        "RHEED representation and condition prediction",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    rheed_axis = figure.add_axes([0.055, 0.72, 0.19, 0.14])
    rheed_axis.imshow(clip, cmap="gray")
    rheed_axis.set_xticks([])
    rheed_axis.set_yticks([])
    rheed_axis.set_title(
        "causal 16-frame RHEED input", fontsize=7.1, color=MUTED, pad=3
    )
    _arrow(canvas, (0.245, 0.79), (0.29, 0.79))
    canvas.plot([0.27, 0.27], [0.71, 0.87], color=INK, lw=1.0)
    _arrow(canvas, (0.27, 0.85), (0.305, 0.885), color=BLUE)
    _arrow(canvas, (0.27, 0.79), (0.305, 0.79), color=BLUE)
    _arrow(canvas, (0.27, 0.73), (0.305, 0.695), color=GREEN)
    _box(
        canvas,
        (0.305, 0.845),
        (0.16, 0.075),
        "DINOv2 key-frame\ndescriptor",
        edge=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        canvas,
        (0.305, 0.75),
        (0.16, 0.075),
        "R3D-18 16-frame\ndescriptor",
        edge=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        canvas,
        (0.305, 0.655),
        (0.16, 0.075),
        "causal R3D + endpoint\nstreak descriptors",
        edge=GREEN,
        fill=PALE_GREEN,
    )
    _arrow(canvas, (0.465, 0.882), (0.515, 0.835), color=BLUE)
    _arrow(canvas, (0.465, 0.787), (0.515, 0.81), color=BLUE)
    _box(
        canvas,
        (0.515, 0.775),
        (0.18, 0.105),
        "hybrid condition head\nPCA + physics summaries\nPLS1 morphology shape",
        edge=MAGENTA,
        fill=PALE_MAGENTA,
        weight="bold",
    )
    _arrow(canvas, (0.695, 0.825), (0.76, 0.825), color=MAGENTA)
    _box(
        canvas,
        (0.76, 0.785),
        (0.17, 0.08),
        "9-D condition z\n(amplitude + morphology)",
        edge=GOLD,
        fill=PALE_GOLD,
        weight="bold",
    )
    _arrow(canvas, (0.465, 0.692), (0.515, 0.69), color=GREEN)
    _box(
        canvas,
        (0.515, 0.64),
        (0.18, 0.10),
        "M19 endpoint support\n+ M20 spot connectivity\nresidual and tail uplift",
        edge=GREEN,
        fill=PALE_GREEN,
        weight="bold",
    )
    _arrow(canvas, (0.695, 0.69), (0.76, 0.69), color=GREEN)
    _box(
        canvas,
        (0.76, 0.65),
        (0.17, 0.08),
        f"physical Sq\n{predicted_sq:.2f} nm",
        edge=BLUE,
        fill=PALE_BLUE,
        weight="bold",
    )
    canvas.text(
        0.60,
        0.615,
        "strict outer LOO: every scaler, neighbor model and head excludes the held growth",
        ha="center",
        fontsize=6.2,
        color=VERMILLION,
    )

    _panel_label(figure, "b", 0.015, 0.59)
    figure.text(
        0.045,
        0.565,
        "M22c layered elliptical-island and gap-completion generator",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    _box(
        canvas,
        (0.05, 0.405),
        (0.11, 0.085),
        "cross-fitted\ncondition z + Sq\n+ spot isolation",
        edge=MAGENTA,
        fill=PALE_MAGENTA,
        weight="bold",
    )
    _arrow(canvas, (0.16, 0.465), (0.205, 0.50), color=BLUE)
    _arrow(canvas, (0.16, 0.43), (0.205, 0.39), color=GREEN)
    _box(
        canvas,
        (0.205, 0.465),
        (0.13, 0.07),
        "spectral ridge +\nIAAFT prior",
        edge=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        canvas,
        (0.205, 0.355),
        (0.13, 0.08),
        "layered elliptical\nisland growth",
        edge=GREEN,
        fill=PALE_GREEN,
    )
    _arrow(canvas, (0.335, 0.50), (0.38, 0.465), color=BLUE)
    _arrow(canvas, (0.335, 0.395), (0.38, 0.43), color=GREEN)
    _box(
        canvas,
        (0.38, 0.395),
        (0.16, 0.11),
        "largest-gap filling\n+ island coalescence\nroughness-aware blend",
        edge=GOLD,
        fill=PALE_GOLD,
        fontsize=6.2,
        weight="bold",
    )
    _arrow(canvas, (0.54, 0.45), (0.575, 0.45))
    _box(
        canvas,
        (0.575, 0.41),
        (0.095, 0.08),
        f"unit Sq\n-> {predicted_sq:.2f} nm",
        edge=MAGENTA,
        fill=PALE_MAGENTA,
        weight="bold",
    )
    for index, array in enumerate(draws):
        left = 0.70 + index * 0.071
        _surface(
            figure,
            [left, 0.385, 0.052, 0.085],
            array,
            title=f"seed {index + 1}",
            title_size=5.4,
            colorbar=False,
        )
    canvas.text(
        0.835,
        0.36,
        f"4 stochastic draws | 128 x 128 | Sample {public}",
        ha="center",
        fontsize=6.1,
        color=MUTED,
    )
    canvas.text(
        0.64,
        0.335,
        "nonretrieval: measured AFM is unavailable at inference",
        ha="center",
        fontsize=6.5,
        color=VERMILLION,
        fontweight="bold",
    )

    _panel_label(figure, "c", 0.015, 0.30)
    figure.text(
        0.045,
        0.275,
        "Growth-level leave-one-out validation",
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    x_values = np.linspace(0.075, 0.61, 27)
    held_index = 22
    canvas.plot([x_values[0], x_values[-1]], [0.16, 0.16], color=INK, lw=0.7)
    for index, x_value in enumerate(x_values):
        canvas.scatter(
            x_value,
            0.16,
            s=19 if index == held_index else 13,
            color=VERMILLION if index == held_index else BLUE,
            edgecolors="none",
            zorder=3,
        )
    canvas.text(
        0.30,
        0.19,
        "26 growths fit",
        ha="center",
        color=BLUE,
        fontsize=7.0,
        fontweight="bold",
    )
    canvas.text(
        x_values[held_index],
        0.125,
        f"held once\nSample {public}",
        ha="center",
        color=VERMILLION,
        fontsize=6.5,
        fontweight="bold",
    )
    _arrow(canvas, (0.63, 0.16), (0.69, 0.16))
    _box(
        canvas,
        (0.69, 0.12),
        (0.12, 0.08),
        "predict without\nheld AFM",
        edge=VERMILLION,
        fill="#FFF0EB",
        weight="bold",
    )
    _arrow(canvas, (0.81, 0.16), (0.85, 0.16))
    _box(
        canvas,
        (0.85, 0.12),
        (0.10, 0.08),
        "compare\nafterward",
        edge=MUTED,
        fill="#F4F5F7",
        weight="bold",
    )
    canvas.text(
        0.50,
        0.065,
        "27 outer folds | growth groups are leakage boundaries | 6081 excluded before fitting",
        ha="center",
        fontsize=6.4,
        color=MUTED,
    )
    figure.subplots_adjust(0, 0, 1, 1)
    return _save_figure(figure, stem)


def _scatter_and_line(
    figure: plt.Figure,
    scatter_position: list[float],
    line_position: list[float],
    ordered: pd.DataFrame,
    examples: list[str],
) -> None:
    truth = ordered["true_target"].to_numpy(float)
    predicted = ordered["predicted_target"].to_numpy(float)
    lower = ordered["interval_lower"].to_numpy(float)
    upper = ordered["interval_upper"].to_numpy(float)
    residual = predicted - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    correlation = float(np.corrcoef(truth, predicted)[0, 1])

    axis = figure.add_axes(scatter_position)
    limit = max(float(np.max(truth)), float(np.max(upper))) + 0.35
    axis.plot([0, limit], [0, limit], color=INK, lw=0.9, linestyle="--", zorder=0)
    axis.errorbar(
        truth,
        predicted,
        yerr=np.vstack([predicted - lower, upper - predicted]),
        fmt="none",
        ecolor="#B9C0C9",
        elinewidth=0.55,
        alpha=0.65,
        zorder=1,
    )
    axis.scatter(
        truth,
        predicted,
        s=18,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=0.9,
        zorder=2,
    )
    for group in examples:
        row = ordered.loc[ordered["growth_run_id"] == group].iloc[0]
        axis.scatter(
            row["true_target"],
            row["predicted_target"],
            s=28,
            color=GREEN,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axis.annotate(
            PUBLIC_ID[group],
            (row["true_target"], row["predicted_target"]),
            xytext=(3, 3),
            textcoords="offset points",
            color=GREEN,
            fontsize=6.1,
            fontweight="bold",
        )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("measured Sq (nm)")
    axis.set_ylabel("predicted Sq (nm)")
    axis.set_title(
        f"linear agreement | n = 27 | r = {correlation:.2f} | MAE = {mae:.2f} nm",
        fontsize=7.0,
    )
    axis.spines[["top", "right"]].set_visible(False)

    line_axis = figure.add_axes(line_position)
    x = np.arange(1, len(ordered) + 1)
    line_axis.fill_between(x, lower, upper, color=GOLD, alpha=0.15, linewidth=0)
    line_axis.plot(
        x, truth, color=INK, marker="o", ms=2.8, label="measured Sq", zorder=3
    )
    line_axis.plot(
        x,
        predicted,
        color=VERMILLION,
        marker="D",
        ms=2.4,
        label="predicted Sq",
        zorder=4,
    )
    line_axis.set_xlabel("sample index (ordered by measured Sq)")
    line_axis.set_ylabel("Sq (nm)")
    line_axis.set_xlim(0.3, len(ordered) + 0.7)
    line_axis.set_xticks(x)
    line_axis.set_xticklabels(
        [f"{index:02d}" for index in x], rotation=90, fontsize=4.8
    )
    line_axis.grid(axis="y", color=GRID, lw=0.45)
    line_axis.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.7)
    line_axis.set_title(f"ordered Sq profile | RMSE = {rmse:.2f} nm", fontsize=7.0)
    line_axis.spines[["top", "right"]].set_visible(False)


def _figure_3(data: dict[str, Any], stem: Path) -> dict[str, str]:
    phase1 = data["phase1"]
    output = data["output"]
    sq = data["sq"].copy()
    confidence = data["confidence"].set_index("growth_run_id")
    examples = ["N6342", "6063", "6099"]
    labels = ["smooth", "intermediate", "rough"]

    figure = plt.figure(figsize=(7.0, 8.35))
    _panel_label(figure, "a", 0.015, 0.995)
    column_titles = [
        "RHEED key frame",
        "M22c generated AFM",
        "measured AFM",
        "normalized radial PSD",
    ]
    for x_value, title in zip([0.075, 0.30, 0.535, 0.79], column_titles, strict=False):
        figure.text(x_value + 0.08, 0.967, title, ha="center", va="top", fontsize=7.7)
    for row_index, (group, regime) in enumerate(zip(examples, labels, strict=False)):
        y = 0.745 - row_index * 0.225
        rheed = _rheed_keyframe(phase1, group)
        generated, predicted_sq = _generated_map(output, group)
        measured = _real_afm(phase1, group)
        real_label = _real_afm_label(phase1, group)
        public = PUBLIC_ID[group]
        _rheed_panel(
            figure,
            [0.055, y, 0.17, 0.17],
            rheed,
            overlay=f"Sample {public}\n{regime}",
        )
        _surface(
            figure,
            [0.285, y, 0.16, 0.17],
            generated,
            title=f"pred. Sq {predicted_sq:.2f} nm\nC {float(confidence.loc[group, 'joint_confidence_index']):.0f}/100",
            colorbar=True,
            compact_bar=True,
        )
        _surface(
            figure,
            [0.525, y, 0.16, 0.17],
            measured,
            title=f"meas. Sq {real_label['displayed_scan_sq_nm']:.2f} nm",
            colorbar=True,
            compact_bar=True,
        )
        psd_axis = figure.add_axes([0.775, y + 0.005, 0.19, 0.16])
        _psd_panel(psd_axis, generated, measured, legend=row_index == 0)
        if row_index < 2:
            psd_axis.set_xticklabels([])
        psd_axis.set_ylabel("normalized PSD")

    ordered = sq.sort_values("true_target").reset_index(drop=True)
    _panel_label(figure, "b", 0.06, 0.265)
    _panel_label(figure, "c", 0.515, 0.265)
    _scatter_and_line(
        figure,
        [0.09, 0.045, 0.34, 0.20],
        [0.55, 0.045, 0.41, 0.20],
        ordered,
        examples,
    )
    return _save_figure(figure, stem)


def _figure_4(data: dict[str, Any], stem: Path) -> dict[str, str]:
    phase1 = data["phase1"]
    output = data["output"]
    ordered = data["sq"].sort_values("true_target").reset_index(drop=True)
    confidence = data["confidence"].set_index("growth_run_id")
    figure = plt.figure(figsize=(8.5, 26.0))
    figure.text(
        0.045,
        0.991,
        "Full-cohort outer-LOO atlas ordered by measured Sq",
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )
    figure.text(
        0.955,
        0.991,
        "27 growths | held growth excluded from every fit | Samples 01-27",
        ha="right",
        va="top",
        fontsize=6.1,
        color=MUTED,
    )
    headers = [
        (0.09, "RHEED key frame"),
        (0.285, "M22c generated AFM"),
        (0.50, "measured AFM\nevaluation only"),
        (0.775, "normalized radial PSD"),
    ]
    for x_value, text_value in headers:
        figure.text(
            x_value,
            0.978,
            text_value,
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
        )

    top = 0.958
    bottom = 0.025
    row_height = (top - bottom) / len(ordered)
    image_height = row_height * 0.78
    for row_index, row in ordered.iterrows():
        group = str(row["growth_run_id"])
        y = top - (row_index + 1) * row_height + row_height * 0.12
        public = PUBLIC_ID[group]
        rheed = _rheed_keyframe(phase1, group)
        generated, predicted_sq = _generated_map(output, group)
        measured = _real_afm(phase1, group)
        real_label = _real_afm_label(phase1, group)
        _rheed_panel(
            figure,
            [0.055, y, 0.075, image_height],
            rheed,
            overlay=f"Sample {public}\nmeas. Sq {float(row['true_target']):.2f} nm",
        )
        _surface(
            figure,
            [0.245, y, 0.075, image_height],
            generated,
            title=f"pred. Sq {predicted_sq:.2f} nm\nC {float(confidence.loc[group, 'joint_confidence_index']):.0f}/100",
            title_size=4.8,
            compact_bar=True,
        )
        _surface(
            figure,
            [0.46, y, 0.075, image_height],
            measured,
            title=f"scan Sq {real_label['displayed_scan_sq_nm']:.2f} nm",
            title_size=4.8,
            compact_bar=True,
        )
        psd_axis = figure.add_axes([0.68, y + 0.003, 0.275, image_height * 0.92])
        _psd_panel(psd_axis, generated, measured, legend=row_index == 0)
        psd_axis.tick_params(labelsize=4.4, length=1.8)
        if row_index < len(ordered) - 1:
            psd_axis.set_xticklabels([])
        else:
            psd_axis.set_xlabel("radial spatial frequency (um$^{-1}$)", fontsize=5.3)
        if row_index == len(ordered) // 2:
            psd_axis.set_ylabel("normalized PSD", fontsize=5.3)
        figure.add_artist(
            plt.Line2D(
                [0.045, 0.96],
                [y - row_height * 0.09, y - row_height * 0.09],
                transform=figure.transFigure,
                color="#E7E9ED",
                lw=0.45,
            )
        )
    figure.text(
        0.5,
        0.008,
        "Rows use public labels only. Generated and measured AFMs have independent physical height bars. Each M22c map is the first saved stochastic realization from its strict outer-LOO fold.",
        ha="center",
        fontsize=5.5,
        color=MUTED,
    )
    return _save_figure(figure, stem)


def _captions(metrics: dict[str, float]) -> str:
    return f"""# Manuscript-ready figure captions

**Figure 1. Overview of the AutoRHEED M20+M22c framework.** (a) During molecular beam epitaxy, the in-situ RHEED stream is converted into an automatically localized, orientation-locked 16-frame clip by causal region-of-interest and clear-moment detection. The displayed input is from Sample 23. (b) Hybrid physics-AI inference combines an R3D-18 temporal embedding with endpoint streak and spot-connectivity descriptors. Strictly cross-fitted heads estimate Sq, the morphology condition, FSMI, and reliability before conditioning the nonretrieval M22c stochastic AFM generator. Measured AFM is unavailable at inference. (c) One physically scaled M22c realization is shown for Sample 23 together with its predicted Sq. The measured AFM is revealed only afterward for evaluation. Every AFM has an independent physical height bar using the Gwyddion.net black-rust-gold-white palette. AFM scan width, 1.0 um; scale bars, 250 nm.

**Figure 2. Physics-guided layered-island generation and leakage-controlled validation.** (a) Real RHEED frames are represented by DINOv2/R3D descriptors and causal endpoint-streak features. The morphology head predicts the nine-dimensional condition vector. The M20 Sq head combines M19 endpoint support with target-blind spot-connectivity features - component merge rate, count, roundness, and area - to correct the rough tail without using the held AFM target. (b) M22c combines a conditional spectral/IAAFT prior with layered elliptical-island growth, largest-gap completion, coalescence, and a roughness-aware blend. Unit-Sq surfaces are scaled to the M20-predicted physical Sq. Four stochastic draws are shown for Sample 23. (c) In each of 27 outer leave-one-growth-out folds, one complete growth group is held out, the remaining 26 growths are fitted, and prediction precedes comparison with AFM. Growth group is the leakage boundary; growth 6081 was excluded before fitting.

**Figure 3. Selected cross-validated M22c predictions and cohort-wide Sq agreement.** (a) Strict outer leave-one-growth-out examples span smooth Sample 23, intermediate Sample 10, and rough Sample 21. Each row contains a real automatically selected RHEED key frame, one genuine M22c stochastic AFM realization, a measured AFM scan shown only for evaluation, and normalized radially averaged power spectral densities (PSDs). Generated maps represent conditional morphology distributions and are not expected to be pixel registered to a measured scan. Every AFM has an independent physical height bar using the Gwyddion.net palette. (b) Cohort-wide measured-versus-predicted Sq scatter with the line of identity and strict outer-LOO intervals. (c) The same 27 measurements and predictions are plotted against sample index after ordering by measured Sq. M20 achieves Pearson r = {metrics["pearson_r"]:.3f}, MAE = {metrics["mae_nm"]:.3f} nm, and RMSE = {metrics["rmse_nm"]:.3f} nm. AFM scan width, 1.0 um; scale bars, 250 nm.

**Figure 4. Full-cohort RHEED-to-AFM M22c atlas ordered by measured surface roughness.** All 27 growths are arranged from top to bottom by increasing measured sample-median Sq. Each row shows the real automatically localized RHEED key frame, the first saved M22c realization from that growth's strict outer-LOO fold, the representative measured AFM scan revealed only for evaluation, independent generated and measured height bars, and normalized radial PSDs. The RHEED label reports the sample-median Sq used for ordering, whereas the measured-AFM label reports the Sq of the displayed scan. Blue solid and vermillion dashed curves denote generated and measured surfaces, respectively. Each prediction excludes the displayed growth from all fitting and uses no measured AFM or AFM retrieval at inference. Public sample labels are anonymized; the private correspondence must not be included in the manuscript or public Supporting Information.
"""


def _readme(metrics: dict[str, float]) -> str:
    return f"""# NanoLetters M22 figure package

This package replaces the M17 figure package with the frozen M20 Sq + M22c AFM method.

## Main figures

- `figures/Figure_1_AutoRHEED_M22_overview.*`
- `figures/Figure_2_M20_M22_model_and_validation.*`
- `figures/Figure_3_M22_selected_results_and_Sq.*`
- `figures/Figure_4_M22_full_cohort_atlas.*`

Each figure is supplied as 600-dpi PNG, 600-dpi LZW TIFF, vector PDF, and editable vector SVG. SVG files can be imported into Canva, Illustrator, Inkscape, or PowerPoint while retaining vector elements and text. Plot data are in `editable/data/M22_Sq_outer_LOO.csv`, and the exact source is in `editable/source/`.

## Quantitative result

Strict outer-LOO n=27: Pearson r={metrics["pearson_r"]:.3f}; MAE={metrics["mae_nm"]:.3f} nm; RMSE={metrics["rmse_nm"]:.3f} nm.

## Privacy and provenance

Main figures use public Sample 01-27 labels. The private ID mapping is stored only in `private/`. M22 is retrospective method-development evidence, not a prospectively untouched validation. Measured AFM and AFM retrieval are unavailable at inference.
"""


def _write_package_metadata(
    output_root: Path,
    data: dict[str, Any],
    generated: dict[str, dict[str, str]],
) -> dict[str, float]:
    sq = data["sq"].sort_values("true_target").reset_index(drop=True).copy()
    truth = sq["true_target"].to_numpy(float)
    predicted = sq["predicted_target"].to_numpy(float)
    metrics = {
        "pearson_r": float(np.corrcoef(truth, predicted)[0, 1]),
        "mae_nm": float(np.mean(np.abs(predicted - truth))),
        "rmse_nm": float(np.sqrt(np.mean(np.square(predicted - truth)))),
    }
    editable_data = output_root / "editable/data"
    editable_source = output_root / "editable/source"
    private = output_root / "private"
    provenance = output_root / "provenance"
    for path in (editable_data, editable_source, private, provenance):
        path.mkdir(parents=True, exist_ok=True)
    sq.insert(0, "ordered_sample_index", np.arange(1, len(sq) + 1))
    sq.insert(
        1,
        "public_sample_id",
        sq["growth_run_id"].astype(str).map(PUBLIC_ID),
    )
    public_columns = [
        "ordered_sample_index",
        "public_sample_id",
        "true_target",
        "predicted_target",
        "interval_lower",
        "interval_upper",
        "absolute_error",
        "confidence",
        "rheed_spot_isolation_score",
    ]
    sq.loc[:, public_columns].rename(
        columns={
            "true_target": "measured_Sq_nm",
            "predicted_target": "predicted_Sq_nm",
            "interval_lower": "prediction_interval_lower_nm",
            "interval_upper": "prediction_interval_upper_nm",
            "confidence": "sq_head_confidence",
        }
    ).to_csv(editable_data / "M22_Sq_outer_LOO.csv", index=False)
    pd.DataFrame(
        {
            "audience": ["internal_provenance_not_for_manuscript"] * len(PUBLIC_ID),
            "public_sample_id": list(PUBLIC_ID.values()),
            "internal_growth_run_id": list(PUBLIC_ID.keys()),
        }
    ).to_csv(private / "sample_id_mapping_internal.csv", index=False)
    shutil.copy2(Path(__file__), editable_source / Path(__file__).name)
    shutil.copy2(
        repo_path(VALIDATION_SCRIPT),
        editable_source / VALIDATION_SCRIPT.name,
    )
    (output_root / "figure_captions.md").write_text(
        _captions(metrics), encoding="utf-8"
    )
    (output_root / "README.md").write_text(_readme(metrics), encoding="utf-8")
    manifest = {
        "package": output_root.name,
        "model": MODEL,
        "selected_method": METHOD,
        "target_prediction_method": data["manifest"]["target_prediction_method"],
        "growth_count": len(sq),
        "metrics": metrics,
        "config": str(data["config_path"]),
        "config_sha256": _sha256(data["config_path"]),
        "retrieval_at_inference": False,
        "measured_afm_patch_at_inference": False,
        "outer_target_used_for_training": False,
        "figures": {
            figure: {
                kind: str(Path(path).resolve().relative_to(output_root.resolve()))
                for kind, path in formats.items()
            }
            for figure, formats in generated.items()
        },
        "editable_formats": ["SVG", "CSV", "Python source"],
    }
    (provenance / "figure_package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def build(config_path: Path, output_root: Path) -> None:
    _style()
    data = _load_data(config_path)
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = {
        "Figure_1": _figure_1(data, figure_dir / "Figure_1_AutoRHEED_M22_overview"),
        "Figure_2": _figure_2(
            data, figure_dir / "Figure_2_M20_M22_model_and_validation"
        ),
        "Figure_3": _figure_3(
            data, figure_dir / "Figure_3_M22_selected_results_and_Sq"
        ),
        "Figure_4": _figure_4(data, figure_dir / "Figure_4_M22_full_cohort_atlas"),
    }
    metrics = _write_package_metadata(output_root, data, generated)
    print(
        json.dumps(
            {
                "status": "built",
                "output": str(output_root),
                "figures": len(generated),
                "formats_per_figure": 4,
                "metrics": metrics,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output folder is not empty: {output}")
    build(args.config.resolve(), output)


if __name__ == "__main__":
    main()
