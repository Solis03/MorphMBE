#!/usr/bin/env python3
"""Create Nano Letters-ready figures for the MorphMBE M17 study.

The script uses only frozen, auditable repository snapshots:

* real RHEED clips selected by the automatic pipeline;
* measured, third-order line-flattened AFM height arrays; and
* strict leave-one-growth-out M17b generated height fields.

Microscopy data are never synthesized for illustration.  AFM height fields are
rendered with the Gwyddion 2.71 ``Gold`` gradient and explicit nanometre
colorbars.  The manuscript-facing labels are anonymized as Sample 01--27.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "nanoletters_m17_figures_20260806"
SOURCE_ROOT = REPORT_ROOT / "source_snapshots"
FIGURE_ROOT = REPORT_ROOT / "figures"
M17_REPORT = (
    REPO_ROOT
    / "reports"
    / "rheed_m17_end_to_end_generation"
    / "20260804_m17_sparse_topology_line3_full27_v1"
    / "full27_loo"
)

SELECTED_METHOD = "M17b_topology_sparse_peak_terrace"
GOLD_POINTS = [
    (0.0, (0.0, 0.0, 0.0)),
    (0.333333, (0.345098, 0.109804, 0.0)),
    (0.666667, (0.737255, 0.501961, 0.0)),
    (1.0, (0.988235, 0.988235, 0.501961)),
]
GWYDDION_GOLD = LinearSegmentedColormap.from_list("Gwyddion Gold", GOLD_POINTS)

# Okabe-Ito colors provide a color-vision-deficiency-safe quantitative system.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
TEAL = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
INK = "#202124"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
PALE_GRAY = "#F3F4F6"


@dataclass(frozen=True)
class SampleSpec:
    public_id: str
    internal_id: str
    regime: str
    clip_sha256: str
    measured_sha256: str
    generated_sha256: str


SAMPLES = (
    SampleSpec(
        public_id="23",
        internal_id="N6342",
        regime="smooth",
        clip_sha256=(
            "b40668b74e6e0a65645d4d29fa2283708883e9c93a46ed691f3484cb5d30f5a4"
        ),
        measured_sha256=(
            "da1a1d1078953919dbbcd4eabd44943935f6bc162fca4fdd693ee4e89dde8433"
        ),
        generated_sha256=(
            "0de31097944eab6ea89ec89938ecaefbcf2101077044c21be42d4af5149d4954"
        ),
    ),
    SampleSpec(
        public_id="04",
        internal_id="6033",
        regime="intermediate",
        clip_sha256=(
            "099b99385db4ee782d0e6c6d0f2fc346e15a812845220927b74214877b1312e3"
        ),
        measured_sha256=(
            "f141b1b45186950f1677401b88ef6274ca5375263f2e38d9100d47714100f8a6"
        ),
        generated_sha256=(
            "fd78abf1669e6a78988d5b8d656c7ada71be58f752b2311ff1fc46b259af9cb1"
        ),
    ),
    SampleSpec(
        public_id="20",
        internal_id="6095",
        regime="rough",
        clip_sha256=(
            "1ce008ed73c3a0ffc91bdcd5ad1455afacb42284ffdf882abaa5e6ae2ea57ad2"
        ),
        measured_sha256=(
            "27161f35ec0c7ed41a703638c00468dfc6cbf3a1c501c29967fad613a80a83f0"
        ),
        generated_sha256=(
            "f8c47e45b2ec26ebd31e5d01b03ed8e50b06920fd0fd4d554f2301a49e6b0409"
        ),
    ),
)

ANONYMIZED_IDS = (
    ("01", "6022"),
    ("02", "6028"),
    ("03", "6029"),
    ("04", "6033"),
    ("05", "6047"),
    ("06", "6048"),
    ("07", "6056"),
    ("08", "6057"),
    ("09", "6062"),
    ("10", "6063"),
    ("11", "6070"),
    ("12", "6072"),
    ("13", "6078"),
    ("14", "6080"),
    ("15", "6082"),
    ("16", "6084"),
    ("17", "6085"),
    ("18", "6090"),
    ("19", "6094"),
    ("20", "6095"),
    ("21", "6099"),
    ("22", "6101"),
    ("23", "N6342"),
    ("24", "N6358"),
    ("25", "N6382"),
    ("26", "N6389"),
    ("27", "N6390"),
)
PUBLIC_BY_INTERNAL = {internal: public for public, internal in ANONYMIZED_IDS}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_path(sample: SampleSpec, kind: str) -> Path:
    suffix = {
        "clip": "rheed_clip.npz",
        "measured": "measured_afm.npy",
        "generated": "generated_loocv.npz",
    }[kind]
    return SOURCE_ROOT / f"sample_{sample.public_id}_{suffix}"


def _verify_sources() -> None:
    for sample in SAMPLES:
        expected = {
            "clip": sample.clip_sha256,
            "measured": sample.measured_sha256,
            "generated": sample.generated_sha256,
        }
        for kind, digest in expected.items():
            path = _snapshot_path(sample, kind)
            if not path.is_file():
                raise FileNotFoundError(f"missing source snapshot: {path}")
            actual = _sha256(path)
            if actual != digest:
                raise RuntimeError(
                    f"source snapshot hash mismatch for {path}: {actual} != {digest}"
                )


def _load_clip(sample: SampleSpec) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(_snapshot_path(sample, "clip"), allow_pickle=False)
    frames = np.asarray(payload["frames_uint8"], dtype=np.uint8)
    indices = np.asarray(payload["frame_indices"], dtype=int)
    if frames.shape[0] != 16 or frames.ndim != 3:
        raise RuntimeError(f"Sample {sample.public_id} does not contain 16 frames")
    if str(payload["growth_run_id"]) != sample.internal_id:
        raise RuntimeError(f"clip identity mismatch for Sample {sample.public_id}")
    return frames, indices


def _load_generated(sample: SampleSpec) -> tuple[np.ndarray, np.ndarray, float, float]:
    payload = np.load(_snapshot_path(sample, "generated"), allow_pickle=False)
    if str(payload["growth_run_id"]) != sample.internal_id:
        raise RuntimeError(f"generated identity mismatch for Sample {sample.public_id}")
    if str(payload["method"]) != SELECTED_METHOD:
        raise RuntimeError(f"unexpected generator for Sample {sample.public_id}")
    if bool(payload["retrieval_at_inference"]):
        raise RuntimeError("retrieval output cannot be shown as a generated AFM")
    if bool(payload["measured_afm_patch_used_at_inference"]):
        raise RuntimeError("held AFM leakage detected in generated snapshot")
    unit_shapes = np.asarray(payload["generated_unit_shapes"], dtype=float)
    predicted_sq = float(payload["predicted_rq_nm"])
    predicted_fsmi = float(payload["predicted_fsmi_nm"])
    height_ensemble = unit_shapes * predicted_sq
    return height_ensemble[0], height_ensemble, predicted_sq, predicted_fsmi


def _load_measured(sample: SampleSpec) -> np.ndarray:
    measured = np.asarray(
        np.load(_snapshot_path(sample, "measured"), allow_pickle=False),
        dtype=float,
    )
    if measured.ndim != 2:
        raise RuntimeError(f"measured AFM is not 2-D for Sample {sample.public_id}")
    return measured


def _center(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=float) - float(np.nanmean(array))


def _display_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([_center(array).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.01, 0.99])
    bound = max(abs(float(low)), abs(float(high)))
    return -bound, bound


def _sq(array: np.ndarray) -> float:
    centered = _center(array)
    return float(np.sqrt(np.nanmean(np.square(centered))))


def _panel_label(axis: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _clean_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _scale_bar(axis: plt.Axes, pixels: int, scan_size_nm: float = 1000.0) -> None:
    length_nm = 250.0
    length_px = length_nm / scan_size_nm * pixels
    x1 = 0.92 * pixels
    x0 = x1 - length_px
    y = 0.90 * pixels
    axis.plot([x0, x1], [y, y], color="black", lw=3.0, solid_capstyle="butt")
    axis.plot([x0, x1], [y, y], color="white", lw=1.8, solid_capstyle="butt")
    axis.text(
        (x0 + x1) / 2,
        y - 0.045 * pixels,
        "250 nm",
        color="white",
        fontsize=5.6,
        ha="center",
        va="bottom",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 0.5},
    )


def _show_afm(
    axis: plt.Axes,
    array: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    title: str | None = None,
) -> mpl.image.AxesImage:
    image = axis.imshow(
        _center(array),
        cmap=GWYDDION_GOLD,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    _clean_image_axis(axis)
    _scale_bar(axis, array.shape[1])
    if title:
        axis.set_title(title, pad=2.5)
    return image


def _show_rheed(
    axis: plt.Axes,
    frame: np.ndarray,
    *,
    title: str | None = None,
) -> None:
    low, high = np.quantile(frame, [0.01, 0.995])
    axis.imshow(frame, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    _clean_image_axis(axis)
    if title:
        axis.set_title(title, pad=2.5)


def _flow_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edge: str = MID_GRAY,
    face: str = "white",
    fontsize: float = 6.5,
    weight: str = "normal",
) -> Rectangle:
    patch = Rectangle(
        xy,
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.75,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    style: str = "-|>",
    lw: float = 0.9,
    mutation: float = 8.0,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            color=color,
            linewidth=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _inset_image(
    figure: plt.Figure,
    parent: plt.Axes,
    bounds: tuple[float, float, float, float],
) -> plt.Axes:
    parent_box = parent.get_position()
    x, y, width, height = bounds
    return figure.add_axes(
        [
            parent_box.x0 + x * parent_box.width,
            parent_box.y0 + y * parent_box.height,
            width * parent_box.width,
            height * parent_box.height,
        ]
    )


def _save_figure(figure: plt.Figure, stem: str) -> dict[str, object]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": FIGURE_ROOT / f"{stem}.png",
        "tiff": FIGURE_ROOT / f"{stem}.tiff",
        "pdf": FIGURE_ROOT / f"{stem}.pdf",
    }
    figure.savefig(paths["png"], dpi=600, facecolor="white")
    figure.savefig(
        paths["tiff"],
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(paths["pdf"], facecolor="white")
    plt.close(figure)
    with Image.open(paths["png"]) as rendered:
        width_px, height_px = rendered.size
        dpi = rendered.info.get("dpi", (600.0, 600.0))
    return {
        "stem": stem,
        "width_px": int(width_px),
        "height_px": int(height_px),
        "dpi_x": float(dpi[0]),
        "dpi_y": float(dpi[1]),
        "files": {key: str(value.relative_to(REPO_ROOT)) for key, value in paths.items()},
    }


def _make_figure_1() -> dict[str, object]:
    sample = SAMPLES[0]
    frames, frame_indices = _load_clip(sample)
    generated, _, predicted_sq, predicted_fsmi = _load_generated(sample)
    measured = _load_measured(sample)
    vmin, vmax = _display_limits(generated, measured)

    figure = plt.figure(figsize=(7.0, 4.45))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[1.02, 0.92, 0.78],
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.055,
        hspace=0.20,
    )

    # (a) In-situ acquisition and automatic causal selection.
    axis_a = figure.add_subplot(grid[0, 0])
    axis_a.set_xlim(0, 1)
    axis_a.set_ylim(0, 1)
    axis_a.axis("off")
    _panel_label(axis_a, "a", x=-0.02, y=1.01)
    axis_a.text(0.005, 0.93, "In-situ acquisition and causal event selection", fontweight="bold")

    # Minimal MBE stack is a schematic, not a synthetic microscopy image.
    axis_a.add_patch(Rectangle((0.01, 0.25), 0.10, 0.12, facecolor="#B8C0CC", edgecolor=INK))
    axis_a.add_patch(Rectangle((0.01, 0.37), 0.10, 0.18, facecolor="#CFE8DF", edgecolor=INK))
    axis_a.text(0.06, 0.61, "MBE growth", ha="center", fontweight="bold")
    axis_a.text(0.06, 0.46, "epilayer", ha="center", va="center", fontsize=5.8)
    axis_a.text(0.06, 0.31, "substrate", ha="center", va="center", fontsize=5.8)
    axis_a.plot([0.02, 0.05, 0.08, 0.10], [0.72, 0.62, 0.67, 0.58], color=VERMILLION, lw=0.8)
    _arrow(axis_a, (0.125, 0.48), (0.18, 0.48))

    thumb_positions = [0.18, 0.225, 0.27, 0.315]
    for offset, x in zip([0, 5, 10, 15], thumb_positions):
        image_axis = _inset_image(figure, axis_a, (x, 0.24, 0.075, 0.46))
        _show_rheed(image_axis, frames[offset])
        image_axis.patch.set_edgecolor("white")
        image_axis.patch.set_linewidth(0.4)
    axis_a.text(0.275, 0.76, "RHEED stream", ha="center", fontweight="bold")
    axis_a.text(
        0.275,
        0.14,
        f"real frames {frame_indices[0]}-{frame_indices[-1]} | Sample {sample.public_id}",
        ha="center",
        fontsize=5.8,
        color=MID_GRAY,
    )
    _arrow(axis_a, (0.405, 0.48), (0.455, 0.48))

    _flow_box(
        axis_a,
        (0.455, 0.29),
        0.14,
        0.38,
        "Automatic ROI\n+ clear-moment\ndetection",
        edge=TEAL,
        face="#ECF8F4",
        fontsize=6.3,
        weight="bold",
    )
    axis_a.text(0.525, 0.19, "causal, bounded latency", ha="center", fontsize=5.7, color=TEAL)
    _arrow(axis_a, (0.60, 0.48), (0.65, 0.48))

    for offset, x in zip([2, 7, 12], [0.65, 0.69, 0.73]):
        image_axis = _inset_image(figure, axis_a, (x, 0.25, 0.085, 0.44))
        _show_rheed(image_axis, frames[offset])
    axis_a.text(0.73, 0.76, "accepted 16-frame clip", ha="center", fontweight="bold")
    axis_a.text(0.73, 0.16, "orientation-locked input", ha="center", fontsize=5.7, color=MID_GRAY)
    _arrow(axis_a, (0.84, 0.48), (0.91, 0.48))
    _flow_box(
        axis_a,
        (0.91, 0.32),
        0.08,
        0.31,
        "model\ninput",
        edge=BLUE,
        face="#EAF3F9",
        fontsize=6.3,
        weight="bold",
    )

    # (b) Physics/AI inference with explicit branch semantics.
    axis_b = figure.add_subplot(grid[1, 0])
    axis_b.set_xlim(0, 1)
    axis_b.set_ylim(0, 1)
    axis_b.axis("off")
    _panel_label(axis_b, "b", x=-0.02, y=1.01)
    axis_b.text(0.005, 0.91, "Hybrid physics-AI inference", fontweight="bold")
    _flow_box(axis_b, (0.02, 0.31), 0.16, 0.40, "16-frame\nRHEED clip", edge=BLUE, face="#EAF3F9")
    _arrow(axis_b, (0.18, 0.51), (0.25, 0.51))
    _flow_box(axis_b, (0.25, 0.52), 0.16, 0.30, "R3D-18 temporal\nembedding", edge=BLUE, face="#EAF3F9", weight="bold")
    _flow_box(axis_b, (0.25, 0.14), 0.16, 0.25, "endpoint streak\nfeatures", edge=TEAL, face="#ECF8F4")
    _arrow(axis_b, (0.18, 0.42), (0.25, 0.27), color=TEAL)
    _arrow(axis_b, (0.41, 0.65), (0.49, 0.56))
    _arrow(axis_b, (0.41, 0.27), (0.49, 0.45), color=TEAL)
    _flow_box(
        axis_b,
        (0.49, 0.30),
        0.17,
        0.39,
        "cross-fitted heads\nSq | FSMI | morphology\ncondition | reliability",
        edge=PURPLE,
        face="#FAF0F7",
        weight="bold",
    )
    _arrow(axis_b, (0.66, 0.50), (0.73, 0.50))
    _flow_box(
        axis_b,
        (0.73, 0.30),
        0.18,
        0.39,
        "M17b non-retrieval\nstochastic AFM\ngenerator",
        edge=ORANGE,
        face="#FFF7E5",
        weight="bold",
    )
    axis_b.text(
        0.575,
        0.09,
        "Measured AFM is not available to the model at inference",
        ha="center",
        fontsize=6.0,
        color=VERMILLION,
        fontweight="bold",
    )

    # (c) Actual quantitative output from the frozen M17b result.
    axis_c = figure.add_subplot(grid[2, 0])
    axis_c.set_xlim(0, 1)
    axis_c.set_ylim(0, 1)
    axis_c.axis("off")
    _panel_label(axis_c, "c", x=-0.02, y=1.01)
    axis_c.text(0.005, 0.90, "Quantitative output", fontweight="bold")
    generated_axis = _inset_image(figure, axis_c, (0.02, 0.05, 0.20, 0.76))
    image = _show_afm(generated_axis, generated, vmin=vmin, vmax=vmax)
    generated_axis.set_title(f"generated AFM | Sample {sample.public_id}", pad=2.5)
    cbar_axis = _inset_image(figure, axis_c, (0.225, 0.10, 0.015, 0.66))
    colorbar = figure.colorbar(image, cax=cbar_axis)
    colorbar.set_label("height (nm)", labelpad=1.5)
    colorbar.ax.tick_params(length=2, width=0.5, pad=1)

    metrics = [
        ("Sq", f"{predicted_sq:.2f} nm"),
        ("FSMI", f"{predicted_fsmi:.2f} nm"),
        ("reliability", "71 / 100"),
    ]
    for index, (label, value) in enumerate(metrics):
        x = 0.30 + 0.16 * index
        axis_c.text(x, 0.65, label, fontsize=6.0, color=MID_GRAY, ha="center")
        axis_c.text(x, 0.44, value, fontsize=10.0, fontweight="bold", ha="center")
    axis_c.plot([0.27, 0.76], [0.31, 0.31], color=LIGHT_GRAY, lw=0.6)
    axis_c.text(
        0.515,
        0.14,
        "one physically scaled realization; ensemble retained for uncertainty analysis",
        ha="center",
        fontsize=5.8,
        color=MID_GRAY,
    )
    measured_axis = _inset_image(figure, axis_c, (0.80, 0.05, 0.18, 0.76))
    _show_afm(measured_axis, measured, vmin=vmin, vmax=vmax)
    measured_axis.set_title("measured AFM | evaluation only", pad=2.5)
    axis_c.text(0.89, 0.00, "not an inference input", ha="center", fontsize=5.6, color=VERMILLION)

    return _save_figure(figure, "Figure_1_AutoRHEED_overview")


def _make_figure_2() -> dict[str, object]:
    sample = SAMPLES[0]
    frames, _ = _load_clip(sample)
    generated, ensemble, predicted_sq, _ = _load_generated(sample)
    measured = _load_measured(sample)
    vmin, vmax = _display_limits(generated, measured)

    figure = plt.figure(figsize=(7.0, 6.15))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=[1.05, 1.05, 0.80],
        left=0.045,
        right=0.985,
        top=0.975,
        bottom=0.055,
        hspace=0.22,
    )

    # (a) Actual input geometry and hybrid condition prediction.
    axis_a = figure.add_subplot(grid[0, 0])
    axis_a.set_xlim(0, 1)
    axis_a.set_ylim(0, 1)
    axis_a.axis("off")
    _panel_label(axis_a, "a", x=-0.03, y=1.01)
    axis_a.text(0.005, 0.93, "RHEED representation and condition prediction", fontweight="bold")
    for index, x in enumerate(np.linspace(0.01, 0.16, 6)):
        frame_axis = _inset_image(figure, axis_a, (x, 0.31, 0.085, 0.44))
        _show_rheed(frame_axis, frames[index * 3])
    axis_a.text(0.12, 0.22, "16 real frames", ha="center", fontsize=5.8, color=MID_GRAY)
    _arrow(axis_a, (0.26, 0.52), (0.32, 0.52))
    _flow_box(axis_a, (0.32, 0.54), 0.17, 0.28, "R3D-18\ntemporal branch", edge=BLUE, face="#EAF3F9", weight="bold")
    _flow_box(axis_a, (0.32, 0.19), 0.17, 0.22, "streak endpoint\nbranch", edge=TEAL, face="#ECF8F4")
    _arrow(axis_a, (0.26, 0.42), (0.32, 0.30), color=TEAL)
    _arrow(axis_a, (0.49, 0.68), (0.56, 0.57))
    _arrow(axis_a, (0.49, 0.30), (0.56, 0.45), color=TEAL)
    _flow_box(
        axis_a,
        (0.56, 0.30),
        0.18,
        0.39,
        "strictly cross-fitted\ncondition heads",
        edge=PURPLE,
        face="#FAF0F7",
        weight="bold",
    )
    _arrow(axis_a, (0.74, 0.50), (0.81, 0.50))
    _flow_box(axis_a, (0.81, 0.59), 0.17, 0.24, f"Sq\n{predicted_sq:.2f} nm", edge=BLUE, face="#EAF3F9", weight="bold")
    _flow_box(axis_a, (0.81, 0.30), 0.17, 0.21, "9-D morphology\ncondition", edge=ORANGE, face="#FFF7E5", weight="bold")
    axis_a.text(
        0.895,
        0.19,
        "held-out growth excluded from fit",
        ha="center",
        fontsize=5.7,
        color=VERMILLION,
    )

    # (b) Generator anatomy with actual stochastic realizations.
    axis_b = figure.add_subplot(grid[1, 0])
    axis_b.set_xlim(0, 1)
    axis_b.set_ylim(0, 1)
    axis_b.axis("off")
    _panel_label(axis_b, "b", x=-0.03, y=1.01)
    axis_b.text(0.005, 0.93, "M17b topology-conditioned sparse-peak terrace generator", fontweight="bold")
    _flow_box(axis_b, (0.01, 0.56), 0.15, 0.24, "learned spectral\npopulation prior", edge=BLUE, face="#EAF3F9")
    _flow_box(axis_b, (0.01, 0.25), 0.15, 0.24, "RHEED-conditioned\nisland topology", edge=TEAL, face="#ECF8F4")
    _flow_box(axis_b, (0.23, 0.41), 0.15, 0.25, "sparse peaks\n+ terrace field", edge=ORANGE, face="#FFF7E5", weight="bold")
    _arrow(axis_b, (0.16, 0.68), (0.23, 0.55), color=BLUE)
    _arrow(axis_b, (0.16, 0.37), (0.23, 0.50), color=TEAL)
    _arrow(axis_b, (0.38, 0.535), (0.44, 0.535))
    _flow_box(axis_b, (0.44, 0.41), 0.13, 0.25, "physical\nSq scaling", edge=PURPLE, face="#FAF0F7", weight="bold")
    _arrow(axis_b, (0.57, 0.535), (0.625, 0.535))

    for index, x in enumerate([0.625, 0.715, 0.805, 0.895]):
        image_axis = _inset_image(figure, axis_b, (x, 0.31, 0.083, 0.45))
        _show_afm(image_axis, ensemble[index], vmin=vmin, vmax=vmax)
        image_axis.set_title(f"seed {index + 1}", fontsize=5.7, pad=1.5)
    axis_b.text(0.80, 0.20, f"four true M17b realizations | Sample {sample.public_id}", ha="center", fontsize=5.8, color=MID_GRAY)
    axis_b.text(
        0.80,
        0.09,
        "no retrieval and no measured AFM patch at inference",
        ha="center",
        fontsize=6.0,
        fontweight="bold",
        color=VERMILLION,
    )

    # (c) Leakage-aware protocol.
    axis_c = figure.add_subplot(grid[2, 0])
    axis_c.set_xlim(0, 1)
    axis_c.set_ylim(0, 1)
    axis_c.axis("off")
    _panel_label(axis_c, "c", x=-0.03, y=1.01)
    axis_c.text(0.005, 0.92, "Growth-level leave-one-out validation", fontweight="bold")
    xs = np.linspace(0.03, 0.64, 27)
    held_index = 22
    for index, x in enumerate(xs):
        color = VERMILLION if index == held_index else BLUE
        axis_c.plot(x, 0.56, marker="o", ms=3.8, color=color, mec="white", mew=0.35)
    axis_c.text(0.335, 0.72, "26 growths fit", ha="center", fontsize=6.2, color=BLUE, fontweight="bold")
    axis_c.text(xs[held_index], 0.38, "held once\nSample 23", ha="center", fontsize=5.8, color=VERMILLION, fontweight="bold")
    _arrow(axis_c, (0.66, 0.56), (0.73, 0.56))
    _flow_box(axis_c, (0.73, 0.40), 0.12, 0.32, "predict\nwithout held AFM", edge=VERMILLION, face="#FFF0EB", weight="bold")
    _arrow(axis_c, (0.85, 0.56), (0.90, 0.56))
    _flow_box(axis_c, (0.90, 0.40), 0.09, 0.32, "compare\nafterward", edge=MID_GRAY, face=PALE_GRAY, weight="bold")
    axis_c.text(
        0.51,
        0.12,
        "27 outer folds | growth groups are the leakage boundary | operator-invalid growth 6081 excluded before fitting",
        ha="center",
        fontsize=5.8,
        color=MID_GRAY,
    )

    return _save_figure(figure, "Figure_2_model_and_validation")


def _radial_psd(array: np.ndarray, scan_size_nm: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    values = _center(array)
    height, width = values.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(height, d=(scan_size_nm / 1000.0) / height))
    fx = np.fft.fftshift(np.fft.fftfreq(width, d=(scan_size_nm / 1000.0) / width))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    positive = radius > 0
    r = radius[positive]
    p = spectrum[positive]
    bins = np.geomspace(max(float(np.min(r)), 1.0), float(np.max(r)), 45)
    indices = np.digitize(r, bins)
    centers: list[float] = []
    power: list[float] = []
    for index in range(1, len(bins)):
        mask = indices == index
        if np.count_nonzero(mask) < 4:
            continue
        centers.append(float(np.exp(np.mean(np.log(r[mask])))))
        power.append(float(np.mean(p[mask])))
    result = np.asarray(power, dtype=float)
    result /= max(float(np.trapezoid(result, np.asarray(centers))), np.finfo(float).eps)
    return np.asarray(centers), result


def _selected_metrics() -> pd.DataFrame:
    standard = pd.read_csv(
        M17_REPORT / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    standard = standard.loc[standard["method"] == SELECTED_METHOD].copy()
    island = pd.read_csv(
        M17_REPORT / "crossfit" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    island = island.loc[island["method"] == SELECTED_METHOD].copy()
    confidence = pd.read_csv(
        M17_REPORT / "confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    )
    surface = pd.read_csv(
        M17_REPORT / "surface_metrics_per_group.csv",
        dtype={"growth_run_id": str},
    )
    result = (
        standard.merge(
            island[["growth_run_id", "island_feature_mae_z"]],
            on="growth_run_id",
            validate="one_to_one",
        )
        .merge(
            confidence[
                [
                    "growth_run_id",
                    "joint_confidence_index",
                    "realized_joint_error_index",
                ]
            ],
            on="growth_run_id",
            validate="one_to_one",
        )
        .merge(
            surface[
                [
                    "growth_run_id",
                    "sq_nm",
                    "functional_surface_morphology_index_nm",
                ]
            ],
            on="growth_run_id",
            validate="one_to_one",
        )
    )
    result["public_sample_id"] = result["growth_run_id"].map(PUBLIC_BY_INTERNAL)
    if result["public_sample_id"].isna().any() or len(result) != 27:
        raise RuntimeError("anonymized sample mapping does not cover the 27-growth cohort")
    return result


def _make_figure_3() -> tuple[dict[str, object], pd.DataFrame]:
    metrics = _selected_metrics()
    metric_by_internal = metrics.set_index("growth_run_id")

    figure = plt.figure(figsize=(7.0, 8.35))
    grid = figure.add_gridspec(
        5,
        5,
        height_ratios=[1.0, 1.0, 1.0, 0.08, 0.93],
        width_ratios=[0.94, 1.0, 1.0, 0.045, 1.08],
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.065,
        hspace=0.34,
        wspace=0.28,
    )

    selected_rows: list[dict[str, object]] = []
    for row, sample in enumerate(SAMPLES):
        frames, frame_indices = _load_clip(sample)
        generated, _, predicted_sq, predicted_fsmi = _load_generated(sample)
        measured = _load_measured(sample)
        entry = metric_by_internal.loc[sample.internal_id]
        vmin, vmax = _display_limits(generated, measured)

        rheed_axis = figure.add_subplot(grid[row, 0])
        generated_axis = figure.add_subplot(grid[row, 1])
        measured_axis = figure.add_subplot(grid[row, 2])
        colorbar_axis = figure.add_subplot(grid[row, 3])
        psd_axis = figure.add_subplot(grid[row, 4])

        keyframe_offset = 7
        _show_rheed(rheed_axis, frames[keyframe_offset])
        rheed_axis.set_title("RHEED key frame" if row == 0 else "", pad=2.5)
        rheed_axis.text(
            0.02,
            0.98,
            f"Sample {sample.public_id}\n{sample.regime}",
            transform=rheed_axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            fontweight="bold",
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.60, "pad": 1.5},
        )
        rheed_axis.text(
            0.02,
            0.03,
            f"frame {frame_indices[keyframe_offset]}",
            transform=rheed_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.3,
            color="white",
        )

        image = _show_afm(
            generated_axis,
            generated,
            vmin=vmin,
            vmax=vmax,
            title="M17b generated AFM" if row == 0 else None,
        )
        generated_axis.text(
            0.02,
            0.03,
            f"pred. Sq {predicted_sq:.2f} nm\nC {entry['joint_confidence_index']:.0f}/100",
            transform=generated_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.3,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.50, "pad": 1.0},
        )

        _show_afm(
            measured_axis,
            measured,
            vmin=vmin,
            vmax=vmax,
            title="measured AFM" if row == 0 else None,
        )
        measured_axis.text(
            0.02,
            0.03,
            f"meas. Sq {entry['sq_nm']:.2f} nm",
            transform=measured_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.3,
            color="white",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.50, "pad": 1.0},
        )

        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        if row == 1:
            colorbar.set_label("height (nm)", labelpad=1.0)
        colorbar.ax.tick_params(length=2, width=0.5, pad=1)

        g_frequency, g_power = _radial_psd(generated)
        m_frequency, m_power = _radial_psd(measured)
        psd_axis.loglog(g_frequency, g_power, color=BLUE, label="generated")
        psd_axis.loglog(m_frequency, m_power, color=VERMILLION, ls="--", label="measured")
        psd_axis.set_xlim(2, 120)
        psd_axis.grid(True, which="major", color=LIGHT_GRAY, lw=0.45, alpha=0.7)
        if row == 2:
            psd_axis.set_xlabel("spatial frequency (µm$^{-1}$)")
        if row == 0:
            psd_axis.set_title("normalized radial PSD", pad=2.5)
            psd_axis.legend(frameon=False, loc="lower left", handlelength=1.6)
        for spine in ("top", "right"):
            psd_axis.spines[spine].set_visible(False)

        selected_rows.append(
            {
                "public_sample_id": sample.public_id,
                "internal_growth_run_id": sample.internal_id,
                "regime": sample.regime,
                "measured_sq_nm": float(entry["sq_nm"]),
                "predicted_sq_nm": float(predicted_sq),
                "predicted_fsmi_nm": float(predicted_fsmi),
                "sq_absolute_error_nm": float(entry["rq_absolute_error_nm"]),
                "normalized_psd_log_distance": float(entry["normalized_psd_log_distance"]),
                "island_feature_mae_z": float(entry["island_feature_mae_z"]),
                "joint_confidence_index": float(entry["joint_confidence_index"]),
                "displayed_measured_scan_sq_nm": _sq(measured),
                "displayed_generated_sq_nm": _sq(generated),
            }
        )

    first_axis = figure.axes[0]
    _panel_label(first_axis, "a", x=-0.12, y=1.08)

    bottom = grid[4, :].subgridspec(1, 2, wspace=0.34)
    parity_axis = figure.add_subplot(bottom[0, 0])
    confidence_axis = figure.add_subplot(bottom[0, 1])
    _panel_label(parity_axis, "b", x=-0.14, y=1.10)
    _panel_label(confidence_axis, "c", x=-0.14, y=1.10)

    measured_values = metrics["sq_nm"].to_numpy(float)
    predicted_values = metrics["generated_rq_nm"].to_numpy(float)
    parity_axis.scatter(
        measured_values,
        predicted_values,
        s=16,
        facecolor="white",
        edgecolor=MID_GRAY,
        linewidth=0.65,
        zorder=2,
        label="all outer-LOO growths",
    )
    selected_markers = {"23": "o", "04": "s", "20": "D"}
    for sample in SAMPLES:
        row = metrics.loc[metrics["growth_run_id"] == sample.internal_id].iloc[0]
        parity_axis.scatter(
            row["sq_nm"],
            row["generated_rq_nm"],
            s=34,
            marker=selected_markers[sample.public_id],
            color=BLUE,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        parity_axis.annotate(
            sample.public_id,
            (row["sq_nm"], row["generated_rq_nm"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.6,
            color=BLUE,
            fontweight="bold",
        )
    limit = max(float(np.max(measured_values)), float(np.max(predicted_values))) * 1.05
    parity_axis.plot([0, limit], [0, limit], color=INK, ls="--", lw=0.8)
    parity_axis.set_xlim(0, limit)
    parity_axis.set_ylim(0, limit)
    parity_axis.set_aspect("equal", adjustable="box")
    parity_axis.set_xlabel("measured Sq (nm)")
    parity_axis.set_ylabel("LOO-predicted Sq (nm)")
    r_value = pearsonr(measured_values, predicted_values).statistic
    mae = float(np.mean(np.abs(measured_values - predicted_values)))
    parity_axis.text(
        0.04,
        0.94,
        f"n = 27 | r = {r_value:.2f} | MAE = {mae:.2f} nm",
        transform=parity_axis.transAxes,
        va="top",
        fontsize=6.0,
    )
    parity_axis.legend(frameon=False, loc="lower right", handletextpad=0.5)
    parity_axis.spines[["top", "right"]].set_visible(False)

    x = metrics["joint_confidence_index"].to_numpy(float)
    y = metrics["realized_joint_error_index"].to_numpy(float)
    confidence_axis.scatter(x, y, s=16, facecolor="white", edgecolor=MID_GRAY, linewidth=0.65)
    for sample in SAMPLES:
        row = metrics.loc[metrics["growth_run_id"] == sample.internal_id].iloc[0]
        confidence_axis.scatter(
            row["joint_confidence_index"],
            row["realized_joint_error_index"],
            s=34,
            marker=selected_markers[sample.public_id],
            color=TEAL,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        confidence_axis.annotate(
            sample.public_id,
            (row["joint_confidence_index"], row["realized_joint_error_index"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.6,
            color=TEAL,
            fontweight="bold",
        )
    rho, pvalue = spearmanr(x, y)
    confidence_axis.set_xlabel("cross-fitted reliability index (0-100)")
    confidence_axis.set_ylabel("realized joint error rank")
    confidence_axis.set_xlim(0, 103)
    confidence_axis.set_ylim(0.05, 1.02)
    confidence_axis.text(
        0.04,
        0.94,
        f"Spearman ρ = {rho:.2f} | p = {pvalue:.3f}",
        transform=confidence_axis.transAxes,
        va="top",
        fontsize=6.0,
    )
    confidence_axis.spines[["top", "right"]].set_visible(False)

    metadata = _save_figure(figure, "Figure_3_selected_results")
    return metadata, pd.DataFrame(selected_rows)


def _write_tables(selected: pd.DataFrame) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    mapping = pd.DataFrame(ANONYMIZED_IDS, columns=["public_sample_id", "internal_growth_run_id"])
    mapping.insert(0, "audience", "internal_provenance_not_for_manuscript")
    mapping.to_csv(REPORT_ROOT / "sample_id_mapping_internal.csv", index=False)
    selected.to_csv(REPORT_ROOT / "selected_case_metrics.csv", index=False)


def _write_build_manifest(figures: Iterable[dict[str, object]]) -> None:
    payload = {
        "experiment": "MorphMBE M17b Nano Letters figure package",
        "git_source_commit": "99bb75b3ed22d385367eb6622f7f05ddbc6a754e",
        "selected_method": SELECTED_METHOD,
        "sample_label_policy": "two-digit public IDs 01-27; internal mapping is not manuscript-facing",
        "selected_samples": [
            {
                "public_sample_id": sample.public_id,
                "internal_growth_run_id": sample.internal_id,
                "roughness_regime": sample.regime,
                "source_sha256": {
                    "rheed_clip": sample.clip_sha256,
                    "measured_afm": sample.measured_sha256,
                    "generated_outer_loo": sample.generated_sha256,
                },
            }
            for sample in SAMPLES
        ],
        "inference_boundary": {
            "validation": "strict outer leave-one-growth-out by growth group",
            "retrieval_at_inference": False,
            "measured_afm_patch_used_at_inference": False,
            "generated_image_interpretation": (
                "stochastic conditional morphology realization; not pixel-registered reconstruction"
            ),
            "sample_23_claim_boundary": (
                "retrospective method-development evidence; not prospectively untouched"
            ),
        },
        "afm_colormap": {
            "name": "Gwyddion 2.71 Gold",
            "mapping": "piecewise-linear",
            "control_points": [
                {"position": p, "rgb": list(rgb)} for p, rgb in GOLD_POINTS
            ],
            "display_range": "pooled generated/measured 1st-99th percentile per pair, symmetric about zero",
            "units": "nm",
        },
        "outputs": list(figures),
    }
    (REPORT_ROOT / "build_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify frozen source hashes and leakage flags without rendering",
    )
    args = parser.parse_args()
    _style()
    _verify_sources()
    for sample in SAMPLES:
        _load_clip(sample)
        _load_generated(sample)
        _load_measured(sample)
    if args.verify_only:
        sample_labels = ", ".join(sample.public_id for sample in SAMPLES)
        print(f"Source verification passed for Samples {sample_labels}.")
        return

    figure_1 = _make_figure_1()
    figure_2 = _make_figure_2()
    figure_3, selected = _make_figure_3()
    _write_tables(selected)
    _write_build_manifest([figure_1, figure_2, figure_3])
    print(f"Wrote Nano Letters figure package to {REPORT_ROOT}")


if __name__ == "__main__":
    main()
