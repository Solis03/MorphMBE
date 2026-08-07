#!/usr/bin/env python3
"""Build a publication-quality full-cohort M17b atlas.

The atlas is ordered by measured sample Sq and uses only frozen, strict
outer-leave-one-growth-out model outputs.  RHEED frames and measured AFM maps
are read without modification; the first saved stochastic M17b realization is
used for each growth.  Manuscript-facing labels are anonymized as Sample
01--27, while internal identifiers remain confined to the private mapping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image

from scripts.make_nanoletters_m17_figures import (
    BLUE,
    GWYDDION_GOLD,
    INK,
    LIGHT_GRAY,
    MID_GRAY,
    SAMPLES,
    VERMILLION,
    _center,
    _display_limits,
    _radial_psd,
    _scale_bar,
    _sq,
    _style,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "nanoletters_m17_figures_20260806"
M17_REPORT = (
    REPO_ROOT
    / "reports"
    / "rheed_m17_end_to_end_generation"
    / "20260804_m17_sparse_topology_line3_full27_v1"
    / "full27_loo"
)
MAPPING_PATH = REPORT_ROOT / "sample_id_mapping_internal.csv"
PHASE1_RELATIVE = Path(
    "outputs/extra_five_integration/"
    "20260729_line3_full28_orientation90_keyframe_locked_v3/"
    "machine_dataset_full28/modeling_manifest.csv"
)
GENERATED_RELATIVE = Path(
    "outputs/rheed_m17_end_to_end_generation/"
    "20260804_m17_sparse_topology_line3_full27_v1/full27_loo/"
    "crossfit/generated_maps/M17b_topology_sparse_peak_terrace"
)
SELECTED_METHOD = "M17b_topology_sparse_peak_terrace"


@dataclass(frozen=True)
class AtlasRecord:
    public_id: str
    internal_id: str
    measured_sq_nm: float
    displayed_scan_sq_nm: float
    predicted_sq_nm: float
    confidence: float
    frame_index: int
    rheed: np.ndarray
    generated: np.ndarray
    measured: np.ndarray
    generated_frequency: np.ndarray
    generated_psd: np.ndarray
    measured_frequency: np.ndarray
    measured_psd: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source(source_repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else source_repo / path


def _discover_source_repo(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        REPO_ROOT,
        REPO_ROOT.parent / "code",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / PHASE1_RELATIVE).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the canonical source repository containing "
        f"{PHASE1_RELATIVE}. Pass --source-repo explicitly."
    )


def _discover_generated_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            REPO_ROOT / GENERATED_RELATIVE,
            REPO_ROOT.parent
            / "code-worktrees"
            / "n6342-sparse-island-20260804"
            / GENERATED_RELATIVE,
        ]
    )
    candidates.extend(
        sorted(
            REPO_ROOT.parent.glob(
                "code-worktrees/*/"
                + str(GENERATED_RELATIVE)
            )
        )
    )
    for candidate in candidates:
        if candidate.is_dir() and len(list(candidate.glob("*.npz"))) == 27:
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate all 27 frozen M17b outer-LOO maps. "
        "Pass --generated-root explicitly."
    )


def _phase_row(phase1: pd.DataFrame, internal_id: str) -> pd.Series:
    rows = phase1.loc[phase1["growth_run_id"].astype(str) == internal_id]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one modeling-manifest row for {internal_id}, found {len(rows)}"
        )
    return rows.iloc[0]


def _load_rheed(
    row: pd.Series,
    *,
    source_repo: Path,
) -> tuple[np.ndarray, int]:
    # The selected_16 cache is the actual 224 x 224, orientation-locked model
    # input.  Reading the raw frame path here would bypass the automatic ROI.
    cache_path = _resolve_source(source_repo, str(row["clip_cache_path"]))
    with np.load(cache_path, allow_pickle=False) as payload:
        frames = np.asarray(payload["frames_uint8"], dtype=float)
        indices = np.asarray(payload["frame_indices"], dtype=int)
        cached_growth = str(payload["growth_run_id"])
    if cached_growth != str(row["growth_run_id"]):
        raise RuntimeError(f"RHEED cache identity mismatch: {cache_path}")
    if frames.shape != (16, 224, 224) or indices.shape != (16,):
        raise RuntimeError(f"invalid selected_16 model-input cache: {cache_path}")
    offset = int(row.get("keyframe_offset_in_clip_x", len(frames) // 2))
    offset = int(np.clip(offset, 0, len(frames) - 1))
    return frames[offset], int(indices[offset])


def _load_measured(row: pd.Series, *, source_repo: Path) -> np.ndarray:
    path = _resolve_source(source_repo, str(row["representative_afm_height_array"]))
    measured = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if measured.ndim != 2 or min(measured.shape) < 128:
        raise RuntimeError(f"invalid measured AFM height map: {path}")
    return measured


def _load_generated(path: Path, *, internal_id: str) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["growth_run_id"]) != internal_id:
            raise RuntimeError(f"generated-map identity mismatch in {path}")
        if str(payload["method"]) != SELECTED_METHOD:
            raise RuntimeError(f"unexpected method in {path}")
        if bool(payload["retrieval_at_inference"]):
            raise RuntimeError(f"retrieval flag is true in {path}")
        if bool(payload["measured_afm_patch_used_at_inference"]):
            raise RuntimeError(f"measured-AFM input flag is true in {path}")
        unit_shapes = np.asarray(payload["generated_unit_shapes"], dtype=float)
        predicted_sq = float(payload["predicted_rq_nm"])
    if unit_shapes.shape != (4, 128, 128):
        raise RuntimeError(f"unexpected generated ensemble shape in {path}")
    generated = unit_shapes[0] * predicted_sq
    if not np.isclose(_sq(generated), predicted_sq, rtol=3e-5):
        raise RuntimeError(f"generated Sq mismatch in {path}")
    return generated, predicted_sq


def load_records(
    *,
    source_repo: Path,
    generated_root: Path,
) -> list[AtlasRecord]:
    mapping = pd.read_csv(
        MAPPING_PATH,
        dtype={"public_sample_id": str, "internal_growth_run_id": str},
    )
    mapping["public_sample_id"] = mapping["public_sample_id"].str.zfill(2)
    if list(mapping["public_sample_id"]) != [f"{index:02d}" for index in range(1, 28)]:
        raise RuntimeError("public Sample 01--27 mapping is incomplete or out of order")
    public_by_internal = mapping.set_index("internal_growth_run_id")[
        "public_sample_id"
    ].to_dict()

    phase1 = pd.read_csv(
        source_repo / PHASE1_RELATIVE,
        dtype={"growth_run_id": str},
    )
    metrics = pd.read_csv(
        M17_REPORT / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    metrics = metrics.loc[metrics["method"] == SELECTED_METHOD].copy()
    confidence = pd.read_csv(
        M17_REPORT / "confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    ).set_index("growth_run_id")
    if len(metrics) != 27 or metrics["growth_run_id"].nunique() != 27:
        raise RuntimeError("selected M17b metric table does not contain 27 unique growths")
    if set(metrics["growth_run_id"]) != set(public_by_internal):
        raise RuntimeError("private mapping and M17b cohort do not cover the same growths")

    anchor_hashes = {sample.internal_id: sample.generated_sha256 for sample in SAMPLES}
    records: list[AtlasRecord] = []
    for metric in metrics.itertuples(index=False):
        internal_id = str(metric.growth_run_id)
        generated_path = generated_root / f"{internal_id}.npz"
        if internal_id in anchor_hashes:
            observed = _sha256(generated_path)
            if observed != anchor_hashes[internal_id]:
                raise RuntimeError(
                    f"frozen generated-map checksum mismatch for {internal_id}: {observed}"
                )
        row = _phase_row(phase1, internal_id)
        rheed, frame_index = _load_rheed(
            row,
            source_repo=source_repo,
        )
        measured = _load_measured(row, source_repo=source_repo)
        generated, predicted_sq = _load_generated(
            generated_path,
            internal_id=internal_id,
        )
        measured_sq = float(metric.true_rq_nm)
        manifest_sq = float(row["primary_rq_nm_median"])
        if not np.isclose(measured_sq, manifest_sq, rtol=2e-5):
            raise RuntimeError(f"sample-median measured Sq mismatch for {internal_id}")
        displayed_scan_sq = _sq(measured)
        if not np.isclose(predicted_sq, float(metric.generated_rq_nm), rtol=2e-5):
            raise RuntimeError(f"predicted Sq mismatch for {internal_id}")
        generated_frequency, generated_psd = _radial_psd(generated)
        measured_frequency, measured_psd = _radial_psd(measured)
        records.append(
            AtlasRecord(
                public_id=public_by_internal[internal_id],
                internal_id=internal_id,
                measured_sq_nm=measured_sq,
                displayed_scan_sq_nm=displayed_scan_sq,
                predicted_sq_nm=predicted_sq,
                confidence=float(confidence.loc[internal_id, "joint_confidence_index"]),
                frame_index=frame_index,
                rheed=rheed,
                generated=generated,
                measured=measured,
                generated_frequency=generated_frequency,
                generated_psd=generated_psd,
                measured_frequency=measured_frequency,
                measured_psd=measured_psd,
            )
        )
    records.sort(key=lambda record: (record.measured_sq_nm, record.public_id))
    values = np.asarray([record.measured_sq_nm for record in records])
    if len(records) != 27 or np.any(np.diff(values) < 0):
        raise RuntimeError("atlas ordering is not monotonic in measured Sq")
    return records


def _image_label(axis: plt.Axes, text: str, *, top: bool = False) -> None:
    axis.text(
        0.025,
        0.965 if top else 0.035,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top" if top else "bottom",
        fontsize=5.5,
        fontweight="bold" if top else "normal",
        color="white",
        linespacing=1.08,
        bbox={
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.55,
            "pad": 1.0,
        },
    )


def _clean_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_atlas(
    records: list[AtlasRecord],
    *,
    output_dir: Path,
    dpi: int,
) -> dict[str, object]:
    _style()
    mpl.rcParams.update(
        {
            "font.size": 6.0,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
        }
    )
    figure = plt.figure(figsize=(8.5, 26.0), facecolor="white")
    grid = figure.add_gridspec(
        len(records),
        5,
        width_ratios=[1.03, 1.03, 1.03, 0.055, 1.62],
        left=0.052,
        right=0.985,
        top=0.966,
        bottom=0.025,
        wspace=0.12,
        hspace=0.18,
    )
    figure.text(
        0.052,
        0.990,
        "Full-cohort outer-LOO atlas ordered by measured Sq",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.985,
        0.990,
        "27 growths | held growth excluded from every fit | Sample 01-27",
        ha="right",
        va="top",
        fontsize=6.2,
        color=MID_GRAY,
    )

    positive_psd = np.concatenate(
        [
            values[np.isfinite(values) & (values > 0)]
            for record in records
            for values in (record.generated_psd, record.measured_psd)
        ]
    )
    y_min = 10.0 ** np.floor(np.log10(np.quantile(positive_psd, 0.005)))
    y_max = 10.0 ** np.ceil(np.log10(np.quantile(positive_psd, 0.995)))
    common_frequency_max = min(
        min(float(record.generated_frequency.max()), float(record.measured_frequency.max()))
        for record in records
    )

    for row_index, record in enumerate(records):
        rheed_axis = figure.add_subplot(grid[row_index, 0])
        generated_axis = figure.add_subplot(grid[row_index, 1])
        measured_axis = figure.add_subplot(grid[row_index, 2])
        color_axis = figure.add_subplot(grid[row_index, 3])
        psd_axis = figure.add_subplot(grid[row_index, 4])

        rheed_low, rheed_high = np.quantile(record.rheed, [0.01, 0.995])
        rheed_axis.imshow(
            record.rheed,
            cmap="gray",
            vmin=rheed_low,
            vmax=rheed_high,
            interpolation="nearest",
        )
        _clean_image_axis(rheed_axis)
        _image_label(
            rheed_axis,
            f"Sample {record.public_id}\nmeas. Sq {record.measured_sq_nm:.2f} nm",
            top=True,
        )
        if record.frame_index >= 0:
            _image_label(rheed_axis, f"frame {record.frame_index}")

        vmin, vmax = _display_limits(record.generated, record.measured)
        for axis, array in (
            (generated_axis, record.generated),
            (measured_axis, record.measured),
        ):
            axis.imshow(
                _center(array),
                cmap=GWYDDION_GOLD,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
            )
            _clean_image_axis(axis)
            _scale_bar(axis, array.shape[1])
        _image_label(
            generated_axis,
            f"pred. Sq {record.predicted_sq_nm:.2f} nm\nC {record.confidence:.0f}/100",
            top=True,
        )
        _image_label(
            measured_axis,
            f"scan Sq {record.displayed_scan_sq_nm:.2f} nm",
            top=True,
        )

        colorbar = ColorbarBase(
            color_axis,
            cmap=GWYDDION_GOLD,
            norm=Normalize(vmin=vmin, vmax=vmax),
            orientation="vertical",
        )
        colorbar.set_ticks([vmin, 0.0, vmax])
        colorbar.set_ticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"])
        color_axis.tick_params(length=1.8, width=0.45, pad=1.0, labelsize=4.5)
        color_axis.yaxis.set_ticks_position("right")
        color_axis.yaxis.set_label_position("right")

        psd_axis.loglog(
            record.generated_frequency,
            record.generated_psd,
            color=BLUE,
            lw=0.85,
            label="generated",
        )
        psd_axis.loglog(
            record.measured_frequency,
            record.measured_psd,
            color=VERMILLION,
            lw=0.85,
            ls="--",
            label="measured",
        )
        psd_axis.set_xlim(1.1, common_frequency_max)
        psd_axis.set_ylim(y_min, y_max)
        psd_axis.grid(True, which="major", color=LIGHT_GRAY, lw=0.35, alpha=0.75)
        psd_axis.spines[["top", "right"]].set_visible(False)
        psd_axis.tick_params(length=2.0, width=0.45, pad=1.4)
        if row_index != len(records) - 1:
            psd_axis.tick_params(labelbottom=False)
        else:
            psd_axis.set_xlabel(
                r"spatial frequency ($\mathrm{\mu m}^{-1}$)",
                labelpad=2.0,
            )
        if row_index not in {0, len(records) // 2, len(records) - 1}:
            psd_axis.tick_params(labelleft=False)
        if row_index == 0:
            rheed_axis.set_title("RHEED key frame", pad=3.0, fontweight="bold")
            generated_axis.set_title("M17b generated AFM", pad=3.0, fontweight="bold")
            measured_axis.set_title("measured AFM\nevaluation only", pad=3.0, fontweight="bold")
            color_axis.set_title("height\n(nm)", pad=3.0, fontweight="bold")
            psd_axis.set_title("normalized radial PSD", pad=3.0, fontweight="bold")
            psd_axis.legend(
                frameon=False,
                loc="lower left",
                fontsize=5.0,
                handlelength=1.6,
                borderaxespad=0.2,
            )

        if row_index < len(records) - 1:
            y = rheed_axis.get_position().y0 - 0.0034
            figure.add_artist(
                Line2D(
                    [0.052, 0.985],
                    [y, y],
                    transform=figure.transFigure,
                    color=LIGHT_GRAY,
                    lw=0.28,
                    zorder=0,
                )
            )

    figure.text(
        0.052,
        0.009,
        "Rows are sorted by measured sample-median Sq; measured-panel labels "
        "report the displayed scan Sq. Each generated panel is the first saved "
        "stochastic realization from its strict outer-LOO fold.",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=MID_GRAY,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Figure_4_full_cohort_atlas"
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "tiff": output_dir / f"{stem}.tiff",
    }
    figure.savefig(
        paths["pdf"],
        facecolor="white",
        metadata={
            "Title": "Full-cohort outer-LOO atlas ordered by measured Sq",
            "Subject": "Anonymized 27-growth RHEED-to-AFM validation atlas",
        },
    )
    figure.savefig(paths["png"], dpi=dpi, facecolor="white")
    figure.savefig(
        paths["tiff"],
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    with Image.open(paths["png"]) as rendered:
        width, height = rendered.size
        rendered_dpi = rendered.info.get("dpi", (dpi, dpi))
    return {
        "records": len(records),
        "first_public_id": records[0].public_id,
        "last_public_id": records[-1].public_id,
        "measured_sq_range_nm": [
            records[0].measured_sq_nm,
            records[-1].measured_sq_nm,
        ],
        "width_px": int(width),
        "height_px": int(height),
        "dpi": [float(rendered_dpi[0]), float(rendered_dpi[1])],
        "files": {key: str(value) for key, value in paths.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_repo = _discover_source_repo(args.source_repo)
    generated_root = _discover_generated_root(args.generated_root)
    records = load_records(
        source_repo=source_repo,
        generated_root=generated_root,
    )
    print(
        "Verified 27 atlas rows ordered by measured Sq: "
        f"{records[0].measured_sq_nm:.2f}--{records[-1].measured_sq_nm:.2f} nm."
    )
    if args.verify_only:
        return
    metadata = make_atlas(records, output_dir=args.output_dir, dpi=args.dpi)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
