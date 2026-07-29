from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_video_afm_story.common import (
    display_path,
    read_id_list,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from .line_flatten import line_flatten, sq_nm


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _raw_stem(path: str | Path) -> str:
    name = Path(path).name
    if name.lower().endswith((".tif", ".tiff")):
        name = re.sub(r"\.tiff?$", "", name, flags=re.IGNORECASE)
    return name


def _normalized_scan_name(path: str | Path) -> str:
    name = _raw_stem(path)
    name = re.sub(
        r"[_ ]+([0-9]+(?:\.[0-9]+)?)\s*nm$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"_1(?=\.\d+$)", "", name)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _nanoscope_qc_records(raw_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not raw_path.parent.exists():
        return records
    source_key = _normalized_scan_name(raw_path)
    for tif in sorted(raw_path.parent.glob("*.tif")):
        match = re.search(
            r"[_ ]([0-9]+(?:\.[0-9]+)?)\s*nm$",
            _raw_stem(tif),
            flags=re.IGNORECASE,
        )
        if match is None or _normalized_scan_name(tif) != source_key:
            continue
        records.append(
            {
                "nanoscope_export_path": display_path(tif),
                "nanoscope_rq_nm": float(match.group(1)),
            }
        )
    return records


def _decision(
    decisions: pd.DataFrame,
    sample_id: str,
    afm_file_id: str,
) -> tuple[str, str, str]:
    rows = decisions.loc[
        (decisions["sample_id"].astype(str) == str(sample_id))
        & (
            (decisions["afm_file_id"].astype(str) == str(afm_file_id))
            | (decisions["afm_file_id"].astype(str) == "*")
        )
    ]
    if rows.empty:
        return "include", "", "not_flagged"
    row = rows.iloc[0]
    return str(row["decision"]), str(row["reason"]), str(row["status"])


def _save_render(
    array: np.ndarray,
    destination: Path,
    *,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    finite = np.asarray(array, dtype=float)
    low, high = np.nanpercentile(finite, [1, 99])
    figure, axis = plt.subplots(figsize=(4.1, 3.7), dpi=160)
    image = axis.imshow(
        finite,
        origin="upper",
        cmap="afmhot",
        vmin=float(low),
        vmax=float(high),
        extent=(0, 1, 1, 0),
    )
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_title(title, fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label("height (nm)")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _process_scans(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        repo_path(config["source_audit"]),
        dtype={"sample_id": str},
    )
    decisions = pd.read_csv(
        repo_path(config["manual_provenance_review"]),
        dtype={"sample_id": str, "afm_file_id": str},
    )
    processed_root = repo_path(config["processed_afm_root"])
    derived_root = repo_path(config["derived_root"])
    orders = [int(value) for value in config["orders"]]
    selected_order = int(config["selected_order"])
    records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []

    for source_row in source.to_dict("records"):
        sample_id = str(source_row["sample_id"])
        scan_id = str(source_row["afm_file_id"])
        source_array = (
            processed_root
            / sample_id
            / scan_id
            / f"{scan_id}_height.npy"
        )
        source_metadata = source_array.with_name(f"{scan_id}_metadata.json")
        if not source_array.exists() or not source_metadata.exists():
            raise FileNotFoundError(
                f"decoded AFM source is incomplete for {sample_id}/{scan_id}"
            )
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        raw_afm = repo_path(
            metadata.get("raw_afm_file")
            or metadata.get("raw_file")
            or source_row["raw_afm_file"]
        )
        if not raw_afm.exists():
            raise FileNotFoundError(f"raw AFM source is missing: {raw_afm}")
        height = np.load(source_array, allow_pickle=False).astype(np.float64)
        unit = str(
            metadata.get("height_unit_exported")
            or metadata.get("height_unit_original")
            or ""
        )
        if unit != str(config["height_unit"]):
            raise RuntimeError(
                f"height unit is not nm for {sample_id}/{scan_id}: {unit}"
            )
        decision, reason, review_status = _decision(
            decisions,
            sample_id,
            scan_id,
        )
        selected_path: Path | None = None
        selected_hash = ""
        order_values: dict[int, float] = {}
        for order in orders:
            corrected, background = line_flatten(height, order=order)
            output_dir = (
                derived_root / f"order_{order}" / sample_id / scan_id
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            corrected_path = output_dir / f"{scan_id}_line_flatten_o{order}.npy"
            background_path = output_dir / f"{scan_id}_background_o{order}.npy"
            np.save(corrected_path, corrected)
            np.save(background_path, background)
            order_values[order] = sq_nm(corrected)
            if order == selected_order:
                selected_path = corrected_path
                selected_hash = _array_sha256(corrected)
                _save_render(
                    corrected,
                    output_dir / f"{scan_id}_line_flatten_o{order}.png",
                    title=(
                        f"{sample_id} / {scan_id}\n"
                        f"line-{order} flattened; Sq={order_values[order]:.3f} nm"
                    ),
                )
            order_metadata = {
                "sample_id": sample_id,
                "afm_file_id": scan_id,
                "source_raw_afm_file": display_path(raw_afm),
                "source_raw_sha256": sha256_file(raw_afm),
                "source_decoded_zsensor": display_path(source_array),
                "source_decoded_sha256": sha256_file(source_array),
                "height_unit": "nm",
                "scan_size_um": metadata.get("scan_size_um"),
                "resolution": list(height.shape),
                "correction": {
                    "method": "independent least-squares polynomial per scan line",
                    "fast_scan_axis": "array axis 1 (x; one fit per row)",
                    "polynomial_order": order,
                    "global_plane_subtraction": False,
                    "post_fit_smoothing": False,
                },
                "sq_nm": order_values[order],
                "output_array": display_path(corrected_path),
                "background_array": display_path(background_path),
            }
            (output_dir / f"{scan_id}_line_flatten_o{order}_metadata.json").write_text(
                json.dumps(order_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        assert selected_path is not None
        scan_x, scan_y = metadata.get("scan_size_um", [np.nan, np.nan])
        record = {
            "sample_id": sample_id,
            "growth_run_id": sample_id,
            "afm_file_id": scan_id,
            "raw_afm_file": display_path(raw_afm),
            "raw_afm_sha256": sha256_file(raw_afm),
            "decoded_zsensor_path": display_path(source_array),
            "decoded_zsensor_sha256": sha256_file(source_array),
            "height_array_path": display_path(selected_path),
            "afm_path": display_path(selected_path),
            "afm_preprocessing_variant": "line3_scanline_flatten_v1",
            "roughness_metric": "Sq_areal_RMS_height_nm",
            "line_flatten_order": selected_order,
            "selected_array_sha256": selected_hash,
            "scan_size_um": (
                float(scan_x) if float(scan_x) == float(scan_y) else np.nan
            ),
            "scan_size_x_um": float(scan_x),
            "scan_size_y_um": float(scan_y),
            "resolution_x": int(height.shape[1]),
            "resolution_y": int(height.shape[0]),
            "height_unit": "nm",
            "channel": metadata.get("primary_channel", ""),
            "sq_nm": order_values[selected_order],
            "rq_recomputed_nm": order_values[selected_order],
            "provenance_decision": decision,
            "provenance_reason": reason,
            "provenance_review_status": review_status,
            "provenance_excluded": decision == "exclude",
            "include_for_modeling": decision != "exclude",
            **{f"sq_order_{order}_nm": order_values[order] for order in orders},
        }
        records.append(record)
        for qc in _nanoscope_qc_records(raw_afm):
            qc_records.append(
                {
                    "sample_id": sample_id,
                    "afm_file_id": scan_id,
                    "raw_afm_file": display_path(raw_afm),
                    **qc,
                    **{
                        f"sq_order_{order}_nm": order_values[order]
                        for order in orders
                    },
                }
            )
    return pd.DataFrame(records), pd.DataFrame(qc_records)


def _deduplicate(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scans = scans.copy()
    scans["duplicate_group_size"] = scans.groupby(
        "selected_array_sha256"
    )["selected_array_sha256"].transform("size")
    scans["duplicate_rank"] = scans.groupby(
        "selected_array_sha256",
        sort=False,
    ).cumcount()
    scans["excluded_by_hash_deduplication"] = scans["duplicate_rank"] > 0
    scans["include_for_modeling"] = (
        ~scans["provenance_excluded"].astype(bool)
        & ~scans["excluded_by_hash_deduplication"].astype(bool)
    )
    duplicates = scans.loc[scans["duplicate_group_size"] > 1].copy()
    return scans, duplicates


def _modeling_targets(
    scans: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling = pd.read_csv(
        repo_path(config["source_modeling_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    removelist = read_id_list(config["removelist_path"])
    excluded_growths = set(map(str, config["explicitly_excluded_growths"]))
    model_ids = set(
        modeling.loc[
            modeling["usable_for_modeling"].astype(bool)
            & modeling["cohort_primary_1um"].astype(bool),
            "sample_id",
        ].astype(str)
    )
    model_ids -= removelist | excluded_growths
    if len(model_ids) != int(config["expected_modeling_growth_count"]):
        raise RuntimeError(
            f"expected {config['expected_modeling_growth_count']} modeling "
            f"growths, found {len(model_ids)}"
        )
    tolerance = float(config["scan_size_tolerance_um"])
    primary = scans.loc[
        scans["sample_id"].isin(model_ids)
        & (
            scans["scan_size_x_um"].sub(
                float(config["primary_scan_size_um"])
            ).abs()
            <= tolerance
        )
        & (
            scans["scan_size_y_um"].sub(
                float(config["primary_scan_size_um"])
            ).abs()
            <= tolerance
        )
        & ~scans["provenance_excluded"]
        & ~scans["excluded_by_hash_deduplication"]
    ].copy()
    target_records: list[dict[str, Any]] = []
    for sample_id, rows in primary.groupby("sample_id", sort=True):
        values = rows["sq_nm"].to_numpy(float)
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        representative = (
            rows.assign(_distance=(rows["sq_nm"] - median).abs())
            .sort_values(["_distance", "afm_file_id"])
            .iloc[0]
        )
        target_records.append(
            {
                "sample_id": str(sample_id),
                "growth_run_id": str(sample_id),
                "primary_afm_scan_count": int(len(rows)),
                "sample_median_sq_nm": median,
                "sample_sq_iqr_nm": float(q75 - q25),
                "sample_sq_q25_nm": float(q25),
                "sample_sq_q75_nm": float(q75),
                "sample_sq_min_nm": float(np.min(values)),
                "sample_sq_max_nm": float(np.max(values)),
                "log_sample_median_sq_nm": float(np.log(max(median, 1e-6))),
                "representative_afm_scan_id": str(
                    representative["afm_file_id"]
                ),
                "representative_afm_height_array": str(
                    representative["height_array_path"]
                ),
                "representative_scan_sq_nm": float(
                    representative["sq_nm"]
                ),
                "aggregation": (
                    "arithmetic median of deduplicated primary 1x1 um "
                    "scan Sq values in nm; log applied after aggregation"
                ),
            }
        )
    targets = pd.DataFrame(target_records).sort_values("sample_id")
    if set(targets["sample_id"]) != model_ids:
        missing = sorted(model_ids - set(targets["sample_id"]))
        raise RuntimeError(f"corrected primary AFM targets are missing: {missing}")

    target_lookup = targets.set_index("sample_id")
    corrected_modeling = modeling.copy()
    corrected_modeling["afm_target_variant"] = "line3_scanline_flatten_v1"
    corrected_modeling["roughness_nomenclature"] = "Sq_areal_RMS_height_nm"
    for index, row in corrected_modeling.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in target_lookup.index:
            continue
        target = target_lookup.loc[sample_id]
        corrected_modeling.at[index, "primary_afm_scan_count"] = int(
            target["primary_afm_scan_count"]
        )
        corrected_modeling.at[index, "primary_rq_nm_median"] = float(
            target["sample_median_sq_nm"]
        )
        corrected_modeling.at[index, "primary_rq_nm_iqr"] = float(
            target["sample_sq_iqr_nm"]
        )
        corrected_modeling.at[index, "primary_rq_nm_min"] = float(
            target["sample_sq_min_nm"]
        )
        corrected_modeling.at[index, "primary_rq_nm_max"] = float(
            target["sample_sq_max_nm"]
        )
        corrected_modeling.at[index, "representative_afm_path"] = str(
            target["representative_afm_height_array"]
        )
        corrected_modeling.at[
            index, "representative_afm_height_array"
        ] = str(target["representative_afm_height_array"])
        corrected_modeling.at[index, "representative_afm_scan_id"] = str(
            target["representative_afm_scan_id"]
        )
    automatic = pd.read_csv(
        repo_path(config["automatic_modeling_manifest"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    corrected_by_sample = corrected_modeling.set_index("sample_id")
    corrected_automatic = automatic.copy()
    target_columns = [
        "primary_afm_scan_count",
        "primary_rq_nm_median",
        "primary_rq_nm_iqr",
        "primary_rq_nm_min",
        "primary_rq_nm_max",
        "representative_afm_path",
        "representative_afm_height_array",
        "representative_afm_scan_id",
        "afm_target_variant",
        "roughness_nomenclature",
    ]
    for index, row in corrected_automatic.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in corrected_by_sample.index:
            continue
        source = corrected_by_sample.loc[sample_id]
        for column in target_columns:
            corrected_automatic.at[index, column] = source[column]
    return primary, targets, corrected_modeling, corrected_automatic


def _qc_metrics(
    qc: pd.DataFrame,
    orders: list[int],
    active_ids: set[str],
    *,
    primary_scan_size_um: float,
    scan_size_tolerance_um: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_primary = qc.loc[
        qc["sample_id"].isin(active_ids)
        & (
            qc["scan_size_x_um"].sub(primary_scan_size_um).abs()
            <= scan_size_tolerance_um
        )
        & (
            qc["scan_size_y_um"].sub(primary_scan_size_um).abs()
            <= scan_size_tolerance_um
        )
        & ~qc["provenance_excluded"].astype(bool)
        & ~qc["excluded_by_hash_deduplication"].astype(bool)
    ]
    for scope, rows in (
        ("all_matched_scans", qc),
        ("active_23_primary_1um_deduplicated", active_primary),
    ):
        for order in orders:
            values = rows[
                ["nanoscope_rq_nm", f"sq_order_{order}_nm"]
            ].dropna()
            if len(values) < 3:
                continue
            truth = values["nanoscope_rq_nm"].to_numpy(float)
            predicted = values[f"sq_order_{order}_nm"].to_numpy(float)
            absolute = np.abs(truth - predicted)
            records.append(
                {
                    "scope": scope,
                    "flatten_order": order,
                    "matched_export_count": int(len(values)),
                    "mae_nm": float(np.mean(absolute)),
                    "median_absolute_error_nm": float(np.median(absolute)),
                    "p90_absolute_error_nm": float(
                        np.percentile(absolute, 90)
                    ),
                    "within_0p2_nm_fraction": float(
                        np.mean(absolute <= 0.2)
                    ),
                    "pearson_r": float(pearsonr(truth, predicted).statistic),
                    "spearman_rho": float(
                        spearmanr(truth, predicted).statistic
                    ),
                }
            )
    return pd.DataFrame(records)


def _figures(
    qc: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    figure_root = repo_path(config["report_root"]) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    orders = [int(value) for value in config["orders"]]

    figure, axes = plt.subplots(
        1,
        len(orders),
        figsize=(3.5 * len(orders), 3.4),
        constrained_layout=True,
    )
    for axis, order in zip(axes, orders):
        x = qc["nanoscope_rq_nm"].to_numpy(float)
        y = qc[f"sq_order_{order}_nm"].to_numpy(float)
        low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        axis.scatter(x, y, s=25, color="#0072B2", alpha=0.78)
        axis.plot([low, high], [low, high], "--", color="#555555", lw=1)
        row = metrics.loc[
            (metrics["scope"] == "all_matched_scans")
            & (metrics["flatten_order"] == order)
        ].iloc[0]
        axis.set_title(
            f"line order {order}\nMAE={row['mae_nm']:.3f} nm, "
            f"r={row['pearson_r']:.3f}"
        )
        axis.set_xlabel("NanoScope export Rq (nm)")
        axis.set_ylabel("recomputed Sq (nm)")
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Independent NanoScope QC supports third-order per-line flattening",
        fontsize=12,
    )
    for suffix in (".png", ".pdf"):
        figure.savefig(
            figure_root / f"Fig1_flatten_order_nanoscope_qc{suffix}",
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    ordered = targets.sort_values("sample_median_sq_nm")
    figure, axis = plt.subplots(figsize=(10.0, 4.1), constrained_layout=True)
    x = np.arange(len(ordered))
    axis.errorbar(
        x,
        ordered["sample_median_sq_nm"],
        yerr=np.vstack(
            [
                ordered["sample_median_sq_nm"]
                - ordered["sample_sq_q25_nm"],
                ordered["sample_sq_q75_nm"]
                - ordered["sample_median_sq_nm"],
            ]
        ),
        fmt="o",
        color="#0072B2",
        ecolor="#7A9DB1",
        capsize=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["sample_id"], rotation=55, ha="right")
    axis.set_ylabel("sample median Sq ± IQR (nm)")
    axis.set_xlabel("growth sample")
    axis.set_title(
        "Corrected 23-growth AFM target distribution "
        "(deduplicated 1 × 1 µm scans)"
    )
    axis.grid(axis="y", alpha=0.18)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            figure_root / f"Fig2_corrected_sample_sq_targets{suffix}",
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _report(
    scans: pd.DataFrame,
    qc: pd.DataFrame,
    metrics: pd.DataFrame,
    duplicates: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    selected_order = int(config["selected_order"])
    best = metrics.loc[
        (metrics["scope"] == "all_matched_scans")
        & (metrics["flatten_order"] == selected_order)
    ].iloc[0]
    flagged = scans.loc[
        scans["provenance_review_status"]
        == "requires_lab_notebook_confirmation"
    ]
    lines = [
        "# AFM metrology repair: third-order scan-line flattening",
        "",
        "## Decision",
        "",
        "The previous global first-order areal plane correction is superseded "
        "for model-target construction. The selected correction fits and "
        "subtracts a cubic polynomial independently from every fast-scan "
        "line. The output is an areal height map, so the recomputed RMS "
        "height is called **Sq**; NanoScope filename values retain their "
        "original **Rq** label as an independent software-export QC record.",
        "",
        "## Evidence",
        "",
        f"- Source decoded ZSensor maps processed: {len(scans)}.",
        f"- Independent labelled NanoScope TIF records matched: {len(qc)}.",
        f"- Selected line-{selected_order} QC MAE: {best['mae_nm']:.4f} nm.",
        f"- Selected line-{selected_order} QC Pearson r: {best['pearson_r']:.4f}.",
        f"- Selected line-{selected_order} within 0.2 nm: "
        f"{100 * best['within_0p2_nm_fraction']:.1f}%.",
        f"- Exact duplicate scan rows identified: {len(duplicates)} "
        f"({duplicates['selected_array_sha256'].nunique()} hash groups).",
        f"- Corrected modeling growths: {len(targets)}.",
        "",
        "## Provenance adjudication",
        "",
        "The local repository cannot replace a lab notebook or acquisition "
        "log. Therefore `6094/N6081_1um_000` is conservatively excluded from "
        "the corrected target, while legacy N69/N74 names are retained but "
        "flagged. These rows still require human confirmation before a paper "
        "freeze.",
        "",
    ]
    for row in flagged[
        [
            "sample_id",
            "afm_file_id",
            "provenance_decision",
            "provenance_reason",
        ]
    ].itertuples(index=False):
        lines.append(
            f"- {row.sample_id}/{row.afm_file_id}: "
            f"{row.provenance_decision}; {row.provenance_reason}"
        )
    lines.extend(
        [
            "",
            "## Target definition",
            "",
            "For each growth, exact selected-array hashes are deduplicated, "
            "unresolved excluded provenance is removed, and Sq is aggregated "
            "as an arithmetic median in nm across primary 1 × 1 µm scans. "
            "Only then is the sample median transformed with the natural log "
            "for model fitting. IQR is retained as within-sample uncertainty.",
            "",
            "## Files",
            "",
            f"- Derived maps: `{config['derived_root']}`",
            f"- Audit tables: `{config['output_root']}`",
            f"- Figures: `{config['report_root']}/figures`",
            "- Raw AFM files and decoded source arrays were read only.",
            "",
        ]
    )
    report_root = repo_path(config["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = repo_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scans, qc = _process_scans(config)
    scans, duplicates = _deduplicate(scans)
    qc = qc.merge(
        scans[
            [
                "sample_id",
                "afm_file_id",
                "scan_size_x_um",
                "scan_size_y_um",
                "provenance_excluded",
                "excluded_by_hash_deduplication",
            ]
        ],
        on=["sample_id", "afm_file_id"],
        how="left",
        validate="many_to_one",
    )
    primary, targets, corrected_modeling, corrected_automatic = (
        _modeling_targets(scans, config)
    )
    active_ids = set(targets["sample_id"].astype(str))
    metrics = _qc_metrics(
        qc,
        [int(value) for value in config["orders"]],
        active_ids,
        primary_scan_size_um=float(config["primary_scan_size_um"]),
        scan_size_tolerance_um=float(config["scan_size_tolerance_um"]),
    )

    write_csv(scans, output_root / "afm_scan_audit.csv")
    write_csv(qc, output_root / "nanoscope_rq_qc_records.csv")
    write_csv(metrics, output_root / "flatten_order_qc_metrics.csv")
    write_csv(duplicates, output_root / "exact_duplicate_scans.csv")
    write_csv(primary, output_root / "primary_deduplicated_scans.csv")
    write_csv(targets, output_root / "sample_sq_targets.csv")
    write_csv(corrected_modeling, output_root / "modeling_manifest.csv")
    write_csv(
        corrected_automatic,
        output_root / "automatic_modeling_manifest.csv",
    )
    _figures(qc, metrics, targets, config)
    _report(scans, qc, metrics, duplicates, targets, config)

    manifest = {
        "experiment_id": config["experiment_id"],
        "source_scan_count": int(len(scans)),
        "nanoscope_qc_record_count": int(len(qc)),
        "duplicate_row_count": int(len(duplicates)),
        "modeling_growth_count": int(len(targets)),
        "selected_flatten_order": int(config["selected_order"]),
        "target_aggregation": config["aggregation"],
        "source_audit_sha256": sha256_file(config["source_audit"]),
        "source_modeling_manifest_sha256": sha256_file(
            config["source_modeling_manifest"]
        ),
        "removelist_sha256": sha256_file(config["removelist_path"]),
        "manual_provenance_review_sha256": sha256_file(
            config["manual_provenance_review"]
        ),
        "raw_data_modified": False,
        "outputs": {
            "scan_audit": display_path(
                output_root / "afm_scan_audit.csv"
            ),
            "sample_targets": display_path(
                output_root / "sample_sq_targets.csv"
            ),
            "modeling_manifest": display_path(
                output_root / "modeling_manifest.csv"
            ),
            "qc_metrics": display_path(
                output_root / "flatten_order_qc_metrics.csv"
            ),
        },
    }
    write_json(manifest, output_root / "experiment_manifest.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build versioned line-flattened AFM metrology targets."
    )
    parser.add_argument(
        "--config",
        default="configs/afm_metrology_line3_v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
