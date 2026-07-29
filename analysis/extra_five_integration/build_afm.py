from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.afm_metrology_repair.line_flatten import line_flatten, sq_nm
from analysis.rheed_video_afm_story.common import (
    display_path,
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)
from rheed2morph.afm.inspect import (
    choose_primary_channel,
    convert_height_to_nm,
    inspect_channels,
    make_safe_id,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _canonical_sample_id(directory_name: str, aliases: dict[str, str]) -> str:
    return str(aliases.get(str(directory_name), str(directory_name)))


def _raw_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )


def _source_inventory(config: dict[str, Any]) -> pd.DataFrame:
    root = repo_path(config["raw_afm_root"])
    aliases = {
        str(key): str(value)
        for key, value in config.get("sample_directory_aliases", {}).items()
    }
    included = set(map(str, config["included_samples"]))
    excluded = set(map(str, config["excluded_samples"]))
    records: list[dict[str, Any]] = []
    for sample_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        canonical = _canonical_sample_id(sample_directory.name, aliases)
        if canonical not in included | excluded:
            decision = "unrecognized_not_used"
            reason = "directory is outside the declared second-batch cohort"
        elif canonical in excluded:
            decision = "excluded"
            reason = "operator-declared rejected sample; N6324 must not be used"
        else:
            decision = "include"
            reason = "declared extra-five growth"
        for raw_path in _raw_files(sample_directory):
            records.append(
                {
                    "sample_directory": sample_directory.name,
                    "sample_id": canonical,
                    "raw_afm_path": display_path(raw_path),
                    "raw_afm_sha256": sha256_file(raw_path),
                    "size_bytes": int(raw_path.stat().st_size),
                    "decision": decision,
                    "decision_reason": reason,
                    "raw_data_modified": False,
                }
            )
    inventory = pd.DataFrame(records).sort_values(
        ["sample_id", "raw_afm_path"]
    ).reset_index(drop=True)
    observed = set(
        inventory.loc[inventory["decision"] == "include", "sample_id"].astype(str)
    )
    if observed != included:
        raise RuntimeError(
            f"extra-five AFM inventory mismatch: expected {sorted(included)}, "
            f"found {sorted(observed)}"
        )
    if set(inventory.loc[inventory["decision"] == "excluded", "sample_id"]) != excluded:
        raise RuntimeError("the declared N6324 exclusion is absent from the inventory")
    return inventory


def _save_selected_render(
    array: np.ndarray,
    destination: Path,
    *,
    title: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    low, high = np.nanpercentile(np.asarray(array, dtype=float), [1, 99])
    figure, axis = plt.subplots(figsize=(3.5, 3.2), dpi=130)
    image = axis.imshow(
        array,
        cmap="afmhot",
        origin="upper",
        extent=(0, 1, 1, 0),
        vmin=float(low),
        vmax=float(high),
    )
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_title(title, fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label("height (nm)")
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _quadrants(array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    height, width = array.shape
    if height % 2 or width % 2:
        raise RuntimeError(
            f"2 × 2 µm harmonization requires even dimensions, got {array.shape}"
        )
    half_y, half_x = height // 2, width // 2
    return [
        ("q00_top_left", array[:half_y, :half_x]),
        ("q01_top_right", array[:half_y, half_x:]),
        ("q10_bottom_left", array[half_y:, :half_x]),
        ("q11_bottom_right", array[half_y:, half_x:]),
    ]


def _decode_and_harmonize(
    inventory: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = repo_path(config["canonical_data_root"])
    decoded_root = data_root / "decoded_afm_2um"
    derived_root = data_root / "afm_line_flattened_1um"
    orders = [int(value) for value in config["orders"]]
    selected_order = int(config["selected_order"])
    scan_records: list[dict[str, Any]] = []
    decode_records: list[dict[str, Any]] = []
    active = inventory.loc[inventory["decision"] == "include"].copy()
    expected_raw = int(config["expected_included_raw_afm_scan_count"])
    if len(active) != expected_raw:
        raise RuntimeError(
            f"expected {expected_raw} included raw AFM scans, found {len(active)}"
        )

    for position, row in enumerate(active.itertuples(index=False), start=1):
        raw_path = repo_path(row.raw_afm_path)
        sample_id = str(row.sample_id)
        parent_scan_id = make_safe_id(raw_path.name)
        result = inspect_channels(raw_path)
        channel = choose_primary_channel(result)
        converted, unit, conversion = convert_height_to_nm(
            result.arrays[channel],
            result.channels[channel].unit,
        )
        height = np.asarray(converted, dtype=np.float64)
        if unit != "nm" or height.ndim != 2 or not np.isfinite(height).all():
            raise RuntimeError(
                f"{sample_id}/{parent_scan_id}: invalid decoded ZSensor map "
                f"(unit={unit}, shape={height.shape})"
            )
        scan_size = result.channels[channel].scan_size_um
        if scan_size is None:
            scan_size = [2.0, 2.0]
        if not np.allclose(scan_size, [2.0, 2.0], rtol=0, atol=1e-6):
            raise RuntimeError(
                f"{sample_id}/{parent_scan_id}: expected 2 × 2 µm, got {scan_size}"
            )
        decoded_dir = decoded_root / sample_id / parent_scan_id
        decoded_dir.mkdir(parents=True, exist_ok=True)
        decoded_path = decoded_dir / f"{parent_scan_id}_zsensor_nm.npy"
        np.save(decoded_path, height.astype(np.float32))
        decoded_metadata = {
            "sample_id": sample_id,
            "parent_afm_scan_id": parent_scan_id,
            "raw_afm_path": display_path(raw_path),
            "raw_afm_sha256": sha256_file(raw_path),
            "channel": channel,
            "height_unit": "nm",
            "unit_conversion": conversion,
            "scan_size_um": [2.0, 2.0],
            "resolution": list(map(int, height.shape)),
            "decoded_array_path": display_path(decoded_path),
            "decoded_array_sha256": sha256_file(decoded_path),
            "decoded_content_sha256": _array_hash(height),
            "raw_data_modified": False,
        }
        write_json(decoded_metadata, decoded_dir / "metadata.json")
        decode_records.append(decoded_metadata)

        for quadrant_name, quadrant in _quadrants(height):
            subfield_id = f"{parent_scan_id}__{quadrant_name}"
            values_by_order: dict[int, float] = {}
            selected_path: Path | None = None
            selected_hash = ""
            for order in orders:
                corrected, background = line_flatten(quadrant, order=order)
                order_dir = (
                    derived_root
                    / f"order_{order}"
                    / sample_id
                    / subfield_id
                )
                order_dir.mkdir(parents=True, exist_ok=True)
                corrected_path = (
                    order_dir / f"{subfield_id}_line_flatten_o{order}.npy"
                )
                background_path = (
                    order_dir / f"{subfield_id}_background_o{order}.npy"
                )
                np.save(corrected_path, corrected)
                np.save(background_path, background)
                values_by_order[order] = sq_nm(corrected)
                metadata = {
                    "sample_id": sample_id,
                    "afm_file_id": subfield_id,
                    "parent_afm_scan_id": parent_scan_id,
                    "quadrant": quadrant_name,
                    "source_raw_afm_file": display_path(raw_path),
                    "source_raw_sha256": sha256_file(raw_path),
                    "source_decoded_zsensor": display_path(decoded_path),
                    "source_decoded_sha256": sha256_file(decoded_path),
                    "source_scan_size_um": [2.0, 2.0],
                    "subfield_scan_size_um": [1.0, 1.0],
                    "height_unit": "nm",
                    "resolution": list(map(int, corrected.shape)),
                    "line_flatten_order": order,
                    "line_flatten_scope": (
                        "crop one non-overlapping 1 × 1 µm subfield first, "
                        "then fit each fast-scan row independently"
                    ),
                    "sq_nm": values_by_order[order],
                    "corrected_array": display_path(corrected_path),
                    "background_array": display_path(background_path),
                }
                write_json(
                    metadata,
                    order_dir / f"{subfield_id}_line_flatten_o{order}_metadata.json",
                )
                if order == selected_order:
                    selected_path = corrected_path
                    selected_hash = _array_hash(corrected)
                    _save_selected_render(
                        corrected,
                        order_dir / f"{subfield_id}_line_flatten_o{order}.png",
                        title=(
                            f"{sample_id} / {quadrant_name}\n"
                            f"1 × 1 µm line-{order}; Sq="
                            f"{values_by_order[order]:.3f} nm"
                        ),
                    )
            assert selected_path is not None
            scan_records.append(
                {
                    "sample_id": sample_id,
                    "growth_run_id": sample_id,
                    "afm_file_id": subfield_id,
                    "parent_afm_scan_id": parent_scan_id,
                    "quadrant": quadrant_name,
                    "raw_afm_file": display_path(raw_path),
                    "raw_afm_sha256": sha256_file(raw_path),
                    "decoded_zsensor_path": display_path(decoded_path),
                    "decoded_zsensor_sha256": sha256_file(decoded_path),
                    "height_array_path": display_path(selected_path),
                    "afm_path": display_path(selected_path),
                    "afm_preprocessing_variant": (
                        "crop_2um_to_nonoverlap_1um_then_line3_scanline_flatten_v1"
                    ),
                    "roughness_metric": "Sq_areal_RMS_height_nm",
                    "line_flatten_order": selected_order,
                    "selected_array_sha256": selected_hash,
                    "source_scan_size_um": 2.0,
                    "scan_size_um": 1.0,
                    "scan_size_x_um": 1.0,
                    "scan_size_y_um": 1.0,
                    "resolution_x": int(quadrant.shape[1]),
                    "resolution_y": int(quadrant.shape[0]),
                    "height_unit": "nm",
                    "channel": channel,
                    "sq_nm": values_by_order[selected_order],
                    "rq_recomputed_nm": values_by_order[selected_order],
                    "provenance_decision": "include",
                    "provenance_reason": "declared extra-five growth",
                    "provenance_review_status": "operator_confirmed",
                    "provenance_excluded": False,
                    "include_for_modeling": True,
                    **{
                        f"sq_order_{order}_nm": values_by_order[order]
                        for order in orders
                    },
                }
            )
        print(
            f"[AFM {position:02d}/{len(active):02d}] "
            f"{sample_id}/{parent_scan_id}",
            flush=True,
        )
    return pd.DataFrame(scan_records), pd.DataFrame(decode_records)


def _deduplicate(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = scans.copy()
    result["duplicate_group_size"] = result.groupby(
        "selected_array_sha256"
    )["selected_array_sha256"].transform("size")
    result["duplicate_rank"] = result.groupby(
        "selected_array_sha256", sort=False
    ).cumcount()
    result["excluded_by_hash_deduplication"] = result["duplicate_rank"] > 0
    result["include_for_modeling"] = (
        ~result["provenance_excluded"].astype(bool)
        & ~result["excluded_by_hash_deduplication"].astype(bool)
    )
    return (
        result,
        result.loc[result["duplicate_group_size"] > 1].copy(),
    )


def _sample_targets(scans: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active = scans.loc[scans["include_for_modeling"].astype(bool)]
    for sample_id, rows in active.groupby("sample_id", sort=True):
        values = rows["sq_nm"].to_numpy(float)
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        representative = (
            rows.assign(_distance=(rows["sq_nm"] - median).abs())
            .sort_values(["_distance", "afm_file_id"])
            .iloc[0]
        )
        records.append(
            {
                "sample_id": str(sample_id),
                "growth_run_id": str(sample_id),
                "primary_afm_scan_count": int(len(rows)),
                "independent_parent_afm_scan_count": int(
                    rows["parent_afm_scan_id"].nunique()
                ),
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
                "representative_scan_sq_nm": float(representative["sq_nm"]),
                "aggregation": (
                    "arithmetic median in nm across hash-deduplicated "
                    "non-overlapping 1 × 1 µm subfields; log afterwards"
                ),
            }
        )
    return pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)


def _combined_tables(
    extra_scans: pd.DataFrame,
    extra_targets: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_scans = pd.read_csv(
        repo_path(config["base_primary_scan_table"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    base_targets = pd.read_csv(
        repo_path(config["base_sample_target_table"]),
        dtype={"sample_id": str, "growth_run_id": str},
    )
    active_extra = extra_scans.loc[extra_scans["include_for_modeling"]].copy()
    combined_scans = pd.concat(
        [base_scans, active_extra],
        ignore_index=True,
        sort=False,
    )
    combined_targets = pd.concat(
        [base_targets, extra_targets],
        ignore_index=True,
        sort=False,
    ).sort_values("sample_id")
    expected = int(config["expected_combined_growth_count"])
    if combined_targets["sample_id"].nunique() != expected:
        raise RuntimeError(
            f"expected {expected} combined growths, found "
            f"{combined_targets['sample_id'].nunique()}"
        )
    if "N6324" in set(combined_targets["sample_id"].astype(str)):
        raise RuntimeError("N6324 entered the combined AFM targets")
    if combined_scans["selected_array_sha256"].duplicated().any():
        duplicates = combined_scans.loc[
            combined_scans["selected_array_sha256"].duplicated(False),
            ["sample_id", "afm_file_id", "selected_array_sha256"],
        ]
        raise RuntimeError(
            "cross-cohort AFM duplicate arrays found:\n"
            + duplicates.to_string(index=False)
        )
    return combined_scans, combined_targets


def _plot_targets(
    combined_targets: pd.DataFrame,
    extra_targets: pd.DataFrame,
    report_root: Path,
) -> None:
    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    ordered = combined_targets.sort_values("sample_median_sq_nm").reset_index(
        drop=True
    )
    extra_ids = set(extra_targets["sample_id"].astype(str))
    colors = [
        "#D55E00" if str(sample) in extra_ids else "#0072B2"
        for sample in ordered["sample_id"]
    ]
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(11.2, 4.5), constrained_layout=True)
    axis.vlines(
        x,
        ordered["sample_sq_q25_nm"],
        ordered["sample_sq_q75_nm"],
        color=colors,
        lw=1.6,
        alpha=0.8,
    )
    axis.scatter(
        x,
        ordered["sample_median_sq_nm"],
        c=colors,
        edgecolor="black",
        linewidth=0.4,
        s=42,
        zorder=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(ordered["sample_id"], rotation=60, ha="right")
    axis.set_ylabel("sample median Sq ± IQR (nm)")
    axis.set_xlabel("growth sample")
    axis.set_title(
        "Harmonized 28-growth AFM targets: original 1 µm scans and "
        "extra-five 1 µm subfields"
    )
    axis.grid(axis="y", alpha=0.18)
    axis.text(
        0.01,
        0.97,
        "blue: original 23   orange: extra five",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    for suffix in (".png", ".pdf"):
        figure.savefig(
            figure_root / f"Fig1_full28_sq_targets{suffix}",
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def run(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    data_root = repo_path(config["canonical_data_root"])
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    for path in (data_root, output_root, report_root):
        path.mkdir(parents=True, exist_ok=True)
    inventory = _source_inventory(config)
    write_csv(inventory, data_root / "raw_afm_source_inventory.csv")
    write_csv(inventory, report_root / "raw_afm_source_inventory.csv")
    scans, decoded = _decode_and_harmonize(inventory, config)
    scans, duplicates = _deduplicate(scans)
    targets = _sample_targets(scans)
    if set(targets["sample_id"]) != set(map(str, config["included_samples"])):
        raise RuntimeError("extra-five AFM target IDs do not match the declaration")
    combined_scans, combined_targets = _combined_tables(
        scans, targets, config
    )
    for frame, name in (
        (decoded, "decoded_afm_audit.csv"),
        (scans, "extra_five_1um_scan_audit.csv"),
        (duplicates, "extra_five_exact_duplicate_subfields.csv"),
        (targets, "extra_five_sample_sq_targets.csv"),
        (combined_scans, "combined_primary_1um_scans.csv"),
        (combined_targets, "combined_sample_sq_targets.csv"),
    ):
        write_csv(frame, output_root / name)
    _plot_targets(combined_targets, targets, report_root)
    historical = pd.DataFrame(
        [
            {
                "path": path,
                "status": "historical_derived_retained_not_used",
                "replacement": config["canonical_data_root"],
                "deletion_performed": False,
            }
            for path in config["historical_extra_five_derived_roots"]
        ]
    )
    write_csv(historical, report_root / "historical_derived_cleanup_audit.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "included_samples": list(map(str, config["included_samples"])),
        "excluded_samples": list(map(str, config["excluded_samples"])),
        "raw_afm_scan_count_included": int(
            (inventory["decision"] == "include").sum()
        ),
        "raw_afm_scan_count_excluded": int(
            (inventory["decision"] == "excluded").sum()
        ),
        "decoded_scan_count": int(len(decoded)),
        "harmonized_1um_subfield_count": int(len(scans)),
        "extra_growth_count": int(len(targets)),
        "combined_growth_count": int(len(combined_targets)),
        "combined_1um_scan_count": int(len(combined_scans)),
        "selected_flatten_order": int(config["selected_order"]),
        "harmonization": (
            "each 2 × 2 µm 512 × 512 ZSensor map is split into four "
            "non-overlapping 1 × 1 µm 256 × 256 subfields before independent "
            "third-order per-row flattening"
        ),
        "group_boundary": (
            "all subfields from one parent scan and sample remain in the same "
            "growth-run fold"
        ),
        "n6324_used": False,
        "raw_data_modified": False,
        "historical_derived_deleted": False,
        "source_inventory_sha256": sha256_file(
            data_root / "raw_afm_source_inventory.csv"
        ),
    }
    write_json(manifest, data_root / "dataset_manifest.json")
    write_json(manifest, output_root / "afm_integration_manifest.json")
    write_json(manifest, report_root / "afm_integration_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/extra_five_line3_full28_v1.json",
    )
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
