"""Prepare paired RHEED-to-AFM-condition data for MVP-2."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rheed2morph.generative.common import (
    REPO_ROOT,
    display_path,
    load_height_array,
    read_csv_rows,
    read_json,
    resolve_repo_path,
    set_seed,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.rheed_features import compute_rheed_features, impute_feature_rows
from rheed2morph.generative.rheed_video import load_or_cache_rheed_tensor
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


VIDEO_COLUMNS = ("rheed_video_path", "video_path", "output_video_path", "processed_video_path", "rheed_path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare paired RHEED and AFM condition data for MVP-2.")
    parser.add_argument("--mvp1-root", type=Path, default=Path("reports/conditional_latent_diffusion_mvp/20260703_041331"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rheed-root", type=Path, default=None)
    parser.add_argument("--afm-data-index", type=Path, default=None)
    parser.add_argument("--condition-table", type=Path, default=None)
    parser.add_argument("--latents-dir", type=Path, default=None)
    parser.add_argument("--scan-size-filter", type=str, default="1um")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--final-fraction", type=float, default=0.25)
    parser.add_argument("--sampling", type=str, default="uniform")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-key-columns", type=str, default="sample_id,group_id,growth_id")
    parser.add_argument("--video-glob", type=str, default="*raw_crop*.mp4")
    parser.add_argument("--allow-unmatched", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _sample_key(text: str) -> str:
    match = re.search(r"([0-9]{4})", str(text))
    return match.group(1) if match else str(text).strip()


def _resolve_possible_output_path(path: Path, base_dir: Path | None = None) -> Path:
    resolved = resolve_repo_path(path, base_dir)
    if resolved.exists():
        return resolved
    text = path.as_posix()
    if text.startswith("outputs/"):
        alt = resolve_repo_path(Path("data") / text[len("outputs/") :], base_dir)
        if alt.exists():
            return alt
    return resolved


def _default_condition_table(mvp1_root: Path) -> Path:
    root = resolve_repo_path(mvp1_root)
    for candidate in (root / "latents_5epoch" / "condition_table.csv", root / "latents" / "condition_table.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No MVP-1 condition_table.csv found under {root}")


def _default_latents_dir(mvp1_root: Path, condition_table: Path) -> Path:
    if condition_table.parent.name.startswith("latents"):
        return condition_table.parent
    root = resolve_repo_path(mvp1_root)
    for candidate in (root / "latents_5epoch", root / "latents"):
        if (candidate / "latent_stats.json").is_file():
            return candidate
    return condition_table.parent


def _condition_schema(condition_table: Path, latents_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(condition_table)
    if not rows:
        raise RuntimeError(f"Condition table is empty: {condition_table}")
    cond_columns = [key for key in rows[0] if key.startswith("cond_")]
    descriptor_columns = [key[len("cond_") :] for key in cond_columns]
    raw_descriptor_columns = [name for name in descriptor_columns if name in rows[0]]
    proto_values = [int(float(row["prototype_id"])) for row in rows if row.get("prototype_id", "") != ""]
    latent_stats_path = latents_dir / "latent_stats.json"
    latent_stats = read_json(latent_stats_path) if latent_stats_path.is_file() else {}
    return {
        "condition_table": display_path(condition_table),
        "latents_dir": display_path(latents_dir),
        "condition_columns": cond_columns,
        "descriptor_columns": descriptor_columns,
        "raw_descriptor_columns": raw_descriptor_columns,
        "target_columns": cond_columns,
        "condition_values": "standardized_descriptor_columns_prefixed_cond_",
        "raw_descriptor_values": "physical_or_plane_corrected_height_descriptor_units",
        "prototype_label_exists": bool(proto_values),
        "prototype_count": int(max(proto_values) + 1) if proto_values else 0,
        "descriptor_train_mean": latent_stats.get("descriptor_train_mean", {}),
        "descriptor_train_std": latent_stats.get("descriptor_train_std", {}),
        "sampling_expectation": "MVP-1 diffusion sampler reads standardized cond_* columns and prototype_id, then appends prototype one-hot sized from the diffusion checkpoint config.",
    }


def _discover_from_manifest(manifest: Path, base_dir: Path, sample_key_columns: list[str]) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest)
    records: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        video_path = None
        for column in VIDEO_COLUMNS:
            value = row.get(column, "").strip()
            if not value:
                continue
            candidate = _resolve_possible_output_path(Path(value), manifest.parent)
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            continue
        sample_id = ""
        rule = ""
        for column in sample_key_columns:
            if row.get(column, "").strip():
                sample_id = _sample_key(row[column])
                rule = f"manifest_column:{column}"
                break
        if not sample_id:
            sample_id = _sample_key(video_path.as_posix())
            rule = "video_path_numeric_token"
        record = {
            "rheed_record_id": f"rheed_{index:05d}",
            "sample_key": sample_id,
            "sample_id": sample_id,
            "rheed_video_path": display_path(video_path),
            "rheed_key_rule": rule,
            "rheed_source": display_path(manifest),
        }
        for column in ("fps", "source_frame_count", "written_frame_count", "material"):
            if row.get(column, "") != "":
                record[column] = row[column]
        records.append(record)
    return records


def _discover_video_files(rheed_root: Path | None, video_glob: str) -> list[dict[str, str]]:
    roots = [resolve_repo_path(rheed_root)] if rheed_root is not None else [
        REPO_ROOT / "data" / "rheed_roi_shadow_right_v2_main_raw_crop_videos_256",
        REPO_ROOT / "data",
        REPO_ROOT / "reports",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.rglob(video_glob)))
    if not paths:
        for root in roots:
            if root.exists():
                paths.extend(sorted(root.rglob("*.mp4")))
    seen: set[Path] = set()
    records: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        sample_id = _sample_key(resolved.as_posix())
        records.append(
            {
                "rheed_record_id": f"rheed_{index:05d}",
                "sample_key": sample_id,
                "sample_id": sample_id,
                "rheed_video_path": display_path(resolved),
                "rheed_key_rule": "video_path_numeric_token",
                "rheed_source": "filesystem_video_glob",
            }
        )
    return records


def _discover_rheed_records(args: argparse.Namespace, sample_key_columns: list[str]) -> list[dict[str, str]]:
    if args.manifest is not None:
        return _discover_from_manifest(resolve_repo_path(args.manifest), REPO_ROOT, sample_key_columns)
    manifest = REPO_ROOT / "data" / "rheed_roi_shadow_right_v2_main_raw_crop_videos_256" / "raw_crop_video_manifest.csv"
    if manifest.is_file():
        records = _discover_from_manifest(manifest, REPO_ROOT, sample_key_columns)
        existing = [row for row in records if resolve_repo_path(Path(row["rheed_video_path"])).exists()]
        if existing:
            return existing
    return _discover_video_files(args.rheed_root, str(args.video_glob))


def _split_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    sample_counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")}
    group_counts = {
        split: len({str(row["group_id"]) for row in rows if row["split"] == split}) for split in ("train", "val", "test")
    }
    return sample_counts, group_counts


def _write_pair_grid(out_path: Path, paired_rows: list[dict[str, Any]], mvp1_root: Path) -> None:
    if not paired_rows:
        return
    ae = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate in (
        mvp1_root / "afm_autoencoder_5epoch" / "checkpoints" / "best.pt",
        mvp1_root / "afm_autoencoder" / "checkpoints" / "best.pt",
    ):
        if candidate.is_file():
            try:
                ae, _payload = load_autoencoder_checkpoint(candidate, str(device))
                ae.to(device).eval()
            except Exception:
                ae = None
            break
    rows: list[list[np.ndarray]] = []
    titles: list[str] = []
    for row in paired_rows[:8]:
        rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"]
        final_frame = rheed[-1, 0]
        afm_path = row.get("network_input_path", "")
        afm = load_height_array(resolve_repo_path(Path(afm_path))) if afm_path else np.zeros((128, 128), dtype=np.float32)
        if ae is not None:
            with torch.no_grad():
                recon = ae(torch.from_numpy(afm[None, None].astype(np.float32)).to(device))[0][0, 0].detach().cpu().numpy()
        else:
            recon = np.zeros_like(afm)
        rows.append([final_frame, afm, recon])
        titles.append(str(row.get("sample_id", row.get("row_id", ""))))
    write_panel_grid(out_path, rows, ["RHEED final frame", "true AFM", "AE reconstruction"], titles)


def prepare_rheed_condition_dataset(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cached_rheed_tensors"
    sample_key_columns = [value.strip() for value in str(args.sample_key_columns).split(",") if value.strip()]
    mvp1_root = resolve_repo_path(args.mvp1_root)
    condition_table = resolve_repo_path(args.condition_table) if args.condition_table else _default_condition_table(mvp1_root)
    latents_dir = resolve_repo_path(args.latents_dir) if args.latents_dir else _default_latents_dir(mvp1_root, condition_table)
    schema = _condition_schema(condition_table, latents_dir)
    condition_rows = read_csv_rows(condition_table)
    condition_by_key: dict[str, dict[str, str]] = {}
    for row in condition_rows:
        for column in sample_key_columns:
            if row.get(column, "").strip():
                condition_by_key[_sample_key(row[column])] = row
                break
    rheed_records = _discover_rheed_records(args, sample_key_columns)
    paired_rows: list[dict[str, Any]] = []
    unmatched_rheed: list[dict[str, Any]] = []
    matched_condition_keys: set[str] = set()
    feature_rows: list[dict[str, Any]] = []
    video_failures: list[dict[str, str]] = []
    for record in rheed_records:
        condition = condition_by_key.get(record["sample_key"])
        if condition is None:
            unmatched_rheed.append(record)
            continue
        try:
            tensor, cache_path, video_meta = load_or_cache_rheed_tensor(
                resolve_repo_path(Path(record["rheed_video_path"])),
                cache_dir=cache_dir,
                frames=int(args.frames),
                image_size=int(args.image_size),
                final_fraction=float(args.final_fraction),
                sampling=str(args.sampling),
            )
        except Exception as exc:
            video_failures.append({**record, "error_message": str(exc)})
            if args.strict:
                raise
            continue
        base: dict[str, Any] = {
            "pair_id": f"pair_{len(paired_rows) + 1:05d}",
            "row_id": condition["row_id"],
            "sample_id": condition.get("sample_id", record["sample_id"]),
            "group_id": condition.get("group_id", condition.get("sample_id", record["sample_id"])),
            "split": condition.get("split", ""),
            "rheed_video_path": record["rheed_video_path"],
            "cached_tensor_path": display_path(cache_path),
            "network_input_path": condition.get("network_input_path", ""),
            "descriptor_height_path": condition.get("descriptor_height_path", ""),
            "prototype_id": condition.get("prototype_id", ""),
            "rheed_key_rule": record["rheed_key_rule"],
            "condition_key_rule": f"condition_column:{sample_key_columns[0]}",
            "source_frame_count": video_meta.get("source_frame_count", ""),
            "frames_used": int(args.frames),
            "image_size": int(args.image_size),
            "final_fraction": float(args.final_fraction),
            "normalization": video_meta.get("normalization", ""),
        }
        for col in schema["raw_descriptor_columns"] + schema["condition_columns"]:
            base[col] = condition.get(col, "")
        paired_rows.append(base)
        matched_condition_keys.add(record["sample_key"])
        features = compute_rheed_features(tensor)
        feature_rows.append({"pair_id": base["pair_id"], "row_id": base["row_id"], "sample_id": base["sample_id"], "group_id": base["group_id"], "split": base["split"], **features})
        if args.limit is not None and len(paired_rows) >= int(args.limit):
            break
    unmatched_afm = []
    for row in condition_rows:
        key = _sample_key(row.get("sample_id", row.get("group_id", "")))
        if key not in matched_condition_keys:
            unmatched_afm.append({"row_id": row.get("row_id", ""), "sample_id": row.get("sample_id", ""), "group_id": row.get("group_id", ""), "split": row.get("split", "")})
    if not paired_rows and not args.allow_unmatched:
        write_csv_rows(out_dir / "unmatched_rheed.csv", unmatched_rheed)
        write_csv_rows(out_dir / "unmatched_afm_conditions.csv", unmatched_afm)
        raise RuntimeError("No matched RHEED-AFM condition pairs were found. See unmatched CSV files.")
    train_mask = np.asarray([row.get("split") == "train" for row in feature_rows], dtype=bool)
    feature_columns = [key for key in feature_rows[0] if key not in {"pair_id", "row_id", "sample_id", "group_id", "split"}] if feature_rows else []
    if feature_rows:
        feature_rows, feature_impute_counts, feature_means, feature_stds = impute_feature_rows(feature_rows, feature_columns, train_mask)
    else:
        feature_impute_counts, feature_means, feature_stds = {}, {}, {}
    write_csv_rows(out_dir / "paired_rheed_condition_index.csv", paired_rows)
    sample_split_counts, group_split_counts = _split_counts(paired_rows)
    split_rows = [
        {"split": split, "sample_count": sample_split_counts[split], "group_count": group_split_counts[split]}
        for split in ("train", "val", "test")
    ]
    write_csv_rows(out_dir / "paired_split_summary.csv", split_rows, ["split", "sample_count", "group_count"])
    write_csv_rows(out_dir / "unmatched_rheed.csv", unmatched_rheed)
    write_csv_rows(out_dir / "unmatched_afm_conditions.csv", unmatched_afm)
    write_csv_rows(out_dir / "rheed_handcrafted_features.csv", feature_rows)
    schema.update(
        {
            "rheed_feature_columns": feature_columns,
            "rheed_feature_imputation_counts": feature_impute_counts,
            "rheed_feature_train_mean": feature_means,
            "rheed_feature_train_std": feature_stds,
            "metadata_columns": ["source_frame_count", "frames_used", "image_size", "final_fraction"],
        }
    )
    write_json(out_dir / "condition_schema.json", schema)
    _write_pair_grid(out_dir / "pair_grid.png", paired_rows, mvp1_root)
    inventory = {
        "mvp1_root": display_path(mvp1_root),
        "condition_table": display_path(condition_table),
        "latents_dir": display_path(latents_dir),
        "rheed_records_found": len(rheed_records),
        "matched_pair_count": len(paired_rows),
        "unmatched_rheed_count": len(unmatched_rheed),
        "unmatched_afm_condition_count": len(unmatched_afm),
        "split_sample_counts": sample_split_counts,
        "split_group_counts": group_split_counts,
        "sample_key_matching_rule": "RHEED numeric token joined to MVP-1 condition sample_id/group_id/growth_id, preferring explicit columns when present.",
        "frames": int(args.frames),
        "final_fraction": float(args.final_fraction),
        "image_size": int(args.image_size),
        "normalization": "percentile_clip_1_99_to_0_1",
        "cached_tensor_count": len(list(cache_dir.glob("*.npz"))),
        "video_read_failures": len(video_failures),
        "video_failure_examples": video_failures[:10],
        "feature_columns": feature_columns,
        "feature_imputation_counts": feature_impute_counts,
        "prototype_label_exists": schema["prototype_label_exists"],
        "prototype_count": schema["prototype_count"],
    }
    write_json(out_dir / "rheed_data_inventory.json", inventory)
    return inventory


def main() -> None:
    args = build_parser().parse_args()
    inventory = prepare_rheed_condition_dataset(args)
    print(f"Wrote paired RHEED-condition data to {display_path(resolve_repo_path(args.out))}")
    print(f"matched_pairs={inventory['matched_pair_count']} rheed_found={inventory['rheed_records_found']} failures={inventory['video_read_failures']}")


if __name__ == "__main__":
    main()
