"""Prepare RHEED SSL and supervised morphology-condition indexes for MVP-6."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

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
from rheed2morph.generative.condition_control_v3_utils import raw_descriptor_to_condition
from rheed2morph.generative.rheed_features import compute_rheed_features, impute_feature_rows
from rheed2morph.generative.rheed_video import VIDEO_SUFFIXES, load_or_cache_rheed_tensor
from rheed2morph.generative.visualization import write_panel_grid


VIDEO_COLUMNS = ("rheed_video_path", "video_path", "output_video_path", "processed_video_path", "input_path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RHEED SSL video/frame indexes and supervised v3-condition pairs.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mvp2-root", type=Path, default=Path("reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816"))
    parser.add_argument("--mvp2-paired-index", type=Path, default=None)
    parser.add_argument("--condition-schema", type=Path, default=Path("reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json"))
    parser.add_argument("--condition-table", type=Path, default=Path("reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/raw_crop_video_manifest.csv"))
    parser.add_argument("--rheed-root", type=Path, default=Path("data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--final-fraction", type=float, default=0.25)
    parser.add_argument("--sampling", type=str, default="uniform")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def sample_key(text: str) -> str:
    match = re.search(r"([0-9]{4})", str(text))
    return match.group(1) if match else str(text).strip()


def resolve_data_path(path: str | Path, base_dir: Path | None = None) -> Path:
    raw = Path(path)
    resolved = resolve_repo_path(raw, base_dir)
    if resolved.exists():
        return resolved
    text = raw.as_posix()
    if text.startswith("outputs/"):
        alt = resolve_repo_path(Path("data") / text[len("outputs/") :], base_dir)
        if alt.exists():
            return alt
    if text.startswith("raw_RHEED_selected/"):
        alt = resolve_repo_path(Path("data/raw") / text, base_dir)
        if alt.exists():
            return alt
    return resolved


def _default_mvp2_pairs(root: Path) -> Path:
    root = resolve_repo_path(root)
    candidates = [
        root / "data_limit64" / "paired_rheed_condition_index.csv",
        root / "data" / "paired_rheed_condition_index.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.glob("**/paired_rheed_condition_index.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No MVP-2 paired index found under {root}")


def _discover_rheed_records(manifest: Path, rheed_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    manifest = resolve_repo_path(manifest)
    if manifest.is_file():
        for index, row in enumerate(read_csv_rows(manifest), start=1):
            if row.get("status", "success") not in {"", "success"}:
                continue
            video_path: Path | None = None
            for column in VIDEO_COLUMNS:
                value = row.get(column, "").strip()
                if not value:
                    continue
                candidate = resolve_data_path(value, manifest.parent)
                if candidate.exists() and candidate.suffix.lower() in VIDEO_SUFFIXES.union({".npy", ".npz"}):
                    video_path = candidate
                    break
            if video_path is None:
                continue
            key = sample_key(row.get("sample_id", video_path.as_posix()))
            record = {
                "video_id": f"rheed_{index:05d}",
                "sample_key": key,
                "sample_id": key,
                "group_id": key,
                "rheed_video_path": display_path(video_path),
                "rheed_source": display_path(manifest),
                "source_frame_count_manifest": row.get("source_frame_count", ""),
                "written_frame_count_manifest": row.get("written_frame_count", ""),
                "fps": row.get("fps", ""),
            }
            records.append(record)
    if records:
        return records
    paths = sorted(resolve_repo_path(rheed_root).rglob("*.mp4")) if resolve_repo_path(rheed_root).exists() else []
    for index, path in enumerate(paths, start=1):
        key = sample_key(path.as_posix())
        records.append(
            {
                "video_id": f"rheed_{index:05d}",
                "sample_key": key,
                "sample_id": key,
                "group_id": key,
                "rheed_video_path": display_path(path),
                "rheed_source": "filesystem_video_glob",
            }
        )
    return records


def _schema_row_from_mvp2(row: dict[str, str], schema: dict[str, Any], v3_by_height: dict[str, dict[str, str]]) -> dict[str, Any]:
    key = row.get("descriptor_height_path", "")
    matched = v3_by_height.get(key) or v3_by_height.get(display_path(resolve_data_path(key))) if key else None
    out: dict[str, Any] = {
        "pair_id": row.get("pair_id", ""),
        "row_id": matched.get("row_id", row.get("row_id", "")) if matched else row.get("row_id", ""),
        "parent_mvp2_row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "group_id": row.get("group_id", row.get("sample_id", "")),
        "split": row.get("split", ""),
        "rheed_video_path": row.get("rheed_video_path", ""),
        "cached_tensor_path": row.get("cached_tensor_path", ""),
        "network_input_path": matched.get("network_input_path", row.get("network_input_path", "")) if matched else row.get("network_input_path", ""),
        "descriptor_height_path": matched.get("descriptor_height_path", row.get("descriptor_height_path", "")) if matched else row.get("descriptor_height_path", ""),
        "prototype_id": matched.get("prototype_id", row.get("prototype_id", "")) if matched else row.get("prototype_id", ""),
        "condition_source": "v3_condition_table_descriptor_height_path" if matched else "mvp2_raw_descriptors_restandardized_to_v3",
        "source_frame_count": row.get("source_frame_count", ""),
        "frames_used": row.get("frames_used", ""),
        "image_size": row.get("image_size", ""),
        "final_fraction": row.get("final_fraction", ""),
        "normalization": row.get("normalization", ""),
    }
    for name in schema["descriptor_columns"]:
        raw_text = matched.get(name, "") if matched else row.get(name, "")
        if raw_text == "":
            raw = float(schema["descriptor_train_mean"][name])
        else:
            raw = float(raw_text)
        out[name] = f"{raw:.10g}"
        out[f"cond_{name}"] = f"{raw_descriptor_to_condition(name, raw, schema):.10g}"
    proto_count = int(schema.get("prototype_count", 0))
    if out.get("prototype_id", "") != "":
        proto = int(float(out["prototype_id"]))
        if proto_count > 0 and not (0 <= proto < proto_count):
            out["prototype_id"] = ""
    return out


def _write_rheed_preview(path: Path, video_rows: list[dict[str, Any]]) -> None:
    panels: list[list[np.ndarray]] = []
    titles: list[str] = []
    for row in video_rows[:8]:
        cache = row.get("cached_tensor_path", "")
        if not cache:
            continue
        frames = np.asarray(np.load(resolve_repo_path(Path(cache)))["frames"], dtype=np.float32)
        panels.append([frames[0, 0], frames[len(frames) // 2, 0], frames[-1, 0]])
        titles.append(str(row.get("sample_id", row.get("video_id", ""))))
    write_panel_grid(path, panels, ["first", "middle", "final"], titles)


def _write_pair_preview(path: Path, paired_rows: list[dict[str, Any]]) -> None:
    panels: list[list[np.ndarray]] = []
    titles: list[str] = []
    for row in paired_rows[:8]:
        try:
            frames = np.asarray(np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"], dtype=np.float32)
            afm = load_height_array(resolve_repo_path(Path(row.get("descriptor_height_path", row.get("network_input_path", "")))))
        except Exception:
            continue
        panels.append([frames[-1, 0], afm])
        titles.append(str(row.get("sample_id", row.get("row_id", ""))))
    write_panel_grid(path, panels, ["RHEED final frame", "true AFM physical"], titles)


def prepare_rheed_ssl_dataset(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    data_dir = out_dir if out_dir.name == "data" else out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "cached_rheed_tensors"
    schema = read_json(resolve_repo_path(args.condition_schema))
    v3_rows = read_csv_rows(resolve_repo_path(args.condition_table))
    v3_by_height = {row.get("descriptor_height_path", ""): row for row in v3_rows if row.get("descriptor_height_path", "")}
    mvp2_pairs = read_csv_rows(resolve_repo_path(args.mvp2_paired_index) if args.mvp2_paired_index else _default_mvp2_pairs(args.mvp2_root))
    paired_by_sample = {sample_key(row.get("sample_id", row.get("rheed_video_path", ""))): row for row in mvp2_pairs}
    records = _discover_rheed_records(args.manifest, args.rheed_root)
    if args.limit is not None:
        records = records[: int(args.limit)]
    video_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            tensor, cache_path, meta = load_or_cache_rheed_tensor(
                resolve_data_path(record["rheed_video_path"]),
                cache_dir=cache_dir,
                frames=int(args.frames),
                image_size=int(args.image_size),
                final_fraction=float(args.final_fraction),
                sampling=str(args.sampling),
            )
        except Exception as exc:
            failures.append({**record, "error_message": str(exc)})
            if args.strict:
                raise
            continue
        paired = record["sample_key"] in paired_by_sample
        split = paired_by_sample[record["sample_key"]].get("split", "unpaired") if paired else "unpaired"
        row = {
            **record,
            "split": split,
            "is_paired": int(paired),
            "cached_tensor_path": display_path(cache_path),
            "source_frame_count": int(meta.get("source_frame_count", tensor.shape[0])),
            "frames_used": int(args.frames),
            "image_size": int(args.image_size),
            "final_fraction": float(args.final_fraction),
            "normalization": meta.get("normalization", ""),
        }
        video_rows.append(row)
        for frame_index in range(tensor.shape[0]):
            frame_rows.append(
                {
                    "frame_id": f"{row['video_id']}_f{frame_index:03d}",
                    "video_id": row["video_id"],
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "split": split,
                    "is_paired": int(paired),
                    "cached_tensor_path": row["cached_tensor_path"],
                    "frame_index": frame_index,
                }
            )
    video_by_sample = {row["sample_key"]: row for row in video_rows}
    paired_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for source in mvp2_pairs:
        key = sample_key(source.get("sample_id", ""))
        video_row = video_by_sample.get(key)
        if video_row is None:
            continue
        row = _schema_row_from_mvp2(source, schema, v3_by_height)
        row["rheed_video_path"] = video_row["rheed_video_path"]
        row["cached_tensor_path"] = video_row["cached_tensor_path"]
        row["source_frame_count"] = video_row["source_frame_count"]
        row["frames_used"] = video_row["frames_used"]
        row["image_size"] = video_row["image_size"]
        row["final_fraction"] = video_row["final_fraction"]
        row["normalization"] = video_row["normalization"]
        paired_rows.append(row)
        tensor = np.asarray(np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"], dtype=np.float32)
        feature_rows.append({"pair_id": row["pair_id"], "row_id": row["row_id"], "sample_id": row["sample_id"], "group_id": row["group_id"], "split": row["split"], **compute_rheed_features(tensor)})
    train_mask = np.asarray([row.get("split") == "train" for row in feature_rows], dtype=bool)
    feature_columns = [key for key in feature_rows[0] if key not in {"pair_id", "row_id", "sample_id", "group_id", "split"}] if feature_rows else []
    if feature_rows:
        feature_rows, impute_counts, means, stds = impute_feature_rows(feature_rows, feature_columns, train_mask)
    else:
        impute_counts, means, stds = {}, {}, {}
    metadata_rows = [
        {
            "video_id": row["video_id"],
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "is_paired": row["is_paired"],
            "source_frame_count": row["source_frame_count"],
            "fps": row.get("fps", ""),
            "frames_used": row["frames_used"],
            "image_size": row["image_size"],
            "final_fraction": row["final_fraction"],
        }
        for row in video_rows
    ]
    write_csv_rows(data_dir / "rheed_ssl_video_index.csv", video_rows)
    write_csv_rows(data_dir / "rheed_ssl_frame_index.csv", frame_rows)
    write_csv_rows(data_dir / "rheed_supervised_pair_index.csv", paired_rows)
    write_csv_rows(data_dir / "rheed_metadata_table.csv", metadata_rows)
    write_csv_rows(data_dir / "video_read_failures.csv", failures)
    write_csv_rows(data_dir / "rheed_handcrafted_features.csv", feature_rows)
    split_rows = []
    for split in ("train", "val", "test", "unpaired"):
        split_rows.append(
            {
                "split": split,
                "sample_count": sum(1 for row in video_rows if row["split"] == split),
                "paired_count": sum(1 for row in paired_rows if row["split"] == split),
                "group_count": len({str(row["group_id"]) for row in video_rows if row["split"] == split}),
            }
        )
    write_csv_rows(data_dir / "split_summary.csv", split_rows)
    enriched_schema = dict(schema)
    enriched_schema.update(
        {
            "rheed_feature_columns": feature_columns,
            "rheed_feature_imputation_counts": impute_counts,
            "rheed_feature_train_mean": means,
            "rheed_feature_train_std": stds,
            "metadata_columns": ["source_frame_count", "frames_used", "image_size", "final_fraction"],
            "source_condition_schema": display_path(resolve_repo_path(args.condition_schema)),
        }
    )
    write_json(data_dir / "condition_schema_v3_mvp6.json", enriched_schema)
    _write_rheed_preview(data_dir / "rheed_preview_grid.png", video_rows)
    _write_pair_preview(data_dir / "paired_rheed_afm_condition_preview_grid.png", paired_rows)
    inventory = {
        "rheed_records_found": len(records),
        "cached_video_count": len(video_rows),
        "cached_frame_count": len(frame_rows),
        "paired_rheed_videos": len(paired_rows),
        "unpaired_rheed_videos": sum(1 for row in video_rows if not int(row["is_paired"])),
        "video_read_failures": len(failures),
        "mvp2_split_preserved": True,
        "frames": int(args.frames),
        "final_fraction": float(args.final_fraction),
        "image_size": int(args.image_size),
        "normalization": "percentile_1_99_to_0_1",
        "feature_columns": feature_columns,
        "metadata_columns": enriched_schema["metadata_columns"],
        "condition_schema": display_path(data_dir / "condition_schema_v3_mvp6.json"),
        "split_summary": split_rows,
    }
    write_json(data_dir / "rheed_ssl_inventory.json", inventory)
    return inventory


def main() -> None:
    args = build_parser().parse_args()
    inventory = prepare_rheed_ssl_dataset(args)
    print(f"Wrote MVP-6 RHEED SSL data to {display_path(resolve_repo_path(args.out))}")
    print(f"paired={inventory['paired_rheed_videos']} unpaired={inventory['unpaired_rheed_videos']} failures={inventory['video_read_failures']}")


if __name__ == "__main__":
    main()
