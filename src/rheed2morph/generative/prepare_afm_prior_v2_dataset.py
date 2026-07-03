"""Prepare a broad AFM-only dataset for prior v2 training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.afm_prior_v2_utils import (
    V2_DESCRIPTOR_NAMES,
    bool_arg,
    compute_afm_descriptors_v2,
    crop_array,
    discover_afm_candidates,
    format_float,
    matches_scan,
    scan_target_um,
    split_groups,
    stable_id,
    standardize_descriptor_rows,
    write_descriptor_plots,
)
from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_json,
    resolve_repo_path,
    robust_normalize_to_unit,
    set_seed,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare broad AFM prior v2 dataset.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--afm-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--include-unpaired-afm", type=bool_arg, default=True)
    parser.add_argument("--scan-size-filter", type=str, default="1um")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--group-split", type=bool_arg, default=True)
    parser.add_argument("--patch-mode", choices=["none", "random", "deterministic"], default="none")
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--min-files-required", type=int, default=60)
    parser.add_argument("--strict", type=bool_arg, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"physical_height": 0, "network_input": 0, "png_fallback": 0}
    parent_seen: set[str] = set()
    for row in rows:
        if row.get("parent_row_id") in parent_seen:
            continue
        parent_seen.add(row.get("parent_row_id", row["row_id"]))
        kind = row.get("source_kind", "")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _row_counts(rows: list[dict[str, Any]], key: str = "split") -> dict[str, int]:
    return {split: sum(1 for row in rows if row.get(key) == split) for split in ("train", "val", "test")}


def _unique_counts(rows: list[dict[str, Any]], field: str, split: str | None = None) -> int:
    return len({row[field] for row in rows if split is None or row.get("split") == split})


def _array_patch_specs(split: str, patch_mode: str, patches_per_image: int) -> list[tuple[str, str, int]]:
    if patch_mode == "none":
        return [("full", "none", 0)]
    if split == "train":
        return [(f"patch_{index:03d}", patch_mode, index) for index in range(max(1, patches_per_image))]
    return [("center", "center", 0)]


def _prepare_rows(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates, discovery = discover_afm_candidates(args.afm_root, args.manifest, bool(args.include_unpaired_afm), args.scan_size_filter)
    if args.limit is not None:
        candidates = candidates[: int(args.limit)]
    groups = [candidate.group_id for candidate in candidates]
    split_for_group = split_groups(groups, int(args.seed), by_group=bool(args.group_split))
    tensor_dir = out_dir / "standardized_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    data_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source_index, candidate in enumerate(candidates, start=1):
        split = split_for_group[candidate.group_id]
        parent_row_id = stable_id(candidate.path.as_posix(), candidate.sample_id, candidate.afm_file_id, prefix="afmv2")
        try:
            source_array = load_height_array(candidate.path)
            descriptor_array = source_array
        except Exception as exc:
            failures.append({"path": display_path(candidate.path), "reason": f"load_failed:{exc}"})
            continue
        patch_specs = _array_patch_specs(split, str(args.patch_mode), int(args.patches_per_image))
        for patch_id, crop_mode, patch_index in patch_specs:
            patch_seed = int(hash(parent_row_id) % 1_000_000) + int(args.seed)
            if crop_mode == "none":
                patch_source = source_array
                patch_descriptor = descriptor_array
            else:
                patch_source = crop_array(source_array, int(args.patch_size), crop_mode, patch_index, int(args.patches_per_image), patch_seed)
                patch_descriptor = crop_array(descriptor_array, int(args.patch_size), crop_mode, patch_index, int(args.patches_per_image), patch_seed)
            row_id = parent_row_id if patch_id == "full" else f"{parent_row_id}_{patch_id}"
            tensor = robust_normalize_to_unit(patch_source, int(args.image_size))
            tensor_path = tensor_dir / f"{row_id}.npy"
            np.save(tensor_path, tensor.astype(np.float32))
            descriptors = compute_afm_descriptors_v2(patch_descriptor)
            scan_text = "" if candidate.scan_size_um is None else format_float(float(candidate.scan_size_um))
            row = {
                "row_id": row_id,
                "parent_row_id": parent_row_id,
                "sample_id": candidate.sample_id,
                "group_id": candidate.group_id,
                "split": split,
                "network_input_path": display_path(tensor_path),
                "descriptor_height_path": display_path(candidate.path),
                "source_path": display_path(candidate.path),
                "source_kind": candidate.source_kind,
                "source_priority": str(source_index),
                "scan_size_um": scan_text,
                "afm_file_id": candidate.afm_file_id,
                "metadata_path": candidate.metadata_path,
                "metadata_source": candidate.metadata_source,
                "height_shape": "x".join(map(str, source_array.shape)),
                "patch_id": patch_id,
                "is_patch": str(patch_id != "full").lower(),
                "metrics_scope": "patch" if patch_id != "full" else "full_image",
            }
            data_rows.append(row)
            descriptor_row = {
                "row_id": row_id,
                "parent_row_id": parent_row_id,
                "sample_id": candidate.sample_id,
                "group_id": candidate.group_id,
                "split": split,
                "source_kind": candidate.source_kind,
                "patch_id": patch_id,
                "metrics_scope": row["metrics_scope"],
            }
            descriptor_row.update({name: format_float(value) for name, value in descriptors.items()})
            descriptor_rows.append(descriptor_row)
    discovery.update(
        {
            "load_failure_count": len(failures),
            "load_failure_examples": failures[:20],
            "source_file_count_after_limit": len(candidates),
            "patch_mode": args.patch_mode,
            "patches_per_image": int(args.patches_per_image),
            "patch_size": int(args.patch_size),
        }
    )
    return data_rows, descriptor_rows, discovery


def _write_prototypes(out_dir: Path, descriptor_rows: list[dict[str, Any]], scaler: dict[str, Any]) -> dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        write_csv_rows(out_dir / "morphology_prototypes_v2.csv", [])
        return {"prototype_status": f"skipped_sklearn_unavailable:{exc}", "prototype_k": 0, "cluster_counts": {}}
    descriptor_columns = list(scaler["descriptor_columns"])
    matrix = np.asarray([[float(row[name]) for name in descriptor_columns] for row in descriptor_rows], dtype=np.float32)
    means = np.asarray([scaler["train_mean"][name] for name in descriptor_columns], dtype=np.float32)
    stds = np.asarray([scaler["train_std"][name] for name in descriptor_columns], dtype=np.float32)
    standardized = (matrix - means[None]) / stds[None]
    train_mask = np.asarray([row.get("split") == "train" for row in descriptor_rows], dtype=bool)
    train = standardized[train_mask] if np.any(train_mask) else standardized
    candidates = [k for k in (4, 6, 8) if train.shape[0] >= k + 2 and len({row["group_id"] for row in descriptor_rows if row.get("split") == "train"}) >= 2]
    if not candidates:
        rows = [
            {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "prototype_id": "",
            }
            for row in descriptor_rows
        ]
        write_csv_rows(out_dir / "morphology_prototypes_v2.csv", rows)
        return {"prototype_status": "skipped_small_dataset", "prototype_k": 0, "cluster_counts": {}}
    labels_by_k: dict[int, np.ndarray] = {}
    silhouettes: dict[str, float] = {}
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_labels = model.fit_predict(train)
        labels = model.predict(standardized)
        labels_by_k[k] = labels
        silhouettes[str(k)] = float(silhouette_score(train, train_labels)) if len(set(train_labels.tolist())) > 1 else float("nan")
    finite_scores = {int(k): v for k, v in silhouettes.items() if math.isfinite(float(v))}
    default_k = max(finite_scores, key=finite_scores.get) if finite_scores else min(candidates, key=lambda k: abs(k - 6))
    rows: list[dict[str, Any]] = []
    default_labels = labels_by_k[default_k]
    for index, row in enumerate(descriptor_rows):
        out = {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "group_id": row["group_id"],
            "split": row["split"],
            "prototype_id": int(default_labels[index]),
        }
        for k, labels in labels_by_k.items():
            out[f"k{k}_prototype_id"] = int(labels[index])
        rows.append(out)
    write_csv_rows(out_dir / "morphology_prototypes_v2.csv", rows)
    counts = {str(label): int(np.sum(default_labels == label)) for label in sorted(set(default_labels.tolist()))}
    return {
        "prototype_status": "ok",
        "prototype_k": int(default_k),
        "prototype_candidates": candidates,
        "prototype_silhouette_scores": silhouettes,
        "cluster_counts": counts,
    }


def _write_preview_grid(out_dir: Path, data_rows: list[dict[str, Any]], prototype_rows: list[dict[str, str]]) -> None:
    if not data_rows:
        return
    proto_by_row = {row["row_id"]: row.get("prototype_id", "") for row in prototype_rows}
    selected: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        selected.extend([row for row in data_rows if row.get("split") == split][:3])
    selected = selected[:9]
    grid_rows: list[list[np.ndarray]] = []
    titles: list[str] = []
    for row in selected:
        image = load_height_array(resolve_repo_path(Path(row["network_input_path"])))
        grid_rows.append([image])
        titles.append(f"{row['split']} s{row['sample_id']} p{proto_by_row.get(row['row_id'], '')}")
    write_panel_grid(out_dir / "afm_preview_grid.png", grid_rows, ["AFM height"], titles)
    write_panel_grid(out_dir / "afm_prior_v2_pair_grid.png", grid_rows, ["AFM height"], titles)
    if prototype_rows:
        proto_grid: list[list[np.ndarray]] = []
        proto_titles: list[str] = []
        by_proto: dict[str, dict[str, Any]] = {}
        by_row = {row["row_id"]: row for row in data_rows}
        for proto_row in prototype_rows:
            proto = proto_row.get("prototype_id", "")
            if proto and proto not in by_proto and proto_row["row_id"] in by_row:
                by_proto[proto] = by_row[proto_row["row_id"]]
        for proto, row in sorted(by_proto.items(), key=lambda item: item[0]):
            proto_grid.append([load_height_array(resolve_repo_path(Path(row["network_input_path"])))])
            proto_titles.append(f"prototype {proto}")
        write_panel_grid(out_dir / "prototype_examples_grid.png", proto_grid, ["example"], proto_titles)


def _write_failure_report(out_dir: Path, inventory: dict[str, Any], min_files_required: int) -> None:
    text = [
        "# AFM Prior V2 Data Discovery Failure",
        "",
        f"Required at least `{min_files_required}` source AFM files for the requested scan filter.",
        f"Found `{inventory['source_file_count']}` source AFM files.",
        "",
        "## Search Roots",
        "",
        *[f"- `{root}`" for root in inventory.get("search_roots", [])],
        "",
        "## Searched Patterns",
        "",
        *[f"- `{pattern}`" for pattern in inventory.get("searched_patterns", [])],
        "",
        "## Candidate Counts",
        "",
        f"- Path candidates before filter: `{inventory.get('candidate_path_count_before_filter', 0)}`",
        f"- Candidates after scan filter: `{inventory.get('candidate_count_after_scan_filter', 0)}`",
        f"- Deduplicated candidates: `{inventory.get('deduplicated_candidate_count', 0)}`",
    ]
    (out_dir / "data_discovery_failure.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_discovery_report(out_dir: Path, inventory: dict[str, Any]) -> None:
    scan_counts = inventory.get("scan_size_counts_raw_candidates", {})
    rows = [
        "# AFM Prior V2 Data Discovery Report",
        "",
        f"- Scan filter: `{inventory['scan_size_filter']}`",
        f"- Source AFM files indexed: `{inventory['source_file_count']}`",
        f"- Indexed training rows after patch policy: `{inventory['row_count']}`",
        f"- Groups: `{inventory['group_count']}`",
        f"- Physical height maps: `{inventory['source_counts'].get('physical_height', 0)}`",
        f"- Network inputs: `{inventory['source_counts'].get('network_input', 0)}`",
        f"- PNG fallback rows: `{inventory['source_counts'].get('png_fallback', 0)}`",
        f"- Increase versus MVP-1 36-file run: `{inventory['source_file_count'] - 36}`",
        f"- Patch mode: `{inventory['patch_mode']}`",
        "",
        "## Split Counts",
        "",
        f"- Files by split: `{inventory['split_source_file_counts']}`",
        f"- Rows by split: `{inventory['split_row_counts']}`",
        f"- Groups by split: `{inventory['split_group_counts']}`",
        "",
        "## Scan Sizes Seen In Raw Candidates",
        "",
    ]
    for key, value in sorted(scan_counts.items()):
        rows.append(f"- `{key}` um: `{value}`")
    rows.extend(
        [
            "",
            "## Path Conventions Observed",
            "",
            "- Physical processed height maps: `data/processed_afm/<sample_id>/<afm_file_id>/<afm_file_id>_height.npy`",
            "- Plane-corrected physical height maps: `data/plane_corrected_afm/<sample_id>/<afm_file_id>/<afm_file_id>_plane_corrected.npy`",
            "- Prepared model input tensors: `reports/afm_prior_v2/<timestamp>/data/standardized_tensors/<row_id>.npy`",
            "- No `network_input.npy` files were required when physical height maps were available.",
        ]
    )
    (out_dir / "data_discovery_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_rows, descriptor_rows, discovery = _prepare_rows(args, out_dir)
    if not data_rows:
        raise RuntimeError("No AFM prior v2 rows were prepared. Check --afm-root, --manifest, and --scan-size-filter.")
    descriptor_rows, scaler = standardize_descriptor_rows(descriptor_rows, V2_DESCRIPTOR_NAMES)
    prototype_info = _write_prototypes(out_dir, descriptor_rows, scaler)
    prototype_rows = []
    proto_path = out_dir / "morphology_prototypes_v2.csv"
    if proto_path.is_file():
        from rheed2morph.generative.common import read_csv_rows

        prototype_rows = read_csv_rows(proto_path)
    proto_by_row = {row["row_id"]: row.get("prototype_id", "") for row in prototype_rows}
    for row in data_rows:
        row["prototype_id"] = proto_by_row.get(row["row_id"], "")
    write_csv_rows(out_dir / "afm_prior_v2_index.csv", data_rows)
    split_rows: list[dict[str, Any]] = []
    for group in sorted({row["group_id"] for row in data_rows}):
        group_rows = [row for row in data_rows if row["group_id"] == group]
        split_rows.append(
            {
                "group_id": group,
                "split": group_rows[0]["split"],
                "row_count": len(group_rows),
                "source_file_count": len({row["parent_row_id"] for row in group_rows}),
            }
        )
    write_csv_rows(out_dir / "afm_prior_v2_splits.csv", split_rows)
    write_csv_rows(out_dir / "afm_prior_v2_descriptors.csv", descriptor_rows)
    write_json(out_dir / "descriptor_scaler_v2.json", scaler)
    source_files = {row["parent_row_id"] for row in data_rows}
    source_file_count = len(source_files)
    source_split_counts = {
        split: len({row["parent_row_id"] for row in data_rows if row["split"] == split}) for split in ("train", "val", "test")
    }
    split_group_counts = {split: _unique_counts(data_rows, "group_id", split) for split in ("train", "val", "test")}
    scan_values = [float(row["scan_size_um"]) for row in data_rows if row.get("scan_size_um", "") not in {"", "nan"}]
    target = scan_target_um(args.scan_size_filter)
    one_um_source_count = len(
        {
            row["parent_row_id"]
            for row in data_rows
            if matches_scan(float(row["scan_size_um"]) if row.get("scan_size_um", "") not in {"", "nan"} else None, 1.0)
        }
    )
    inventory: dict[str, Any] = {
        **discovery,
        **prototype_info,
        "scan_size_filter": args.scan_size_filter,
        "image_size": int(args.image_size),
        "source_file_count": source_file_count,
        "one_um_source_file_count": one_um_source_count,
        "row_count": len(data_rows),
        "descriptor_row_count": len(descriptor_rows),
        "group_count": len({row["group_id"] for row in data_rows}),
        "split_source_file_counts": source_split_counts,
        "split_row_counts": _row_counts(data_rows),
        "split_group_counts": split_group_counts,
        "source_counts": _source_counts(data_rows),
        "descriptor_columns": V2_DESCRIPTOR_NAMES,
        "descriptor_imputation_counts": scaler["nan_imputation_counts"],
        "scan_size_values_indexed": sorted({format_float(value) for value in scan_values}),
        "mvp1_afm_file_count": 36,
        "increase_over_mvp1": source_file_count - 36,
        "group_id_rule": "metadata group_id when available, otherwise sample_id inferred from path",
        "metrics_scope": "patch rows for train only when patching is enabled; validation/test rows use full image or center crop",
        "strict": bool(args.strict),
        "min_files_required": int(args.min_files_required),
        "target_scan_um": target,
    }
    write_json(out_dir / "afm_prior_v2_inventory.json", inventory)
    write_descriptor_plots(out_dir, descriptor_rows, V2_DESCRIPTOR_NAMES)
    _write_preview_grid(out_dir, data_rows, prototype_rows)
    _write_discovery_report(out_dir, inventory)
    if source_file_count < int(args.min_files_required):
        _write_failure_report(out_dir, inventory, int(args.min_files_required))
        if bool(args.strict):
            raise RuntimeError(
                f"Only found {source_file_count} AFM files, below --min-files-required={args.min_files_required}. "
                f"See {out_dir / 'data_discovery_failure.md'}."
            )
    return inventory


def main() -> None:
    args = build_parser().parse_args()
    inventory = prepare_dataset(args)
    print(f"Wrote AFM prior v2 dataset to {display_path(resolve_repo_path(args.out))}")
    print(
        "source_files={source_file_count} rows={row_count} groups={group_count} increase_over_mvp1={increase_over_mvp1}".format(
            **inventory
        )
    )


if __name__ == "__main__":
    main()
