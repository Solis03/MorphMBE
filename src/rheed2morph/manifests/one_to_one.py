#!/usr/bin/env python3
"""Build one-to-one RHEED-to-AFM manifests for clean baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIR_ROOT = REPO_ROOT / "data" / "pair"
DEFAULT_DESCRIPTOR_AUX_CSV = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "manifests"
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
MATERIAL_RE = re.compile(r"\b([A-Z][a-z]Sb)\b")
N_TAG_RE = re.compile(r"n\d{4}", re.IGNORECASE)


@dataclass(frozen=True)
class CandidateRecord:
    sample_id: str
    group_id: str
    material: str
    rheed_path: Path
    afm_path: Path
    source_afm_path: Path | None
    afm_scan_size_um: float | None
    scan_size_source: str = "missing"
    resolution_h: int | None = None
    resolution_w: int | None = None
    channel_name: str = "unknown"
    is_plane_corrected: bool = False
    is_rendered_image: bool = False
    is_physical_height_map: bool = False


@dataclass(frozen=True)
class ManifestSelection:
    sample_id: str
    group_id: str
    material: str
    rheed_path: Path
    afm_path: Path
    afm_scan_size_um: float | None
    selection_reason: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def canonical_size_label(size_um: float) -> str:
    if math.isclose(size_um, 0.5, rel_tol=1e-9, abs_tol=1e-9):
        return "0p5um"
    if math.isclose(size_um, round(size_um), rel_tol=1e-9, abs_tol=1e-9):
        return f"{int(round(size_um))}um"
    text = f"{size_um:.3f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}um"


def scan_size_bucket(size_um: float | None) -> str:
    if size_um is None or not np.isfinite(size_um):
        return "nan"
    return f"{float(size_um):.3f}"


def normalize_scan_size_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"(?<=\d)p(?=\d)", ".", lowered)
    lowered = lowered.replace("-", " ").replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def size_to_um(value: float, unit: str | None) -> float | None:
    if unit is None:
        if 0.05 <= value <= 20.0:
            return float(value)
        if 50.0 <= value <= 20_000.0:
            return float(value) / 1000.0
        return None
    if unit == "um":
        return float(value)
    if unit == "nm":
        return float(value) / 1000.0
    return None


def numeric_size_to_um(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    if 0.01 <= value <= 20.0:
        return float(value)
    if 50.0 <= value <= 20_000.0:
        return float(value) / 1000.0
    return float(value) if value > 0 else None


def metadata_path_candidates(path: Path) -> list[Path]:
    candidates: list[Path] = []
    if path.suffix.lower() == ".npy":
        if path.name.endswith("_plane_corrected.npy"):
            base = path.name.removesuffix("_plane_corrected.npy")
            candidates.append(path.with_name(f"{base}_plane_corrected_metadata.json"))
            candidates.append(path.with_name(f"{base}_metadata.json"))
        else:
            candidates.append(path.with_name(f"{path.stem}_metadata.json"))
    elif path.suffix.lower() == ".png":
        candidates.append(path.with_name(f"{path.stem.removesuffix('_network_input')}_metadata.json"))
        candidates.append(path.with_name(f"{path.stem}_metadata.json"))
    return candidates


def metadata_scan_size_um(path: Path) -> float | None:
    for metadata_path in metadata_path_candidates(path):
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("scan_size_um",):
            value = payload.get(key)
            if isinstance(value, list | tuple) and len(value) >= 2:
                x_val = numeric_size_to_um(float(value[0]))
                y_val = numeric_size_to_um(float(value[1]))
                if x_val is not None and y_val is not None:
                    return float((x_val + y_val) / 2.0)
            if isinstance(value, int | float):
                parsed = numeric_size_to_um(float(value))
                if parsed is not None:
                    return parsed
        channels = payload.get("channels")
        if isinstance(channels, dict):
            for channel_payload in channels.values():
                if not isinstance(channel_payload, dict):
                    continue
                value = channel_payload.get("scan_size_um")
                if isinstance(value, list | tuple) and len(value) >= 2:
                    x_val = numeric_size_to_um(float(value[0]))
                    y_val = numeric_size_to_um(float(value[1]))
                    if x_val is not None and y_val is not None:
                        return float((x_val + y_val) / 2.0)
    return None


def parse_afm_scan_size_um(path_text: str) -> float | None:
    normalized = normalize_scan_size_text(Path(path_text).stem)
    patterns = [
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>um|nm)\s*x\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>um|nm)?"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*x\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<u>um|nm)"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*x\s*(?P<b>\d+(?:\.\d+)?)"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<u>um|nm)"),
    ]
    for pattern in patterns:
        match = pattern.search(normalized)
        if match is None:
            continue
        groups = match.groupdict()
        if "b" in groups and groups.get("b") is not None:
            first = size_to_um(float(groups["a"]), groups.get("ua") or groups.get("u"))
            second = size_to_um(float(groups["b"]), groups.get("ub") or groups.get("u") or groups.get("ua"))
            if first is None or second is None:
                continue
            return float((first + second) / 2.0)
        single = size_to_um(float(groups["a"]), groups.get("u"))
        if single is not None:
            return single
    return None


def infer_afm_scan_size_um(primary_path: Path, fallback_paths: Sequence[Path] = ()) -> tuple[float | None, str]:
    ordered_paths = [primary_path, *fallback_paths]
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in ordered_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)

    for path in unique_paths:
        from_metadata = metadata_scan_size_um(path)
        if from_metadata is not None:
            return from_metadata, f"metadata:{display_path(path)}"

    for path in unique_paths:
        from_filename = parse_afm_scan_size_um(path.name)
        if from_filename is not None:
            return from_filename, f"filename:{path.name}"
    return None, "missing"


def parse_bool_text(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_optional_float_text(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if np.isfinite(parsed) else None


def infer_material(sample_id: str, afm_path: Path, rheed_path: Path | None = None) -> str:
    candidates = [afm_path.stem]
    if rheed_path is not None:
        candidates.append(rheed_path.stem)
    for text in candidates:
        match = MATERIAL_RE.search(text)
        if match is not None:
            return match.group(1)
    stripped = N_TAG_RE.sub("", afm_path.stem).strip(" _-")
    if stripped:
        token = stripped.split("_")[0].split()[0]
        if any(char.isalpha() for char in token):
            return token
    return "unknown"


def require_video_backend() -> Any:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing video dependency `imageio-ffmpeg`; install it before building one-to-one manifests."
        ) from exc
    return imageio_ffmpeg


def visible_video_files(sample_root: Path) -> list[Path]:
    rheed_root = sample_root / "RHEED"
    if not rheed_root.is_dir():
        return []
    return [
        path
        for path in sorted(rheed_root.iterdir())
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VIDEO_SUFFIXES
    ]


def probe_video(path: Path) -> tuple[int, float] | None:
    imageio_ffmpeg = require_video_backend()
    try:
        frame_count, duration_seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    except Exception:
        return None
    if frame_count <= 0 or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return None
    return int(frame_count), float(duration_seconds)


def choose_canonical_rheed_path(sample_root: Path) -> tuple[Path | None, str]:
    candidates = []
    for path in visible_video_files(sample_root):
        probed = probe_video(path)
        if probed is None:
            continue
        frame_count, duration_seconds = probed
        candidates.append(
            (
                1 if "main" in path.stem.lower() else 0,
                duration_seconds,
                frame_count,
                path.name.lower(),
                path.resolve(),
            )
        )
    if not candidates:
        return None, "no decodable video"
    _, duration_seconds, _, _, selected = sorted(candidates, reverse=True)[0]
    if "main" in selected.stem.lower():
        return selected, "contains_main"
    return selected, f"longest_decodable:{duration_seconds:.3f}s"


def load_candidate_records(
    manifest_path: Path | None,
    pair_root: Path,
    descriptor_aux_csv: Path,
) -> tuple[list[CandidateRecord], list[dict[str, Any]], dict[str, Any]]:
    if manifest_path is not None:
        return load_candidate_records_from_manifest(manifest_path)
    return load_candidate_records_from_descriptors(pair_root, descriptor_aux_csv)


def load_candidate_records_from_manifest(
    manifest_path: Path,
) -> tuple[list[CandidateRecord], list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(manifest_path)
    candidates: list[CandidateRecord] = []
    warnings: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            warnings.append({"sample_id": "", "afm_path": row.get("afm_path", ""), "reason": "missing sample_id"})
            continue
        group_id = row.get("group_id", "").strip() or sample_id
        rheed_path = resolve_existing_path(Path(row.get("rheed_path", "")))
        afm_path = resolve_existing_path(Path(row.get("afm_path", "")))
        parsed_size = parse_optional_float_text(row.get("scan_size_um"))
        size_source = row.get("scan_size_source", "").strip()
        if parsed_size is None:
            parsed_size, inferred_source = infer_afm_scan_size_um(afm_path)
            size_source = size_source or inferred_source
        elif not size_source:
            size_source = "input_manifest:scan_size_um"
        if parsed_size is None:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "afm_path": display_path(afm_path),
                    "reason": "scan size not found in metadata or filename",
                }
            )
        material = row.get("material", "").strip() or infer_material(sample_id, afm_path, rheed_path)
        candidates.append(
            CandidateRecord(
                sample_id=sample_id,
                group_id=group_id,
                material=material,
                rheed_path=rheed_path,
                afm_path=afm_path,
                source_afm_path=None,
                afm_scan_size_um=parsed_size,
                scan_size_source=size_source,
                resolution_h=(int(row["resolution_h"]) if row.get("resolution_h", "").strip() else None),
                resolution_w=(int(row["resolution_w"]) if row.get("resolution_w", "").strip() else None),
                channel_name=row.get("channel_name", "").strip() or "unknown",
                is_plane_corrected=parse_bool_text(row.get("is_plane_corrected")),
                is_rendered_image=parse_bool_text(row.get("is_rendered_image")),
                is_physical_height_map=parse_bool_text(row.get("is_physical_height_map")),
            )
        )
    summary = {
        "source": display_path(manifest_path),
        "mode": "input_manifest",
        "candidate_count": len(candidates),
        "warning_count": len(warnings),
    }
    return candidates, warnings, summary


def load_candidate_records_from_descriptors(
    pair_root: Path,
    descriptor_aux_csv: Path,
) -> tuple[list[CandidateRecord], list[dict[str, Any]], dict[str, Any]]:
    aux_rows = read_csv(descriptor_aux_csv)
    rheed_by_sample: dict[str, tuple[Path, str]] = {}
    warnings: list[dict[str, Any]] = []
    for sample_root in sorted(path for path in pair_root.iterdir() if path.is_dir()):
        rheed_path, reason = choose_canonical_rheed_path(sample_root)
        if rheed_path is None:
            warnings.append(
                {
                    "sample_id": sample_root.name,
                    "afm_path": "",
                    "reason": f"missing canonical rheed video: {reason}",
                }
            )
            continue
        rheed_by_sample[sample_root.name] = (rheed_path, reason)

    candidates: list[CandidateRecord] = []
    for row in aux_rows:
        sample_id = row["sample_id"].strip()
        rheed_info = rheed_by_sample.get(sample_id)
        if rheed_info is None:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "afm_path": row.get("network_input_path", ""),
                    "reason": "no canonical rheed for descriptor row",
                }
            )
            continue
        network_input_path = resolve_existing_path(Path(row["network_input_path"]))
        source_afm_path = resolve_existing_path(Path(row["afm_path"]))
        parsed_size, size_source = infer_afm_scan_size_um(
            source_afm_path,
            fallback_paths=[network_input_path],
        )
        if parsed_size is None:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "afm_path": display_path(network_input_path),
                    "reason": "scan size not found in metadata or filename",
                }
            )
        material = infer_material(sample_id, source_afm_path, rheed_info[0])
        candidates.append(
            CandidateRecord(
                sample_id=sample_id,
                group_id=sample_id,
                material=material,
                rheed_path=rheed_info[0],
                afm_path=network_input_path,
                source_afm_path=source_afm_path,
                afm_scan_size_um=parsed_size,
                scan_size_source=size_source,
                resolution_h=None,
                resolution_w=None,
                channel_name="ZSensor",
                is_plane_corrected=True,
                is_rendered_image=False,
                is_physical_height_map=True,
            )
        )
    summary = {
        "source": display_path(descriptor_aux_csv),
        "mode": "descriptor_aux_csv",
        "candidate_count": len(candidates),
        "warning_count": len(warnings),
        "canonical_rheed_sample_count": len(rheed_by_sample),
    }
    return candidates, warnings, summary


def size_matches(size_um: float | None, target_um: float, tolerance: float) -> bool:
    if size_um is None or not np.isfinite(size_um):
        return False
    lower = target_um * (1.0 - tolerance)
    upper = target_um * (1.0 + tolerance)
    return lower <= float(size_um) <= upper


def parse_resolution(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, mmap_mode="r")
            if array.ndim >= 2:
                return int(array.shape[-2]), int(array.shape[-1])
            return None
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                return int(image.height), int(image.width)
    except Exception:
        return None
    return None


def zsensor_priority(candidate: CandidateRecord) -> int:
    texts = [candidate.afm_path.stem.lower()]
    if candidate.source_afm_path is not None:
        texts.append(candidate.source_afm_path.stem.lower())
    text = " ".join(texts)
    return 1 if any(token in text for token in ("zsensor", "zsens", "z sensor", "zsensr")) else 0


def extension_priority(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return 2
    if suffix == ".png":
        return 1
    return 0


def candidate_sort_key(candidate: CandidateRecord) -> tuple[Any, ...]:
    resolution = (
        (candidate.resolution_h, candidate.resolution_w)
        if candidate.resolution_h is not None and candidate.resolution_w is not None
        else parse_resolution(candidate.afm_path)
    )
    resolution_rank = 0 if resolution is None else int(resolution[0] * resolution[1])
    return (
        1 if candidate.is_plane_corrected else 0,
        1 if candidate.is_physical_height_map else 0,
        0 if candidate.is_rendered_image else 1,
        zsensor_priority(candidate),
        extension_priority(candidate.afm_path),
        1 if candidate.afm_scan_size_um is not None else 0,
        resolution_rank,
        str(candidate.afm_path).lower(),
    )


def candidate_resolution_text(candidate: CandidateRecord) -> str:
    resolution = parse_resolution(candidate.afm_path)
    if resolution is None:
        return "unknown"
    return f"{resolution[0]}x{resolution[1]}"


def select_representative_candidate(
    candidates: Sequence[CandidateRecord],
    selection_context: str,
) -> ManifestSelection:
    if not candidates:
        raise ValueError("Cannot select representative from empty candidate list.")
    ordered = sorted(candidates, key=candidate_sort_key, reverse=True)
    selected = ordered[0]
    reason_parts = [
        selection_context,
        f"zsensor_pref={'yes' if zsensor_priority(selected) else 'no'}",
        f"suffix={selected.afm_path.suffix.lower() or 'none'}",
        f"valid_size={'yes' if selected.afm_scan_size_um is not None else 'no'}",
        f"resolution={candidate_resolution_text(selected)}",
    ]
    if len(ordered) > 1:
        reason_parts.append(f"tie_break=sorted_path:{selected.afm_path.name}")
    return ManifestSelection(
        sample_id=selected.sample_id,
        group_id=selected.group_id,
        material=selected.material,
        rheed_path=selected.rheed_path,
        afm_path=selected.afm_path,
        afm_scan_size_um=selected.afm_scan_size_um,
        selection_reason="; ".join(reason_parts),
    )


def build_target_manifest(
    grouped_candidates: dict[str, list[CandidateRecord]],
    target_um: float,
    tolerance: float,
) -> tuple[list[ManifestSelection], dict[str, Any]]:
    rows: list[ManifestSelection] = []
    matched_groups = 0
    dropped_groups = 0
    for group_id in sorted(grouped_candidates):
        matches = [candidate for candidate in grouped_candidates[group_id] if size_matches(candidate.afm_scan_size_um, target_um, tolerance)]
        if not matches:
            dropped_groups += 1
            continue
        matched_groups += 1
        rows.append(
            select_representative_candidate(
                matches,
                selection_context=f"target_size={target_um:.6f}um within_tol={tolerance:.3f}",
            )
        )
    summary = {
        "target_size_um": target_um,
        "selected_pair_count": len(rows),
        "matched_group_count": matched_groups,
        "dropped_group_count": dropped_groups,
    }
    return rows, summary


def most_common_size_um(candidates: Iterable[CandidateRecord]) -> float | None:
    counts = Counter(scan_size_bucket(candidate.afm_scan_size_um) for candidate in candidates if candidate.afm_scan_size_um is not None)
    counts.pop("nan", None)
    if not counts:
        return None
    return float(counts.most_common(1)[0][0])


def build_all_size_representative_manifest(
    grouped_candidates: dict[str, list[CandidateRecord]],
    tolerance: float,
    preferred_targets: Sequence[float],
    most_common_size: float | None,
) -> list[ManifestSelection]:
    rows: list[ManifestSelection] = []
    for group_id in sorted(grouped_candidates):
        candidates = grouped_candidates[group_id]
        selected: ManifestSelection | None = None
        for target_um in preferred_targets:
            matches = [candidate for candidate in candidates if size_matches(candidate.afm_scan_size_um, target_um, tolerance)]
            if matches:
                selected = select_representative_candidate(
                    matches,
                    selection_context=f"representative_priority_target={target_um:.6f}um",
                )
                break
        if selected is None and most_common_size is not None:
            matches = [candidate for candidate in candidates if size_matches(candidate.afm_scan_size_um, most_common_size, tolerance)]
            if matches:
                selected = select_representative_candidate(
                    matches,
                    selection_context=f"representative_priority_most_common={most_common_size:.6f}um",
                )
        if selected is None:
            selected = select_representative_candidate(
                candidates,
                selection_context="representative_fallback_best_available",
            )
        rows.append(selected)
    return rows


def validate_manifest_rows(rows: Sequence[ManifestSelection], manifest_name: str) -> dict[str, Any]:
    group_ids = [row.group_id for row in rows]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{manifest_name}: duplicate group_id values found.")
    pair_keys = [(row.group_id, str(row.rheed_path)) for row in rows]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError(f"{manifest_name}: duplicate (group_id, rheed_path) pairs found.")
    for row in rows:
        if not str(row.afm_path):
            raise ValueError(f"{manifest_name}: empty afm_path for group {row.group_id}")
        if not row.afm_path.exists():
            raise ValueError(f"{manifest_name}: AFM path does not exist: {row.afm_path}")
        if not row.rheed_path.exists():
            raise ValueError(f"{manifest_name}: RHEED path does not exist: {row.rheed_path}")
    return {
        "manifest_name": manifest_name,
        "row_count": len(rows),
        "group_count": len(set(group_ids)),
        "material_count": len({row.material for row in rows}),
    }


def manifest_rows_to_dicts(rows: Sequence[ManifestSelection]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "sample_id": row.sample_id,
                "group_id": row.group_id,
                "material": row.material,
                "rheed_path": display_path(row.rheed_path),
                "afm_path": display_path(row.afm_path),
                "afm_scan_size_um": "" if row.afm_scan_size_um is None else f"{float(row.afm_scan_size_um):.6f}",
                "selection_reason": row.selection_reason,
            }
        )
    return out


def write_manifest(path: Path, rows: Sequence[ManifestSelection]) -> None:
    write_csv(
        path,
        manifest_rows_to_dicts(rows),
        ["sample_id", "group_id", "material", "rheed_path", "afm_path", "afm_scan_size_um", "selection_reason"],
    )


def build_size_summary_rows(
    grouped_candidates: dict[str, list[CandidateRecord]],
    target_summaries: dict[str, dict[str, Any]],
    all_size_rows: Sequence[ManifestSelection],
) -> list[dict[str, Any]]:
    size_counts = Counter()
    size_groups: dict[str, set[str]] = defaultdict(set)
    for group_id, candidates in grouped_candidates.items():
        for candidate in candidates:
            bucket = scan_size_bucket(candidate.afm_scan_size_um)
            size_counts[bucket] += 1
            size_groups[bucket].add(group_id)

    rows: list[dict[str, Any]] = []
    for bucket in sorted(size_counts):
        row: dict[str, Any] = {
            "scan_size_um": bucket,
            "afm_file_count": size_counts[bucket],
            "unique_group_count": len(size_groups[bucket]),
            "selected_pairs_all_size_representative": sum(
                1 for selection in all_size_rows if scan_size_bucket(selection.afm_scan_size_um) == bucket
            ),
        }
        for label, summary in target_summaries.items():
            row[f"selected_pairs_{label}"] = summary["selected_pair_count"]
            row[f"dropped_groups_{label}"] = summary["dropped_group_count"]
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean one-to-one RHEED-to-AFM manifests by scan size."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional input candidate manifest.")
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--descriptor-aux-csv", type=Path, default=DEFAULT_DESCRIPTOR_AUX_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-sizes", type=float, nargs="+", default=[1.0, 0.5, 5.0])
    parser.add_argument("--size-tolerance", type=float, default=0.10)
    return parser


def build_one_to_one_manifests(
    manifest_path: Path | None,
    pair_root: Path,
    descriptor_aux_csv: Path,
    out_dir: Path,
    target_sizes: Sequence[float],
    size_tolerance: float,
) -> dict[str, Any]:
    candidates, warnings, source_summary = load_candidate_records(manifest_path, pair_root, descriptor_aux_csv)
    grouped_candidates: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        grouped_candidates[candidate.group_id].append(candidate)

    target_summaries: dict[str, dict[str, Any]] = {}
    validation_rows: list[dict[str, Any]] = []
    manifest_outputs: dict[str, str] = {}
    for target_size in target_sizes:
        label = canonical_size_label(target_size)
        rows, summary = build_target_manifest(grouped_candidates, target_size, size_tolerance)
        manifest_path_out = out_dir / f"manifest_{label}_one_to_one.csv"
        validate_summary = validate_manifest_rows(rows, manifest_path_out.name)
        write_manifest(manifest_path_out, rows)
        target_summaries[label] = summary
        validation_rows.append(validate_summary)
        manifest_outputs[label] = display_path(manifest_path_out)
        print(
            f"{manifest_path_out.name}: selected {summary['selected_pair_count']} pairs, "
            f"dropped {summary['dropped_group_count']} groups"
        )

    common_size = most_common_size_um(candidates)
    all_size_rows = build_all_size_representative_manifest(
        grouped_candidates,
        tolerance=size_tolerance,
        preferred_targets=target_sizes,
        most_common_size=common_size,
    )
    all_manifest_path = out_dir / "manifest_all_size_representative_one_to_one.csv"
    validate_summary = validate_manifest_rows(all_size_rows, all_manifest_path.name)
    write_manifest(all_manifest_path, all_size_rows)
    validation_rows.append(validate_summary)
    manifest_outputs["all_size_representative"] = display_path(all_manifest_path)
    print(f"{all_manifest_path.name}: selected {len(all_size_rows)} representative pairs")

    summary_rows = build_size_summary_rows(grouped_candidates, target_summaries, all_size_rows)
    summary_path = out_dir / "manifest_size_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["scan_size_um", "afm_file_count", "unique_group_count"]
    write_csv(summary_path, summary_rows, fieldnames)

    warning_path = out_dir / "manifest_build_warnings.csv"
    write_csv(warning_path, warnings, ["sample_id", "afm_path", "reason"])

    build_summary = {
        "source_summary": source_summary,
        "group_count": len(grouped_candidates),
        "candidate_count": len(candidates),
        "size_tolerance": size_tolerance,
        "target_sizes_um": list(target_sizes),
        "most_common_size_um": common_size,
        "manifest_outputs": manifest_outputs,
        "target_summary_by_label": target_summaries,
        "validation": validation_rows,
        "warning_count": len(warnings),
        "warning_csv": display_path(warning_path),
        "size_summary_csv": display_path(summary_path),
    }
    (out_dir / "manifest_build_summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return build_summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair_root = resolve_existing_path(args.pair_root)
    descriptor_aux_csv = resolve_existing_path(args.descriptor_aux_csv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)
    manifest_path = None if args.manifest is None else resolve_existing_path(args.manifest)

    if manifest_path is not None and not manifest_path.is_file():
        raise SystemExit(f"Input manifest does not exist: {manifest_path}")
    if manifest_path is None:
        if not pair_root.is_dir():
            raise SystemExit(f"Pair root does not exist: {pair_root}")
        if not descriptor_aux_csv.is_file():
            raise SystemExit(f"Descriptor aux CSV does not exist: {descriptor_aux_csv}")
    summary = build_one_to_one_manifests(
        manifest_path=manifest_path,
        pair_root=pair_root,
        descriptor_aux_csv=descriptor_aux_csv,
        out_dir=out_dir,
        target_sizes=args.target_sizes,
        size_tolerance=args.size_tolerance,
    )
    print(
        f"Built one-to-one manifests for {summary['group_count']} groups with {summary['candidate_count']} AFM candidates."
    )
    print(f"Size summary: {summary['size_summary_csv']}")
    print(f"Warnings: {summary['warning_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
