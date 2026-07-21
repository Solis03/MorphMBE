"""Shared helpers for prospective unseen RHEED keyframe selection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SAMPLE_IDS = ["N6342", "N6358", "N6382", "N6389", "N6390"]
PACKAGE_REL = Path("publication_freeze/prospective_unseen_single_frame_v1")
FREEZE_REL = Path("publication_freeze/rheed_afm_single_frame_v1_2026-07-18")


def repo_root_from(path: Path | None = None) -> Path:
    """Return the repository root by walking upward from path or cwd."""

    start = (path or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "publication_freeze").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


def package_root(repo_root: Path) -> Path:
    return repo_root / PACKAGE_REL


def relpath(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False)
    tmp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path, mode: str = "fast", chunk_size: int = 1024 * 1024) -> tuple[str | None, str]:
    """Return no hash, a fast file fingerprint, or a full streaming SHA256."""

    if mode == "none":
        return None, "not computed (--video-hash-mode none)"
    if mode == "full":
        return sha256_file(path), "full sha256"
    if mode != "fast":
        raise ValueError("video hash mode must be one of: none, fast, full")
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest(), "fast fingerprint: size, mtime, first chunk, last chunk"


def which_or_error(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(
            f"Missing required dependency `{name}`. Install it on macOS with: brew install ffmpeg"
        )
    return found


def run_command(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, text=True, capture_output=True, timeout=timeout)


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        if denominator == 0:
            return None
        return float(num) / denominator
    return float(value)


def simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small repository-owned config without requiring PyYAML."""

    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, dict[str, Any], str] | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            value = _yaml_scalar(text[2:].strip())
            if not isinstance(parent, list):
                if pending_key is None:
                    raise ValueError(f"Unexpected list item in {path}: {raw}")
                _, pending_parent, key = pending_key
                new_list: list[Any] = []
                pending_parent[key] = new_list
                stack.append((pending_key[0], new_list))
                parent = new_list
            parent.append(value)
            continue
        key, _, value_text = text.partition(":")
        key = key.strip()
        if value_text.strip():
            parent[key] = _yaml_scalar(value_text.strip())
            pending_key = None
        else:
            new_map: dict[str, Any] = {}
            parent[key] = new_map
            pending_key = (indent, parent, key)
            stack.append((indent, new_map))
    return root


def _yaml_scalar(text: str) -> Any:
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_yaml_scalar(item.strip()) for item in inner.split(",")]
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text.strip("'\"")


def ensure_package_dirs(root: Path) -> None:
    for rel in [
        "code/keyframe_selector",
        "config",
        "metadata/samples",
        "manifests",
        "keyframes/raw",
        "keyframes/roi",
        "keyframes/model_ready",
        "previews",
        "logs",
        "cache/thumbnails",
        "cache/frames",
        "provenance",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SampleConfig:
    sample_ids: list[str]
    data_root: Path
    freeze_id: str
    freeze_path: Path
    frames_before: int
    frames_after: int
    frame_stride: int
    display_transforms: dict[str, str]


def load_config(repo_root: Path) -> SampleConfig:
    cfg = simple_yaml(package_root(repo_root) / "config" / "unseen_samples.yaml")
    freeze = cfg.get("freeze_reference", {})
    default_context = cfg.get("default_context", {})
    return SampleConfig(
        sample_ids=[str(item) for item in cfg.get("sample_ids", EXPECTED_SAMPLE_IDS)],
        data_root=repo_root / str(cfg.get("data_root", "data/compressedfile")),
        freeze_id=str(freeze.get("freeze_id", "rheed_afm_single_frame_v1_2026-07-18")),
        freeze_path=repo_root / str(freeze.get("freeze_path", FREEZE_REL.as_posix())),
        frames_before=int(default_context.get("frames_before", 0)),
        frames_after=int(default_context.get("frames_after", 0)),
        frame_stride=int(default_context.get("frame_stride", 1)),
        display_transforms={str(key): str(value) for key, value in dict(cfg.get("display_transforms", {})).items()},
    )


def sample_numeric(sample_id: str) -> int | None:
    digits = "".join(ch for ch in sample_id if ch.isdigit())
    return int(digits) if digits else None
