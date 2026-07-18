from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | Path) -> dict[str, Any]:
    # The project config is JSON-compatible YAML, avoiding an extra parser dependency.
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    output_root = repo_path(config["output_root"])
    report_root = repo_path(config["report_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    (output_root / "clip_cache").mkdir(parents=True, exist_ok=True)
    (report_root / "clip_previews").mkdir(parents=True, exist_ok=True)
    (report_root / "figures").mkdir(parents=True, exist_ok=True)
    return output_root, report_root


def read_id_list(path: str | Path) -> set[str]:
    file_path = repo_path(path)
    if not file_path.exists():
        return set()
    ids: set[str] = set()
    for raw_line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"(\d+)", line)
        if match:
            ids.add(match.group(1))
    return ids


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    file_path = repo_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def write_json(value: Any, path: str | Path) -> None:
    file_path = repo_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return display_path(value)
    return str(value)


def finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def median_abs_deviation(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    med = np.median(finite)
    return float(np.median(np.abs(finite - med)))


def parse_stage(video_id: str) -> str:
    text = str(video_id).lower()
    if "oxide" in text or "sio2" in text:
        return "oxide_desorption"
    if "ramp" in text:
        return "ramp"
    if "gasb" in text:
        return "GaSb"
    if "alsb" in text:
        return "AlSb"
    if "substrate" in text or "initial" in text:
        return "initial_substrate"
    return "unknown"


def infer_material(sample_id: str, afm_ids: list[str]) -> str:
    joined = " ".join(afm_ids).lower()
    if "gdsb" in joined:
        return "GdSb"
    if "gasb" in joined:
        return "GaSb"
    if "alsb" in joined:
        return "AlSb"
    return "unknown"


def save_parquet(df: pd.DataFrame, path: str | Path) -> str:
    file_path = repo_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(file_path, index=False)
    except ImportError:
        df.to_csv(file_path.with_suffix(file_path.suffix + ".csv_fallback"), index=False)
    return display_path(file_path)
