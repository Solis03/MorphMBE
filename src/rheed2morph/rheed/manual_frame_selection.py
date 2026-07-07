"""Build a manifest from human-edited RHEED frame selection files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
RANK_RE = re.compile(r"(?:^|[_-])rank0*(\d+)(?:$|[_-])", re.IGNORECASE)
FRAME_RE = re.compile(r"frame(?:_idx)?=?0*(\d+)", re.IGNORECASE)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def parse_manual_selection_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.lower().endswith(".png"):
        rank_match = re.search(r"rank0*(\d+)", text, flags=re.IGNORECASE)
        frame_match = re.search(r"frame0*(\d+)", text, flags=re.IGNORECASE)
        parsed: dict[str, Any] = {"kind": "filename", "raw": text}
        if rank_match:
            parsed["rank"] = int(rank_match.group(1))
        if frame_match:
            parsed["frame_idx"] = int(frame_match.group(1))
        if "rank" in parsed or "frame_idx" in parsed:
            return parsed
    rank_match = RANK_RE.search(text)
    if rank_match:
        return {"kind": "rank", "rank": int(rank_match.group(1)), "raw": text}
    if text.lower().startswith("rank"):
        digits = re.sub(r"\D", "", text)
        if digits:
            return {"kind": "rank", "rank": int(digits), "raw": text}
    frame_match = FRAME_RE.search(text)
    if frame_match:
        return {"kind": "frame_idx", "frame_idx": int(frame_match.group(1)), "raw": text}
    return {"kind": "invalid", "raw": text}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_source_video(frame_selection_dir: Path, candidate_rows: Sequence[dict[str, str]]) -> str:
    source_path = frame_selection_dir / "source_video.txt"
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8").strip()
    if candidate_rows:
        return candidate_rows[0].get("video_path", "")
    return ""


def match_selection(selection: dict[str, Any], candidates: Sequence[dict[str, str]]) -> dict[str, str] | None:
    if selection["kind"] == "rank" or ("rank" in selection and selection["kind"] == "filename"):
        rank = int(selection["rank"])
        for row in candidates:
            if int(float(row.get("candidate_rank", "0") or 0)) == rank:
                return row
    if selection["kind"] == "frame_idx" or ("frame_idx" in selection and selection["kind"] == "filename"):
        frame_idx = int(selection["frame_idx"])
        for row in candidates:
            if int(float(row.get("frame_idx", "-1") or -1)) == frame_idx:
                return row
    return None


def build_manifest(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    manual_files = sorted(root.rglob("manual_selected_frames.txt"))
    for manual_file in manual_files:
        frame_selection_dir = manual_file.parent
        sample_dir = frame_selection_dir.parent
        sample_id = sample_dir.name
        candidate_csv = frame_selection_dir / "candidate_frames.csv"
        candidates = read_csv(candidate_csv)
        source_video = load_source_video(frame_selection_dir, candidates)
        selections = [
            parsed
            for parsed in (parse_manual_selection_line(line) for line in manual_file.read_text(encoding="utf-8").splitlines())
            if parsed is not None
        ]
        real_selections = [selection for selection in selections if selection.get("kind") != "invalid"]
        invalid_selections = [selection for selection in selections if selection.get("kind") == "invalid"]
        if not real_selections and not invalid_selections:
            issue_rows.append(
                {
                    "sample_id": sample_id,
                    "manual_selection_file": display_path(manual_file),
                    "status": "pending",
                    "selection": "",
                    "message": "no uncommented selections",
                }
            )
            continue
        for selection in invalid_selections:
            issue_rows.append(
                {
                    "sample_id": sample_id,
                    "manual_selection_file": display_path(manual_file),
                    "status": "invalid",
                    "selection": selection.get("raw", ""),
                    "message": "could not parse selection",
                }
            )
        for selection in real_selections:
            match = match_selection(selection, candidates)
            if match is None:
                issue_rows.append(
                    {
                        "sample_id": sample_id,
                        "manual_selection_file": display_path(manual_file),
                        "status": "missing",
                        "selection": selection.get("raw", ""),
                        "message": "selection not found in candidate_frames.csv",
                    }
                )
                continue
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source_video": source_video,
                    "selected_rank": match.get("candidate_rank", ""),
                    "selected_frame_idx": match.get("frame_idx", ""),
                    "selected_timestamp_sec": match.get("timestamp_sec", ""),
                    "selected_png_path": match.get("candidate_png_path", ""),
                    "quality_score": match.get("quality_score", ""),
                    "manual_selection_file": display_path(manual_file),
                }
            )
    return manifest_rows, issue_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = resolve_path(args.root)
    out = resolve_path(args.out)
    manifest_rows, issue_rows = build_manifest(root)
    manifest_fields = [
        "sample_id",
        "source_video",
        "selected_rank",
        "selected_frame_idx",
        "selected_timestamp_sec",
        "selected_png_path",
        "quality_score",
        "manual_selection_file",
    ]
    issue_fields = ["sample_id", "manual_selection_file", "status", "selection", "message"]
    write_csv(out, manifest_rows, manifest_fields)
    issue_path = out.with_name(out.stem + "_issues.csv")
    write_csv(issue_path, issue_rows, issue_fields)
    pending = sum(1 for row in issue_rows if row.get("status") == "pending")
    invalid = sum(1 for row in issue_rows if row.get("status") in {"invalid", "missing"})
    print(f"Wrote {len(manifest_rows)} selected frame rows to {display_path(out)}")
    print(f"Wrote {len(issue_rows)} issue rows to {display_path(issue_path)}")
    print(f"Pending manual files: {pending}; invalid or missing selections: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
