"""ffprobe/ffmpeg based MPG metadata and deterministic frame extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .common import parse_rate, run_command, sha256_file, which_or_error


def ffprobe_version() -> str:
    exe = which_or_error("ffprobe")
    result = run_command([exe, "-version"], timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe version check failed")
    return result.stdout.splitlines()[0]


def ffmpeg_version() -> str:
    exe = which_or_error("ffmpeg")
    result = run_command([exe, "-version"], timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg version check failed")
    return result.stdout.splitlines()[0]


def probe_video(path: Path) -> dict[str, Any]:
    exe = which_or_error("ffprobe")
    args = [
        exe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(args, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or 0.0
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    duration = stream.get("duration")
    return {
        "codec": stream.get("codec_name") or "",
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": float(fps),
        "duration_sec": float(duration) if duration not in (None, "N/A") else 0.0,
        "frame_count": int(frame_count) if frame_count not in (None, "N/A") else 0,
        "frame_count_method": "ffprobe -count_frames nb_read_frames",
    }


def extract_frame(video_path: Path, frame_index: int, output_path: Path, display_transform: str | None = None) -> Path:
    """Extract one source frame losslessly as RGB PNG using ffmpeg select."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    exe = which_or_error("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp.png")
    tmp_path.unlink(missing_ok=True)
    args = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-frames:v",
        "1",
        "-vcodec",
        "png",
        "-compression_level",
        "0",
        "-vsync",
        "0",
        "-f",
        "image2",
        str(tmp_path),
    ]
    result = run_command(args, timeout=240)
    if result.returncode != 0 or not tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"ffmpeg could not extract frame {frame_index} from {video_path}")
    if display_transform:
        transform_frame_png(tmp_path, display_transform)
    os.replace(tmp_path, output_path)
    return output_path


def transform_frame_png(path: Path, display_transform: str) -> None:
    if display_transform == "none":
        return
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if display_transform == "rotate_clockwise_90":
            transformed = rgb.transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError(f"Unsupported display transform: {display_transform}")
        transformed.save(path, format="PNG", compress_level=0)


def decode_test(video_path: Path, output_path: Path) -> tuple[bool, str]:
    try:
        extract_frame(video_path, 0, output_path)
        return True, "decoded frame 0 with ffmpeg"
    except Exception as exc:
        return False, str(exc)


def frame_sha(path: Path) -> str:
    return sha256_file(path)
