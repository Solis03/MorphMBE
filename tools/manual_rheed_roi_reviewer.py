#!/usr/bin/env python3
"""Local GUI for manual RHEED keyframe and ROI annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rheed2morph.rheed.manual_roi import (
    discover_video_records,
    filtered_records,
    select_start_index,
    sorted_frame_paths,
    video_is_complete,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/rheed_keyframe_selection"))
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--start-from-first", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Discover data and initialize state without opening GUI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    all_records = discover_video_records(args.root)
    records = filtered_records(all_records, args.sample_id, args.video_id)
    if not records:
        raise SystemExit("No matching videos found.")
    start_index = select_start_index(records, args.start_from_first, args.sample_id, args.video_id)
    if args.smoke_test:
        record = records[start_index]
        frames = sorted_frame_paths(record.frames_dir)
        print(f"records: {len(records)}")
        print(f"start: {record.sample_id}/{record.video_id}")
        print(f"frames: {len(frames)}")
        print(f"complete: {video_is_complete(record)}")
        return

    from rheed2morph.rheed.manual_roi_qt import launch_reviewer

    launch_reviewer(records, start_index)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
