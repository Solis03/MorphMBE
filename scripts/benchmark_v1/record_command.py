#!/usr/bin/env python3
"""Run a command and append stdout, stderr, exit code, and runtime to JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cmd:
        raise SystemExit("no command provided")
    if args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    start = time.monotonic()
    result = subprocess.run(args.cmd, text=True, capture_output=True, check=False)
    runtime = time.monotonic() - start
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": args.cmd,
        "exit_code": result.returncode,
        "runtime_seconds": runtime,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(result.stdout, end="")
    print(result.stderr, end="", file=__import__("sys").stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
