#!/usr/bin/env python3
"""Write a benchmark v1 environment snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from rheed2morph.benchmark.hashing import repo_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    payload = snapshot(root)
    write_json(root / args.output, payload)
    print(json.dumps({"status": "ok", "output": args.output}, indent=2, sort_keys=True))


def snapshot(root: Path) -> dict[str, Any]:
    packages = {
        dist.metadata["Name"].lower(): dist.version
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata
    }
    torch_info: dict[str, Any] = {
        "version": packages.get("torch"),
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
    }
    try:
        import torch

        torch_info.update(
            {
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        torch_info["error"] = repr(exc)
    return {
        "schema_version": "benchmark_environment_snapshot_v1",
        "protocol_version": "benchmark_v1",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "installed_packages": dict(sorted(packages.items())),
        "torch": torch_info,
        "cuda_available": torch_info["cuda_available"],
        "gpu_name": torch_info["gpu_name"],
        "scikit_learn_version": packages.get("scikit-learn"),
        "numpy_version": packages.get("numpy"),
        "scipy_version": packages.get("scipy"),
        "pandas_version": packages.get("pandas"),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_branch": git(root, "branch", "--show-current"),
        "git_dirty": bool(git(root, "status", "--porcelain").strip()),
        "hostname": socket.gethostname(),
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


if __name__ == "__main__":
    main()
