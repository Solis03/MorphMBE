"""Installed console entry point for the desktop replay application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_realtime_ui.json",
    )
    return parser.parse_args()


def repository_root_from_config(
    config_path: str | Path,
    config: dict[str, object],
) -> Path:
    """Resolve the package root for configs in ``configs/`` or subfolders."""

    path = Path(config_path).resolve()
    configured = Path(str(config.get("repository_root", "."))).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    for candidate in path.parents:
        if (
            (candidate / "src" / "rheed2morph").is_dir()
            and (candidate / "configs").is_dir()
        ):
            return candidate
    return (path.parent / configured).resolve()


def main() -> None:
    args = _arguments()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository = repository_root_from_config(config_path, config)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from .model import build_deployment_bundle, save_deployment_bundle
    from .ui import RealtimeMainWindow

    config["repository_root"] = str(repository)
    bundle_path = repository / config["deployment_bundle"]
    if not bundle_path.exists():
        print("Preparing the configured MorphMBE deployment cache...")
        bundle = build_deployment_bundle(config, progress=print)
        save_deployment_bundle(bundle, bundle_path)
    application = QApplication(sys.argv)
    application.setApplicationName("MorphMBE Realtime Morphology")
    window = RealtimeMainWindow(config)
    window.show()
    raise SystemExit(application.exec())


__all__ = ["main"]
