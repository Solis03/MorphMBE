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


def main() -> None:
    args = _arguments()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository = config_path.parent.parent
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from .model import build_deployment_bundle, save_deployment_bundle
    from .ui import RealtimeMainWindow

    config["repository_root"] = str(repository)
    bundle_path = repository / config["deployment_bundle"]
    if not bundle_path.exists():
        print("Preparing the M15b + frozen M12a deployment cache...")
        bundle = build_deployment_bundle(config, progress=print)
        save_deployment_bundle(bundle, bundle_path)
    application = QApplication(sys.argv)
    application.setApplicationName("MorphMBE Realtime Morphology")
    window = RealtimeMainWindow(config)
    window.show()
    raise SystemExit(application.exec())


__all__ = ["main"]
