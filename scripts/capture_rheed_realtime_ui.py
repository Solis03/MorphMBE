#!/usr/bin/env python3
"""Capture a reproducible offscreen screenshot of one replay prediction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from rheed2morph.realtime.ui import RealtimeMainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--video-contains", default="")
    parser.add_argument("--config", default="configs/rheed_realtime_ui.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["repository_root"] = str(config_path.parent.parent)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    window = RealtimeMainWindow(config)
    window.show()
    window.sample_combo.setCurrentText(str(args.sample_id))
    if args.video_contains:
        for index in range(window.video_combo.count()):
            if args.video_contains.lower() in window.video_combo.itemText(
                index
            ).lower():
                window.video_combo.setCurrentIndex(index)
                break
    window.speed.setValue(1.0)
    started = time.monotonic()
    state = {"started": False}

    def poll() -> None:
        if not state["started"] and window._model_ready:
            state["started"] = True
            window.start_session()
        if window.trend.times:
            window.grab().save(str(destination))
            window.close()
            app.quit()
            return
        if time.monotonic() - started > float(args.timeout_seconds):
            window.close()
            app.quit()
            raise TimeoutError("UI screenshot timed out before prediction")
        QTimer.singleShot(500, poll)

    QTimer.singleShot(200, poll)
    exit_code = app.exec()
    if not destination.exists():
        raise RuntimeError("UI screenshot was not created")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
