"""PySide6 GUI for manually selecting prospective unseen RHEED keyframes."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .common import EXPECTED_SAMPLE_IDS, load_config, load_json, relpath
from .decoder import extract_frame
from .manifests import discover_mpgs, metadata_path, save_selection


class ImageCanvas(QWidget):
    """Frame display widget that maps mouse-drawn ROIs to source pixels."""

    roi_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(760, 560)
        self.pixmap: QPixmap | None = None
        self.source_width = 0
        self.source_height = 0
        self.frame_index = 0
        self.roi: dict | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_roi: dict | None = None
        self.display_rect = QRectF()

    def set_image(self, frame_path: Path, frame_index: int) -> None:
        image = QImage(str(frame_path))
        if image.isNull():
            raise ValueError(f"Could not load frame: {frame_path}")
        self.pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB888))
        self.source_width = int(image.width())
        self.source_height = int(image.height())
        self.frame_index = int(frame_index)
        self.update()

    def set_roi(self, roi: dict | None) -> None:
        self.roi = roi
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self.pixmap is None or self.source_width <= 0 or self.source_height <= 0:
            painter.end()
            return
        self.display_rect = self._fit_rect()
        painter.drawPixmap(self.display_rect, self.pixmap, QRectF(0, 0, self.source_width, self.source_height))
        for roi, color in ((self.roi, Qt.GlobalColor.green), (self.drag_roi, Qt.GlobalColor.yellow)):
            if roi is None:
                continue
            painter.setPen(QPen(color, 2))
            painter.drawRect(self._source_roi_to_display_rect(roi))
        painter.end()

    def _fit_rect(self) -> QRectF:
        scale = min(self.width() / self.source_width, self.height() / self.source_height)
        width = self.source_width * scale
        height = self.source_height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _source_roi_to_display_rect(self, roi: dict) -> QRectF:
        scale_x = self.display_rect.width() / self.source_width
        scale_y = self.display_rect.height() / self.source_height
        return QRectF(
            self.display_rect.x() + int(roi["x"]) * scale_x,
            self.display_rect.y() + int(roi["y"]) * scale_y,
            int(roi["width"]) * scale_x,
            int(roi["height"]) * scale_y,
        )

    def _display_to_source_point(self, point: QPoint) -> tuple[int, int]:
        if self.display_rect.isNull():
            self.display_rect = self._fit_rect()
        x = round((point.x() - self.display_rect.x()) / self.display_rect.width() * self.source_width)
        y = round((point.y() - self.display_rect.y()) / self.display_rect.height() * self.source_height)
        return (
            max(0, min(self.source_width, int(x))),
            max(0, min(self.source_height, int(y))),
        )

    def _roi_from_points(self, first: tuple[int, int], second: tuple[int, int]) -> dict:
        x0, y0 = first
        x1, y1 = second
        left = max(0, min(self.source_width, min(x0, x1)))
        top = max(0, min(self.source_height, min(y0, y1)))
        right = max(0, min(self.source_width, max(x0, x1)))
        bottom = max(0, min(self.source_height, max(y0, y1)))
        return {
            "x": int(left),
            "y": int(top),
            "width": int(right - left),
            "height": int(bottom - top),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "reference_frame_index": int(self.frame_index),
            "coordinate_space": "source_frame_pixels",
        }

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.pixmap is None:
            return
        self.drag_start = self._display_to_source_point(event.position().toPoint())
        self.drag_roi = None
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start is None:
            return
        self.drag_roi = self._roi_from_points(self.drag_start, self._display_to_source_point(event.position().toPoint()))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start is None:
            return
        self.roi = self._roi_from_points(self.drag_start, self._display_to_source_point(event.position().toPoint()))
        self.drag_start = None
        self.drag_roi = None
        self.roi_changed.emit(self.roi)
        self.update()


class KeyframeSelector(QMainWindow):
    def __init__(self, repo_root: Path, package_root: Path) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.package_root = package_root
        self.config = load_config(repo_root)
        self.sample_ids = list(self.config.sample_ids)
        self.current_sample_index = self._first_pending_index()
        self.current_frame_index = 0
        self.selected_frame_index: int | None = None
        self.roi: dict | None = None
        self.current_frame_path: Path | None = None
        self.setWindowTitle("Prospective Unseen RHEED Keyframe Selector")
        self.resize(1420, 900)
        self._build_ui()
        self.load_sample(self.current_sample_index)

    def _build_ui(self) -> None:
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QGridLayout(root_widget)
        root.setColumnStretch(1, 1)

        left = QVBoxLayout()
        root.addLayout(left, 0, 0)
        self.progress_label = QLabel()
        self.sample_list = QComboBox()
        self.sample_list.addItems(self.sample_ids)
        self.sample_list.currentIndexChanged.connect(self.load_sample)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.video_combo = QComboBox()
        self.video_combo.currentIndexChanged.connect(self.video_changed)
        self.video_path_label = QLabel()
        self.video_path_label.setWordWrap(True)
        for widget in [self.progress_label, self.sample_list, self.status_label, QLabel("MPG source"), self.video_combo, self.video_path_label]:
            left.addWidget(widget)
        for text, callback in [
            ("Previous Sample", self.previous_sample),
            ("Next Sample", self.next_sample),
            ("Next Pending", self.next_pending),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            left.addWidget(button)
        left.addStretch(1)

        center = QVBoxLayout()
        root.addLayout(center, 0, 1)
        self.canvas = ImageCanvas()
        self.canvas.roi_changed.connect(self.on_roi_changed)
        center.addWidget(self.canvas, stretch=1)
        nav = QHBoxLayout()
        center.addLayout(nav)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.set_frame_index)
        nav.addWidget(self.frame_slider, stretch=1)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.set_frame_index)
        nav.addWidget(self.frame_spin)
        for delta in [-30, -10, -5, -1, 1, 5, 10, 30]:
            button = QPushButton(f"{delta:+d}")
            button.clicked.connect(lambda _checked=False, value=delta: self.step_frame(value))
            nav.addWidget(button)

        right = QVBoxLayout()
        root.addLayout(right, 0, 2)
        meta_box = QGroupBox("Frame Metadata")
        right.addWidget(meta_box)
        form = QFormLayout(meta_box)
        self.frame_info = QLabel()
        self.frame_info.setWordWrap(True)
        form.addRow(self.frame_info)
        self.timestamp_spin = QDoubleSpinBox()
        self.timestamp_spin.setDecimals(3)
        self.timestamp_spin.setRange(0, 999999)
        self.timestamp_spin.valueChanged.connect(self.timestamp_changed)
        form.addRow("timestamp sec", self.timestamp_spin)
        self.set_keyframe_button = QPushButton("Set as keyframe")
        self.set_keyframe_button.clicked.connect(self.set_as_keyframe)
        form.addRow(self.set_keyframe_button)
        self.selected_label = QLabel("selected: none")
        form.addRow(self.selected_label)

        context_box = QGroupBox("Context")
        right.addWidget(context_box)
        context_form = QFormLayout(context_box)
        self.before_spin = QSpinBox()
        self.before_spin.setRange(0, 100000)
        self.before_spin.setValue(self.config.frames_before)
        self.after_spin = QSpinBox()
        self.after_spin.setRange(0, 100000)
        self.after_spin.setValue(self.config.frames_after)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 1000)
        self.stride_spin.setValue(self.config.frame_stride)
        context_form.addRow("frames_before", self.before_spin)
        context_form.addRow("frames_after", self.after_spin)
        context_form.addRow("frame_stride", self.stride_spin)

        roi_box = QGroupBox("ROI")
        right.addWidget(roi_box)
        roi_form = QFormLayout(roi_box)
        self.roi_x_edit = QLineEdit()
        self.roi_y_edit = QLineEdit()
        self.roi_w_edit = QLineEdit()
        self.roi_h_edit = QLineEdit()
        for label, widget in [
            ("x", self.roi_x_edit),
            ("y", self.roi_y_edit),
            ("width", self.roi_w_edit),
            ("height", self.roi_h_edit),
        ]:
            roi_form.addRow(label, widget)
        apply_roi = QPushButton("Apply ROI fields")
        apply_roi.clicked.connect(self.apply_roi_fields)
        roi_form.addRow(apply_roi)
        reset_roi = QPushButton("Reset ROI")
        reset_roi.clicked.connect(self.reset_roi)
        roi_form.addRow(reset_roi)

        notes_box = QGroupBox("Notes")
        right.addWidget(notes_box)
        notes_layout = QVBoxLayout(notes_box)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Optional selection notes")
        notes_layout.addWidget(self.notes)
        self.save_button = QPushButton("Save selection")
        self.save_button.clicked.connect(self.save_current_selection)
        right.addWidget(self.save_button)
        right.addStretch(1)

    def _first_pending_index(self) -> int:
        for index, sample_id in enumerate(self.sample_ids):
            path = metadata_path(self.package_root, sample_id)
            if not path.exists():
                return index
            if not self._complete_with_roi(load_json(path)):
                return index
        return 0

    def _complete_with_roi(self, payload: dict) -> bool:
        if payload.get("sample", {}).get("selection_status") != "completed":
            return False
        selection = payload.get("selection") or {}
        return isinstance(selection.get("roi"), dict) and isinstance(selection.get("roi_xyxy"), list) and bool(selection.get("roi_keyframe_png"))

    def sample_payload(self, sample_id: str | None = None) -> dict:
        sid = sample_id or self.sample_ids[self.current_sample_index]
        path = metadata_path(self.package_root, sid)
        if not path.exists():
            discover_mpgs(self.repo_root, self.package_root)
        return load_json(path)

    def load_sample(self, index: int) -> None:
        if index < 0 or index >= len(self.sample_ids):
            return
        self.current_sample_index = index
        self.sample_list.blockSignals(True)
        self.sample_list.setCurrentIndex(index)
        self.sample_list.blockSignals(False)
        sample_id = self.sample_ids[index]
        payload = self.sample_payload(sample_id)
        status = payload.get("sample", {}).get("selection_status", "missing")
        if status == "completed" and not self._complete_with_roi(payload):
            status = "needs_roi_review"
        completed = sum(1 for sid in self.sample_ids if metadata_path(self.package_root, sid).exists() and self._complete_with_roi(load_json(metadata_path(self.package_root, sid))))
        self.progress_label.setText(f"{index + 1} of {len(self.sample_ids)}; completed {completed}")
        self.status_label.setText(f"{sample_id}: {status}")
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        candidates = payload.get("source_video_candidates") or []
        for item in candidates:
            if str(item.get("extension", "")).lower() == ".mpg":
                self.video_combo.addItem(item.get("filename", ""), item.get("repo_relative_path"))
        selected_rel = (payload.get("source_video") or {}).get("repo_relative_path")
        if selected_rel:
            for i in range(self.video_combo.count()):
                if self.video_combo.itemData(i) == selected_rel:
                    self.video_combo.setCurrentIndex(i)
                    break
        self.video_combo.blockSignals(False)
        selection = payload.get("selection") or {}
        if selection:
            self.selected_frame_index = int(selection.get("selected_frame_index_0based"))
            self.roi = self._roi_from_selection(selection)
            self.before_spin.setValue(int(selection.get("requested_frames_before", 0)))
            self.after_spin.setValue(int(selection.get("requested_frames_after", 0)))
            self.stride_spin.setValue(int(selection.get("frame_stride", 1)))
            self.notes.setPlainText(str(selection.get("notes", "")))
        else:
            self.selected_frame_index = None
            self.roi = None
            self.notes.setPlainText("")
        self.update_roi_fields()
        self.video_changed()

    def selected_video_relpath(self) -> str | None:
        if self.video_combo.count() == 0:
            return None
        return str(self.video_combo.currentData())

    def video_changed(self) -> None:
        source_rel = self.selected_video_relpath()
        if not source_rel:
            self.video_path_label.setText("No MPG candidate. This sample is blocked until a source MPG exists.")
            return
        payload = self.sample_payload()
        candidate = next((item for item in payload.get("source_video_candidates", []) if item.get("repo_relative_path") == source_rel), {})
        frame_count = int(candidate.get("frame_count") or 1)
        fps = float(candidate.get("fps") or 0)
        duration = float(candidate.get("duration_sec") or 0)
        self.video_path_label.setText(source_rel)
        max_frame = max(0, frame_count - 1)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(0, max_frame)
        self.frame_spin.setRange(0, max_frame)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        frame = self.selected_frame_index if self.selected_frame_index is not None else 0
        frame = max(0, min(frame, max_frame))
        self.timestamp_spin.blockSignals(True)
        self.timestamp_spin.setRange(0, max(duration, 999999 if duration == 0 else duration))
        self.timestamp_spin.blockSignals(False)
        self.set_frame_index(frame)
        self._update_frame_info(candidate)

    def set_frame_index(self, value: int) -> None:
        source_rel = self.selected_video_relpath()
        if not source_rel:
            return
        self.current_frame_index = int(value)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_spin.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        payload = self.sample_payload()
        candidate = next((item for item in payload.get("source_video_candidates", []) if item.get("repo_relative_path") == source_rel), {})
        display_transform = self.config.display_transforms.get(self.sample_ids[self.current_sample_index])
        fps = float(candidate.get("fps") or 0)
        if fps:
            self.timestamp_spin.blockSignals(True)
            self.timestamp_spin.setValue(self.current_frame_index / fps)
            self.timestamp_spin.blockSignals(False)
        source = self.repo_root / source_rel
        cache_path = self.package_root / "cache" / "frames" / self.sample_ids[self.current_sample_index] / f"{self.current_frame_index:06d}.png"
        if not cache_path.exists():
            try:
                extract_frame(source, self.current_frame_index, cache_path, display_transform=display_transform)
            except Exception as exc:
                QMessageBox.critical(self, "Frame decode failed", str(exc))
                return
        self.current_frame_path = cache_path
        self.canvas.set_image(cache_path, self.current_frame_index)
        self.canvas.set_roi(self.roi)
        self._update_frame_info(candidate)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.current_frame_path and self.current_frame_path.exists():
            self.canvas.update()

    def timestamp_changed(self, value: float) -> None:
        payload = self.sample_payload()
        source_rel = self.selected_video_relpath()
        candidate = next((item for item in payload.get("source_video_candidates", []) if item.get("repo_relative_path") == source_rel), {})
        fps = float(candidate.get("fps") or 0)
        if fps:
            self.set_frame_index(round(value * fps))

    def _update_frame_info(self, candidate: dict) -> None:
        total = int(candidate.get("frame_count") or 0)
        fps = float(candidate.get("fps") or 0)
        timestamp = self.current_frame_index / fps if fps else 0
        self.frame_info.setText(
            f"sample: {self.sample_ids[self.current_sample_index]}\n"
            f"frame index: {self.current_frame_index}\n"
            f"frame number: {self.current_frame_index + 1}\n"
            f"timestamp: {timestamp:.3f} sec\n"
            f"total frames: {total}\n"
            f"fps: {fps:.6g}\n"
            f"duration: {candidate.get('duration_sec', '')}\n"
            f"resolution: {candidate.get('width', '')} x {candidate.get('height', '')}\n"
            f"display transform: {candidate.get('display_transform', 'none')}\n"
            f"codec: {candidate.get('codec', '')}\n"
            f"video: {candidate.get('filename', '')}"
        )

    def step_frame(self, delta: int) -> None:
        self.set_frame_index(self.current_frame_index + delta)

    def set_as_keyframe(self) -> None:
        self.selected_frame_index = self.current_frame_index
        self.selected_label.setText(f"selected: frame {self.selected_frame_index} ({self.selected_frame_index + 1})")

    def _roi_from_selection(self, selection: dict) -> dict | None:
        roi = selection.get("roi")
        if isinstance(roi, dict):
            return roi
        xyxy = selection.get("roi_xyxy")
        source_rel = self.selected_video_relpath()
        payload = self.sample_payload()
        candidate = next((item for item in payload.get("source_video_candidates", []) if item.get("repo_relative_path") == source_rel), {})
        if isinstance(xyxy, list) and len(xyxy) == 4:
            x0, y0, x1, y1 = [int(value) for value in xyxy]
            return {
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "source_width": int(candidate.get("display_width") or candidate.get("width") or 0),
                "source_height": int(candidate.get("display_height") or candidate.get("height") or 0),
                "reference_frame_index": int(selection.get("selected_frame_index_0based") or 0),
                "coordinate_space": "source_frame_pixels",
            }
        return None

    def on_roi_changed(self, roi: dict) -> None:
        self.roi = roi
        self.update_roi_fields()

    def update_roi_fields(self) -> None:
        if self.roi is None:
            for edit in [self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit]:
                edit.setText("")
            self.canvas.set_roi(None)
            return
        self.roi_x_edit.setText(str(int(self.roi["x"])))
        self.roi_y_edit.setText(str(int(self.roi["y"])))
        self.roi_w_edit.setText(str(int(self.roi["width"])))
        self.roi_h_edit.setText(str(int(self.roi["height"])))
        self.canvas.set_roi(self.roi)

    def apply_roi_fields(self) -> None:
        try:
            roi = {
                "x": int(self.roi_x_edit.text()),
                "y": int(self.roi_y_edit.text()),
                "width": int(self.roi_w_edit.text()),
                "height": int(self.roi_h_edit.text()),
                "source_width": int(self.canvas.source_width),
                "source_height": int(self.canvas.source_height),
                "reference_frame_index": int(self.current_frame_index),
                "coordinate_space": "source_frame_pixels",
            }
            self._validate_roi(roi)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid ROI", str(exc))
            return
        self.roi = roi
        self.update_roi_fields()

    def reset_roi(self) -> None:
        self.roi = None
        self.update_roi_fields()

    def _validate_roi(self, roi: dict) -> None:
        if int(roi["width"]) <= 0 or int(roi["height"]) <= 0:
            raise ValueError("ROI width and height must be greater than 0.")
        if int(roi["x"]) < 0 or int(roi["y"]) < 0:
            raise ValueError("ROI x/y must be non-negative.")
        if int(roi["x"]) + int(roi["width"]) > int(roi["source_width"]):
            raise ValueError("ROI exceeds source frame width.")
        if int(roi["y"]) + int(roi["height"]) > int(roi["source_height"]):
            raise ValueError("ROI exceeds source frame height.")

    def save_current_selection(self) -> None:
        source_rel = self.selected_video_relpath()
        if not source_rel:
            QMessageBox.critical(self, "Save failed", "No MPG source selected.")
            return
        if self.selected_frame_index is None:
            QMessageBox.critical(self, "Save failed", "Use 'Set as keyframe' before saving.")
            return
        if self.current_frame_path is None or self.current_frame_index != self.selected_frame_index:
            self.set_frame_index(self.selected_frame_index)
        if self.current_frame_path is None:
            QMessageBox.critical(self, "Save failed", "No decoded displayed frame is available.")
            return
        if self.roi is None:
            QMessageBox.critical(self, "Save failed", "Draw or enter an ROI before saving.")
            return
        try:
            self._validate_roi(self.roi)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        message = (
            f"Save {self.sample_ids[self.current_sample_index]} frame {self.selected_frame_index} with "
            f"frames_before={self.before_spin.value()}, frames_after={self.after_spin.value()}, "
            f"frame_stride={self.stride_spin.value()}, ROI="
            f"({self.roi['x']}, {self.roi['y']}, {self.roi['width']}, {self.roi['height']})?"
        )
        if QMessageBox.question(self, "Confirm selection", message) != QMessageBox.StandardButton.Yes:
            return
        try:
            save_selection(
                self.repo_root,
                self.package_root,
                self.sample_ids[self.current_sample_index],
                source_rel,
                self.selected_frame_index,
                self.before_spin.value(),
                self.after_spin.value(),
                self.stride_spin.value(),
                self.notes.toPlainText(),
                self.current_frame_path,
                self.roi,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", "Selection saved.")
        self.next_pending()

    def previous_sample(self) -> None:
        self.load_sample((self.current_sample_index - 1) % len(self.sample_ids))

    def next_sample(self) -> None:
        self.load_sample((self.current_sample_index + 1) % len(self.sample_ids))

    def next_pending(self) -> None:
        for offset in range(1, len(self.sample_ids) + 1):
            index = (self.current_sample_index + offset) % len(self.sample_ids)
            path = metadata_path(self.package_root, self.sample_ids[index])
            if not path.exists() or not self._complete_with_roi(load_json(path)):
                self.load_sample(index)
                return
        self.next_sample()


def clear_cache(package_root: Path) -> None:
    cache = package_root / "cache"
    if cache.exists():
        shutil.rmtree(cache)
    (cache / "thumbnails").mkdir(parents=True, exist_ok=True)
    (cache / "frames").mkdir(parents=True, exist_ok=True)


def launch(repo_root: Path, package_root: Path) -> None:
    if not (package_root / "manifests" / "discovered_mpg_files.csv").exists():
        discover_mpgs(repo_root, package_root)
    app = QApplication.instance() or QApplication([])
    window = KeyframeSelector(repo_root, package_root)
    window.show()
    app.exec()


def smoke(repo_root: Path, package_root: Path) -> None:
    if not (package_root / "manifests" / "discovered_mpg_files.csv").exists():
        discover_mpgs(repo_root, package_root)
    for sample_id in EXPECTED_SAMPLE_IDS:
        payload = load_json(metadata_path(package_root, sample_id))
        selection = payload.get("selection") or {}
        complete_with_roi = payload.get("sample", {}).get("selection_status") == "completed" and isinstance(selection.get("roi"), dict)
        status = "completed" if complete_with_roi else "needs_roi_review"
        print(sample_id, status, len(payload.get("source_video_candidates") or []))
