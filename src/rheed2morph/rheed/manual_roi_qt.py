"""PySide6 GUI for manual RHEED keyframe and ROI annotation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rheed2morph.rheed.manual_roi import (
    ROI,
    VideoRecord,
    clip_indices,
    discover_video_records,
    fit_display_transform,
    frame_path_for_index,
    numeric_frame_index,
    roi_from_selection,
    save_roi_preview,
    selection_from_record,
    sorted_frame_paths,
    update_video_selection,
    validate_selection,
    video_is_complete,
)


class ImageCanvas(QWidget):
    """Image display widget that maps drawn rectangles to source pixels."""

    roi_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(760, 640)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pixmap: QPixmap | None = None
        self.source_width = 0
        self.source_height = 0
        self.frame_index = 0
        self.roi: ROI | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_roi: ROI | None = None
        self.transform = None

    def set_image(self, frame_path: Path, frame_index: int) -> None:
        image = QImage(str(frame_path))
        if image.isNull():
            raise ValueError(f"Could not load frame: {frame_path}")
        self.pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB888))
        self.source_width = int(image.width())
        self.source_height = int(image.height())
        self.frame_index = int(frame_index)
        self.update()

    def set_roi(self, roi: ROI | None) -> None:
        self.roi = roi
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self.pixmap is None or self.source_width <= 0 or self.source_height <= 0:
            painter.end()
            return
        self.transform = fit_display_transform(self.source_width, self.source_height, self.width(), self.height())
        target = QRectF(
            self.transform.display_x,
            self.transform.display_y,
            self.transform.display_width,
            self.transform.display_height,
        )
        painter.drawPixmap(target, self.pixmap, QRectF(0, 0, self.source_width, self.source_height))
        for roi, color in ((self.roi, Qt.GlobalColor.green), (self.drag_roi, Qt.GlobalColor.yellow)):
            if roi is None:
                continue
            x0, y0, x1, y1 = self.transform.source_to_display_rect(roi)
            painter.setPen(QPen(color, 2))
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        painter.end()

    def source_point(self, point: QPoint) -> tuple[int, int]:
        if self.transform is None:
            self.transform = fit_display_transform(self.source_width, self.source_height, self.width(), self.height())
        return self.transform.display_to_source_point(float(point.x()), float(point.y()))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.pixmap is None:
            return
        self.drag_start = self.source_point(event.position().toPoint())
        self.drag_roi = None
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start is None:
            return
        current = self.source_point(event.position().toPoint())
        self.drag_roi = ROI.from_points(
            self.drag_start,
            current,
            self.source_width,
            self.source_height,
            self.frame_index,
        )
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.drag_start is None:
            return
        current = self.source_point(event.position().toPoint())
        self.roi = ROI.from_points(
            self.drag_start,
            current,
            self.source_width,
            self.source_height,
            self.frame_index,
        )
        self.drag_start = None
        self.drag_roi = None
        self.roi_changed.emit(self.roi)
        self.update()

    def cancel_drag(self) -> None:
        self.drag_start = None
        self.drag_roi = None
        self.update()


class ReviewerWindow(QMainWindow):
    """Main reviewer window."""

    def __init__(self, records: list[VideoRecord], start_index: int) -> None:
        super().__init__()
        self.records = records
        self.current_index = max(0, min(start_index, len(records) - 1))
        self.frame_indices: list[int] = []
        self.current_frame_index = 0
        self.roi: ROI | None = None
        self.setWindowTitle("RHEED Keyframe and ROI Reviewer")
        self.resize(1500, 950)
        self._build_ui()
        self._build_shortcuts()
        self.load_record(self.current_index)

    @property
    def record(self) -> VideoRecord:
        return self.records[self.current_index]

    def _build_ui(self) -> None:
        main = QWidget()
        self.setCentralWidget(main)
        root = QGridLayout(main)
        root.setColumnStretch(1, 1)

        left = QVBoxLayout()
        root.addLayout(left, 0, 0)
        self.sample_label = QLabel()
        self.video_label = QLabel()
        self.video_label.setWordWrap(True)
        self.status_label = QLabel()
        self.position_label = QLabel()
        self.remaining_label = QLabel()
        self.clip_indices_label = QLabel()
        self.clip_indices_label.setWordWrap(True)
        for widget in (
            QLabel("<b>Video Status</b>"),
            self.sample_label,
            self.video_label,
            self.status_label,
            self.position_label,
            self.remaining_label,
            self.clip_indices_label,
        ):
            left.addWidget(widget)
        left.addSpacing(16)
        self._add_button(left, "Previous Video", self.previous_video)
        self._add_button(left, "Next Video", self.next_video)
        self._add_button(left, "Skip", self.next_video)
        left.addSpacing(16)
        self._add_button(left, "Save", self.save_selection)
        self._add_button(left, "Save and Next", self.save_and_next)
        left.addSpacing(16)
        self._add_button(left, "Reset ROI", self.reset_roi)
        left.addStretch(1)

        center = QVBoxLayout()
        root.addLayout(center, 0, 1)
        self.canvas = ImageCanvas()
        self.canvas.roi_changed.connect(self.on_roi_changed)
        center.addWidget(self.canvas, stretch=1)
        browser = QHBoxLayout()
        center.addLayout(browser)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.set_frame_by_index)
        browser.addWidget(self.frame_slider, stretch=1)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.set_frame_by_index)
        browser.addWidget(self.frame_spin)
        self._add_button(browser, "K", self.set_keyframe_current)

        right = QVBoxLayout()
        root.addLayout(right, 0, 2)
        selection_box = QGroupBox("Selection")
        right.addWidget(selection_box)
        form = QFormLayout(selection_box)
        self.keyframe_edit = QLineEdit()
        self.clip_count_edit = QLineEdit()
        self.roi_x_edit = QLineEdit()
        self.roi_y_edit = QLineEdit()
        self.roi_w_edit = QLineEdit()
        self.roi_h_edit = QLineEdit()
        self.source_w_edit = QLineEdit()
        self.source_h_edit = QLineEdit()
        self.source_w_edit.setReadOnly(True)
        self.source_h_edit.setReadOnly(True)
        for label, widget in (
            ("keyframe_index", self.keyframe_edit),
            ("clip_frame_count", self.clip_count_edit),
            ("ROI x", self.roi_x_edit),
            ("ROI y", self.roi_y_edit),
            ("ROI width", self.roi_w_edit),
            ("ROI height", self.roi_h_edit),
            ("source width", self.source_w_edit),
            ("source height", self.source_h_edit),
        ):
            form.addRow(label, widget)
        self._add_button(form, "Set Keyframe to Current (K)", self.set_keyframe_current)
        self._add_button(form, "Apply ROI Fields", self.apply_roi_fields)

        preview_box = QGroupBox("ROI Clip Preview")
        right.addWidget(preview_box, stretch=1)
        preview_layout = QVBoxLayout(preview_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.preview_label)
        preview_layout.addWidget(scroll)
        self._add_button(preview_layout, "Refresh Preview", self.refresh_preview)

    def _add_button(self, layout, text: str, callback: Callable[[], None]) -> None:
        button = QPushButton(text)
        button.clicked.connect(callback)
        if isinstance(layout, QFormLayout):
            layout.addRow(button)
        else:
            layout.addWidget(button)

    def _build_shortcuts(self) -> None:
        for text, callback in (
            ("Save", self.save_selection),
            ("Save and Next", self.save_and_next),
            ("Previous Video", self.previous_video),
            ("Next Video", self.next_video),
            ("Reset ROI", self.reset_roi),
        ):
            action = QAction(text, self)
            action.triggered.connect(callback)
            self.addAction(action)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Left:
            self.step_frame(-10 if shift else -1)
        elif key == Qt.Key.Key_Right:
            self.step_frame(10 if shift else 1)
        elif key == Qt.Key.Key_Home and self.frame_indices:
            self.set_frame_by_index(self.frame_indices[0])
        elif key == Qt.Key.Key_End and self.frame_indices:
            self.set_frame_by_index(self.frame_indices[-1])
        elif key == Qt.Key.Key_K:
            self.set_keyframe_current()
        elif key == Qt.Key.Key_R:
            self.reset_roi()
        elif key == Qt.Key.Key_Escape:
            self.canvas.cancel_drag()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.save_and_next()
        else:
            super().keyPressEvent(event)

    def load_record(self, index: int) -> None:
        self.current_index = max(0, min(index, len(self.records) - 1))
        self.frame_indices = [numeric_frame_index(path) for path in sorted_frame_paths(self.record.frames_dir)]
        if not self.frame_indices:
            raise ValueError(f"No PNG frames found in {self.record.frames_dir}")
        selection = selection_from_record(self.record)
        keyframe = selection.get("keyframe_index")
        self.current_frame_index = int(keyframe) if keyframe is not None else self.frame_indices[0]
        self.current_frame_index = max(self.frame_indices[0], min(self.current_frame_index, self.frame_indices[-1]))
        self.keyframe_edit.setText("" if keyframe is None else str(keyframe))
        self.clip_count_edit.setText("" if selection.get("clip_frame_count") is None else str(selection.get("clip_frame_count")))
        self.roi = roi_from_selection(selection)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setRange(self.frame_indices[0], self.frame_indices[-1])
        self.frame_spin.setRange(self.frame_indices[0], self.frame_indices[-1])
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.set_frame_by_index(self.current_frame_index)
        self.update_roi_fields()
        self.update_status()
        self.refresh_preview()

    def update_status(self) -> None:
        incomplete = sum(1 for record in self.records if not video_is_complete(record))
        self.sample_label.setText(f"sample_id: {self.record.sample_id}")
        self.video_label.setText(f"video_id: {self.record.video_id}")
        self.status_label.setText(f"completed: {video_is_complete(self.record)}")
        self.position_label.setText(f"video: {self.current_index + 1} / {len(self.records)}")
        self.remaining_label.setText(f"unfinished: {incomplete}")

    def set_frame_by_index(self, value: int) -> None:
        if not self.frame_indices:
            return
        value = max(self.frame_indices[0], min(int(value), self.frame_indices[-1]))
        if value not in set(self.frame_indices):
            value = min(self.frame_indices, key=lambda item: abs(item - value))
        self.current_frame_index = value
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_spin.setValue(value)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        frame_path = frame_path_for_index(self.record.frames_dir, value)
        self.canvas.set_image(frame_path, value)
        self.canvas.set_roi(self.roi)
        with Image.open(frame_path) as image:
            self.source_w_edit.setText(str(image.width))
            self.source_h_edit.setText(str(image.height))

    def step_frame(self, delta: int) -> None:
        self.set_frame_by_index(self.current_frame_index + delta)

    def set_keyframe_current(self) -> None:
        self.keyframe_edit.setText(str(self.current_frame_index))
        if self.roi is not None:
            self.roi = ROI(
                x=self.roi.x,
                y=self.roi.y,
                width=self.roi.width,
                height=self.roi.height,
                source_width=self.roi.source_width,
                source_height=self.roi.source_height,
                reference_frame_index=self.current_frame_index,
            )
            self.update_roi_fields()
        self.refresh_preview()

    def on_roi_changed(self, roi: ROI) -> None:
        self.roi = roi
        self.update_roi_fields()
        self.refresh_preview()

    def update_roi_fields(self) -> None:
        if self.roi is None:
            for edit in (self.roi_x_edit, self.roi_y_edit, self.roi_w_edit, self.roi_h_edit):
                edit.setText("")
            self.canvas.set_roi(None)
            return
        self.roi_x_edit.setText(str(self.roi.x))
        self.roi_y_edit.setText(str(self.roi.y))
        self.roi_w_edit.setText(str(self.roi.width))
        self.roi_h_edit.setText(str(self.roi.height))
        self.canvas.set_roi(self.roi)

    def apply_roi_fields(self) -> None:
        try:
            keyframe = int(self.keyframe_edit.text() or self.current_frame_index)
            roi = ROI(
                x=int(self.roi_x_edit.text()),
                y=int(self.roi_y_edit.text()),
                width=int(self.roi_w_edit.text()),
                height=int(self.roi_h_edit.text()),
                source_width=int(self.source_w_edit.text()),
                source_height=int(self.source_h_edit.text()),
                reference_frame_index=keyframe,
            )
            validate_selection(self.record.frames_dir, keyframe, int(self.clip_count_edit.text() or "1"), roi)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid ROI", str(exc))
            return
        self.roi = roi
        self.update_roi_fields()
        self.refresh_preview()

    def refresh_preview(self) -> None:
        if self.roi is None:
            self.preview_label.setText("ROI not set")
            self.clip_indices_label.setText("clip indices: ROI not set")
            return
        try:
            keyframe = int(self.keyframe_edit.text() or self.current_frame_index)
            clip_count = int(self.clip_count_edit.text() or "1")
            indices = clip_indices(keyframe, clip_count, self.frame_indices[0], self.frame_indices[-1])
            self.clip_indices_label.setText(f"clip indices: {indices}")
            preview_path = save_roi_preview(self.record, keyframe, clip_count, self.roi, thumb_height=120)
            pixmap = QPixmap(str(preview_path))
            self.preview_label.setPixmap(pixmap)
            self.preview_label.resize(pixmap.size())
        except Exception as exc:
            self.preview_label.setText(f"Preview unavailable: {exc}")
            self.clip_indices_label.setText(f"clip preview unavailable: {exc}")

    def reset_roi(self) -> None:
        self.roi = None
        self.update_roi_fields()
        self.refresh_preview()

    def save_selection(self) -> None:
        try:
            keyframe = int(self.keyframe_edit.text())
            clip_count = int(self.clip_count_edit.text())
            if self.roi is None:
                raise ValueError("ROI is required before saving.")
            roi = ROI(
                x=self.roi.x,
                y=self.roi.y,
                width=self.roi.width,
                height=self.roi.height,
                source_width=self.roi.source_width,
                source_height=self.roi.source_height,
                reference_frame_index=keyframe,
            )
            validate_selection(self.record.frames_dir, keyframe, clip_count, roi)
            update_video_selection(self.record.metadata_path, self.record.video_id, keyframe, clip_count, roi)
            save_roi_preview(self.record, keyframe, clip_count, roi)
            self._reload_current_record()
            self.update_status()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _reload_current_record(self) -> None:
        root = self.record.metadata_path.parents[1]
        for record in discover_video_records(root):
            if record.sample_id == self.record.sample_id and record.video_id == self.record.video_id:
                self.records[self.current_index] = record
                return

    def save_and_next(self) -> None:
        before = video_is_complete(self.record)
        self.save_selection()
        after = video_is_complete(self.record)
        if after or not before:
            self.next_unfinished_or_next()

    def next_unfinished_or_next(self) -> None:
        for offset in range(1, len(self.records) + 1):
            index = (self.current_index + offset) % len(self.records)
            if not video_is_complete(self.records[index]):
                self.load_record(index)
                return
        self.next_video()

    def next_video(self) -> None:
        self.load_record((self.current_index + 1) % len(self.records))

    def previous_video(self) -> None:
        self.load_record((self.current_index - 1) % len(self.records))


def launch_reviewer(records: list[VideoRecord], start_index: int) -> None:
    """Launch the PySide6 reviewer."""

    app = QApplication.instance() or QApplication([])
    window = ReviewerWindow(records, start_index)
    window.show()
    app.exec()
