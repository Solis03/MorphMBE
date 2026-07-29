"""PySide6 desktop interface for simulated real-time morphology monitoring."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .catalog import VideoEntry, discover_videos, group_by_sample, read_removelist
from .selector import ReplaySelection
from .session import SessionRecorder
from .workers import (
    PredictionJob,
    PredictionResult,
    PredictionWorker,
    ReplayWorker,
)


class VideoCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(620, 520)
        self._frame: np.ndarray | None = None
        self._selection: ReplaySelection | None = None
        self._frame_index = 0
        self._seconds = 0.0
        self._event = False

    def set_selection(self, selection: ReplaySelection) -> None:
        self._selection = selection
        self.update()

    def set_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        seconds: float,
        event: bool,
    ) -> None:
        self._frame = np.ascontiguousarray(frame)
        self._frame_index = int(frame_index)
        self._seconds = float(seconds)
        self._event = bool(event)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#071018"))
        if self._frame is None:
            painter.setPen(QColor("#89a2b2"))
            painter.setFont(QFont("Arial", 17))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "等待 RHEED 视频输入",
            )
            return
        frame = self._frame
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        available = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        scale = min(available.width() / width, available.height() / height)
        target = QRectF(
            available.x() + (available.width() - width * scale) / 2,
            available.y() + (available.height() - height * scale) / 2,
            width * scale,
            height * scale,
        )
        painter.drawPixmap(target, QPixmap.fromImage(image), QRectF(image.rect()))
        if self._selection is not None:
            roi = self._selection.model_input_roi.rect
            color = QColor("#22d3a6")
            rect = QRectF(
                target.x() + roi.x * scale,
                target.y() + roi.y * scale,
                roi.width * scale,
                roi.height * scale,
            )
            painter.setPen(QPen(color, 3))
            painter.drawRect(rect)
            painter.setPen(color)
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(
                rect.adjusted(4, 16, 0, 0),
                "模型输入 / 完整点阵 ROI",
            )
            physics_roi = self._selection.physics_roi.rect
            physics_color = QColor("#f472b6")
            physics_rect = QRectF(
                target.x() + physics_roi.x * scale,
                target.y() + physics_roi.y * scale,
                physics_roi.width * scale,
                physics_roi.height * scale,
            )
            physics_pen = QPen(physics_color, 2)
            physics_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(physics_pen)
            painter.drawRect(physics_rect)
            painter.setPen(physics_color)
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(
                physics_rect.adjusted(4, 32, 0, 0),
                "物理特征 ROI（不用于生成器）",
            )
        banner = (
            f"frame {self._frame_index:05d}   t = {self._seconds:6.2f} s"
        )
        if self._event:
            banner += "   ● 形貌推理触发"
        painter.fillRect(
            QRectF(target.x(), target.y(), target.width(), 34),
            QColor(0, 0, 0, 155),
        )
        painter.setPen(QColor("#ffffff" if not self._event else "#ffdf65"))
        painter.setFont(QFont("Menlo", 12, QFont.Weight.Bold))
        painter.drawText(
            QRectF(target.x() + 10, target.y(), target.width() - 20, 34),
            Qt.AlignmentFlag.AlignVCenter,
            banner,
        )


class AFMCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(5.1, 4.2), facecolor="#0b1620")
        super().__init__(self.figure)
        self.setParent(parent)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#0b1620")
        axis.text(
            0.5,
            0.5,
            "Awaiting generated AFM",
            ha="center",
            va="center",
            color="#8fa8b8",
            fontsize=16,
        )
        axis.set_axis_off()
        self.figure.tight_layout()
        self.draw_idle()

    def update_prediction(self, result: PredictionResult) -> None:
        prediction = result.prediction
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        image = axis.imshow(
            prediction.height_nm,
            cmap="afmhot",
            extent=(0.0, 1.0, 1.0, 0.0),
            interpolation="bilinear",
        )
        axis.set(
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=(
                f"Generated AFM · frame {result.job.event.frame_index}\n"
                f"Rq {prediction.rq.value:.2f} nm · "
                f"model confidence {prediction.model_confidence * 100:.0f}%"
            ),
        )
        colorbar = self.figure.colorbar(
            image, ax=axis, fraction=0.048, pad=0.04
        )
        colorbar.set_label("Relative height (nm)")
        for item in (
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
            colorbar.ax.yaxis.label,
            *colorbar.ax.get_yticklabels(),
        ):
            item.set_color("#dfeaf0")
        axis.set_facecolor("#0b1620")
        colorbar.outline.set_edgecolor("#6f8796")
        self.figure.tight_layout()
        self.draw_idle()


class TrendCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(6.6, 2.4), facecolor="#0b1620")
        super().__init__(self.figure)
        self.setParent(parent)
        self.times: list[float] = []
        self.rq: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.confidence: list[float] = []
        self.refresh()

    def clear_data(self) -> None:
        self.times.clear()
        self.rq.clear()
        self.lower.clear()
        self.upper.clear()
        self.confidence.clear()
        self.refresh()

    def append(self, result: PredictionResult) -> None:
        prediction = result.prediction
        self.times.append(result.job.event_time_seconds)
        self.rq.append(prediction.rq.value)
        self.lower.append(prediction.rq.interval_lower)
        self.upper.append(prediction.rq.interval_upper)
        self.confidence.append(prediction.model_confidence)
        self.refresh()

    def refresh(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#0b1620")
        if self.times:
            times = np.asarray(self.times)
            rq = np.asarray(self.rq)
            axis.plot(times, rq, color="#99b7c7", linewidth=1.5, zorder=1)
            lower = np.asarray(self.lower)
            upper = np.asarray(self.upper)
            axis.fill_between(
                times,
                lower,
                upper,
                color="#5b7890",
                alpha=0.18,
                label="90% empirical interval",
            )
            scatter = axis.scatter(
                times,
                rq,
                c=self.confidence,
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                s=65,
                edgecolors="#ffffff",
                linewidths=0.7,
                zorder=3,
            )
            colorbar = self.figure.colorbar(
                scatter, ax=axis, fraction=0.035, pad=0.025
            )
            colorbar.set_label("Confidence index")
            colorbar.outline.set_edgecolor("#6f8796")
            colorbar.ax.yaxis.label.set_color("#dfeaf0")
            for label in colorbar.ax.get_yticklabels():
                label.set_color("#dfeaf0")
        axis.set(
            xlabel="RHEED stream time (s)",
            ylabel="Predicted Rq (nm)",
            title="Roughness timeline · point color encodes confidence",
        )
        axis.grid(color="#426070", alpha=0.25, linewidth=0.7)
        for item in (
            axis.title,
            axis.xaxis.label,
            axis.yaxis.label,
            *axis.get_xticklabels(),
            *axis.get_yticklabels(),
        ):
            item.set_color("#dfeaf0")
        for spine in axis.spines.values():
            spine.set_color("#486070")
        self.figure.tight_layout()
        self.draw_idle()


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        caption = QLabel(title)
        caption.setObjectName("metricCaption")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.detail = QLabel("")
        self.detail.setObjectName("metricDetail")
        layout.addWidget(caption)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)


class RealtimeMainWindow(QMainWindow):
    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self.config = dict(config)
        self.repository = Path(config["repository_root"]).resolve()
        self.entries: dict[str, list[VideoEntry]] = {}
        self.selection: ReplaySelection | None = None
        self.replay_worker: ReplayWorker | None = None
        self.prediction_worker: PredictionWorker | None = None
        self.recorder: SessionRecorder | None = None
        self._model_ready = False
        self._build_ui()
        self._load_catalog()
        self._start_prediction_worker()

    def _build_ui(self) -> None:
        self.setWindowTitle("MorphMBE · RHEED → AFM 实时形貌监测")
        self.resize(1540, 980)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(11)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("MorphMBE  实时表面形貌监测")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "双 ROI / 自动旋转顶点 · M15b causal R3D + 角覆盖 TTA · "
            "M12a 非检索式 AFM 生成"
        )
        subtitle.setObjectName("appSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)
        self.model_badge = QLabel("模型加载中")
        self.model_badge.setObjectName("modelBadge")
        header.addWidget(self.model_badge)
        outer.addLayout(header)

        controls = QFrame()
        controls.setObjectName("controlBar")
        control_layout = QGridLayout(controls)
        control_layout.setContentsMargins(12, 9, 12, 9)
        self.sample_combo = QComboBox()
        self.video_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("原始视频 · 模拟实时回放", "video")
        self.mode_combo.addItem("工业相机 · 接口预留", "camera")
        self.mode_combo.setItemData(
            1,
            "相机适配器接口已预留；本版本验证对象是 raw 视频模拟流。",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.speed = QDoubleSpinBox()
        self.speed.setRange(1.0, 4.0)
        self.speed.setSingleStep(0.1)
        self.speed.setSuffix("× 时长")
        self.speed.setValue(
            float(self.config.get("default_playback_duration_ratio", 1.67))
        )
        self.start_button = QPushButton("开始分析与回放")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        for column, (label, widget) in enumerate(
            (
                ("输入模式", self.mode_combo),
                ("样品编号", self.sample_combo),
                ("原始视频", self.video_combo),
                ("回放速度", self.speed),
            )
        ):
            box = QVBoxLayout()
            caption = QLabel(label)
            caption.setObjectName("controlCaption")
            box.addWidget(caption)
            box.addWidget(widget)
            control_layout.addLayout(box, 0, column)
        control_layout.addWidget(self.start_button, 0, 4)
        control_layout.addWidget(self.pause_button, 0, 5)
        control_layout.addWidget(self.stop_button, 0, 6)
        control_layout.setColumnStretch(2, 2)
        outer.addWidget(controls)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setObjectName("panel")
        left_layout = QVBoxLayout(left)
        left_header = QHBoxLayout()
        rheed_title = QLabel("RHEED 视频流")
        rheed_title.setObjectName("panelTitle")
        self.stream_state = QLabel("未启动")
        self.stream_state.setObjectName("stateBadge")
        left_header.addWidget(rheed_title)
        left_header.addStretch(1)
        left_header.addWidget(self.stream_state)
        left_layout.addLayout(left_header)
        self.video_canvas = VideoCanvas()
        left_layout.addWidget(self.video_canvas, 1)
        self.roi_note = QLabel(
            "青色实框：真正送入 M15b/M12a 的完整点阵裁剪；"
            "内部 tracking ROI 只定位旋转顶点，不参与生成"
        )
        self.roi_note.setObjectName("note")
        left_layout.addWidget(self.roi_note)

        right = QFrame()
        right.setObjectName("panel")
        right_layout = QVBoxLayout(right)
        result_title = QLabel("生成式 AFM 预测")
        result_title.setObjectName("panelTitle")
        right_layout.addWidget(result_title)
        self.afm_canvas = AFMCanvas()
        right_layout.addWidget(self.afm_canvas, 1)
        metrics = QGridLayout()
        self.rq_card = MetricCard("粗糙度 Rq", "— nm")
        self.fsmi_card = MetricCard("表面形貌复杂度 FSMI", "— nm")
        self.confidence_card = MetricCard("预测置信度（误差相关）", "— %")
        metrics.addWidget(self.rq_card, 0, 0)
        metrics.addWidget(self.fsmi_card, 0, 1)
        metrics.addWidget(self.confidence_card, 0, 2)
        right_layout.addLayout(metrics)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("model confidence  %p%")
        right_layout.addWidget(self.confidence_bar)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([780, 690])
        outer.addWidget(split, 5)

        lower = QSplitter(Qt.Orientation.Horizontal)
        chart_frame = QFrame()
        chart_frame.setObjectName("panel")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(8, 6, 8, 6)
        self.trend = TrendCanvas()
        chart_layout.addWidget(self.trend)
        terminal_frame = QFrame()
        terminal_frame.setObjectName("terminalFrame")
        terminal_layout = QVBoxLayout(terminal_frame)
        terminal_layout.setContentsMargins(8, 6, 8, 6)
        terminal_title = QLabel("PIPELINE LOG")
        terminal_title.setObjectName("terminalTitle")
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(400)
        self.terminal.setObjectName("terminal")
        terminal_layout.addWidget(terminal_title)
        terminal_layout.addWidget(self.terminal)
        lower.addWidget(chart_frame)
        lower.addWidget(terminal_frame)
        lower.setSizes([900, 560])
        outer.addWidget(lower, 2)

        self.sample_combo.currentTextChanged.connect(self._update_videos)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.start_button.clicked.connect(self.start_session)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_session)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#071018; color:#dfeaf0; }
            #appTitle { font-size:25px; font-weight:700; color:#f3f8fa; }
            #appSubtitle { color:#8ea7b6; font-size:12px; }
            #modelBadge, #stateBadge {
                background:#173140; border:1px solid #2b6171;
                border-radius:11px; padding:5px 11px; color:#86e8cb;
            }
            #controlBar, #panel, #metricCard {
                background:#0b1620; border:1px solid #203644;
                border-radius:9px;
            }
            #controlCaption, #metricCaption { color:#7f99a9; font-size:10px; }
            #panelTitle { font-size:15px; font-weight:650; }
            #metricValue { font-size:21px; font-weight:700; color:#f5fafc; }
            #metricDetail, #note { color:#829aa9; font-size:10px; }
            QComboBox, QDoubleSpinBox {
                background:#10232e; border:1px solid #31505f;
                border-radius:5px; padding:6px 8px; min-height:22px;
            }
            QPushButton {
                background:#173140; border:1px solid #31505f;
                border-radius:6px; padding:8px 13px;
            }
            QPushButton:hover { background:#214657; }
            QPushButton:disabled { color:#566873; background:#101c24; }
            #primaryButton {
                background:#087f6d; border-color:#19a78f; font-weight:650;
            }
            #primaryButton:hover { background:#0a927c; }
            #terminalFrame { background:#050a0e; border:1px solid #203644; }
            #terminalTitle { color:#49d6ad; font-family:Menlo; font-size:10px; }
            #terminal {
                background:#050a0e; color:#a9c7b9; border:0;
                font-family:Menlo; font-size:10px;
            }
            QProgressBar {
                background:#10232e; border:1px solid #31505f;
                border-radius:5px; text-align:center; min-height:18px;
            }
            QProgressBar::chunk { background:#28b88f; border-radius:4px; }
            """
        )

    def _load_catalog(self) -> None:
        excluded = read_removelist(
            self.repository / self.config["removelist_path"]
        )
        entries = discover_videos(
            self.repository / self.config["raw_video_root"],
            excluded_sample_ids=excluded,
        )
        self.entries = group_by_sample(entries)
        self.sample_combo.clear()
        self.sample_combo.addItems(sorted(self.entries))
        if "6063" in self.entries:
            self.sample_combo.setCurrentText("6063")
        self.append_log(
            f"发现 {len(entries)} 个 raw 视频、{len(self.entries)} 个可用样品；"
            f"removelist 排除 {len(excluded)} 个编号"
        )

    def _update_videos(self, sample_id: str) -> None:
        self.video_combo.clear()
        for entry in self.entries.get(str(sample_id), []):
            self.video_combo.addItem(entry.label, entry)

    def _mode_changed(self, index: int) -> None:
        camera = self.mode_combo.itemData(index) == "camera"
        self.start_button.setEnabled(self._model_ready and not camera)
        if camera:
            self.append_log(
                "工业相机 FrameSource 接口已预留；当前验证版本使用 raw 视频模拟流"
            )

    def _start_prediction_worker(self) -> None:
        self.prediction_worker = PredictionWorker(
            bundle_path=self.repository / self.config["deployment_bundle"],
            device=str(self.config.get("prediction_device", "auto")),
        )
        self.prediction_worker.log.connect(self.append_log)
        self.prediction_worker.ready.connect(self._model_loaded)
        self.prediction_worker.result.connect(self._prediction_ready)
        self.prediction_worker.failed.connect(self._worker_failed)
        self.prediction_worker.start()

    def _model_loaded(self, model_id: str) -> None:
        self._model_ready = True
        self.model_badge.setText("M15b + M12a · READY")
        self.model_badge.setToolTip(model_id)
        self.start_button.setEnabled(
            self.mode_combo.currentData() == "video"
        )
        self.append_log(f"模型就绪：{model_id}")

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.terminal.appendPlainText(f"[{stamp}] {message}")

    def start_session(self) -> None:
        entry = self.video_combo.currentData()
        if not isinstance(entry, VideoEntry):
            QMessageBox.warning(self, "没有输入", "请选择一个原始 RHEED 视频。")
            return
        self.stop_session(silent=True)
        ratio = float(self.speed.value())
        self.selection = None
        self.video_canvas._selection = None
        self.afm_canvas.show_placeholder()
        self.trend.clear_data()
        self.rq_card.value.setText("— nm")
        self.fsmi_card.value.setText("— nm")
        self.confidence_card.value.setText("— %")
        self.confidence_bar.setValue(0)
        result_root = self.repository / self.config["result_root"]
        self.recorder = SessionRecorder(
            result_root,
            sample_id=entry.sample_id,
            source=entry.path,
            playback_ratio=ratio,
        )
        self.append_log(
            f"启动样品 {entry.sample_id}：{entry.path.name}，目标时长 {ratio:.2f}×"
        )
        self.replay_worker = ReplayWorker(
            sample_id=entry.sample_id,
            source=entry.path,
            config=self.config,
            playback_ratio=ratio,
        )
        self.replay_worker.log.connect(self.append_log)
        self.replay_worker.prepared.connect(self._replay_prepared)
        self.replay_worker.frame.connect(self.video_canvas.set_frame)
        self.replay_worker.prediction_job.connect(self._submit_prediction)
        self.replay_worker.completed.connect(self._replay_completed)
        self.replay_worker.failed.connect(self._worker_failed)
        self.replay_worker.start()
        self.stream_state.setText("ANALYZING")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def _replay_prepared(
        self,
        selection: ReplaySelection,
        fps: float,
    ) -> None:
        self.selection = selection
        self.video_canvas.set_selection(selection)
        duration = selection.frame_count / max(fps, 1e-6)
        slowed = duration * float(self.speed.value())
        self.stream_state.setText("STREAMING")
        self.append_log(
            f"源视频 {fps:.2f} fps / {duration:.1f} s；"
            f"模拟回放约 {slowed:.1f} s；预测事件 {len(selection.events)} 个"
        )
        if self.recorder is not None:
            self.recorder.record_selection(selection, fps=fps)

    def _submit_prediction(self, job: PredictionJob) -> None:
        if self.prediction_worker is not None:
            accepted = self.prediction_worker.submit(job)
            if not accepted:
                self.append_log(
                    f"帧 {job.event.frame_index}: 推理队列已满，"
                    "实时节流跳过该旋转周期"
                )

    def _prediction_ready(self, result: PredictionResult) -> None:
        prediction = result.prediction
        self.afm_canvas.update_prediction(result)
        self.trend.append(result)
        self.rq_card.value.setText(f"{prediction.rq.value:.2f} nm")
        self.rq_card.detail.setText(
            f"预计 |误差| {prediction.rq.expected_absolute_error:.2f} nm  ·  "
            f"区间 [{prediction.rq.interval_lower:.2f}, "
            f"{prediction.rq.interval_upper:.2f}]"
            + ("  ·  SUPPORT CLIP" if prediction.rq.support_clipped else "")
        )
        self.fsmi_card.value.setText(f"{prediction.fsmi.value:.2f} nm")
        self.fsmi_card.detail.setText(
            f"预计 |误差| {prediction.fsmi.expected_absolute_error:.2f} nm  ·  "
            f"区间 [{prediction.fsmi.interval_lower:.2f}, "
            f"{prediction.fsmi.interval_upper:.2f}]"
            + ("  ·  SUPPORT CLIP" if prediction.fsmi.support_clipped else "")
        )
        percent = int(round(prediction.model_confidence * 100))
        self.confidence_card.value.setText(f"{percent}%")
        self.confidence_card.detail.setText(
            f"角覆盖+TTA {min(prediction.rq.tta_confidence, prediction.fsmi.tta_confidence) * 100:.0f}%  ·  "
            f"多头一致 {min(prediction.rq.head_agreement_confidence, prediction.fsmi.head_agreement_confidence) * 100:.0f}%  ·  "
            f"保守综合 {prediction.combined_confidence * 100:.0f}%"
        )
        self.confidence_bar.setValue(percent)
        if percent < 40:
            chunk = "#d45d5d"
        elif percent < 70:
            chunk = "#d5a947"
        else:
            chunk = "#28b88f"
        self.confidence_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background:{chunk}; border-radius:4px; }}"
        )
        saved = self.recorder.record(result) if self.recorder else None
        self.append_log(
            f"预测完成 frame={result.job.event.frame_index} · "
            f"Rq={prediction.rq.value:.2f} nm · "
            f"FSMI={prediction.fsmi.value:.2f} nm · "
            f"confidence={percent}% · "
            f"latency={prediction.inference_seconds:.2f}s"
        )
        if saved is not None:
            self.append_log(f"派生结果已保存：{saved}")

    def toggle_pause(self) -> None:
        if self.replay_worker is None:
            return
        paused = self.pause_button.text() == "暂停"
        self.replay_worker.set_paused(paused)
        self.pause_button.setText("继续" if paused else "暂停")
        self.stream_state.setText("PAUSED" if paused else "STREAMING")

    def stop_session(self, *, silent: bool = False) -> None:
        if self.replay_worker is not None:
            if self.replay_worker.isRunning():
                self.replay_worker.stop()
                self.replay_worker.wait(3000)
            self.replay_worker = None
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(
            self._model_ready and self.mode_combo.currentData() == "video"
        )
        self.stream_state.setText("STOPPED")
        if not silent:
            self.append_log("回放已停止；raw 数据未被修改")

    def _replay_completed(self) -> None:
        self.stream_state.setText("VIDEO COMPLETE")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(self._model_ready)
        self.append_log("视频回放结束；已提交的形貌预测将继续完成")

    def _worker_failed(self, message: str) -> None:
        self.append_log(f"ERROR · {message}")
        self.stream_state.setText("ERROR")
        self.start_button.setEnabled(self._model_ready)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        QMessageBox.critical(self, "实时管线错误", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_session(silent=True)
        if self.prediction_worker is not None:
            self.prediction_worker.stop()
            self.prediction_worker.wait(5000)
        super().closeEvent(event)
