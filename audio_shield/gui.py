#!/usr/bin/env python3
"""
CineCast Audio Shield — 交互界面 (GUI)

PyQt6 + pyqtgraph 界面：
- 文件列表（带状态标记）
- 波形展示与噪音点标记
- 播放控制与精准定位
- 滑动选择与删除
"""

import logging
import os
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 延迟导入：仅在实际运行 GUI 时才需要 PyQt6 / pyqtgraph
# ---------------------------------------------------------------------------
_QT_AVAILABLE = False
try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QLabel,
        QSlider,
        QFileDialog,
        QMessageBox,
        QSplitter,
        QStatusBar,
        QProgressBar,
    )

    import pyqtgraph as pg  # type: ignore

    _QT_AVAILABLE = True
except ImportError:
    pass

from audio_shield.scanner import AudioScanner, FileStatus, AudioFileInfo
from audio_shield.analyzer import detect_audio_glitches
from audio_shield.editor import AudioBufferManager


def _check_qt():
    """确保 PyQt6 可用，否则给出友好提示。"""
    if not _QT_AVAILABLE:
        raise RuntimeError(
            "PyQt6 and pyqtgraph are required for the GUI. "
            "Install them with: pip install PyQt6 pyqtgraph"
        )


# ========================== 后台分析线程 ==========================


class AnalysisWorker(QThread if _QT_AVAILABLE else object):
    """后台线程：逐文件运行噪音检测。"""

    if _QT_AVAILABLE:
        progress = pyqtSignal(int, int)            # (current, total)
        file_done = pyqtSignal(str, list)           # (file_path, glitches)
        all_done = pyqtSignal()

    def __init__(self, files: List[AudioFileInfo], sensitivity: float = 0.4):
        if _QT_AVAILABLE:
            super().__init__()
        self.files = files
        self.sensitivity = sensitivity

    def run(self):
        total = len(self.files)
        for i, finfo in enumerate(self.files):
            try:
                glitches = detect_audio_glitches(
                    finfo.file_path, sensitivity=self.sensitivity
                )
            except Exception as exc:
                logger.error(f"分析失败: {finfo.file_path} — {exc}")
                glitches = []
            self.file_done.emit(finfo.file_path, glitches)
            self.progress.emit(i + 1, total)
        self.all_done.emit()


# ========================== 主窗口 ==========================


class AudioShieldWindow(QMainWindow if _QT_AVAILABLE else object):
    """CineCast Audio Shield 主窗口"""

    def __init__(self):
        _check_qt()
        super().__init__()
        self.setWindowTitle("CineCast Audio Shield — 音频盾")
        self.setMinimumSize(1100, 700)

        self.scanner: Optional[AudioScanner] = None
        self.editor = AudioBufferManager()
        self._current_file: Optional[AudioFileInfo] = None
        self._current_glitches: List[float] = []
        self._selection_start: Optional[float] = None
        self._selection_end: Optional[float] = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---- 左侧：文件列表 ----
        left = QVBoxLayout()
        self.btn_open_folder = QPushButton("📂 打开文件夹")
        self.btn_scan = QPushButton("🔍 开始扫描分析")
        self.btn_scan.setEnabled(False)
        self.file_list = QListWidget()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left.addWidget(self.btn_open_folder)
        left.addWidget(self.btn_scan)
        left.addWidget(self.file_list)
        left.addWidget(self.progress_bar)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(320)

        # ---- 右侧：波形 + 控件 ----
        right = QVBoxLayout()

        # 波形图
        self.waveform_plot = pg.PlotWidget(title="波形")
        self.waveform_plot.setLabel("bottom", "时间", units="s")
        self.waveform_plot.setLabel("left", "振幅")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        right.addWidget(self.waveform_plot, stretch=3)

        # 信息标签
        self.label_info = QLabel("请先打开一个文件夹，然后点击「开始扫描分析」")
        right.addWidget(self.label_info)

        # 播放控制
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_prev_glitch = QPushButton("⬅ 上一个噪音点")
        self.btn_next_glitch = QPushButton("➡ 下一个噪音点")
        ctrl_layout.addWidget(self.btn_prev_glitch)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_next_glitch)
        right.addLayout(ctrl_layout)

        # 编辑控制
        edit_layout = QHBoxLayout()
        self.btn_delete = QPushButton("🗑️ 删除选中区域")
        self.btn_undo = QPushButton("↩️ 撤销")
        self.btn_normalize = QPushButton("📊 归一化")
        self.btn_save = QPushButton("💾 保存")
        edit_layout.addWidget(self.btn_delete)
        edit_layout.addWidget(self.btn_undo)
        edit_layout.addWidget(self.btn_normalize)
        edit_layout.addWidget(self.btn_save)
        right.addLayout(edit_layout)

        right_widget = QWidget()
        right_widget.setLayout(right)

        # 拼装
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪")

    # --------------------------------------------------------- 信号连接
    def _connect_signals(self):
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.btn_scan.clicked.connect(self._on_start_scan)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.btn_delete.clicked.connect(self._on_delete_selection)
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_normalize.clicked.connect(self._on_normalize)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_prev_glitch.clicked.connect(self._on_prev_glitch)
        self.btn_next_glitch.clicked.connect(self._on_next_glitch)

        # 波形拖动选择
        self.waveform_plot.scene().sigMouseClicked.connect(self._on_waveform_click)

    # --------------------------------------------------------- 槽函数
    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹")
        if folder:
            self.scanner = AudioScanner(folder)
            self.scanner.scan()
            self._refresh_file_list()
            self.btn_scan.setEnabled(True)
            self.statusBar().showMessage(f"已扫描 {len(self.scanner.files)} 个文件")

    def _on_start_scan(self):
        if not self.scanner or not self.scanner.files:
            return
        self.btn_scan.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.scanner.files))

        self._worker = AnalysisWorker(self.scanner.files)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.file_done.connect(self._on_file_analyzed)
        self._worker.all_done.connect(self._on_analysis_complete)
        self._worker.start()

    def _on_analysis_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self.statusBar().showMessage(f"分析中 {current}/{total}...")

    def _on_file_analyzed(self, file_path: str, glitches: list):
        if glitches:
            self.scanner.update_status(
                file_path, FileStatus.NEEDS_FIX, glitches
            )
        else:
            self.scanner.update_status(file_path, FileStatus.PASSED)
        self._refresh_file_list()

    def _on_analysis_complete(self):
        self.progress_bar.setVisible(False)
        self.btn_scan.setEnabled(True)
        needs_fix = len(self.scanner.get_needs_fix_files())
        self.statusBar().showMessage(f"分析完成！{needs_fix} 个文件需要修复")

    def _on_file_selected(self, row: int):
        if row < 0 or not self.scanner:
            return
        finfo = self.scanner.files[row]
        self._current_file = finfo
        self._current_glitches = list(finfo.glitches)
        self._load_waveform(finfo.file_path)
        self.label_info.setText(
            f"文件: {finfo.filename} | 状态: {finfo.status.value} | "
            f"噪音点: {len(finfo.glitches)}"
        )

    def _on_delete_selection(self):
        if self._selection_start is None or self._selection_end is None:
            QMessageBox.information(self, "提示", "请先在波形图上选择一段区域")
            return
        start = min(self._selection_start, self._selection_end)
        end = max(self._selection_start, self._selection_end)
        self.editor.delete_range(start, end)
        self._draw_waveform_from_editor()
        self._selection_start = None
        self._selection_end = None
        self.statusBar().showMessage(f"已删除 [{start:.3f}s - {end:.3f}s]")

    def _on_undo(self):
        if self.editor.undo():
            self._draw_waveform_from_editor()
            self.statusBar().showMessage("已撤销")
        else:
            self.statusBar().showMessage("没有可撤销的操作")

    def _on_normalize(self):
        self.editor.normalize()
        self._draw_waveform_from_editor()
        self.statusBar().showMessage("已归一化")

    def _on_save(self):
        if not self._current_file:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", self._current_file.filename, "MP3 (*.mp3);;WAV (*.wav)"
        )
        if path:
            fmt = "wav" if path.lower().endswith(".wav") else "mp3"
            self.editor.save_result(path, file_format=fmt)
            self.statusBar().showMessage(f"已保存: {path}")

    def _on_prev_glitch(self):
        self._jump_to_glitch(-1)

    def _on_next_glitch(self):
        self._jump_to_glitch(1)

    def _on_waveform_click(self, event):
        """记录波形图上的点击位置用于选择区域"""
        pos = event.scenePos()
        mouse_point = self.waveform_plot.plotItem.vb.mapSceneToView(pos)
        time_sec = mouse_point.x()
        if self._selection_start is None:
            self._selection_start = time_sec
            self.statusBar().showMessage(f"选择起点: {time_sec:.3f}s — 再次点击设置终点")
        else:
            self._selection_end = time_sec
            start = min(self._selection_start, self._selection_end)
            end = max(self._selection_start, self._selection_end)
            self._draw_selection(start, end)
            self.statusBar().showMessage(
                f"已选择: [{start:.3f}s - {end:.3f}s]  点击「删除选中区域」确认"
            )

    # --------------------------------------------------------- 内部方法
    def _refresh_file_list(self):
        self.file_list.clear()
        if not self.scanner:
            return
        for finfo in self.scanner.files:
            self.file_list.addItem(repr(finfo))

    def _load_waveform(self, file_path: str):
        """加载音频文件并绘制波形"""
        self.editor.load(file_path)
        self._draw_waveform_from_editor()

    def _draw_waveform_from_editor(self):
        """从 editor 中的当前音频绘制波形"""
        self.waveform_plot.clear()
        audio = self.editor.audio
        if len(audio) == 0:
            return

        # 转为 numpy 数组
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        # 归一化到 [-1, 1]
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak

        sr = audio.frame_rate
        time_axis = np.arange(len(samples)) / sr

        # 降采样以加速绘制（最多显示 50000 个点）
        max_points = 50000
        if len(samples) > max_points:
            step = len(samples) // max_points
            samples = samples[::step]
            time_axis = time_axis[::step]

        self.waveform_plot.plot(time_axis, samples, pen=pg.mkPen("c", width=1))

        # 绘制噪音标记
        for t in self._current_glitches:
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen("r", width=2))
            self.waveform_plot.addItem(line)

    def _draw_selection(self, start: float, end: float):
        """在波形图上绘制选择区域高亮"""
        region = pg.LinearRegionItem(
            values=[start, end],
            brush=pg.mkBrush(255, 0, 0, 50),
            movable=False,
        )
        self.waveform_plot.addItem(region)

    def _jump_to_glitch(self, direction: int):
        """跳转到上一个/下一个噪音点"""
        if not self._current_glitches:
            self.statusBar().showMessage("没有噪音点")
            return

        # 根据当前视图中心确定位置
        view_range = self.waveform_plot.viewRange()
        current_center = (view_range[0][0] + view_range[0][1]) / 2

        if direction > 0:
            candidates = [t for t in self._current_glitches if t > current_center + 0.1]
            target = candidates[0] if candidates else self._current_glitches[0]
        else:
            candidates = [t for t in self._current_glitches if t < current_center - 0.1]
            target = candidates[-1] if candidates else self._current_glitches[-1]

        # 设置视图到噪音点前 1 秒
        pre_listen = max(0, target - 1.0)
        window = 5.0  # 显示 5 秒窗口
        self.waveform_plot.setXRange(pre_listen, pre_listen + window)
        self.statusBar().showMessage(f"跳转到噪音点: {target:.3f}s")


def launch_gui():
    """启动 Audio Shield GUI"""
    _check_qt()
    app = QApplication(sys.argv)
    window = AudioShieldWindow()
    window.show()
    sys.exit(app.exec())
