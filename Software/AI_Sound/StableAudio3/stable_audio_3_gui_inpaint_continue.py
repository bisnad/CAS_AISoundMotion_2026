"""
stableaudio3_gui_inpaint_continue.py

GUI for inpainting and continuation, with model size/variant switching.
ALL Stable Audio 3-specific logic lives in stable_audio_3_core.py -- this
file contains ONLY GUI code. Compare generate_inpaint_or_continue() in
that module directly against hello_stable_audio_3.py's Sections 3/4.
"""

import sys
import random

import numpy as np
import torch
import torchaudio
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QSize, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QRadioButton,
    QButtonGroup, QProgressBar, QStatusBar, QScrollArea, QSizePolicy,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None

import stable_audio_3_core as sa3


MAX_PLOT_POINTS = 4000
REGION_BRUSH_NORMAL = (100, 150, 255, 80)
REGION_BRUSH_SELECTED = (255, 170, 40, 140)
PLAYHEAD_POLL_INTERVAL_MS = 30
WAVEFORM_MIN_HEIGHT = 120
WAVEFORM_PREFERRED_WIDTH = 400


def tensor_to_playback_numpy(audio_tensor):
    arr = audio_tensor.detach().cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(arr.T)


def compute_minmax_envelope(mono_samples, target_points):
    n = len(mono_samples)
    if n <= target_points:
        return None
    bucket_size = max(1, n // target_points)
    n_buckets = n // bucket_size
    trimmed = mono_samples[: n_buckets * bucket_size].reshape(n_buckets, bucket_size)
    mins = trimmed.min(axis=1)
    maxs = trimmed.max(axis=1)
    y = np.empty(n_buckets * 2, dtype=np.float32)
    y[0::2] = mins
    y[1::2] = maxs
    return y, bucket_size


# ============================================================================
# QThread workers -- thin wrappers around stable_audio_3_core
# ============================================================================

class LoadModelWorker(QThread):
    finished = pyqtSignal(object, object, int, int, str, str)
    failed = pyqtSignal(str)

    def __init__(self, model_variant, device):
        super().__init__()
        self.model_variant = model_variant
        self.device = device

    def run(self):
        try:
            model, conditioner, sample_rate, io_channels, device = sa3.load_model(self.model_variant, self.device)
            self.finished.emit(model, conditioner, sample_rate, io_channels, device, self.model_variant)
        except Exception as exc:
            self.failed.emit(str(exc))


class InpaintWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, int)
    failed = pyqtSignal(str)

    def __init__(self, generate_kwargs, sample_rate):
        super().__init__()
        self.generate_kwargs = generate_kwargs
        self.sample_rate = sample_rate

    def run(self):
        try:
            num_regions = len(self.generate_kwargs["regions"])
            if num_regions > 1:
                self.progress.emit(f"Running {num_regions} sequential inpainting passes ...")
            audio = sa3.generate_inpaint_or_continue(**self.generate_kwargs)
            self.finished.emit(audio, self.sample_rate)
        except Exception as exc:
            self.failed.emit(str(exc))


# ============================================================================
# Waveform widget (GUI-only)
# ============================================================================

class WaveformView(pg.PlotWidget):
    region_created = pyqtSignal(object)
    regions_changed = pyqtSignal()
    region_selected = pyqtSignal(object)
    keep_from_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setBackground("w")
        self.showGrid(x=True, y=False)
        self.setLabel("bottom", "Time", units="s")
        self.setLabel("left", "Amplitude")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(WAVEFORM_MIN_HEIGHT)
        self._preferred_size = QSize(WAVEFORM_PREFERRED_WIDTH, WAVEFORM_MIN_HEIGHT)

        self.duration_sec = 0.0
        self.regions = []
        self.selected_region = None
        self.mode = "inpaint"
        self.keep_from_line = None

        self._drag_temp_region = None
        self._drag_start_x = None

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(color=(20, 20, 20), width=2))
        self.playhead_line.setZValue(1000)
        self.playhead_line.setVisible(False)
        self.addItem(self.playhead_line, ignoreBounds=True)

        self._install_drag_override()

    def sizeHint(self):
        return self._preferred_size

    def minimumSizeHint(self):
        return self._preferred_size

    def set_waveform(self, audio_tensor, sample_rate):
        self.clear()
        self.regions = []
        self.selected_region = None
        self.keep_from_line = None

        mono = audio_tensor.mean(dim=0).cpu().numpy().astype(np.float32)
        n = len(mono)
        self.duration_sec = n / sample_rate

        envelope = compute_minmax_envelope(mono, MAX_PLOT_POINTS)
        if envelope is None:
            x = np.arange(n) / sample_rate
            y = mono
        else:
            y, bucket_size = envelope
            bucket_duration = bucket_size / sample_rate
            x = np.repeat(np.arange(len(y) // 2) * bucket_duration, 2)

        self.plot(x, y, pen=pg.mkPen(color=(30, 100, 200), width=1))
        self.setXRange(0, self.duration_sec, padding=0.02)

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(color=(20, 20, 20), width=2))
        self.playhead_line.setZValue(1000)
        self.playhead_line.setVisible(False)
        self.addItem(self.playhead_line, ignoreBounds=True)

    def show_playhead(self, position_sec):
        self.playhead_line.setPos(position_sec)
        self.playhead_line.setVisible(True)

    def hide_playhead(self):
        self.playhead_line.setVisible(False)

    def set_mode(self, mode):
        self.mode = mode
        for region in self.regions:
            region.setVisible(mode == "inpaint")
        if self.keep_from_line is not None:
            self.keep_from_line.setVisible(mode == "continue")

    def ensure_keep_from_line(self, initial_pos):
        if self.keep_from_line is None:
            self.keep_from_line = pg.InfiniteLine(
                pos=initial_pos, angle=90, movable=True,
                pen=pg.mkPen(color=(200, 30, 30), width=2), bounds=[0, self.duration_sec],
            )
            self.keep_from_line.sigPositionChangeFinished.connect(
                lambda: self.keep_from_changed.emit(self.keep_from_line.value())
            )
            self.addItem(self.keep_from_line, ignoreBounds=True)
        self.keep_from_line.setVisible(self.mode == "continue")
        return self.keep_from_line

    def _install_drag_override(self):
        view_box = self.getPlotItem().vb
        original_drag_event = view_box.mouseDragEvent

        def custom_drag_event(ev, axis=None):
            if self.mode != "inpaint" or ev.button() != Qt.LeftButton:
                original_drag_event(ev, axis=axis)
                return
            ev.accept()
            start_x = view_box.mapToView(ev.buttonDownPos()).x()
            current_x = view_box.mapToView(ev.pos()).x()
            if ev.isStart():
                self._drag_start_x = start_x
                self._drag_temp_region = pg.LinearRegionItem(values=[start_x, current_x], brush=pg.mkBrush(100, 200, 100, 80))
                self.addItem(self._drag_temp_region, ignoreBounds=True)
            if self._drag_temp_region is not None:
                lo, hi = sorted([self._drag_start_x, current_x])
                lo, hi = max(0.0, lo), min(self.duration_sec, hi)
                self._drag_temp_region.setRegion([lo, hi])
            if ev.isFinish():
                width = 0.0
                if self._drag_temp_region is not None:
                    lo, hi = self._drag_temp_region.getRegion()
                    width = hi - lo
                    self.removeItem(self._drag_temp_region)
                    self._drag_temp_region = None
                if width >= 0.05:
                    self.add_region(lo, hi)
                else:
                    self.set_selected_region(None)

        view_box.mouseDragEvent = custom_drag_event

    def _bind_region_click_handler(self, region):
        original_click_event = region.mouseClickEvent

        def on_click(ev):
            if ev.button() == Qt.LeftButton:
                self.set_selected_region(region)
                ev.accept()
            else:
                original_click_event(ev)

        region.mouseClickEvent = on_click

    def set_selected_region(self, region):
        if self.selected_region is not None:
            self.selected_region.setBrush(pg.mkBrush(*REGION_BRUSH_NORMAL))
            self.selected_region.update()
        self.selected_region = region
        if region is not None:
            region.setBrush(pg.mkBrush(*REGION_BRUSH_SELECTED))
            region.update()
        self.region_selected.emit(region)

    def add_region(self, start_sec, end_sec):
        region = pg.LinearRegionItem(values=[start_sec, end_sec], brush=pg.mkBrush(*REGION_BRUSH_NORMAL), bounds=[0, self.duration_sec])
        region.setZValue(10)
        region.sigRegionChangeFinished.connect(self.regions_changed.emit)
        self._bind_region_click_handler(region)
        self.addItem(region, ignoreBounds=True)
        self.regions.append(region)
        self.region_created.emit(region)
        self.regions_changed.emit()
        self.set_selected_region(region)
        return region

    def remove_region(self, region):
        self.removeItem(region)
        if region in self.regions:
            self.regions.remove(region)
        if self.selected_region is region:
            self.selected_region = None
            self.region_selected.emit(None)
        self.regions_changed.emit()

    def remove_selected_region(self):
        if self.selected_region is not None:
            self.remove_region(self.selected_region)

    def clear_regions(self):
        for region in list(self.regions):
            self.removeItem(region)
        self.regions = []
        self.selected_region = None
        self.region_selected.emit(None)
        self.regions_changed.emit()

    def get_region_bounds(self):
        return [tuple(r.getRegion()) for r in self.regions]


# ============================================================================
# Main window
# ============================================================================

class InpaintContinueGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Audio 3 -- Inpainting / Continuation GUI")

        self.model = None
        self.conditioner = None
        self.model_device = None
        self.current_model_variant = None
        self.sample_rate = None
        self.io_channels = 2

        self.source_audio_tensor = None
        self.source_duration_sec = 0.0
        self.result_audio_tensor = None

        self.load_worker = None
        self.run_worker = None

        self._playhead_view = None
        self._playhead_offset_sec = 0.0
        self._playhead_stream_time_at_start = 0.0
        self._playhead_segment_duration_sec = 0.0
        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(PLAYHEAD_POLL_INTERVAL_MS)
        self._playhead_timer.timeout.connect(self._on_playhead_tick)

        self._build_ui()
        self._populate_playback_devices()
        self._reload_model()

    def _build_ui(self):
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.addWidget(self._build_model_selection_group())
        outer.addWidget(self._build_source_group())
        outer.addWidget(self._build_waveform_group())
        outer.addWidget(self._build_mode_group())
        outer.addWidget(self._build_params_group())
        outer.addWidget(self._build_playback_group())
        outer.addLayout(self._build_action_row())
        outer.addWidget(self._build_result_waveform_group())

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        outer.addWidget(self.progress_bar)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content)
        self.setCentralWidget(scroll_area)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Starting up...")
        self.resize(900, 1000)

    def _build_model_selection_group(self):
        group = QGroupBox("Model selection")
        outer = QVBoxLayout()

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self.size_group = QButtonGroup(self)
        self.size_small_radio = QRadioButton("small")
        self.size_medium_radio = QRadioButton("medium")
        self.size_small_radio.setChecked(True)
        self.size_group.addButton(self.size_small_radio)
        self.size_group.addButton(self.size_medium_radio)
        self.size_small_radio.toggled.connect(self._on_model_selection_changed)
        self.size_medium_radio.toggled.connect(self._on_model_selection_changed)
        size_row.addWidget(self.size_small_radio)
        size_row.addWidget(self.size_medium_radio)
        outer.addLayout(size_row)

        self.small_domain_widget = QWidget()
        domain_row = QHBoxLayout(self.small_domain_widget)
        domain_row.setContentsMargins(0, 0, 0, 0)
        domain_row.addWidget(QLabel("Small domain:"))
        self.domain_group = QButtonGroup(self)
        self.domain_sfx_radio = QRadioButton("sfx")
        self.domain_music_radio = QRadioButton("music")
        self.domain_sfx_radio.setChecked(True)
        self.domain_group.addButton(self.domain_sfx_radio)
        self.domain_group.addButton(self.domain_music_radio)
        self.domain_sfx_radio.toggled.connect(self._on_model_selection_changed)
        self.domain_music_radio.toggled.connect(self._on_model_selection_changed)
        domain_row.addWidget(self.domain_sfx_radio)
        domain_row.addWidget(self.domain_music_radio)
        outer.addWidget(self.small_domain_widget)

        variant_row = QHBoxLayout()
        variant_row.addWidget(QLabel("Variant:"))
        self.variant_group = QButtonGroup(self)
        self.variant_posttrained_radio = QRadioButton("post-trained (8 steps, no CFG needed)")
        self.variant_base_radio = QRadioButton("base (~50 steps, CFG meaningful)")
        self.variant_posttrained_radio.setChecked(True)
        self.variant_group.addButton(self.variant_posttrained_radio)
        self.variant_group.addButton(self.variant_base_radio)
        self.variant_posttrained_radio.toggled.connect(self._on_model_selection_changed)
        self.variant_base_radio.toggled.connect(self._on_model_selection_changed)
        variant_row.addWidget(self.variant_posttrained_radio)
        variant_row.addWidget(self.variant_base_radio)
        outer.addLayout(variant_row)

        self.resolved_variant_label = QLabel("Resolved model: --")
        self.resolved_variant_label.setStyleSheet("font-weight: bold;")
        outer.addWidget(self.resolved_variant_label)

        reload_btn = QPushButton("Reload model now")
        reload_btn.clicked.connect(self._reload_model)
        outer.addWidget(reload_btn)

        group.setLayout(outer)
        return group

    def _build_source_group(self):
        group = QGroupBox("Source audio")
        layout = QVBoxLayout()
        row = QHBoxLayout()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.source_path_edit.setPlaceholderText("Load a .wav file to inpaint or continue")
        browse_btn = QPushButton("Load .wav...")
        browse_btn.clicked.connect(self._on_browse_source)
        row.addWidget(self.source_path_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)
        self.source_duration_label = QLabel("Duration: -- s")
        layout.addWidget(self.source_duration_label)
        play_source_btn = QPushButton("Play full source audio")
        play_source_btn.clicked.connect(self._on_play_source_audio)
        layout.addWidget(play_source_btn)
        group.setLayout(layout)
        return group

    def _build_waveform_group(self):
        group = QGroupBox(
            "Source waveform -- left-drag empty space to create a region / "
            "click a region to select it / drag red line to set continuation point"
        )
        layout = QVBoxLayout()
        self.waveform_view = WaveformView()
        layout.addWidget(self.waveform_view)
        self.waveform_view.region_selected.connect(self._on_region_selected)
        self.waveform_view.keep_from_changed.connect(self._on_keep_from_changed)

        btn_row = QHBoxLayout()
        self.play_region_btn = QPushButton("Play selected region")
        self.play_region_btn.clicked.connect(self._on_play_selected_region)
        self.play_region_btn.setEnabled(False)
        self.remove_region_btn = QPushButton("Remove selected region")
        self.remove_region_btn.clicked.connect(self._on_remove_selected_region)
        self.remove_region_btn.setEnabled(False)
        clear_regions_btn = QPushButton("Clear all regions")
        clear_regions_btn.clicked.connect(self._on_clear_regions)
        btn_row.addWidget(self.play_region_btn)
        btn_row.addWidget(self.remove_region_btn)
        btn_row.addWidget(clear_regions_btn)
        layout.addLayout(btn_row)
        group.setLayout(layout)
        return group

    def _build_result_waveform_group(self):
        group = QGroupBox("Generated result waveform -- left-drag to select a region to preview/play")
        layout = QVBoxLayout()
        self.result_waveform_view = WaveformView()
        self.result_waveform_view.set_mode("inpaint")
        layout.addWidget(self.result_waveform_view)

        btn_row = QHBoxLayout()
        play_full_result_btn = QPushButton("Play full result")
        play_full_result_btn.clicked.connect(self._on_play_result_audio)
        clear_result_regions_btn = QPushButton("Clear regions")
        clear_result_regions_btn.clicked.connect(self.result_waveform_view.clear_regions)
        self.save_btn = QPushButton("Save result...")
        self.save_btn.clicked.connect(self._on_save_result)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(play_full_result_btn)
        btn_row.addWidget(clear_result_regions_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)
        group.setLayout(layout)
        return group

    def _build_mode_group(self):
        group = QGroupBox("Mode")
        outer = QVBoxLayout()
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.inpaint_radio = QRadioButton("Inpaint region(s)")
        self.inpaint_radio.setChecked(True)
        self.continue_radio = QRadioButton("Continue clip")
        self.mode_group.addButton(self.inpaint_radio)
        self.mode_group.addButton(self.continue_radio)
        self.inpaint_radio.toggled.connect(self._on_mode_toggled)
        mode_row.addWidget(self.inpaint_radio)
        mode_row.addWidget(self.continue_radio)
        outer.addLayout(mode_row)

        self.continue_widget = QWidget()
        continue_form = QFormLayout(self.continue_widget)
        self.keep_from_label = QLabel("Kept from: -- s (drag the red line in the waveform)")
        continue_form.addRow(self.keep_from_label)
        self.continue_total_duration_spin = QDoubleSpinBox()
        self.continue_total_duration_spin.setRange(1.0, 100000.0)
        self.continue_total_duration_spin.setSingleStep(1.0)
        self.continue_total_duration_spin.setValue(20.0)
        continue_form.addRow("Total duration (s):", self.continue_total_duration_spin)
        outer.addWidget(self.continue_widget)
        self.continue_widget.setVisible(False)
        self._keep_from_sec = 0.0

        group.setLayout(outer)
        return group

    def _build_params_group(self):
        group = QGroupBox("Generation parameters")
        layout = QFormLayout()

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText("footsteps on gravel")
        self.prompt_edit.setFixedHeight(50)
        layout.addRow("Prompt:", self.prompt_edit)

        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setFixedHeight(50)
        self.negative_prompt_edit.setPlaceholderText("Optional. Leave empty to omit.")
        layout.addRow("Negative prompt:", self.negative_prompt_edit)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(8)
        layout.addRow("Steps:", self.steps_spin)

        cfg_row = QHBoxLayout()
        self.cfg_scale_checkbox = QCheckBox("Use CFG scale")
        self.cfg_scale_checkbox.setChecked(False)
        self.cfg_scale_checkbox.toggled.connect(lambda checked: self.cfg_scale_spin.setEnabled(checked))
        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(0.0, 15.0)
        self.cfg_scale_spin.setSingleStep(0.5)
        self.cfg_scale_spin.setValue(1.0)
        self.cfg_scale_spin.setEnabled(False)
        cfg_row.addWidget(self.cfg_scale_checkbox)
        cfg_row.addWidget(self.cfg_scale_spin)
        layout.addRow("CFG scale:", cfg_row)

        self.cfg_scale_hint_label = QLabel("")
        self.cfg_scale_hint_label.setWordWrap(True)
        self.cfg_scale_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addRow("", self.cfg_scale_hint_label)

        seed_row = QHBoxLayout()
        self.seed_fixed_checkbox = QCheckBox("Fixed seed")
        self.seed_fixed_checkbox.setChecked(True)
        self.seed_fixed_checkbox.toggled.connect(lambda checked: self.seed_spin.setEnabled(checked))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        seed_row.addWidget(self.seed_fixed_checkbox)
        seed_row.addWidget(self.seed_spin)
        layout.addRow("Seed:", seed_row)

        group.setLayout(layout)
        return group

    def _build_playback_group(self):
        group = QGroupBox("Playback device")
        layout = QFormLayout()
        self.device_combo = QComboBox()
        layout.addRow("Output device:", self.device_combo)
        if sd is None:
            self.device_combo.addItem("sounddevice not installed -- playback disabled")
            self.device_combo.setEnabled(False)
        group.setLayout(layout)
        return group

    def _build_action_row(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.run_btn.setEnabled(False)
        row.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop playback")
        self.stop_btn.clicked.connect(self._on_stop_playback)
        row.addWidget(self.stop_btn)
        return row

    # ------------------------------------------------------------------
    # Model selection / loading
    # ------------------------------------------------------------------

    def _current_selection(self):
        size = "small" if self.size_small_radio.isChecked() else "medium"
        variant = "base" if self.variant_base_radio.isChecked() else "post-trained"
        small_domain = "music" if self.domain_music_radio.isChecked() else "sfx"
        return size, variant, small_domain

    def _on_model_selection_changed(self, checked=None):
        if checked is False:
            return
        size, variant, small_domain = self._current_selection()
        self.small_domain_widget.setVisible(size == "small")
        resolved = sa3.resolve_model_variant(size, variant, small_domain)
        self.resolved_variant_label.setText(f"Resolved model: {resolved}")
        self._apply_variant_defaults(variant)
        self._reload_model()

    def _apply_variant_defaults(self, variant):
        if variant == "base":
            self.steps_spin.setValue(50)
            self.cfg_scale_checkbox.setChecked(True)
            self.cfg_scale_spin.setValue(7.0)
            self.cfg_scale_hint_label.setText(
                "Base checkpoint: matches the paper's own base-model evaluation setup "
                "(50 steps, CFG scale 7). NOTE: the confirmed inpainting-clipping finding "
                "was specific to cfg_scale=7.0 on the POST-TRAINED checkpoint; this has not "
                "been separately verified for base-checkpoint inpainting/continuation."
            )
        else:
            self.steps_spin.setValue(8)
            self.cfg_scale_checkbox.setChecked(False)
            self.cfg_scale_spin.setValue(1.0)
            self.cfg_scale_hint_label.setText(
                "Post-trained checkpoint: cfg_scale=7.0 was confirmed to cause severe "
                "clipping on this checkpoint's inpainting path (both small and medium). "
                "CFG scale disabled by default."
            )

    def _reload_model(self):
        size, variant, small_domain = self._current_selection()
        model_variant = sa3.resolve_model_variant(size, variant, small_domain)
        device = sa3.detect_default_device()

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Loading '{model_variant}' on device='{device}' ...")

        self.load_worker = LoadModelWorker(model_variant, device)
        self.load_worker.finished.connect(self._on_model_loaded)
        self.load_worker.failed.connect(self._on_model_load_failed)
        self.load_worker.start()

    def _on_model_loaded(self, model, conditioner, sample_rate, io_channels, device, model_variant):
        self.model = model
        self.conditioner = conditioner
        self.sample_rate = sample_rate
        self.io_channels = io_channels
        self.model_device = device
        self.current_model_variant = model_variant
        self.progress_bar.setVisible(False)
        self._update_run_enabled()
        self.status_bar.showMessage(
            f"Model '{model_variant}' loaded on device='{device}'. sample_rate={sample_rate}"
        )

    def _on_model_load_failed(self, error_message):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Model load failed", error_message)
        self.status_bar.showMessage("Model load failed.")

    def _update_run_enabled(self):
        self.run_btn.setEnabled(self.model is not None and self.source_audio_tensor is not None)

    # ------------------------------------------------------------------
    # Source audio handling
    # ------------------------------------------------------------------

    def _on_browse_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load source audio", "", "WAV files (*.wav)")
        if not path:
            return
        if self.sample_rate is None:
            QMessageBox.warning(self, "Model not ready", "Wait for the model to finish loading first.")
            return
        try:
            audio = sa3.load_wav_matching_model(path, self.sample_rate, self.io_channels)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load audio", str(exc))
            return

        self.source_audio_tensor = audio
        self.source_duration_sec = audio.shape[1] / self.sample_rate
        self.source_path_edit.setText(path)
        self.source_duration_label.setText(f"Duration: {self.source_duration_sec:.2f} s")

        self.waveform_view.set_waveform(audio, self.sample_rate)
        self.waveform_view.set_mode("inpaint" if self.inpaint_radio.isChecked() else "continue")
        self.waveform_view.ensure_keep_from_line(self.source_duration_sec)
        self._keep_from_sec = self.source_duration_sec
        self.keep_from_label.setText(f"Kept from: {self._keep_from_sec:.2f} s (drag the red line in the waveform)")
        self.continue_total_duration_spin.setValue(max(20.0, self.source_duration_sec * 2))

        self._update_run_enabled()
        self.status_bar.showMessage(f"Loaded source audio: {path} ({self.source_duration_sec:.2f}s)")

    def _on_play_source_audio(self):
        if self.source_audio_tensor is None:
            QMessageBox.information(self, "No source audio", "Load a source .wav file first.")
            return
        duration = self.source_audio_tensor.shape[1] / self.sample_rate
        self._play_numpy_audio(
            tensor_to_playback_numpy(self.source_audio_tensor), self.sample_rate,
            view=self.waveform_view, offset_sec=0.0, duration_sec=duration,
        )

    def _on_mode_toggled(self, checked):
        mode = "inpaint" if checked else "continue"
        self.continue_widget.setVisible(mode == "continue")
        self.waveform_view.set_mode(mode)

    def _on_region_selected(self, region):
        has_selection = region is not None
        self.play_region_btn.setEnabled(has_selection)
        self.remove_region_btn.setEnabled(has_selection)

    def _on_play_selected_region(self):
        region = self.waveform_view.selected_region
        if region is None or self.source_audio_tensor is None:
            QMessageBox.information(self, "No region selected", "Click a region in the waveform first.")
            return
        start_sec, end_sec = region.getRegion()
        start_idx = max(0, int(start_sec * self.sample_rate))
        end_idx = min(self.source_audio_tensor.shape[1], int(end_sec * self.sample_rate))
        if end_idx <= start_idx:
            return
        segment = self.source_audio_tensor[:, start_idx:end_idx]
        self._play_numpy_audio(
            tensor_to_playback_numpy(segment), self.sample_rate,
            view=self.waveform_view, offset_sec=start_sec, duration_sec=end_sec - start_sec,
        )

    def _on_remove_selected_region(self):
        self.waveform_view.remove_selected_region()

    def _on_clear_regions(self):
        self.waveform_view.clear_regions()

    def _on_keep_from_changed(self, value):
        self._keep_from_sec = max(0.0, min(self.source_duration_sec, value))
        self.keep_from_label.setText(f"Kept from: {self._keep_from_sec:.2f} s (drag the red line in the waveform)")

    # ------------------------------------------------------------------
    # Playback devices and playhead tracking
    # ------------------------------------------------------------------

    def _populate_playback_devices(self):
        if sd is None:
            return
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.device_combo.addItem(f"Error listing devices: {exc}")
            return
        default_output = None
        try:
            default_output = sd.default.device[1]
        except Exception:
            pass
        for idx, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) > 0:
                self.device_combo.addItem(f"[{idx}] {dev['name']} ({dev['max_output_channels']} ch)", userData=idx)
        if default_output is not None:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == default_output:
                    self.device_combo.setCurrentIndex(i)
                    break

    def _selected_output_device_index(self):
        if sd is None or self.device_combo.count() == 0:
            return None
        return self.device_combo.currentData()

    def _play_numpy_audio(self, numpy_audio, sample_rate, view=None, offset_sec=0.0, duration_sec=0.0):
        if sd is None:
            QMessageBox.warning(self, "Playback unavailable", "The 'sounddevice' package is not installed.")
            return
        device_index = self._selected_output_device_index()
        try:
            sd.stop()
            self._stop_playhead_tracking()
            sd.play(numpy_audio, samplerate=sample_rate, device=device_index)
            if view is not None:
                self._start_playhead_tracking(view, offset_sec, duration_sec)
        except Exception as exc:
            QMessageBox.critical(self, "Playback failed", str(exc))

    def _start_playhead_tracking(self, view, offset_sec, duration_sec):
        try:
            stream = sd.get_stream()
            stream_time_now = stream.time if stream is not None else 0.0
        except Exception:
            stream_time_now = 0.0
        self._playhead_view = view
        self._playhead_offset_sec = offset_sec
        self._playhead_stream_time_at_start = stream_time_now
        self._playhead_segment_duration_sec = duration_sec
        view.show_playhead(offset_sec)
        self._playhead_timer.start()

    def _stop_playhead_tracking(self):
        self._playhead_timer.stop()
        if self._playhead_view is not None:
            self._playhead_view.hide_playhead()
        self._playhead_view = None

    def _on_playhead_tick(self):
        if self._playhead_view is None:
            return
        try:
            stream = sd.get_stream()
            stream_time_now = stream.time if stream is not None else None
        except Exception:
            stream_time_now = None
        if stream_time_now is None:
            self._stop_playhead_tracking()
            return
        elapsed = stream_time_now - self._playhead_stream_time_at_start
        if elapsed >= self._playhead_segment_duration_sec:
            self._stop_playhead_tracking()
            return
        self._playhead_view.show_playhead(self._playhead_offset_sec + elapsed)

    def _on_play_result_audio(self):
        if self.result_audio_tensor is None:
            QMessageBox.information(self, "No result", "Run inpainting/continuation first.")
            return
        duration = self.result_audio_tensor.shape[1] / self.sample_rate
        self._play_numpy_audio(
            tensor_to_playback_numpy(self.result_audio_tensor), self.sample_rate,
            view=self.result_waveform_view, offset_sec=0.0, duration_sec=duration,
        )

    def _on_stop_playback(self):
        if sd is not None:
            sd.stop()
        self._stop_playhead_tracking()

    # ------------------------------------------------------------------
    # Run (inpaint or continue) -- reads widget state, delegates to core
    # ------------------------------------------------------------------

    def _current_seed(self):
        if self.seed_fixed_checkbox.isChecked():
            return int(self.seed_spin.value())
        return random.randint(0, 2**31 - 1)

    def _on_run_clicked(self):
        if self.model is None or self.source_audio_tensor is None:
            QMessageBox.warning(self, "Not ready", "Load the model and a source audio file first.")
            return

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Enter a text prompt before running.")
            return

        try:
            if self.inpaint_radio.isChecked():
                regions = self.waveform_view.get_region_bounds()
                if not regions:
                    QMessageBox.warning(self, "No regions", "Left-drag on the source waveform to create a region.")
                    return
                for start, end in regions:
                    if start < 0 or end > self.source_duration_sec:
                        raise ValueError(f"Region [{start:.2f}, {end:.2f}] must lie within [0, {self.source_duration_sec:.2f}].")
                duration = self.source_duration_sec
            else:
                keep_from = self._keep_from_sec
                total_duration = float(self.continue_total_duration_spin.value())
                if total_duration <= self.source_duration_sec:
                    raise ValueError(
                        f"Total duration ({total_duration:.2f}s) must exceed source duration "
                        f"({self.source_duration_sec:.2f}s)."
                    )
                regions = [(keep_from, total_duration)]
                duration = total_duration
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        negative_prompt = self.negative_prompt_edit.toPlainText().strip() or None
        cfg_scale = float(self.cfg_scale_spin.value()) if self.cfg_scale_checkbox.isChecked() else None

        generate_kwargs = dict(
            model=self.model,
            device=self.model_device,
            source_audio_tensor=self.source_audio_tensor,
            source_sample_rate=self.sample_rate,
            regions=regions,
            prompt=prompt,
            duration=duration,
            steps=int(self.steps_spin.value()),
            seed=self._current_seed(),
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
        )

        self.status_bar.showMessage(f"Running with model='{self.current_model_variant}' ...")
        self.progress_bar.setVisible(True)
        self.run_btn.setEnabled(False)

        self.run_worker = InpaintWorker(generate_kwargs, self.sample_rate)
        self.run_worker.progress.connect(self.status_bar.showMessage)
        self.run_worker.finished.connect(self._on_run_finished)
        self.run_worker.failed.connect(self._on_run_failed)
        self.run_worker.start()

    def _on_run_finished(self, audio_tensor, sample_rate):
        self.result_audio_tensor = audio_tensor
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.result_waveform_view.set_waveform(audio_tensor, sample_rate)
        self.result_waveform_view.set_mode("inpaint")
        self.status_bar.showMessage("Done.")

    def _on_run_failed(self, error_message):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Run failed", error_message)
        self.status_bar.showMessage("Run failed.")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _on_save_result(self):
        if self.result_audio_tensor is None:
            QMessageBox.information(self, "No result", "Run inpainting/continuation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result audio", "result.wav", "WAV files (*.wav)")
        if not path:
            return
        try:
            sa3.save_wav(path, self.result_audio_tensor, self.sample_rate)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status_bar.showMessage(f"Saved result to {path}")

    def closeEvent(self, event):
        self._stop_playhead_tracking()
        if sd is not None:
            sd.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = InpaintContinueGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
