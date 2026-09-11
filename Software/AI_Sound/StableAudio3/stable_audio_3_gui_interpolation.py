"""
stableaudio3_gui_interpolation.py

GUI for text-embedding interpolation, with model size/variant switching.
ALL Stable Audio 3-specific logic (encoding, mask-aware blending, the
conditioner.forward patch/restore pipeline) lives in
stable_audio_3_core.py's generate_interpolated() -- this file contains
ONLY GUI code. Compare that function directly against
hello_stable_audio_3.py's Section 5; they are the same logic.
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
    QButtonGroup, QProgressBar, QStatusBar, QScrollArea, QSizePolicy, QSlider,
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


class InterpolationWorker(QThread):
    finished = pyqtSignal(object, int, dict)
    failed = pyqtSignal(str)

    def __init__(self, generate_kwargs, sample_rate):
        super().__init__()
        self.generate_kwargs = generate_kwargs
        self.sample_rate = sample_rate

    def run(self):
        try:
            audio, stats = sa3.generate_interpolated(**self.generate_kwargs)
            self.finished.emit(audio, self.sample_rate, stats)
        except Exception as exc:
            self.failed.emit(str(exc))


# ============================================================================
# Waveform widget (GUI-only)
# ============================================================================

class WaveformView(pg.PlotWidget):
    region_created = pyqtSignal(object)
    regions_changed = pyqtSignal()
    region_selected = pyqtSignal(object)

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

    def _install_drag_override(self):
        view_box = self.getPlotItem().vb
        original_drag_event = view_box.mouseDragEvent

        def custom_drag_event(ev, axis=None):
            if ev.button() != Qt.LeftButton:
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

    def clear_regions(self):
        for region in list(self.regions):
            self.removeItem(region)
        self.regions = []
        self.selected_region = None
        self.region_selected.emit(None)
        self.regions_changed.emit()

    def get_selected_bounds(self):
        if self.selected_region is None:
            return None
        return self.selected_region.getRegion()


# ============================================================================
# Main window
# ============================================================================

class InterpolationGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Audio 3 -- Text Embedding Interpolation GUI")

        self.model = None
        self.conditioner = None
        self.model_device = None
        self.current_model_variant = None
        self.sample_rate = None

        self.result_audio_tensor = None

        self.load_worker = None
        self.interp_worker = None

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
        outer.addWidget(self._build_prompts_group())
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
        self.resize(760, 850)

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

        warning_label = QLabel(
            "Note: the conditioner.forward patch used for interpolation was empirically "
            "verified against 'small-sfx' specifically. The paper describes the same "
            "conditioning architecture as shared across all sizes/variants, so switching "
            "models should continue to work, but this has not been separately re-verified "
            "for 'medium' or any '-base' checkpoint."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #a05a00; font-size: 11px;")
        outer.addWidget(warning_label)

        reload_btn = QPushButton("Reload model now")
        reload_btn.clicked.connect(self._reload_model)
        outer.addWidget(reload_btn)

        group.setLayout(outer)
        return group

    def _build_prompts_group(self):
        group = QGroupBox("Prompts")
        layout = QFormLayout()

        self.prompt_a_edit = QLineEdit("a violin sound")
        layout.addRow("Prompt A:", self.prompt_a_edit)

        self.prompt_b_edit = QLineEdit("a frog sound")
        layout.addRow("Prompt B:", self.prompt_b_edit)

        self.negative_prompt_edit = QLineEdit()
        self.negative_prompt_edit.setPlaceholderText("Optional. Leave empty to omit.")
        layout.addRow("Negative prompt:", self.negative_prompt_edit)

        interp_row = QHBoxLayout()
        self.interp_slider = QSlider(Qt.Horizontal)
        self.interp_slider.setRange(0, 100)
        self.interp_slider.setValue(50)
        self.interp_slider.valueChanged.connect(self._on_interp_slider_changed)
        self.interp_spin = QDoubleSpinBox()
        self.interp_spin.setRange(0.0, 1.0)
        self.interp_spin.setSingleStep(0.01)
        self.interp_spin.setValue(0.50)
        self.interp_spin.setToolTip("0.0 = pure prompt A, 1.0 = pure prompt B.")
        self.interp_spin.valueChanged.connect(self._on_interp_spin_changed)
        interp_row.addWidget(QLabel("A"))
        interp_row.addWidget(self.interp_slider)
        interp_row.addWidget(QLabel("B"))
        interp_row.addWidget(self.interp_spin)
        layout.addRow("Interpolation factor:", interp_row)

        self.blend_stats_label = QLabel("Blend stats: -- (run once to see mask-aware blend breakdown)")
        self.blend_stats_label.setWordWrap(True)
        self.blend_stats_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addRow("", self.blend_stats_label)

        group.setLayout(layout)
        return group

    def _build_params_group(self):
        group = QGroupBox("Generation parameters")
        layout = QFormLayout()

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(8)
        layout.addRow("Steps:", self.steps_spin)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 380.0)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setValue(10.0)
        layout.addRow("Duration (sec):", self.duration_spin)

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
        self.generate_btn = QPushButton("Generate interpolated audio")
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        self.generate_btn.setEnabled(False)
        row.addWidget(self.generate_btn)
        return row

    def _build_result_waveform_group(self):
        group = QGroupBox("Interpolated result waveform -- left-drag to select a region to preview/play")
        layout = QVBoxLayout()
        self.result_waveform_view = WaveformView()
        layout.addWidget(self.result_waveform_view)

        btn_row = QHBoxLayout()
        self.play_result_btn = QPushButton("Play result (region if selected, else full)")
        self.play_result_btn.clicked.connect(self._on_play_result)
        self.play_result_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop playback")
        self.stop_btn.clicked.connect(self._on_stop_playback)
        clear_regions_btn = QPushButton("Clear regions")
        clear_regions_btn.clicked.connect(self.result_waveform_view.clear_regions)
        self.save_btn = QPushButton("Save result...")
        self.save_btn.clicked.connect(self._on_save_result)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.play_result_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(clear_regions_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        group.setLayout(layout)
        return group

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
            self.cfg_scale_hint_label.setText("Base checkpoint: 50 steps, CFG scale 7 (paper's own evaluation setup).")
        else:
            self.steps_spin.setValue(8)
            self.cfg_scale_checkbox.setChecked(False)
            self.cfg_scale_spin.setValue(1.0)
            self.cfg_scale_hint_label.setText("Post-trained checkpoint: does not rely on CFG at inference.")

    def _reload_model(self):
        size, variant, small_domain = self._current_selection()
        model_variant = sa3.resolve_model_variant(size, variant, small_domain)
        device = sa3.detect_default_device()

        self.generate_btn.setEnabled(False)
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
        self.model_device = device
        self.current_model_variant = model_variant
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_bar.showMessage(f"Model '{model_variant}' loaded on device='{device}'. sample_rate={sample_rate}")

    def _on_model_load_failed(self, error_message):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Model load failed", error_message)
        self.status_bar.showMessage("Model load failed.")

    # ------------------------------------------------------------------
    # Interpolation factor slider <-> spinbox sync
    # ------------------------------------------------------------------

    def _on_interp_slider_changed(self, value):
        self.interp_spin.blockSignals(True)
        self.interp_spin.setValue(value / 100.0)
        self.interp_spin.blockSignals(False)

    def _on_interp_spin_changed(self, value):
        self.interp_slider.blockSignals(True)
        self.interp_slider.setValue(int(round(value * 100)))
        self.interp_slider.blockSignals(False)

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

    def _on_play_result(self):
        if self.result_audio_tensor is None:
            QMessageBox.information(self, "No result", "Generate interpolated audio first.")
            return
        bounds = self.result_waveform_view.get_selected_bounds()
        if bounds is None:
            segment, offset_sec = self.result_audio_tensor, 0.0
            duration_sec = segment.shape[1] / self.sample_rate
        else:
            start_sec, end_sec = bounds
            start_idx = max(0, int(start_sec * self.sample_rate))
            end_idx = min(self.result_audio_tensor.shape[1], int(end_sec * self.sample_rate))
            if end_idx <= start_idx:
                return
            segment = self.result_audio_tensor[:, start_idx:end_idx]
            offset_sec, duration_sec = start_sec, end_sec - start_sec
        self._play_numpy_audio(
            tensor_to_playback_numpy(segment), self.sample_rate,
            view=self.result_waveform_view, offset_sec=offset_sec, duration_sec=duration_sec,
        )

    def _on_stop_playback(self):
        if sd is not None:
            sd.stop()
        self._stop_playhead_tracking()

    # ------------------------------------------------------------------
    # Generation -- reads widget state, delegates to core via the worker
    # ------------------------------------------------------------------

    def _current_seed(self):
        if self.seed_fixed_checkbox.isChecked():
            return int(self.seed_spin.value())
        return random.randint(0, 2**31 - 1)

    def _on_generate_clicked(self):
        if self.model is None or self.conditioner is None:
            QMessageBox.warning(self, "Model not ready", "Wait for the model to finish loading first.")
            return

        prompt_a = self.prompt_a_edit.text().strip()
        prompt_b = self.prompt_b_edit.text().strip()
        if not prompt_a or not prompt_b:
            QMessageBox.warning(self, "Missing prompts", "Enter both Prompt A and Prompt B.")
            return

        negative_prompt = self.negative_prompt_edit.text().strip() or None
        cfg_scale = float(self.cfg_scale_spin.value()) if self.cfg_scale_checkbox.isChecked() else None

        generate_kwargs = dict(
            model=self.model,
            conditioner=self.conditioner,
            device=self.model_device,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            interp_factor=float(self.interp_spin.value()),
            duration=float(self.duration_spin.value()),
            steps=int(self.steps_spin.value()),
            seed=self._current_seed(),
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
        )

        self.status_bar.showMessage(
            f"Interpolating '{prompt_a}' <-> '{prompt_b}' at factor={self.interp_spin.value():.2f} ..."
        )
        self.progress_bar.setVisible(True)
        self.generate_btn.setEnabled(False)

        self.interp_worker = InterpolationWorker(generate_kwargs, self.sample_rate)
        self.interp_worker.finished.connect(self._on_interpolation_finished)
        self.interp_worker.failed.connect(self._on_interpolation_failed)
        self.interp_worker.start()

    def _on_interpolation_finished(self, audio_tensor, sample_rate, stats):
        self.result_audio_tensor = audio_tensor
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.play_result_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.result_waveform_view.set_waveform(audio_tensor, sample_rate)
        self.blend_stats_label.setText(
            f"Blend stats: {stats['blended']} positions blended, {stats['only_a']} kept from A only, "
            f"{stats['only_b']} kept from B only, {stats['padding']} shared padding."
        )
        self.status_bar.showMessage("Interpolated generation complete.")

    def _on_interpolation_failed(self, error_message):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Generation failed", error_message)
        self.status_bar.showMessage("Interpolated generation failed.")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _on_save_result(self):
        if self.result_audio_tensor is None:
            QMessageBox.information(self, "No result", "Generate interpolated audio first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result audio", "interpolated.wav", "WAV files (*.wav)")
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
    window = InterpolationGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
