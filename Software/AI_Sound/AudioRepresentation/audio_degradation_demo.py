#!/usr/bin/env python3
"""
Audio Degradation Demo — Aliasing & Bit-Depth Reduction
=========================================================

Interactive PyQt5 GUI demonstrating:
  1. Aliasing from sample-rate reduction (sample-and-hold, no anti-alias filter)
  2. Quantization distortion from bit-depth reduction

Sources: synthetic sine, or an audio file (via soundfile).
Plots: waveform (time domain) and spectrum (frequency domain).
"""

import sys
import threading
import numpy as np

import sounddevice as sd
import soundfile as sf

from PyQt5 import QtWidgets, QtCore

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --------------------------------------------------------------------------
# DSP helpers
# --------------------------------------------------------------------------

def generate_sine(freq, duration, fs, amplitude=0.6):
    t = np.arange(int(duration * fs)) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def apply_sample_rate_reduction(x, factor):
    """Sample-and-hold decimation without anti-aliasing filter."""
    factor = max(1, int(round(factor)))
    if factor == 1:
        return x.copy()
    n = len(x)
    idx = (np.arange(n) // factor) * factor
    idx = np.clip(idx, 0, n - 1)
    return x[idx]


def apply_bit_depth_reduction(x, bits):
    """Uniform quantization of a [-1, 1] signal to 2**bits levels."""
    bits = max(1, min(24, int(round(bits))))
    levels = 2 ** bits
    x_clipped = np.clip(x, -1.0, 1.0)
    q = np.round((x_clipped + 1.0) * 0.5 * (levels - 1))
    return q / (levels - 1) * 2.0 - 1.0


def process_audio(x, sr_factor, bits):
    y = apply_sample_rate_reduction(x, sr_factor)
    return apply_bit_depth_reduction(y, bits)


# --------------------------------------------------------------------------
# Audio playback engine
# --------------------------------------------------------------------------

class AudioEngine(QtCore.QObject):
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.fs = 44100
        self.source = np.zeros(0, dtype=np.float64)
        self.playback_buffer = np.zeros(0, dtype=np.float64)
        self.lock = threading.Lock()
        self.pos = 0
        self.stream = None
        self.device_index = None
        self.sr_factor = 1
        self.bits = 16

    def set_device(self, device_index):
        self.device_index = device_index

    def load_source(self, data, fs):
        data = data.astype(np.float64)
        new_buf = process_audio(data, self.sr_factor, self.bits) if len(data) else np.zeros(0)
        with self.lock:
            self.source = data
            self.fs = fs
            self.pos = 0
            self.playback_buffer = new_buf

    def update_params(self, sr_factor, bits):
        # Heavy DSP happens outside the lock; only the swap is locked.
        with self.lock:
            src = self.source
        new_buf = process_audio(src, sr_factor, bits) if len(src) else np.zeros(0)
        with self.lock:
            if src is self.source:  # source unchanged meanwhile
                self.sr_factor = sr_factor
                self.bits = bits
                self.playback_buffer = new_buf

    def get_snapshot(self):
        """Return (source, processed, fs, playhead) consistently."""
        with self.lock:
            return self.source, self.playback_buffer, self.fs, self.pos

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            buf = self.playback_buffer
            n = len(buf)
            if n == 0:
                outdata[:] = 0
                return
            out = np.empty(frames, dtype=np.float64)
            filled = 0
            pos = self.pos
            while filled < frames:  # loop seamlessly
                take = min(frames - filled, n - pos)
                out[filled:filled + take] = buf[pos:pos + take]
                filled += take
                pos += take
                if pos >= n:
                    pos = 0
            self.pos = pos
            outdata[:, 0] = out

    def start(self):
        self.stop()
        if len(self.playback_buffer) == 0:
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=self.fs,
                channels=1,
                dtype="float32",
                device=self.device_index,
                callback=self._callback,
                blocksize=1024,
            )
            self.stream.start()
        except Exception as e:
            self.stream = None
            self.error_occurred.emit(str(e))

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            self.pos = 0


# --------------------------------------------------------------------------
# Waveform + Spectrum plot widget
# --------------------------------------------------------------------------

class ScopeCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6.0))
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_wave = self.fig.add_subplot(211)
        self.ax_wave.set_xlabel("Time (ms)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_title("Waveform (original vs. degraded)")
        self.wave_orig, = self.ax_wave.plot([], [], color="#999999", lw=1.0, label="Original")
        self.wave_proc, = self.ax_wave.plot([], [], color="#d62728", lw=1.2, label="Degraded")
        self.ax_wave.legend(loc="upper right", fontsize=8)
        self.ax_wave.set_ylim(-1.05, 1.05)

        self.ax_spec = self.fig.add_subplot(212)
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Magnitude (dB)")
        self.ax_spec.set_title("Spectrum (original vs. degraded)")
        self.spec_orig, = self.ax_spec.plot([], [], color="#999999", lw=1.0, label="Original")
        self.spec_proc, = self.ax_spec.plot([], [], color="#d62728", lw=1.2, label="Degraded")
        self.nyq_line = self.ax_spec.axvline(0, color="#1f77b4", ls="--", lw=1.0, label="New Nyquist")
        self.ax_spec.legend(loc="upper right", fontsize=8)
        self.ax_spec.set_ylim(-100, 5)

        self.fig.tight_layout()  # once; axes limits change but layout does not

    def update_waveform(self, orig, proc, fs, window_ms=30.0):
        n = int(fs * window_ms / 1000.0)
        n = max(2, min(n, len(orig), len(proc)))
        t_ms = np.arange(n) / fs * 1000.0
        self.wave_orig.set_data(t_ms, orig[:n])
        self.wave_proc.set_data(t_ms, proc[:n])
        self.ax_wave.set_xlim(0, t_ms[-1] if n > 1 else window_ms)

    def update_spectrum(self, orig, proc, fs, new_nyquist):
        def spectrum(x):
            if len(x) < 2:
                return np.array([0.0]), np.array([-100.0])
            n = min(len(x), 16384)
            xw = x[:n] * np.hanning(n)
            spec = np.fft.rfft(xw)
            mag_db = 20 * np.log10(np.abs(spec) / (n / 2) + 1e-12)
            return np.fft.rfftfreq(n, 1 / fs), mag_db

        f1, m1 = spectrum(orig)
        f2, m2 = spectrum(proc)
        self.spec_orig.set_data(f1, m1)
        self.spec_proc.set_data(f2, m2)
        self.nyq_line.set_xdata([new_nyquist, new_nyquist])
        self.ax_spec.set_xlim(0, fs / 2)

    def redraw(self):
        self.draw_idle()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    SINE_DURATION = 3.0
    NATIVE_FS = 44100
    WAVE_WINDOW_MS = 30.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Degradation Demo — Aliasing & Bit-Depth")
        self.resize(950, 850)

        self.engine = AudioEngine()
        self.engine.error_occurred.connect(self.on_engine_error)

        self.loaded_file_data = None
        self.current_mode = "sine"
        self.sine_freq = 440.0
        self._params_dirty = False
        self._device_indices = []

        self._build_ui()
        self._populate_devices()
        self._on_source_changed()

        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(80)
        self.update_timer.timeout.connect(self.refresh_plots)
        self.update_timer.start()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # --- Source selection ---
        src_group = QtWidgets.QGroupBox("Audio Source")
        src_layout = QtWidgets.QHBoxLayout(src_group)

        self.radio_sine = QtWidgets.QRadioButton("Synthetic sine wave")
        self.radio_file = QtWidgets.QRadioButton("External audio file")
        self.radio_sine.setChecked(True)
        self.radio_sine.toggled.connect(self._on_source_changed)
        src_layout.addWidget(self.radio_sine)
        src_layout.addWidget(self.radio_file)

        self.sine_freq_spin = QtWidgets.QDoubleSpinBox()
        self.sine_freq_spin.setRange(20, 20000)
        self.sine_freq_spin.setValue(440)
        self.sine_freq_spin.setSuffix(" Hz")
        self.sine_freq_spin.valueChanged.connect(self._on_sine_freq_changed)
        src_layout.addWidget(QtWidgets.QLabel("Sine freq:"))
        src_layout.addWidget(self.sine_freq_spin)

        self.load_file_btn = QtWidgets.QPushButton("Load file…")
        self.load_file_btn.clicked.connect(self._on_load_file)
        src_layout.addWidget(self.load_file_btn)
        self.file_label = QtWidgets.QLabel("(no file loaded)")
        self.file_label.setStyleSheet("color: gray;")
        src_layout.addWidget(self.file_label, 1)
        main_layout.addWidget(src_group)

        # --- Output device ---
        dev_group = QtWidgets.QGroupBox("Audio Output Device")
        dev_layout = QtWidgets.QHBoxLayout(dev_group)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        dev_layout.addWidget(self.device_combo, 1)
        self.refresh_devices_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_devices_btn.clicked.connect(self._populate_devices)
        dev_layout.addWidget(self.refresh_devices_btn)
        main_layout.addWidget(dev_group)

        # --- Degradation controls ---
        deg_group = QtWidgets.QGroupBox("Degradation Controls")
        deg_layout = QtWidgets.QGridLayout(deg_group)

        deg_layout.addWidget(QtWidgets.QLabel("Sample-rate reduction:"), 0, 0)
        self.sr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sr_slider.setRange(1, 64)
        self.sr_slider.setValue(1)
        self.sr_slider.valueChanged.connect(self._on_params_changed)
        deg_layout.addWidget(self.sr_slider, 0, 1)
        self.sr_value_label = QtWidgets.QLabel()
        deg_layout.addWidget(self.sr_value_label, 0, 2)

        deg_layout.addWidget(QtWidgets.QLabel("Bit resolution:"), 1, 0)
        self.bit_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bit_slider.setRange(2, 16)
        self.bit_slider.setValue(16)
        self.bit_slider.valueChanged.connect(self._on_params_changed)
        deg_layout.addWidget(self.bit_slider, 1, 1)
        self.bit_value_label = QtWidgets.QLabel()
        deg_layout.addWidget(self.bit_value_label, 1, 2)

        # Layout fix: slider column takes all spare space, label column has a
        # fixed width wide enough for the longest possible text, so label
        # changes can never resize the sliders.
        deg_layout.setColumnStretch(0, 0)
        deg_layout.setColumnStretch(1, 1)
        deg_layout.setColumnStretch(2, 0)
        fm = self.sr_value_label.fontMetrics()
        longest = max(
            (fm.horizontalAdvance(self._sr_text(f)) for f in range(1, 65)),
            default=0,
        )
        longest = max(longest, fm.horizontalAdvance(self._bit_text(16)))
        label_w = longest + 16
        for lbl in (self.sr_value_label, self.bit_value_label):
            lbl.setFixedWidth(label_w)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        label_w0 = fm.horizontalAdvance("Sample-rate reduction:") + 8
        deg_layout.setColumnMinimumWidth(0, label_w0)

        main_layout.addWidget(deg_group)

        # --- Transport ---
        transport_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play)
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        transport_layout.addWidget(self.play_btn)
        transport_layout.addWidget(self.stop_btn)
        transport_layout.addStretch(1)

        transport_layout.addWidget(QtWidgets.QLabel("Waveform window:"))
        self.wave_window_spin = QtWidgets.QSpinBox()
        self.wave_window_spin.setRange(2, 200)
        self.wave_window_spin.setValue(int(self.WAVE_WINDOW_MS))
        self.wave_window_spin.setSuffix(" ms")
        self.wave_window_spin.valueChanged.connect(self._on_wave_window_changed)
        transport_layout.addWidget(self.wave_window_spin)
        main_layout.addLayout(transport_layout)

        # --- Plots ---
        self.canvas = ScopeCanvas(self)
        main_layout.addWidget(self.canvas, 1)

        self.status = self.statusBar()
        self.status.showMessage("Ready.")
        self._update_param_labels()

    # ----------------------------------------------------------- devices --
    def _populate_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self._device_indices = []
        try:
            devices = sd.query_devices()
        except Exception as e:
            self.status.showMessage(f"Could not query devices: {e}")
            self.device_combo.blockSignals(False)
            return
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0:
                label = f"{i}: {d['name']} ({d['max_output_channels']} ch, {int(d['default_samplerate'])} Hz)"
                self.device_combo.addItem(label)
                self._device_indices.append(i)
        if default_out in self._device_indices:
            self.device_combo.setCurrentIndex(self._device_indices.index(default_out))
        elif self._device_indices:
            self.device_combo.setCurrentIndex(0)
        self.device_combo.blockSignals(False)
        self._on_device_changed()

    def _on_device_changed(self, *_):
        if not self._device_indices:
            return
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self._device_indices):
            self.engine.set_device(self._device_indices[idx])
            if self.engine.stream is not None:
                self.engine.start()

    # ------------------------------------------------------------ source --
    def _on_source_changed(self, *_):
        self.current_mode = "sine" if self.radio_sine.isChecked() else "file"
        self.sine_freq_spin.setEnabled(self.current_mode == "sine")
        self.load_file_btn.setEnabled(self.current_mode == "file")
        self._rebuild_source()

    def _on_sine_freq_changed(self, val):
        self.sine_freq = val
        if self.current_mode == "sine":
            self._rebuild_source()

    def _on_load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load audio file", "",
            "Audio files (*.wav *.flac *.ogg *.aiff *.aif *.mp3);;All files (*)"
        )
        if not path:
            return
        try:
            data, fs = sf.read(path, always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float64)
            peak = np.max(np.abs(data)) if len(data) else 1.0
            if peak > 0:
                data = data / peak * 0.9
            self.loaded_file_data = (data, fs)
            self.file_label.setText(path.split("/")[-1].split("\\")[-1])
            self.radio_file.setChecked(True)
            self._rebuild_source()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load error", f"Could not load file:\n{e}")

    def _rebuild_source(self):
        fs = self.NATIVE_FS
        if self.current_mode == "sine" or self.loaded_file_data is None:
            data = generate_sine(self.sine_freq, self.SINE_DURATION, fs)
        else:
            raw, raw_fs = self.loaded_file_data
            data = self._resample_linear(raw, raw_fs, fs) if raw_fs != fs else raw
        self.engine.sr_factor = self.sr_slider.value()
        self.engine.bits = self.bit_slider.value()
        self.engine.load_source(data, fs)

    @staticmethod
    def _resample_linear(x, fs_in, fs_out):
        if fs_in == fs_out or len(x) == 0:
            return x
        n_out = int(round(len(x) / fs_in * fs_out))
        t_in = np.arange(len(x)) / fs_in
        t_out = np.arange(n_out) / fs_out
        return np.interp(t_out, t_in, x)

    # ------------------------------------------------------------ params --
    def _on_params_changed(self, *_):
        # Cheap: update labels and flag for the timer to apply the DSP.
        self._update_param_labels()
        self._params_dirty = True

    def _on_wave_window_changed(self, val):
        self.WAVE_WINDOW_MS = float(val)

    def _apply_params_to_engine(self):
        self.engine.update_params(self.sr_slider.value(), self.bit_slider.value())

    def _sr_text(self, sr_factor):
        eff_fs = self.NATIVE_FS / sr_factor
        return f"÷{sr_factor}  (≈ {eff_fs:,.0f} Hz, Nyquist ≈ {eff_fs / 2:,.0f} Hz)"

    @staticmethod
    def _bit_text(bits):
        return f"{bits}-bit  ({2 ** bits} levels)"

    def _update_param_labels(self):
        self.sr_value_label.setText(self._sr_text(self.sr_slider.value()))
        self.bit_value_label.setText(self._bit_text(self.bit_slider.value()))

    # --------------------------------------------------------- transport --
    def _on_play(self):
        self._flush_params()
        self.engine.start()
        self.status.showMessage("Playing…")

    def _on_stop(self):
        self.engine.stop()
        self.status.showMessage("Stopped.")

    def on_engine_error(self, msg):
        self.status.showMessage(f"Audio error: {msg}")
        QtWidgets.QMessageBox.warning(self, "Playback error", msg)

    def _flush_params(self):
        if self._params_dirty:
            self._params_dirty = False
            self._apply_params_to_engine()

    # ------------------------------------------------------------ plots --
    def refresh_plots(self):
        self._flush_params()

        orig, proc, fs, pos = self.engine.get_snapshot()
        if len(orig) == 0 or len(proc) == 0:
            return
        new_nyquist = (fs / self.sr_slider.value()) / 2.0

        n_win = int(fs * self.WAVE_WINDOW_MS / 1000.0)
        n_win = max(2, min(n_win, len(orig), len(proc)))
        start = pos if pos + n_win <= len(orig) else max(0, len(orig) - n_win)
        orig_win = orig[start:start + n_win]
        proc_win = proc[start:start + n_win] if start + n_win <= len(proc) else proc[:n_win]
        self.canvas.update_waveform(orig_win, proc_win, fs, self.WAVE_WINDOW_MS)

        n_spec = min(len(orig), 16384)
        self.canvas.update_spectrum(orig[:n_spec], proc[:n_spec], fs, new_nyquist)
        self.canvas.redraw()

    def closeEvent(self, event):
        self.update_timer.stop()
        self.engine.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()