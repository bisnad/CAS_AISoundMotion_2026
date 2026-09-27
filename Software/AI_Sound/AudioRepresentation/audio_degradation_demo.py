#!/usr/bin/env python3
"""
Audio Degradation Demo — Aliasing & Bit-Depth Reduction
=========================================================

Interactive PyQt5 GUI that demonstrates two classic digital-audio
degradation effects in real time:

1. Aliasing / frequency foldover — caused by reducing the effective
   sampling rate (simulated via sample-and-hold decimation without an
   anti-aliasing filter, so high frequencies "fold over" the new
   Nyquist limit instead of being removed cleanly).

2. Quantization distortion — caused by reducing the bit resolution
   used to represent the amplitude of each sample (fewer quantization
   levels -> more quantization noise / harsh high-frequency artifacts).

You can play either:
  - A synthetic sine wave (frequency adjustable), or
  - An external audio file (wav/flac/ogg/... loaded via soundfile,
    resampled to the device sample rate).

Sliders let you change:
  - Effective sample rate (via decimation factor)
  - Bit depth (2-16 bits)

A dropdown lets you pick the audio OUTPUT device (via sounddevice).
Two stacked live plots show the effect of your settings:
  - Top: waveform view (original vs. degraded), showing the
    stair-stepping / held samples from sample-rate reduction and the
    "stepped" amplitude levels from bit-depth reduction directly in
    the time domain.
  - Bottom: spectrum view (original vs. degraded), showing foldover
    artifacts appearing below the new Nyquist frequency.

Dependencies
------------
    pip install PyQt5 sounddevice soundfile numpy scipy matplotlib

Run
---
    python audio_degradation_demo.py
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
    """Generate a mono sine wave at the given sample rate."""
    t = np.arange(int(duration * fs)) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def apply_sample_rate_reduction(x, factor):
    """
    Simulate a reduced sampling rate WITHOUT an anti-aliasing filter.

    We use a naive sample-and-hold decimation: every `factor` samples,
    we hold the value of the first sample of that group for the whole
    group, at the ORIGINAL sample rate. This mimics what a cheap
    sample-rate reducer / bitcrusher "downsample" stage does: it does
    not band-limit the signal first, so frequency content above the
    new (lower) Nyquist frequency folds back (aliases) into the
    audible range instead of being attenuated.

    factor = 1 means no reduction (bypass).
    """
    factor = max(1, int(round(factor)))
    if factor == 1:
        return x.copy()
    n = len(x)
    idx = (np.arange(n) // factor) * factor
    idx = np.clip(idx, 0, n - 1)
    return x[idx]


def apply_bit_depth_reduction(x, bits):
    """
    Simulate reduced amplitude (bit) resolution via uniform quantization.

    bits: number of bits used to represent amplitude (e.g. 16, 8, 4, 2).
    The signal is assumed normalized to [-1, 1]; it is clipped, then
    quantized to 2**bits evenly spaced levels across that range.

    Lower bit depths introduce quantization error that behaves like
    added broadband noise / harsh distortion, which is particularly
    audible as harshness in high-frequency content and as an audible
    noise floor.
    """
    bits = int(round(bits))
    bits = max(1, min(24, bits))
    levels = 2 ** bits
    x_clipped = np.clip(x, -1.0, 1.0)
    # map [-1, 1] -> [0, levels-1], round, map back
    q = np.round((x_clipped + 1.0) * 0.5 * (levels - 1))
    q = q / (levels - 1) * 2.0 - 1.0
    return q


def process_audio(x, sr_factor, bits):
    """Apply both degradation stages in sequence."""
    y = apply_sample_rate_reduction(x, sr_factor)
    y = apply_bit_depth_reduction(y, bits)
    return y


# --------------------------------------------------------------------------
# Audio playback engine
# --------------------------------------------------------------------------

class AudioEngine(QtCore.QObject):
    """
    Wraps sounddevice OutputStream playback of a numpy buffer with a
    callback so that changes to sliders can be picked up (by
    reprocessing the *source* buffer and swapping the playback buffer)
    without restarting the stream every time.
    """

    position_changed = QtCore.pyqtSignal(int)
    playback_finished = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.fs = 44100
        self.source = np.zeros(0, dtype=np.float64)   # untouched original
        self.playback_buffer = np.zeros(0, dtype=np.float64)  # processed
        self.lock = threading.Lock()
        self.pos = 0
        self.stream = None
        self.device_index = None
        self.sr_factor = 1
        self.bits = 16

    def set_device(self, device_index):
        self.device_index = device_index

    def load_source(self, data, fs):
        with self.lock:
            self.source = data.astype(np.float64)
            self.fs = fs
            self.pos = 0
            self._reprocess_locked()

    def update_params(self, sr_factor, bits):
        with self.lock:
            self.sr_factor = sr_factor
            self.bits = bits
            self._reprocess_locked()

    def _reprocess_locked(self):
        """Recompute playback_buffer from source using current params.
        Caller must hold self.lock."""
        if len(self.source) == 0:
            self.playback_buffer = np.zeros(0)
            return
        self.playback_buffer = process_audio(self.source, self.sr_factor, self.bits)

    def get_processed_preview(self, n_samples=None):
        """Return a copy of the current processed buffer (for plotting)."""
        with self.lock:
            if n_samples is None:
                return self.playback_buffer.copy(), self.fs
            return self.playback_buffer[:n_samples].copy(), self.fs

    def get_playhead(self):
        with self.lock:
            return self.pos

    def _callback(self, outdata, frames, time_info, status):
        if status:
            pass  # underflows etc. are common while scrubbing sliders; ignore
        with self.lock:
            buf = self.playback_buffer
            n = len(buf)
            if n == 0:
                outdata[:] = 0
                return
            end = self.pos + frames
            if end <= n:
                chunk = buf[self.pos:end]
                self.pos = end
            else:
                chunk = np.concatenate([buf[self.pos:n], np.zeros(end - n)])
                self.pos = 0  # loop
            outdata[:, 0] = chunk
        self.position_changed.emit(self.pos)

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
# Waveform + Spectrum plot widget (stacked)
# --------------------------------------------------------------------------

class ScopeCanvas(FigureCanvas):
    """Two stacked live plots sharing one figure:
    top = time-domain waveform (original vs. degraded),
    bottom = frequency-domain spectrum (original vs. degraded)."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6.0), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        # --- waveform (top) axis ---
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_wave.set_xlabel("Time (ms)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_title("Waveform (original vs. degraded)")
        self.wave_orig, = self.ax_wave.plot([], [], color="#999999", lw=1.0, label="Original")
        self.wave_proc, = self.ax_wave.plot([], [], color="#d62728", lw=1.2, label="Degraded")
        self.ax_wave.legend(loc="upper right", fontsize=8)
        self.ax_wave.set_ylim(-1.05, 1.05)

        # --- spectrum (bottom) axis ---
        self.ax_spec = self.fig.add_subplot(212)
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Magnitude (dB)")
        self.ax_spec.set_title("Spectrum (original vs. degraded)")
        self.spec_orig, = self.ax_spec.plot([], [], color="#999999", lw=1.0, label="Original")
        self.spec_proc, = self.ax_spec.plot([], [], color="#d62728", lw=1.2, label="Degraded")
        self.nyq_line = self.ax_spec.axvline(0, color="#1f77b4", ls="--", lw=1.0, label="New Nyquist")
        self.ax_spec.legend(loc="upper right", fontsize=8)
        self.ax_spec.set_ylim(-100, 5)

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
            n = min(len(x), 65536)
            x = x[:n] * np.hanning(n)
            spec = np.fft.rfft(x)
            mag_db = 20 * np.log10(np.abs(spec) / (n / 2) + 1e-12)
            freqs = np.fft.rfftfreq(n, 1 / fs)
            return freqs, mag_db

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
    SINE_DURATION = 3.0     # seconds, looped
    NATIVE_FS = 44100       # device playback rate; source audio is resampled/repeated to fit callback
    WAVE_WINDOW_MS = 30.0   # width of the live waveform window

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Degradation Demo — Aliasing & Bit-Depth")
        self.resize(950, 850)

        self.engine = AudioEngine()
        self.engine.error_occurred.connect(self.on_engine_error)

        self.loaded_file_data = None   # (mono float64 array, fs) as loaded, before any resampling to NATIVE_FS
        self.current_mode = "sine"     # "sine" or "file"
        self.sine_freq = 440.0

        self._build_ui()
        self._populate_devices()
        self._on_source_changed()  # initializes sine source
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(80)
        self.update_timer.timeout.connect(self.refresh_plots)
        self.update_timer.start()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # --- Source selection group ---
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

        # --- Output device group ---
        dev_group = QtWidgets.QGroupBox("Audio Output Device")
        dev_layout = QtWidgets.QHBoxLayout(dev_group)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        dev_layout.addWidget(self.device_combo, 1)
        self.refresh_devices_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_devices_btn.clicked.connect(self._populate_devices)
        dev_layout.addWidget(self.refresh_devices_btn)
        main_layout.addWidget(dev_group)

        # --- Degradation controls group ---
        deg_group = QtWidgets.QGroupBox("Degradation Controls")
        deg_layout = QtWidgets.QGridLayout(deg_group)

        # Sample rate reduction slider (expressed as decimation factor 1..64)
        deg_layout.addWidget(QtWidgets.QLabel("Sample-rate reduction:"), 0, 0)
        self.sr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sr_slider.setRange(1, 64)
        self.sr_slider.setValue(1)
        self.sr_slider.valueChanged.connect(self._on_params_changed)
        deg_layout.addWidget(self.sr_slider, 0, 1)
        self.sr_value_label = QtWidgets.QLabel()
        deg_layout.addWidget(self.sr_value_label, 0, 2)

        # Bit depth slider
        deg_layout.addWidget(QtWidgets.QLabel("Bit resolution:"), 1, 0)
        self.bit_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bit_slider.setRange(2, 16)
        self.bit_slider.setValue(16)
        self.bit_slider.valueChanged.connect(self._on_params_changed)
        deg_layout.addWidget(self.bit_slider, 1, 1)
        self.bit_value_label = QtWidgets.QLabel()
        deg_layout.addWidget(self.bit_value_label, 1, 2)

        main_layout.addWidget(deg_group)

        # --- Transport controls ---
        transport_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play)
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        transport_layout.addWidget(self.play_btn)
        transport_layout.addWidget(self.stop_btn)
        transport_layout.addStretch(1)

        # waveform window width control
        transport_layout.addWidget(QtWidgets.QLabel("Waveform window:"))
        self.wave_window_spin = QtWidgets.QSpinBox()
        self.wave_window_spin.setRange(2, 200)
        self.wave_window_spin.setValue(int(self.WAVE_WINDOW_MS))
        self.wave_window_spin.setSuffix(" ms")
        self.wave_window_spin.valueChanged.connect(self._on_wave_window_changed)
        transport_layout.addWidget(self.wave_window_spin)

        main_layout.addLayout(transport_layout)

        # --- Waveform + Spectrum plots (stacked) ---
        self.canvas = ScopeCanvas(self)
        main_layout.addWidget(self.canvas, 1)

        # --- Status bar ---
        self.status = self.statusBar()
        self.status.showMessage("Ready.")

        self._update_param_labels()

    # ----------------------------------------------------------- devices --
    def _populate_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception as e:
            self.status.showMessage(f"Could not query devices: {e}")
            self.device_combo.blockSignals(False)
            return
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None
        self._device_indices = []
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0:
                label = f"{i}: {d['name']} ({d['max_output_channels']} ch, {int(d['default_samplerate'])} Hz)"
                self.device_combo.addItem(label)
                self._device_indices.append(i)
        # select default output device if possible
        if default_out in self._device_indices:
            self.device_combo.setCurrentIndex(self._device_indices.index(default_out))
        elif self._device_indices:
            self.device_combo.setCurrentIndex(0)
        self.device_combo.blockSignals(False)
        self._on_device_changed()

    def _on_device_changed(self):
        if not getattr(self, "_device_indices", None):
            return
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self._device_indices):
            dev_index = self._device_indices[idx]
            self.engine.set_device(dev_index)
            was_playing = self.engine.stream is not None
            if was_playing:
                self.engine.start()  # restart on new device

    # ------------------------------------------------------------ source --
    def _on_source_changed(self):
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
                data = data.mean(axis=1)  # downmix to mono
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
        if self.current_mode == "sine":
            data = generate_sine(self.sine_freq, self.SINE_DURATION, fs)
        else:
            if self.loaded_file_data is None:
                data = generate_sine(self.sine_freq, self.SINE_DURATION, fs)
            else:
                raw, raw_fs = self.loaded_file_data
                if raw_fs != fs:
                    data = self._resample_linear(raw, raw_fs, fs)
                else:
                    data = raw
        self.engine.load_source(data, fs)
        self._apply_params_to_engine()

    @staticmethod
    def _resample_linear(x, fs_in, fs_out):
        """Simple, dependency-free linear-interpolation resampler
        (used only to bring source material to the playback rate;
        not part of the demonstrated degradation effects)."""
        if fs_in == fs_out or len(x) == 0:
            return x
        duration = len(x) / fs_in
        n_out = int(round(duration * fs_out))
        t_in = np.arange(len(x)) / fs_in
        t_out = np.arange(n_out) / fs_out
        return np.interp(t_out, t_in, x)

    # ------------------------------------------------------------ params --
    def _on_params_changed(self):
        self._update_param_labels()
        self._apply_params_to_engine()

    def _on_wave_window_changed(self, val):
        self.WAVE_WINDOW_MS = float(val)

    def _apply_params_to_engine(self):
        sr_factor = self.sr_slider.value()
        bits = self.bit_slider.value()
        self.engine.update_params(sr_factor, bits)

    def _update_param_labels(self):
        sr_factor = self.sr_slider.value()
        bits = self.bit_slider.value()
        eff_fs = self.NATIVE_FS / sr_factor
        self.sr_value_label.setText(f"÷{sr_factor}  (≈ {eff_fs:,.0f} Hz, Nyquist ≈ {eff_fs/2:,.0f} Hz)")
        self.bit_value_label.setText(f"{bits}-bit  ({2**bits} levels)")

    # --------------------------------------------------------- transport --
    def _on_play(self):
        self.engine.start()
        self.status.showMessage("Playing…")

    def _on_stop(self):
        self.engine.stop()
        self.status.showMessage("Stopped.")

    def on_engine_error(self, msg):
        self.status.showMessage(f"Audio error: {msg}")
        QtWidgets.QMessageBox.warning(self, "Playback error", msg)

    # ------------------------------------------------------------ plots --
    def refresh_plots(self):
        orig = self.engine.source
        proc, fs = self.engine.get_processed_preview()
        sr_factor = self.sr_slider.value()
        new_nyquist = (fs / sr_factor) / 2.0

        if len(orig) == 0:
            return

        # Waveform view: a short scrolling window anchored at the current
        # playhead position, so the live waveform tracks what is audible
        # right now (falls back to the start of the buffer if not playing).
        n_win = int(fs * self.WAVE_WINDOW_MS / 1000.0)
        n_win = max(2, min(n_win, len(orig), len(proc)))
        pos = self.engine.get_playhead()
        start = pos if pos + n_win <= len(orig) else max(0, len(orig) - n_win)
        orig_win = orig[start:start + n_win]
        proc_win = proc[start:start + n_win] if start + n_win <= len(proc) else proc[:n_win]
        self.canvas.update_waveform(orig_win, proc_win, fs, self.WAVE_WINDOW_MS)

        # Spectrum view: use a longer chunk for good frequency resolution.
        n_spec = min(len(orig), 65536)
        self.canvas.update_spectrum(orig[:n_spec], proc[:n_spec], fs, new_nyquist)

        self.canvas.redraw()

    def closeEvent(self, event):
        self.engine.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
