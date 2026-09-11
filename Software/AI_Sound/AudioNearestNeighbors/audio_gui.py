import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from threading import Thread, Event
import time

"""
AudioGui for the NearestNeighbors tool.

Layout (top to bottom):
- output device dropdown
- load audio folder button + progress bar + status label (background-
  threaded, matching AudioClustering's load-file workflow)
- excerpt timing controls: excerpt length (seconds) and excerpt offset
  (seconds) spinboxes + an "apply" button. Since excerpt boundaries are
  baked into how the source audio was sliced, applying a change here
  re-slices and re-analyzes the currently loaded folder from scratch
  (same background-thread mechanism as loading a folder) rather than
  taking effect instantly.
- audio feature checklist (multi-select - which descriptors are
  concatenated into the search distance vector)
- start / stop buttons: start begins (or resumes) stepping the
  nearest-neighbor search AND playing back the growing result in real
  time; stop pauses both
- progress bar (fraction of excerpts consumed) + generated-duration label
- save audio file... button (writes everything generated so far to disk)

The generation loop (calling model.step() repeatedly at a fixed rate) runs
on its own background thread, matching the update-loop pattern used by
AudioAnalysis/AudioClustering's GUIs, so the Qt event loop / playback are
never blocked by the nearest-neighbor search itself.
"""

config = {
    "model": None,
    "synthesis": None,
    "step_interval_seconds": 0.05,
}


class _LoadFolderWorker(QObject):
    """
    Runs model.load_and_extract() on a background thread, forwarding its
    progress_callback(fraction, message) calls as a Qt signal so the GUI
    thread can update a progress bar safely. Used both for the initial
    "load audio folder..." action and for re-extraction triggered by
    applying new excerpt timing values.
    """
    progress = pyqtSignal(float, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, run_fn):
        super().__init__()
        self.run_fn = run_fn

    def run(self):
        try:
            self.run_fn(progress_callback=lambda f, m: self.progress.emit(f, m))
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))


class AudioGui(QtWidgets.QWidget):

    def __init__(self, config):
        super().__init__()

        self.model = config["model"]
        self.synthesis = config["synthesis"]
        self.step_interval_seconds = config.get("step_interval_seconds", 0.05)

        self._load_thread = None
        self._load_worker = None

        self._gen_thread = None
        self._gen_thread_event = None

        # ---------------- output device (top of window) ----------------

        self.q_output_device_label = QtWidgets.QLabel("output device:")
        self.q_output_device_combo = QtWidgets.QComboBox()
        self.output_device_mapping = []
        for idx, name in self.synthesis.list_output_devices():
            self.output_device_mapping.append(idx)
            self.q_output_device_combo.addItem("{}: {}".format(idx, name))
        self.q_output_device_combo.currentIndexChanged.connect(self.change_output_device)

        self.q_device_grid = QtWidgets.QGridLayout()
        self.q_device_grid.addWidget(self.q_output_device_label, 0, 0)
        self.q_device_grid.addWidget(self.q_output_device_combo, 0, 1)

        # ---------------- load audio folder ----------------

        self.q_load_folder_button = QtWidgets.QPushButton("load audio folder...", self)
        self.q_load_folder_button.clicked.connect(self.load_folder)

        self.q_load_progress = QtWidgets.QProgressBar(self)
        self.q_load_progress.setMinimum(0)
        self.q_load_progress.setMaximum(100)

        self.q_load_status_label = QtWidgets.QLabel("")
        self.q_load_status_label.setWordWrap(True)

        self.q_load_grid = QtWidgets.QVBoxLayout()
        self.q_load_grid.addWidget(self.q_load_folder_button)
        self.q_load_grid.addWidget(self.q_load_progress)
        self.q_load_grid.addWidget(self.q_load_status_label)

        # ---------------- excerpt timing controls ----------------

        self.q_excerpt_length_label = QtWidgets.QLabel("excerpt length (s):")
        self.q_excerpt_length_spin = QtWidgets.QDoubleSpinBox(self)
        self.q_excerpt_length_spin.setDecimals(2)
        self.q_excerpt_length_spin.setMinimum(0.05)
        self.q_excerpt_length_spin.setMaximum(60.0)
        self.q_excerpt_length_spin.setSingleStep(0.1)
        self.q_excerpt_length_spin.setValue(self.model.audio_excerpt_seconds)

        self.q_excerpt_offset_label = QtWidgets.QLabel("excerpt offset (s):")
        self.q_excerpt_offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.q_excerpt_offset_spin.setDecimals(2)
        self.q_excerpt_offset_spin.setMinimum(0.05)
        self.q_excerpt_offset_spin.setMaximum(60.0)
        self.q_excerpt_offset_spin.setSingleStep(0.1)
        self.q_excerpt_offset_spin.setValue(self.model.audio_excerpt_offset_seconds)

        self.q_excerpt_apply_button = QtWidgets.QPushButton("apply excerpt timing", self)
        self.q_excerpt_apply_button.clicked.connect(self.apply_excerpt_timing)

        self.q_excerpt_grid = QtWidgets.QGridLayout()
        self.q_excerpt_grid.addWidget(self.q_excerpt_length_label, 0, 0)
        self.q_excerpt_grid.addWidget(self.q_excerpt_length_spin, 0, 1)
        self.q_excerpt_grid.addWidget(self.q_excerpt_offset_label, 1, 0)
        self.q_excerpt_grid.addWidget(self.q_excerpt_offset_spin, 1, 1)
        self.q_excerpt_grid.addWidget(self.q_excerpt_apply_button, 2, 0, 1, 2)

        # ---------------- feature checklist ----------------

        self.q_feature_list_label = QtWidgets.QLabel("audio features used for search:")
        self.q_feature_list = QtWidgets.QListWidget()
        self._populate_feature_list()
        self.q_feature_list.itemChanged.connect(self.change_selected_features)

        self.q_feature_grid = QtWidgets.QVBoxLayout()
        self.q_feature_grid.addWidget(self.q_feature_list_label)
        self.q_feature_grid.addWidget(self.q_feature_list)

        # ---------------- start / stop ----------------

        self.q_start_button = QtWidgets.QPushButton("start", self)
        self.q_start_button.clicked.connect(self.start)

        self.q_stop_button = QtWidgets.QPushButton("stop", self)
        self.q_stop_button.clicked.connect(self.stop)

        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_button, 0, 0)
        self.q_button_grid.addWidget(self.q_stop_button, 0, 1)

        # ---------------- progress ----------------

        self.q_gen_progress = QtWidgets.QProgressBar(self)
        self.q_gen_progress.setMinimum(0)
        self.q_gen_progress.setMaximum(100)

        self.q_gen_status_label = QtWidgets.QLabel("")

        self.q_gen_grid = QtWidgets.QVBoxLayout()
        self.q_gen_grid.addWidget(self.q_gen_progress)
        self.q_gen_grid.addWidget(self.q_gen_status_label)

        self.q_progress_timer = QTimer(self)
        self.q_progress_timer.setInterval(200)
        self.q_progress_timer.timeout.connect(self.refresh_progress)
        self.q_progress_timer.start()

        # ---------------- save ----------------

        self.q_save_button = QtWidgets.QPushButton("save audio file...", self)
        self.q_save_button.clicked.connect(self.save_file)

        # ---------------- overall layout ----------------

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.q_device_grid, 0, 0)
        self.q_grid.addLayout(self.q_load_grid, 1, 0)
        self.q_grid.addLayout(self.q_excerpt_grid, 2, 0)
        self.q_grid.addLayout(self.q_feature_grid, 3, 0)
        self.q_grid.addLayout(self.q_button_grid, 4, 0)
        self.q_grid.addLayout(self.q_gen_grid, 5, 0)
        self.q_grid.addWidget(self.q_save_button, 6, 0)

        for row in range(7):
            self.q_grid.setRowStretch(row, 0)
        self.q_grid.setRowStretch(3, 1)

        self.setLayout(self.q_grid)

        self.setGeometry(50, 50, 420, 720)
        self.setWindowTitle("Audio Nearest Neighbors")

    # ---------------- feature checklist helpers ----------------

    def _populate_feature_list(self):
        self.q_feature_list.blockSignals(True)
        self.q_feature_list.clear()

        available = self.model.get_available_feature_names()
        selected = set(self.model.audio_feature_names)

        for name in available:
            item = QtWidgets.QListWidgetItem(name, self.q_feature_list)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selected else Qt.Unchecked)

        self.q_feature_list.blockSignals(False)

    def change_selected_features(self, item):
        selected = []
        for i in range(self.q_feature_list.count()):
            it = self.q_feature_list.item(i)
            if it.checkState() == Qt.Checked:
                selected.append(it.text())

        was_running = self._gen_thread is not None and self._gen_thread.is_alive()
        if was_running:
            self.stop()

        self.model.select_audio_features(selected)
        self.synthesis.reset_playback()

        if was_running:
            self.start()

    # ---------------- device ----------------

    def change_output_device(self, idx):
        if 0 <= idx < len(self.output_device_mapping):
            device_id = self.output_device_mapping[idx]
            self.synthesis.set_output_device(device_id)

    # ---------------- busy state (shared by load-folder and apply-timing) ----------------

    def _set_busy(self, busy, message=""):
        self.q_load_folder_button.setEnabled(not busy)
        self.q_excerpt_apply_button.setEnabled(not busy)
        self.q_excerpt_length_spin.setEnabled(not busy)
        self.q_excerpt_offset_spin.setEnabled(not busy)
        self.q_start_button.setEnabled(not busy)
        self.q_feature_list.setEnabled(not busy)
        self.q_load_status_label.setText(message)

    def _run_background(self, run_fn, on_finished):
        was_running = self._gen_thread is not None and self._gen_thread.is_alive()
        if was_running:
            self.stop()

        self._set_busy(True, "processing...")
        self.q_load_progress.setValue(0)

        self._load_thread = QThread(self)
        self._load_worker = _LoadFolderWorker(run_fn)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(on_finished)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def _on_load_progress(self, fraction, message):
        self.q_load_progress.setValue(int(fraction * 100))
        self.q_load_status_label.setText(message)

    def _on_load_failed(self, error_message):
        self._set_busy(False, "failed: {}".format(error_message))

    # ---------------- load folder (background thread) ----------------

    def load_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select audio folder")
        if not folder_path:
            return

        self.model.audio_folder_path = folder_path
        self._run_background(self.model.load_and_extract, self._on_load_finished)

    def _on_load_finished(self):
        self._populate_feature_list()
        self.model.reset()
        self.synthesis.reset_playback()
        self._set_busy(False, "loaded {} excerpts.".format(self.model.audio_excerpts.shape[0]))

    # ---------------- excerpt timing (background thread - re-extracts) ----------------

    def apply_excerpt_timing(self):
        if self.model.audio_excerpts is None:
            # nothing loaded yet - just update the values, they take effect
            # once a folder is loaded
            self.model.set_excerpt_timing(
                self.q_excerpt_length_spin.value(), self.q_excerpt_offset_spin.value()
            )
            self.q_load_status_label.setText(
                "excerpt timing updated ({}s / {}s) - load an audio folder to apply.".format(
                    self.model.audio_excerpt_seconds, self.model.audio_excerpt_offset_seconds
                )
            )
            return

        excerpt_seconds = self.q_excerpt_length_spin.value()
        offset_seconds = self.q_excerpt_offset_spin.value()

        def run_fn(progress_callback=None):
            self.model.set_excerpt_timing(excerpt_seconds, offset_seconds, progress_callback=progress_callback)

        self._run_background(run_fn, self._on_excerpt_timing_applied)

    def _on_excerpt_timing_applied(self):
        self._populate_feature_list()
        self.synthesis.reset_playback()
        self._set_busy(False, "re-analyzed with excerpt length={}s, offset={}s ({} excerpts).".format(
            self.model.audio_excerpt_seconds, self.model.audio_excerpt_offset_seconds,
            self.model.audio_excerpts.shape[0]
        ))

    # ---------------- generation + playback ----------------

    def start(self):
        if self.model.audio_excerpts is None:
            self.q_gen_status_label.setText("load an audio folder first.")
            return

        self.synthesis.start()

        self._gen_thread_event = Event()
        self._gen_thread = Thread(target=self._generation_loop, daemon=True)
        self._gen_thread.start()

    def stop(self):
        if self._gen_thread_event is not None:
            self._gen_thread_event.set()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=1.0)
            self._gen_thread = None

        self.synthesis.stop()

    def _generation_loop(self):
        while self._gen_thread_event.is_set() == False:
            if not self.model.is_done():
                self.model.step()
            time.sleep(self.step_interval_seconds)

    def refresh_progress(self):
        progress = self.model.get_progress()
        self.q_gen_progress.setValue(int(progress * 100))

        played_seconds = self.synthesis.get_progress_seconds()
        generated_seconds = self.model.get_output_waveform().shape[0] / float(self.model.audio_sample_rate)

        status = "generated: {:.1f}s, played: {:.1f}s".format(generated_seconds, played_seconds)
        if self.model.is_done():
            status += "  (search complete)"
        elif self.synthesis.is_caught_up():
            status += "  (waiting for more audio...)"

        self.q_gen_status_label.setText(status)

    # ---------------- save ----------------

    def save_file(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save audio file", "gen_waveform.wav", "WAV files (*.wav)"
        )
        if not file_path:
            return

        self.synthesis.save_to_file(file_path)
        self.q_gen_status_label.setText("saved to '{}'.".format(file_path))
