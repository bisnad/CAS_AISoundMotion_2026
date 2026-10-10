import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal

import clustering_pipeline as cp
import audio_export

"""
AudioGui for the clustering tool.

- Several audio features can be checked and combined for clustering.
- Excerpt length and overlap are entered in SECONDS (hop = length - overlap).
  Internally the pipeline still works in milliseconds, and the config keys
  excerpt_length_ms / excerpt_offset_ms are unchanged.
- "apply analysis settings" re-slices / computes only missing features /
  re-clusters on a background QThread; "load audio file..." uses the same
  widgets and sits at the top, below the output device pulldown.
- "save clusters as audio files..." writes one WAV per cluster (audio_export.py).
"""

config = {
    "model": None,
    "synthesis": None,
    "control": None,
    "pipeline": None,
    "excerpt_length_ms": 100,
    "excerpt_offset_ms": 90,
}

SECONDS_DECIMALS = 3   # 1 ms resolution


class _AnalysisWorker(QObject):
    """Runs pipeline.prepare() + model.set_data() + clustering off the GUI thread."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, pipeline, model, file_path, length_ms, offset_ms, feature_names):
        super().__init__()
        self.pipeline = pipeline
        self.model = model
        self.file_path = file_path
        self.length_ms = length_ms
        self.offset_ms = offset_ms
        self.feature_names = feature_names

    def run(self):
        try:
            excerpts, features, sample_rate = self.pipeline.prepare(
                self.file_path, self.length_ms, self.offset_ms, self.feature_names)

            self.model.set_data(excerpts, features, self.feature_names)
            self.model.create_clusters()

            self.finished.emit({
                "excerpt_count": int(excerpts.shape[0]),
                "sample_rate": int(sample_rate),
                "length_ms": self.length_ms,
                "offset_ms": self.offset_ms,
                "dimensions": self.model.get_feature_dimensions(),
            })
        except Exception as e:
            self.failed.emit(str(e))


class _ExportWorker(QObject):
    """Writes one audio file per cluster off the GUI thread."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, model, synthesis, folder):
        super().__init__()
        self.model = model
        self.synthesis = synthesis
        self.folder = folder

    def run(self):
        try:
            written = audio_export.export_clusters(
                self.model, self.synthesis, self.folder,
                progress=lambda done, total: self.progress.emit(done, total))
            self.finished.emit({"folder": self.folder, "files": written})
        except Exception as e:
            self.failed.emit(str(e))


class AudioGui(QtWidgets.QWidget):

    def __init__(self, config):
        super().__init__()

        self.model = config["model"]
        self.synthesis = config["synthesis"]
        self.control = config.get("control", None)
        self.pipeline = config["pipeline"]

        self._thread = None
        self._worker = None

        # ---------------- output device + load file (top) ----------------

        self.q_output_device_combo = QtWidgets.QComboBox()
        self.output_device_mapping = []
        for idx, name in self.synthesis.list_output_devices():
            self.output_device_mapping.append(idx)
            self.q_output_device_combo.addItem("{}: {}".format(idx, name))
        self.q_output_device_combo.currentIndexChanged.connect(self.change_output_device)

        self.q_load_file_button = QtWidgets.QPushButton("load audio file...", self)
        self.q_load_file_button.clicked.connect(self.load_file)

        self.q_status_label = QtWidgets.QLabel("")
        self.q_status_label.setWordWrap(True)

        device_box = QtWidgets.QGroupBox("output / input")
        device_layout = QtWidgets.QFormLayout(device_box)
        device_layout.addRow("output device:", self.q_output_device_combo)
        device_layout.addRow(self.q_load_file_button)
        device_layout.addRow(self.q_status_label)

        # ---------------- excerpts (seconds) ----------------

        length_s = float(config.get("excerpt_length_ms", 100)) / 1000.0
        offset_s = float(config.get("excerpt_offset_ms", 90)) / 1000.0
        overlap_s = max(0.0, length_s - offset_s)

        self.q_length_spin = QtWidgets.QDoubleSpinBox()
        self.q_length_spin.setDecimals(SECONDS_DECIMALS)
        self.q_length_spin.setSingleStep(0.01)
        self.q_length_spin.setRange(0.001, 60.0)
        self.q_length_spin.setSuffix(" s")
        self.q_length_spin.setValue(length_s)

        self.q_overlap_spin = QtWidgets.QDoubleSpinBox()
        self.q_overlap_spin.setDecimals(SECONDS_DECIMALS)
        self.q_overlap_spin.setSingleStep(0.01)
        self.q_overlap_spin.setRange(0.0, max(0.0, length_s - 0.001))
        self.q_overlap_spin.setSuffix(" s")
        self.q_overlap_spin.setValue(overlap_s)

        self.q_excerpt_info_label = QtWidgets.QLabel("")
        self.q_excerpt_info_label.setWordWrap(True)

        self.q_length_spin.valueChanged.connect(self.change_excerpt_length)
        self.q_overlap_spin.valueChanged.connect(self.update_excerpt_info)

        excerpt_box = QtWidgets.QGroupBox("excerpts")
        excerpt_layout = QtWidgets.QFormLayout(excerpt_box)
        excerpt_layout.addRow("excerpt length:", self.q_length_spin)
        excerpt_layout.addRow("overlap:", self.q_overlap_spin)
        excerpt_layout.addRow(self.q_excerpt_info_label)

        # ---------------- audio features (multi select) ----------------

        self.q_feature_list = QtWidgets.QListWidget()
        self.q_feature_list.setMinimumHeight(190)
        selected = set(self.model.get_selected_features())
        for name in cp.get_all_feature_names():
            text = name + ("  (slow)" if cp.is_slow_feature(name) else "")
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in selected else Qt.Unchecked)
            self.q_feature_list.addItem(item)

        self.q_apply_button = QtWidgets.QPushButton("apply analysis settings", self)
        self.q_apply_button.clicked.connect(self.apply_settings)

        self.q_feature_info_label = QtWidgets.QLabel("")
        self.q_feature_info_label.setWordWrap(True)

        feature_box = QtWidgets.QGroupBox("audio features (check several to combine)")
        feature_layout = QtWidgets.QVBoxLayout(feature_box)
        feature_layout.addWidget(self.q_feature_list)
        feature_layout.addWidget(self.q_apply_button)
        feature_layout.addWidget(self.q_feature_info_label)

        # ---------------- clustering ----------------

        self.q_method_combo = QtWidgets.QComboBox()
        self.q_method_combo.addItem("kmeans")
        self.q_method_combo.addItem("minibatch_kmeans")
        self.q_method_combo.setCurrentText(self.model.cluster_method)
        self.q_method_combo.currentTextChanged.connect(self.change_method)

        self.q_cluster_count_spin = QtWidgets.QSpinBox(self)
        self.q_cluster_count_spin.setRange(2, 500)
        self.q_cluster_count_spin.setValue(self.model.cluster_count)
        self.q_cluster_count_spin.valueChanged.connect(self.change_cluster_count)

        self.q_normalize_toggle = QtWidgets.QCheckBox("normalize features (z-score)", self)
        self.q_normalize_toggle.setChecked(self.model.normalize)
        self.q_normalize_toggle.stateChanged.connect(lambda: self.toggle_normalize(self.q_normalize_toggle))

        self.q_equal_weight_toggle = QtWidgets.QCheckBox("equal weight per feature", self)
        self.q_equal_weight_toggle.setChecked(self.model.equal_feature_weight)
        self.q_equal_weight_toggle.setToolTip(
            "Scale each feature block by 1/sqrt(dimensions) so a high-dimensional feature "
            "(e.g. mel spectrogram) does not outweigh a low-dimensional one (e.g. RMS).")
        self.q_equal_weight_toggle.stateChanged.connect(lambda: self.toggle_equal_weight(self.q_equal_weight_toggle))

        cluster_box = QtWidgets.QGroupBox("clustering")
        cluster_layout = QtWidgets.QFormLayout(cluster_box)
        cluster_layout.addRow("cluster method:", self.q_method_combo)
        cluster_layout.addRow("cluster count:", self.q_cluster_count_spin)
        cluster_layout.addRow(self.q_normalize_toggle)
        cluster_layout.addRow(self.q_equal_weight_toggle)

        # ---------------- start / stop / save ----------------

        self.q_start_button = QtWidgets.QPushButton("start", self)
        self.q_start_button.clicked.connect(self.start)
        self.q_stop_button = QtWidgets.QPushButton("stop", self)
        self.q_stop_button.clicked.connect(self.stop)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.q_start_button)
        button_layout.addWidget(self.q_stop_button)

        self.q_save_button = QtWidgets.QPushButton("save clusters as audio files...", self)
        self.q_save_button.setToolTip("Writes one WAV file per cluster into a folder you choose.")
        self.q_save_button.clicked.connect(self.save_clusters)

        # ---------------- cluster list ----------------

        self.q_cluster_list = QtWidgets.QListWidget()
        self.q_cluster_list.currentRowChanged.connect(self.change_cluster_selection)
        self.refresh_cluster_list()

        self.q_refresh_timer = QTimer(self)
        self.q_refresh_timer.setInterval(1000)
        self.q_refresh_timer.timeout.connect(self.refresh_cluster_list)
        self.q_refresh_timer.start()

        # ---------------- OSC ----------------

        self.q_osc_label = QtWidgets.QLabel("")
        self.q_osc_label.setWordWrap(True)
        if self.control is not None:
            self.q_osc_label.setText(
                "OSC control listening on {}:{} (/synth/clusterlabel, /synth/audiofeature "
                "[one or more names], /synth/clustercount, /synth/clustermethod)".format(
                    self.control.ip, self.control.port))

        # ---------------- layout ----------------

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(device_box)
        layout.addWidget(excerpt_box)
        layout.addWidget(feature_box)
        layout.addWidget(cluster_box)
        layout.addLayout(button_layout)
        layout.addWidget(self.q_save_button)
        layout.addWidget(QtWidgets.QLabel("clusters (select to play):"))
        layout.addWidget(self.q_cluster_list, 1)
        layout.addWidget(self.q_osc_label)
        self.setLayout(layout)

        self.setGeometry(50, 50, 460, 920)
        self.setWindowTitle("Audio Clustering")

        self.update_excerpt_info()
        self.update_feature_info()

    # ---------------- helpers ----------------

    def get_checked_features(self):
        names = []
        for row in range(self.q_feature_list.count()):
            item = self.q_feature_list.item(row)
            if item.checkState() == Qt.Checked:
                names.append(item.data(Qt.UserRole))
        return names

    def get_length_ms(self):
        return round(self.q_length_spin.value() * 1000.0, 3)

    def get_offset_ms(self):
        """Hop between excerpt starts in ms = length - overlap (at least 1 ms)."""
        return max(1.0, round((self.q_length_spin.value() - self.q_overlap_spin.value()) * 1000.0, 3))

    def update_excerpt_info(self, *args):
        length_ms = self.get_length_ms()
        offset_ms = self.get_offset_ms()
        sample_rate = self.pipeline.get_sample_rate()
        count = cp.count_excerpts(self.pipeline.get_duration_ms(), length_ms, offset_ms)

        text = "hop {:.3f} s, about {} excerpts".format(offset_ms / 1000.0, count)
        minimum = cp.min_excerpt_length_ms(sample_rate)
        if length_ms < minimum:
            text += " - length is below the analysis minimum of {:.3f} s at {} Hz".format(minimum / 1000.0, sample_rate)
        elif count > cp.MAX_EXCERPTS:
            text += " - above the limit of {}".format(cp.MAX_EXCERPTS)
        self.q_excerpt_info_label.setText(text)

    def update_feature_info(self, dimensions=None):
        if dimensions is None:
            dimensions = self.model.get_feature_dimensions()
        if not dimensions:
            self.q_feature_info_label.setText("no features selected")
            return
        parts = ["{} ({})".format(n, d) for n, d in dimensions.items()]
        self.q_feature_info_label.setText("clustering on {} dimensions in total: {}".format(
            sum(dimensions.values()), ", ".join(parts)))

    def change_excerpt_length(self, length_s):
        self.q_overlap_spin.setMaximum(max(0.0, length_s - 0.001))
        self.update_excerpt_info()

    # ---------------- live handlers ----------------

    def start(self):
        self.synthesis.start()

    def stop(self):
        self.synthesis.stop()

    def change_output_device(self, idx):
        if 0 <= idx < len(self.output_device_mapping):
            self.synthesis.set_output_device(self.output_device_mapping[idx])

    def change_method(self, method_text):
        self.model.set_cluster_method(method_text)
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
        self.refresh_cluster_list()

    def change_cluster_count(self, value):
        self.model.set_cluster_count(value)
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
        self.refresh_cluster_list()

    def toggle_normalize(self, widget):
        self.model.set_normalize(widget.isChecked())
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
        self.refresh_cluster_list()

    def toggle_equal_weight(self, widget):
        self.model.set_equal_feature_weight(widget.isChecked())
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
        self.refresh_cluster_list()

    def change_cluster_selection(self, row):
        if row < 0:
            return
        item = self.q_cluster_list.item(row)
        if item is None:
            return
        label = item.data(Qt.UserRole)
        if label is not None:
            self.synthesis.setClusterLabel(int(label))

    def refresh_cluster_list(self):
        sizes = self.model.get_cluster_sizes()
        current_label = self.synthesis.get_cluster_label()

        self.q_cluster_list.blockSignals(True)
        self.q_cluster_list.clear()

        selected_row = -1
        for row, (label, count) in enumerate(sorted(sizes.items())):
            item = QtWidgets.QListWidgetItem("cluster {} ({} excerpts)".format(label, count))
            item.setData(Qt.UserRole, label)
            self.q_cluster_list.addItem(item)
            if label == current_label:
                selected_row = row

        if selected_row >= 0:
            self.q_cluster_list.setCurrentRow(selected_row)

        self.q_cluster_list.blockSignals(False)

    # ---------------- background tasks ----------------

    def _set_busy(self, busy, message=""):
        for widget in (self.q_load_file_button, self.q_apply_button, self.q_save_button,
                       self.q_start_button, self.q_stop_button, self.q_cluster_list,
                       self.q_feature_list, self.q_length_spin, self.q_overlap_spin,
                       self.q_method_combo, self.q_cluster_count_spin,
                       self.q_normalize_toggle, self.q_equal_weight_toggle):
            widget.setEnabled(not busy)
        self.q_status_label.setText(message)

    def _run_worker(self, worker):
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.finished.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        return self._thread

    # ---------------- analysis ----------------

    def apply_settings(self):
        self._start_analysis(None)

    def load_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load audio file", "", "Audio files (*.wav *.flac *.aiff *.aif *.ogg);;All files (*.*)")
        if not file_path:
            return
        self._start_analysis(file_path)

    def _start_analysis(self, file_path):
        feature_names = self.get_checked_features()
        if not feature_names:
            self.q_status_label.setText("select at least one audio feature.")
            return

        length_ms = self.get_length_ms()
        offset_ms = self.get_offset_ms()

        self.synthesis.stop()
        self._set_busy(True, "analyzing {} feature(s), {:.3f} s excerpts, {:.3f} s hop...".format(
            len(feature_names), length_ms / 1000.0, offset_ms / 1000.0))

        worker = _AnalysisWorker(self.pipeline, self.model, file_path, length_ms, offset_ms, feature_names)
        worker.finished.connect(self._on_analysis_finished)
        worker.failed.connect(self._on_analysis_failed)
        self._run_worker(worker).start()

    def _on_analysis_finished(self, info):
        self.synthesis.set_excerpt_params(info["length_ms"], info["offset_ms"], info["sample_rate"])
        self.synthesis.setClusterLabel(0)

        self.refresh_cluster_list()
        self.update_excerpt_info()
        self.update_feature_info(info["dimensions"])

        self._set_busy(False, "{} excerpts of {:.3f} s (hop {:.3f} s) at {} Hz.".format(
            info["excerpt_count"], info["length_ms"] / 1000.0, info["offset_ms"] / 1000.0, info["sample_rate"]))

    def _on_analysis_failed(self, error_message):
        self._set_busy(False, "analysis failed: {}".format(error_message))
        self.update_excerpt_info()

    # ---------------- save clusters ----------------

    def save_clusters(self):
        if self.model.get_label_count() == 0:
            self.q_status_label.setText("nothing to save: no clusters yet.")
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose folder for cluster audio files", "results/audio")
        if not folder:
            return

        self._set_busy(True, "saving {} clusters to '{}'...".format(self.model.get_label_count(), folder))

        worker = _ExportWorker(self.model, self.synthesis, folder)
        worker.progress.connect(self._on_save_progress)
        worker.finished.connect(self._on_save_finished)
        worker.failed.connect(self._on_save_failed)
        self._run_worker(worker).start()

    def _on_save_progress(self, done, total):
        self.q_status_label.setText("saving cluster {} of {}...".format(done, total))

    def _on_save_finished(self, info):
        total_seconds = sum(f[2] for f in info["files"])
        self._set_busy(False, "saved {} files ({:.1f} s of audio) to '{}'.".format(
            len(info["files"]), total_seconds, info["folder"]))

    def _on_save_failed(self, error_message):
        self._set_busy(False, "saving failed: {}".format(error_message))
