import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal

"""
AudioGui for the clustering tool.

NEW in this revision: a "load audio file..." button that lets the user pick
a different audio file, re-run excerpt extraction + feature computation +
clustering on it, and swap the results into the existing
model/synthesis/control objects - without restarting the application.

Because extraction + feature computation across potentially thousands of
excerpts can take anywhere from a couple of seconds to tens of seconds,
this runs on a background QThread (_LoadFileWorker) rather than blocking
the GUI thread. While loading:
- the load/start/stop buttons and cluster list are disabled
- a status label shows "loading..." / the eventual result or error
- once finished, model.set_data() swaps in the new excerpts/features,
  create_clusters() re-clusters using whatever method/count/normalize
  settings are already configured, and the cluster list refreshes.

Layout (top to bottom):
- output device dropdown
- audio feature dropdown (which descriptor to cluster on)
- cluster method dropdown (kmeans / minibatch_kmeans) + cluster count spinbox
- normalize-features toggle
- load audio file button + status label
- start / stop buttons (control cluster audio PLAYBACK)
- cluster list (one row per cluster label, showing excerpt count)
- OSC address display
"""

config = {
    "model": None,
    "synthesis": None,
    "control": None,
    "build_clustering_data": None,
    "excerpt_length_ms": 100,
    "excerpt_offset_ms": 90,
    "analysis_seconds": 60.0,
}


class _LoadFileWorker(QObject):
    """
    Runs build_clustering_data(file_path, ...) on a background thread.
    Emits finished(audio_excerpts, audio_features, sample_rate) on success,
    or failed(error_message) on exception.
    """
    finished = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, build_fn, file_path, excerpt_length_ms, excerpt_offset_ms, analysis_seconds):
        super().__init__()
        self.build_fn = build_fn
        self.file_path = file_path
        self.excerpt_length_ms = excerpt_length_ms
        self.excerpt_offset_ms = excerpt_offset_ms
        self.analysis_seconds = analysis_seconds

    def run(self):
        try:
            audio_excerpts, audio_features, sample_rate = self.build_fn(
                self.file_path,
                excerpt_length_ms=self.excerpt_length_ms,
                excerpt_offset_ms=self.excerpt_offset_ms,
                analysis_seconds=self.analysis_seconds,
            )
            self.finished.emit(audio_excerpts, audio_features, sample_rate)
        except Exception as e:
            self.failed.emit(str(e))


class AudioGui(QtWidgets.QWidget):

    def __init__(self, config):
        super().__init__()

        self.model = config["model"]
        self.synthesis = config["synthesis"]
        self.control = config.get("control", None)
        self.build_clustering_data = config.get("build_clustering_data", None)
        self.excerpt_length_ms = config.get("excerpt_length_ms", 100)
        self.excerpt_offset_ms = config.get("excerpt_offset_ms", 90)
        self.analysis_seconds = config.get("analysis_seconds", 60.0)

        self._load_thread = None
        self._load_worker = None

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

        # ---------------- feature / clustering controls ----------------

        self.q_feature_label = QtWidgets.QLabel("audio feature:")
        self.q_feature_combo = QtWidgets.QComboBox()
        self._populate_feature_combo()
        self.q_feature_combo.currentIndexChanged.connect(self.change_feature)

        self.q_method_label = QtWidgets.QLabel("cluster method:")
        self.q_method_combo = QtWidgets.QComboBox()
        self.q_method_combo.addItem("kmeans")
        self.q_method_combo.addItem("minibatch_kmeans")
        self.q_method_combo.setCurrentText(self.model.cluster_method)
        self.q_method_combo.currentTextChanged.connect(self.change_method)

        self.q_cluster_count_label = QtWidgets.QLabel("cluster count:")
        self.q_cluster_count_spin = QtWidgets.QSpinBox(self)
        self.q_cluster_count_spin.setMinimum(2)
        self.q_cluster_count_spin.setMaximum(500)
        self.q_cluster_count_spin.setValue(self.model.cluster_count)
        self.q_cluster_count_spin.valueChanged.connect(self.change_cluster_count)

        self.q_normalize_toggle = QtWidgets.QCheckBox("normalize features", self)
        self.q_normalize_toggle.setChecked(self.model.normalize)
        self.q_normalize_toggle.stateChanged.connect(lambda: self.toggle_normalize(self.q_normalize_toggle))

        self.q_feature_grid = QtWidgets.QGridLayout()
        self.q_feature_grid.addWidget(self.q_feature_label, 0, 0)
        self.q_feature_grid.addWidget(self.q_feature_combo, 0, 1)
        self.q_feature_grid.addWidget(self.q_method_label, 1, 0)
        self.q_feature_grid.addWidget(self.q_method_combo, 1, 1)
        self.q_feature_grid.addWidget(self.q_cluster_count_label, 2, 0)
        self.q_feature_grid.addWidget(self.q_cluster_count_spin, 2, 1)
        self.q_feature_grid.addWidget(self.q_normalize_toggle, 3, 0, 1, 2)

        # ---------------- load audio file ----------------

        self.q_load_file_button = QtWidgets.QPushButton("load audio file...", self)
        self.q_load_file_button.clicked.connect(self.load_file)

        self.q_load_status_label = QtWidgets.QLabel("")
        self.q_load_status_label.setWordWrap(True)

        self.q_load_grid = QtWidgets.QVBoxLayout()
        self.q_load_grid.addWidget(self.q_load_file_button)
        self.q_load_grid.addWidget(self.q_load_status_label)

        # ---------------- start / stop playback ----------------

        self.q_start_button = QtWidgets.QPushButton("start", self)
        self.q_start_button.clicked.connect(self.start)

        self.q_stop_button = QtWidgets.QPushButton("stop", self)
        self.q_stop_button.clicked.connect(self.stop)

        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_button, 0, 0)
        self.q_button_grid.addWidget(self.q_stop_button, 0, 1)

        # ---------------- cluster list ----------------

        self.q_cluster_list_label = QtWidgets.QLabel("clusters (select to play):")
        self.q_cluster_list = QtWidgets.QListWidget()
        self.q_cluster_list.currentRowChanged.connect(self.change_cluster_selection)

        self.q_cluster_grid = QtWidgets.QVBoxLayout()
        self.q_cluster_grid.addWidget(self.q_cluster_list_label)
        self.q_cluster_grid.addWidget(self.q_cluster_list)

        self.refresh_cluster_list()

        self.q_refresh_timer = QTimer(self)
        self.q_refresh_timer.setInterval(1000)
        self.q_refresh_timer.timeout.connect(self.refresh_cluster_list)
        self.q_refresh_timer.start()

        # ---------------- OSC control address display ----------------

        self.q_osc_label = QtWidgets.QLabel("")
        if self.control is not None:
            self.q_osc_label.setText(
                "OSC control listening on {}:{}  (/synth/clusterlabel, /synth/audiofeature, "
                "/synth/clustercount, /synth/clustermethod)".format(self.control.ip, self.control.port)
            )
        self.q_osc_label.setWordWrap(True)

        # ---------------- overall layout ----------------

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.q_device_grid, 0, 0)
        self.q_grid.addLayout(self.q_feature_grid, 1, 0)
        self.q_grid.addLayout(self.q_load_grid, 2, 0)
        self.q_grid.addLayout(self.q_button_grid, 3, 0)
        self.q_grid.addLayout(self.q_cluster_grid, 4, 0)
        self.q_grid.addWidget(self.q_osc_label, 5, 0)

        for row in range(6):
            self.q_grid.setRowStretch(row, 0)
        self.q_grid.setRowStretch(4, 1)

        self.setLayout(self.q_grid)

        self.setGeometry(50, 50, 420, 620)
        self.setWindowTitle("Audio Clustering")

    # ---------------- feature combo helper ----------------

    def _populate_feature_combo(self):
        self.q_feature_combo.blockSignals(True)
        self.q_feature_combo.clear()
        self.feature_names = self.model.get_feature_names()
        for name in self.feature_names:
            self.q_feature_combo.addItem(name)
        current_feature = self.model.get_current_feature()
        if current_feature in self.feature_names:
            self.q_feature_combo.setCurrentIndex(self.feature_names.index(current_feature))
        self.q_feature_combo.blockSignals(False)

    # ---------------- handlers ----------------

    def start(self):
        self.synthesis.start()

    def stop(self):
        self.synthesis.stop()

    def change_output_device(self, idx):
        if 0 <= idx < len(self.output_device_mapping):
            device_id = self.output_device_mapping[idx]
            self.synthesis.set_output_device(device_id)

    def change_feature(self, idx):
        if 0 <= idx < len(self.feature_names):
            feature_name = self.feature_names[idx]
            self.synthesis.selectAudioFeature(feature_name)
            self.refresh_cluster_list()

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
            item = QtWidgets.QListWidgetItem("cluster {}  ({} excerpts)".format(label, count))
            item.setData(Qt.UserRole, label)
            self.q_cluster_list.addItem(item)
            if label == current_label:
                selected_row = row

        if selected_row >= 0:
            self.q_cluster_list.setCurrentRow(selected_row)

        self.q_cluster_list.blockSignals(False)

    # ---------------- load audio file (background thread) ----------------

    def _set_busy(self, busy, message=""):
        self.q_load_file_button.setEnabled(not busy)
        self.q_start_button.setEnabled(not busy)
        self.q_stop_button.setEnabled(not busy)
        self.q_cluster_list.setEnabled(not busy)
        self.q_feature_combo.setEnabled(not busy)
        self.q_method_combo.setEnabled(not busy)
        self.q_cluster_count_spin.setEnabled(not busy)
        self.q_normalize_toggle.setEnabled(not busy)
        self.q_load_status_label.setText(message)

    def load_file(self):
        if self.build_clustering_data is None:
            self.q_load_status_label.setText("load-file pipeline not configured.")
            return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load audio file", "", "Audio files (*.wav *.flac *.aiff *.aif *.ogg);;All files (*.*)"
        )
        if not file_path:
            return

        # stop playback while the new file is being processed, since the
        # cluster currently playing may be replaced or become invalid
        self.synthesis.stop()

        self._set_busy(True, "loading and analyzing '{}'...".format(file_path))

        self._load_thread = QThread(self)
        self._load_worker = _LoadFileWorker(
            self.build_clustering_data, file_path,
            self.excerpt_length_ms, self.excerpt_offset_ms, self.analysis_seconds
        )
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    def _on_load_finished(self, audio_excerpts, audio_features, sample_rate):
        self.model.set_data(audio_excerpts, audio_features)
        self.model.create_clusters()

        # keep synthesis in sync with the newly loaded excerpts/sample rate
        self.synthesis.model = self.model
        self.synthesis.audio_sample_rate = sample_rate
        self.synthesis.setClusterLabel(0)

        self._populate_feature_combo()
        self.refresh_cluster_list()

        self._set_busy(False, "loaded {} excerpts at {} Hz.".format(audio_excerpts.shape[0], sample_rate))

    def _on_load_failed(self, error_message):
        self._set_busy(False, "failed to load file: {}".format(error_message))
