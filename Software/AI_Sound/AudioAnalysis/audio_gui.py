import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vispy import scene
from vispy.scene import SceneCanvas, visuals
from threading import Thread, Event
import time
from time import sleep

config = {"pipeline": None, "sender": None, "receiver": None}

"""
BarViewOptimized / DataViewOptimized / CanvasOptimized are reused verbatim
from mocap_gui.py (same vispy-based real-time bar chart visualization).
"""


class BarViewOptimized:
    def __init__(self, max_value_count, colors, parent_view=None):
        self.max_value_count = max_value_count
        self.value_count = max_value_count
        self.colors = colors
        self.parent_view = parent_view

        self.bar_width = 1.0 / self.value_count
        self.bar_centers_x = np.linspace(self.bar_width / 2, 1.0 - self.bar_width / 2, self.value_count)

        self.vertices = np.zeros((self.value_count * 4, 3), dtype=np.float32)
        self.colors_arr = np.zeros((self.value_count * 4, 4), dtype=np.float32)

        self.faces = np.zeros((self.value_count * 2, 3), dtype=np.uint32)

        for i in range(self.value_count):
            base_face = i * 2
            base_vert = i * 4
            self.faces[base_face] = [base_vert, base_vert + 1, base_vert + 2]
            self.faces[base_face + 1] = [base_vert, base_vert + 2, base_vert + 3]

        self._initialize_geometry()

        self.mesh = visuals.Mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.colors_arr,
            mode='triangles',
            parent=self.parent_view
        )

    def _initialize_geometry(self):
        half_width = self.bar_width / 2

        for i, center_x in enumerate(self.bar_centers_x):
            base_index = i * 4
            bottom_y = 0.0
            top_y = 0.01

            bl = (center_x - half_width, bottom_y, 0)
            br = (center_x + half_width, bottom_y, 0)
            tr = (center_x + half_width, top_y, 0)
            tl = (center_x - half_width, top_y, 0)

            self.vertices[base_index] = bl
            self.vertices[base_index + 1] = br
            self.vertices[base_index + 2] = tr
            self.vertices[base_index + 3] = tl

            color = self.colors[i] if i < len(self.colors) else (1, 1, 1, 0.8)
            self.colors_arr[base_index:base_index+4] = color

    def resetValueCount(self, value_count):
        if value_count == self.value_count:
            return

        self.value_count = value_count
        self.bar_width = 0.9 / self.value_count
        self.bar_centers_x = np.linspace(self.bar_width / 2, 1.0 - self.bar_width / 2, self.value_count)

        self.faces = np.zeros((self.value_count * 2, 3), dtype=np.uint32)
        for i in range(self.value_count):
            base_face = i * 2
            base_vert = i * 4
            self.faces[base_face] = [base_vert, base_vert + 1, base_vert + 2]
            self.faces[base_face + 1] = [base_vert, base_vert + 2, base_vert + 3]

        self._initialize_geometry()

    def update(self, values):
        value_count = min(values.shape[0], self.max_value_count)

        if value_count != self.value_count:
            self.resetValueCount(value_count)

        half_width = self.bar_width / 2

        for i in range(self.value_count):
            base_index = i * 4
            center_x = self.bar_centers_x[i]
            value = values[i] if i < len(values) else 0

            bottom_y = min(value, -0.0001)
            top_y = max(value, 0.0001)

            bl = (center_x - half_width, bottom_y, 0)
            br = (center_x + half_width, bottom_y, 0)
            tr = (center_x + half_width, top_y, 0)
            tl = (center_x - half_width, top_y, 0)

            self.vertices[base_index] = bl
            self.vertices[base_index + 1] = br
            self.vertices[base_index + 2] = tr
            self.vertices[base_index + 3] = tl

        self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=self.colors_arr)


class DataViewOptimized:
    def __init__(self, title, max_value_dim, value_range, time_steps, colors):
        self.title = title
        self.max_value_dim = max_value_dim
        self.value_dim = max_value_dim
        self.value_range = value_range
        self.time_steps = time_steps
        self.colors = colors

        if value_range[0] > value_range[1]:
            self.autoscale = True
        else:
            self.autoscale = False

        self.canvas = SceneCanvas()
        self.grid = self.canvas.central_widget.add_grid()

        yaxis = scene.AxisWidget(
            orientation='left',
            axis_font_size=12,
            axis_label_margin=50,
            tick_label_margin=5
        )
        yaxis.width_max = 80
        self.grid.add_widget(yaxis, row=0, col=0)

        self.bar_view = self.grid.add_view(0, 1, bgcolor="black")

        self.bars = BarViewOptimized(self.max_value_dim, self.colors, self.bar_view.scene)

        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(self.value_range[0], self.value_range[1]))

        yaxis.link_view(self.bar_view)

    def set_value_range(self, value_range):
        self.value_range = value_range
        if value_range[0] > value_range[1]:
            self.autoscale = True
        else:
            self.autoscale = False
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(value_range[0], value_range[1]))

    def update_data(self, data):

        data_dim = min(data.shape[0], self.max_value_dim)
        data = data[:data_dim]

        if self.autoscale:
            range_changed = False
            min_value = np.min(data)
            max_value = np.max(data)

            if self.value_range[0] > min_value:
                self.value_range[0] = min_value
                range_changed = True
            if self.value_range[1] < max_value:
                self.value_range[1] = max_value
                range_changed = True

            if range_changed:
                self.bar_view.camera.set_range(x=(0.0, 1.0), y=(self.value_range[0], self.value_range[1]))

        self.bars.update(data)


class CanvasOptimized:
    def __init__(self, size):
        self.size = size
        self.canvas = SceneCanvas(size=size)
        self.grid = self.canvas.central_widget.add_grid()
        self.views = {}

    def add_sensor_view(self, name, max_value_dim, value_range, time_steps, colors):
        sensor_view = DataViewOptimized(name, max_value_dim, value_range, time_steps, colors)
        self.grid.add_widget(sensor_view.canvas.central_widget, len(self.views), 0)
        self.views[name] = sensor_view

    def set_value_range(self, name, value_range):
        if name in self.views:
            self.views[name].set_value_range(value_range)

    def update_data(self, new_data):

        key = list(new_data.keys())[0]
        value = list(new_data.values())[0]

        if key in self.views:
            self.views[key].update_data(value)


class AudioGui(QtWidgets.QWidget):

    def __init__(self, config):
        super().__init__()

        self.pipeline = config["pipeline"]
        self.sender = config["sender"]
        self.receiver = config["receiver"]

        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)

        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)

        fps = 1.0 / self.pipeline.updateInterval

        self.q_fps = QtWidgets.QSpinBox(self)
        self.q_fps.setMinimum(0)
        self.q_fps.setMaximum(200)
        self.q_fps.setValue(int(fps))
        self.q_fps.valueChanged.connect(self.change_fps)

        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_buttom, 0, 0)
        self.q_button_grid.addWidget(self.q_stop_buttom, 0, 1)
        self.q_button_grid.addWidget(self.q_fps, 0, 2)

        # ---------------- device selection (top of window) ----------------

        self.q_input_device_label = QtWidgets.QLabel("input device:")
        self.q_input_device_combo = QtWidgets.QComboBox()
        self.input_device_mapping = []
        for idx, name in self.receiver.list_input_devices():
            self.input_device_mapping.append(idx)
            self.q_input_device_combo.addItem("{}: {}".format(idx, name))
        self.q_input_device_combo.currentIndexChanged.connect(self.change_input_device)

        self.q_output_device_label = QtWidgets.QLabel("output device:")
        self.q_output_device_combo = QtWidgets.QComboBox()
        self.output_device_mapping = []
        for idx, name in self.receiver.list_output_devices():
            self.output_device_mapping.append(idx)
            self.q_output_device_combo.addItem("{}: {}".format(idx, name))
        self.q_output_device_combo.currentIndexChanged.connect(self.change_output_device)

        self.q_device_grid = QtWidgets.QGridLayout()
        self.q_device_grid.addWidget(self.q_input_device_label, 0, 0)
        self.q_device_grid.addWidget(self.q_input_device_combo, 0, 1)
        self.q_device_grid.addWidget(self.q_output_device_label, 1, 0)
        self.q_device_grid.addWidget(self.q_output_device_combo, 1, 1)

        # ---------------- source / transport controls ----------------

        self.q_load_file_button = QtWidgets.QPushButton("load audio file...", self)
        self.q_load_file_button.clicked.connect(self.load_file)

        self.q_mic_toggle = QtWidgets.QCheckBox("use microphone", self)
        self.q_mic_toggle.setChecked(self.receiver.mode == "mic")
        self.q_mic_toggle.stateChanged.connect(lambda: self.toggle_mic(self.q_mic_toggle))

        self.q_loop_toggle = QtWidgets.QCheckBox("loop", self)
        self.q_loop_toggle.setChecked(self.receiver.get_loop())
        self.q_loop_toggle.stateChanged.connect(lambda: self.toggle_loop(self.q_loop_toggle))

        self.q_source_grid = QtWidgets.QGridLayout()
        self.q_source_grid.addWidget(self.q_load_file_button, 0, 0)
        self.q_source_grid.addWidget(self.q_mic_toggle, 0, 1)
        self.q_source_grid.addWidget(self.q_loop_toggle, 0, 2)

        # transport sliders: play head, region start, region end (all in seconds,
        # represented as integer slider ticks at millisecond resolution)
        self._slider_scale = 1000.0  # slider units per second

        duration = self.receiver.get_file_duration()
        duration_ticks = max(int(duration * self._slider_scale), 1)

        self.q_playhead_label = QtWidgets.QLabel("play head: 0.00 s")
        self.q_playhead_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_playhead_slider.setMinimum(0)
        self.q_playhead_slider.setMaximum(duration_ticks)
        self.q_playhead_slider.setValue(0)
        self.q_playhead_slider.valueChanged.connect(self.change_playhead)

        self.q_region_start_label = QtWidgets.QLabel("region start: 0.00 s")
        self.q_region_start_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_region_start_slider.setMinimum(0)
        self.q_region_start_slider.setMaximum(duration_ticks)
        self.q_region_start_slider.setValue(0)
        self.q_region_start_slider.valueChanged.connect(self.change_region_start)

        self.q_region_end_label = QtWidgets.QLabel("region end: {:.2f} s".format(duration))
        self.q_region_end_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_region_end_slider.setMinimum(0)
        self.q_region_end_slider.setMaximum(duration_ticks)
        self.q_region_end_slider.setValue(duration_ticks)
        self.q_region_end_slider.valueChanged.connect(self.change_region_end)

        self.q_transport_grid = QtWidgets.QGridLayout()
        self.q_transport_grid.addWidget(self.q_playhead_label, 0, 0)
        self.q_transport_grid.addWidget(self.q_playhead_slider, 0, 1)
        self.q_transport_grid.addWidget(self.q_region_start_label, 1, 0)
        self.q_transport_grid.addWidget(self.q_region_start_slider, 1, 1)
        self.q_transport_grid.addWidget(self.q_region_end_label, 2, 0)
        self.q_transport_grid.addWidget(self.q_region_end_slider, 2, 1)

        # timer-driven play-head display update (reads receiver's actual
        # play head so the slider tracks playback without fighting user drags)
        from PyQt5.QtCore import QTimer
        self._suppress_playhead_signal = False
        self.q_playhead_timer = QTimer(self)
        self.q_playhead_timer.setInterval(100)
        self.q_playhead_timer.timeout.connect(self.refresh_playhead_display)
        self.q_playhead_timer.start()

        # canvas
        self.canvas = CanvasOptimized((400, 400))
        self.canvas.add_sensor_view("data", 128, [1000.0, -1000.0], 100, [(1.0, 1.0, 1.0, 0.8)] * 128)

        self.canvas_active = True

        self.q_canvas_toggle = QtWidgets.QCheckBox("canvas", self)
        self.q_canvas_toggle.stateChanged.connect(lambda: self.toggle_canvas(self.q_canvas_toggle))
        self.q_canvas_toggle.setChecked(self.canvas_active)

        self.q_canvas_grid = QtWidgets.QGridLayout()
        self.q_canvas_grid.addWidget(self.canvas.canvas.native, 0, 0)
        self.q_canvas_grid.addWidget(self.q_canvas_toggle, 1, 0)

        # send items - the CHECKBOX now controls ONLY whether a descriptor is
        # sent via OSC. Whether a descriptor is DISPLAYED in the canvas is
        # controlled purely by highlighting/selecting it in the list
        # (see currentItemChanged -> on_item_highlighted below). Computation
        # (pipeline.enabled) is the union of "checked for OSC" OR "currently
        # highlighted for display", so unchecking an item only stops OSC
        # sending if it is not also being viewed, and viewing an unchecked
        # item still works (it is computed on demand while highlighted).
        self.sendItems = dict(self.pipeline.enabled)
        self.showItem = ""

        self.q_sendItems = QtWidgets.QListWidget()
        for key, value in self.sendItems.items():
            QtWidgets.QListWidgetItem(key, self.q_sendItems)

        self.q_sendItems_layout = QtWidgets.QVBoxLayout()
        self.q_sendItems_layout.addWidget(self.q_sendItems)

        for i in range(self.q_sendItems.count()):
            q_sendItem = self.q_sendItems.item(i)
            q_sendItem.setFlags(q_sendItem.flags() | Qt.ItemIsUserCheckable)

            q_checked = self.sendItems[q_sendItem.text()]

            if q_checked == False:
                q_sendItem.setCheckState(Qt.Unchecked)
            else:
                q_sendItem.setCheckState(Qt.Checked)

        self.q_sendItems.itemChanged.connect(lambda: self.change_send_item(self.q_sendItems))
        self.q_sendItems.currentItemChanged.connect(self.on_item_highlighted)

        # osc sender
        self.sender_active = True

        self.q_sender_toggle = QtWidgets.QCheckBox("osc", self)
        self.q_sender_toggle.stateChanged.connect(lambda: self.toggle_sender(self.q_sender_toggle))
        self.q_sender_toggle.setChecked(self.sender_active)

        sender_ip_string, sender_port = self.sender.get_address()
        sender_ip_list = sender_ip_string.split(".")
        sender_ip = [int(el) for el in sender_ip_list]

        self.sender_ip = sender_ip
        self.sender_port = sender_port

        self.q_sender_ip = []
        for i in range(4):
            _w = QtWidgets.QSpinBox(self)
            _w.setMinimum(0)
            _w.setMaximum(255)
            _w.setValue(self.sender_ip[i])
            _w.valueChanged.connect(lambda: self.change_sender_ip(_w))

            self.q_sender_ip.append(_w)
        self.q_sender_port = QtWidgets.QSpinBox(self)
        self.q_sender_port.setMinimum(0)
        self.q_sender_port.setMaximum(65535)
        self.q_sender_port.setValue(self.sender_port)
        self.q_sender_port.valueChanged.connect(lambda: self.change_sender_port(self.q_sender_port))

        self.q_sender_grid = QtWidgets.QGridLayout()
        self.q_sender_grid.addWidget(self.q_sender_toggle, 0, 0)
        self.q_sender_grid.addWidget(self.q_sender_ip[0], 0, 1)
        self.q_sender_grid.addWidget(self.q_sender_ip[1], 0, 2)
        self.q_sender_grid.addWidget(self.q_sender_ip[2], 0, 3)
        self.q_sender_grid.addWidget(self.q_sender_ip[3], 0, 4)
        self.q_sender_grid.addWidget(self.q_sender_port, 0, 5)

        # ---------------- overall layout: devices -> source -> transport -> rest ----------------

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.q_device_grid, 0, 0)
        self.q_grid.addLayout(self.q_source_grid, 1, 0)
        self.q_grid.addLayout(self.q_transport_grid, 2, 0)
        self.q_grid.addLayout(self.q_button_grid, 3, 0)
        self.q_grid.addLayout(self.q_canvas_grid, 4, 0)
        self.q_grid.addLayout(self.q_sendItems_layout, 5, 0)
        self.q_grid.addLayout(self.q_sender_grid, 6, 0)

        for row in range(7):
            self.q_grid.setRowStretch(row, 0)

        self.setLayout(self.q_grid)

        self.setGeometry(50, 50, 560, 780)
        self.setWindowTitle("Audio Analysis")

    def start(self):
        """
        Starts both the analysis/OSC update thread and the audio
        source/transport (file playback or microphone capture), so
        pressing "start" actually produces sound as well as descriptors.
        """
        self.receiver.start()

        self.data_thread_event = Event()
        self.data_thread = Thread(target=self.update)

        self.data_thread.start()

    def stop(self):
        """
        Stops both the analysis/OSC update thread and the audio
        source/transport.
        """
        self.data_thread_event.set()
        self.data_thread.join()

        self.receiver.stop()

    def change_fps(self, fps):

        update_interval = 1.0 / int(fps)
        self.pipeline.setUpdateInterval(update_interval)

    # ---------------- source / transport handlers ----------------

    def load_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load audio file", "", "Audio files (*.wav *.flac *.aiff *.aif *.ogg);;All files (*.*)"
        )
        if not file_path:
            return

        was_playing = hasattr(self, "data_thread") and self.data_thread.is_alive()
        if was_playing:
            self.stop()

        self.q_mic_toggle.blockSignals(True)
        self.q_mic_toggle.setChecked(False)
        self.q_mic_toggle.blockSignals(False)

        self.receiver.load_file(file_path)
        self.pipeline.reconfigure()

        duration = self.receiver.get_file_duration()
        duration_ticks = max(int(duration * self._slider_scale), 1)

        for slider, value in (
            (self.q_playhead_slider, 0),
            (self.q_region_start_slider, 0),
            (self.q_region_end_slider, duration_ticks),
        ):
            slider.blockSignals(True)
            slider.setMaximum(duration_ticks)
            slider.setValue(value)
            slider.blockSignals(False)

        self.q_region_end_label.setText("region end: {:.2f} s".format(duration))
        self.q_region_start_label.setText("region start: 0.00 s")
        self.q_playhead_label.setText("play head: 0.00 s")

        if was_playing:
            self.start()

    def toggle_mic(self, widget):
        use_mic = widget.isChecked()

        was_playing = hasattr(self, "data_thread") and self.data_thread.is_alive()
        if was_playing:
            self.stop()

        self.receiver.set_mode("mic" if use_mic else "file")
        self.pipeline.reconfigure()

        if was_playing:
            self.start()

    def toggle_loop(self, widget):
        self.receiver.set_loop(widget.isChecked())

    def change_playhead(self, value):
        if self._suppress_playhead_signal:
            return
        seconds = value / self._slider_scale
        self.receiver.set_play_head(seconds)
        self.q_playhead_label.setText("play head: {:.2f} s".format(seconds))

    def change_region_start(self, value):
        seconds = value / self._slider_scale
        end_seconds = self.q_region_end_slider.value() / self._slider_scale
        self.receiver.set_region(seconds, end_seconds)
        self.q_region_start_label.setText("region start: {:.2f} s".format(seconds))

    def change_region_end(self, value):
        seconds = value / self._slider_scale
        start_seconds = self.q_region_start_slider.value() / self._slider_scale
        self.receiver.set_region(start_seconds, seconds)
        self.q_region_end_label.setText("region end: {:.2f} s".format(seconds))

    def refresh_playhead_display(self):
        if self.receiver.mode != "file":
            return
        seconds = self.receiver.get_play_head()
        ticks = int(seconds * self._slider_scale)

        self._suppress_playhead_signal = True
        self.q_playhead_slider.setValue(ticks)
        self._suppress_playhead_signal = False

        self.q_playhead_label.setText("play head: {:.2f} s".format(seconds))

    def change_input_device(self, idx):
        if 0 <= idx < len(self.input_device_mapping):
            device_id = self.input_device_mapping[idx]
            self.receiver.set_input_device(device_id)

    def change_output_device(self, idx):
        if 0 <= idx < len(self.output_device_mapping):
            device_id = self.output_device_mapping[idx]
            self.receiver.set_output_device(device_id)

    # ---------------- update loop ----------------

    def update(self):

        while self.data_thread_event.is_set() == False:

            start_time = time.time()

            self.pipeline.update()
            self.update_osc()
            if self.canvas_active == True:
                self.update_view()

            end_time = time.time()

            next_update_interval = max(self.pipeline.updateInterval - (end_time - start_time), 0.0)

            sleep(next_update_interval)

    def update_osc(self):

        if self.sendItems["rms"] == True:
            osc_values = np.reshape(self.pipeline.rms, (-1)).tolist()
            self.sender.send("/audio/0/rms", osc_values)
        if self.sendItems.get("rms_smooth", False) == True:
            osc_values = np.reshape(self.pipeline.rmsSmooth, (-1)).tolist()
            self.sender.send("/audio/0/rms_smooth", osc_values)
        if self.sendItems["zero_crossing_rate"] == True:
            osc_values = np.reshape(self.pipeline.zero_crossing_rate, (-1)).tolist()
            self.sender.send("/audio/0/zero_crossing_rate", osc_values)
        if self.sendItems["chroma_stft"] == True:
            osc_values = np.reshape(self.pipeline.chroma_stft, (-1)).tolist()
            self.sender.send("/audio/0/chroma_stft", osc_values)
        if self.sendItems["tonnetz"] == True:
            osc_values = np.reshape(self.pipeline.tonnetz, (-1)).tolist()
            self.sender.send("/audio/0/tonnetz", osc_values)
        if self.sendItems["mel_spectrogram"] == True:
            osc_values = np.reshape(self.pipeline.mel_spectrogram, (-1)).tolist()
            self.sender.send("/audio/0/mel_spectrogram", osc_values)
        if self.sendItems["mfcc"] == True:
            osc_values = np.reshape(self.pipeline.mfcc, (-1)).tolist()
            self.sender.send("/audio/0/mfcc", osc_values)
        if self.sendItems["spectral_centroid"] == True:
            osc_values = np.reshape(self.pipeline.spectral_centroid, (-1)).tolist()
            self.sender.send("/audio/0/spectral_centroid", osc_values)
        if self.sendItems.get("spectral_centroid_smooth", False) == True:
            osc_values = np.reshape(self.pipeline.spectral_centroidSmooth, (-1)).tolist()
            self.sender.send("/audio/0/spectral_centroid_smooth", osc_values)
        if self.sendItems["spectral_bandwidth"] == True:
            osc_values = np.reshape(self.pipeline.spectral_bandwidth, (-1)).tolist()
            self.sender.send("/audio/0/spectral_bandwidth", osc_values)
        if self.sendItems["spectral_contrast"] == True:
            osc_values = np.reshape(self.pipeline.spectral_contrast, (-1)).tolist()
            self.sender.send("/audio/0/spectral_contrast", osc_values)
        if self.sendItems["spectral_flatness"] == True:
            osc_values = np.reshape(self.pipeline.spectral_flatness, (-1)).tolist()
            self.sender.send("/audio/0/spectral_flatness", osc_values)
        if self.sendItems["spectral_rolloff"] == True:
            osc_values = np.reshape(self.pipeline.spectral_rolloff, (-1)).tolist()
            self.sender.send("/audio/0/spectral_rolloff", osc_values)
        if self.sendItems["poly_features"] == True:
            osc_values = np.reshape(self.pipeline.poly_features, (-1)).tolist()
            self.sender.send("/audio/0/poly_features", osc_values)
        if self.sendItems["onset_strength"] == True:
            osc_values = np.reshape(self.pipeline.onset_strength, (-1)).tolist()
            self.sender.send("/audio/0/onset_strength", osc_values)
        if self.sendItems.get("onset_strength_smooth", False) == True:
            osc_values = np.reshape(self.pipeline.onset_strengthSmooth, (-1)).tolist()
            self.sender.send("/audio/0/onset_strength_smooth", osc_values)
        if self.sendItems["chroma_cqt"] == True:
            osc_values = np.reshape(self.pipeline.chroma_cqt, (-1)).tolist()
            self.sender.send("/audio/0/chroma_cqt", osc_values)
        if self.sendItems["chroma_cens"] == True:
            osc_values = np.reshape(self.pipeline.chroma_cens, (-1)).tolist()
            self.sender.send("/audio/0/chroma_cens", osc_values)
        if self.sendItems["chroma_vqt"] == True:
            osc_values = np.reshape(self.pipeline.chroma_vqt, (-1)).tolist()
            self.sender.send("/audio/0/chroma_vqt", osc_values)
        if self.sendItems["tempo"] == True:
            osc_values = np.reshape(self.pipeline.tempo, (-1)).tolist()
            self.sender.send("/audio/0/tempo", osc_values)
        if self.sendItems["tempogram"] == True:
            osc_values = np.reshape(self.pipeline.tempogram, (-1)).tolist()
            self.sender.send("/audio/0/tempogram", osc_values)
        if self.sendItems["fourier_tempogram"] == True:
            osc_values = np.reshape(self.pipeline.fourier_tempogram, (-1)).tolist()
            self.sender.send("/audio/0/fourier_tempogram", osc_values)
        if self.sendItems["tempogram_ratio"] == True:
            osc_values = np.reshape(self.pipeline.tempogram_ratio, (-1)).tolist()
            self.sender.send("/audio/0/tempogram_ratio", osc_values)

    def update_view(self):
        """
        Displays whichever descriptor is currently highlighted/selected in
        the list (self.showItem), independent of its OSC checkbox state.
        """
        if not self.showItem:
            return

        attr_map = {
            "rms": "rms",
            "zero_crossing_rate": "zero_crossing_rate",
            "chroma_stft": "chroma_stft",
            "tonnetz": "tonnetz",
            "mel_spectrogram": "mel_spectrogram",
            "mfcc": "mfcc",
            "spectral_centroid": "spectral_centroid",
            "spectral_bandwidth": "spectral_bandwidth",
            "spectral_contrast": "spectral_contrast",
            "spectral_flatness": "spectral_flatness",
            "spectral_rolloff": "spectral_rolloff",
            "poly_features": "poly_features",
            "onset_strength": "onset_strength",
            "chroma_cqt": "chroma_cqt",
            "chroma_cens": "chroma_cens",
            "chroma_vqt": "chroma_vqt",
            "tempo": "tempo",
            "tempogram": "tempogram",
            "fourier_tempogram": "fourier_tempogram",
            "tempogram_ratio": "tempogram_ratio",
        }

        attr_name = attr_map.get(self.showItem)
        if attr_name is not None:
            values = getattr(self.pipeline, attr_name)
            view_data = {"data": np.reshape(values, (-1))}
            self.canvas.update_data(view_data)

    def on_item_highlighted(self, current, previous):
        """
        Called whenever the highlighted/current item in the descriptor list
        changes. Highlighting alone now controls what is displayed in the
        canvas - it is completely independent of the item's OSC checkbox.
        Highlighting a descriptor also force-enables its computation in the
        pipeline (so it has fresh data to display) even if its OSC checkbox
        is unchecked; un-highlighting it drops that forced enable unless the
        checkbox is still checked (for OSC sending).
        """
        if previous is not None:
            prev_text = previous.text()
            if not self.sendItems.get(prev_text, False):
                self.pipeline.set_enabled(prev_text, False)

        if current is not None:
            text = current.text()
            self.showItem = text
            self.canvas.set_value_range("data", [1000, -1000])
            self.pipeline.set_enabled(text, True)
        else:
            self.showItem = ""

    def change_send_item(self, widget):
        """
        The checkbox controls ONLY whether a descriptor is sent via OSC.
        Computation is re-derived as the union of "checked" OR "currently
        highlighted for display", so unchecking a highlighted item keeps it
        computed (for display) but stops sending it; checking an item
        always enables its computation (for OSC), regardless of display.
        """
        for i in range(widget.count()):
            q_sendItem = widget.item(i)

            q_text = q_sendItem.text()
            q_state = q_sendItem.checkState()

            checked = (q_state == Qt.Checked)
            self.sendItems[q_text] = checked

            is_highlighted = (q_text == self.showItem)
            self.pipeline.set_enabled(q_text, checked or is_highlighted)

    def toggle_canvas(self, widget):
        self.canvas_active = widget.isChecked()

    def toggle_sender(self, widget):
        self.sender_on = widget.isChecked()

        self.sender.set_active(self.sender_on)

    def change_sender_ip(self, widget):

        sender_widget = super().sender()

        if sender_widget == self.q_sender_ip[0]:
            self.sender_ip[0] = sender_widget.value()
        elif sender_widget == self.q_sender_ip[1]:
            self.sender_ip[1] = sender_widget.value()
        elif sender_widget == self.q_sender_ip[2]:
            self.sender_ip[2] = sender_widget.value()
        else:
            self.sender_ip[3] = sender_widget.value()

        sender_active = self.sender.get_active()
        sender_ip = "{}.{}.{}.{}".format(self.sender_ip[0], self.sender_ip[1], self.sender_ip[2], self.sender_ip[3])
        sender_port = self.sender_port

        if sender_active == True:
            self.sender.set_active(False)

        self.sender.set_address(sender_ip, sender_port)

        if sender_active == True:
            self.sender.set_active(True)

    def change_sender_port(self, widget):
        self.sender_port = widget.value()

        sender_active = self.sender.get_active()
        sender_ip = "{}.{}.{}.{}".format(self.sender_ip[0], self.sender_ip[1], self.sender_ip[2], self.sender_ip[3])
        sender_port = self.sender_port

        if sender_active == True:
            self.sender.set_active(False)

        self.sender.set_address(sender_ip, sender_port)

        if sender_active == True:
            self.sender.set_active(True)
