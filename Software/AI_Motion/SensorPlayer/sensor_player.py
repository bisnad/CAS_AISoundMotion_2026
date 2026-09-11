"""
Imports
"""

import sys
import time
import queue
import colorsys
import numpy as np

# OSC
from pythonosc import udp_client

# GUI & VisPy
from PyQt5 import QtWidgets, QtCore
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas, visuals

"""
Playback Thread
"""

class PlaybackThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(float)
    playback_finished = QtCore.pyqtSignal()
    
    def __init__(self, vis_queue, parent=None):
        super().__init__(parent)
        self.vis_queue = vis_queue
        self.events = []  
        self.ip = "127.0.0.1"
        self.port = 9007
        self.is_playing = False
        
        self.current_sec = 0.0
        self.start_sec = 0.0
        self.end_sec = 0.0
        self.loop = False
        
    def load_events(self, events, ip, port):
        self.events = events
        self.ip = ip
        self.port = port
        
    def configure_playback(self, current_sec, start_sec, end_sec, loop):
        self.current_sec = current_sec
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.loop = loop
        
    def stop(self):
        self.is_playing = False
        
    def run(self):
        if not self.events:
            self.playback_finished.emit()
            return
            
        self.is_playing = True
        client = udp_client.SimpleUDPClient(self.ip, self.port)
        
        # Determine logical starting point
        actual_start = max(self.start_sec, min(self.current_sec, self.end_sec))
        
        while self.is_playing:
            # Slice events
            active_events = [e for e in self.events if actual_start <= e[0] <= self.end_sec]
            
            if not active_events:
                if self.loop:
                    actual_start = self.start_sec
                    continue
                else:
                    break
                    
            thread_start_time = time.perf_counter()
            first_event_ts = actual_start
            last_gui_update = thread_start_time
            
            for timestamp, addr, values in active_events:
                if not self.is_playing:
                    break 
                    
                # Dynamic abort if end slider is dragged backwards mid-playback
                if timestamp > self.end_sec:
                    break
                    
                target_time = thread_start_time + (timestamp - first_event_ts)
                
                # Hybrid Precision Wait
                while True:
                    if not self.is_playing:
                        break
                    now = time.perf_counter()
                    if now >= target_time:
                        break
                    delta = target_time - now
                    if delta > 0.002:  
                        time.sleep(0.001)
                    else:
                        pass 
                
                # Check again in case we broke out of the sleep loop early
                if not self.is_playing:
                    break
                
                # Fire OSC message
                client.send_message(addr, values)
                self.vis_queue.put((timestamp, addr, values))
                
                # Update GUI Slider
                if now - last_gui_update > 0.05:
                    self.progress_update.emit(timestamp)
                    last_gui_update = now
            
            # Cascading break to exit the outer while loop safely
            if not self.is_playing:
                break
                
            if self.loop and self.is_playing:
                actual_start = self.start_sec # Instant wrap
            else:
                if active_events and self.is_playing:
                    self.progress_update.emit(min(self.end_sec, active_events[-1][0]))
                break
                
        self.is_playing = False
        self.playback_finished.emit()


"""
VisPy-based GUI Components
"""

def generate_colors(n):
    return [colorsys.hsv_to_rgb(i/n, 0.8, 0.9) + (1.0,) for i in range(n)]

class TimePlotView:
    def __init__(self, value_count, time_count, colors, parent_view):
        self.value_count = value_count
        self.time_count = time_count
        self.colors = colors
        self.parent_view = parent_view

        self.values = np.zeros((self.value_count, self.time_count), dtype=np.float32)
        self.idx = 0
        self.time_line = np.linspace(0.0, 1.0, self.time_count, dtype=np.float32)
        self._initialize_line_visual()

    def _initialize_line_visual(self):
        self.pos_array = np.zeros((self.value_count * self.time_count, 2), dtype=np.float32)
        self.pos_array[:, 0] = np.tile(self.time_line, self.value_count)
        self.color_array = np.zeros((self.value_count * self.time_count, 4), dtype=np.float32)
        for i in range(self.value_count):
            self.color_array[i * self.time_count : (i+1) * self.time_count] = self.colors[i]

        connections = []
        for i in range(self.value_count):
            start_idx = i * self.time_count
            for j in range(self.time_count - 1):
                connections.append([start_idx + j, start_idx + j + 1])
        self.connect_array = np.array(connections, dtype=np.uint32)

        if hasattr(self, 'line'):
            self.line.set_data(pos=self.pos_array, color=self.color_array, connect=self.connect_array)
        else:
            self.line = visuals.Line(pos=self.pos_array, color=self.color_array,
                                     connect=self.connect_array, width=1.5,
                                     method='gl', parent=self.parent_view)

    def set_time_steps(self, new_count):
        if new_count == self.time_count: return
        ordered_vals = np.concatenate((self.values[:, self.idx:], self.values[:, :self.idx]), axis=1)
        new_values = np.zeros((self.value_count, new_count), dtype=np.float32)
        keep = min(self.time_count, new_count)
        new_values[:, -keep:] = ordered_vals[:, -keep:]
        self.values = new_values
        self.time_count = new_count
        self.idx = 0
        self.time_line = np.linspace(0.0, 1.0, self.time_count, dtype=np.float32)
        self._initialize_line_visual()

    def update(self, values):
        self.values[:len(values), self.idx] = values
        self.idx = (self.idx + 1) % self.time_count
        ordered_vals = np.concatenate((self.values[:, self.idx:], self.values[:, :self.idx]), axis=1)
        self.pos_array[:, 1] = ordered_vals.ravel()
        self.line.set_data(pos=self.pos_array)

class BarView:
    def __init__(self, max_value_count, colors, parent_view=None):
        self.value_count = max_value_count
        self.colors = colors
        self.parent_view = parent_view
        
        self.bar_width = 0.9 / self.value_count
        self.bar_centers_x = np.linspace(self.bar_width / 2, 1.0 - self.bar_width / 2, self.value_count)

        self.vertices = np.zeros((self.value_count * 4, 3), dtype=np.float32)
        self.colors_arr = np.zeros((self.value_count * 4, 4), dtype=np.float32)
        self.faces = np.zeros((self.value_count * 2, 3), dtype=np.uint32)
        self._initialize_geometry()

        self.mesh = visuals.Mesh(vertices=self.vertices, faces=self.faces,
                                 vertex_colors=self.colors_arr, mode='triangles',
                                 parent=self.parent_view)

    def _initialize_geometry(self):
        half_width = self.bar_width / 2
        for i in range(self.value_count):
            base_face = i * 2
            base_vert = i * 4
            self.faces[base_face] = [base_vert, base_vert + 1, base_vert + 2]
            self.faces[base_face + 1] = [base_vert, base_vert + 2, base_vert + 3]
            center_x = self.bar_centers_x[i]
            self.vertices[base_vert] = (center_x - half_width, 0, 0)
            self.vertices[base_vert + 1] = (center_x + half_width, 0, 0)
            self.vertices[base_vert + 2] = (center_x + half_width, 0, 0)
            self.vertices[base_vert + 3] = (center_x - half_width, 0, 0)
            color = self.colors[i] if i < len(self.colors) else (1, 1, 1, 0.8)
            self.colors_arr[base_vert:base_vert+4] = color

    def update(self, values):
        n = min(len(values), self.value_count)
        vals = np.array(values[:n], dtype=np.float32)
        bottom_y = np.minimum(vals, -0.0001)
        top_y = np.maximum(vals, 0.0001)
        self.vertices[0:n*4:4, 1] = bottom_y
        self.vertices[1:n*4:4, 1] = bottom_y
        self.vertices[2:n*4:4, 1] = top_y
        self.vertices[3:n*4:4, 1] = top_y
        self.mesh.mesh_data.set_vertices(self.vertices)
        self.mesh.mesh_data_changed()

class SensorView:
    def __init__(self, title, value_dim, time_steps):
        self.grid = scene.widgets.Grid()
        self.y_min = -1.0
        self.y_max = 1.0
        self.value_dim = value_dim
        
        title_label = scene.Label(title, color='black', font_size=11)
        title_label.height_max = 30
        title_label.height_min = 30
        title_label.stretch = (1, 0.001) 
        self.grid.add_widget(title_label, row=0, col=0, col_span=3)

        yaxis = scene.AxisWidget(orientation='left', axis_font_size=10, 
                                 axis_label_margin=40, tick_label_margin=5,
                                 text_color='black', axis_color='black', tick_color='black')
        yaxis.width_max = 60
        yaxis.width_min = 60
        yaxis.stretch = (0.001, 1) 
        self.grid.add_widget(yaxis, row=1, col=0)
        
        self.bar_view = self.grid.add_view(row=1, col=1, bgcolor="white")
        self.bar_view.stretch = (0.001, 1)
        
        self.plot_right = self.grid.add_view(row=1, col=2, bgcolor="white")
        self.plot_right.stretch = (1, 1)
        
        colors = generate_colors(value_dim)
        self.bars = BarView(value_dim, colors, self.bar_view.scene)
        self.plots = TimePlotView(value_dim, time_steps, colors, self.plot_right.scene)

        self.bar_view.camera = "panzoom"
        self.plot_right.camera = "panzoom"
        self._apply_camera_range()
        yaxis.link_view(self.plot_right)
        
    def _apply_camera_range(self):
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(self.y_min, self.y_max))
        self.plot_right.camera.set_range(x=(0.0, 1.0), y=(self.y_min, self.y_max))
        
    def update_data(self, data, autoscale_active):
        self.bars.update(data)
        self.plots.update(data)
        if autoscale_active and len(data) > 0:
            d_min, d_max = min(data), max(data)
            needs_update = False
            if d_min < self.y_min:
                self.y_min = d_min - abs(d_min * 0.2)
                needs_update = True
            if d_max > self.y_max:
                self.y_max = d_max + abs(d_max * 0.2)
                needs_update = True
            if needs_update:
                self._apply_camera_range()

class AdaptiveCanvas:
    def __init__(self, size):
        self.canvas = SceneCanvas(size=size, keys="interactive", bgcolor="white")
        self.main_grid = self.canvas.central_widget.add_grid()
        self.views = {}
        self.time_steps = 100
        self.max_dimension = 0

    def add_sensor_view(self, name, value_dim):
        sensor_view = SensorView(name, value_dim, self.time_steps)
        self.main_grid.add_widget(sensor_view.grid, row=len(self.views), col=0)
        self.views[name] = sensor_view
        
        if value_dim > self.max_dimension:
            self.max_dimension = value_dim
            
        target_width = max(60, self.max_dimension * 2 + 10)
        for view in self.views.values():
            view.bar_view.width_max = target_width
            view.bar_view.width_min = target_width

    def set_time_steps(self, steps):
        self.time_steps = steps
        for view in self.views.values():
            view.plots.set_time_steps(steps)

    def update_data(self, latest_data, autoscale_active):
        for addr, values in latest_data.items():
            if addr not in self.views:
                self.add_sensor_view(addr, len(values))
            self.views[addr].update_data(values, autoscale_active)

"""
Main Application
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Player")
        self.resize(1000, 800)
        
        self.events = []
        self.total_time = 0.0
        self.current_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        
        self.vis_queue = queue.Queue()
        self.playback_thread = None
        
        self.vis_active = True
        self.autoscale_active = True
        
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 1. VisPy Canvas
        self.adaptive_canvas = AdaptiveCanvas((900, 600))
        main_layout.addWidget(self.adaptive_canvas.canvas.native, stretch=1)
        
        # 2. File & Target Settings
        file_target_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load .npz")
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        
        ip_label = QtWidgets.QLabel("IP:")
        self.ip_input = QtWidgets.QLineEdit("127.0.0.1")
        self.ip_input.setFixedWidth(100)
        port_label = QtWidgets.QLabel("Port:")
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(9007)
        self.port_input.setFixedWidth(70)
        
        file_target_layout.addWidget(self.load_btn)
        file_target_layout.addWidget(self.file_label, 1)
        file_target_layout.addWidget(ip_label)
        file_target_layout.addWidget(self.ip_input)
        file_target_layout.addWidget(port_label)
        file_target_layout.addWidget(self.port_input)
        main_layout.addLayout(file_target_layout)
        
        # 3. Slider Controls Grid
        slider_grid = QtWidgets.QGridLayout()
        
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_label = QtWidgets.QLabel("0.00s")
        self.time_label.setFixedWidth(50)
        
        self.start_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.start_slider.setRange(0, 1000)
        self.start_label = QtWidgets.QLabel("0.00s")
        self.start_label.setFixedWidth(50)
        
        self.end_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.end_slider.setRange(0, 1000)
        self.end_slider.setValue(1000)
        self.end_label = QtWidgets.QLabel("0.00s")
        self.end_label.setFixedWidth(50)
        
        self.loop_checkbox = QtWidgets.QCheckBox("Loop")
        
        slider_grid.addWidget(QtWidgets.QLabel("Time:"), 0, 0)
        slider_grid.addWidget(self.time_slider, 0, 1)
        slider_grid.addWidget(self.time_label, 0, 2)
        slider_grid.addWidget(self.loop_checkbox, 0, 3)
        
        slider_grid.addWidget(QtWidgets.QLabel("Start:"), 1, 0)
        slider_grid.addWidget(self.start_slider, 1, 1)
        slider_grid.addWidget(self.start_label, 1, 2)
        
        slider_grid.addWidget(QtWidgets.QLabel("End:"), 2, 0)
        slider_grid.addWidget(self.end_slider, 2, 1)
        slider_grid.addWidget(self.end_label, 2, 2)
        
        main_layout.addLayout(slider_grid)
        
        # 4. Buttons Row (Play, Stop, Vis Controls)
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.play_btn = QtWidgets.QPushButton("Play")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        self.vis_btn = QtWidgets.QPushButton("Vis (ON)")
        self.autoscale_btn = QtWidgets.QPushButton("Auto-Scale (ON)")
        
        self.time_steps_input = QtWidgets.QSpinBox()
        self.time_steps_input.setRange(10, 5000)
        self.time_steps_input.setValue(100)
        self.time_steps_input.setPrefix("Steps: ")
        self.time_steps_input.setFixedWidth(100)
        
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        self.fps_label.setFixedWidth(65)
        
        controls_layout.addWidget(self.vis_btn)
        controls_layout.addWidget(self.autoscale_btn)
        controls_layout.addWidget(self.time_steps_input)
        controls_layout.addWidget(self.fps_label)
        main_layout.addLayout(controls_layout)

        # Connections
        self.load_btn.clicked.connect(self.load_file)
        self.play_btn.clicked.connect(self.start_playback)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.vis_btn.clicked.connect(self.toggle_vis)
        self.autoscale_btn.clicked.connect(self.toggle_autoscale)
        self.time_steps_input.valueChanged.connect(self.adaptive_canvas.set_time_steps)
        
        self.time_slider.sliderMoved.connect(self.scrub_time)
        self.start_slider.sliderMoved.connect(self.scrub_start)
        self.end_slider.sliderMoved.connect(self.scrub_end)
        self.loop_checkbox.stateChanged.connect(self.toggle_loop)
        
        # GUI Update Loop
        self.frames = 0
        self.last_fps_time = time.time()
        
        self.process_timer = QtCore.QTimer()
        self.process_timer.timeout.connect(self.process_queue)
        self.process_timer.start(16)

    def load_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Recording", "", "NumPy Archives (*.npz)"
        )
        if not filepath: return
            
        self.file_label.setText("Loading and parsing...")
        self.file_label.setStyleSheet("color: orange;")
        QtWidgets.QApplication.processEvents()
        
        try:
            with np.load(filepath) as data:
                addresses = set(k.replace('_timestamps', '').replace('_values', '') for k in data.files)
                raw_events = []
                for addr in addresses:
                    ts_arr = data[f"{addr}_timestamps"]
                    val_arr = data[f"{addr}_values"]
                    for i in range(len(ts_arr)):
                        raw_events.append((float(ts_arr[i]), addr, val_arr[i].tolist()))
                
                raw_events.sort(key=lambda x: x[0])
                self.events = raw_events
                
                if self.events:
                    self.total_time = self.events[-1][0]
                else:
                    self.total_time = 0.0
                    
            filename = filepath.split('/')[-1]
            self.file_label.setText(f"Loaded: {filename}")
            self.file_label.setStyleSheet("color: green;")
            
            # Reset timeline defaults
            self.start_time = 0.0
            self.end_time = self.total_time
            self.current_time = 0.0
            
            self.time_slider.setValue(0)
            self.start_slider.setValue(0)
            self.end_slider.setValue(1000)
            
            self.time_label.setText("0.00s")
            self.start_label.setText("0.00s")
            self.end_label.setText(f"{self.total_time:.2f}s")
            
            self.play_btn.setEnabled(True)
            
        except Exception as e:
            self.file_label.setText(f"Error loading file: {str(e)}")
            self.file_label.setStyleSheet("color: red;")
            self.events = []
            self.play_btn.setEnabled(False)

    def scrub_time(self, val):
        if self.total_time == 0: return
        self.current_time = (val / 1000.0) * self.total_time
        self.time_label.setText(f"{self.current_time:.2f}s")
        
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self.stop_playback()
            self.start_playback()

    def scrub_start(self, val):
        if self.total_time == 0: return
        self.start_time = (val / 1000.0) * self.total_time
        if self.start_time > self.end_time:
            self.start_time = self.end_time
            self.start_slider.blockSignals(True)
            self.start_slider.setValue(int((self.start_time/self.total_time)*1000))
            self.start_slider.blockSignals(False)
        self.start_label.setText(f"{self.start_time:.2f}s")
        
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self.playback_thread.start_sec = self.start_time
        if self.current_time < self.start_time:
            self.scrub_time(int((self.start_time / self.total_time) * 1000))

    def scrub_end(self, val):
        if self.total_time == 0: return
        self.end_time = (val / 1000.0) * self.total_time
        if self.end_time < self.start_time:
            self.end_time = self.start_time
            self.end_slider.blockSignals(True)
            self.end_slider.setValue(int((self.end_time/self.total_time)*1000))
            self.end_slider.blockSignals(False)
        self.end_label.setText(f"{self.end_time:.2f}s")
        
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self.playback_thread.end_sec = self.end_time
        if self.current_time > self.end_time:
            self.scrub_time(int((self.end_time / self.total_time) * 1000))

    def toggle_loop(self):
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self.playback_thread.loop = self.loop_checkbox.isChecked()

    def start_playback(self):
        if not self.events: return
        
        if self.current_time >= self.end_time or self.current_time < self.start_time:
            self.current_time = self.start_time
        
        while not self.vis_queue.empty():
            try: self.vis_queue.get_nowait()
            except queue.Empty: break
                
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self.ip_input.setEnabled(False)
        self.port_input.setEnabled(False)
        self.start_slider.setEnabled(False)
        self.end_slider.setEnabled(False)
        
        # Instantiate a fresh thread for every playback session
        self.playback_thread = PlaybackThread(self.vis_queue)
        self.playback_thread.progress_update.connect(self.update_progress)
        self.playback_thread.playback_finished.connect(self.on_playback_finished)
        
        self.playback_thread.load_events(self.events, self.ip_input.text(), self.port_input.value())
        self.playback_thread.configure_playback(self.current_time, self.start_time, self.end_time, self.loop_checkbox.isChecked())
        self.playback_thread.start()

    def stop_playback(self):
        if self.playback_thread is not None and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread.wait()

    def toggle_vis(self):
        self.vis_active = not self.vis_active
        self.vis_btn.setText(f"Vis ({'ON' if self.vis_active else 'OFF'})")
        
        if self.vis_active:
            self.adaptive_canvas.canvas.native.show()
            if hasattr(self, 'last_window_size'):
                self.resize(self.last_window_size)
        else:
            self.last_window_size = self.size()
            self.adaptive_canvas.canvas.native.hide()
            self.centralWidget().adjustSize()
            self.resize(self.width(), 1)

    def toggle_autoscale(self):
        self.autoscale_active = not self.autoscale_active
        self.autoscale_btn.setText(f"Auto-Scale ({'ON' if self.autoscale_active else 'OFF'})")

    def update_progress(self, current_time):
        self.current_time = current_time
        self.time_label.setText(f"{current_time:.2f}s")
        if self.total_time > 0:
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int((current_time / self.total_time) * 1000))
            self.time_slider.blockSignals(False)

    def on_playback_finished(self):
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Restore GUI Elements
        self.load_btn.setEnabled(True)
        self.ip_input.setEnabled(True)
        self.port_input.setEnabled(True)
        self.start_slider.setEnabled(True)
        self.end_slider.setEnabled(True)

    def process_queue(self):
        messages = []
        while not self.vis_queue.empty():
            try: messages.append(self.vis_queue.get_nowait())
            except queue.Empty: break
        
        if not messages: return

        if self.vis_active:
            latest_data = {}
            for _, addr, vals in messages:
                latest_data[addr] = vals
            self.adaptive_canvas.update_data(latest_data, self.autoscale_active)
            
            self.frames += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps_label.setText(f"FPS: {self.frames / (now - self.last_fps_time):.1f}")
                self.frames = 0
                self.last_fps_time = now

    def closeEvent(self, event):
        self.stop_playback()
        event.accept()

"""
Run Application
"""

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    QtWidgets.QApplication.setStyle("Fusion")
    
    win = MainWindow()
    win.show()
    app.run()