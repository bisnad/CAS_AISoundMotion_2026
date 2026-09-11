"""
Imports
"""

import sys
import math
import time
import datetime
import queue
import threading
import colorsys
import os
import numpy as np

# OSC
from pythonosc import dispatcher
from pythonosc import osc_server

# GUI & VisPy
from PyQt5 import QtWidgets, QtCore
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas, visuals

"""
OSC Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

"""
OSC Receiver
"""

class OscReceiver:
    def __init__(self, ip, port, msg_queue):
        self.ip = ip
        self.port = port
        self.msg_queue = msg_queue
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self.receive)
        self.server = osc_server.BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        
    def start(self):
        print(f"OscReceiver started on {self.ip}:{self.port}")
        self.thread.start()
    
    def stop(self):
        print("OscReceiver stopping...")
        self.server.shutdown()
        self.thread.join()
    
    def receive(self, addr, *args):
        self.msg_queue.put((time.time(), addr, args))

"""
Motion Recorder
"""

class MotionRecorder:
    def __init__(self):
        self.recording_active = False
        self.recording_start_time = None
        self.class_id = 0
        self.data = {} 
        os.makedirs("recordings", exist_ok=True)
        
    def set_class_id(self, class_id):
        if not self.recording_active:
            self.class_id = class_id
        
    def toggle_record(self):
        if self.recording_active:
            self.stop_record()
        else:
            self.start_record()
    
    def start_record(self):
        if self.recording_active: return
        self.data = {}
        self.recording_active = True
        self.recording_start_time = time.time()
        
    def stop_record(self):
        if not self.recording_active: return
        self.recording_active = False
        self._save_recording()
        
    def record_batch(self, messages):
        if not self.recording_active: return
        for timestamp, addr, values in messages:
            rel_time = timestamp - self.recording_start_time
            if addr not in self.data:
                self.data[addr] = {"timestamps": [], "values": []}
            self.data[addr]["timestamps"].append(rel_time)
            self.data[addr]["values"].append(values)
            
    def _save_recording(self):
        if not self.data: return
        filename = f"recordings/Sensor_class_{self.class_id}_time_{int(self.recording_start_time)}.npz"
        save_dict = {}
        for addr, d in self.data.items():
            save_dict[addr + "_timestamps"] = np.array(d["timestamps"], dtype=np.float64)
            save_dict[addr + "_values"] = np.array(d["values"], dtype=np.float32)
        np.savez_compressed(filename, **save_dict)
        print(f"Saved: {filename}")

"""
GUI Components
"""

def generate_colors(n):
    return [colorsys.hsv_to_rgb(i/n, 0.8, 0.9) + (1.0,) for i in range(n)]

class TimePlotView:
    def __init__(self, value_count, time_count, colors, parent_view):
        self.value_count = value_count
        self.time_count = time_count
        self.colors = colors
        self.parent_view = parent_view

        # OPTIMIZATION: Shape inverted to (dimensions, time_steps) for C-contiguous GPU alignment
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
            self.line = visuals.Line(pos=self.pos_array,
                                     color=self.color_array,
                                     connect=self.connect_array,
                                     width=1.5,
                                     method='gl',
                                     parent=self.parent_view)

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
        # Update specific column
        self.values[:len(values), self.idx] = values
        self.idx = (self.idx + 1) % self.time_count

        ordered_vals = np.concatenate((self.values[:, self.idx:], self.values[:, :self.idx]), axis=1)
        
        # OPTIMIZATION: ravel() is now instantaneous because the array memory is already laid out linearly
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

        self.mesh = visuals.Mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.colors_arr,
            mode='triangles',
            parent=self.parent_view
        )

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

        # OPTIMIZATION: Bypass set_data() and push directly to internal meshdata
        # Prevents VisPy from recalculating colors/faces/bounds
        self.mesh.mesh_data.set_vertices(self.vertices)
        self.mesh.mesh_data_changed()

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

        self.mesh = visuals.Mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.colors_arr,
            mode='triangles',
            parent=self.parent_view
        )

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
        vals = np.array(values[:n])
        
        bottom_y = np.minimum(vals, -0.0001)
        top_y = np.maximum(vals, 0.0001)
        
        self.vertices[0:n*4:4, 1] = bottom_y
        self.vertices[1:n*4:4, 1] = bottom_y
        self.vertices[2:n*4:4, 1] = top_y
        self.vertices[3:n*4:4, 1] = top_y

        self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=self.colors_arr)

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
        
        # Track the maximum dimension
        if value_dim > self.max_dimension:
            self.max_dimension = value_dim
            
        # FIX: Always apply the target width to ALL views every time a new sensor arrives,
        # ensuring even the small dimension sensors take the global maximum width.
        target_width = max(60, self.max_dimension * 4 + 10)
        
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
Main Application & Thread Event Loop
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, osc_queue, recorder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Sensor Recorder")
        
        self.osc_queue = osc_queue
        self.recorder = recorder
        self.vis_active = True
        self.autoscale_active = True
        
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        
        self.adaptive_canvas = AdaptiveCanvas((900, 700))
        main_layout.addWidget(self.adaptive_canvas.canvas.native)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.class_input = QtWidgets.QSpinBox(self)
        self.class_input.setPrefix("Class: ")
        self.class_input.setFixedWidth(80)

        self.record_indicator = QtWidgets.QLabel(self)
        self.record_indicator.setFixedSize(16, 16)
        self.record_indicator.setStyleSheet("background-color: gray; border-radius: 8px;")
        
        self.time_steps_input = QtWidgets.QSpinBox(self)
        self.time_steps_input.setRange(10, 5000)
        self.time_steps_input.setValue(100)
        self.time_steps_input.setPrefix("Steps: ")
        self.time_steps_input.setFixedWidth(100)
        
        self.start_btn = QtWidgets.QPushButton("Start Record", self)
        self.stop_btn = QtWidgets.QPushButton("Stop Record", self)
        self.vis_btn = QtWidgets.QPushButton("Vis (ON)", self)
        self.autoscale_btn = QtWidgets.QPushButton("Auto-Scale (ON)", self)
        
        self.fps_label = QtWidgets.QLabel("FPS: 0.0", self)
        self.fps_label.setFixedWidth(65)
        self.time_label = QtWidgets.QLabel("Rec Time: 0:00:00", self)
        
        controls_layout.addWidget(self.class_input)
        controls_layout.addWidget(self.record_indicator)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.time_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.vis_btn)
        controls_layout.addWidget(self.autoscale_btn)
        controls_layout.addWidget(self.time_steps_input)
        controls_layout.addWidget(self.fps_label)
        
        main_layout.addLayout(controls_layout)
        
        self.class_input.valueChanged.connect(self.recorder.set_class_id)
        self.time_steps_input.valueChanged.connect(self.adaptive_canvas.set_time_steps)
        self.start_btn.clicked.connect(self.start_recording_ui)
        self.stop_btn.clicked.connect(self.stop_recording_ui)
        self.vis_btn.clicked.connect(self.toggle_vis)
        self.autoscale_btn.clicked.connect(self.toggle_autoscale)
        
        self.frames = 0
        self.last_fps_time = time.time()
        
        self.process_timer = QtCore.QTimer()
        self.process_timer.timeout.connect(self.process_queue)
        self.process_timer.start(16)

    def start_recording_ui(self):
        self.recorder.start_record()
        # Change circle to red
        self.record_indicator.setStyleSheet("background-color: red; border-radius: 8px;")

    def stop_recording_ui(self):
        self.recorder.stop_record()
        # Change circle back to gray
        self.record_indicator.setStyleSheet("background-color: gray; border-radius: 8px;")

    def toggle_vis(self):
        self.vis_active = not self.vis_active
        self.vis_btn.setText(f"Vis ({'ON' if self.vis_active else 'OFF'})")
        
        if self.vis_active:
            # Show the canvas and restore the previous window dimensions
            self.adaptive_canvas.canvas.native.show()
            if hasattr(self, 'last_window_size'):
                self.resize(self.last_window_size)
        else:
            # Save current window size before hiding
            self.last_window_size = self.size()
            
            # Hide the canvas
            self.adaptive_canvas.canvas.native.hide()
            
            # Force the internal layout engine to flush the empty space immediately
            self.centralWidget().adjustSize()
            
            # Command the window to shrink to 1px tall (PyQt will mathematically 
            # intercept this and snap it to the exact height of the remaining controls)
            self.resize(self.width(), 1)

    def toggle_autoscale(self):
        self.autoscale_active = not self.autoscale_active
        state = "ON" if self.autoscale_active else "OFF"
        self.autoscale_btn.setText(f"Auto-Scale ({state})")

    def process_queue(self):
        messages = []
        while not self.osc_queue.empty():
            try:
                messages.append(self.osc_queue.get_nowait())
            except queue.Empty:
                break
        
        if not messages: return

        if self.recorder.recording_active:
            self.recorder.record_batch(messages)
            elapsed = time.time() - self.recorder.recording_start_time
            time_text = str(datetime.timedelta(seconds=int(elapsed)))
            self.time_label.setText(f"Rec Time: {time_text}")

        if self.vis_active:
            latest_data = {}
            for _, addr, vals in messages:
                latest_data[addr] = vals
            self.adaptive_canvas.update_data(latest_data, self.autoscale_active)
            
            self.frames += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                fps = self.frames / (now - self.last_fps_time)
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.frames = 0
                self.last_fps_time = now

    def closeEvent(self, event):
        print("Closing main window...")
        self.recorder.stop_record()
        event.accept()

"""
Run Application
"""

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    
    msg_queue = queue.Queue()
    recorder = MotionRecorder()
    osc_receiver = OscReceiver(osc_receive_ip, osc_receive_port, msg_queue)
    
    osc_receiver.start()
    win = MainWindow(msg_queue, recorder)
    win.show()
    
    try:
        app.run()
    finally:
        osc_receiver.stop()