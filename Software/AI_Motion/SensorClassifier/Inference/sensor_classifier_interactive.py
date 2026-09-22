"""
Real-time Sensor Classifier
===================================

This application receives sensor data via OSC (Open Sound Control), 
classifies motion patterns using a trained LSTM neural network, and displays 
the classification results in real-time through a bar chart visualization.

Features:
- Real-time motion data reception via OSC
- LSTM-based motion classification
- Interactive visualization with dynamic class labels from directories
- Toggleable visualization for performance saving
- Graceful application shutdown
- Multi-threaded OSC communication
"""

"""
Imports
"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import colorsys

# OSC
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# PyTorch
import torch
import torch.nn.functional as F
from torch import nn

# GUI
from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas, visuals

"""
Configurations
"""

# Device Settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Sensor Data Settings


# Example Mobile Phone (Sensor2Osc)
sensor_data_path = "data/sensors/sensors_phone/"  # Path to the training folders to extract class names
sensor_data_norm_path = "data/results/sensors_phone/stats/"
sensor_data_ids = ["/accelerometer", "/gyroscope"] 
sensor_data_window_length = 90

"""
# Example Imu
sensor_data_path = "data/sensors/sensors_imu/"  # Path to the training folders to extract class names
sensor_data_norm_path = "data/results/sensors_imu/stats/"
sensor_data_ids = ["/imu/1/accelerometer", "/imu/1/gyroscope"] 
sensor_data_window_length = 90
"""

# Model Settings

model_hidden_dim = 64
model_layer_count = 3
model_dropout = 0.3
model_weights_file = "data/results/sensors_phone/weights/classifier_weights_epoch_400.pth"
#model_weights_file = "data/results/sensors_imu/weights/classifier_weights_epoch_400.pth"

# OSC Settings

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 9008

# GUI Settings

canvas_size = (600, 400)

"""
Load Classes, Sensor Dimensions, and Normalization Parameters
"""

def find_classes(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Data directory '{directory}' does not exist.")
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    return classes

def find_sensor_dims(directory, sensor_ids, extensions=(".npz",)):
    """
    Scan the class directories for the first available recording and
    determine the per-sensor feature dimensionality by inspecting the
    shape of each '<sensor_id>_values' array, mirroring how the training
    script concatenates sensor channels (load_sensor_file / axis=1).
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Data directory '{directory}' does not exist.")

    sample_file = None
    for root, _, fnames in os.walk(directory, followlinks=True):
        for fname in sorted(fnames):
            if fname.lower().endswith(tuple(extensions)):
                sample_file = os.path.join(root, fname)
                break
        if sample_file:
            break

    if sample_file is None:
        raise FileNotFoundError(
            f"No recordings with extensions {extensions} found under '{directory}' "
            f"to infer sensor dimensions from."
        )

    print(f"Inferring sensor dimensions from sample recording: {sample_file}")
    data = np.load(sample_file)

    dims = []
    for sid in sensor_ids:
        key = sid + "_values"
        if key not in data:
            raise KeyError(
                f"Sensor id '{sid}' (key '{key}') not found in sample recording "
                f"'{sample_file}'. Available keys: {list(data.keys())}"
            )
        dims.append(int(data[key].shape[1]))

    print(f"Detected sensor dimensions: {dict(zip(sensor_ids, dims))}")
    return dims

try:
    # Dynamically load class names from directory structure
    class_names = find_classes(sensor_data_path)
    class_count = len(class_names)
    class_labels = class_names
    print(f"Detected {class_count} classes: {class_names}")

    # Auto-determine per-sensor feature dimensions from a sample recording,
    sensor_data_dims = find_sensor_dims(sensor_data_path, sensor_data_ids)
except FileNotFoundError as e:
    print(f"Error detecting classes or sensor dimensions: {e}")
    sys.exit(1)
except KeyError as e:
    print(f"Error detecting sensor dimensions: {e}")
    sys.exit(1)

# Load normalization stats
try:
    data_mean = np.load(os.path.join(sensor_data_norm_path, "mean.npy"))
    data_std = np.load(os.path.join(sensor_data_norm_path, "std.npy"))
    model_input_dim = data_mean.shape[0]
    print(f"Loaded normalization stats. Input dimension: {model_input_dim}")
except FileNotFoundError as e:
    print(f"Could not load normalization data: {e}")
    sys.exit(1)

"""
Create Model
"""

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, class_count, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, batch_first=True, dropout=dropout if layer_count > 1 else 0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, class_count)
        self.relu = nn.ReLU()
        
        self.init_weights(self.rnn)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        
    """
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
    """
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    # forget gate is the second of the 4 chunks: i, f, g, o
                    param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        x = h[-1]  # Take final hidden state from last layer
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

"""
Live Classification Components
"""

class LiveClassifier(QtCore.QObject):
    """Handles real-time motion classification using the trained model."""
    
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, classifier: Classifier, parent=None):
        super().__init__(parent=parent)
        self.classifier = classifier.eval()
        
    def update(self, input_data: np.ndarray):
        try:
            with torch.no_grad():
                # Normalize input data identically to training phase
                input_norm = (input_data - data_mean) / (data_std + 1e-8)
                input_norm = np.nan_to_num(input_norm)
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get model predictions
                outputs = self.classifier(input_tensor)
                class_probs = F.softmax(outputs, dim=1)
                class_probs = class_probs.detach().cpu().numpy()[0]
                
                self.new_data.emit(class_probs)
                
        except Exception as e:
            print(f"Error in classification: {e}")

"""
OSC Communication
"""

class OscReceiver(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)
    
    def __init__(self, ip, port, parent=None):
        super().__init__(parent=parent)
        self.ip = ip
        self.port = port
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self.receive)
        self.server = osc_server.BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        
    def start(self):
        print(f"OscReceiver listening on {self.ip}:{self.port}")
        self.server.serve_forever()
    
    def stop(self):
        print("OscReceiver stop")
        self.server.shutdown()
    
    def receive(self, addr, *args):
        osc_address = addr
        osc_values = args
        values_dict = {osc_address: osc_values}
        self.new_data.emit(values_dict)

class SensorDataReceiver(QtCore.QObject):
    new_data = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, sensor_ids, sensor_data_dims, window_length, parent=None):
        super().__init__(parent=parent)
        self.sensor_ids = sensor_ids
        self.sensor_data_dims = sensor_data_dims
        self.window_length = window_length
        self.sensor_values = [np.zeros((self.window_length, self.sensor_data_dims[sI])) for sI in range(len(self.sensor_data_dims))]
        self.sensor_updated = [False] * len(self.sensor_ids)
        self.running = False
        
    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False

    def receive(self, new_data):
        if not self.running:
            return

        sensor_id = list(new_data.keys())[0]
        sensor_value = list(new_data.values())[0]
        
        if sensor_id not in self.sensor_ids:
            return
    
        sensor_index = self.sensor_ids.index(sensor_id)
        
        self.sensor_values[sensor_index] = np.roll(self.sensor_values[sensor_index], -1, axis=0)
        self.sensor_values[sensor_index][-1] = sensor_value
        self.sensor_updated[sensor_index] = True
 
        if self.sensor_updated.count(True) == len(self.sensor_ids):
            sensor_values_combined = np.concatenate(self.sensor_values, axis=1)
            self.new_data.emit(sensor_values_combined)
            self.sensor_updated = [False] * len(self.sensor_ids)

class ClassificationSender:
    """Sends classification results via OSC."""
    
    def __init__(self, ip: str, port: int):
        self.osc_sender = SimpleUDPClient(ip, port)
        print(f"Classification sender initialized: {ip}:{port}")
        
    def send(self, class_probs: np.ndarray):
        try:
            osc_values = class_probs.tolist()
            self.osc_sender.send_message("/motion/class", osc_values)
        except Exception as e:
            print(f"Error sending classification data: {e}")
        

"""
GUI Components
"""

class ClassificationBarView:
    """Visualization component for classification results with labels."""
    
    def __init__(self, class_count: int, class_labels: List[str], 
                 colors: List[Tuple[float, float, float]], parent_view):
        self.class_count = class_count
        self.class_labels = class_labels
        self.parent_view = parent_view
        
        self.bar_width = 0.8 / class_count
        self.bar_spacing = 1.0 / class_count
        bar_centers_x = np.linspace(self.bar_spacing / 2, 1.0 - self.bar_spacing / 2, class_count)
        
        self.bars = []
        self.labels = []
        
        for i in range(class_count):
            bar = visuals.Rectangle(
                center=(bar_centers_x[i], 0.0),
                width=self.bar_width,
                height=0.01,
                color=colors[i]
            )
            self.bars.append(bar)
            
            label = visuals.Text(
                text=class_labels[i],
                pos=(bar_centers_x[i], -0.1),
                color='black',
                font_size=12,
                anchor_x='center',
                anchor_y='top'
            )
            self.labels.append(label)
            
        self.compound = visuals.Compound(self.bars + self.labels, parent=self.parent_view)
        
    def update(self, values: np.ndarray):
        for bar, value in zip(self.bars, values):
            bar.center = (bar.center[0], value / 2)
            bar.height = max(abs(value), 0.001)

class VisualizationCanvas:
    """Main visualization canvas containing the bar chart."""
    
    def __init__(self, class_labels: List[str], colors: List[Tuple[float, float, float]], size: Tuple[int, int]):
        self.size = size
        self.canvas = SceneCanvas(size=size, keys="interactive")
        self.grid = self.canvas.central_widget.add_grid()
        self.vis_active = True
        
        self.bar_view = self.grid.add_view(0, 0, bgcolor="white")
        
        self.bars = ClassificationBarView(
            class_count=len(colors),
            class_labels=class_labels,
            colors=colors,
            parent_view=self.bar_view.scene
        )
        
        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(-0.2, 1.0))
        
        # Connect FPS tracking
        self.fps_callback_fn = None
        self.canvas.measure_fps(window=1, callback=self._internal_fps_callback)
        
    def _internal_fps_callback(self, fps):
        """Passes the internal VisPy float calculation to the GUI label callback."""
        if self.fps_callback_fn and self.vis_active:
            self.fps_callback_fn(fps)
            
    def update(self, new_data: np.ndarray):
        if self.vis_active:
            self.bars.update(new_data)

class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""
    
    closing = QtCore.pyqtSignal()
    start_classification = QtCore.pyqtSignal()
    stop_classification = QtCore.pyqtSignal()
    
    def __init__(self, canvas: VisualizationCanvas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Real-Time Sensor Classifier")
        self.setWindowIcon(QtGui.QIcon())
        
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        
        self.canvas = canvas
        main_layout.addWidget(self.canvas.canvas.native)
        
        self._create_controls(main_layout)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self._connect_signals()
        
    def _create_controls(self, main_layout: QtWidgets.QVBoxLayout):
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Action Buttons
        self.start_button = QtWidgets.QPushButton("Start", self)
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.vis_button = QtWidgets.QPushButton("Disable Visualization", self)
        self.exit_button = QtWidgets.QPushButton("Exit", self)
        
        for button in [self.start_button, self.stop_button, self.vis_button, self.exit_button]:
            button.setMinimumWidth(130)
            button.setMinimumHeight(30)
            
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.vis_button)
        controls_layout.addWidget(self.exit_button)
        controls_layout.addStretch()
        
        # GUI FPS label on the bottom right
        self.fps_label = QtWidgets.QLabel("FPS: 0.0", self)
        self.fps_label.setFixedWidth(70)
        controls_layout.addWidget(self.fps_label)
        
        main_layout.addLayout(controls_layout)
        
    def _connect_signals(self):
        self.start_button.clicked.connect(self.start_classification.emit)
        self.stop_button.clicked.connect(self.stop_classification.emit)
        self.vis_button.clicked.connect(self.toggle_vis)
        self.exit_button.clicked.connect(self.close)
        
        # Link the canvas FPS tracking straight into our new PyQt label
        self.canvas.fps_callback_fn = self.update_fps
        
    def toggle_vis(self):
        self.canvas.vis_active = not self.canvas.vis_active
        self.vis_button.setText("Disable Visualization" if self.canvas.vis_active else "Enable Visualization")
        
        if self.canvas.vis_active:
            # Show canvas and restore standard size
            self.canvas.canvas.native.show()
            self.adjustSize()
        else:
            # Hide canvas and collapse the vertical space
            self.canvas.canvas.native.hide()
            self.fps_label.setText("FPS: --")
            self.centralWidget().adjustSize()
            self.resize(self.width(), 1)
            
    def update_fps(self, fps: float):
        """Called automatically by VisPy roughly once per second."""
        self.fps_label.setText(f"FPS: {float(fps):.1f}")
        
    def closeEvent(self, event):
        print("Closing main window")
        self.closing.emit()
        event.accept()

# =============================================================================
# Main Application
# =============================================================================

class MotionClassifierApp(QtCore.QObject):
    """Main application controller that coordinates all components."""
    
    def __init__(self):
        super().__init__()
        self.osc_thread = None
        self.components = {}
        
    def initialize(self):
        try:
            self._initialize_model()
            self._initialize_osc_components()
            self._initialize_gui()
            self._connect_components()
            print("Application initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize application: {e}")
            return False
            
    def _initialize_model(self):
        classifier = Classifier(
            input_dim=model_input_dim,
            hidden_dim=model_hidden_dim,
            layer_count=model_layer_count,
            class_count=class_count,
            dropout=model_dropout
        )
        
        classifier.to(device)
        
        if device == 'cuda':
            classifier.load_state_dict(torch.load(model_weights_file))
        else:
            classifier.load_state_dict(torch.load(model_weights_file, map_location=torch.device("cpu")))
            
        self.components['classifier'] = LiveClassifier(classifier)
        print(f"Model loaded on {device}")
        
    def _initialize_osc_components(self):
        self.components['osc_receiver'] = OscReceiver(osc_receive_ip, osc_receive_port)
        self.components['sensor_receiver'] = SensorDataReceiver(sensor_data_ids, sensor_data_dims, sensor_data_window_length)
        self.components['classify_sender'] = ClassificationSender(osc_send_ip, osc_send_port)
        
    def _initialize_gui(self):
        bar_colors = [colorsys.hsv_to_rgb(1.0 / class_count * i, 1.0, 1.0) for i in range(class_count)]
        self.components['canvas'] = VisualizationCanvas(class_labels, bar_colors, canvas_size)
        self.components['main_window'] = MainWindow(self.components['canvas'])
        
    def _connect_components(self):
        self.osc_thread = QtCore.QThread(parent=self.components['main_window'])
        self.components['osc_receiver'].moveToThread(self.osc_thread)
        
        self.components['osc_receiver'].new_data.connect(self.components['sensor_receiver'].receive)
        self.components['sensor_receiver'].new_data.connect(self.components['classifier'].update)
        
        self.components['classifier'].new_data.connect(self.components['classify_sender'].send)
        self.components['classifier'].new_data.connect(self.components['canvas'].update)
        
        self.components['main_window'].start_classification.connect(self.components['sensor_receiver'].start)
        self.components['main_window'].stop_classification.connect(self.components['sensor_receiver'].stop)
        
        self.osc_thread.started.connect(self.components['osc_receiver'].start)
        self.components['main_window'].closing.connect(self._shutdown)
        
    def run(self):
        if not self.initialize():
            return False
            
        self.components['main_window'].show()
        self.osc_thread.start()
        print("Application started")
        return True
        
    def _shutdown(self):
        print("Shutting down application...")
        if 'sensor_receiver' in self.components:
            self.components['sensor_receiver'].stop()
        if 'osc_receiver' in self.components:
            self.components['osc_receiver'].stop()
        if self.osc_thread and self.osc_thread.isRunning():
            self.osc_thread.quit()
            self.osc_thread.wait(3000)
        print("Application shutdown complete")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    qt_app = use_app("pyqt5")
    qt_app.create()
    app = MotionClassifierApp()
    
    if app.run():
        try:
            qt_app.run()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
    else:
        print("Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()