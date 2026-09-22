"""
Motion Capture Classification using LSTM Networks - Inference Script
"""

"""
imports
"""

# General imports
import os
import sys
import time
import numpy as np
from collections import deque
import colorsys

# Pytorch imports
import torch
import torch.nn as nn

# OSC imports
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# GUI Imports
from PyQt5 import QtWidgets, QtCore, QtGui
from vispy import scene
from vispy.app import use_app
from vispy.scene import SceneCanvas

# Mocap imports
from common import mocap_tools as mocap
mocap_tools_inst = mocap.Mocap_Tools()

"""
Configurations
"""

# Device Settings
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Mediapipe NPZ
mocap_data_file_path = "data/mocap/classes"
mocap_data_file_extensions = [".npz"] 
mocap_topology_file = "data/configs/Mediapipe_config.json"
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # skeleton without face, hands
mocap_data_window_length = 30
mocap_pos_scale = 100.0

"""
# XSens NPZ
mocap_data_file_path = "E:/Data/mocap/ick3/Solos/Ricardo/Xsens/npz/Classes"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # skeleton without root
mocap_data_window_length = 90
mocap_pos_scale = 1.0
"""

"""
# Mediapipe NPZ
mocap_data_file_path = "data/weights/E:/Data/mocap/Yurika/Mediapipe_v2/Classes"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
mocap_data_window_length = 30
mocap_pos_scale = 100.0
"""

"""
# Mediapipe FBX
mocap_data_file_path = "E:/Data/mocap/Yurika/Mediapipe_v2_fbx/Classes"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
mocap_data_window_length = 30
mocap_pos_scale = 100.0
"""

"""
# Mocap settings Yolo MotionBert
mocap_data_file_path = "E:/Data/mocap/Yurika/YoloMB/Classes"
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
mocap_data_window_length = 30
mocap_pos_scale = 1.0
"""

# SciPy expects a specific rotation order for its Euler conversions.
# In Mocap_Tools, 0 maps to "xyz" which is the standard default.
mocap_rot_sequence = [0, 1, 2] 

# Dynamically compute feature size based on mocap_data_types
features_per_joint = 0
for dtype in mocap_data_types:
    if "pos" in dtype:
        features_per_joint += 3
    elif "rot" in dtype:
        features_per_joint += 4

num_features = len(mocap_joint_indices) * features_per_joint

# Model Settings (Matching training script)
model_hidden_dim = 64
model_layer_count = 1
model_dropout = 0.3

# Training Settings / Paths (Aligned with Training Script output paths)

save_stats_path = "data/results/stats"
model_weights_file = "data/results/weights/classifier_weights_epoch_200.pth"

# OSC Settings
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007
osc_send_ip = "127.0.0.1"
osc_send_port = 9008

"""
Helper Functions to extract Classes
"""

def find_classes(directory):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found. Creating dummy classes for visualization.")
        return ["Class_A", "Class_B"], {"Class_A": 0, "Class_B": 1}
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# Automatically detect classes from folders
classes, _ = find_classes(mocap_data_file_path)
class_count = len(classes)

"""
Load Normalization Stats
"""

if os.path.exists(save_stats_path):
    mean_np = np.load(os.path.join(save_stats_path, "mean.npy"))
    std_np = np.load(os.path.join(save_stats_path, "std.npy"))
else:
    print(f"Warning: Normalization stats not found at {save_stats_path}. Using zero mean and unit std.")
    mean_np = np.zeros(num_features)
    std_np = np.ones(num_features)

data_mean = torch.tensor(mean_np, dtype=torch.float32).to(device)
data_std = torch.tensor(std_np, dtype=torch.float32).to(device)

"""
Classifier
"""

class Classifier(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.init_weights(self.embedding)
        self.init_weights(self.lstm)
        self.init_weights(self.classifier)

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
                    param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        out, _ = self.lstm(x)
        final_state = self.fc_dropout(out[:, -1, :])
        return self.classifier(final_state)

"""
OSC Worker
"""

class OSCWorker(QtCore.QThread):
    probs_signal = QtCore.pyqtSignal(np.ndarray)
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Buffer needs +2 size to compute velocity and acceleration while keeping requested window size
        self.buffer_pos = deque(maxlen=mocap_data_window_length + 2)
        self.buffer_rot = deque(maxlen=mocap_data_window_length + 2)
        
        self.current_pos_data = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.server = None
        self.is_running = False
        
        # OSC Out Client
        self.osc_client = SimpleUDPClient(osc_send_ip, osc_send_port)

    def set_running(self, state):
        self.is_running = state
        if not state:
            # Clear buffers on stop so old data doesn't corrupt new sessions
            self.buffer_pos.clear()
            self.buffer_rot.clear()
            self.current_pos_data = None

    def osc_pos_handler(self, address, *args):
        if not self.is_running: return
        raw_data = np.array(args)
        pos_data = []
        for j in mocap_joint_indices:
            idx = j * 3
            pos_data.append(raw_data[idx : idx+3] * mocap_pos_scale)
        self.current_pos_data = pos_data

    def osc_rot_handler(self, address, *args):
        if not self.is_running or self.current_pos_data is None: return
            
        raw_data = np.array(args)
        rot_data = []
        for j in mocap_joint_indices:
            idx = j * 4
            rot_data.append(raw_data[idx : idx+4])
            
        self.buffer_pos.append(self.current_pos_data)
        self.buffer_rot.append(rot_data)
        self.current_pos_data = None

        if len(self.buffer_pos) == self.buffer_pos.maxlen:
            self.osc_classification_handler()

    def osc_classification_handler(self):
        pos = np.array(self.buffer_pos)
        rot_quat = np.array(self.buffer_rot)
        
        # FIX: Convert scalar-last quaternions [x, y, z, w] to continuous Euler angles
        # This matches the training script's underlying continuous motion profile 
        # and eliminates artificial velocity spikes caused by quaternion double-cover flips.
        
        # quat_to_euler natively handles the [x,y,z,w] output now, but we must ensure it is un-wrapped
        rot_euler_raw = mocap_tools_inst.quat_to_euler(rot_quat, mocap_rot_sequence)
        
        # Unwrap the angles along the time axis to prevent 360-degree jumps
        rot_rad = np.deg2rad(rot_euler_raw)
        rot_rad_unwrapped = np.unwrap(rot_rad, axis=0)
        rot = np.rad2deg(rot_rad_unwrapped)
        
        # Initialize a dictionary to store computed features for this window
        features = {}
        
        if "pos" in mocap_data_types:
            features["pos"] = pos[2:] # Drop first 2 frames to align with accelerations
            
        if "rot" in mocap_data_types:
            # We must map the Euler back to quat for the feature vector if the network expects quats
            # OR if the network expects Euler, keep it as `rot`. Assuming your training script 
            # fed quats to the model (size 4), we convert the unwrapped euler back to quat:
            rot_quat_clean = mocap_tools_inst.euler_to_quat(rot, mocap_rot_sequence)
            features["rot"] = rot_quat_clean[2:]
            
        if "vel_pos" in mocap_data_types or "acc_pos" in mocap_data_types:
            vel_pos = np.diff(pos, axis=0)
            if "vel_pos" in mocap_data_types:
                features["vel_pos"] = vel_pos[1:] # Drop first frame to align with accelerations
            if "acc_pos" in mocap_data_types:
                features["acc_pos"] = np.diff(vel_pos, axis=0)

        if "vel_rot" in mocap_data_types or "acc_rot" in mocap_data_types:
            # We compute velocity on the CLEAN unwrapped quaternion to prevent flip spikes
            vel_rot = np.diff(rot_quat_clean, axis=0)
            if "vel_rot" in mocap_data_types:
                features["vel_rot"] = vel_rot[1:]
            if "acc_rot" in mocap_data_types:
                features["acc_rot"] = np.diff(vel_rot, axis=0)
        
        # Dynamically select and order the feature arrays
        selected_arrays = [features[dtype] for dtype in mocap_data_types]
        
        # Stack features along the last axis
        mocap_data = np.concatenate(selected_arrays, axis=-1)
        
        # Flatten joints into singular vector per frame: (T, J, F) -> (T, J * F)
        mocap_data = mocap_data.reshape(mocap_data.shape[0], -1)
        
        with torch.no_grad():
            batch_x = torch.FloatTensor(mocap_data).unsqueeze(0).to(device)
            
            # Apply normalization
            batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
            batch_x_norm = torch.nan_to_num(batch_x_norm)
            
            output = self.model(batch_x_norm)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # Broadcast the class probabilities
        self.probs_signal.emit(probs)
        
        for i, prob in enumerate(probs):
            self.osc_client.send_message(f"/classifier/{classes[i]}", float(prob))
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            fps = self.frame_count / (current_time - self.fps_start_time)
            self.fps_signal.emit(fps)
            self.fps_start_time = current_time
            self.frame_count = 0

    def run(self):
        dispatcher = Dispatcher()
        dispatcher.map("/mocap/0/joint/pos_local", self.osc_pos_handler)
        dispatcher.map("/mocap/0/joint/rot_local", self.osc_rot_handler)
        
        self.server = BlockingOSCUDPServer((osc_receive_ip, osc_receive_port), dispatcher)
        print("OSC Receiver listening on port {} ...".format(osc_receive_port))
        self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
        self.wait()

"""
Main GUI Window
"""

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Mocap Classifier")
        self.resize(800, 600)
        self.show_visualization = True

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        
        # Main vertical layout
        layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(layout)

        # 1. VisPy Canvas Setup (White Background)
        use_app('PyQt5')
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor='white')
        layout.addWidget(self.canvas.native, stretch=1) # Give the canvas the majority of space
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-1, -0.2, class_count * 2, 1.4))
        
        self.bars = []

        bar_colors = MainWindow.generate_distinct_colors(class_count)

        for i, cls in enumerate(classes):
            x_pos = i * 2
            scene.visuals.Text(
                cls,
                pos=(x_pos, -0.1),
                color='black',
                font_size=14,
                parent=self.view.scene
            )
            bar = scene.visuals.Rectangle(
                center=(x_pos, 0.001),
                width=1.2,
                height=0.002,
                color=bar_colors[i],
                parent=self.view.scene
            )
            self.bars.append(bar)

        # 2. Bottom Control Panel
        control_layout = QtWidgets.QHBoxLayout()
        
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_toggle_vis = QtWidgets.QPushButton("Disable Visualisation")
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        
        self.fps_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.fps_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_toggle_vis.clicked.connect(self.toggle_vis)
        self.btn_exit.clicked.connect(self.close)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_toggle_vis)
        control_layout.addWidget(self.btn_exit)
        control_layout.addStretch()
        control_layout.addWidget(self.fps_label)
        
        layout.addLayout(control_layout)

        # Initialize and start PyTorch Model + OSC Worker
        self.init_model_and_worker()

    @staticmethod
    def generate_distinct_colors(n, saturation=0.8, value=0.9, alpha=1.0):
        if n <= 0:
            return []
        
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((r, g, b, alpha))
        return colors

    def init_model_and_worker(self):
        # Match training network dimensions and layers
        self.classifier = Classifier(
            num_features=num_features, 
            hidden_dim=model_hidden_dim, 
            num_classes=class_count, 
            num_layers=model_layer_count,
            dropout=model_dropout
        ).to(device)
        
        if os.path.exists(model_weights_file):
            self.classifier.load_state_dict(torch.load(model_weights_file, map_location=device))
        else:
            print(f"Warning: Model weights not found at {model_weights_file}")
            
        self.classifier.eval()

        self.osc_worker = OSCWorker(self.classifier)
        self.osc_worker.probs_signal.connect(self.update_graph)
        self.osc_worker.fps_signal.connect(self.update_fps)
        self.osc_worker.start()

    def start_processing(self):
        self.osc_worker.set_running(True)
        self.fps_label.setText("FPS: Calculating...")

    def stop_processing(self):
        self.osc_worker.set_running(False)
        self.fps_label.setText("FPS: Stopped")
        # Reset bars to zero visually
        self.update_graph(np.zeros(class_count))

    def toggle_vis(self):
        self.show_visualization = not self.show_visualization
        
        if self.show_visualization:
            self.btn_toggle_vis.setText("Disable Visualisation")
            # Show canvas and restore standard size
            self.canvas.native.show()
            self.adjustSize()
        else:
            self.btn_toggle_vis.setText("Enable Visualisation")
            # Hide canvas and collapse the vertical space
            self.canvas.native.hide()
            self.fps_label.setText("FPS: --")
            self.centralWidget().adjustSize()
            self.resize(self.width(), 1)

    @QtCore.pyqtSlot(np.ndarray)
    def update_graph(self, probs):
        if not self.show_visualization:
            return
            
        for i, prob in enumerate(probs):
            h = max(prob, 0.002) 
            self.bars[i].height = h
            self.bars[i].center = (i * 2, h / 2.0)
            
        self.canvas.update()

    @QtCore.pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        self.osc_worker.stop()
        event.accept()

"""
Run Application
"""

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())