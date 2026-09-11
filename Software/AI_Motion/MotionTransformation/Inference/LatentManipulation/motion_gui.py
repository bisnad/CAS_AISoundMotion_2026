import torch
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.Vector as qVector
import pyqtgraph.opengl as gl
from pathlib import Path

from threading import Thread, Event
import time
from time import sleep
import datetime

import motion_synthesis
from common import fbx_tools as fbx  # Added for FBX exporting

config = {"synthesis": None,
          "sender": None,
          "update_interval": 0.02,
          "view_min": np.array([-100, -100, -100], dtype=np.float32),
          "view_max": np.array([100, 100, 100], dtype=np.float32),
          "view_ele": 90,
          "view_azi": -90,
          "view_dist": 250,
          "view_line_width": 2.0,
          "osc_ip": "127.0.0.1",
          "osc_port": 9004
    }

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts['rotationMethod'] = 'quaternion'

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()

        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos

        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.LeftButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())

        elif ev.buttons() == QtCore.Qt.MiddleButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        else:
            super().mouseMoveEvent(ev)


class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.synthesis = config["synthesis"]
        self.sender = config["sender"]
        
        self.edges = self.synthesis.edge_list
        self.pose_thread_interval = config["update_interval"]
        
        # Recording state
        self.is_recording = False
        self.record_buffer_pos = []
        self.record_buffer_rot = []
        
        self.view_min = config["view_min"]
        self.view_max = config["view_max"]
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_line_width = config["view_line_width"]

        view_center = config.get("view_center", np.array([0, 0, 100], dtype=np.float32))
        self.view_center = qVector(view_center[0], view_center[1], view_center[2])
        
        # dynamic canvas
        self.pose_canvas = CustomGLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)

        self.pose_canvas.setCameraParams(center=self.view_center)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)

        # Buttons
        self.q_start_buttom = QtWidgets.QPushButton("Start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("Stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  

        self.q_record_button = QtWidgets.QPushButton("Record", self)
        self.q_record_button.setCheckable(True)
        self.q_record_button.clicked.connect(self.toggle_recording)
        
        self.q_exit_button = QtWidgets.QPushButton("Exit", self)
        self.q_exit_button.clicked.connect(self.exit_application)
        
        self.q_button_grid = QtWidgets.QHBoxLayout()
        self.q_button_grid.addWidget(self.q_start_buttom)
        self.q_button_grid.addWidget(self.q_stop_buttom)
        self.q_button_grid.addWidget(self.q_record_button)
        self.q_button_grid.addWidget(self.q_exit_button)

        # -----------------------------
        # GUI ELEMENTS (Controls)
        # -----------------------------
        self.controls_layout = QtWidgets.QVBoxLayout()

        num_seqs = len(self.synthesis.orig_sequences)
        max_seq_idx = max(0, num_seqs - 1)
        max_incr = max(1, self.synthesis.seq_window_length // 2)

        # OSC Setup Group
        osc_group = QtWidgets.QGroupBox("OSC Controls")
        osc_layout = QtWidgets.QFormLayout()
        
        self.q_osc_ip = QtWidgets.QLineEdit(config.get("osc_ip", "127.0.0.1"))
        self.q_osc_ip.textChanged.connect(self.change_osc_ip)
        osc_layout.addRow("OSC IP:", self.q_osc_ip)
        
        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1024, 65535)
        self.q_osc_port.setValue(config.get("osc_port", 9004))
        self.q_osc_port.valueChanged.connect(self.change_osc_port)
        osc_layout.addRow("OSC Port:", self.q_osc_port)
        osc_group.setLayout(osc_layout)

        # Sequence 1 Controls
        seq1_group = QtWidgets.QGroupBox("Sequence 1 Controls")
        seq1_layout = QtWidgets.QFormLayout()
        
        self.seq1_idx_box = QtWidgets.QSpinBox()
        self.seq1_idx_box.setRange(0, max_seq_idx)
        self.seq1_idx_box.setValue(self.synthesis.orig_seq1_index)
        self.seq1_idx_box.valueChanged.connect(self.on_seq1_idx_changed)
        
        self.seq1_start_box = QtWidgets.QDoubleSpinBox()
        self.seq1_start_box.setDecimals(3)
        self.seq1_end_box = QtWidgets.QDoubleSpinBox()
        self.seq1_end_box.setDecimals(3)
        
        self.seq1_start_box.valueChanged.connect(self.update_seq1_ranges)
        self.seq1_end_box.valueChanged.connect(self.update_seq1_ranges)
        
        self.seq1_incr_box = QtWidgets.QSpinBox()
        self.seq1_incr_box.setRange(1, max_incr)
        self.seq1_incr_box.setValue(self.synthesis.orig_seq1_frame_incr)
        self.seq1_incr_box.valueChanged.connect(self.on_seq1_incr_changed)

        seq1_layout.addRow("Mocap Index:", self.seq1_idx_box)
        seq1_layout.addRow("Start Time (s):", self.seq1_start_box)
        seq1_layout.addRow("End Time (s):", self.seq1_end_box)
        seq1_layout.addRow("Frame Increment:", self.seq1_incr_box)
        seq1_group.setLayout(seq1_layout)

        # Sequence 2 Controls
        seq2_group = QtWidgets.QGroupBox("Sequence 2 Controls")
        seq2_layout = QtWidgets.QFormLayout()
        
        self.seq2_idx_box = QtWidgets.QSpinBox()
        self.seq2_idx_box.setRange(0, max_seq_idx)
        self.seq2_idx_box.setValue(self.synthesis.orig_seq2_index)
        self.seq2_idx_box.valueChanged.connect(self.on_seq2_idx_changed)
        
        self.seq2_start_box = QtWidgets.QDoubleSpinBox()
        self.seq2_start_box.setDecimals(3)
        self.seq2_end_box = QtWidgets.QDoubleSpinBox()
        self.seq2_end_box.setDecimals(3)
        
        self.seq2_start_box.valueChanged.connect(self.update_seq2_ranges)
        self.seq2_end_box.valueChanged.connect(self.update_seq2_ranges)
        
        self.seq2_incr_box = QtWidgets.QSpinBox()
        self.seq2_incr_box.setRange(1, max_incr)
        self.seq2_incr_box.setValue(self.synthesis.orig_seq2_frame_incr)
        self.seq2_incr_box.valueChanged.connect(self.on_seq2_incr_changed)

        seq2_layout.addRow("Mocap Index:", self.seq2_idx_box)
        seq2_layout.addRow("Start Time (s):", self.seq2_start_box)
        seq2_layout.addRow("End Time (s):", self.seq2_end_box)
        seq2_layout.addRow("Frame Increment:", self.seq2_incr_box)
        seq2_group.setLayout(seq2_layout)

        self.init_seq1_bounds()
        self.init_seq2_bounds()

        # Latent Encoding Mix Controls
        mix_group = QtWidgets.QGroupBox("Latent Encoding Mix")
        mix_layout = QtWidgets.QFormLayout()
        
        self.mix_master_box = QtWidgets.QDoubleSpinBox()
        self.mix_master_box.setRange(-1e6, 1e6)
        self.mix_master_box.setSingleStep(0.1)
        self.mix_master_box.valueChanged.connect(self.on_mix_master_changed)
        mix_layout.addRow("Master Mix:", self.mix_master_box)

        self.mix_boxes = []
        for d in range(self.synthesis.latent_dim):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self.apply_mix_offsets)
            self.mix_boxes.append(box)
            mix_layout.addRow(f"Dim {d}:", box)
        mix_group.setLayout(mix_layout)

        # Latent Encoding Offset Controls
        offset_group = QtWidgets.QGroupBox("Latent Encoding Offset")
        offset_layout = QtWidgets.QFormLayout()

        self.offset_master_box = QtWidgets.QDoubleSpinBox()
        self.offset_master_box.setRange(-1e6, 1e6)
        self.offset_master_box.setSingleStep(0.1)
        self.offset_master_box.valueChanged.connect(self.on_offset_master_changed)
        offset_layout.addRow("Master Offset:", self.offset_master_box)

        self.offset_boxes = []
        for d in range(self.synthesis.latent_dim):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self.apply_mix_offsets)
            self.offset_boxes.append(box)
            offset_layout.addRow(f"Dim {d}:", box)
        offset_group.setLayout(offset_layout)

        # Build layout
        self.controls_layout.addWidget(osc_group)
        self.controls_layout.addWidget(seq1_group)
        self.controls_layout.addWidget(seq2_group)
        self.controls_layout.addWidget(mix_group)
        self.controls_layout.addWidget(offset_group)
        
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(self.controls_layout)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(320)

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addWidget(self.pose_canvas, 0, 0)
        self.q_grid.addWidget(scroll_area, 0, 1, 2, 1)
        self.q_grid.addLayout(self.q_button_grid, 1, 0)
        self.q_grid.setRowStretch(0, 1)
        self.q_grid.setColumnStretch(0, 1)
        self.q_grid.setColumnStretch(1, 0)
        
        self.setLayout(self.q_grid)
        self.setGeometry(50, 50, 900, 700)
        self.setWindowTitle("Motion Transformation")

        # Visual update signals
        self.poseCanvasUpdater = PoseCanvasUpdater()
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)

        # ---------------------------------------------------------
        # Sync Timer for updating GUI from OSC/External changes
        # ---------------------------------------------------------
        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.timeout.connect(self.sync_gui_from_synthesis)
        self.sync_timer.start(100)

    # --- Feature Addition Methods ---
    def change_osc_ip(self, text):
        self.sender.config["ip"] = text
        if hasattr(self.sender, 'client'):
            try:
                from pythonosc import udp_client
                self.sender.client = udp_client.SimpleUDPClient(text, self.q_osc_port.value())
            except ImportError:
                pass

    def change_osc_port(self, val):
        self.sender.config["port"] = val
        if hasattr(self.sender, 'client'):
            try:
                from pythonosc import udp_client
                self.sender.client = udp_client.SimpleUDPClient(self.q_osc_ip.text(), val)
            except ImportError:
                pass

    def toggle_recording(self):
        self.is_recording = self.q_record_button.isChecked()
        if self.is_recording:
            self.q_record_button.setText("Stop Recording")
            self.record_buffer_pos = []
            self.record_buffer_rot = []
            print("Recording started...")
        else:
            self.q_record_button.setText("Record")
            self.save_recording()

    def save_recording(self):
        if len(self.record_buffer_rot) == 0:
            print("No frames were recorded.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_autoencoder_motion_{timestamp}.fbx"
        
        rot_local_quat = np.array(self.record_buffer_rot)
        rot_local_euler = np.zeros((*rot_local_quat.shape[:-1], 3))
        
        try:
            from common.mocap_tools import Mocap_Tools
            mocap_t = Mocap_Tools()
            if hasattr(mocap_t, 'quat_to_euler'):
                rot_local_euler = mocap_t.quat_to_euler(rot_local_quat, [0,1,2])
            else:
                from scipy.spatial.transform import Rotation
                flat_quats = rot_local_quat.reshape(-1, 4)
                flat_eulers = Rotation.from_quat(flat_quats).as_euler('xyz', degrees=True)
                rot_local_euler = flat_eulers.reshape((*rot_local_quat.shape[:-1], 3))
        except Exception as e:
            print(f"Warning: Could not properly convert quaternions to euler angles. ({e})")
            
        from common.fbx_tools import FBX_Mocap_Data, FBX_Tools
        
        fbx_data = FBX_Mocap_Data()
        skel = self.synthesis.skeleton
        
        fbx_data.skeleton_joints = skel.get("joints", [])
        fbx_data.skeleton_children = skel.get("children", [])
        fbx_data.skeleton_parents = skel.get("parents", [])
        fbx_data.skeleton_joint_offsets = skel.get("offsets", [])
        fbx_data.skeleton_root_node = None 
        fbx_data.skeleton_nodes = []
        
        fbx_data.motion_rot_sequence = [0, 1, 2] 
        fbx_data.motion_frame_rate = float(self.synthesis.mocap_fps)
        fbx_data.motion_frame_count = len(self.record_buffer_pos)
        fbx_data.motion_pos_local = np.array(self.record_buffer_pos)
        fbx_data.motion_rot_local_euler = rot_local_euler
        fbx_data.system_unit = "cm" 

        exporter = fbx.FBX_Tools()
        try:
            exporter.write([fbx_data], filename)
            print(f"Successfully saved recording to: {filename}")
        except Exception as e:
            print(f"Error saving FBX file: {e}")

    def exit_application(self):
        if hasattr(self, 'pose_thread_event') and not self.pose_thread_event.is_set():
            self.stop()
        self.close()

    # --- GUI -> Synthesis Callback Methods ---
    def init_seq1_bounds(self):
        duration = self.synthesis.seq1_length / float(self.synthesis.mocap_fps)
        self.seq1_start_box.blockSignals(True)
        self.seq1_end_box.blockSignals(True)
        self.seq1_start_box.setRange(0.0, duration)
        self.seq1_end_box.setRange(0.0, duration)
        self.seq1_start_box.setValue(0.0)
        self.seq1_end_box.setValue(duration)
        self.seq1_start_box.blockSignals(False)
        self.seq1_end_box.blockSignals(False)
        self.update_seq1_ranges()

    def init_seq2_bounds(self):
        duration = self.synthesis.seq2_length / float(self.synthesis.mocap_fps)
        self.seq2_start_box.blockSignals(True)
        self.seq2_end_box.blockSignals(True)
        self.seq2_start_box.setRange(0.0, duration)
        self.seq2_end_box.setRange(0.0, duration)
        self.seq2_start_box.setValue(0.0)
        self.seq2_end_box.setValue(duration)
        self.seq2_start_box.blockSignals(False)
        self.seq2_end_box.blockSignals(False)
        self.update_seq2_ranges()

    def update_seq1_ranges(self):
        start_val = self.seq1_start_box.value()
        end_val = self.seq1_end_box.value()
        duration = self.synthesis.seq1_length / float(self.synthesis.mocap_fps)

        self.seq1_start_box.setRange(0.0, end_val)
        self.seq1_end_box.setRange(start_val, duration)

        fps = self.synthesis.mocap_fps
        self.synthesis.setSeq1FrameRange(int(start_val * fps), int(end_val * fps))

    def update_seq2_ranges(self):
        start_val = self.seq2_start_box.value()
        end_val = self.seq2_end_box.value()
        duration = self.synthesis.seq2_length / float(self.synthesis.mocap_fps)

        self.seq2_start_box.setRange(0.0, end_val)
        self.seq2_end_box.setRange(start_val, duration)

        fps = self.synthesis.mocap_fps
        self.synthesis.setSeq2FrameRange(int(start_val * fps), int(end_val * fps))

    def on_seq1_idx_changed(self, val):
        self.synthesis.setSeq1Index(val)
        self.synthesis.update()
        self.init_seq1_bounds()

    def on_seq2_idx_changed(self, val):
        self.synthesis.setSeq2Index(val)
        self.synthesis.update()
        self.init_seq2_bounds()

    def on_seq1_incr_changed(self, val):
        self.synthesis.setSeq1FrameIncrement(val)

    def on_seq2_incr_changed(self, val):
        self.synthesis.setSeq2FrameIncrement(val)

    def on_mix_master_changed(self, val):
        for box in self.mix_boxes:
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.apply_mix_offsets()

    def on_offset_master_changed(self, val):
        for box in self.offset_boxes:
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.apply_mix_offsets()

    def apply_mix_offsets(self):
        mix_vals = [box.value() for box in self.mix_boxes]
        offset_vals = [box.value() for box in self.offset_boxes]
        self.synthesis.setEncodingMix(mix_vals)
        self.synthesis.setEncodingOffset(offset_vals)

    # --- Synthesis -> GUI Syncing (Adapts to OSC inputs) ---
    def sync_gui_from_synthesis(self):
        fps = float(self.synthesis.mocap_fps)
        if fps <= 0: return

        def sync_box(box, target_value):
            if abs(box.value() - target_value) > 1e-4:
                box.blockSignals(True)
                box.setValue(target_value)
                box.blockSignals(False)

        # Seq 1 updates
        sync_box(self.seq1_idx_box, self.synthesis.orig_seq1_index)
        sync_box(self.seq1_incr_box, self.synthesis.orig_seq1_frame_incr)

        s1_start_time = self.synthesis.orig_seq1_frame_range[0] / fps
        s1_end_time = self.synthesis.orig_seq1_frame_range[1] / fps

        self.seq1_start_box.blockSignals(True)
        self.seq1_end_box.blockSignals(True)
        self.seq1_start_box.setRange(0.0, s1_end_time)
        self.seq1_end_box.setRange(s1_start_time, self.synthesis.seq1_length / fps)
        self.seq1_start_box.setValue(s1_start_time)
        self.seq1_end_box.setValue(s1_end_time)
        self.seq1_start_box.blockSignals(False)
        self.seq1_end_box.blockSignals(False)

        # Seq 2 updates
        sync_box(self.seq2_idx_box, self.synthesis.orig_seq2_index)
        sync_box(self.seq2_incr_box, self.synthesis.orig_seq2_frame_incr)

        s2_start_time = self.synthesis.orig_seq2_frame_range[0] / fps
        s2_end_time = self.synthesis.orig_seq2_frame_range[1] / fps
        self.seq2_start_box.blockSignals(True)
        self.seq2_end_box.blockSignals(True)
        self.seq2_start_box.setRange(0.0, s2_end_time)
        self.seq2_end_box.setRange(s2_start_time, self.synthesis.seq2_length / fps)
        self.seq2_start_box.setValue(s2_start_time)
        self.seq2_end_box.setValue(s2_end_time)
        self.seq2_start_box.blockSignals(False)
        self.seq2_end_box.blockSignals(False)

        # Latent Mix updates
        if hasattr(self.synthesis, 'encoding_mix'):
            mix_array = self.synthesis.encoding_mix[0, :, 0].detach().cpu().numpy()
            for d, box in enumerate(self.mix_boxes):
                if d < len(mix_array):
                    sync_box(box, float(mix_array[d]))

        # Latent Offset updates
        if hasattr(self.synthesis, 'encoding_offset'):
            offset_array = self.synthesis.encoding_offset[0, :, 0].detach().cpu().numpy()
            for d, box in enumerate(self.offset_boxes):
                if d < len(offset_array):
                    sync_box(box, float(offset_array[d]))

    # --- Original threading/rendering Methods ---
    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update_loop) 
        self.pose_thread.start()
        
    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()
                
    def update_loop(self): 
        while self.pose_thread_event.is_set() == False:
            start_time = time.time()            
            
            self.update_pred_seq()
            self.poseCanvasUpdater.request_canvas_update.emit() 
            self.update_osc()
            
            end_time = time.time()   
            next_update_interval = max(self.pose_thread_interval - (end_time - start_time), 0.0)
            sleep(next_update_interval)

    def update_pred_seq(self):
        self.synthesis.update()       
        self.synth_pose_wpos = self.synthesis.synth_pose_wpos
        self.synth_pose_wrot = self.synthesis.synth_pose_wrot
        self.synth_pose_lrot = self.synthesis.synth_pose_lrot
        
    def update_osc(self):
        self.synth_pose_wpos_rh = np.copy(self.synth_pose_wpos)
        self.synth_pose_wrot_rh = np.copy(self.synth_pose_wrot)
        self.synth_pose_lrot_rh = np.copy(self.synth_pose_lrot)
        
        self.sender.send("/mocap/0/joint/pos_world", self.synth_pose_wpos_rh)
        self.sender.send("/mocap/0/joint/rot_world", self.synth_pose_wrot_rh)
        self.sender.send("/mocap/0/joint/rot_local", self.synth_pose_lrot_rh)

        # Buffer frames when recording
        if hasattr(self, 'is_recording') and self.is_recording:
            pos_local = np.zeros_like(self.synth_pose_wpos)
            pos_local[0] = self.synth_pose_wpos[0] 
            
            self.record_buffer_pos.append(pos_local)
            self.record_buffer_rot.append(np.copy(self.synth_pose_lrot))

    def update_pose_plot(self):
        pose = self.synth_pose_wpos

        points_data = pose
        lines_data = pose[np.array(self.edges).flatten()]
        
        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        self.pose_canvas_points.setData(pos=pose, color=(1.0, 1.0, 1.0, 0.5))