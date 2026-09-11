import torch
import numpy as np
import datetime
import os
import json

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.Vector as qVector
import pyqtgraph.opengl as gl
from pathlib import Path

from threading import Thread, Event
import time
from time import sleep

import motion_synthesis
from common import mocap_tools as mocap
from common import bvh_tools as bvh
from common import fbx_tools as fbx

config = {
    "synthesis": None,
    "sender": None,
    "update_interval": 0.02,
    "view_min": np.array([-100, -100, -100], dtype=np.float32),
    "view_max": np.array([100, 100, 100], dtype=np.float32),
    "view_center": np.array([0, 0, 100], dtype=np.float32),
    "view_ele": 15,    # Adjusted for realistic viewing
    "view_azi": 45,    # Adjusted for realistic viewing
    "view_dist": 250,
    "view_line_width": 2.0,
    "osc_ip": "127.0.0.1",
    "osc_port": 9007,
    "recording_export_format": "npz", 
    "recording_npz_topology_json": None
}

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        kwargs['rotationMethod'] = 'quaternion'
        super().__init__(*args, **kwargs)

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

        self.is_recording = False
        self.record_buffer_pos = []
        self.record_buffer_rot = []
        self.record_buffer_wpos = []
        self.record_buffer_wrot = []

        self.recording_export_format = config.get("recording_export_format", "fbx").lower()
        self.recording_npz_topology_json = config.get("recording_npz_topology_json", None)

        self.view_min = config["view_min"]
        self.view_max = config["view_max"]
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_line_width = config["view_line_width"]

        view_center = config.get("view_center", np.array([0, 0, 100], dtype=np.float32))
        self.view_center = qVector(view_center[0], view_center[1], view_center[2])

        self.pose_canvas = CustomGLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)

        self.pose_canvas.setCameraParams(center=self.view_center)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)

        self.q_start_buttom = QtWidgets.QPushButton("Start", self)
        self.q_start_buttom.clicked.connect(self.start) 

        self.q_stop_buttom = QtWidgets.QPushButton("Stop", self)
        self.q_stop_buttom.clicked.connect(self.stop) 

        self.q_record_button = QtWidgets.QPushButton("Record", self)
        self.q_record_button.setCheckable(True)
        self.q_record_button.clicked.connect(self.toggle_recording)

        self.q_exit_button = QtWidgets.QPushButton("Exit", self)
        self.q_exit_button.clicked.connect(self.exit_application)

        # --- Continuation / Live mode controls ---
        # Style used to highlight whichever mode (continuation vs live) is currently active.
        self._continuation_active_style = "background-color: #2e7d32; color: white; font-weight: bold;"
        self._live_active_style = "background-color: #2e7d32; color: white; font-weight: bold;"
        self._inactive_style = ""

        self.q_continuation_button = QtWidgets.QPushButton("Continuation", self)
        self.q_continuation_button.clicked.connect(self.trigger_continuation)

        self.q_live_button = QtWidgets.QPushButton("Live", self)
        self.q_live_button.clicked.connect(self.return_to_live)

        self._refresh_mode_buttons()

        self.q_button_grid = QtWidgets.QHBoxLayout()
        self.q_button_grid.addWidget(self.q_start_buttom)
        self.q_button_grid.addWidget(self.q_stop_buttom)
        self.q_button_grid.addWidget(self.q_record_button)
        self.q_button_grid.addWidget(self.q_continuation_button)
        self.q_button_grid.addWidget(self.q_live_button)
        self.q_button_grid.addWidget(self.q_exit_button)

        self.q_controls_layout = QtWidgets.QFormLayout()
        expanding_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.q_osc_ip = QtWidgets.QLineEdit(config.get("osc_ip", "127.0.0.1"))
        self.q_osc_ip.setSizePolicy(expanding_policy)
        self.q_osc_ip.textChanged.connect(self.change_osc_ip)
        self.q_controls_layout.addRow("OSC Send IP:", self.q_osc_ip)

        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1024, 65535)
        self.q_osc_port.setValue(config.get("osc_port", 9004))
        self.q_osc_port.setSizePolicy(expanding_policy)
        self.q_osc_port.valueChanged.connect(self.change_osc_port)
        self.q_controls_layout.addRow("OSC Send Port:", self.q_osc_port)

        self.q_seq_index = QtWidgets.QSpinBox()
        self.q_seq_index.setMinimum(0)
        self.q_seq_index.setMaximum(max(0, len(self.synthesis.orig_sequences) - 1))
        self.q_seq_index.setValue(self.synthesis.orig_seq_index)
        self.q_seq_index.setSizePolicy(expanding_policy)
        self.q_seq_index.valueChanged.connect(self.change_seq_index)
        self.q_controls_layout.addRow("Seq Index:", self.q_seq_index)

        self.q_start_time = QtWidgets.QDoubleSpinBox()
        self.q_start_time.setMinimum(0.0)
        self.q_start_time.setDecimals(3)
        self.q_start_time.setSingleStep(0.1)
        self.q_start_time.setSizePolicy(expanding_policy)
        self.update_start_time_bounds()
        current_time = self.synthesis.orig_seq_start_frame_index / max(1, self.synthesis.mocap_fps)
        self.q_start_time.setValue(current_time)
        self.q_start_time.valueChanged.connect(self.change_start_time)
        self.q_controls_layout.addRow("Start Time (s):", self.q_start_time)

        self.q_pi_temp = QtWidgets.QDoubleSpinBox()
        self.q_pi_temp.setRange(0.1, 2.5)
        self.q_pi_temp.setSingleStep(0.1)
        self.q_pi_temp.setValue(self.synthesis.pi_temperature)
        self.q_pi_temp.setSizePolicy(expanding_policy)
        self.q_pi_temp.valueChanged.connect(lambda v: self.synthesis.setPiTemp(v))
        self.q_controls_layout.addRow("MDN PI Temp:", self.q_pi_temp)

        self.q_gauss_temp = QtWidgets.QDoubleSpinBox()
        self.q_gauss_temp.setRange(0.0, 2.0)
        self.q_gauss_temp.setSingleStep(0.1)
        self.q_gauss_temp.setValue(self.synthesis.gaussian_temp)
        self.q_gauss_temp.setSizePolicy(expanding_policy)
        self.q_gauss_temp.valueChanged.connect(lambda v: self.synthesis.setGaussianTemp(v))
        self.q_controls_layout.addRow("MDN Gauss Temp:", self.q_gauss_temp)

        self.q_top_p = QtWidgets.QDoubleSpinBox()
        self.q_top_p.setRange(0.5, 0.95)
        self.q_top_p.setSingleStep(0.05)
        self.q_top_p.setValue(self.synthesis.top_p)
        self.q_top_p.setSizePolicy(expanding_policy)
        self.q_top_p.valueChanged.connect(lambda v: self.synthesis.setTopP(v))
        self.q_controls_layout.addRow("MDN TopP:", self.q_top_p)

        self.q_cutoff = QtWidgets.QDoubleSpinBox()
        self.q_cutoff.setRange(0.0, 10.0)
        self.q_cutoff.setSingleStep(0.1)
        self.q_cutoff.setValue(self.synthesis.one_euro_filter.mincutoff)
        self.q_cutoff.setSizePolicy(expanding_policy)
        self.q_cutoff.valueChanged.connect(lambda v: self.synthesis.one_euro_filter.setMinCutOff(v))
        self.q_controls_layout.addRow("1€ Cutoff:", self.q_cutoff)

        self.q_beta = QtWidgets.QDoubleSpinBox()
        self.q_beta.setRange(0.0, 1.0)
        self.q_beta.setSingleStep(0.01)
        self.q_beta.setValue(self.synthesis.one_euro_filter.beta)
        self.q_beta.setSizePolicy(expanding_policy)
        self.q_beta.valueChanged.connect(lambda v: self.synthesis.one_euro_filter.setBeta(v))
        self.q_controls_layout.addRow("1€ Beta:", self.q_beta)

        # --- Blend duration control (frames to cross-fade between continuation and live) ---
        self.q_blend_frames = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.q_blend_frames.setRange(1, 120)
        self.q_blend_frames.setValue(int(getattr(self.synthesis, "blend_frames_max", 15)))
        self.q_blend_frames.setSizePolicy(expanding_policy)
        self.q_blend_frames.valueChanged.connect(self.change_blend_frames_max)

        self.q_blend_frames_label = QtWidgets.QLabel(str(self.q_blend_frames.value()))
        self.q_blend_frames_label.setMinimumWidth(30)
        self.q_blend_frames_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        q_blend_row = QtWidgets.QHBoxLayout()
        q_blend_row.addWidget(self.q_blend_frames)
        q_blend_row.addWidget(self.q_blend_frames_label)
        self.q_controls_layout.addRow("Blend Frames:", q_blend_row)

        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(self.q_controls_layout)
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
        self.setWindowTitle("Motion Continuation")

        self.poseCanvasUpdater = PoseCanvasUpdater()
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)

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
            self.record_buffer_wpos = []
            self.record_buffer_wrot = []
            print("Recording started...")
        else:
            self.q_record_button.setText("Record")
            self.save_recording()

    # --- Continuation / Live mode handlers ---
    def _refresh_mode_buttons(self):
        is_continuing = bool(getattr(self.synthesis, "is_continuing", False))
        self.q_continuation_button.setStyleSheet(self._continuation_active_style if is_continuing else self._inactive_style)
        self.q_live_button.setStyleSheet(self._inactive_style if is_continuing else self._live_active_style)

    def trigger_continuation(self):
        self.synthesis.trigger_continuation()
        self._refresh_mode_buttons()

    def return_to_live(self):
        self.synthesis.return_to_live()
        self._refresh_mode_buttons()

    def change_blend_frames_max(self, val):
        self.synthesis.blend_frames_max = int(val)
        self.q_blend_frames_label.setText(str(int(val)))

    def _quat_to_euler(self, quat_data):
        mocap_t = mocap.Mocap_Tools()
        return mocap_t.quat_to_euler(quat_data, getattr(self.synthesis, "rot_sequence", [0, 1, 2]))

    def _build_recording_mocap_dict(self):
        mocap_data = {
            "frame_rate": float(self.synthesis.mocap_fps),
            "rot_sequence": getattr(self.synthesis, "rot_sequence", [0, 1, 2]),
            "skeleton": self.synthesis.skeleton,
            "motion": {
                "pos_local": np.array(self.record_buffer_pos, dtype=np.float32),
                "rot_local": np.array(self.record_buffer_rot, dtype=np.float32),
                "pos_world": np.array(self.record_buffer_wpos, dtype=np.float32),
                "rot_world": np.array(self.record_buffer_wrot, dtype=np.float32),
            }
        }

        mocap_data["motion"]["rot_local_euler"] = self._quat_to_euler(mocap_data["motion"]["rot_local"])
        return mocap_data

    def _export_recording_fbx(self, base_path):
        mocap_t = mocap.Mocap_Tools()
        fbx_exporter = fbx.FBX_Tools()
        mocap_data = self._build_recording_mocap_dict()
        fbx_scene = mocap_t.mocap_to_fbx([mocap_data])
        fbx_exporter.write(fbx_scene, base_path + ".fbx")

    def _export_recording_bvh(self, base_path):
        mocap_t = mocap.Mocap_Tools()
        bvh_exporter = bvh.BVH_Tools()
        mocap_data = self._build_recording_mocap_dict()
        bvh_data = mocap_t.mocap_to_bvh(mocap_data)
        bvh_exporter.write(bvh_data, base_path + ".bvh")

    def _export_recording_npz(self, base_path):
        mocap_t = mocap.Mocap_Tools()
        mocap_data = self._build_recording_mocap_dict()
        npz_dict = mocap_t.mocap_to_npz([mocap_data])
        
        # Robust non-pickled export
        np.savez_compressed(base_path + ".npz", **npz_dict)

        topo = {
            "parents": list(self.synthesis.skeleton.get("parents", [])),
            "children": [list(c) for c in self.synthesis.skeleton.get("children", [])],
            "joints": list(self.synthesis.skeleton.get("joints", []))
        }
        with open(base_path + "_topology.json", "w") as f:
            json.dump(topo, f, indent=2)

    def save_recording(self):
        if len(self.record_buffer_rot) == 0:
            print("No frames were recorded.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"recorded_motion_{timestamp}"

        try:
            if self.recording_export_format == "fbx":
                self._export_recording_fbx(base_path)
                print(f"Successfully saved recording to: {base_path}.fbx")
            elif self.recording_export_format == "bvh":
                self._export_recording_bvh(base_path)
                print(f"Successfully saved recording to: {base_path}.bvh")
            elif self.recording_export_format == "npz":
                self._export_recording_npz(base_path)
                print(f"Successfully saved recording to: {base_path}.npz")
            else:
                raise ValueError(f"Unsupported recording export format: {self.recording_export_format}")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def update_start_time_bounds(self):
        fps = max(1, self.synthesis.mocap_fps)
        seq_length = self.synthesis.seq_length
        total_frames = self.synthesis.orig_sequences[self.synthesis.orig_seq_index].shape[0]

        max_start_frame = max(0, total_frames - seq_length)
        max_time_seconds = max_start_frame / fps
        self.q_start_time.setMaximum(max_time_seconds)

    def change_seq_index(self, val):
        self.synthesis.setOrigSeqIndex(val)
        self.update_start_time_bounds()

    def change_start_time(self, val):
        fps = max(1, self.synthesis.mocap_fps)
        frame_index = int(val * fps)
        self.synthesis.setOrigSeqStartFrameIndex(frame_index)

    def sync_gui_with_synthesis(self):
        def update_val(widget, val, tolerance=1e-4):
            if abs(widget.value() - val) > tolerance:
                widget.blockSignals(True)
                widget.setValue(val)
                widget.blockSignals(False)

        update_val(self.q_seq_index, self.synthesis.orig_seq_index)
        self.update_start_time_bounds()
        current_time = self.synthesis.orig_seq_start_frame_index / max(1, self.synthesis.mocap_fps)
        update_val(self.q_start_time, current_time)
        update_val(self.q_pi_temp, self.synthesis.pi_temperature)
        update_val(self.q_gauss_temp, self.synthesis.gaussian_temp)
        update_val(self.q_top_p, self.synthesis.top_p)
        update_val(self.q_cutoff, self.synthesis.one_euro_filter.mincutoff)
        update_val(self.q_beta, self.synthesis.one_euro_filter.beta)

        blend_val = int(getattr(self.synthesis, "blend_frames_max", self.q_blend_frames.value()))
        if self.q_blend_frames.value() != blend_val:
            self.q_blend_frames.blockSignals(True)
            self.q_blend_frames.setValue(blend_val)
            self.q_blend_frames_label.setText(str(blend_val))
            self.q_blend_frames.blockSignals(False)

        # Keep the mode buttons in sync with synthesis state (e.g. changes via OSC)
        self._refresh_mode_buttons()

    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        self.pose_thread.start()

    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()

    def exit_application(self):
        if hasattr(self, 'pose_thread_event') and not self.pose_thread_event.is_set():
            self.stop()
        self.close()

    def update(self):
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

        if self.is_recording:
            pos_local = np.zeros_like(self.synth_pose_wpos)
            pos_local[0] = self.synth_pose_wpos[0]

            self.record_buffer_pos.append(np.copy(pos_local))
            self.record_buffer_rot.append(np.copy(self.synth_pose_lrot))
            self.record_buffer_wpos.append(np.copy(self.synth_pose_wpos))
            self.record_buffer_wrot.append(np.copy(self.synth_pose_wrot))

    def update_pose_plot(self):
        self.sync_gui_with_synthesis()
        
        pose = np.copy(self.synth_pose_wpos)
        
        # Apply Y-up to Z-up transformation so character stands up in pyqtgraph
        theta = np.radians(90.0) 
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_x_mat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t]
        ], dtype=np.float32)
        pose_z_up = np.dot(pose, rot_x_mat.T)
        pose_z_up[..., 0] *= -1.0 # Handedness fix

        points_data = pose_z_up
        lines_data = pose_z_up[np.array(self.edges).flatten()]

        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        self.pose_canvas_points.setData(pos=pose_z_up, color=(1.0, 1.0, 1.0, 0.5))
