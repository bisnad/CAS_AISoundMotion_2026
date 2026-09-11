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

config = {"mapping": None,
          "synthesis": None,
          "sender": None,
          "update_interval": 0.02,
          "view_min": np.array([-100, -100, -100], dtype=np.float32),
          "view_max": np.array([100, 100, 100], dtype=np.float32),
          "view_ele": 90,
          "view_azi": -90,
          "view_dist": 250,
          "view_line_width": 2.0,
          "osc_ip": "127.0.0.1",
          "osc_port": 9004,
          "mocap_fps": 50
    }

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Important: avoid Euler pole / roll weirdness
        self.opts['rotationMethod'] = 'quaternion'

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()

        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos

        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.LeftButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                # Restore original "move center position" behavior
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                # Keep the standard orbit mapping
                self.orbit(-diff.x(), diff.y())

        elif ev.buttons() == QtCore.Qt.MiddleButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')

        else:
            super().mouseMoveEvent(ev)

class MappingCanvas(pg.PlotWidget):
    
    def __init__(self, points2D, callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.points2D = points2D
        self.callback = callback
        
        self.scatter = pg.ScatterPlotItem(
            x=points2D[:, 0],
            y=points2D[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10
        )
        
        self.addItem(self.scatter)
        
        # Interactive points and encodings (red points are user-selected)
        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.addItem(self.click_scatter)
        self.add_point_interval_ms = 100
        self.remove_point_interval_ms = 10
        
        self.left_move_timer = QtCore.QTimer()
        self.left_move_timer.setInterval(self.add_point_interval_ms)
        self.left_move_timer.timeout.connect(self._add_point_by_timer)
        
        self.right_move_timer = QtCore.QTimer()
        self.right_move_timer.setInterval(self.remove_point_interval_ms)
        self.right_move_timer.timeout.connect(self._remove_point_by_timer)
        self.pending_mouse_pos = None
        
        # Interaction state
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.right_button_pressed = False
        self.last_pos = None
        
    def add_click_point(self, pos):
        
        self.click_points.append({'pos': (pos[0], pos[1])})
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )
        
    def remove_click_point(self, idx):
        
        self.click_points.pop(idx)
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )

    def remove_click_points(self):
        self.click_points.clear()
        self.click_scatter.setData([], [])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = True
            self.pending_mouse_pos = event.pos()
            self.left_move_timer.start()
            
            if self.callback: 
                self.callback("left_press", event.pos())
            
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = True
            self.pending_mouse_pos = event.pos()
            self.right_move_timer.start()
            
            if self.callback: 
                self.callback("right_press", event.pos())
            
        elif event.button() == Qt.MiddleButton:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = False
            self.left_move_timer.stop()
            
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = False
            self.right_move_timer.stop()
        else:
            super().mouseReleaseEvent(event)
            
    def mouseMoveEvent(self, event):
        self.pending_mouse_pos = event.pos()
        
        if event.button() == Qt.MiddleButton:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Clear points/encodings on 'C' key."""
        if event.key() == QtCore.Qt.Key_C:
            self.remove_click_points()
            #self.synthesis.clearEncodings()
        super().keyPressEvent(event)
            
    def _add_point_by_timer(self):
        if self.left_button_pressed and self.pending_mouse_pos is not None and self.callback:
            self.callback("left_move", self.pending_mouse_pos)

    def _remove_point_by_timer(self):
        if self.right_button_pressed and self.pending_mouse_pos is not None and self.callback:
            self.callback("right_move", self.pending_mouse_pos)
        
class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.mapping = config["mapping"]
        self.synthesis = config["synthesis"]
        self.sender = config["sender"]
        self.mocap_fps = config.get("mocap_fps", 50)
        
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
        
        # mapping canvas
        self.mapping_canvas = MappingCanvas(self.mapping.Z_tsne, callback=self.handle_mapping_mouse)
        
        # pose canvas
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

        self.q_clear_buttom = QtWidgets.QPushButton("Clear", self)
        self.q_clear_buttom.clicked.connect(self.clear)  

        self.q_record_button = QtWidgets.QPushButton("Record", self)
        self.q_record_button.setCheckable(True)
        self.q_record_button.clicked.connect(self.toggle_recording)
        
        self.q_exit_button = QtWidgets.QPushButton("Exit", self)
        self.q_exit_button.clicked.connect(self.exit_application)
        
        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_buttom,0,0)
        self.q_button_grid.addWidget(self.q_stop_buttom,0,1)
        self.q_button_grid.addWidget(self.q_clear_buttom,0,2)
        self.q_button_grid.addWidget(self.q_record_button,0,3)
        self.q_button_grid.addWidget(self.q_exit_button,0,4)
        
        # OSC IP and Port Layout
        self.q_osc_layout = QtWidgets.QFormLayout()
        
        self.q_osc_ip = QtWidgets.QLineEdit(config.get("osc_ip", "127.0.0.1"))
        self.q_osc_ip.textChanged.connect(self.change_osc_ip)
        self.q_osc_layout.addRow("OSC IP:", self.q_osc_ip)
        
        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1024, 65535)
        self.q_osc_port.setValue(config.get("osc_port", 9004))
        self.q_osc_port.valueChanged.connect(self.change_osc_port)
        self.q_osc_layout.addRow("OSC Port:", self.q_osc_port)

        # Wrap OSC layout in a container so it aligns to the top right without stretching
        osc_container = QtWidgets.QWidget()
        osc_vbox = QtWidgets.QVBoxLayout()
        osc_vbox.addLayout(self.q_osc_layout)
        osc_vbox.addStretch()  # Forces the inputs to stay at the top
        osc_container.setLayout(osc_vbox)
        osc_container.setMinimumWidth(200)

        # Add all three columns to the Canvas Grid (Mapping Canvas | Pose Canvas | OSC Controls)
        self.canvas_grid = QtWidgets.QGridLayout()
        self.canvas_grid.addWidget(self.mapping_canvas, 0, 0)
        self.canvas_grid.addWidget(self.pose_canvas, 0, 1)
        self.canvas_grid.addWidget(osc_container, 0, 2)
        
        self.canvas_grid.setColumnStretch(0, 1) # Mapping expands
        self.canvas_grid.setColumnStretch(1, 1) # Skeleton expands
        self.canvas_grid.setColumnStretch(2, 0) # OSC controls fixed width

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.canvas_grid, 0, 0)
        self.q_grid.addLayout(self.q_button_grid, 1, 0)
        
        self.q_grid.setRowStretch(0, 1)
        self.q_grid.setRowStretch(1, 0)
        
        self.setLayout(self.q_grid)
        # Increased window width slightly to accommodate the third column natively
        self.setGeometry(50, 50, 512 * 2 + 200, 612)
        self.setWindowTitle("Motion Autoencoder - Latent Exploration")

        # Signals that can be emitted 
        self.poseCanvasUpdater = PoseCanvasUpdater()
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)
        
        # Timer for continuous add (while dragging left button)
        self.add_mapping_timer = QtCore.QTimer()
        self.add_mapping_timer.setInterval(100)  # ms
        self.add_mapping_timer.timeout.connect(self.continuous_add_mapping)
        
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
        filename = f"recorded_latent_motion_{timestamp}.fbx"
        
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
        skel = getattr(self.synthesis, 'skeleton', {})
        
        fbx_data.skeleton_joints = skel.get("joints", [])
        fbx_data.skeleton_children = skel.get("children", [])
        fbx_data.skeleton_parents = skel.get("parents", [])
        fbx_data.skeleton_joint_offsets = skel.get("offsets", [])
        fbx_data.skeleton_root_node = None 
        fbx_data.skeleton_nodes = []
        
        fbx_data.motion_rot_sequence = [0, 1, 2] 
        fbx_data.motion_frame_rate = float(self.mocap_fps)
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

    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        self.pose_thread.start()
        
    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()

    def clear(self):
        self.mapping_canvas.remove_click_points()
        self.synthesis.clearEncodings()
        
    def continuous_add_mapping(self):
        if self.mapping_canvas.left_button_pressed and self.mapping_canvas.last_mouse_pos is not None:
            x, y = self.mapping_canvas.last_mouse_pos[0], self.mapping_canvas.last_mouse_pos[1]
            self.mapping_canvas.add_click_point([x, y])
            
            encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
            encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
            self.synthesis.addEncoding(encoding)
        
    def handle_mapping_mouse(self, button, scene_pos):
        if button == "left_press":
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos):
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                self.mapping_canvas.left_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                self.mapping_canvas.add_click_point([x, y])
                
                encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
                encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
                self.synthesis.addEncoding(encoding)
            return
        
        elif button == "left_move":
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos) and self.mapping_canvas.left_button_pressed:
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                self.mapping_canvas.left_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                self.mapping_canvas.add_click_point([x, y])
                
                encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
                encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
                self.synthesis.addEncoding(encoding)
            return
            
        elif button == "left_release":
            self.mapping_canvas.left_button_pressed = False
            return
            
        elif button == "right_press":
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos):
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                self.mapping_canvas.right_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                to_remove_indices = []
                radius=0.5
                
                for i, p in enumerate(self.mapping_canvas.click_points):
                    dx = p['pos'][0] - x
                    dy = p['pos'][1] - y
                    dist = (dx*dx + dy*dy)**0.5
                    if dist < radius:
                        to_remove_indices.append(i)
                        
                for idx in reversed(to_remove_indices):
                    self.mapping_canvas.remove_click_point(idx)
                    self.synthesis.removeEncoding(idx)
            return

        elif button == "right_move":
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos) and self.mapping_canvas.right_button_pressed:
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                self.mapping_canvas.right_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                to_remove_indices = []
                radius=0.5
                
                for i, p in enumerate(self.mapping_canvas.click_points):
                    dx = p['pos'][0] - x
                    dy = p['pos'][1] - y
                    dist = (dx*dx + dy*dy)**0.5
                    if dist < radius:
                        to_remove_indices.append(i)
                
                for idx in reversed(to_remove_indices):
                    self.mapping_canvas.remove_click_point(idx)
                    self.synthesis.removeEncoding(idx)
            return
            
        elif button == "right_release":
            self.mapping_canvas.right_button_pressed = False
            return

                
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