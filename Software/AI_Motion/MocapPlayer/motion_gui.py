import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.Vector as qVector
import pyqtgraph.opengl as gl
from pathlib import Path

from threading import Thread, Event
import time

import motion_player

config = {"player": None,
          "sender": None,
          "view_min": np.array([-100, -100, -100], dtype=np.float32),
          "view_max": np.array([100, 100, 100], dtype=np.float32),
          "view_center": np.array([0, 0, 100], dtype=np.float32),
          "view_ele": 90,
          "view_azi": -90,
          "view_dist": 250,
          "view_scale": 1.0,
          "view_line_width": 3.0,
          "view_joint_size": 9.0
    }

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, default_dist=250, default_azi=-90, default_ele=90, *args, **kwargs):
        super().__init__(*args, rotationMethod='quaternion', **kwargs)

        self.default_dist = default_dist
        self.default_azi = default_azi
        self.default_ele = default_ele

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.setCameraParams(
                distance=self.default_dist,
                azimuth=self.default_azi,
                elevation=self.default_ele
            )
        super().mouseDoubleClickEvent(ev)

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

class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.player = config["player"]
        self.sender = config["sender"]
        
        fps = int(self.player.get_fps())
        self.pose_thread_interval = 1.0 / fps
        self.pose_thread_event = None
    
        # header
        file_name = getattr(self.player, 'file_name', "No file loaded")
        self.q_file_label = QtWidgets.QLabel(Path(file_name).stem)

        self.topology_file_name = config.get("topology_file_name", "")
        topo_name = Path(self.topology_file_name).stem if self.topology_file_name else "No topology loaded"
        self.q_topo_label = QtWidgets.QLabel("Topology: " + topo_name)

        self.q_time_label = QtWidgets.QLabel("Time: 0.00s | Loop: 0.00s - 0.00s", self)

        # Pose canvas setup
        self.view_min = config["view_min"]
        self.view_max = config["view_max"]   
        view_center = config["view_center"]
        self.view_center = qVector(view_center[0], view_center[1], view_center[2])
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_scale = config["view_scale"]
        self.view_line_width = config["view_line_width"]
        self.view_joint_size = config["view_joint_size"]
        
        self.pose_canvas = CustomGLViewWidget(
            default_dist=self.view_dist,
            default_azi=self.view_azi,
            default_ele=self.view_ele
        )
        self.pose_canvas.setCameraParams(center=self.view_center)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)
        
        # Track items drawn to the canvas mapped by skeleton ID
        self.skeleton_items = {}
        
        # Buttons
        self.q_load_buttom = QtWidgets.QPushButton("load Mocap", self)
        self.q_load_buttom.clicked.connect(self.choose_file)  

        self.q_load_topo_button = QtWidgets.QPushButton("Load Topo", self)
        self.q_load_topo_button.clicked.connect(self.choose_topo_file)

        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  

        self.q_exit_button = QtWidgets.QPushButton("Exit", self)
        self.q_exit_button.clicked.connect(self.exit_application)
        
        self.q_fps = QtWidgets.QSpinBox(self)
        self.q_fps.setMinimum(1)
        self.q_fps.setMaximum(200)
        self.q_fps.setValue(fps)
        self.q_fps.valueChanged.connect(self.change_fps)  
        
        # Skeleton Selector
        self.q_skeleton_selector = QtWidgets.QComboBox(self)
        self.q_skeleton_selector.addItem("None")
        
        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_load_buttom, 0, 0)
        self.q_button_grid.addWidget(self.q_load_topo_button, 0, 1)
        self.q_button_grid.addWidget(self.q_start_buttom, 0, 2)
        self.q_button_grid.addWidget(self.q_stop_buttom, 0, 3)
        self.q_button_grid.addWidget(self.q_exit_button, 0, 4)
        self.q_button_grid.addWidget(QtWidgets.QLabel("FPS:"), 0, 5, alignment=Qt.AlignRight)
        self.q_button_grid.addWidget(self.q_fps, 0, 6)
        self.q_button_grid.addWidget(QtWidgets.QLabel("Follow:"), 0, 7, alignment=Qt.AlignRight)
        self.q_button_grid.addWidget(self.q_skeleton_selector, 0, 8)
        
        # We will use sliders with a 0-1000 range to represent percentage of max play time
        self.q_time_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_time_slider.setMinimum(0)
        self.q_time_slider.setMaximum(1000)
        self.q_time_slider.setValue(0)
        self.q_time_slider.sliderMoved.connect(self.change_play_time)  
        
        self.q_start_time_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_start_time_slider.setMinimum(0)
        self.q_start_time_slider.setMaximum(1000)
        self.q_start_time_slider.setValue(0)
        self.q_start_time_slider.sliderMoved.connect(self.change_start_time)  
        
        self.q_end_time_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_end_time_slider.setMinimum(0)
        self.q_end_time_slider.setMaximum(1000)
        self.q_end_time_slider.setValue(1000)
        self.q_end_time_slider.sliderMoved.connect(self.change_end_time)  
        
        # osc sender config
        self.sender_active = True
        self.q_sender_toggle = QtWidgets.QCheckBox("osc", self)
        self.q_sender_toggle.stateChanged.connect(lambda:self.toggle_sender(self.q_sender_toggle))  
        self.q_sender_toggle.setChecked(self.sender_active)
          
        if self.sender:
            sender_ip_string, sender_port = self.sender.get_address()
            self.sender_ip = [int(el) for el in sender_ip_string.split(".")]
            self.sender_port = sender_port
        else:
            self.sender_ip = [127, 0, 0, 1]
            self.sender_port = 9005

        self.q_sender_ip = []
        for i in range(4):
            _w = QtWidgets.QSpinBox(self)
            _w.setMinimum(0)
            _w.setMaximum(255)
            _w.setValue(self.sender_ip[i])
            _w.valueChanged.connect(self.change_sender_address)  
            self.q_sender_ip.append(_w)
            
        self.q_sender_port = QtWidgets.QSpinBox(self)
        self.q_sender_port.setMinimum(0)
        self.q_sender_port.setMaximum(65535)
        self.q_sender_port.setValue(self.sender_port)
        self.q_sender_port.valueChanged.connect(self.change_sender_address)  
        
        self.q_sender_grid = QtWidgets.QGridLayout()
        self.q_sender_grid.addWidget(self.q_sender_toggle,0,0)
        self.q_sender_grid.addWidget(self.q_sender_ip[0],0,1)
        self.q_sender_grid.addWidget(self.q_sender_ip[1],0,2)
        self.q_sender_grid.addWidget(self.q_sender_ip[2],0,3)
        self.q_sender_grid.addWidget(self.q_sender_ip[3],0,4)
        self.q_sender_grid.addWidget(self.q_sender_port,0,5)
    
        # final layout
        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addWidget(self.q_file_label, 0, 0)
        self.q_grid.addWidget(self.q_topo_label, 1, 0)
        self.q_grid.addWidget(self.q_time_label, 2, 0)
        self.q_grid.addWidget(self.pose_canvas, 3, 0)
        self.q_grid.addWidget(self.q_time_slider, 4, 0)
        self.q_grid.addWidget(self.q_start_time_slider, 5, 0)
        self.q_grid.addWidget(self.q_end_time_slider, 6, 0)
        self.q_grid.addLayout(self.q_button_grid, 7, 0)
        self.q_grid.addLayout(self.q_sender_grid, 8, 0)
        
        self.q_grid.setRowStretch(0, 0)
        self.q_grid.setRowStretch(1, 0)
        self.q_grid.setRowStretch(2, 0)
        self.q_grid.setRowStretch(3, 1)
        self.q_grid.setRowStretch(4, 0)
        self.q_grid.setRowStretch(5, 0)
        self.q_grid.setRowStretch(6, 0)
        self.q_grid.setRowStretch(7, 0)
        self.q_grid.setRowStretch(8, 0)
        
        self.setLayout(self.q_grid)

        self.setGeometry(50,50,512,612)
        self.setWindowTitle("Mocap Player")
        
        self.gui_needs_repaint = False
        self.poseCanvasUpdater = PoseCanvasUpdater()
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_gui_and_canvas)

        self.update_gui_labels()
                
    def choose_file(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'mocap',"Mocap Files (*.bvh *.fbx *.npz)")
        file_name = file_name[0]
        if len(file_name) == 0:
            return
        self.load_file(file_name)

    def choose_topo_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Topology File', '', "Topology Files (*.json *.yaml *.txt);;All Files (*.*)"
        )
        if len(file_name) == 0:
            return
            
        self.topology_file_name = file_name
        self.q_topo_label.setText("Topology: " + Path(file_name).stem)
        
        # Optional: If you want it to immediately reload the current NPZ file using the new topology
        if hasattr(self.player, 'file_name') and self.player.file_name.lower().endswith(".npz"):
            self.load_file(self.player.file_name)
        
    def load_file(self, file_name):
        sender_active = self.sender.get_active() if self.sender else False
        if sender_active:
            self.sender.set_active(False)
            
        self.q_file_label.setText("Loading...")
        self.q_file_label.repaint()
        
        # Branch loading logic based on extension
        if file_name.lower().endswith(".npz"):
            if not getattr(self, 'topology_file_name', ""):
                print("Warning: Loading NPZ file but no topology file is set.")
            self.player.load(file_name, self.topology_file_name)
        else:
            # BVH and FBX don't need the topology file
            self.player.load(file_name)
        
        self.change_fps(self.player.get_fps())
        self.q_file_label.setText("Mocap: " + Path(file_name).stem)
        
        self.q_time_slider.setValue(0)
        self.update_gui_labels()
        self.update_gui_and_canvas()
        
        if sender_active:
            self.sender.set_active(True)

    def start(self):
        if self.pose_thread_event is not None and not self.pose_thread_event.is_set():
            return 
            
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.playback_loop)
        self.pose_thread.start()
        
    def stop(self):
        if self.pose_thread_event is None:
            return
        self.pose_thread_event.set()
        self.pose_thread.join()

    def exit_application(self):
        """Cleanly stops the thread and closes the application."""
        # Stop the prediction thread if it is currently running
        if hasattr(self, 'pose_thread_event') and not self.pose_thread_event.is_set():
            self.stop()
            
        # Close the PyQt window (triggers app.lastWindowClosed in the main script)
        self.close()
        
    def change_fps(self, fps):
        fps = int(fps)
        self.player.set_fps(fps)
        self.q_fps.blockSignals(True)
        self.q_fps.setValue(fps)
        self.q_fps.blockSignals(False)
        self.pose_thread_interval = 1.0 / fps
        
    def playback_loop(self):
        last_time = time.time()
        while not self.pose_thread_event.is_set():
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            self.player.update(dt) 
            self.poseCanvasUpdater.request_canvas_update.emit()
            
            time.sleep(max(0, self.pose_thread_interval - (time.time() - current_time)))
            
    def update_gui_and_canvas(self):
        """Called by the main Qt thread to update drawing and UI"""
        poses = self.player.get_frame()
        
        # Pre-process active IDs to dynamically update the dropdown
        active_ids = {pose["id"] for pose in poses}
        
        if not hasattr(self, 'known_skeleton_ids'):
            self.known_skeleton_ids = set()
            
        if active_ids != self.known_skeleton_ids:
            current_selection = self.q_skeleton_selector.currentText()
            self.q_skeleton_selector.blockSignals(True)
            self.q_skeleton_selector.clear()
            self.q_skeleton_selector.addItem("None")
            for sid in sorted(list(active_ids)):
                self.q_skeleton_selector.addItem(str(sid))
                
            idx = self.q_skeleton_selector.findText(current_selection)
            if idx >= 0:
                self.q_skeleton_selector.setCurrentIndex(idx)
                
            self.known_skeleton_ids = active_ids
            self.q_skeleton_selector.blockSignals(False)
            
        selected_text = self.q_skeleton_selector.currentText()
        target_center = qVector(0, 0, 0)
        target_found = False

        # Ensure the points tracking dictionary exists dynamically
        if not hasattr(self, 'skeleton_points'):
            self.skeleton_points = {}
            
        for pose in poses:
            skel_id = pose["id"]
            skeleton_hierarchy = pose["skeleton"]
            pos_world = pose["pos_world"] * self.view_scale
            
            # --- Calculating Center Point tracking (Root Joint) ---
            if str(skel_id) == selected_text and len(pos_world) > 0:
                # Identify the root joint (the one with parent index -1)
                parents_list = list(skeleton_hierarchy["parents"])
                
                # Default to index 0 if for some reason -1 is missing from the hierarchy
                root_idx = parents_list.index(-1) if -1 in parents_list else 0 
                
                root_pos = pos_world[root_idx]
                target_center = qVector(root_pos[0], root_pos[1], root_pos[2])
                target_found = True
            
            # --- Rendering Lines (Bones) ---
            lines = []
            for i, parent_idx in enumerate(skeleton_hierarchy["parents"]):
                if parent_idx != -1:
                    lines.append(pos_world[i])
                    lines.append(pos_world[parent_idx])
                    
            if len(lines) > 0:
                line_data = np.array(lines)
                if skel_id not in self.skeleton_items:
                    skel_mesh = gl.GLLinePlotItem(pos=line_data, color=(1.0, 1.0, 1.0, 0.8), mode='lines', width=self.view_line_width)
                    self.pose_canvas.addItem(skel_mesh)
                    self.skeleton_items[skel_id] = skel_mesh
                else:
                    self.skeleton_items[skel_id].setData(pos=line_data)
                    
            # --- Rendering Points (Joints) ---
            if len(pos_world) > 0:
                if skel_id not in self.skeleton_points:
                    skel_pts = gl.GLScatterPlotItem(pos=pos_world, color=(1.0, 1.0, 1.0, 0.5), size=self.view_joint_size)
                    self.pose_canvas.addItem(skel_pts)
                    self.skeleton_points[skel_id] = skel_pts
                else:
                    self.skeleton_points[skel_id].setData(pos=pos_world)
            
            # --- OSC Sending (Original Array Format) ---
            if self.sender and getattr(self.sender, 'active', False):
                
                osc_pos_local = np.reshape(pose["pos_local"], (-1)).tolist()
                osc_rot_local = np.reshape(pose["rot_local"], (-1)).tolist()
                osc_pos_world = np.reshape(pose["pos_world"], (-1)).tolist()
                osc_rot_world = np.reshape(pose["rot_world"], (-1)).tolist()
                
                self.sender.send(f"/mocap/{skel_id}/joint/pos_local", osc_pos_local) 
                self.sender.send(f"/mocap/{skel_id}/joint/rot_local", osc_rot_local) 
                self.sender.send(f"/mocap/{skel_id}/joint/pos_world", osc_pos_world) 
                self.sender.send(f"/mocap/{skel_id}/joint/rot_world", osc_rot_world) 

        # --- View Center Updating (Fixed for manual panning) ---
        if not hasattr(self, 'last_selected_text'):
            self.last_selected_text = "None"
            self.pose_canvas.setCameraParams(center=qVector(0, 0, 0))

        if selected_text != "None" and target_found:
            # Constantly update tracking for the selected skeleton
            self.pose_canvas.setCameraParams(center=target_center)
        elif selected_text == "None" and self.last_selected_text != "None":
            # Only force back to the origin exactly when switching to "None"
            # This allows the user to manually control-drag afterwards
            self.pose_canvas.setCameraParams(center=qVector(0, 0, 0))

        self.last_selected_text = selected_text

        # Cleanup disappeared skeletons (Bones)
        for skel_id in list(self.skeleton_items.keys()):
            if skel_id not in active_ids:
                self.pose_canvas.removeItem(self.skeleton_items[skel_id])
                del self.skeleton_items[skel_id]
                
        # Cleanup disappeared skeletons (Joints)
        for skel_id in list(getattr(self, 'skeleton_points', {}).keys()):
            if skel_id not in active_ids:
                self.pose_canvas.removeItem(self.skeleton_points[skel_id])
                del self.skeleton_points[skel_id]

        self.update_gui_labels()
        
    def update_gui_labels(self):
        t = getattr(self.player, 'play_time', 0.0)
        start_t = getattr(self.player, 'start_time', 0.0)
        end_t = getattr(self.player, 'end_time', 0.0)
        max_t = getattr(self.player, 'max_time', 1.0) 
        
        self.q_time_label.setText(f"Time: {t:.2f}s | Loop: {start_t:.2f}s - {end_t:.2f}s")
        
        if max_t > 0:
            self.q_time_slider.blockSignals(True)
            self.q_time_slider.setValue(int((t / max_t) * 1000))
            self.q_time_slider.blockSignals(False)

            self.q_start_time_slider.blockSignals(True)
            self.q_start_time_slider.setValue(int((start_t / max_t) * 1000))
            self.q_start_time_slider.blockSignals(False)

            self.q_end_time_slider.blockSignals(True)
            self.q_end_time_slider.setValue(int((end_t / max_t) * 1000))
            self.q_end_time_slider.blockSignals(False)
        
    def change_play_time(self, val):
        max_t = getattr(self.player, 'max_time', 0.0)
        self.player.set_play_time((val / 1000.0) * max_t)
        self.update_gui_and_canvas() 

    def change_start_time(self, val):
        max_t = getattr(self.player, 'max_time', 0.0)
        self.player.set_start_time((val / 1000.0) * max_t)
        self.update_gui_and_canvas() 
            
    def change_end_time(self, val):
        max_t = getattr(self.player, 'max_time', 0.0)
        self.player.set_end_time((val / 1000.0) * max_t)
        self.update_gui_and_canvas() 

    def toggle_sender(self, widget):
        if self.sender:
            self.sender.set_active(widget.isChecked())

    def change_sender_address(self):
        if not self.sender:
            return
            
        for i in range(4):
            self.sender_ip[i] = self.q_sender_ip[i].value()
        self.sender_port = self.q_sender_port.value()
        
        sender_active = self.sender.get_active()
        sender_ip = f"{self.sender_ip[0]}.{self.sender_ip[1]}.{self.sender_ip[2]}.{self.sender_ip[3]}"
        
        if sender_active:
            self.sender.set_active(False)
            
        self.sender.set_address(sender_ip, self.sender_port)
            
        if sender_active:
            self.sender.set_active(True)