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


import motion_player

config = {"player": None,
          "sender": None,
          "view_min": np.array([-100, -100, -100], dtype=np.float32),
          "view_max": np.array([100, 100, 100], dtype=np.float32),
          "view_center": np.array([0, 0, 100], dtype=np.float32),
          "view_ele": 90,
          "view_azi": -90,
          "view_dist": 250,
          "view_line_width": 2.0
    }

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.player = config["player"]
        self.sender = config["sender"]
        
        fps = int(self.player.get_fps())
        self.pose_thread_interval = 1.0 / fps
        self.pose_thread_event = None
    
        # header
        file_name = self.player.get_file_name()
        play_frame = self.player.get_play_frame()
        start_play_frame = self.player.get_start_play_frame()
        end_play_frame = self.player.get_end_play_frame()
        
        self.q_file_label = QtWidgets.QLabel(Path(file_name).stem)
        self.q_frame_label = QtWidgets.QLabel(f"sequence start frame {start_play_frame} end frame {end_play_frame} play frame {play_frame}", self)

        # pose canvas
        self._create_edges()

        
        self.view_min = config["view_min"]
        self.view_max = config["view_max"]   
        view_center = config["view_center"]
        self.view_center = qVector(view_center[0], view_center[1], view_center[2])
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_line_width = config["view_line_width"]
        
        self.pose_canvas = gl.GLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)
        self.pose_canvas.setCameraParams(center=self.view_center)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)
        
        self.q_load_buttom = QtWidgets.QPushButton("load", self)
        self.q_load_buttom.clicked.connect(self.choose_file)  

        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  
        
        self.q_fps = QtWidgets.QSpinBox(self)
        self.q_fps.setMinimum(0)
        self.q_fps.setMaximum(200)
        self.q_fps.setValue(fps)
        self.q_fps.valueChanged.connect(self.change_fps)  
        
        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_load_buttom,0,0)
        self.q_button_grid.addWidget(self.q_start_buttom,0,1)
        self.q_button_grid.addWidget(self.q_stop_buttom,0,2)
        self.q_button_grid.addWidget(self.q_fps,0,3)
        
        self.q_frame_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_frame_slider.setMinimum(0)
        self.q_frame_slider.setMaximum(end_play_frame)
        self.q_frame_slider.setValue(play_frame)
        self.q_frame_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.q_frame_slider.setTickInterval(1000)
        self.q_frame_slider.sliderMoved.connect(self.change_play_frame)  
        
        self.q_start_frame_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_start_frame_slider.setMinimum(0)
        self.q_start_frame_slider.setMaximum(end_play_frame)
        self.q_start_frame_slider.setValue(start_play_frame)
        self.q_start_frame_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.q_start_frame_slider.setTickInterval(1000)
        self.q_start_frame_slider.sliderMoved.connect(self.change_start_play_frame)  
        
        self.q_end_frame_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.q_end_frame_slider.setMinimum(0)
        self.q_end_frame_slider.setMaximum(end_play_frame)
        self.q_end_frame_slider.setValue(end_play_frame)
        self.q_end_frame_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.q_end_frame_slider.setTickInterval(1000)
        self.q_end_frame_slider.sliderMoved.connect(self.change_end_play_frame)  
        
        
        # osc sender
        self.sender_active = True
        
        self.q_sender_toggle = QtWidgets.QCheckBox("osc", self)
        self.q_sender_toggle.stateChanged.connect(lambda:self.toggle_sender(self.q_sender_toggle))  
        self.q_sender_toggle.setChecked(self.sender_active)
          
        sender_ip_string, sender_port = self.sender.get_address()
        sender_ip_list = sender_ip_string.split(".")
        sender_ip = [ int(el) for el in sender_ip_list ]
        
        self.sender_ip = sender_ip
        self.sender_port = sender_port

        self.q_sender_ip = []
        for i in range(4):
            _w = QtWidgets.QSpinBox(self)
            _w.setMinimum(0)
            _w.setMaximum(255)
            _w.setValue(self.sender_ip[i])
            _w.valueChanged.connect(lambda:self.change_sender_ip(_w))  

            self.q_sender_ip.append(_w)
        self.q_sender_port = QtWidgets.QSpinBox(self)
        self.q_sender_port.setMinimum(0)
        self.q_sender_port.setMaximum(65535)
        self.q_sender_port.setValue(self.sender_port)
        self.q_sender_port.valueChanged.connect(lambda:self.change_sender_port(self.q_sender_port))  
        
        self.q_sender_grid = QtWidgets.QGridLayout()
        self.q_sender_grid.addWidget(self.q_sender_toggle,0,0)
        self.q_sender_grid.addWidget(self.q_sender_ip[0],0,1)
        self.q_sender_grid.addWidget(self.q_sender_ip[1],0,2)
        self.q_sender_grid.addWidget(self.q_sender_ip[2],0,3)
        self.q_sender_grid.addWidget(self.q_sender_ip[3],0,4)
        self.q_sender_grid.addWidget(self.q_sender_port,0,5)
    

        # final layout

        self.q_grid = QtWidgets.QGridLayout()
        
        self.q_grid.addWidget(self.q_file_label,0,0)
        self.q_grid.addWidget(self.q_frame_label,1,0)
        self.q_grid.addWidget(self.pose_canvas,2,0)
        self.q_grid.addWidget(self.q_frame_slider,3,0)
        self.q_grid.addWidget(self.q_start_frame_slider,4,0)
        self.q_grid.addWidget(self.q_end_frame_slider,5,0)
        self.q_grid.addLayout(self.q_button_grid,6,0)
        self.q_grid.addLayout(self.q_sender_grid,7,0)

        
        self.q_grid.setRowStretch(0, 0)
        self.q_grid.setRowStretch(1, 0)
        self.q_grid.setRowStretch(2, 1)
        self.q_grid.setRowStretch(3, 0)
        self.q_grid.setRowStretch(4, 0)
        self.q_grid.setRowStretch(5, 0)
        self.q_grid.setRowStretch(6, 0)
        self.q_grid.setRowStretch(7, 0)
        
        self.setLayout(self.q_grid)

        self.setGeometry(50,50,512,612)
        self.setWindowTitle("Mocap Player")
        
        self.gui_needs_repaint = False

        # Signals that can be emitted 
        self.poseCanvasUpdater = PoseCanvasUpdater()
        # Update graph whenever the 'request_graph_update' signal is emitted 
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)
        
    def _create_edges(self):
        
        skeleton = self.player.get_skeleton()
        
        # create edge list
        children = skeleton["children"]
        self.edges = []
        for parent_joint_index in range(len(children)):
            for child_joint_index in children[parent_joint_index]:
                self.edges.append([parent_joint_index, child_joint_index])
                
    def choose_file(self):
        
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'mocap',"Mocap Files (*.bvh *.fbx)")
        
        file_name = file_name[0]

        if len(file_name) == 0:
            return
        
        self.load_file(file_name)
        
    def load_file(self, file_name):
        
        #print("load_file: ", file_name)
        
        sender_active = self.sender.get_active()
        
        if sender_active == True:
            self.sender.set_active(False)
            
        self.q_file_label.setText("Loading...")
        self.q_file_label.update()
        
        #self.q_file_label.repaint()
        
        self.player.load(file_name)
        self.pose_thread_interval = 1.0 / self.player.get_fps() 
        self._create_edges()
        
        self.q_file_label.setText(Path(file_name).stem)
        self.update_gui()

        fps = int(self.player.get_fps())
        play_frame = self.player.get_play_frame()
        start_play_frame = self.player.get_start_play_frame()
        end_play_frame = self.player.get_end_play_frame()
        
        #self.q_frame_label.setText(f"sequence start frame {start_play_frame} end frame {end_play_frame} play frame {play_frame}")

        self.q_fps.setValue(fps)
        self.q_frame_slider.setMaximum(end_play_frame)
        self.q_start_frame_slider.setMaximum(end_play_frame)
        self.q_end_frame_slider.setMaximum(end_play_frame)
        self.q_end_frame_slider.setValue(end_play_frame)
        self.q_start_frame_slider.setValue(start_play_frame)
        
        self.q_frame_slider.update()
        #self.q_frame_slider.repaint()
        self.q_start_frame_slider.update()
        #self.q_start_frame_slider.repaint()
        self.q_end_frame_slider.update()
        #self.q_end_frame_slider.repaint()
        
        if sender_active == True:
            self.sender.set_active(True)

        self.gui_needs_repaint = True
        
    def start(self):

        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        
        self.pose_thread.start()
        
    def stop(self):
        
        if self.pose_thread_event is None:
            return

        self.pose_thread_event.set()
        self.pose_thread.join()
        
    def change_fps(self, fps):
        
        fps = int(fps)
        
        self.player.set_fps(fps)
        
        fps = self.player.get_fps()
        
        self.q_fps.setValue(fps)
        
        self.pose_thread_interval = 1.0 / fps
        
    def update(self):
        
        while self.pose_thread_event.is_set() == False:
 
            start_time = time.time()            
            
            if self.gui_needs_repaint == True:
                self.q_file_label.repaint()
                self.q_frame_slider.repaint()
                self.q_start_frame_slider.repaint()
                self.q_end_frame_slider.repaint()
                
                self.gui_needs_repaint = False

            self.update_gui()
            self.update_player()
            
            self.poseCanvasUpdater.request_canvas_update.emit()  

            self.update_osc()
            
            end_time = time.time()   
            
            #print("update time ", end_time - start_time, " interval ", self.pose_thread_interval)
            
            next_update_interval = max(self.pose_thread_interval - (end_time - start_time), 0.0)
            
            sleep(self.pose_thread_interval)

    def update_gui(self):
        
        play_frame = self.player.get_play_frame()
        start_play_frame = self.player.get_start_play_frame()
        end_play_frame = self.player.get_end_play_frame()
        
        self.q_frame_label.setText(f"sequence start frame {start_play_frame} end frame {end_play_frame} play frame {play_frame}")
        self.q_frame_label.update()
        self.q_frame_slider.setValue(play_frame)
            
    def update_player(self):
        
        self.player.update()       
        self.player_pose_wpos =  self.player.get_pose("pos_world")
        self.player_pose_wrot = self.player.get_pose("rot_world")
        self.player_pose_lpos =  self.player.get_pose("pos_local")
        self.player_pose_lrot = self.player.get_pose("rot_local")
        
        #print(self.player_pose_lrot)
        
    def update_osc(self):
        
        osc_values = np.reshape(self.player_pose_lpos, (-1)).tolist()
        self.sender.send("/mocap/0/joint/pos_local", osc_values) 
        
        osc_values = np.reshape(self.player_pose_lrot, (-1)).tolist()
        self.sender.send("/mocap/0/joint/rot_local", osc_values) 
            
        osc_values = np.reshape(self.player_pose_wpos, (-1)).tolist()
        self.sender.send("/mocap/0/joint/pos_world", osc_values) 
            
        osc_values = np.reshape(self.player_pose_wrot, (-1)).tolist()
        self.sender.send("/mocap/0/joint/rot_world", osc_values) 


    def update_pose_plot(self):
        
        pose = self.player_pose_wpos

        points_data = pose
        lines_data = pose[np.array(self.edges).flatten()]

        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        self.pose_canvas_points.setData(pos=pose, color=(1.0, 1.0, 1.0, 0.5))
        
    def change_play_frame(self, frame):
        self.player.set_play_frame(frame)
        
    def change_start_play_frame(self, frame):
        
        self.player.set_start_play_frame(frame)
        start_play_frame = self.player.get_start_play_frame()
        self.q_start_frame_slider.setValue(start_play_frame)
            
    def change_end_play_frame(self, frame):
        
        self.player.set_end_play_frame(frame)
        end_play_frame = self.player.get_end_play_frame()
        self.q_end_frame_slider.setValue(end_play_frame)

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
        
        
