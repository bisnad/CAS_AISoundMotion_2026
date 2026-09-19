"""
Imports
"""

import os, sys, time, subprocess
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import motion_player 
import motion_sender
import motion_control
import motion_gui

"""
Setup Motion Player
"""

"""
motion_player.config["file_name"] = "data/mocap/Muriel_Take4_Blumen_Baile.fbx"
motion_player.config["fps"] = 50
"""

motion_player.config = { 
    "file_name": "../../../../CAS_AISoundMotion_Data_2026/Mocap/Stocos/Duets/XSens_npz_60fps/Jason_Muriel_Take1.npz",
    "topology_file_name": "data/configs/XSensLive_config.json",
    "fps": 60
    }

"""
motion_player.config = { 
    "file_name": "../../../../CAS_AISoundMotion_Data_2026/Mocap/Stocos/Solos/Mediapipe_npz_50fps/Muriel_Blumen_Baile.npz",
    "topology_file_name": "data/configs/Mediapipe_config.json",
    "fps": 50
    }
"""

"""
motion_player.config = { 
    "file_name": "../../../../CAS_AISoundMotion_Data_2026/Mocap/Stocos/Duets/Yolo_npz_50fps/Jason_Muriel_Take1.npz",
    "topology_file_name": "data/configs/Human36M_config.json",
    "fps": 50
    }
"""

player = motion_player.MotionPlayer(motion_player.config)

"""
Setup OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9007

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Setup Motion GUI
"""

motion_gui.config["player"] = player
motion_gui.config["sender"] = osc_sender
motion_gui.config["view_scale"] = 1.0

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
Setup OSC Receiver
"""

motion_control.config["gui"] = gui
motion_control.config["ip"] = "0.0.0.0"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()