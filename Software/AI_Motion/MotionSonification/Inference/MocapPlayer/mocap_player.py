
import os, sys, time, subprocess

import motion_player 
import motion_sender
import motion_control
import motion_gui

"""
Setup Motion Player
"""

# Eleni (in Domain)
motion_player.config = { 
    "file_name": "../data/mocap/Eline_Session-002.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }

"""
# Eleni (out Domain)
motion_player.config = { 
    "file_name": "../data/mocap/Eline_Session-001.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }
"""

"""
# Tim (out Domain)
motion_player.config = { 
    "file_name": "../data/mocap/Tim_Session_002.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }
"""

"""
# Eleni (out of Domain)

motion_player.config = { 
    "file_name": "E:/Data/mocap/Eleni/Solos/CuteStudio_09.02.2026/xsens2osc_30hz/Eleni_kld0.1_evaluation.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }
"""

"""
# Diane (in Domain)
motion_player.config = { 
    "file_name": "E:/Data/mocap/Diane/Solos/ZHdK_10.10.2025/xsens2osc_30hz/trial-002.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }
"""

"""
# Diane (out of Domain)

motion_player.config = { 
    "file_name": "E:/Data/mocap/Diane/Solos/ZHdK_06.02.2026/xsens2osc_30hz/Diane_kld0.1_jointdrop.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }

# Tim (in Domain)
motion_player.config = { 
    "file_name": "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/xsens2osc_30hz/trial-002.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }

# Tim (out of Domain)
motion_player.config = { 
    "file_name": "E:/Data/mocap/Tim/Solos/ZHdK_07.02.2026/xsens2osc_30hz/Tim_kl1.0_dectrain_evaluation.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }


"""

"""
motion_player.config = { 
    "file_name": "E:/Data/mocap/Eleni/Solos/ZHdK_04.12.2025/xsens2osc_30hz/Eline_Session-002.pkl",
    "root_joint_index": 0,
    "pos_scale": 100.0,
    "parents" : [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21],
    "children": [[1, 5, 9], [2], [3], [4], [], [6], [7], [8], [], [10], [11], [12], [13, 17, 21], [14], [15], [16], [], [18], [19], [20], [], [22], []],
    "fps": 30
    }
"""

player = motion_player.MotionPlayer(motion_player.config)


"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9007

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Setup Motion GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["player"] = player
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
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