"""
Things that are unclear at the moment:
    
How the travel distance is calculated (especially the mean, std, max, min), at the moment, I calculate only the mean
Same issue applies to the area covered

"""


"""
imports
"""

import analysis as ma
import motion_receiver
import motion_sender
import motion_pipeline
import motion_gui

import json
import sys

from matplotlib import pyplot as plt
import numpy as np

"""
Mocap Settings
"""

mocap_config_path = "configs/xsens_fbx_config.json"
#mocap_config_path = "configs/zed34_fbx_config.json"

# load mocap config
with open(mocap_config_path) as json_data:
    mocap_config = json.load(json_data)

"""
OSC Receiver
"""

input_pos_data = np.zeros((len(mocap_config["joint_names"]), mocap_config["pos_dim"]), dtype=np.float32)

motion_receiver.config["ip"] = "0.0.0.0"
motion_receiver.config["port"] = 9007
motion_receiver.config["data"] = [ input_pos_data ]
motion_receiver.config["messages"] = ["/mocap/*/joint/pos_world"]

osc_receiver = motion_receiver.MotionReceiver(motion_receiver.config)

"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9008

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Data Pipeline
"""

pipeline = motion_pipeline.MotionPipeline(osc_receiver, mocap_config)

"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["pipeline"] = pipeline
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)


"""
Start Application
"""

osc_receiver.start()
gui.show()
app.exec_()

osc_receiver.stop()


