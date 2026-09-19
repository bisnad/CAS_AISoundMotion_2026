# -------------------------------------------------------------------------------------------------
# Motion Transformation - Inference Script for Direct Latent Dimension Manipulation
# Employs a Convolutional Variational Beta-Autoencoder
# Can be trained on 6D joint rotation representations and optionally on the root joint trajectory
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control
from common.rotation_utils_numpy import  RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import  RotationUtilsTorch as rot_to

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg

import os, sys, time, subprocess
import numpy as np
import math
import pickle

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap

# -------------------------------------------------------------------------------------------------
# Compute Unit
# -------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device} device")

# -------------------------------------------------------------------------------------------------
# Mocap Settings
# -------------------------------------------------------------------------------------------------

# Example 1: FBX
mocap_file_path = "data/mocap/"
mocap_files = ["Muriel_Blumen_Baile.fbx", "Muriel_RO_37-4-1.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_root_trajectory = False

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

vae_latent_dim = 16
vae_conv_channel_counts = [128, 128, 128]
vae_conv_kernel_sizes = [3, 3, 3, 4]
vae_conv_strides = [2, 2, 2, 2]
vae_conv_dilations = [1, 2, 4, 1]
vae_window_length = 64

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

#vae_weights_file = "data/results/weights/vae_weight_epoch_200.pt"
#vae_weights_file = "../vae_cnn_v2/results_Muriel_Take1_double_Bind_fbx/weights/vae_weight_epoch_200.pt"
#vae_weights_file = "../vae_cnn_v2/results_Muriel_Take1_double_Bind_bvh/weights/vae_weight_epoch_200.pt"
vae_weights_file = "../vae_cnn_v2/results_Yurika_MotionClasses_Mediapipe_npz/weights/vae_weight_epoch_200.pt"

# -------------------------------------------------------------------------------------------------
# OSC Settings
# -------------------------------------------------------------------------------------------------

osc_send_ip = "127.0.0.1"
osc_send_port = 9004

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9005

# -------------------------------------------------------------------------------------------------
# Load Mocap Data
# -------------------------------------------------------------------------------------------------

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
npz_tools = npz.NPZ_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file, mocap_topology_file in zip(mocap_files, mocap_topology_files):

    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path + mocap_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path + mocap_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] 
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".npz") or mocap_file.endswith(".NPZ"):
        npz_data, topo_data = npz_tools.load(mocap_file_path + mocap_file, mocap_topology_file)
        mocap_data = mocap_tools.npz_to_mocap(npz_data, topo_data, mocap_fps)[0]

    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale

    if not mocap_root_trajectory:
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 0] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 2] = 0.0

    mocap_data["motion"]["rot_local"] = rot_np.quat_to_r6d(mocap_data["motion"]["rot_local"])
    all_mocap_data.append(mocap_data)

skeleton = all_mocap_data[0]["skeleton"]
joint_count = all_mocap_data[0]["motion"]["rot_local"].shape[1]
joint_dim = all_mocap_data[0]["motion"]["rot_local"].shape[2]
pose_dim = joint_count * joint_dim
input_dim = pose_dim

all_pose_sequences = []

for mocap_data in all_mocap_data:
    
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence = np.reshape(pose_sequence, (-1, joint_count * joint_dim))
    
    if mocap_root_trajectory:
        root_positions = mocap_data["motion"]["pos_local"][:, 0, :]
        pose_sequence = np.concatenate((root_positions, pose_sequence), axis=1)
    
    all_pose_sequences.append(pose_sequence)

pose_dim = all_pose_sequences[0].shape[1]

# Initialize to None for the no-root-trajectory case
root_pos_mean = None
root_pos_std = None

if mocap_root_trajectory:
    # 1. Extract the root trajectory
    root_sequence = pose_sequence[:, :3]
    
    # 2. Add extra dimensions to match the (1, 1, 3) format used in the decoder
    root_pos_mean = np.mean(root_sequence, axis=0, keepdims=True)
    root_pos_mean = np.expand_dims(root_pos_mean, axis=0) # Shape becomes (1, 1, 3)
    
    root_pos_std = np.std(root_sequence, axis=0, keepdims=True)
    root_pos_std = np.expand_dims(root_pos_std, axis=0)   # Shape becomes (1, 1, 3)
    
    # 3. Handle zero variance (e.g., stationary sequences) to prevent division by zero or collapse
    root_pos_std[root_pos_std == 0.0] = 1.0

# -------------------------------------------------------------------------------------------------
# Setup Model
# -------------------------------------------------------------------------------------------------

motion_model.config["vae_input_dim"] = pose_dim
motion_model.config["vae_latent_dim"] = vae_latent_dim
motion_model.config["vae_conv_channel_counts"] = vae_conv_channel_counts
motion_model.config["vae_conv_kernel_sizes"] = vae_conv_kernel_sizes
motion_model.config["vae_conv_strides"] = vae_conv_strides
motion_model.config["vae_conv_dilations"] = vae_conv_dilations
motion_model.config["vae_window_length"] = vae_window_length
motion_model.config["device"] = device
motion_model.config["vae_weights_path"] = vae_weights_file

vae = motion_model.createModels(motion_model.config) 

# -------------------------------------------------------------------------------------------------
# Setup Motion Synthesis
# -------------------------------------------------------------------------------------------------

motion_synthesis.config["skeleton"] = skeleton
motion_synthesis.config["model"] = vae
motion_synthesis.config["device"] = device
motion_synthesis.config["seq_window_length"] = vae_window_length
motion_synthesis.config["seq_window_offset"] = 1
motion_synthesis.config["root_trajectory"] = mocap_root_trajectory
motion_synthesis.config["root_pos_mean"] = root_pos_mean
motion_synthesis.config["root_pos_std"] = root_pos_std
motion_synthesis.config["mocap_fps"] = mocap_fps
motion_synthesis.config["orig_sequences"] = all_pose_sequences
motion_synthesis.config["orig_seq1_index"] = 0
motion_synthesis.config["orig_seq2_index"] = 0
motion_synthesis.config["use_live_seq1"] = True
motion_synthesis.config["use_live_seq2"] = False

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

# -------------------------------------------------------------------------------------------------
# Setup OSC Sender
# -------------------------------------------------------------------------------------------------

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)

# -------------------------------------------------------------------------------------------------
# Setup GUI
# -------------------------------------------------------------------------------------------------

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender
motion_gui.config["osc_ip"] = osc_send_ip
motion_gui.config["osc_port"] = osc_send_port

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

# -------------------------------------------------------------------------------------------------
# Setup OSC Control
# -------------------------------------------------------------------------------------------------

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["latent_dim"] = vae_latent_dim
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

# -------------------------------------------------------------------------------------------------
# Start Application
# -------------------------------------------------------------------------------------------------

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
