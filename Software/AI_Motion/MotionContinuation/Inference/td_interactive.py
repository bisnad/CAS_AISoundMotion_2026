# -------------------------------------------------------------------------------------------------
# Motion Continuation Model - Inference Script
# Employs a Transformer Decoder MDN Architecture
# Can be trained on 6D joint rotation representations and optionally on the root joint trajectory
# -------------------------------------------------------------------------------------------------

import motion_model
import motion_synthesis
import motion_sender
import motion_receiver
import motion_gui
import motion_control
from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

import torch
import os, sys, json, copy
import numpy as np

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

"""
# Example: FBX
mocap_file_path = "data/mocap/"
mocap_files = ["Muriel_Embodied_Machine.fbx.fbx"]
mocap_topology_files = [None for _ in mocap_files]
mocap_valid_time_ranges = [None for _ in mocap_files]
mocap_pos_scale = 1.0
mocap_fps = 50
"""

# Example: NPZ
mocap_file_path = "data/mocap/"
mocap_files = ["MadelineHarms_Improvisation.npz"]
mocap_valid_time_ranges = [ [ [ 10.0, 884.0 ] ] ] # in seconds
mocap_topology_files = [ "data/configs/Mediapipe_config.json" ] # only used for .npz files
mocap_pos_scale = 100.0
mocap_fps = 25

train_root_trajectory = False

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

mdn_num_mixtures = 20
decoder_layer_count = 6
decoder_head_count = 8
decoder_embed_dim = 512
decoder_ff_dim = 2048
decoder_dropout = 0.1
sequence_length = 64

decoder_weights_file = "data/results/weights/decoder_weights_epoch_400.pth"

gaussian_temp = 0.2
pi_temperature = 1.5
top_p = 0.9
motion_smooth_cutoff = 5.0
motion_smooth_beta = 0.01

osc_send_ip = "127.0.0.1"
osc_send_port = 9004
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9006
osc_live_receive_port = 9007

# -------------------------------------------------------------------------------------------------
# Load Mocap Data
# -------------------------------------------------------------------------------------------------
bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
npz_tools = npz.NPZ_Tools()
mocap_tools = mocap.Mocap_Tools()

def load_mocap_file_unified(mocap_file, topology_file, valid_time_ranges, mocap_file_path, mocap_fps):
    mocap_abs_path = os.path.join(mocap_file_path, mocap_file)
    ext = os.path.splitext(mocap_file)[1].lower()

    if ext == ".bvh":
        bvh_data = bvh_tools.load(mocap_abs_path)
        mocap_data_raw = mocap_tools.bvh_to_mocap(bvh_data)
        segments = mocap_tools.resample_euler_mocap_data(mocap_data_raw, mocap_fps, valid_time_ranges)
        for seg in segments:
            seg["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(seg["motion"]["rot_local_euler"], seg["rot_sequence"])
            del seg["motion"]["rot_local_euler"]
        return segments

    if ext == ".fbx":
        fbx_data = fbx_tools.load(mocap_abs_path)
        mocap_data_raw = mocap_tools.fbx_to_mocap(fbx_data)[0]
        segments = mocap_tools.resample_euler_mocap_data(mocap_data_raw, mocap_fps, valid_time_ranges)
        for seg in segments:
            seg["motion"]["rot_local"] = mocap_tools.euler_to_quat(seg["motion"]["rot_local_euler"], seg["rot_sequence"])
            del seg["motion"]["rot_local_euler"]
        return segments

    if ext == ".npz":
        topology_file = mocap_topology_files[i]
        if topology_file is None:
            raise ValueError(
                f"NPZ file '{mocap_file}' requires a topology JSON file path in mocap_topology_files"
            )

        npz_data, topo_data = npz_tools.load(mocap_abs_path, topology_file)
        segments = mocap_tools.npz_to_mocap(
            npz_data, 
            topo_data, 
            mocap_fps)
        return segments

    raise ValueError(f"Unsupported mocap file format: {mocap_file}")

def skeleton_signature(skel):
    return (
        tuple(skel["parents"]),
        tuple(tuple(c) for c in skel["children"]),
        len(skel["joints"])
    )

all_mocap_data = []
reference_skeleton_signature = None
reference_rot_sequence = None

for i, mocap_file in enumerate(mocap_files):
    print("process file ", mocap_file)
    segments = load_mocap_file_unified(
        mocap_file=mocap_file, topology_file=mocap_topology_files[i],
        valid_time_ranges=mocap_valid_time_ranges[i], mocap_file_path=mocap_file_path, mocap_fps=mocap_fps
    )

    for mocap_data in segments:
        mocap_data["skeleton"]["offsets"] = mocap_data["skeleton"]["offsets"].astype(np.float32) * mocap_pos_scale
        mocap_data["motion"]["pos_local"] = mocap_data["motion"]["pos_local"].astype(np.float32) * mocap_pos_scale
        mocap_data["motion"]["rot_local"] = rot_np.normalize(mocap_data["motion"]["rot_local"].astype(np.float32))

        if not train_root_trajectory:
            mocap_data["skeleton"]["offsets"][0, 0] = 0.0
            mocap_data["skeleton"]["offsets"][0, 2] = 0.0
            mocap_data["motion"]["pos_local"][:, 0, 0] = 0.0
            mocap_data["motion"]["pos_local"][:, 0, 2] = 0.0

        sig = skeleton_signature(mocap_data["skeleton"])
        if reference_skeleton_signature is None:
            reference_skeleton_signature = sig
            reference_rot_sequence = mocap_data.get("rot_sequence", [0, 1, 2])
        elif sig != reference_skeleton_signature:
            raise ValueError(f"Incompatible skeleton in '{mocap_file}'. All clips must share the same hierarchy.")

        all_mocap_data.append(mocap_data)

all_pose_sequences = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence_6d = rot_np.quat_to_r6d(pose_sequence)
    joint_count = pose_sequence_6d.shape[1]
    joint_dim = 6
    pose_dim = joint_count * joint_dim
    pose_sequence_6d_flat = np.reshape(pose_sequence_6d, (-1, pose_dim))

    if train_root_trajectory:
        root_positions = mocap_data["motion"]["pos_local"][:, 0, :]
        pose_sequence_input = np.concatenate((root_positions, pose_sequence_6d_flat), axis=1)
    else:
        pose_sequence_input = pose_sequence_6d_flat

    all_pose_sequences.append(pose_sequence_input.astype(np.float32))

input_dim = all_pose_sequences[0].shape[1]

all_excerpts = np.concatenate(all_pose_sequences, axis=0)
if train_root_trajectory:
    root_pos_mean = np.mean(all_excerpts[:, :3], axis=0, keepdims=True)
    root_pos_std = np.std(all_excerpts[:, :3], axis=0, keepdims=True)
    root_pos_std[root_pos_std == 0] = 1.0

# -------------------------------------------------------------------------------------------------
# Setup Model
# -------------------------------------------------------------------------------------------------
motion_model.config["input_dim"] = input_dim
motion_model.config["embed_dim"] = decoder_embed_dim
motion_model.config["num_heads"] = decoder_head_count
motion_model.config["num_layers"] = decoder_layer_count
motion_model.config["ff_dim"] = decoder_ff_dim
motion_model.config["dropout"] = decoder_dropout
motion_model.config["pos_encoding_max_length"] = sequence_length
motion_model.config["num_mixtures"] = mdn_num_mixtures
motion_model.config["device"] = device
motion_model.config["weights_path"] = decoder_weights_file

model = motion_model.createModel(motion_model.config)

# -------------------------------------------------------------------------------------------------
# Setup Motion Synthesis
# -------------------------------------------------------------------------------------------------
synthesis_config = motion_synthesis.config
synthesis_config["skeleton"] = all_mocap_data[0]["skeleton"]
synthesis_config["rot_sequence"] = reference_rot_sequence
synthesis_config["source_mocap_format"] = os.path.splitext(mocap_files[0])[1].lower().replace(".", "")
synthesis_config["model"] = model
synthesis_config["seq_length"] = sequence_length
synthesis_config["orig_sequences"] = all_pose_sequences
synthesis_config["orig_seq_index"] = 0
synthesis_config["use_live_motion"] = True
synthesis_config["device"] = device
synthesis_config["train_root_trajectory"] = train_root_trajectory
synthesis_config["root_pos_mean"] = root_pos_mean if train_root_trajectory else None
synthesis_config["root_pos_std"] = root_pos_std if train_root_trajectory else None
synthesis_config["joint_count"] = joint_count
synthesis_config["mocap_fps"] = mocap_fps
synthesis_config["smooth_cutoff"] = motion_smooth_cutoff
synthesis_config["smooth_beta"] = motion_smooth_beta
synthesis_config["gaussian_temp"] = gaussian_temp
synthesis_config["pi_temperature"] = pi_temperature
synthesis_config["top_p"] = top_p

synthesis = motion_synthesis.MotionSynthesis(synthesis_config)

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
motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender
motion_gui.config["update_interval"] = 1.0 / mocap_fps
motion_gui.config["osc_ip"] = osc_send_ip
motion_gui.config["osc_port"] = osc_send_port

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent)

# -------------------------------------------------------------------------------------------------
# Setup OSC Control
# -------------------------------------------------------------------------------------------------
motion_control.config["motion_seq"] = pose_sequence_input
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

# -------------------------------------------------------------------------------------------------
# Setup OSC Receiver (Live Motion)
# -------------------------------------------------------------------------------------------------

motion_receiver_config = {
    "ip": osc_receive_ip,
    "port": osc_live_receive_port,
    "synthesis": synthesis
}

osc_receiver = motion_receiver.OscReceiver(motion_receiver_config)

# -------------------------------------------------------------------------------------------------
# Start Application
# -------------------------------------------------------------------------------------------------

osc_receiver.start()
osc_control.start()
gui.show()
app.exec_()
osc_control.stop()
osc_receiver.stop()