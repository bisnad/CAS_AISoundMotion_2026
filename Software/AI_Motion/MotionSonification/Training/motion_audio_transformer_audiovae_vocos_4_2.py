"""
same as motion_audio_transformer_vae_vocos_4.py but with an error fixed in the PositionalEncoding class that mistakingly assumed a time first instead of batch first approach
"""

"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa
import auraloss

# vocos specific imports
from vocos import Vocos 

# mocap specific imports

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, q_conj_np, qmul_np, slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

"""
# Diane Version

mocap_data_path = "E:/data/mocap/Diane/Solos/ZHdK_10.10.2025/xsens2osc_30hz/"
mocap_data_files = ["trial-002.pkl"]
mocap_valid_ranges = [[181, 11368]]


# Eleni Version

mocap_data_path = "E:/data/mocap/Eleni/Solos/ZHdK_04.12.2025/xsens2osc_30hz/"
mocap_data_files = ["Eline_Session-002.pkl"]
mocap_valid_ranges = [[389, 25080]]


# Tim Version

mocap_data_path = "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/xsens2osc_30hz/"
mocap_data_files = ["trial-002.pkl"]
mocap_valid_ranges = [[654, 12492]]

"""

mocap_data_path = "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/xsens2osc_30hz/"
mocap_data_files = ["trial-002.pkl"]
mocap_valid_ranges = [[654, 12492]]

mocap_joint_count = 23 # will be overwritten for fbx and bvh files
mocap_joint_dim = 4 # will be overwritten for fbx and bvh files
mocap_root_joint_index = 0 # will be overwritten for fbx and bvh files
mocap_pose_dim = mocap_joint_count * mocap_joint_dim

mocap_pos_scale = 0.1
mocap_fps = 30 # 60
mocap_dim = -1 # automatically calculated
mocap_input_seq_length = 60 # (56 at 60 fps, 60 at 30 or 50 fps)
mocap_output_seq_length = 10 # for non teacher forcing
mocap_input_output_seq_length = mocap_input_seq_length + mocap_output_seq_length
load_mocap_stat = False
mocap_mean_file = "results/stat/mocap_mean.pt"
mocap_std_file = "results/stat/mocap_std.pt"

"""
Audio Settings
"""

"""
# Diane Version

audio_data_path = "E:/data/audio/Diane/48khz/"
audio_data_files = ["4d69949b.wav"]
audio_valid_ranges = [[5.0, 377.91]]

# Eleni Version

audio_data_path = "E:/data/audio/Eleni/48khz/"
audio_data_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]

# Tim Version

audio_data_path = "E:/data/audio/Tim/48khz/"
audio_data_files = ["SajuHariPlacePrizeEntry2010.wav"]
audio_valid_ranges = [[0.0, 394.06]]

"""

audio_data_path = "E:/data/audio/Tim/48khz/"
audio_data_files = ["SajuHariPlacePrizeEntry2010.wav"]
audio_valid_ranges = [[0.0, 394.06]]

audio_sample_rate = 48000
audio_channels = 1
audio_dim = -1 # automatically calculated
audio_waveform_input_seq_length = int(audio_sample_rate / mocap_fps * mocap_input_seq_length)
audio_samples_per_mocap_frame = audio_sample_rate // mocap_fps

load_audio_latents_stat = False
audio_latents_mean_file = "results/stat/latents_mean.pt"
audio_latents_std_file = "results/stat/latents_std.pt"

"""
Vocos Settings
"""

vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"

"""
Vocos VAE Model Settings
"""

latent_dim = 32 # 128
audio_dim = latent_dim
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]
audio_mel_count_vae = 8 
audio_latents_input_sequence_length = None

"""
# Diane Versions

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld128_kld1.0/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld128_kld1.0/weights/decoder_weights_epoch_400"

# Eleni Versions

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld128_kld1.0/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld128_kld1.0/weights/decoder_weights_epoch_400"

# Tim Versions

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld128_kld1.0/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld128_kld1.0/weights/decoder_weights_epoch_400"

"""

encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"

"""
Dataset Settings
"""

mocap_frame_incr = 4
batch_size = 128 # 32
test_percentage = 0.1

"""
Model Settings
"""

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1   
#pos_encoding_max_length = max(mocap_input_seq_length, audio_latents_input_seq_length)
pos_encoding_max_length = mocap_input_seq_length # in this implementation, the mocap_input_seq_length is always longer than the audio_latents_input_seq_length

"""
Training Settings
"""

learning_rate = 1e-4
non_teacher_forcing_step_count = 10
model_save_interval = 50
load_weights = False
save_weights = True

"""
# Diane Version

transformer_load_weights_path = "results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

# Eleni Version

transformer_load_weights_path = "results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

# Tim Version

transformer_load_weights_path = "results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

"""

transformer_load_weights_path = "results_Eleni_Take2_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

epochs = 200

"""
Mocap Visualisation Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Load Data - Mocap
"""

# load mocap data
bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        pkl_data = pickle.load(f)
        
    return pkl_data

def pkl_to_mocap(pkl_data):
    
    mocap_data = {}
    mocap_data["motion"] = {}
    
    # unique osc message addresses
    sensor_ids = pkl_data["sensor_ids"]
    sensor_values = pkl_data["sensor_values"]
    
    # get unique sensor ids
    unique_sensor_ids = list(set(sensor_ids))
    #print("unique_sensor_ids ", unique_sensor_ids)
    
    # get rid of all unique sensor ids that don't correspond to skeleton 0
    unique_sensor_ids = [ unique_sensor_id for unique_sensor_id in unique_sensor_ids if "/mocap/0" in unique_sensor_id ]
    #print("unique_sensor_ids ", unique_sensor_ids)
    
    # get indices of unique sensor ids into sensor values list
    unique_sensor_indices = {}
    for unique_sensor_id in unique_sensor_ids:
        unique_indices = [i for i, s in enumerate(sensor_ids) if s == unique_sensor_id]
        unique_sensor_indices[unique_sensor_id] = unique_indices
    
    # create motion data
    for unique_sensor_id in unique_sensor_ids:
        
        indices = unique_sensor_indices[unique_sensor_id]
        values = [sensor_values[i] for i in indices]
        
        mocap_data["motion"][unique_sensor_id.replace("/mocap/0/joint/", "")] = np.array(values)


    return mocap_data

def remove_root_position(mocap_data, root_joint_index):
    
    pos_local = mocap_data["motion"]["pos_local"]
    pos_world = mocap_data["motion"]["pos_world"]
    
    #print("pos_local s ", pos_local.shape)
    #print("pos_world s ", pos_world.shape)
    
    pos_local = pos_local.reshape(-1, mocap_joint_count, 3)
    pos_world = pos_world.reshape(-1, mocap_joint_count, 3)
    
    #print("pos_local 2 s ", pos_local.shape)
    #print("pos_world 2 s ", pos_world.shape)
    
    pos_local[:, root_joint_index, :] = 0.0
    pos_world_root = np.expand_dims(pos_world[:, root_joint_index, :], 1)
    
    #print("pos_world_root s ", pos_world_root.shape)
    
    pos_world -= pos_world_root
    
    pos_local = pos_local.reshape(-1, mocap_joint_count * 3)
    pos_world = pos_world.reshape(-1, mocap_joint_count * 3)
    
    mocap_data["motion"]["pos_local"] = pos_local
    mocap_data["motion"]["pos_world"] = pos_world

def quat_relative(q1, q2, normalize_inputs=True):
    """
    Relative rotation that takes q1 to q2, batched.
    q1, q2: shape (N, 4) (or any broadcastable (..., 4))
    Returns q_rel: shape (N, 4), with q_rel * q1 = q2.
    """
    if normalize_inputs:
        q1 = qnormalize_np(q1)
        q2 = qnormalize_np(q2)

    q1_inv = q_conj_np(q1)   # for unit quats, inverse == conjugate
    return qmul_np(q2, q1_inv)

def remove_root_rotation(mocap_data, root_joint_index):
    rot_local = mocap_data["motion"]["rot_local"]
    rot_world = mocap_data["motion"]["rot_world"]

    rot_local = rot_local.reshape(-1, mocap_joint_count, 4)
    rot_world = rot_world.reshape(-1, mocap_joint_count, 4)

    # set root local rotation to identity
    rot_local[:, root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))

    # extract root world rotation (F,4)
    rot_world_root = rot_world[:, root_joint_index, :]

    # inverse root rotation (F,4)
    rot_world_root_inv = q_conj_np(qnormalize_np(rot_world_root))

    # broadcast to (F,J,4)
    rot_world_root_inv = np.expand_dims(rot_world_root_inv, 1)
    rot_world_root_inv = np.repeat(rot_world_root_inv, repeats=mocap_joint_count, axis=1)

    # apply: q_new = q_root_inv * q_world
    rot_world = qmul_np(rot_world_root_inv, rot_world)

    rot_local = rot_local.reshape(-1, mocap_joint_count * 4)
    rot_world = rot_world.reshape(-1, mocap_joint_count * 4)

    mocap_data["motion"]["rot_local"] = rot_local
    mocap_data["motion"]["rot_world"] = rot_world
    
    
mocap_all_data = [] 
    
for mocap_data_file, mocap_valid_range in zip(mocap_data_files, mocap_valid_ranges):
    
    bvh_data = None
    fbx_data = None
    pkl_data = None
    
    if mocap_data_file.endswith(".bvh") or mocap_data_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_data_file.endswith(".fbx") or mocap_data_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
    elif mocap_data_file.endswith(".pkl"):
        pkl_data = load_pkl(mocap_data_path + "/" + mocap_data_file)
        mocap_data = pkl_to_mocap(pkl_data)
    
        
    if bvh_data is not None or fbx_data is not None:
        
        mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
        print("pos_local shape", mocap_data["motion"]["pos_local"].shape)
        #print("rot_local_euler shape", mocap_data["motion"]["rot_local_euler"].shape)
    
        mocap_data["motion"]["pos_local"] = mocap_data["motion"]["pos_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        mocap_data["motion"]["rot_local_euler"] = mocap_data["motion"]["rot_local_euler"][mocap_valid_range[0]:mocap_valid_range[1], ...]
    
        # set x and z offset of root joint to zero
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
        
    elif pkl_data is not None:
        
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
        mocap_data["motion"]["pos_world"] *= mocap_pos_scale
        
        mocap_data["motion"]["pos_world"] = mocap_data["motion"]["pos_world"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        mocap_data["motion"]["pos_local"] = mocap_data["motion"]["pos_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        mocap_data["motion"]["rot_world"] = mocap_data["motion"]["rot_world"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        mocap_data["motion"]["rot_local"] = mocap_data["motion"]["rot_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        
        
    if bvh_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif fbx_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif pkl_data is not None:
        remove_root_position(mocap_data, mocap_root_joint_index)
        remove_root_rotation(mocap_data, mocap_root_joint_index)

    mocap_all_data.append(mocap_data)
    
# get mocap info

mocap_skeleton = None
edge_list = None
poseRenderer = None

if "skeleton" in mocap_all_data[0]:

    mocap_skeleton = mocap_all_data[0]["skeleton"]
    
    offsets = mocap_skeleton["offsets"].astype(np.float32)
    parents = mocap_skeleton["parents"]
    children = mocap_skeleton["children"]
    mocap_root_joint_index = mocap_skeleton["joints"].index(mocap_skeleton["root"])

    mocap_motion = mocap_all_data[0]["motion"]["rot_local"]
    
    joint_count = mocap_motion.shape[1]
    joint_dim = mocap_motion.shape[2]
    pose_dim = joint_count * joint_dim
    mocap_dim = pose_dim
    
    # create edge list
    def get_edge_list(children):
        edge_list = []
    
        for parent_joint_index in range(len(children)):
            for child_joint_index in children[parent_joint_index]:
                edge_list.append([parent_joint_index, child_joint_index])
        
        return edge_list
    
    edge_list = get_edge_list(children)
    
    poseRenderer = PoseRenderer(edge_list)
    
else:
    
    pose_dim = mocap_all_data[0]["motion"]["rot_local"].shape[1]
    mocap_dim = pose_dim
    
# calc mean and std on all mocap data

if load_mocap_stat == True:
    mocap_mean = torch.load(mocap_mean_file)
    mocap_std = torch.load(mocap_std_file)
    
    mocap_mean.to(device)
    mocap_std.to(device)
else:

    mocap_sequences_concat = [ mocap_data["motion"]["rot_local"] for mocap_data in mocap_all_data ]
    mocap_sequences_concat = np.concatenate(mocap_sequences_concat, axis=0)
    mocap_sequences_concat = mocap_sequences_concat.reshape(mocap_sequences_concat.shape[0], -1)
    
    mocap_mean = np.mean(mocap_sequences_concat, axis=0, keepdims=True)
    mocap_std = np.std(mocap_sequences_concat, axis=0, keepdims=True)
    
    mocap_mean = torch.from_numpy(mocap_mean).to(dtype=torch.float32)
    mocap_std = torch.from_numpy(mocap_std).to(dtype=torch.float32)
    
    print("mocap_mean s ", mocap_mean.shape)
    print("mocap_std s ", mocap_std.shape)
    
    torch.save(mocap_mean, mocap_mean_file)
    torch.save(mocap_std, mocap_std_file)
    
    mocap_mean = mocap_mean.to(device)
    mocap_std = mocap_std.to(device)

    
"""
Load Data - Audio
"""

audio_all_data = []

for audio_data_file, audio_valid_range in zip(audio_data_files, audio_valid_ranges):   
    
    audio_data, _ = torchaudio.load(audio_data_path + audio_data_file)
  
    print("audio_data s ", audio_data.shape)
    
    audio_range_begin = audio_valid_range[0]
    audio_range_end = audio_valid_range[1]
    
    print("audio_valid_range ", audio_valid_range)
    
    if audio_range_begin > 0 and audio_range_end > 0:
        audio_valid_range_sample = [ int(audio_valid_range[0] * audio_sample_rate), int(audio_valid_range[1] * audio_sample_rate)]    
    else: 
        audio_valid_range_sample = [ 0, audio_data.shape[-1]]   

    print("audio_valid_range_sample ", audio_valid_range_sample)
    
    audio_data = audio_data[:, audio_valid_range_sample[0]:audio_valid_range_sample[1]]
    
    print("audio_data 2 s ", audio_data.shape)
    
    audio_all_data.append(audio_data)

"""
Load Vocos Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()

audio_mels_input_sequence = vocos.feature_extractor(torch.rand(size=(1, audio_waveform_input_seq_length), dtype=torch.float32).to(device))
tmp = vocos.decode(audio_mels_input_sequence)

if audio_waveform_input_seq_length != tmp.shape[-1]:
    print("Warning: reconstructing audio waveform from mels does not produce same number as audio samples as in original audio waveform")
    print("orig waveform sample count ", audio_waveform_input_seq_length)
    print("reconsrtucted waveform sample count ", tmp.shape[-1])

audio_mel_input_seq_length = audio_mels_input_sequence.shape[-1]
audio_latents_input_seq_length = audio_mel_input_seq_length // audio_mel_count_vae
audio_mel_filter_count = audio_mels_input_sequence.shape[1]

print("audio_mel_input_seq_length ", audio_mel_input_seq_length)
print("audio_latents_input_seq_length ", audio_latents_input_seq_length)
print("audio_mel_filter_count ", audio_mel_filter_count)

pos_encoding_max_length = max(mocap_input_seq_length, audio_latents_input_seq_length)

"""
for pose_count in range(8, 256):
    
    sample_count = pose_count * audio_samples_per_mocap_frame
    
    dummy_wave = torch.zeros((1, sample_count)).to(device)
    dummy_mels = vocos.feature_extractor(dummy_wave)
    dummy_wave2 = vocos.decode(dummy_mels)
    
    if dummy_wave.shape[-1] == dummy_wave2.shape[-1] and dummy_mels.shape[-1] % audio_mel_count_vae == 0:
        print("pc ", pose_count)
        print("dummy_mels s ", dummy_mels.shape)
"""

"""
Load Audio Autoencoder
"""

class Encoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        #print("conv_kernel_size ", conv_kernel_size)
        #print("stride ", stride)
        
        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_input_size ", dense_layer_input_size)
        #print("self.dense_layer_sizes[0] ", self.dense_layer_sizes[0])
        
        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x1 s ", x.shape)
        
        x = self.flatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x3 s ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", std.shape)

        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

encoder = Encoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

print(encoder)

encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))

encoder.eval()
    
# Decoder 
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        print("stride ", stride)
                
        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_output_size ", dense_layer_output_size)

        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x1 s ", x.shape)
        
        x = self.unflatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x3 s ", x.shape)

        return x
    
vae_conv_channel_counts_reversed = vae_conv_channel_counts.copy()
vae_conv_channel_counts_reversed.reverse()
    
vae_dense_layer_sizes_reversed = vae_dense_layer_sizes.copy()
vae_dense_layer_sizes_reversed.reverse()

vae_conv_channel_counts_reversed

decoder = Decoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts_reversed, vae_conv_kernel_size, vae_dense_layer_sizes_reversed).to(device)

print(decoder)

decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

decoder.eval()

# compute and normalise audio latents

if load_audio_latents_stat == True:
    audio_latents_mean = torch.load(audio_latents_mean_file)
    audio_latents_std = torch.load(audio_latents_std_file)
    
    audio_latents_mean = audio_latents_mean.to(device)
    audio_latents_std = audio_latents_std.to(device)
    
else:
    
    with torch.no_grad():
        
        audio_latents_all = []
        
        for audio_waveform_data in audio_all_data: 
            audio_mels = vocos.feature_extractor(audio_waveform_data.to(device))
            
            #print("audio_mels s ", audio_mels.shape)
            
            audio_mel_count = audio_mels.shape[-1]
            
            for amI in range(0, audio_mel_count - audio_mel_count_vae * batch_size, batch_size):
                
                audio_mels_excerpt = audio_mels[:, :, amI:amI + audio_mel_count_vae * batch_size]
                
                #print("amI ", amI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
                
                audio_encoder_in = audio_mels_excerpt.reshape((1, audio_mel_filter_count, batch_size, audio_mel_count_vae))
            
                #print("amI ", amI, " audio_encoder_in s ", audio_encoder_in.shape)
                
                audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
                
                #print("amI ", amI, " audio_encoder_in 2 s ", audio_encoder_in.shape)
                
                audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
                audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
                audio_encoder_out = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
            
                #print("amI ", amI, " audio_encoder_out s ", audio_encoder_out.shape)
                
                audio_latents_all.append(audio_encoder_out.detach().cpu())
                
        audio_latents_all = torch.cat(audio_latents_all, dim=0)
        
        #print("audio_latents_all s ", audio_latents_all.shape)
        
        audio_latents_mean = torch.mean(audio_latents_all, dim=0, keepdim=True)
        audio_latents_std = torch.std(audio_latents_all, dim=0, keepdim=True)
        
        #print("audio_latents_mean s ", audio_latents_mean.shape)
        #print("audio_latents_std s ", audio_latents_std.shape)
        
        torch.save(audio_latents_mean.detach(), audio_latents_mean_file)
        torch.save(audio_latents_std.detach(), audio_latents_std_file)
        
        audio_latents_mean = audio_latents_mean.to(device)
        audio_latents_std = audio_latents_std.to(device)


"""
Create Dataset
"""

X_mocap = []
X_audio = []
Y_audio = []

mocap_input_output_seq_length
mocap_frame_incr

for sI in range(len(mocap_all_data)):
    
    with torch.no_grad():
        
        mocap_data = mocap_all_data[sI]["motion"]["rot_local"].reshape(-1, pose_dim)
        audio_data = audio_all_data[sI][0]
    
        #print(sI)
        #print("mocap_data s ", mocap_data.shape)
        #print("audio_data s ", audio_data.shape)
        
        mocap_frame_count = mocap_data.shape[0]
        
        # the range begin of 3 enshures no negative audio_waveform_start index and the -10 for the range end ensures that the audio_waveform_end doesn't exceed the audio waveform length
        for mfI in range(3, mocap_frame_count - mocap_input_output_seq_length - 10 - mocap_frame_incr, mocap_frame_incr):
            
            print("mfI ", mfI, " out of ", (mocap_frame_count - mocap_input_output_seq_length - 10))
            
            # mocap sequence part
            
            # get mocap sequence
            mocap_excerpt_start = mfI
            mocap_excerpt_end = mfI + mocap_input_output_seq_length
            
            #print("mocap_excerpt_start ", mocap_excerpt_start)
            #print("mocap_excerpt_end ", mocap_excerpt_end)
            
            mocap_excerpt = mocap_data[mocap_excerpt_start:mocap_excerpt_end, :]
            mocap_excerpt = torch.from_numpy(mocap_excerpt).unsqueeze(0).to(torch.float32).to(device)
            
            #print("mfI ", mfI, " me s ", mocap_excerpt.shape)
            
            # normalise mocap excerpt
            mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
            
            X_mocap.append(mocap_excerpt_norm.cpu())
            
            # audio sequence part (input squence)
            asI = mfI * audio_samples_per_mocap_frame
            audio_waveform_start = asI
            audio_waveform_end = audio_waveform_start + audio_waveform_input_seq_length
            
            audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            
            #print("mfI ", mfI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
            
            audio_encoder_in = audio_mels_excerpt.reshape((1, audio_mel_filter_count, audio_latents_input_seq_length, audio_mel_count_vae))
    
            #print("mfI ", mfI, " audio_encoder_in s ", audio_encoder_in.shape)
        
            audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
        
            #print("mfI ", mfI, " audio_encoder_in 2 s ", audio_encoder_in.shape)
        
            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            audio_latents = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
            
            #print("mfI ", mfI, " audio_latents s ", audio_latents.shape)
            
            audio_latents_norm = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
            
            #print("mfI ", mfI, " audio_latents_norm s ", audio_latents_norm.shape)
            
            audio_latents_norm = audio_latents_norm.unsqueeze(0)
            
            #print("mfI ", mfI, " audio_latents_norm 2 s ", audio_latents_norm.shape)
            
            X_audio.append(audio_latents_norm.cpu())
    
            # audio sequence part (target squence)
            
            y_audio_grouped = []
            
            for tsI in range(mocap_output_seq_length):
                
                mfI2 = mfI + tsI + 1
                
                #print("mfI2 ", mfI2)
                
                asI = mfI2 * audio_samples_per_mocap_frame
                audio_waveform_start = asI
                audio_waveform_end = audio_waveform_start + audio_waveform_input_seq_length
                
                audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
                audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
                
                #print("mfI2 ", mfI2, " audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
                #print("mfI2 ", mfI2, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
                
                audio_encoder_in = audio_mels_excerpt.reshape((1, audio_mel_filter_count, audio_latents_input_seq_length, audio_mel_count_vae))
    
                #print("mfI2 ", mfI2, " audio_encoder_in s ", audio_encoder_in.shape)
            
                audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
            
                #print("mfI2 ", mfI2, " audio_encoder_in 2 s ", audio_encoder_in.shape)
            
                audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
                audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
                audio_latents = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
                
                #print("mfI2 ", mfI2, " audio_latents s ", audio_latents.shape)
                
                audio_latents_norm = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
                
                #print("mfI2 ", mfI2, " audio_latents_norm s ", audio_latents_norm.shape)
                
                audio_latents_norm = audio_latents_norm.unsqueeze(0)
                
                #print("mfI2 ", mfI2, " audio_latents_norm 2 s ", audio_latents_norm.shape)
    
                y_audio_grouped.append(audio_latents_norm.cpu())
                
            y_audio_grouped = torch.cat(y_audio_grouped, dim=0)
            
            #print("y_audio_grouped s ", y_audio_grouped.shape)
            
            y_audio_grouped = y_audio_grouped.unsqueeze(0)
            
            #print("y_audio_grouped 2 s ", y_audio_grouped.shape)
            
            Y_audio.append(y_audio_grouped)


X_mocap = torch.cat(X_mocap, dim=0)
X_audio = torch.cat(X_audio, dim=0)
Y_audio = torch.cat(Y_audio, dim=0)


print("X_mocap s ", X_mocap.shape)
print("X_audio s ", X_audio.shape)
print("Y_audio s ", Y_audio.shape)


class SequenceDataset(Dataset):
    def __init__(self, X_mocap, X_audio, Y_audio):
        self.X_mocap = X_mocap
        self.X_audio = X_audio
        self.Y_audio = Y_audio
    
    def __len__(self):
        return self.X_mocap.shape[0]
    
    def __getitem__(self, idx):
        return self.X_mocap[idx, ...], self.X_audio[idx, ...], self.Y_audio[idx, ...]

full_dataset = SequenceDataset(X_mocap, X_audio, Y_audio)

x_item_mocap, x_item_audio, y_item_audio = full_dataset[0]

print("x_item_mocap s ", x_item_mocap.shape)
print("x_item_audio s ", x_item_audio.shape)
print("y_item_audio s ", y_item_audio.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

x_batch_mocap, x_batch_audio, y_batch_audio = next(iter(train_loader))

print("x_batch_mocap s ", x_batch_mocap.shape)
print("x_batch_audio s ", x_batch_audio.shape)
print("y_batch_audio s ", y_batch_audio.shape)

"""
Create Models - PositionalEncoding
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # for batch-first: [1, max_len, dim_model]
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        
        #print("token_embedding s ", token_embedding.shape)
        #print("pos_encoding s ", self.pos_encoding.shape)
        
        # token_embedding: [batch_size, seq_len, dim_model]
        seq_len = token_embedding.size(1)
        # broadcast over batch dimension
        pe = self.pos_encoding[:, :seq_len, :]
        
        return self.dropout(token_embedding + pe)


"""
Create Models - Transformer
"""


class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        mocap_dim,
        audio_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim) # map mocap data to embedding
        self.audio2embed = nn.Linear(audio_dim, embed_dim) # map audio data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2audio = nn.Linear(embed_dim, audio_dim) # map embedding to audio data
        
    def get_src_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.ones(size, size)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
       
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, mocap_data, audio_data):
        
        #print("forward")
        
        #print("data s ", data.shape)

        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        
        mocap_embedded = self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim)
        mocap_embedded = self.positional_encoder(mocap_embedded)
        
        audio_embedded = self.audio2embed(audio_data) * math.sqrt(self.embed_dim)
        audio_embedded = self.positional_encoder(audio_embedded)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask =tgt_mask)
        
        out = self.embed2audio(decoder_out)
        
        return out

transformer = Transformer(mocap_dim=mocap_dim, 
                          audio_dim=audio_dim,
                          embed_dim=transformer_embed_dim, 
                          num_heads=transformer_head_count, 
                          num_encoder_layers=transformer_layer_count, 
                          num_decoder_layers=transformer_layer_count, 
                          dropout_p=transformer_dropout,
                          pos_encoding_max_length=pos_encoding_max_length).to(device)

print(transformer)

if load_weights and transformer_load_weights_path:
    transformer.load_state_dict(torch.load(transformer_load_weights_path, map_location=device))


# test model

x_mocap_batch, x_audio_batch, y_audio_batch = next(iter(train_loader))

transformer_mocap_input = x_mocap_batch[:, :mocap_input_seq_length, ...].to(device)
transformer_audio_input = x_audio_batch.to(device)
transformer_audio_target = y_audio_batch[:, 0, ...].to(device)

transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

print("transformer_mocap_input s ", transformer_mocap_input.shape)
print("transformer_audio_input s ", transformer_audio_input.shape)
print("transformer_audio_target s ", transformer_audio_target.shape)
print("transformer_audio_output s ", transformer_audio_output.shape)

"""
Training
"""

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

n1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

# Define perceptial loss: MR-STFT with perceptual mel weighting
perc_loss = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",          # use mel-scaled spectrograms
    n_bins=128,           # number of mel bins for the perceptual weighting
    sample_rate=48000,    # set to the actual SR used (48 kHz in your code)
    perceptual_weighting=True
)

def audio_latents_loss(y_latents, yhat_latents):
    
    _all = mse_loss(yhat_latents, y_latents)

    return _all

def audio_mel_loss(y_mels, yhat_mels):
    
    _aml = mse_loss(yhat_mels, y_mels)

    return _aml

def audio_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

def loss(y_latents_norm, yhat_latents_norm):
    
    #print("loss begin")
    
    #print("y_latents_norm s ", y_latents_norm.shape, " yhat_latents_norm s ", yhat_latents_norm.shape)
    
    batch_size = y_latents_norm.shape[0]
    
    #print("batch_size s ", batch_size)
    
    y_latents_norm = y_latents_norm.reshape((-1, latent_dim))
    yhat_latents_norm = yhat_latents_norm.reshape((-1, latent_dim))
    
    #print("y_latents_norm 2 s ", y_latents_norm.shape, " yhat_latents_norm 2 s ", yhat_latents_norm.shape)
    
    # calculate audio latent loss
    #start_time = time.time()
    
    _audio_latents_loss = audio_latents_loss(y_latents_norm, yhat_latents_norm)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio latent loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("1")
    
    # convert latents norm to latents
    #start_time = time.time()
    
    y_latents = y_latents_norm * audio_latents_std + audio_latents_mean
    yhat_latents = yhat_latents_norm * audio_latents_std + audio_latents_mean
    
    #print("y_latents s ", y_latents.shape, " yhat_latents s ", yhat_latents.shape)
    
    # decode latents into mels
    y_mels_regrouped = decoder(y_latents)
    yhat_mels_regrouped = decoder(yhat_latents)
    
    #print("2")
    
    #print("y_mels_regrouped s ", y_mels_regrouped.shape, " yhat_mels_regrouped s ", yhat_mels_regrouped.shape)
    
    # regroup mels
    y_mels = y_mels_regrouped.reshape(batch_size, -1, 1, audio_mel_filter_count, audio_mel_count_vae)
    yhat_mels = yhat_mels_regrouped.reshape(batch_size, -1, 1, audio_mel_filter_count, audio_mel_count_vae)
    
    #print("y_mels s ", y_mels.shape, " yhat_mels s ", yhat_mels.shape)

    y_mels = y_mels.permute(0, 2, 3, 1, 4) 
    yhat_mels = yhat_mels.permute(0, 2, 3, 1, 4) 
    
    #print("y_mels 2 s ", y_mels.shape, " yhat_mels 2 s ", yhat_mels.shape)
    
    y_mels = y_mels.reshape(batch_size, audio_mel_filter_count, -1) 
    yhat_mels = yhat_mels.reshape(batch_size, audio_mel_filter_count, -1) 
    
    #print("y_mels 3 s ", y_mels.shape, " yhat_mels 3 s ", yhat_mels.shape)
    
    # calculate audio mel loss
    _audio_mel_loss = audio_mel_loss(y_mels, yhat_mels)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio mel loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("3")
    
    # convert audio mels to audio waveforms
    #start_time = time.time()
    
    y_waveform = vocos.decode(y_mels)
    yhat_waveform = vocos.decode(yhat_mels)
    
    #print("4")
    
    #print("y_waveform s ", y_waveform.shape, " yhat_waveform s ", yhat_waveform.shape)
    
    y_waveform = y_waveform.unsqueeze(1)
    yhat_waveform = yhat_waveform.unsqueeze(1)
    
    #print("y_waveform 2 s ", y_waveform.shape, " yhat_waveform 2 s ", yhat_waveform.shape)
    
    # calculate audio perceptual loss
    _audio_perc_loss = audio_perc_loss(y_waveform, yhat_waveform)
    
    #end_time = time.time()
    #elapsed_seconds = end_time - start_time
    #print(f"calculate audio perceptual loss: Elapsed time: {elapsed_seconds:.4f} seconds")
    
    #print("5")
    
    _total_loss = 0.0
    _total_loss += _audio_latents_loss * 0.33
    _total_loss += _audio_mel_loss * 0.33
    _total_loss += _audio_perc_loss * 0.33
    
    #print("loss end")
    
    return _total_loss

def train_step(x_mocap, x_audio, y_audio):

    #print("x_mocap s ", x_mocap.shape)
    #print("x_audio s ", x_audio.shape)
    #print("y_audio s ", y_audio.shape)
    
    # first step (teacher forcing)
    
    transformer_mocap_input = x_mocap[:, :mocap_input_seq_length, ...]
    transformer_audio_input = x_audio
    transformer_audio_target = y_audio[:, 0, ...]
    
    #print("transformer_mocap_input s ", transformer_mocap_input.shape)
    #print("transformer_audio_input s ", transformer_audio_input.shape)
    #print("transformer_audio_target s ", transformer_audio_target.shape)
    
    transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

    # todo: also compute mel loss and waveform perceptual loss
    step_loss = loss(transformer_audio_target, transformer_audio_output) 
    _loss = step_loss.detach().cpu()
    
    # Backpropagation
    optimizer.zero_grad()
    step_loss.backward()
    optimizer.step()
    
    # next steps(non-teacher forcing)
    for mfI in range(1, mocap_output_seq_length):
        
        #print("mfI ", mfI)
        
        transformer_mocap_input = x_mocap[:, mfI:mocap_input_seq_length + mfI, ...].to(device)
        transformer_audio_input = transformer_audio_output.detach().clone()
        transformer_audio_target = y_audio[:, mfI, ...]
        
        #print("transformer_mocap_input s ", transformer_mocap_input.shape)
        #print("transformer_audio_input s ", transformer_audio_input.shape)
        #print("transformer_audio_target s ", transformer_audio_target.shape)
        
        transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

        step_loss = loss(transformer_audio_target, transformer_audio_output) 
        _loss += step_loss.detach().cpu()

        # Backpropagation
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()
        
    _loss /= (mocap_output_seq_length + 1)

    return _loss

"""
x_mocap_batch, x_audio_batch, y_audio_batch = next(iter(train_loader))
x_mocap_batch = x_mocap_batch.to(device)
x_audio_batch = x_audio_batch.to(device)
y_audio_batch = y_audio_batch.to(device)
_loss = train_step(x_mocap_batch, x_audio_batch, y_audio_batch)
"""

@torch.no_grad()
def test_step(x_mocap, x_audio, y_audio):
    
    transformer.eval()

    #print("x_mocap s ", x_mocap.shape)
    #print("x_audio s ", x_audio.shape)
    #print("y_audio s ", y_audio.shape)
    
    # first step (teacher forcing)
    
    transformer_mocap_input = x_mocap[:, :mocap_input_seq_length, ...]
    transformer_audio_input = x_audio
    transformer_audio_target = y_audio[:, 0, ...]
    
    #print("transformer_mocap_input s ", transformer_mocap_input.shape)
    #print("transformer_audio_input s ", transformer_audio_input.shape)
    #print("transformer_audio_target s ", transformer_audio_target.shape)
    
    transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

    # todo: also compute mel loss and waveform perceptual loss
    step_loss = loss(transformer_audio_target, transformer_audio_output) 
    _loss = step_loss.detach().cpu()

    # next steps(non-teacher forcing)
    for mfI in range(1, mocap_output_seq_length):
        
        #print("mfI ", mfI)
        
        transformer_mocap_input = x_mocap[:, mfI:mocap_input_seq_length + mfI, ...].to(device)
        transformer_audio_input = transformer_audio_output.detach().clone()
        transformer_audio_target = y_audio[:, mfI, ...]
        
        #print("transformer_mocap_input s ", transformer_mocap_input.shape)
        #print("transformer_audio_input s ", transformer_audio_input.shape)
        #print("transformer_audio_target s ", transformer_audio_target.shape)
        
        transformer_audio_output = transformer(transformer_mocap_input, transformer_audio_input)

        step_loss = loss(transformer_audio_target, transformer_audio_output) 
        _loss += step_loss.detach().cpu()

    _loss /= (mocap_output_seq_length + 1)
    
    transformer.train()

    return _loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []

        for train_batch in train_dataloader:
            x_mocap = train_batch[0].to(device)
            x_audio = train_batch[1].to(device)
            y_audio = train_batch[2].to(device)
            
            _loss = train_step(x_mocap, x_audio, y_audio)
            
            _loss = _loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            x_mocap = test_batch[0].to(device)
            x_audio = test_batch[1].to(device)
            y_audio = test_batch[2].to(device)
            
            _loss = test_step(x_mocap, x_audio, y_audio)

            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epochs))

# inference

audio_window_length = audio_samples_per_mocap_frame * 4
audio_window_env = torch.hann_window(audio_window_length)

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)


def create_mocap_anim(mocap_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    pose_sequence = mocap_data[mocap_start_frame_index:mocap_start_frame_index + mocap_frame_count]

    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=1000 / mocap_fps, loop=0)
    

def create_orig_audio(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    audio_waveform_excerpt_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_excerpt_end_index = audio_waveform_excerpt_start_index + mocap_frame_count * audio_samples_per_mocap_frame

    audio_waveform_excerpt = waveform_data[audio_waveform_excerpt_start_index:audio_waveform_excerpt_end_index]
    
    torchaudio.save(file_name, audio_waveform_excerpt.unsqueeze(0), audio_sample_rate)

@torch.no_grad()
def create_vocos_audio(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    audio_waveform_excerpt_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_excerpt_sample_count = mocap_frame_count * audio_samples_per_mocap_frame
    audio_waveform_excerpt_end_index = audio_waveform_excerpt_start_index + audio_waveform_excerpt_sample_count

    orig_audio_waveform = waveform_data[audio_waveform_excerpt_start_index:audio_waveform_excerpt_end_index]

    gen_audio_waveform = torch.zeros((audio_waveform_excerpt_sample_count), dtype=torch.float32)
    
    #print("gen_audio_waveform s ", gen_audio_waveform.shape)
    
    for aSI in range(0, audio_waveform_excerpt_sample_count - audio_waveform_input_seq_length, audio_samples_per_mocap_frame):
        
        #print("aSI ", aSI)
        
        audio_waveform = orig_audio_waveform[aSI:aSI + audio_waveform_input_seq_length].to(device)

        #print("audio_waveform s ", audio_waveform.shape)

        audio_waveform = audio_waveform.reshape((1, audio_waveform_input_seq_length))
        audio_mels = vocos.feature_extractor(audio_waveform)
        
        #print("audio_mels_2 s ", audio_mels.shape)
        
        audio_waveform_2 = vocos.decode(audio_mels)
        
        #print("audio_waveform_2 s ", audio_waveform_2.shape)
        
        audio_waveform_2_window = audio_waveform_2.reshape(-1)[-audio_window_length:]
        audio_waveform_2_window = audio_waveform_2_window.detach().cpu()
        
        #print("audio_waveform_2_window s ", audio_waveform_2_window.shape)
        #print("audio_window_env s ", audio_window_env.shape)
        #print("gen_audio_waveform[aSI:aSI + audio_window_length] s ", gen_audio_waveform[aSI:aSI + audio_window_length].shape)
        
        gen_audio_waveform[aSI:aSI + audio_window_length] += audio_waveform_2_window * audio_window_env

        torchaudio.save(file_name, gen_audio_waveform.unsqueeze(0), audio_sample_rate)

@torch.no_grad()
def create_gen_audio(mocap_data, waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    transformer.eval()
    
    # prepare mocap data
    mocap_end_frame_index = mocap_start_frame_index + mocap_frame_count
    mocap_data = mocap_data[mocap_start_frame_index:mocap_end_frame_index, ...].to(device)
    mocap_data_norm = (mocap_data - mocap_mean) / (mocap_std + 1e-8)
    mocap_data_norm = mocap_data_norm.unsqueeze(0)
    
    #print("mocap_data_norm s ", mocap_data_norm.shape)
    
    # prepare audio data

    audio_waveform_sample_count = mocap_frame_count * audio_samples_per_mocap_frame
    gen_audio_waveform = torch.zeros((audio_waveform_sample_count), dtype=torch.float32)

    audio_waveform_start_sample_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    audio_waveform_end_sample_index =  audio_waveform_start_sample_index + audio_waveform_input_seq_length
    audio_waveform_data = waveform_data[audio_waveform_start_sample_index:audio_waveform_end_sample_index]
    audio_waveform_data = audio_waveform_data.unsqueeze(0).to(device)
    
    #print("audio_waveform_data s ", audio_waveform_data.shape)

    audio_mels_data = vocos.feature_extractor(audio_waveform_data)
    
    #print("audio_mels_data s ", audio_mels_data.shape)
    
    audio_encoder_in = audio_mels_data.reshape((1, audio_mel_filter_count, -1, audio_mel_count_vae))
    
    #print("audio_encoder_in s ", audio_encoder_in.shape)
    
    audio_encoder_in = audio_encoder_in.permute((2, 0, 1, 3))
                
    #print("audio_encoder_in 2 s ", audio_encoder_in.shape)
    
    audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
    audio_encoder_out_std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
    audio_latents = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)
    
    #print("audio_latents_data s ", audio_latents_data.shape)
    
    audio_latents_norm = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
            
    #print("audio_latents_norm s ", audio_latents_norm.shape)
    
    audio_latents_norm = audio_latents_norm.unsqueeze(0)
    
    #print("audio_latents_norm 2 s ", audio_latents_norm.shape)
    
    # predict audio
    
    x_mocap = mocap_data_norm[:, :mocap_input_seq_length, ...]
    
    #print("x_mocap s ", x_mocap.shape)
    
    x_audio = audio_latents_norm
    
    #print("x_audio s ", x_audio.shape)
    
    yhat_audio = transformer(x_mocap, x_audio)
    
    #print("yhat_audio s ", yhat_audio.shape)

    yhat_audio_latents_norm = yhat_audio.detach().squeeze(0)
    
    #print("yhat_audio_latents_norm s ", yhat_audio_latents_norm.shape)
    
    yhat_audio_latents = yhat_audio_latents_norm * audio_latents_std + audio_latents_mean

    #print("yhat_audio_latents s ", yhat_audio_latents.shape)
    
    yhat_audio_mels = decoder(yhat_audio_latents)
    
    #print("yhat_audio_mels s ", yhat_audio_mels.shape)
    
    yhat_audio_mels = yhat_audio_mels.permute((1, 2, 0, 3))
    
    #print("yhat_audio_mels 2 s ", yhat_audio_mels.shape)
    
    yhat_audio_mels = yhat_audio_mels.reshape((1, audio_mel_filter_count, -1))
    
    #print("yhat_audio_mels 3 s ", yhat_audio_mels.shape)
    
    yhat_audio_waveform = vocos.decode(yhat_audio_mels)
    
    #print("yhat_audio_waveform s ", yhat_audio_waveform.shape)
    
    yhat_audio_window = yhat_audio_waveform.reshape(-1)[-audio_window_length:]
    yhat_audio_window = yhat_audio_window.detach().cpu()
    
    #print("yhat_audio_window s ", yhat_audio_window.shape)
    
    gen_audio_waveform[:audio_window_length] += yhat_audio_window * audio_window_env

    for mFI in range(1, mocap_frame_count - mocap_input_seq_length):  
        
        #print("mFI ", mFI)
    
        aSI = mFI * audio_samples_per_mocap_frame
        
        x_mocap = mocap_data_norm[:, mFI:mFI + mocap_input_seq_length, ...]
        x_audio = yhat_audio.detach()
        
        #print("x_mocap s ", x_mocap.shape)
        #print("x_audio s ", x_audio.shape)
        
        yhat_audio = transformer(x_mocap, x_audio)
        
        #print("yhat_audio s ", yhat_audio.shape)
        
        yhat_audio_latents_norm = yhat_audio.detach().squeeze(0)
        
        #print("yhat_audio_latents_norm s ", yhat_audio_latents_norm.shape)
        
        yhat_audio_latents = yhat_audio_latents_norm * audio_latents_std + audio_latents_mean

        #print("yhat_audio_latents s ", yhat_audio_latents.shape)
        
        yhat_audio_mels = decoder(yhat_audio_latents)
        
        #print("yhat_audio_mels s ", yhat_audio_mels.shape)
        
        yhat_audio_mels = yhat_audio_mels.permute((1, 2, 0, 3))
        
        #print("yhat_audio_mels 2 s ", yhat_audio_mels.shape)
        
        yhat_audio_mels = yhat_audio_mels.reshape((1, audio_mel_filter_count, -1))
        
        #print("yhat_audio_mels 3 s ", yhat_audio_mels.shape)
        
        yhat_audio_waveform = vocos.decode(yhat_audio_mels)
        
        #print("yhat_audio_waveform s ", yhat_audio_waveform.shape)
        
        yhat_audio_window = yhat_audio_waveform.reshape(-1)[-audio_window_length:]
        yhat_audio_window = yhat_audio_window.detach().cpu()
        
        #print("yhat_audio_window s ", yhat_audio_window.shape)
        
        gen_audio_waveform[aSI:aSI+ audio_window_length] += yhat_audio_window * audio_window_env

    torchaudio.save(file_name, gen_audio_waveform.unsqueeze(0), audio_sample_rate)
    
    transformer.train()
    
    
"""
generate audio with orig mocap data
"""
    
test_mocap_data = torch.from_numpy(mocap_all_data[0]["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)
test_audio_data = audio_all_data[0][0]

#print("test_mocap_data s ", test_mocap_data.shape)
#print("test_audio_data s ", test_audio_data.shape)

test_mocap_start_times = [100, 200, 300]
test_mocap_duration = 30
    
for test_mocap_start_time in test_mocap_start_times:
    if poseRenderer is not None:
        create_mocap_anim(test_mocap_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/anims/orig_mocap_{}-{}.gif".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_orig_audio(test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/orig_audio_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_vocos_audio(test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/vocos_audio_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_gen_audio(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/gen_audio_{}-{}_epoch_{}_orig.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))


"""
generate audio with alternative mocap data
"""

"""
# Diane Version

test_mocap_data_path = "E:/data/mocap/Diane/Solos/ZHdK_10.10.2025/fbx_60hz/"

test_mocap_data_files = ["trial-002.fbx",
                         "trial-003.fbx",
                         "trial-004.fbx",
                         "trial-005.fbx",
                         "trial-006.fbx"]
test_mocap_valid_ranges = [[364, 22739],
                           [531, 22905],
                           [549, 22924],
                           [613, 22988],
                           [1171, 23545]]
test_mocap_start_times = [[100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300],
                          [100, 200, 300]]

# Eleni Version

test_mocap_data_path = "E:/Data/mocap/Eleni/Solos/ZHdK_04.12.2025/fbx_60hz/"
test_mocap_data_files = ["Eline_Session-001.fbx",
                         "Eline_Session-003.fbx"]
test_mocap_valid_ranges = [[583, 49965],
                           [874, 50257]]
test_mocap_start_times = [[30, 120, 240],
                          [30, 120, 240]]

# Tim Version

test_mocap_data_path = "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/fbx_60hz/"
test_mocap_data_files = ["trial-001.fbx"]
test_mocap_valid_ranges = [[1301, 24945]]
test_mocap_start_times = [[0, 120, 240]]



"""


test_mocap_data_path = "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/xsens2osc_30hz/"
test_mocap_data_files = ["trial-001.pkl"]
test_mocap_valid_ranges = [[649, 12487]]
test_mocap_start_times = [[30, 120, 240]]


test_mocap_duration = 30
test_audio_data = audio_all_data[0][0]

for fI in range(len(test_mocap_data_files)):
    
    bvh_data = None
    fbx_data = None
    pkl_data = None
    
    test_mocap_data_file = test_mocap_data_files[fI]
    test_mocap_valid_range = test_mocap_valid_ranges[fI]
    test_mocap_start_times2 = test_mocap_start_times[fI]

    if test_mocap_data_file.endswith(".bvh") or test_mocap_data_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(test_mocap_data_path + test_mocap_data_file)
        test_mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif test_mocap_data_file.endswith(".fbx") or test_mocap_data_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(test_mocap_data_path + test_mocap_data_file)
        test_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
    elif test_mocap_data_file.endswith(".pkl"):
        pkl_data = load_pkl(test_mocap_data_path + "/" + test_mocap_data_file)
        test_mocap_data = pkl_to_mocap(pkl_data)
        
    if bvh_data is not None or fbx_data is not None:
        
        test_mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
        test_mocap_data["motion"]["pos_local"] *= mocap_pos_scale

        test_mocap_data["motion"]["pos_local"] = test_mocap_data["motion"]["pos_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        test_mocap_data["motion"]["rot_local_euler"] = test_mocap_data["motion"]["rot_local_euler"][mocap_valid_range[0]:mocap_valid_range[1], ...]
    
        # set x and z offset of root joint to zero
        test_mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
        test_mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
        
    elif pkl_data is not None:
        
        test_mocap_data["motion"]["pos_local"] *= mocap_pos_scale
        test_mocap_data["motion"]["pos_world"] *= mocap_pos_scale
        
        test_mocap_data["motion"]["pos_world"] = test_mocap_data["motion"]["pos_world"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        test_mocap_data["motion"]["pos_local"] = test_mocap_data["motion"]["pos_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        test_mocap_data["motion"]["rot_world"] = test_mocap_data["motion"]["rot_world"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        test_mocap_data["motion"]["rot_local"] = test_mocap_data["motion"]["rot_local"][mocap_valid_range[0]:mocap_valid_range[1], ...]
        
    if bvh_data is not None:
        test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])
        test_mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif fbx_data is not None:
        test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])
        test_mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif pkl_data is not None:
        remove_root_position(test_mocap_data, mocap_root_joint_index)
        remove_root_rotation(test_mocap_data, mocap_root_joint_index)
        
    test_mocap_data = torch.from_numpy(test_mocap_data["motion"]["rot_local"]).to(torch.float32)
    test_mocap_data = test_mocap_data.reshape(-1, mocap_pose_dim)

    for test_mocap_start_time in test_mocap_start_times2:
        if poseRenderer is not None:
            create_mocap_anim(test_mocap_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/anims/test_mocap_{}_{}-{}.gif".format(test_mocap_data_file, test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
        create_gen_audio(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, test_mocap_duration * mocap_fps, "results/audio/test_gen_audio_{}_{}-{}_epoch_{}.wav".format(test_mocap_data_file, test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))
    
