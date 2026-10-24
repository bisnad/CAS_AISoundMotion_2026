import os
import math
import numpy as np
import pickle
import threading
import queue
import time
import sounddevice as sd
import torch
import torchaudio
from torch import nn
from torchaudio.functional import highpass_biquad

# osc specific imports
from pythonosc import dispatcher
from pythonosc import osc_server

# vocos specific
from vocos import Vocos

# mocap specific
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, q_conj_np, qmul_np, slerp
from common.pose_renderer import PoseRenderer

# -------------------------------------------------------
# Settings
# -------------------------------------------------------

# compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')


# mocap parameters

"""
# Eleni
mocap_data_file = "E:/Data/mocap/Eleni/Solos/ZHdK_04.12.2025/xsens2osc_30hz/Eline_Session-002.pkl"

# Diane
mocap_data_file = "E:/Data/mocap/Diane/Solos/ZHdK_10.10.2025/xsens2osc_30hz/trial-002.pkl"

# Tim
mocap_data_file = "E:/Data/mocap/Tim/Solos/ZHdK_09.12.2025/xsens2osc_30hz/trial-002.pkl"
"""

mocap_data_file = "data/mocap/Eline_Session-002.pkl"

mocap_pos_scale = 0.1
mocap_fps = 30 # 50, 60
mocap_input_seq_length = 60 # (56 at 60 fps, 60 at 30 or 50 fps)
mocap_joint_count = 23 # will be overwritten for fbx and bvh files
mocap_joint_dim = 4 # will be overwritten for fbx and bvh files
mocap_root_joint_index = 0 # will be overwritten for fbx and bvh files
mocap_pose_dim = mocap_joint_count * mocap_joint_dim

# audio parameters

"""
# Eleni
audio_data_file = "E:/data/audio/Eleni/48khz/4_5870821179501060412.wav"

# Diane
audio_data_file = "E:/Data/audio/Diane/48khz/4d69949b.wav"

# Tim
audio_data_file = "E:/Data/audio/Tim/48khz/SajuHariPlacePrizeEntry2010.wav"
"""

print(sd.query_devices())
audio_data_file = "data/audio/Eline_Session-002.wav"
audio_output_device = 8
audio_sample_rate = 48000
audio_channels = 1
audio_samples_per_mocap_frame = audio_sample_rate // mocap_fps
audio_waveform_input_seq_length = int(audio_sample_rate / mocap_fps * mocap_input_seq_length)

audio_window_size = audio_samples_per_mocap_frame * 4
gen_buffer_size = audio_window_size
audio_window_offset = audio_window_size // 4 # the correct value would be: audio_sample_rate // mocap_fps 

audio_mel_filter_count = None # to be computed
audio_mel_input_seq_length = None # to be computed
audio_latents_input_seq_length = None # to be computed

play_buffer_size = audio_window_size * 4 # * 16
playback_latency = 1.0
hpf_cutoff_hz = 15.0

# OSC parameters
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

# FIFO Queue Settings
#max_fifo_queue_length = 32 # 32
max_fifo_queue_length = play_buffer_size // audio_window_offset * 2 # 32

print("play_buffer_size ", play_buffer_size)
print("audio_window_offset ", audio_window_offset)
print("max_fifo_queue_length ", max_fifo_queue_length)

# Vocos pre-trained model
vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"

# VAE parameters
latent_dim = 32 # 128
vae_audio_mel_count = 8
vae_conv_channel_counts = [16, 32, 64, 128]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [512]

"""
# Eleni

# kld 0.1
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_std.pt"

# kld 0.1 and joint dropout
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Eleni_Take2fbx_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Eleni_Take2fbx_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_std.pt"

# kld 1.0 and dec training
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Eleni_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_std.pt"

# Diane

# kld 0.1
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_std.pt"

# kld 0.1 and joint dropout
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_std.pt"

# kld 1.0 and dec training
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Diane_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/decoder_only_weights_epoch_200"
audio_latents_mean_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_std.pt"

# Tim

# kld 0.1
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/stat/latents_std.pt"

# kld 0.1 and joint dropout
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/stat/latents_std.pt"

# kld 1.0 and dec training
encoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../VAE/Audio_VAE/Audio_VAE_Vocos/Training/results_Tim_audio_vae_vocos_cnn_ld32_kld1.0_decoder_training/weights/decoder_only_weights_epoch_200"
audio_latents_mean_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_mean.pt"
audio_latents_std_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/stat/latents_std.pt"

"""

encoder_weights_file = "data/results/weights/encoder_weights_epoch_400"
decoder_weights_file = "data/results/weights/decoder_weights_epoch_400"
audio_latents_mean_file = "data/results/stat/latents_mean.pt"
audio_latents_std_file = "data/results/stat/latents_std.pt"

# Transformer parameters


transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1
pos_encoding_max_length = mocap_input_seq_length

"""
# Eleni

# kld 0.1
transformer_weights_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

# kld 0.1 and joint dropout
transformer_weights_file = "../Training/results_Eleni_Take2fbx_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/weights/transformer_weights_epoch_200"

# kld 1.0 and dec training
transformer_weights_file = "../Training/results_Eleni_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/weights/transformer_weights_epoch_200"

# Diane

# kld 0.1
transformer_weights_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

# kld 0.1 and joint dropout
transformer_weights_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/weights/transformer_weights_epoch_200"

# kld 1.0 and dec training
transformer_weights_file = "../Training/results_Diane_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/weights/transformer_weights_epoch_200"


# Tim

# kld 0.1
transformer_weights_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1/weights/transformer_weights_epoch_200"

# kld 0.1 and joint dropout
transformer_weights_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld0.1_jointdropout/weights/transformer_weights_epoch_200"

# kld 1.0 and dec training
transformer_weights_file = "../Training/results_Tim_Take2osc_30hz_transformer_audiovae_ld32_kld1.0_deconly/weights/transformer_weights_epoch_200"
"""

transformer_weights_file = "data/results/weights/transformer_weights_epoch_200"

# -------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------

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

def load_mocap(file_path, scale):
    bvh_tools = bvh.BVH_Tools()
    fbx_tools = fbx.FBX_Tools()
    mocap_tools = mocap.Mocap_Tools()
    
    bvh_data = None
    fbx_data = None
    pkl_data = None
    
    if file_path.endswith(".bvh") or file_path.endswith(".BVH"):
        bvh_data = bvh_tools.load(file_path)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif file_path.endswith(".fbx") or file_path.endswith(".FBX"):
        fbx_data = fbx_tools.load(file_path)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
    elif file_path.endswith(".pkl"):
        pkl_data = load_pkl(file_path)
        mocap_data = pkl_to_mocap(pkl_data)
    
    if bvh_data is not None or fbx_data is not None:
        
        mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
        print("pos_local shape", mocap_data["motion"]["pos_local"].shape)
        #print("rot_local_euler shape", mocap_data["motion"]["rot_local_euler"].shape)

        # set x and z offset of root joint to zero
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
        
    elif pkl_data is not None:
        
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
        mocap_data["motion"]["pos_world"] *= mocap_pos_scale

    if bvh_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif fbx_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif pkl_data is not None:
        remove_root_position(mocap_data, mocap_root_joint_index)
        remove_root_rotation(mocap_data, mocap_root_joint_index)

    return mocap_data

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr

# -------------------------------------------------------
# Model Definitions
# -------------------------------------------------------

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

        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y

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
        
        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)

        x = self.flatten(x)

        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)

        mu = self.fc_mu(x)
        std = self.fc_std(x)

        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

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
        
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y

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
        
        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        
        x = self.unflatten(x)

        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)

        return x

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


class Transformer(nn.Module):
    """Transformer mapping mocap sequence → audio latents."""
    def __init__(self, mocap_dim, audio_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pos_encoding_max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim)
        self.audio2embed = nn.Linear(audio_dim, embed_dim)
        self.positional_encoder = PositionalEncoding(dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        self.embed2audio = nn.Linear(embed_dim, audio_dim)

    def get_src_mask(self, size):
        return torch.zeros(size, size)

    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, mocap_data, audio_data):
        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        mocap_embedded = self.positional_encoder(self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim))
        audio_embedded = self.positional_encoder(self.audio2embed(audio_data) * math.sqrt(self.embed_dim))
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask=tgt_mask)
        return self.embed2audio(decoder_out)
    
# -------------------------------------------------------
# Audio Synthesis Helper Functions
# -------------------------------------------------------

@torch.no_grad()
def encode_audio(audio, vocos_model, encoder_model):
    """Encode waveform to vocos latents."""
    
    #print("audio s ", audio.shape)
    
    vocos_input = audio.reshape(1, 1, -1).float().to(device)
    
    #print("vocos_input s ", vocos_input.shape)

    mel = vocos_model.feature_extractor(vocos_input)
    
    #print("mel s ", mel.shape)

    encoder_input = mel.reshape((1, audio_mel_filter_count, -1, vae_audio_mel_count))
    
    #print("encoder_input s ", encoder_input.shape)

    encoder_input = encoder_input.permute((2, 0, 1, 3))
    
    #print("encoder_input 2 s ", encoder_input.shape)

    mu, std = encoder_model(encoder_input)
    std = torch.nn.functional.softplus(std) + 1e-8
    latent = encoder_model.reparameterize(mu, std)
    
    #print("latent s ", latent.shape)

    return latent

@torch.no_grad()
def decode_audio(latent, vocos_model, decoder_model):
    """Decode vocos latents to waveform."""
    
    decoder_output = decoder_model(latent)
    
    #print("decoder_output s ", decoder_output.shape)
    
    mels = decoder_output.permute((1, 2, 0, 3))

    #print("mels s ", mels.shape)
    
    mels = mels.reshape((1, 1, audio_mel_filter_count, -1))
    
    #print("mels 2 s ", mels.shape)
    
    wav = vocos_model.decode(mels.squeeze(1))
    
    #print("wav s ", wav.shape)
    
    wav = wav.reshape(-1)
    
    #print("wav 2 s ", wav.shape)
    
    return wav

@torch.no_grad()
def audio_synthesis(mocap_seq, audio_latents):
    
    #print("audio_synthesis mocap_seq s ", mocap_seq.shape, " audio_latents s ", audio_latents.shape)
    
    """Predict next audio latents from mocap+current latents."""
    m_in = torch.tensor(mocap_seq, dtype=torch.float32).reshape(mocap_input_seq_length, -1).to(device)
    m_in = (m_in - mocap_mean) / (mocap_std + 1e-8)
    a_in = (audio_latents - audio_latents_mean) / (audio_latents_std + 1e-8)
    m_in = m_in.unsqueeze(0)
    a_in = a_in.unsqueeze(0)
    pred = transformer(m_in, a_in).squeeze(0)
    pred = pred * audio_latents_std + audio_latents_mean

    #print("audio_synth m_in s ", m_in.shape, " a_in s ", a_in.shape, " pred s ", pred.shape)

    return pred

# -------------------------------------------------------
# Load Audio and Mocap Data
# -------------------------------------------------------

mocap_data = load_mocap(mocap_data_file, mocap_pos_scale)
mocap_sequence = mocap_data["motion"]["rot_local"]
mocap_pose_dim = mocap_sequence.shape[1]
mocap_mean = torch.tensor(np.mean(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)
mocap_std = torch.tensor(np.std(mocap_sequence.reshape(mocap_sequence.shape[0], -1), axis=0, keepdims=True), dtype=torch.float32).to(device)
audio_waveform, _ = torchaudio.load(audio_data_file)
audio_waveform = audio_waveform[0]

if "skeleton" in mocap_data:
    mocap_skeleton = mocap_data["skeleton"]
    mocap_root_joint_index = mocap_skeleton["joints"].index(mocap_skeleton["root"])

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------

vocos = Vocos.from_pretrained(vocos_pretrained_config).to(device).eval()

audio_mels_input_sequence = vocos.feature_extractor(torch.rand(size=(1, audio_waveform_input_seq_length), dtype=torch.float32).to(device))
tmp = vocos.decode(audio_mels_input_sequence)

if audio_waveform_input_seq_length != tmp.shape[-1]:
    print("Warning: reconstructing audio waveform from mels does not produce same number as audio samples as in original audio waveform")
    print("orig waveform sample count ", audio_waveform_input_seq_length)
    print("reconsrtucted waveform sample count ", tmp.shape[-1])

audio_mel_input_seq_length = audio_mels_input_sequence.shape[-1]
audio_latents_input_seq_length = audio_mel_input_seq_length // vae_audio_mel_count
audio_mel_filter_count = audio_mels_input_sequence.shape[1]

print("audio_mel_input_seq_length ", audio_mel_input_seq_length)
print("audio_latents_input_seq_length ", audio_latents_input_seq_length)
print("audio_mel_filter_count ", audio_mel_filter_count)

encoder = Encoder(latent_dim, vae_audio_mel_count, audio_mel_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
decoder = Decoder(latent_dim, vae_audio_mel_count, audio_mel_filter_count, list(reversed(vae_conv_channel_counts)), vae_conv_kernel_size, list(reversed(vae_dense_layer_sizes))).to(device)
decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))
audio_latents_mean = torch.load(audio_latents_mean_file).to(device)
audio_latents_std = torch.load(audio_latents_std_file).to(device)

transformer = Transformer(
    mocap_pose_dim,
    latent_dim,
    transformer_embed_dim,
    transformer_head_count,
    transformer_layer_count,
    transformer_layer_count,
    transformer_dropout,
    pos_encoding_max_length
).to(device)

print(transformer)

transformer.load_state_dict(torch.load(transformer_weights_file, map_location=device))

# Precompute audio window
hann_window = torch.from_numpy(np.hanning(audio_window_size)).float().to(device)

# =========================================================
# FIFO Queues
# =========================================================

mocap_queue =queue.Queue(maxsize=max_fifo_queue_length)
audio_queue = queue.Queue(maxsize=max_fifo_queue_length)
export_audio_buffer = []

# =========================================================
# Motion Capture Receiver
# =========================================================

class MocapReceiver:
    """Receives mocap OSC messages and pushes frames to mocap_queue."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/0/joint/rot_local", self.updateMocapQueue)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.thread = None

    def start(self):
        """Start OSC server in background thread."""
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def stop(self):
        """Gracefully shutdown OSC server."""
        self.server.shutdown()
        self.server.server_close()

    def join(self):
        """Wait for server thread to exit."""
        if self.thread is not None:
            self.thread.join()

    def updateMocapQueue(self, address, *args):
        """OSC handler for receiving mocap frames."""
        while mocap_queue.full():
            mocap_queue.get_nowait()  # Drop oldest
        mocap_frame = np.asarray(args, dtype=np.float32)
        
        # set root rotation to zero
        mocap_frame = mocap_frame.reshape((mocap_joint_count, mocap_joint_dim))
        mocap_frame[mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
        mocap_frame = mocap_frame.flatten()
        
        # convert numpy array to torch tensor
        mocap_frame = torch.tensor(mocap_frame, dtype=torch.float32).to(device)

        #print("mocap_frame ", mocap_frame)

        mocap_queue.put(mocap_frame)
    

# =========================================================
# Init Mocap and Audio Context
# =========================================================

input_waveform = audio_waveform[:audio_waveform_input_seq_length].unsqueeze(0).to(device)
input_audio_latents = encode_audio(input_waveform, vocos, encoder)

print("input_audio_latents s ", input_audio_latents.shape)

input_mocap_sequence = torch.zeros((mocap_input_seq_length, mocap_joint_count, mocap_joint_dim),
                                   dtype=torch.float32).to(device)
input_mocap_sequence[:, :, 0] = 1.0  # Example init
input_mocap_sequence = input_mocap_sequence.reshape((mocap_input_seq_length, mocap_pose_dim))

print("input_mocap_sequence s ", input_mocap_sequence.shape)

# debug
test_waveform = decode_audio(input_audio_latents, vocos, decoder)

test_waveform.shape

# =========================================================
# Audio Producer Thread
# =========================================================

shutdown_event = threading.Event()

mocap_idx = 0
save_counter = 0 # debug

def producer_thread():
    """Continuously predicts and enqueues new audio buffers."""
    global mocap_idx, input_audio_latents, input_mocap_sequence
    global save_counter # debug
    while not shutdown_event.is_set():
        if not audio_queue.full():
            input_mocap_sequence = torch.roll(input_mocap_sequence, shifts=-1, dims=0)
            if not mocap_queue.empty():
                input_mocap_sequence[-1] = mocap_queue.get_nowait().to(device)
            #print("input_mocap_sequence s ", input_mocap_sequence.shape)
            #print("input_mocap_sequence[:8] ", input_mocap_sequence[:8])

            output_audio_latents = audio_synthesis(input_mocap_sequence, input_audio_latents)

            #print("output_audio_latents s ", output_audio_latents.shape)

            gen_waveform = decode_audio(output_audio_latents, vocos, decoder)
            

            """
            # debug begin
            if save_counter % 100 == 0 and save_counter < 1000:
                torchaudio.save("test_{}.wav".format(save_counter), gen_waveform.detach().cpu().unsqueeze(0), audio_sample_rate)
            save_counter += 1
            # debug end
            """
            
            #print("gen_waveform s ", gen_waveform.shape)
            
            gen_waveform = gen_waveform[-audio_window_size:]
            
            #print("gen_waveform 2 s ", gen_waveform.shape)
            
            # Optional highpass:
            # gen_waveform = highpass_biquad(gen_waveform, audio_sample_rate, hpf_cutoff_hz)
            audio_queue.put(gen_waveform.cpu().numpy())
            mocap_idx = (mocap_idx + 1) % mocap_sequence.shape[0]
            input_audio_latents = output_audio_latents.detach()
        else:
            time.sleep(0.01)  # Avoid busy-waiting

# =========================================================
# AUDIO CALLBACK (playback)
# =========================================================

# This must be maintained across audio_callback calls:
last_chunk = np.zeros(audio_window_size, dtype=np.float32)

def audio_callback(out_data, frames, time_info, status):
        
    #print("audio_callback")
    
    """Overlap-add from audio_queue to output audio buffer."""
    global export_audio_buffer, last_chunk
    output = np.zeros((frames, audio_channels), dtype=np.float32)
    cursor = 0
  
    # Start with second half of last block from previous callback
    #output[cursor:cursor+audio_window_offset, 0] += last_chunk[-audio_window_offset:]
    output[cursor:cursor+audio_window_size // 2, 0] += last_chunk[-audio_window_size // 2:]

    while cursor < frames:
        
        #print("cursor ", cursor, " frames ", frames, " queue ", audio_queue.qsize())
        
        try:
            chunk = audio_queue.get_nowait()
            chunk = chunk * hann_window.cpu().numpy()
        except queue.Empty:
            chunk = np.zeros(audio_window_size, dtype=np.float32)  # Output silence
        chunk_size = output[cursor:cursor+audio_window_size, 0].shape[0]
        output[cursor:cursor+chunk_size, 0] += chunk[:chunk_size]
        
        cursor += audio_window_offset
        last_chunk[:] = chunk[:]

    out_data[:] = output
    export_audio_buffer.append(output[:, 0])

# =========================================================
# MAIN RUNTIME
# =========================================================
if __name__ == "__main__":
    
    mocap_receiver = MocapReceiver(osc_receive_ip, osc_receive_port)

    mocap_receiver.start()
    threading.Thread(target=producer_thread, daemon=True).start()

    sd.sleep(2000)  # Allow queue to prefill

    with sd.OutputStream(
        samplerate=audio_sample_rate,
        device=audio_output_device,
        channels=audio_channels,
        callback=audio_callback,
        blocksize=play_buffer_size,
        latency=playback_latency
    ):
        print("Streaming audio... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopping...")
            shutdown_event.set()
            mocap_receiver.stop()
            mocap_receiver.join()

    # Save streamed audio to WAV
    export_audio_buffer_np = np.concatenate(export_audio_buffer, axis=0)
    export_audio_buffer_tensor = torch.from_numpy(export_audio_buffer_np).unsqueeze(0)
    torchaudio.save("audio_export.wav", export_audio_buffer_tensor, audio_sample_rate)