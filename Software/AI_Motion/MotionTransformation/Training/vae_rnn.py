# -------------------------------------------------------------------------------------------------
# Motion Transformation - Training Script
# Recurrent Variational Beta-Autoencoder with modern motion data handling
# Keeps the LSTM-based model architecture, but uses the CNN script's motion I/O and 6D pipeline
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
import time
import json
import numpy as np

import torch
import torch.nn.functional as nnF
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import OrderedDict

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer

from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

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
mocap_valid_time_ranges = [None] * len(mocap_files) # in seconds
mocap_topology_files = [None] * len(mocap_files)  # only used for .npz files
mocap_pos_scale = 1.0
mocap_fps = 50


"""
# Example 2: BVH
mocap_file_path = "../../../Data/Mocap/Xsens/Stocos/Solos/bvh_50hz/"
mocap_files = ["Muriel_Take1_double_Bind.bvh"]
mocap_valid_time_ranges = [None]
mocap_topology_files = [None]
mocap_pos_scale = 1.0
mocap_fps = 50
"""

"""
# Example 3: NPZ
mocap_file_path = "E:/data/mocap/Yurika/Mediapipe_v2/All/"
mocap_files = [
    "Yurika_Everyday_Mediapipe_realtime.npz",
    "Yurika_Geometry_Mediapipe_realtime.npz",
    "Yurika_Rythm_Mediapipe_realtime.npz"
]
mocap_valid_time_ranges = [None, None, None]
mocap_topology_files = [
    "data/configs/Mediapipe_config.json",
    "data/configs/Mediapipe_config.json",
    "data/configs/Mediapipe_config.json"
]
mocap_pos_scale = 100.0
mocap_fps = 30
"""

mocap_loss_weights_file = None
train_root_trajectory = False

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------

save_path = "results/Muriel_Blumen_RO_fbx_5/"
save_weights_path = os.path.join(save_path, "weights/")
save_history_path = os.path.join(save_path, "history/")
save_anims_path = os.path.join(save_path, "anims/")
save_anim_formats = ["gif", "fbx"] 

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_anims_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

latent_dim = 16
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [512]
ae_dropout = 0.1

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

mocap_window_length = 64
mocap_window_offset = 1
test_percentage = 0.1
batch_size = 128
epochs = 200

ae_learning_rate = 1e-3
learning_rate_cycle_decay = 0.5

pos_rec_loss_scale = 0.9
rot_rec_loss_scale = 1.0

target_beta = 0.25
target_beta_cycles = 4
target_beta_ratio = 0.5
kl_free_bits = 0.5

save_weights = True
save_weights_interval = 50

load_weights = False
load_weights_file = "results/Muriel_Blumen_RO_fbx/weights/vae_weight_epoch_200.pt"

# -------------------------------------------------------------------------------------------------
# Render Settings
# -------------------------------------------------------------------------------------------------

view_ele = 0.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

# -------------------------------------------------------------------------------------------------
# Load Mocap Data
# -------------------------------------------------------------------------------------------------

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
npz_tools = npz.NPZ_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

if len(mocap_topology_files) != len(mocap_files):
    raise ValueError("mocap_topology_files must have the same length as mocap_files")

for i, mocap_file in enumerate(mocap_files):
    print("process file", mocap_file)

    valid_time_ranges = mocap_valid_time_ranges[i]
    mocap_abs_path = os.path.join(mocap_file_path, mocap_file)
    file_ext = os.path.splitext(mocap_file)[1].lower()

    if file_ext == ".bvh":
        bvh_data = bvh_tools.load(mocap_abs_path)
        mocap_data_raw = mocap_tools.bvh_to_mocap(bvh_data)
        segments = mocap_tools.resample_euler_mocap_data(mocap_data_raw, mocap_fps, valid_time_ranges)

        for segment in segments:
            segment["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(
                segment["motion"]["rot_local_euler"],
                segment["rot_sequence"]
            )
            all_mocap_data.append(segment)

    elif file_ext == ".fbx":
        fbx_data = fbx_tools.load(mocap_abs_path)
        mocap_data_raw = mocap_tools.fbx_to_mocap(fbx_data)[0]
        segments = mocap_tools.resample_euler_mocap_data(mocap_data_raw, mocap_fps, valid_time_ranges)

        for segment in segments:
            segment["motion"]["rot_local"] = mocap_tools.euler_to_quat(
                segment["motion"]["rot_local_euler"],
                segment["rot_sequence"]
            )
            all_mocap_data.append(segment)

    elif file_ext == ".npz":
        topology_file = mocap_topology_files[i]
        if topology_file is None:
            raise ValueError(
                f"NPZ file '{mocap_file}' requires a topology JSON file path in mocap_topology_files"
            )

        npz_data, topo_data = npz_tools.load(mocap_abs_path, topology_file)
        segments = mocap_tools.npz_to_mocap(npz_data, topo_data, mocap_fps)

        for segment in segments:
            all_mocap_data.append(segment)

    else:
        raise ValueError(f"Unsupported mocap format: {mocap_file}")

if len(all_mocap_data) == 0:
    raise ValueError("No mocap data loaded.")

for mocap_data in all_mocap_data:
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale

    if not train_root_trajectory:
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 0] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 2] = 0.0

    mocap_data["motion"]["rot_local"] = rot_np.quat_to_r6d(mocap_data["motion"]["rot_local"])

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = 6
pose_rot_dim = joint_count * joint_dim
vae_input_dim = pose_rot_dim + (3 if train_root_trajectory else 0)

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

def get_edge_list(children_):
    edge_list_ = []
    for parent_joint_index in range(len(children_)):
        for child_joint_index in children_[parent_joint_index]:
            edge_list_.append([parent_joint_index, child_joint_index])
    return edge_list_

def get_execution_order(parents_):
    execution_order_ = []
    roots = [j for j, p in enumerate(parents_) if p == -1]
    queue = roots[:]
    while queue:
        j = queue.pop(0)
        execution_order_.append(j)
        for c in range(len(parents_)):
            if parents_[c] == j:
                queue.append(c)
    return execution_order_

edge_list = get_edge_list(children)
execution_order = get_execution_order(parents)

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file, "r") as f:
        joint_loss_weights = json.load(f)["joint_loss_weights"]
else:
    joint_loss_weights = [1.0] * joint_count

# -------------------------------------------------------------------------------------------------
# Create Dataset
# -------------------------------------------------------------------------------------------------

pose_sequence_excerpts = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_local"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_rot_dim))

    if train_root_trajectory:
        root_positions = mocap_data["motion"]["pos_local"][:, 0, :]
        pose_sequence = np.concatenate((root_positions, pose_sequence), axis=1)

    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]

    for seq_excerpt_start in np.arange(
        frame_range_start,
        frame_range_end - mocap_window_length,
        mocap_window_offset
    ):
        excerpt = pose_sequence[seq_excerpt_start:seq_excerpt_start + mocap_window_length]
        pose_sequence_excerpts.append(excerpt)

pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)

root_pos_mean_tensor = None
root_pos_std_tensor = None
root_pos_mean = None
root_pos_std = None

if train_root_trajectory:
    root_traj_excerpts = pose_sequence_excerpts[:, :, :3]
    root_pos_mean = np.mean(root_traj_excerpts, axis=(0, 1), keepdims=True)
    root_pos_std = np.std(root_traj_excerpts, axis=(0, 1), keepdims=True)
    root_pos_std[root_pos_std == 0] = 1.0

    pose_sequence_excerpts[:, :, :3] = (root_traj_excerpts - root_pos_mean) / root_pos_std

    root_pos_mean_tensor = torch.from_numpy(root_pos_mean.astype(np.float32)).to(device)
    root_pos_std_tensor = torch.from_numpy(root_pos_std.astype(np.float32)).to(device)

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts

    def __len__(self):
        return self.sequence_excerpts.shape[0]

    def __getitem__(self, idx):
        return self.sequence_excerpts[idx, ...]

full_dataset = SequenceDataset(pose_sequence_excerpts)

dataset_size = len(full_dataset)
if dataset_size < 2:
    raise ValueError("Need at least 2 sequence excerpts for train/test split.")

test_size = max(1, int(test_percentage * dataset_size))
train_size = dataset_size - test_size
if train_size < 1:
    train_size = dataset_size - 1
    test_size = 1

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------------------------------
# Create Model
# -------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout=0.3):
        super().__init__()
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size
        self.dense_layer_sizes = dense_layer_sizes

        self.rnn = nn.LSTM(
            input_size=self.pose_dim,
            hidden_size=self.rnn_layer_size,
            num_layers=self.rnn_layer_count,
            batch_first=True,
            dropout=dropout if self.rnn_layer_count > 1 else 0.0,
        )

        dense_layers = []
        in_dim = self.rnn_layer_size
        for layer_index, out_dim in enumerate(self.dense_layer_sizes):
            dense_layers.append((f"encoder_dense_{layer_index}", nn.Linear(in_dim, out_dim)))
            dense_layers.append((f"encoder_relu_{layer_index}", nn.ReLU()))
            dense_layers.append((f"encoder_dropout_{layer_index}", nn.Dropout(dropout)))
            in_dim = out_dim

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers)) if dense_layers else nn.Identity()
        last_dim = self.dense_layer_sizes[-1] if len(self.dense_layer_sizes) > 0 else self.rnn_layer_size
        self.fc_mu = nn.Linear(last_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(last_dim, self.latent_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.dense_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout=0.3):
        super().__init__()
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        dense_layers = []
        in_dim = self.latent_dim
        for layer_index, out_dim in enumerate(self.dense_layer_sizes):
            dense_layers.append((f"decoder_dense_{layer_index}", nn.Linear(in_dim, out_dim)))
            dense_layers.append((f"decoder_relu_{layer_index}", nn.ReLU()))
            dense_layers.append((f"decoder_dropout_{layer_index}", nn.Dropout(dropout)))
            in_dim = out_dim

        self.dense_layers = nn.Sequential(OrderedDict(dense_layers)) if dense_layers else nn.Identity()
        decoder_input_dim = self.dense_layer_sizes[-1] if len(self.dense_layer_sizes) > 0 else self.latent_dim

        self.rnn = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=self.rnn_layer_size,
            num_layers=self.rnn_layer_count,
            batch_first=True,
            dropout=dropout if self.rnn_layer_count > 1 else 0.0,
        )
        self.output_layer = nn.Linear(self.rnn_layer_size, self.pose_dim)

    def forward(self, z):
        x = self.dense_layers(z)
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.rnn(x)
        yhat = self.output_layer(x)
        return yhat

class RecurrentMotionVAE(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout=0.3):
        super().__init__()
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            sequence_length, pose_dim, latent_dim,
            rnn_layer_count, rnn_layer_size, dense_layer_sizes, dropout
        )
        self.decoder = Decoder(
            sequence_length, pose_dim, latent_dim,
            rnn_layer_count, rnn_layer_size, list(reversed(dense_layer_sizes)), dropout
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        yhat = self.decode(z)
        return yhat, mu, logvar

vae = RecurrentMotionVAE(
    sequence_length=mocap_window_length,
    pose_dim=vae_input_dim,
    latent_dim=latent_dim,
    rnn_layer_count=ae_rnn_layer_count,
    rnn_layer_size=ae_rnn_layer_size,
    dense_layer_sizes=ae_dense_layer_sizes,
    dropout=ae_dropout,
).to(device)

if load_weights:
    if not load_weights_file:
        raise ValueError("load_weights is True but load_weights_file is empty.")
    vae.load_state_dict(torch.load(load_weights_file, map_location=device))
    print(f"Loaded weights from {load_weights_file}")

# -------------------------------------------------------------------------------------------------
# Losses and Kinematics
# -------------------------------------------------------------------------------------------------

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32).reshape(1, 1, -1).to(device)

def kl_loss(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()

def split_root_and_rot_torch(x):
    if train_root_trajectory:
        root_traj = x[:, :, :3]
        rot_6d = x[:, :, 3:].reshape(x.shape[0], x.shape[1], joint_count, 6)
    else:
        root_traj = torch.zeros((x.shape[0], x.shape[1], 3), dtype=x.dtype, device=x.device)
        rot_6d = x.reshape(x.shape[0], x.shape[1], joint_count, 6)
    return root_traj, rot_6d

def forward_kinematics(rotation_matrices, root_positions):
    toffsets = torch.tensor(offsets, dtype=torch.float32, device=rotation_matrices.device)
    expanded_offsets = toffsets.expand(
        rotation_matrices.shape[0],
        rotation_matrices.shape[1],
        offsets.shape[0],
        offsets.shape[1]
    )

    num_joints = offsets.shape[0]
    positions_world = [None] * num_joints
    rotations_world = [None] * num_joints

    for jI in execution_order:
        if parents[jI] == -1:
            positions_world[jI] = root_positions
            rotations_world[jI] = rotation_matrices[:, :, jI]
        else:
            parent_idx = parents[jI]
            parent_rot = rotations_world[parent_idx]
            parent_pos = positions_world[parent_idx]
            local_offset = expanded_offsets[:, :, jI].unsqueeze(2)

            rotated_offset = torch.matmul(local_offset, parent_rot).squeeze(2)
            positions_world[jI] = rotated_offset + parent_pos

            local_rot = rotation_matrices[:, :, jI]
            rotations_world[jI] = torch.matmul(local_rot, parent_rot)

    return torch.stack(positions_world, dim=2)

def pos_rec_loss(y, yhat):
    y_root, y_rot6d = split_root_and_rot_torch(y)
    yhat_root, yhat_rot6d = split_root_and_rot_torch(yhat)

    if train_root_trajectory:
        y_root = y_root * root_pos_std_tensor + root_pos_mean_tensor
        yhat_root = yhat_root * root_pos_std_tensor + root_pos_mean_tensor

    y_mat = rot_to.r6d_to_mat(y_rot6d)
    yhat_mat = rot_to.r6d_to_mat(yhat_rot6d)

    y_pos = forward_kinematics(y_mat, y_root)
    yhat_pos = forward_kinematics(yhat_mat, yhat_root)

    pos_diff = torch.norm(y_pos - yhat_pos, dim=3)
    return torch.mean(pos_diff * joint_loss_weights)

def rot_rec_loss(y, yhat):
    _, y_rot6d = split_root_and_rot_torch(y)
    _, yhat_rot6d = split_root_and_rot_torch(yhat)

    y_mat = rot_to.r6d_to_mat(y_rot6d)
    yhat_mat = rot_to.r6d_to_mat(yhat_rot6d)

    rel = torch.matmul(y_mat.transpose(-1, -2), yhat_mat)
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -0.9999, 0.9999))

    return torch.mean(angle * joint_loss_weights)

def ae_loss(y, yhat, mu, logvar, beta):
    pos_loss = pos_rec_loss(y, yhat)
    rot_loss = rot_rec_loss(y, yhat)
    kld = kl_loss(mu, logvar, free_bits=kl_free_bits)

    total_loss = (pos_loss * pos_rec_loss_scale) + (rot_loss * rot_rec_loss_scale) + (kld * beta)

    if train_root_trajectory:
        traj_mse = torch.mean((y[:, :, :3] - yhat[:, :, :3]) ** 2)
        total_loss = total_loss + (traj_mse * pos_rec_loss_scale)

    return total_loss, pos_loss, rot_loss, kld

# -------------------------------------------------------------------------------------------------
# Training Logic
# -------------------------------------------------------------------------------------------------

def create_beta_schedule(epochs_, target_beta_=0.25, cycles=4, ratio=0.5):
    schedule = []
    period = epochs_ / cycles
    for epoch in range(epochs_):
        cycle_step = epoch % period
        phase = cycle_step / period
        if phase < ratio:
            beta = target_beta_ * (phase / ratio)
        else:
            beta = target_beta_
        schedule.append(beta)
    return schedule

beta_schedule = create_beta_schedule(
    epochs_=epochs,
    target_beta_=target_beta,
    cycles=target_beta_cycles,
    ratio=target_beta_ratio
)

optimizer = torch.optim.AdamW(vae.parameters(), lr=ae_learning_rate, weight_decay=1e-4)
cycle_length = max(1, int(epochs / target_beta_cycles))
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cycle_length, T_mult=1, eta_min=1e-6)

def vae_train_step(y, beta):
    vae.train()
    yhat, mu, logvar = vae(y)
    loss, pos_loss, rot_loss, kld = ae_loss(y, yhat, mu, logvar, beta)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
    optimizer.step()

    return loss, pos_loss, rot_loss, kld

@torch.no_grad()
def vae_test_step(y, beta):
    vae.eval()
    yhat, mu, logvar = vae(y)
    loss, pos_loss, rot_loss, kld = ae_loss(y, yhat, mu, logvar, beta)
    return loss, pos_loss, rot_loss, kld

def train(train_dataloader_, test_dataloader_, epochs_):
    loss_history = {"lr": [], "beta": [], "train": [], "test": [], "pos": [], "rot": [], "kl": []}

    for epoch in range(epochs_):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        current_beta = beta_schedule[epoch] if epoch < len(beta_schedule) else beta_schedule[-1]

        loss_history["lr"].append(current_lr)
        loss_history["beta"].append(current_beta)

        train_loss_per_epoch = []
        test_loss_per_epoch = []
        pos_loss_per_epoch = []
        rot_loss_per_epoch = []
        kl_loss_per_epoch = []

        for train_batch in train_dataloader_:
            train_batch = train_batch.to(device)
            loss, pos_loss, rot_loss, kld = vae_train_step(train_batch, current_beta)

            train_loss_per_epoch.append(loss.detach().cpu().numpy())
            pos_loss_per_epoch.append(pos_loss.detach().cpu().numpy())
            rot_loss_per_epoch.append(rot_loss.detach().cpu().numpy())
            kl_loss_per_epoch.append(kld.detach().cpu().numpy())

        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        pos_loss_per_epoch = np.mean(np.array(pos_loss_per_epoch))
        rot_loss_per_epoch = np.mean(np.array(rot_loss_per_epoch))
        kl_loss_per_epoch = np.mean(np.array(kl_loss_per_epoch))

        for test_batch in test_dataloader_:
            test_batch = test_batch.to(device)
            loss, _, _, _ = vae_test_step(test_batch, current_beta)
            test_loss_per_epoch.append(loss.detach().cpu().numpy())

        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))

        if save_weights and ((epoch + 1) % save_weights_interval == 0):
            torch.save(vae.state_dict(), f"{save_weights_path}vae_weight_epoch_{epoch + 1}.pt")

        loss_history["train"].append(train_loss_per_epoch)
        loss_history["test"].append(test_loss_per_epoch)
        loss_history["pos"].append(pos_loss_per_epoch)
        loss_history["rot"].append(rot_loss_per_epoch)
        loss_history["kl"].append(kl_loss_per_epoch)

        print(
            f"epoch: {epoch + 1:03d} | "
            f"train: {train_loss_per_epoch:01.4f} | "
            f"test: {test_loss_per_epoch:01.4f} | "
            f"pos: {pos_loss_per_epoch:01.4f} | "
            f"rot: {rot_loss_per_epoch:01.4f} | "
            f"kl: {kl_loss_per_epoch:01.4f} | "
            f"time: {(time.time() - start):01.2f}"
        )

        if epoch > 0 and ((epoch + 1) % cycle_length == 0):
            scheduler.base_lrs = [base_lr * learning_rate_cycle_decay for base_lr in scheduler.base_lrs]
        scheduler.step(epoch + 1)

    return loss_history

def plot_training_history(loss_history, save_path):
    epochs_ = len(loss_history["train"])
    x = np.arange(1, epochs_ + 1)

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Recurrent VAE Training History", fontsize=18, y=0.98)

    axs[0, 0].plot(x, loss_history["train"], label="Train Loss", color="tab:blue", linewidth=2)
    axs[0, 0].plot(x, loss_history["test"], label="Test Loss", color="tab:orange", linewidth=2)
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(x, loss_history["pos"], label="Pos Rec", color="tab:green", linewidth=2)
    axs[0, 1].plot(x, loss_history["rot"], label="Rot Rec", color="tab:red", linewidth=2)
    axs[0, 1].set_title("Reconstruction Losses")
    axs[0, 1].legend()

    axs[1, 0].plot(x, loss_history["kl"], label="KL Loss", color="tab:brown", linewidth=2)
    axs[1, 0].set_title("KL Divergence")
    axs[1, 0].legend()

    axs[1, 1].plot(x, loss_history["beta"], label="Beta", color="tab:pink", linewidth=2)
    axs[1, 1].set_title("Beta Schedule")
    axs[1, 1].legend()

    axs[2, 0].plot(x, loss_history["lr"], label="Learning Rate", color="tab:cyan", linewidth=2)
    axs[2, 0].set_title("Learning Rate")
    axs[2, 0].legend()

    axs[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------------------------------------------
# Inference Helpers
# -------------------------------------------------------------------------------------------------

@torch.no_grad()
def encode_sequences(orig_sequence, frame_indices):
    vae.eval()
    latent_vectors = []

    for excerpt_start_frame in frame_indices:
        excerpt_end_frame = excerpt_start_frame + mocap_window_length
        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame].copy()

        if excerpt.shape[0] != mocap_window_length:
            continue

        if train_root_trajectory:
            excerpt[:, :3] = (excerpt[:, :3] - root_pos_mean.reshape(1, 3)) / root_pos_std.reshape(1, 3)

        excerpt = np.expand_dims(excerpt, axis=0).astype(np.float32)
        excerpt = torch.from_numpy(excerpt).to(device)

        mu, logvar = vae.encode(excerpt)
        latent_vector = vae.reparameterize(mu, logvar)
        latent_vectors.append(torch.squeeze(latent_vector).detach().cpu().numpy())

    vae.train()
    return latent_vectors

@torch.no_grad()
def decode_sequence_encodings(sequence_encodings, seq_overlap, base_pose=None):
    vae.eval()

    seq_env = np.hanning(mocap_window_length).astype(np.float32) + 0.01
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + mocap_window_length

    matrix_accum = np.zeros((gen_seq_length, joint_count, 3, 3), dtype=np.float32)
    weight_accum = np.zeros((gen_seq_length, 1, 1), dtype=np.float32)

    if train_root_trajectory:
        root_accum = np.zeros((gen_seq_length, 3), dtype=np.float32)
        root_weight_accum = np.zeros((gen_seq_length, 1), dtype=np.float32)

    for excerpt_index, latent_vector in enumerate(sequence_encodings):
        latent_vector = np.expand_dims(latent_vector, axis=0).astype(np.float32)
        latent_vector = torch.from_numpy(latent_vector).to(device)

        excerpt_dec = vae.decode(latent_vector).squeeze(0).detach().cpu().numpy()

        if train_root_trajectory:
            root_traj_dec = excerpt_dec[:, :3]
            excerpt_rot = excerpt_dec[:, 3:]
        else:
            excerpt_rot = excerpt_dec

        excerpt_rot = np.reshape(excerpt_rot, (-1, joint_count, 6))
        excerpt_rot_tensor = torch.from_numpy(excerpt_rot.astype(np.float32))
        excerpt_mats = rot_to.r6d_to_mat(excerpt_rot_tensor).cpu().numpy()

        gen_frame = excerpt_index * seq_overlap

        for si in range(mocap_window_length):
            mix_weight = seq_env[si]
            matrix_accum[gen_frame + si] += excerpt_mats[si] * mix_weight
            weight_accum[gen_frame + si] += mix_weight

            if train_root_trajectory:
                root_accum[gen_frame + si] += root_traj_dec[si] * mix_weight
                root_weight_accum[gen_frame + si] += mix_weight

    weight_accum[weight_accum == 0.0] = 1.0
    avg_matrices = matrix_accum / np.expand_dims(weight_accum, axis=-1)

    x_avg = avg_matrices[:, :, :, 0]
    y_avg = avg_matrices[:, :, :, 1]
    rot6d = np.concatenate((x_avg, y_avg), axis=-1).reshape(gen_seq_length, -1)

    if train_root_trajectory:
        root_weight_accum[root_weight_accum == 0.0] = 1.0
        avg_root = root_accum / root_weight_accum
        avg_root = avg_root * root_pos_std.reshape(1, 3) + root_pos_mean.reshape(1, 3)
        gen_sequence = np.concatenate((avg_root, rot6d), axis=1)
    else:
        gen_sequence = rot6d

    vae.train()
    return gen_sequence.astype(np.float32)

def create_2d_latent_space_representation(sequence_excerpts):
    encodings = []
    excerpt_count = sequence_excerpts.shape[0]

    for eI in range(0, excerpt_count, batch_size):
        excerpt_batch = sequence_excerpts[eI:eI + batch_size]
        excerpt_batch = torch.from_numpy(excerpt_batch).to(device)

        with torch.no_grad():
            mu, logvar = vae.encode(excerpt_batch)
            encoding_batch = vae.reparameterize(mu, logvar)

        encodings.append(encoding_batch.detach().cpu())

    encodings = torch.cat(encodings, dim=0).numpy()
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)
    z_tsne = tsne.fit_transform(encodings)
    return z_tsne

def create_2d_latent_space_image(z_tsne, highlight_excerpt_ranges, file_name):
    z_tsne_x = z_tsne[:, 0]
    z_tsne_y = z_tsne[:, 1]

    plot_colors = ["green", "red", "blue", "magenta", "orange"]
    plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(z_tsne_x, z_tsne_y, s=0.1, c="grey", alpha=0.5)

    for hI, hR in enumerate(highlight_excerpt_ranges):
        ax.scatter(z_tsne_x[hR[0]:hR[1]], z_tsne_y[hR[0]:hR[1]], s=0.8, c=plot_colors[hI], alpha=0.5)

    ax.set_xlabel("c1")
    ax.set_ylabel("c2")
    fig.savefig(file_name, dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Export Helpers
# -------------------------------------------------------------------------------------------------

poseRenderer = PoseRenderer(edge_list)

def split_root_and_rot_numpy(pose_sequence):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence
    return root_trajectory, rot_sequence

def export_sequence_anim(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    root_trajectory, rot_sequence = split_root_and_rot_numpy(pose_sequence)

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    rot_quat = rot_np.r6d_to_quat(rot_seq_6d)

    pos_local = np.repeat(np.expand_dims(offsets, axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory

    pos_world, _ = mocap_tools.local_to_world(rot_quat, pos_local, mocap_data["skeleton"])

    theta = np.radians(90.0)
    cost, sint = np.cos(theta), np.sin(theta)
    rotx_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cost, -sint],
        [0.0, sint, cost]
    ], dtype=np.float32)

    skel_sequence_vis = np.dot(pos_world, rotx_mat.T)
    skel_sequence_vis[..., 0] *= -1.0

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_vis)
    skel_images = poseRenderer.create_pose_images(
        skel_sequence_vis,
        view_min,
        view_max,
        view_ele,
        view_azi,
        view_line_width,
        view_size,
        view_size
    )
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    root_trajectory, rot_sequence = split_root_and_rot_numpy(pose_sequence)

    pred_dataset = {
        "frame_rate": mocap_data.get("frame_rate", mocap_fps),
        "rot_sequence": mocap_data.get("rot_sequence", [0, 1, 2]),
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(
        pred_dataset["motion"]["rot_local"],
        pred_dataset["rot_sequence"]
    )

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    root_trajectory, rot_sequence = split_root_and_rot_numpy(pose_sequence)

    pred_dataset = {
        "frame_rate": mocap_data.get("frame_rate", mocap_fps),
        "rot_sequence": mocap_data.get("rot_sequence", [0, 1, 2]),
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(
        pred_dataset["motion"]["rot_local"],
        pred_dataset["rot_sequence"]
    )

    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    fbx_tools.write(pred_fbx, file_name)

def export_sequence_npz(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    root_trajectory, rot_sequence = split_root_and_rot_numpy(pose_sequence)

    pred_dataset = {
        "frame_rate": mocap_data.get("frame_rate", mocap_fps),
        "rot_sequence": mocap_data.get("rot_sequence", [0, 1, 2]),
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory:
        pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)

    npz_dict = mocap_tools.mocap_to_npz([pred_dataset])
    np.savez_compressed(file_name, **npz_dict)

# -------------------------------------------------------------------------------------------------
# Train / Load
# -------------------------------------------------------------------------------------------------

if not load_weights:
    loss_history = train(train_dataloader, test_dataloader, epochs)
    utils.save_loss_as_csv(loss_history, f"{save_history_path}history_{epochs}.csv")
    plot_training_history(loss_history, f"{save_history_path}history_{epochs}.png")

    if save_weights:
        torch.save(vae.state_dict(), f"{save_weights_path}vae_weight_epoch_{epochs}.pt")

# -------------------------------------------------------------------------------------------------
# Build Original Sequence
# -------------------------------------------------------------------------------------------------

orig_rot = all_mocap_data[0]["motion"]["rot_local"].astype(np.float32)
orig_rot = np.reshape(orig_rot, (-1, pose_rot_dim))

if train_root_trajectory:
    orig_pos = all_mocap_data[0]["motion"]["pos_local"][:, 0, :].astype(np.float32)
    orig_sequence = np.concatenate((orig_pos, orig_rot), axis=1)
else:
    orig_sequence = orig_rot

# -------------------------------------------------------------------------------------------------
# Create Latent Space Plot
# -------------------------------------------------------------------------------------------------

z_tsne = create_2d_latent_space_representation(pose_sequence_excerpts)
create_2d_latent_space_image(z_tsne, [], f"{save_path}latent_space_plot_epoch_{epochs}.png")

# -------------------------------------------------------------------------------------------------
# Export Original Sequence
# -------------------------------------------------------------------------------------------------

seq_start = 1000
seq_length = 1000

if "gif" in save_anim_formats:
    export_sequence_anim(
        orig_sequence[seq_start:seq_start + seq_length],
        f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.gif"
    )
if "fbx" in save_anim_formats:
    export_sequence_fbx(
        orig_sequence[seq_start:seq_start + seq_length],
        f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.fbx"
    )
if "bvh" in save_anim_formats:
    export_sequence_bvh(
        orig_sequence[seq_start:seq_start + seq_length],
        f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.bvh"
    )
if "npz" in save_anim_formats:
    export_sequence_npz(
        orig_sequence[seq_start:seq_start + seq_length],
        f"{save_anims_path}orig_sequence_seq_start_{seq_start}_length_{seq_length}.npz"
    )

# -------------------------------------------------------------------------------------------------
# Reconstruct Original Sequence
# -------------------------------------------------------------------------------------------------

seq_start = 1000
seq_length = 1000
seq_overlap = mocap_window_length // 4
base_pose = orig_sequence[0].copy()

seq_indices = [frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)

if "gif" in save_anim_formats:
    export_sequence_anim(
        gen_sequence,
        f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif"
    )
if "fbx" in save_anim_formats:
    export_sequence_fbx(
        gen_sequence,
        f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx"
    )
if "bvh" in save_anim_formats:
    export_sequence_bvh(
        gen_sequence,
        f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.bvh"
    )
if "npz" in save_anim_formats:
    export_sequence_npz(
        gen_sequence,
        f"{save_anims_path}rec_sequences_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.npz"
    )

# -------------------------------------------------------------------------------------------------
# Random Walk in Latent Space
# -------------------------------------------------------------------------------------------------

seq_start = 1000
seq_length = 1000

seq_indices = [seq_start]
seq_encodings = encode_sequences(orig_sequence, seq_indices)

for index in range(0, seq_length // seq_overlap):
    random_step = (np.random.random((latent_dim)).astype(np.float32) - 0.5) * 2.0
    seq_encodings.append(seq_encodings[index] + random_step)

gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)

if "gif" in save_anim_formats:
    export_sequence_anim(
        gen_sequence,
        f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif"
    )
if "fbx" in save_anim_formats:
    export_sequence_fbx(
        gen_sequence,
        f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx"
    )
if "bvh" in save_anim_formats:
    export_sequence_bvh(
        gen_sequence,
        f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.bvh"
    )
if "npz" in save_anim_formats:
    export_sequence_npz(
        gen_sequence,
        f"{save_anims_path}seq_randwalk_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.npz"
    )

# -------------------------------------------------------------------------------------------------
# Sequence Offset Following
# -------------------------------------------------------------------------------------------------

seq_start = 1000
seq_length = 1000

seq_indices = [seq_index for seq_index in range(seq_start, seq_start + seq_length, seq_overlap)]
seq_encodings = encode_sequences(orig_sequence, seq_indices)

offset_seq_encodings = []
for index in range(len(seq_encodings)):
    sin_value = np.sin(index / max(1, (len(seq_encodings) - 1)) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_seq_encoding = seq_encodings[index] + offset
    offset_seq_encodings.append(offset_seq_encoding)

gen_sequence = decode_sequence_encodings(offset_seq_encodings, seq_overlap, base_pose)

if "gif" in save_anim_formats:
    export_sequence_anim(
        gen_sequence,
        f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.gif"
    )
if "fbx" in save_anim_formats:
    export_sequence_fbx(
        gen_sequence,
        f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.fbx"
    )
if "bvh" in save_anim_formats:
    export_sequence_bvh(
        gen_sequence,
        f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.bvh"
    )
if "npz" in save_anim_formats:
    export_sequence_npz(
        gen_sequence,
        f"{save_anims_path}seq_offset_epoch_{epochs}_seq_start_{seq_start}_length_{seq_length}.npz"
    )

# -------------------------------------------------------------------------------------------------
# Interpolate Two Original Sequences
# -------------------------------------------------------------------------------------------------

seq1_start = 0
seq2_start = 1000
seq_length = 1000

seq1_indices = [seq_index for seq_index in range(seq1_start, seq1_start + seq_length, seq_overlap)]
seq2_indices = [seq_index for seq_index in range(seq2_start, seq2_start + seq_length, seq_overlap)]

seq1_encodings = encode_sequences(orig_sequence, seq1_indices)
seq2_encodings = encode_sequences(orig_sequence, seq2_indices)

mix_encodings = []
mix_count = min(len(seq1_encodings), len(seq2_encodings))

for index in range(mix_count):
    mix_factor = index / max(1, (mix_count - 1))
    mix_encoding = seq1_encodings[index] * (1.0 - mix_factor) + seq2_encodings[index] * mix_factor
    mix_encodings.append(mix_encoding)

gen_sequence = decode_sequence_encodings(mix_encodings, seq_overlap, base_pose)

if "gif" in save_anim_formats:
    export_sequence_anim(
        gen_sequence,
        f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.gif"
    )
if "fbx" in save_anim_formats:
    export_sequence_fbx(
        gen_sequence,
        f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.fbx"
    )
if "bvh" in save_anim_formats:
    export_sequence_bvh(
        gen_sequence,
        f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.bvh"
    )
if "npz" in save_anim_formats:
    export_sequence_npz(
        gen_sequence,
        f"{save_anims_path}seq_mix_epoch_{epochs}_seq1_start_{seq1_start}_seq2_start_{seq2_start}_length_{seq_length}.npz"
    )

print("Done.")