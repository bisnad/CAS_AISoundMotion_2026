"""
Motion Capture Classification using Structure-Aware LSTM
Reads NPZ-Format motion data 
"""

"""
Imports
"""

import os, time
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Mocap imports
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap

"""
Settings
"""

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Mocap Settings
"""

# XSens NPZ
mocap_data_file_path = "E:/Data/mocap/ick3/Solos/Ricardo/Xsens/npz/Classes"
mocap_data_file_extensions = [".npz"] 
mocap_topology_file = "data/configs/XSensLive_config.json"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_fps = 60
mocap_joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # skeleton without root
mocap_data_window_length = 90
mocap_data_window_offset = 15
mocap_pos_scale = 1.0
mocap_stats_load = False

"""
# Mediapipe NPZ
mocap_data_file_path = "E:/Data/mocap/Yurika/Mediapipe_v2/Classes"
mocap_data_file_extensions = [".npz"] 
mocap_topology_file = "data/configs/Mediapipe_config.json"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_fps = 30
mocap_joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # skeleton without face, hands
mocap_data_window_length = 30
mocap_data_window_offset = 10
mocap_pos_scale = 100.0
mocap_stats_load = False
"""

"""
# Mediapipe FBX
mocap_data_file_path = "E:/Data/mocap/Yurika/Mediapipe_v2_fbx/Classes"
mocap_data_file_extensions = [".fbx"] 
mocap_topology_file = "data/configs/Mediapipe_config.json"
#mocap_data_types = ["pos", "rot", "vel_pos", "vel_rot", "acc_pos", "acc_rot"]
mocap_data_types = ["rot", "vel_rot", "acc_rot"]
mocap_fps = 30
mocap_joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] # skeleton without face, hands
mocap_data_window_length = 30
mocap_data_window_offset = 10
mocap_pos_scale = 100.0
mocap_stats_load = False
"""


"""
# Yolo MotionBert NPZ
mocap_data_file_path = "E:/Data/mocap/Yurika/YoloMB/Classes"
mocap_data_file_extensions = [".npz"] 
mocap_fps = 30
mocap_parents = [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]
mocap_children = [[1, 4, 7],[2],[3],[],[5],[6],[],[8],[9, 11, 14],[10],[],[12],[13],[],[15],[16],[]]
mocap_joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # full skeleton without torso
mocap_data_window_length = 30
mocap_data_window_offset = 15
mocap_pos_scale = 1.0
mocap_stats_load = False
"""

"""
Model Settings
"""

class_count = None
model_input_dim = None
model_hidden_dim = 64
model_layer_count = 2
model_dropout = 0.5

"""
Training Settings
"""

test_percentage = 0.2
batch_size = 128
epochs = 200
learning_rate = 1e-4
label_smoothing = 0.4
weight_decay = 1e-3
load_weights = False
save_weights = True
model_weights_file = "results_lstm/weights/classifier_epoch_200.pth"

"""
Save Paths Settings
"""

save_path = "results_ICK_Ricardo_XSens_npz"
save_stats_path = save_path + "/stats"
save_history_path = save_path + "/history"
save_weights_path = save_path + "/weights"

os.makedirs(save_stats_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_weights_path, exist_ok=True)

"""
Load Class and Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
npz_tools = npz.NPZ_Tools()
mocap_tools = mocap.Mocap_Tools()

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def load_class_filenames(directory, extensions):
    _, class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(tuple(extensions)):
                    instances.append((os.path.join(root, fname), class_index))
    return instances

def load_mocap_file(mocap_file_path):
    if mocap_file_path.lower().endswith(".bvh"):
        bvh_data = bvh_tools.load(mocap_file_path)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(
            mocap_data["motion"]["rot_local_euler"],
            mocap_data["rot_sequence"]
        )
    elif mocap_file_path.lower().endswith(".fbx"):
        fbx_data = fbx_tools.load(mocap_file_path)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0]
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(
            mocap_data["motion"]["rot_local_euler"],
            mocap_data["rot_sequence"]
        )
    elif mocap_file_path.lower().endswith(".npz"):
        npz_data, topo_data = npz_tools.load(mocap_file_path, mocap_topology_file)
        mocap_data = mocap_tools.npz_to_mocap(npz_data, topo_data, mocap_fps)[0]
    else:
        raise ValueError(f"Unsupported mocap file: {mocap_file_path}")

    # ALWAYS dynamically ensure pos_world and rot_world exist before slicing
    if "pos_world" not in mocap_data["motion"] or "rot_world" not in mocap_data["motion"]:
        pos_world, rot_world = mocap_tools.local_to_world(
            mocap_data["motion"]["rot_local"],
            mocap_data["motion"]["pos_local"],
            mocap_data["skeleton"]
        )
        mocap_data["motion"]["pos_world"] = pos_world
        mocap_data["motion"]["rot_world"] = rot_world

    # 1. Update Skeleton Topology, Offsets, Names, and root
    if "skeleton" in mocap_data:
        if "offsets" in mocap_data["skeleton"]:
            mocap_data["skeleton"]["offsets"] = mocap_data["skeleton"]["offsets"][mocap_joint_indices, :]
            mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
            
        if "parents" in mocap_data["skeleton"]:
            global_parents = mocap_data["skeleton"]["parents"]
            global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(mocap_joint_indices)}
            subset_parents = [-1] * len(mocap_joint_indices)
            
            for l_idx, g_idx in enumerate(mocap_joint_indices):
                current_ancestor = global_parents[g_idx]
                
                # Trace up the original hierarchy until we find an ancestor in our subset
                while current_ancestor != -1:
                    if current_ancestor in global_to_local:
                        subset_parents[l_idx] = global_to_local[current_ancestor]
                        break
                    current_ancestor = global_parents[current_ancestor]
            
            mocap_data["skeleton"]["parents"] = subset_parents
            
            # Rebuild children array based on the newly mapped subset parents
            if "children" in mocap_data["skeleton"]:
                subset_children = [[] for _ in range(len(mocap_joint_indices))]
                for l_idx, p_idx in enumerate(subset_parents):
                    if p_idx != -1:
                        subset_children[p_idx].append(l_idx)
                
                mocap_data["skeleton"]["children"] = subset_children

        # Update joint names and find the new root
        if "joints" in mocap_data["skeleton"]:
            global_joints = mocap_data["skeleton"]["joints"]
            subset_joints = [global_joints[i] for i in mocap_joint_indices]
            mocap_data["skeleton"]["joints"] = subset_joints
            
        # Update the root joint name based on the new topology
        if "root" in mocap_data["skeleton"] and "parents" in mocap_data["skeleton"]:
            if -1 in subset_parents:
                new_root_idx = subset_parents.index(-1)
                mocap_data["skeleton"]["root"] = subset_joints[new_root_idx]

    # 2. Slice Motion Data (Positions and Rotations)
    if "motion" in mocap_data:
        for key in ["pos_local", "rot_local", "pos_world", "rot_world"]:
            if key in mocap_data["motion"]:
                # Keep only the subset joints
                mocap_data["motion"][key] = mocap_data["motion"][key][:, mocap_joint_indices, :]
                
                # Scale positional values if needed
                if "pos" in key:
                    mocap_data["motion"][key] *= mocap_pos_scale

    return mocap_data


def load_class_data(class_files):
    class_data = []
    mocap_file_data_all = []
    
    for class_file, class_index in class_files:
        mocap_file_data = load_mocap_file(class_file)
        mocap_file_data_all.append(mocap_file_data)
        
        pos = mocap_file_data["motion"]["pos_local"]
        rot = mocap_file_data["motion"]["rot_local"]
        
        # Initialize a dictionary to store computed features
        features = {
            "pos": pos,
            "rot": rot
        }
        
        # Conditionally compute velocities if they (or their accelerations) are requested
        if "vel_pos" in mocap_data_types or "acc_pos" in mocap_data_types:
            features["vel_pos"] = np.concatenate(
                (np.zeros((1, pos.shape[1], pos.shape[2])), np.diff(pos, axis=0)), 
                axis=0
            )
        if "vel_rot" in mocap_data_types or "acc_rot" in mocap_data_types:
            features["vel_rot"] = np.concatenate(
                (np.zeros((1, rot.shape[1], rot.shape[2])), np.diff(rot, axis=0)), 
                axis=0
            )

        # Conditionally compute accelerations
        if "acc_pos" in mocap_data_types:
            features["acc_pos"] = np.concatenate(
                (np.zeros((1, features["vel_pos"].shape[1], features["vel_pos"].shape[2])), np.diff(features["vel_pos"], axis=0)), 
                axis=0
            )
        if "acc_rot" in mocap_data_types:
            features["acc_rot"] = np.concatenate(
                (np.zeros((1, features["vel_rot"].shape[1], features["vel_rot"].shape[2])), np.diff(features["vel_rot"], axis=0)), 
                axis=0
            )
            
        # Dynamically select and order the feature arrays based on the mocap_data_types list
        selected_arrays = [features[data_type] for data_type in mocap_data_types]
        
        # Stack all requested variables along the last axis
        mocap_data = np.concatenate(selected_arrays, axis=-1)
        class_data.append((mocap_data, class_index))
        
    return class_data, mocap_file_data_all


def calculate_class_weights(class_labels, num_classes):
    unique, counts = np.unique(class_labels, return_counts=True)
    total_samples = len(class_labels)

    class_weights = np.zeros(num_classes, dtype=np.float32)
    for class_id, count in zip(unique, counts):
        class_weights[class_id] = total_samples / (num_classes * count)

    return torch.FloatTensor(class_weights)

"""
Normalise Data and Create Dataset
"""

def create_dataset_with_split(class_data, window_length, window_offset, test_percentage):
    train_motion, train_labels, test_motion, test_labels = [], [], [], []

    for data, class_id in class_data:
        split_point = int((1 - test_percentage) * data.shape[0])

        for dI in range(0, split_point - window_length, window_offset):
            train_labels.append(class_id)
            train_motion.append(data[dI:dI + window_length])

        for dI in range(split_point, data.shape[0] - window_length, window_offset):
            test_labels.append(class_id)
            test_motion.append(data[dI:dI + window_length])

    return (
        np.array(train_labels, dtype=np.int64),
        np.stack(train_motion, dtype=np.float32),
        np.array(test_labels, dtype=np.int64),
        np.stack(test_motion, dtype=np.float32),
    )

def calc_norm_values(train_motion_data):
    flattened = np.reshape(train_motion_data, (-1, train_motion_data.shape[-1]))
    mean = np.mean(flattened, axis=0)
    std = np.std(flattened, axis=0)
    std[std == 0] = 1e-8
    return mean, std

class MocapDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

"""
Classifier
"""

class Classifier(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        out, _ = self.lstm(x)
        final_state = self.fc_dropout(out[:, -1, :])
        return self.classifier(final_state)

"""
Training Functions
"""

def train_step(batch_x, batch_y):
    batch_yhat = classifier(batch_x)
    _loss = class_loss(batch_yhat, batch_y)

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()

    return _loss

def test_step(batch_x, batch_y):
    with torch.no_grad():
        batch_yhat = classifier(batch_x)
        _loss = class_loss(batch_yhat, batch_y)
    return _loss

def test_model(data_loader):
    correct = 0
    total = 0

    classifier.eval()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = classifier(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return correct / total

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {"train": [], "test": []}

    for epoch in range(epochs):
        start = time.time()

        train_loss_per_epoch = []
        classifier.train()

        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _train_loss = train_step(batch_x, batch_y)
            train_loss_per_epoch.append(_train_loss.detach().cpu().numpy())

        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        loss_history["train"].append(train_loss_per_epoch)

        test_loss_per_epoch = []
        classifier.eval()

        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _test_loss = test_step(batch_x, batch_y)
            test_loss_per_epoch.append(_test_loss.detach().cpu().numpy())

        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))
        loss_history["test"].append(test_loss_per_epoch)

        train_correct = test_model(train_dataloader)
        test_correct = test_model(test_dataloader)

        scheduler.step()

        print(
            'epoch {} : train: loss {:01.4f} corr {:01.2f} test: loss {:01.4f} correct {:01.2f} time {:01.2f}'.format(
                epoch + 1,
                train_loss_per_epoch,
                train_correct * 100,
                test_loss_per_epoch,
                test_correct * 100,
                time.time() - start
            )
        )

    return loss_history

"""
Save Training History
"""

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_columns,
            delimiter=',',
            lineterminator='\n'
        )
        csv_writer.writeheader()

        for row in range(csv_row_count):
            csv_row = {}
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]
            csv_writer.writerow(csv_row)

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs_count = len(loss_history[keys[0]])

    for key in keys:
        plt.plot(range(epochs_count), loss_history[key], label=key)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()

"""
Main Execution
"""

classes, class_to_idx = find_classes(mocap_data_file_path)
class_count = len(classes)
class_files = load_class_filenames(mocap_data_file_path, mocap_data_file_extensions)
class_data, _ = load_class_data(class_files)

train_labels, train_data, test_labels, test_data = create_dataset_with_split(
    class_data,
    mocap_data_window_length,
    mocap_data_window_offset,
    test_percentage
)

# Flatten joints and per-joint features into one per-frame feature vector
# Before: (N, T, J, F_joint)
# After:  (N, T, J * F_joint)
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], -1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], -1)

model_input_dim = train_data.shape[-1]

if mocap_stats_load == False:
    mean_np, std_np = calc_norm_values(train_data)
    np.save(save_stats_path + "/mean.npy", mean_np)
    np.save(save_stats_path + "/std.npy", std_np)
else:
    mean_np = np.load(save_stats_path + "/mean.npy")
    std_np = np.load(save_stats_path + "/std.npy")

# Pre-normalize full datasets
train_data_norm = (train_data - mean_np) / (std_np + 1e-8)
test_data_norm = (test_data - mean_np) / (std_np + 1e-8)

train_data_norm = np.nan_to_num(train_data_norm)
test_data_norm = np.nan_to_num(test_data_norm)

class_weights = calculate_class_weights(train_labels, len(classes)).to(device)

train_loader = DataLoader(
    MocapDataset(train_data_norm, train_labels),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    MocapDataset(test_data_norm, test_labels),
    batch_size=batch_size,
    shuffle=True
)

classifier = Classifier(
    num_features=model_input_dim,
    hidden_dim=model_hidden_dim,
    num_classes=class_count,
    num_layers=model_layer_count,
    dropout=model_dropout
).to(device)

print(classifier)

batch = next(iter(train_loader))
batch_x = batch[0].to(device)
batch_y = batch[1].to(device)

print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)

batch_yhat = classifier(batch_x)
print("batch_yhat s ", batch_yhat.shape)

if load_weights:
    classifier.load_state_dict(torch.load(model_weights_file, map_location=device))

if save_weights:
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    class_loss = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing
    )

    print("Starting Training...")
    history = train(train_loader, test_loader, epochs)

    save_loss_as_csv(history, save_history_path + "/history_{}.csv".format(epochs))
    save_loss_as_image(history, save_history_path + "/history_{}.png".format(epochs))

    torch.save(
        classifier.state_dict(),
        save_weights_path + "/classifier_weights_epoch_{}.pth".format(epochs)
    )

batch_x, batch_y = next(iter(test_loader))
batch_x = batch_x.to(device)
batch_yhat = classifier(batch_x)
_, pred_labels = torch.max(batch_yhat, 1)

for i in range(min(batch_size, batch_x.shape[0])):
    print("motion {} pred class {} true class {}".format(i, pred_labels[i], batch_y[i]))