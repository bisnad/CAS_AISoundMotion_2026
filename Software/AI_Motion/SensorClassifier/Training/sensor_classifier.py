"""
Sensor Classification using LSTM Networks

This module implements a neural network classifier for sensor data using PyTorch.
It natively reads .npz recordings and derives class labels from the directory structure.
"""

"""
Imports
"""

import os, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""
Settings
"""

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

"""
Sensor Data Settings
"""

# Example: Mobile Phone Data (Sensor2Osc)
sensor_data_file_path = "data/sensors_phone/"
sensor_data_file_extensions = [".npz"] 
sensor_data_ids = ["/accelerometer", "/gyroscope"] # OSC addresses to extract
sensor_data_window_length = 60
sensor_data_window_offset = 20
sensor_stats_load = False # load previously calculated stats instead of calculating new ones

"""
# Example: Imu Data
sensor_data_file_path = "data/sensors_imu/"
sensor_data_file_extensions = [".npz"] 
sensor_data_ids = ["/imu/1/accelerometer", "/imu/1/gyroscope"] # OSC addresses to extract
sensor_data_window_length = 60
sensor_data_window_offset = 20
sensor_stats_load = False # load previously calculated stats instead of calculating new ones
"""

"""
Model Settings
"""

class_count = None # automatically determined from folders
model_input_dim = None # automatically determined from sensors
#model_hidden_dim = 128
#model_layer_count = 3
model_hidden_dim = 64
model_layer_count = 3
model_dropout = 0.3

"""
Training Settings
"""

test_percentage = 0.2
batch_size = 64
epochs = 400
learning_rate = 1e-4
weight_decay = 0.001
label_smoothing = 0.1
load_weights = False
save_weights = True
model_weights_file = "results/sensors_imu_2/weights/classifier_epoch_400.pth"

"""
Save Paths Settings
"""

save_path = "results/sensors_phone"
#save_path = "results/sensors_imu"
save_stats_path = save_path + "/stats"
save_history_path = save_path + "/history"
save_weights_path = save_path + "/weights"

os.makedirs(save_stats_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_weights_path, exist_ok=True)

"""
Load Class and Sensor Data (Directory-Based)
"""

def find_classes(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Data directory '{directory}' does not exist.")
        
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    
    print("--- Directory-Based Class Mapping ---")
    for cls, idx in class_to_idx.items():
        print(f"Directory '{cls}' -> Class ID {idx}")
    print("-------------------------------------")
    
    return classes, class_to_idx

def load_class_filenames(directory, extensions):
    _, class_to_idx = find_classes(directory)
    instances = []
    
    for target_class, class_index in class_to_idx.items():
        target_dir = os.path.join(directory, target_class)
        class_file_count = 0
        
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(tuple(extensions)):
                    instances.append((os.path.join(root, fname), class_index))
                    class_file_count += 1
                    
        print(f"Found {class_file_count} recording(s) in '{target_class}'")
        
    return instances

def load_sensor_file(file_path, sensor_ids):
    data = np.load(file_path)
    sensor_values = []
    min_len = float('inf')
    
    # Extract values for requested OSC addresses
    for sid in sensor_ids:
        key = sid + "_values"
        if key in data:
            val = data[key]
            sensor_values.append(val)
            if val.shape[0] < min_len:
                min_len = val.shape[0]
        else:
            print(f"Warning: {key} not found in {file_path}")
            
    if not sensor_values:
        return None
        
    # Truncate all arrays to the minimum time length and concatenate features
    sensor_values = [val[:min_len, :] for val in sensor_values]
    combined = np.concatenate(sensor_values, axis=1)
    return combined

def load_class_data(class_files, sensor_ids):
    class_data = []
    for class_file, class_index in class_files:
        sensor_data = load_sensor_file(class_file, sensor_ids)
        if sensor_data is not None:
            # The class_index here directly comes from the folder name
            class_data.append((sensor_data, class_index))
    return class_data

def calculate_class_weights(class_labels, num_classes):
    """Calculate inverse frequency weights for classes"""
    unique, counts = np.unique(class_labels, return_counts=True)
    total_samples = len(class_labels)
    class_weights = np.zeros(num_classes)
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
            
    return np.array(train_labels, dtype=np.int64), np.stack(train_motion, dtype=np.float32), \
           np.array(test_labels, dtype=np.int64), np.stack(test_motion, dtype=np.float32)

def calc_norm_values(train_motion_data):
    flattened = np.reshape(train_motion_data, (-1, train_motion_data.shape[-1]))
    mean = np.mean(flattened, axis=0)
    std = np.std(flattened, axis=0)
    std[std == 0] = 1e-8
    return mean, std

class SensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, class_count, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, batch_first=True, dropout=dropout if layer_count > 1 else 0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, class_count)
        self.relu = nn.ReLU()
        
        self.init_weights(self.rnn)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    # forget gate is the second of the 4 chunks: i, f, g, o
                    param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        x = h[-1]  # Take final hidden state from last layer
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

"""
Training Functions
"""

def train_step(batch_x, batch_y):
    batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
    batch_x_norm = torch.nan_to_num(batch_x_norm)

    batch_yhat = classifier(batch_x_norm)
    _loss = class_loss(batch_yhat, batch_y) 

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss

def test_step(batch_x, batch_y):
    batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)

    with torch.no_grad():
        batch_yhat = classifier(batch_x_norm)
        _loss = class_loss(batch_yhat, batch_y) 
        
    return _loss

def test_model(data_loader):
    correct = 0
    total = 0

    classifier.eval() # Ensure dropout is disabled for eval
    with torch.no_grad():
        for data in data_loader:
            batch_x, batch_y = data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
        
            outputs = classifier(batch_x_norm)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    return correct / total

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {"train": [], "test": []}
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss_per_epoch = []
        classifier.train() # Enable dropout during training
        
        for train_data in train_dataloader:
            batch_x, batch_y = train_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            _train_loss = train_step(batch_x, batch_y)
            _train_loss = _train_loss.detach().cpu().numpy()
            train_loss_per_epoch.append(_train_loss)
            
        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        loss_history["train"].append(train_loss_per_epoch)
        
        test_loss_per_epoch = []
        classifier.eval() # Disable dropout during test loss calculation
        
        for test_data in test_dataloader:
            batch_x, batch_y = test_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            _test_loss = test_step(batch_x, batch_y)
            _test_loss = _test_loss.detach().cpu().numpy()
            test_loss_per_epoch.append(_test_loss)
            
        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))
        loss_history["test"].append(test_loss_per_epoch)
        
        train_correct = test_model(train_dataloader)
        test_correct = test_model(test_dataloader)
        
        scheduler.step()
        
        print('epoch {} : train: loss {:01.4f} corr {:01.2f} test: loss {:01.4f} correct {:01.2f} time {:01.2f}'.format(
            epoch + 1, train_loss_per_epoch, train_correct * 100, test_loss_per_epoch, test_correct * 100, time.time()-start))

    return loss_history

"""
Save Training History
"""

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
            csv_row = {key: loss_history[key][row] for key in loss_history.keys()}
            csv_writer.writerow(csv_row)

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    plt.figure()
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.close()

"""
Main Execution
"""

# detect classes and load corresponding sensor files
classes, class_to_idx = find_classes(sensor_data_file_path)
class_count = len(classes)
class_files = load_class_filenames(sensor_data_file_path, sensor_data_file_extensions)
class_data = load_class_data(class_files, sensor_data_ids)
    
# create dataset 
train_labels, train_data, test_labels, test_data = create_dataset_with_split(
    class_data, sensor_data_window_length, sensor_data_window_offset, test_percentage)

model_input_dim = train_data.shape[-1]

# normalise sensor data
if not sensor_stats_load:
    mean_np, std_np = calc_norm_values(train_data)
    np.save(save_stats_path + "/mean.npy", mean_np)
    np.save(save_stats_path + "/std.npy", std_np)
else:
    mean_np = np.load(save_stats_path + "/mean.npy")
    std_np = np.load(save_stats_path + "/std.npy")
    
data_mean = torch.tensor(mean_np, dtype=torch.float32).to(device)
data_std = torch.tensor(std_np, dtype=torch.float32).to(device)

# Calculate class weights for unbalanced datasets
class_weights = calculate_class_weights(train_labels, len(classes)).to(device)

# create data loaders
train_loader = DataLoader(SensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=True)

# create classifier model
classifier = Classifier(input_dim=model_input_dim, hidden_dim=model_hidden_dim, 
                        layer_count=model_layer_count, class_count=class_count, 
                        dropout=model_dropout).to(device)

print(classifier)

# test classifier pass
batch_x, batch_y = next(iter(train_loader))
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

print("batch_x shape:", batch_x.shape)
print("batch_y shape:", batch_y.shape)
batch_yhat = classifier(batch_x)
print("batch_yhat shape:", batch_yhat.shape)

if load_weights:
    classifier.load_state_dict(torch.load(model_weights_file))

# run training
if save_weights:
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    class_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)

    print("Starting Training...")
    history = train(train_loader, test_loader, epochs)
    
    save_loss_as_csv(history, save_history_path + "/history_{}.csv".format(epochs))
    save_loss_as_image(history, save_history_path + "/history_{}.png".format(epochs))

    # save trained model
    torch.save(classifier.state_dict(), save_weights_path + "/classifier_weights_epoch_{}.pth".format(epochs))

# Test the model on some test data
classifier.eval()
with torch.no_grad():
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.to(device)
    batch_x_norm = (batch_x - data_mean) / data_std 
    batch_x_norm = torch.nan_to_num(batch_x_norm)
    batch_yhat = classifier(batch_x_norm)
    _, pred_labels = torch.max(batch_yhat, 1)

    for i in range(min(batch_size, 10)):  # print first 10 for brevity
        print("motion {} pred class {} true class {}".format(i, pred_labels[i], batch_y[i]))