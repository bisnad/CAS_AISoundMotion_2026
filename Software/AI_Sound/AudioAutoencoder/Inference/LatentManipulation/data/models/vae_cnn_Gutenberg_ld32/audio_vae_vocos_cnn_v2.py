# -------------------------------------------------------------------------------------------------
# Trains an CNN-based variational audio autoencoder on short sequences of log-mel spectrograms: 
# it encodes 48 kHz waveform excerpts into spectrograms representations, 
# compresses them into latent vectors, reconstructs and decodes them to audio, 
# and optimizes mel MSE, perceptually weighted multi-resolution STFT, and cyclic KL-divergence losses.
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchaudio
import simpleaudio as sa
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
import os, time
import json
import csv
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE

import colorsys
from matplotlib.colors import hsv_to_rgb

from vocos import Vocos

import auraloss

# -------------------------------------------------------------------------------------------------
# Compute Unit
# -------------------------------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# -------------------------------------------------------------------------------------------------
# Audio Settings
# -------------------------------------------------------------------------------------------------

audio_data_path = "E:/data/audio/Gutenberg/"
audio_data_files = ["Night_and_Day_by_Virginia_Woolf_48khz.wav"]
audio_valid_ranges = [[-1.0, -1.0]]

audio_sample_rate = 48000 # numer of audio samples per sec
audio_channels = 1

audio_window_length_vocos = 65280 # 256 mel frames worth of audio
audio_window_length_vae = 1792 # 8 mel frames worth of audio
audio_mel_count_vocos = None # will be calculated
audio_mel_count_vae = None

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------

save_path = "results_vae_cnn_Gutenberg_ld32"
save_weights_path = os.path.join(save_path, "weights/")
save_history_path = os.path.join(save_path, "histories/")
save_audio_path = os.path.join(save_path, "audio/")

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_audio_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

latent_dim = 32 # 32
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

audio_excerpt_count = 100000 # total number of randomly-positioned excerpts to draw across all audio files
test_percentage = 0.1

batch_size = 128
epochs = 400

ae_learning_rate = 1e-4
ae_rec_loss_scale = 5.0
ae_beta = 0.0 # will be calculated
ae_beta_cycle_duration = 100
ae_beta_min_const_duration = 20
ae_beta_max_const_duration = 20
ae_min_beta = 0.0
ae_max_beta = 0.1 # 0.1

save_history = True
save_weights = True
load_weights = False
model_save_interval = 50
encoder_weights_file = "results/weights/encoder_weights_epoch_400"
decoder_weights_file = "results/weights/decoder_weights_epoch_400"

# -------------------------------------------------------------------------------------------------
# Inference Settings
# -------------------------------------------------------------------------------------------------

audio_test_starts_sec = [
    [5, 50, 100, 140, 160, 214, 270, 340]
]

audio_test_duration_sec = 20
perturbation_sizes = [ 0.0, 0.1, 0.2, 0.5, 1.0 ]

# -------------------------------------------------------------------------------------------------
# Fix Seeds
# -------------------------------------------------------------------------------------------------

def set_all_seeds(seed: int):
    # Python's built-in RNG
    random.seed(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch RNG (CPU)
    torch.manual_seed(seed)
    # PyTorch RNG (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # (optional) PyTorch backend for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

# -------------------------------------------------------------------------------------------------
# Load Vocoder Model
# -------------------------------------------------------------------------------------------------

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)

# freeze model parameters
for param in vocos.parameters():
    param.requires_grad = False

# determine number of mel spectra procuced by waveform of length audio_window_length_vocos
vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vocos), dtype=torch.float32).to(device))
audio_mel_count_vocos = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

print("audio_mel_count_vocos ", audio_mel_count_vocos, " audio_mel_filter_count ", audio_mel_filter_count)

# assert that the waveform length for vocos feature extraction is the same as the waveform length after vocos decodes the features
assert vocos.decode(vocoder_features).shape[-1] == audio_window_length_vocos, "the length of the waveform that vocos encodes into mels and decodes from mels must be identical"

# determine number of mel spectra procuced by waveform of length audio_window_length_vae
vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vae), dtype=torch.float32).to(device))
audio_mel_count_vae = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

print("audio_mel_count_vae ", audio_mel_count_vae, " audio_mel_filter_count ", audio_mel_filter_count)

#assert that the number of mels spectra produced by Vocos is an integer multiple of the number of mels consumed by the vae
assert audio_mel_count_vocos % audio_mel_count_vae == 0, "vocos mel count must be an integer multiple of vae mel count "

audio_vae_mels_per_vocos_mels = audio_mel_count_vocos // audio_mel_count_vae

# assert that the waveform length for vocos feature extraction is the same as the waveform length after vocos decodes the features
assert vocos.decode(vocoder_features).shape[-1] == audio_window_length_vae, "the length of the waveform that vocos encodes into mels and decodes from mels must be identical"

# -------------------------------------------------------------------------------------------------
# Load Audio
# -------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------
# Create Dataset
# -------------------------------------------------------------------------------------------------

# usable length per file: number of valid excerpt start positions (must fit a full audio_window_length_vocos window)
audio_file_usable_lengths = []

for sI in range(len(audio_all_data)):
    audio_data = audio_all_data[sI][0]
    usable_length = max(0, audio_data.shape[0] - audio_window_length_vocos)
    audio_file_usable_lengths.append(usable_length)

audio_file_usable_lengths = np.array(audio_file_usable_lengths, dtype=np.float64)

if audio_file_usable_lengths.sum() <= 0:
    raise ValueError("No audio file has enough valid samples for the requested audio_window_length_vocos.")

# sample which file each excerpt is drawn from, proportional to each file's usable length
audio_file_sample_probs = audio_file_usable_lengths / audio_file_usable_lengths.sum()
audio_excerpt_file_indices = np.random.choice(len(audio_all_data), size=audio_excerpt_count, p=audio_file_sample_probs)

audio_dataset = []

with torch.no_grad():
    
    for excerpt_index in range(audio_excerpt_count):
        
        file_index = int(audio_excerpt_file_indices[excerpt_index])
        audio_data = audio_all_data[file_index][0]
        
        usable_length = audio_file_usable_lengths[file_index]
        
        audio_waveform_start = int(np.random.randint(0, int(usable_length) + 1))
        audio_waveform_end = audio_waveform_start + audio_window_length_vocos
        
        audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0)
        
        audio_dataset.append(audio_waveform_excerpt)

audio_dataset = torch.cat(audio_dataset, dim=0)

print("audio_dataset s ", audio_dataset.shape)

class AudioDataset(Dataset):
    def __init__(self, audio):
        self.audio = audio
    
    def __len__(self):
        return self.audio.shape[0]
    
    def __getitem__(self, idx):
        return self.audio[idx, ...]
    
full_dataset = AudioDataset(audio_dataset)

item_audio = full_dataset[0]

print("item_audio s ", item_audio.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

batch_audio = next(iter(train_loader))

print("batch_audio s ", batch_audio.shape)

# -------------------------------------------------------------------------------------------------
# Create Models
# -------------------------------------------------------------------------------------------------

# create encoder model

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

encoder = Encoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

print(encoder)

# test encoder
audio_batch = next(iter(train_loader)).to(device)
audio_batch_mels = vocos.feature_extractor(audio_batch.unsqueeze(1))
audio_encoder_in = audio_batch_mels[:, :, :, -audio_mel_count_vae:]
audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
audio_encoder_out = encoder.reparameterize(audio_encoder_out_mu, audio_encoder_out_std)

print("audio_batch s ", audio_batch.shape)
print("audio_batch_mels s ", audio_batch_mels.shape)
print("audio_encoder_in s ", audio_encoder_in.shape)
print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)
print("audio_encoder_out_std s ", audio_encoder_out_std.shape)
print("audio_encoder_out s ", audio_encoder_out.shape)

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    
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
    
vae_conv_channel_counts_reversed = vae_conv_channel_counts.copy()
vae_conv_channel_counts_reversed.reverse()
    
vae_dense_layer_sizes_reversed = vae_dense_layer_sizes.copy()
vae_dense_layer_sizes_reversed.reverse()

decoder = Decoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts_reversed, vae_conv_kernel_size, vae_dense_layer_sizes_reversed).to(device)

print(decoder)

# test decoder
audio_decoder_in = audio_encoder_out
audio_decoder_out = decoder(audio_decoder_in)

audio_features = torch.cat([audio_batch_mels[:, 0, :, :-audio_mel_count_vae], audio_decoder_out[:, 0, :, :]], dim=2)
audio_batch = vocos.decode(audio_features)

print("audio_decoder_in s ", audio_decoder_in.shape)
print("audio_decoder_out s ", audio_decoder_out.shape)
print("audio_features s ", audio_features.shape)
print("audio_batch s ", audio_batch.shape)

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

# -------------------------------------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------------------------------------

mse_loss = nn.MSELoss()

# KL Divergence

def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl

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

def ae_mel_loss(y_mels, yhat_mels):
    
    _aml = mse_loss(yhat_mels, y_mels)

    return _aml

def ae_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

def ae_loss(y_wave, yhat_wave, y_mels, yhat_mels, mu, std):

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    # ae mel_rec loss
    _ae_mel_rec_loss = ae_mel_loss(y_mels, yhat_mels)
    
    # ae_perc_rec_loss
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _ae_mel_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta
    
    return _total_loss, _ae_mel_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

# -------------------------------------------------------------------------------------------------
# Training 
# -------------------------------------------------------------------------------------------------

def calc_ae_beta_values():
    
    ae_beta_values = []

    for e in range(epochs):
        
        cycle_step = e % ae_beta_cycle_duration

        if cycle_step < ae_beta_min_const_duration:
            ae_beta_value = ae_min_beta
            ae_beta_values.append(ae_beta_value)
        elif cycle_step > ae_beta_cycle_duration - ae_beta_max_const_duration:
            ae_beta_value = ae_max_beta
            ae_beta_values.append(ae_beta_value)
        else:
            lin_step = cycle_step - ae_beta_min_const_duration
            ae_beta_value = ae_min_beta + (ae_max_beta - ae_min_beta) * lin_step / (ae_beta_cycle_duration - ae_beta_min_const_duration - ae_beta_max_const_duration)
            ae_beta_values.append(ae_beta_value)
            
    return ae_beta_values

ae_beta_values = calc_ae_beta_values()

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=50, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

def ae_train_step(y_wave):
    
    batch_size = y_wave.shape[0]

    y_mels = vocos.feature_extractor(y_wave)
    
    # regroup mels
    # from: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
    # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    y_mels_regrouped = y_mels.reshape(batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae)   
    
    # permute tensor so that audio_mel_group_count can be combined with batch size
    # from: batch_size,1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    # to: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    y_mels_regrouped = y_mels_regrouped.permute(0, 3, 1, 2, 4) 
    
    # reshape tensor to combine batch_size and audio_vae_mels_per_vocos_mels
    # from: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    # to: batch_size x audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    y_mels_regrouped = y_mels_regrouped.reshape(-1, 1, audio_mel_filter_count, audio_mel_count_vae)
    
    # encode mels 
    audio_encoder_out_mu, audio_encoder_out_std = encoder(y_mels_regrouped)

    mu = audio_encoder_out_mu
    std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
    decoder_input = encoder.reparameterize(mu, std)
    
    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
 
    yhat_mels_regrouped = decoder(decoder_input)
    
    # ae mel rec loss
    _ae_mel_rec_loss = ae_mel_loss(y_mels_regrouped, yhat_mels_regrouped)
    
    # convert the mel grouping back to original
    # from: batch_size x audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    # to: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    yhat_mels = yhat_mels_regrouped.reshape(batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae) 
    
    # from: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae        
    # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    yhat_mels = yhat_mels.permute(0, 2, 3, 1, 4) 
    
    # from: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    # to: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
    yhat_mels = yhat_mels.reshape(batch_size, audio_mel_filter_count, audio_mel_count_vocos) 
    
    yhat_wave = vocos.decode(yhat_mels)
    
    # ae perc rec loss
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave.unsqueeze(1))

    _total_loss = 0.0
    _total_loss += _ae_mel_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta

    # Backpropagation
    ae_optimizer.zero_grad()
    _total_loss.backward()

    ae_optimizer.step()
    
    return _total_loss, _ae_mel_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

@torch.no_grad()
def ae_test_step(y_wave):
    
    batch_size = y_wave.shape[0]

    y_mels = vocos.feature_extractor(y_wave)
    
    y_mels_regrouped = y_mels.reshape(batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae)   
    y_mels_regrouped = y_mels_regrouped.permute(0, 3, 1, 2, 4) 
    y_mels_regrouped = y_mels_regrouped.reshape(-1, 1, audio_mel_filter_count, audio_mel_count_vae)
    
    audio_encoder_out_mu, audio_encoder_out_std = encoder(y_mels_regrouped)

    mu = audio_encoder_out_mu
    std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
    decoder_input = encoder.reparameterize(mu, std)
    
    _ae_kld_loss = variational_loss(mu, std)
 
    yhat_mels_regrouped = decoder(decoder_input)
    
    _ae_mel_rec_loss = ae_mel_loss(y_mels_regrouped, yhat_mels_regrouped)
    
    yhat_mels = yhat_mels_regrouped.reshape(batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae) 
    yhat_mels = yhat_mels.permute(0, 2, 3, 1, 4) 
    yhat_mels = yhat_mels.reshape(batch_size, audio_mel_filter_count, audio_mel_count_vocos) 
    
    yhat_wave = vocos.decode(yhat_mels)
    
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave.unsqueeze(1))

    _total_loss = 0.0
    _total_loss += _ae_mel_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta
    
    return _total_loss, _ae_mel_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

# ae_train_step

audio_batch = next(iter(train_loader)).to(device)
test_loss = ae_train_step(audio_batch)

def train(train_loader, test_loader, epochs):
    
    global ae_beta
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae mel"] = []
    loss_history["ae perc"] = []
    loss_history["ae kld"] = []
    loss_history["beta"] = []
    loss_history["lr"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_beta = ae_beta_values[epoch]
        
        loss_history["beta"].append(ae_beta)
        loss_history["lr"].append(ae_optimizer.param_groups[0]["lr"])
        
        ae_train_loss_per_epoch = []
        ae_mel_loss_per_epoch = []
        ae_perc_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for train_batch in train_loader:
            train_batch = train_batch.unsqueeze(1).to(device)
            
            _ae_loss, _ae_mel_loss, _ae_perc_loss, _ae_kld_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_mel_loss = _ae_mel_loss.detach().cpu().numpy()
            _ae_perc_loss = _ae_perc_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()

            ae_train_loss_per_epoch.append(_ae_loss)
            ae_mel_loss_per_epoch.append(_ae_mel_loss)
            ae_perc_loss_per_epoch.append(_ae_perc_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_mel_loss_per_epoch = np.mean(np.array(ae_mel_loss_per_epoch))
        ae_perc_loss_per_epoch = np.mean(np.array(ae_perc_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))
        
        ae_test_loss_per_epoch = []
        
        for test_batch in test_loader:
            test_batch = test_batch.unsqueeze(1).to(device)
            
            _ae_loss, _, _, _ = ae_test_step(test_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            
            ae_test_loss_per_epoch.append(_ae_loss)

        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "{}encoder_weights_epoch_{}".format(save_weights_path, epoch))
            torch.save(decoder.state_dict(), "{}decoder_weights_epoch_{}".format(save_weights_path, epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae mel"].append(ae_mel_loss_per_epoch)
        loss_history["ae perc"].append(ae_perc_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} mel {:01.4f} perc {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, ae_mel_loss_per_epoch, ae_perc_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        ae_scheduler.step()
        
    return loss_history

# -------------------------------------------------------------------------------------------------
# Plot Training History
# -------------------------------------------------------------------------------------------------

def plot_training_history(loss_history, save_path):
    epochs_ = len(loss_history["ae train"])
    x = np.arange(1, epochs_ + 1)

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("CNN VAE Training History", fontsize=18, y=0.98)

    axs[0, 0].plot(x, loss_history["ae train"], label="Train Loss", color="tab:blue", linewidth=2)
    axs[0, 0].plot(x, loss_history["ae test"], label="Test Loss", color="tab:orange", linewidth=2)
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].legend()

    axs[0, 1].plot(x, loss_history["ae mel"], label="Mel MSE", color="tab:green", linewidth=2)
    axs[0, 1].plot(x, loss_history["ae perc"], label="Perceptual STFT", color="tab:red", linewidth=2)
    axs[0, 1].set_title("Reconstruction Losses")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].legend()

    axs[1, 0].plot(x, loss_history["ae kld"], label="KL Loss", color="tab:brown", linewidth=2)
    axs[1, 0].set_title("KL Divergence")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].legend()

    axs[1, 1].plot(x, loss_history["beta"], label="Beta", color="tab:pink", linewidth=2)
    axs[1, 1].set_title("Beta Schedule")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].legend()

    axs[2, 0].plot(x, loss_history["lr"], label="Learning Rate", color="tab:cyan", linewidth=2)
    axs[2, 0].set_title("Learning Rate")
    axs[2, 0].set_xlabel("Epochs")
    axs[2, 0].legend()

    axs[2, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

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

# -------------------------------------------------------------------------------------------------
# Load Test Waveforms (only for files that have at least one test start time)
# -------------------------------------------------------------------------------------------------

if len(audio_test_starts_sec) != len(audio_data_files):
    raise ValueError("audio_test_starts_sec must contain exactly one inner list per entry in audio_data_files.")

audio_test_waveforms = {}   # file_index -> waveform tensor
audio_test_file_indices = [ fI for fI in range(len(audio_data_files)) if len(audio_test_starts_sec[fI]) > 0 ]

for file_index in audio_test_file_indices:
    waveform, _ = torchaudio.load(audio_data_path + audio_data_files[file_index])
    audio_test_waveforms[file_index] = waveform

# -------------------------------------------------------------------------------------------------
# Create Latent Space Plot
# -------------------------------------------------------------------------------------------------

def create_2d_latent_space_representation(waveform, window_offset, max_windows=2000):

    encoder.eval()

    waveform_length = waveform.shape[1]
    window_count = int(waveform_length - audio_window_length_vocos) // window_offset
    window_count = min(window_count, max_windows)

    encodings = []

    with torch.no_grad():
        for i in range(window_count):

            window_start = i * window_offset
            window_end = window_start + audio_window_length_vocos

            target_audio = waveform[:, window_start:window_end]

            audio_features = vocos.feature_extractor(target_audio.to(device))
            audio_encoder_in = audio_features[:, :, -audio_mel_count_vae:].unsqueeze(1)

            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            mu = audio_encoder_out_mu
            std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            audio_encoder_out = encoder.reparameterize(mu, std)

            encodings.append(audio_encoder_out.detach().cpu())

    encoder.train()

    encodings = torch.cat(encodings, dim=0).numpy()
    return encodings

def distinct_hsv_colors(n, s=0.8, v=0.9):
    """
    Generate n HSV colors (h, s, v in [0, 1]) with
    evenly spaced hues and fixed saturation/value.
    """
    if n <= 0:
        return []
    # Golden ratio conjugate gives decent spacing on [0, 1)
    phi = 0.618033988749895
    colors = []
    h = 0.0
    for _ in range(n):
        colors.append((h % 1.0, s, v))
        h += phi
    return colors

def create_audio_space_image(Z_tsne, audio_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(audio_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.2)
        
    for hI, hR in enumerate(audio_excerpt_ranges):
        
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='o', facecolors="none", s=10.0, linewidths=0.4, edgecolors=plot_colors[hI], alpha=0.5)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Train / History Plot / Latent Space Plot
# -------------------------------------------------------------------------------------------------

if save_weights == True:

    # train model
    loss_history = train(train_loader, test_loader, epochs)

    # save training history
    save_loss_as_csv(loss_history, "{}history_{}.csv".format(save_history_path, epochs))
    plot_training_history(loss_history, "{}history_{}.png".format(save_history_path, epochs))

    # save model weights
    torch.save(encoder.state_dict(), "{}encoder_weights_epoch_{}".format(save_weights_path, epochs))
    torch.save(decoder.state_dict(), "{}decoder_weights_epoch_{}".format(save_weights_path, epochs))

    # build one combined latent space representation across all test files, tracking excerpt-range
    # offsets per file so highlight ranges still map correctly onto the concatenated encodings
    audio_space_window_offset = audio_window_length_vocos // 2

    all_encodings = []
    audio_ranges = []
    running_offset = 0

    for file_index in audio_test_file_indices:

        waveform = audio_test_waveforms[file_index]
        file_encodings = create_2d_latent_space_representation(waveform, audio_space_window_offset)

        for audio_start_sec in audio_test_starts_sec[file_index]:
            audio_ranges.append([
                running_offset + int(audio_start_sec * audio_sample_rate) // audio_space_window_offset,
                running_offset + int((audio_start_sec + audio_test_duration_sec) * audio_sample_rate) // audio_space_window_offset
            ])

        all_encodings.append(file_encodings)
        running_offset += file_encodings.shape[0]

    all_encodings = np.concatenate(all_encodings, axis=0)

    tsne = TSNE(n_components=2, max_iter=5000, verbose=1)
    Z_tsne = tsne.fit_transform(all_encodings)

    create_audio_space_image(Z_tsne, audio_ranges, "{}/audio_space_plot_epoch_{}.png".format(save_path, epochs))

# -------------------------------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------------------------------

def create_ref_audio(waveform, file_name):

    torchaudio.save("{}".format(file_name), waveform, audio_sample_rate)

def create_voc_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_window_length_vocos // 2
    audio_window_env = torch.hann_window(audio_window_length_vocos)
    
    audio_window_count = int(waveform_length - audio_window_length_vocos) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)

    for i in range(audio_window_count):
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_window_length_vocos
        
        target_audio = waveform[:, window_start:window_end]
        
        with torch.no_grad():
            audio_features = vocos.feature_extractor(target_audio.to(device))
            voc_audio = vocos.decode(audio_features).detach().cpu()

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_window_length_vocos] += voc_audio[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

def encode_audio(waveform):
    
    encoder.eval()
    
    with torch.no_grad():

        waveform = waveform.to(device)
 
        y_mels = vocos.feature_extractor(waveform.to(device))
        mel_count = y_mels.shape[-1]
        mel_count = (mel_count // audio_mel_count_vae) * audio_mel_count_vae
        y_mels = y_mels[:, :, -mel_count:]
        y_mels = y_mels.reshape((1, audio_mel_filter_count, -1, audio_mel_count_vae))
        y_mels = y_mels.permute((2, 0, 1, 3))

        latent_vectors = []
        
        for i in range(mel_count // audio_mel_count_vae):
            
            audio_encoder_in = y_mels[i, ...].unsqueeze(0)
            
            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            mu = audio_encoder_out_mu
            std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            audio_encoder_out = encoder.reparameterize(mu, std)

            latent_vector = audio_encoder_out.detach().cpu().numpy()
        
            latent_vectors.append(latent_vector)
            
    encoder.train()
    
    return latent_vectors
  
def decode_audio_encodings(encodings, file_name):
    
    decoder.eval()
    
    with torch.no_grad():
        
        encoding_count = len(encodings)
        
        yhat_mels_all = []
        
        for i in range(encoding_count):
            
            encoding = torch.Tensor(encodings[i]).to(device)
            yhat_mels = decoder(encoding)
            
            yhat_mels = yhat_mels.reshape(audio_mel_filter_count, audio_mel_count_vae).detach()
            
            yhat_mels_all.append(yhat_mels)
            
        yhat_mels_all = torch.cat(yhat_mels_all, dim=1).unsqueeze(0)
         
        yhat_audio = vocos.decode(yhat_mels_all).detach().cpu()

    torchaudio.save("{}".format(file_name), torch.reshape(yhat_audio, (1, -1)), audio_sample_rate)

    decoder.train()

# -------------------------------------------------------------------------------------------------
# Create Original and Vocos Waveform
# -------------------------------------------------------------------------------------------------

for file_index in audio_test_file_indices:

    waveform = audio_test_waveforms[file_index]
    file_tag = os.path.splitext(audio_data_files[file_index])[0]

    for test_start_time_sec in audio_test_starts_sec[file_index]:

        start_time_frames = test_start_time_sec * audio_sample_rate
        end_time_frames = start_time_frames + audio_test_duration_sec * audio_sample_rate

        create_ref_audio(waveform[:, start_time_frames:end_time_frames], "{}audio_ref_{}_{}-{}.wav".format(save_audio_path, file_tag, test_start_time_sec, test_start_time_sec + audio_test_duration_sec))
        create_voc_audio(waveform[:, start_time_frames:end_time_frames], "{}audio_voc_{}_{}-{}.wav".format(save_audio_path, file_tag, test_start_time_sec, test_start_time_sec + audio_test_duration_sec))

# -------------------------------------------------------------------------------------------------
# Reconstruct Original Waveform
# -------------------------------------------------------------------------------------------------

for file_index in audio_test_file_indices:

    waveform = audio_test_waveforms[file_index]
    file_tag = os.path.splitext(audio_data_files[file_index])[0]

    for test_start_time_sec in audio_test_starts_sec[file_index]:

        start_time_frames = test_start_time_sec * audio_sample_rate
        end_time_frames = start_time_frames + audio_test_duration_sec * audio_sample_rate
            
        latent_vectors = encode_audio(waveform[:, start_time_frames:end_time_frames])
        decode_audio_encodings(latent_vectors, "{}rec_audio_epochs_{}_{}_{}-{}.wav".format(save_audio_path, epochs, file_tag, test_start_time_sec, test_start_time_sec + audio_test_duration_sec))

# -------------------------------------------------------------------------------------------------
# Latent Perturbation
# -------------------------------------------------------------------------------------------------

for file_index in audio_test_file_indices:

    waveform = audio_test_waveforms[file_index]
    file_tag = os.path.splitext(audio_data_files[file_index])[0]

    for audio_start_sec in audio_test_starts_sec[file_index]:

        start_time_frames = int(audio_start_sec * audio_sample_rate)
        end_time_frames = int((audio_start_sec + audio_test_duration_sec) * audio_sample_rate)
        latent_vectors = encode_audio(waveform[:, start_time_frames:end_time_frames])
        
        z = torch.randn((1, latent_dim))  # or your encoded latent
        
        perturbed_encodings = []
        
        for index in range(len(latent_vectors)):
            
            sigma = perturbation_sizes[ min(index // (len(latent_vectors) // len(perturbation_sizes)), len(perturbation_sizes) - 1) ]
            
            epsilon = torch.randn_like(z) * sigma  # ~ N(0, sigma^2 I)
            
            latent_vector_perturbed = latent_vectors[index] + epsilon.numpy()
        
            perturbed_encodings.append(latent_vector_perturbed)
            
        decode_audio_encodings(perturbed_encodings, "{}perturb_audio_{}_{}-{}.wav".format(save_audio_path, file_tag, audio_start_sec, audio_start_sec + audio_test_duration_sec))

# -------------------------------------------------------------------------------------------------
# Latent Interpolation
# -------------------------------------------------------------------------------------------------

# interpolates between consecutive test start times within the same audio file
# (interpolation is not performed across different audio files)

for file_index in audio_test_file_indices:

    waveform = audio_test_waveforms[file_index]
    file_tag = os.path.splitext(audio_data_files[file_index])[0]
    file_starts_sec = audio_test_starts_sec[file_index]

    for i in range(len(file_starts_sec) - 1):

        test1_start_time_sec = file_starts_sec[i]
        test2_start_time_sec = file_starts_sec[i + 1]

        start1_time_frames = int(test1_start_time_sec * audio_sample_rate)
        end1_time_frames = int(start1_time_frames + audio_test_duration_sec * audio_sample_rate)
        
        start2_time_frames = int(test2_start_time_sec * audio_sample_rate)
        end2_time_frames = int(start2_time_frames + audio_test_duration_sec * audio_sample_rate)
        
        latent_vectors_1 = encode_audio(waveform[:, start1_time_frames:end1_time_frames])
        latent_vectors_2 = encode_audio(waveform[:, start2_time_frames:end2_time_frames])
        
        mix_encodings = []
        
        for index in range(len(latent_vectors_1)):
            mix_factor = index / (len(latent_vectors_1) - 1)
            mix_encoding = latent_vectors_1[index] * (1.0 - mix_factor) + latent_vectors_2[index] * mix_factor
            mix_encodings.append(mix_encoding)
        
        decode_audio_encodings(mix_encodings, "{}mix_audio_{}_{}-{}_{}-{}.wav".format(save_audio_path, file_tag, test1_start_time_sec, test1_start_time_sec + audio_test_duration_sec, test2_start_time_sec, test2_start_time_sec + audio_test_duration_sec))

