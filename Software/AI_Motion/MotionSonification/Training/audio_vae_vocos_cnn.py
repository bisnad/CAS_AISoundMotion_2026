"""
Same as audio_vae_vocos_cnn_v3.5 but with perceptual loss instead of simple MSE less on mel spectra
"""

"""
Imports
"""

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

from vocos import Vocos

import auraloss


"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Audio Settings
"""

audio_file_path = "E:/Data/audio/Diane/48khz/"
audio_files = ["4d69949b.wav"]

audio_sample_rate = 48000 # numer of audio samples per sec
audio_channels = 1

audio_window_length_vocos = 65280 # 256 mel frames worth of audio
audio_window_length_vae = 1792 # 8 mel frames worth of audio
audio_mel_count_vocos = None # will be calculated
audio_mel_count_vae = None
audio_window_offset = 960 # don't change if this vae will later be used for the motion2audio transformer model

"""
Fix Seeds
"""

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

"""
Vocoder Model
"""

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

"""
VAE Model Settings
"""

latent_dim = 32
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]

save_weights = True
load_weights = False
encoder_weights_file = "results/weights/encoder_weights_epoch_400"
decoder_weights_file = "results/weights/decoder_weights_epoch_400"

"""
Training Settings
"""

data_count = 100000 # 100000
batch_size = 32


train_percentage = 0.9 # train / test split
test_percentage  = 0.1
ae_learning_rate = 1e-4
ae_rec_loss_scale = 5.0
ae_beta = 0.0 # will be calculated
ae_beta_cycle_duration = 100
ae_beta_min_const_duration = 20
ae_beta_max_const_duration = 20
ae_min_beta = 0.0
ae_max_beta = 0.1


epochs = 400
model_save_interval = 50
save_history = True

"""
Create Dataset
"""

class AudioDataset(Dataset):
    def __init__(self, audio_file_path, audio_files, audio_window_length_vocos, audio_data_count):
        self.audio_file_path = audio_file_path
        self.audio_files = audio_files
        self.audio_window_length_vocos = audio_window_length_vocos
        self.audio_data_count = audio_data_count
        
        self.audio_waveforms = []
        
        for audio_file in self.audio_files:
            audio_waveform, _ = torchaudio.load(self.audio_file_path + "/" + audio_file)
            self.audio_waveforms.append(audio_waveform)
    
    def __len__(self):
        return self.audio_data_count
    
    def __getitem__(self, idx):
        
        audio_index = torch.randint(0, len(self.audio_waveforms), size=(1,))
        audio_waveform = self.audio_waveforms[audio_index]
        
        audio_length = audio_waveform.shape[1]
        audio_excerpt_start = torch.randint(0, audio_length - self.audio_window_length_vocos, size=(1,))
        audio_excerpt = audio_waveform[:, audio_excerpt_start:audio_excerpt_start+self.audio_window_length_vocos]
        audio_excerpt = audio_excerpt[0]
        
        return audio_excerpt


full_dataset = AudioDataset(audio_file_path, audio_files, audio_window_length_vocos, data_count)
dataset_size = len(full_dataset)

data_item = full_dataset[0]

print("data_item s ", data_item.shape)

dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

batch_x = next(iter(dataloader))

print("batch_x s ", batch_x.shape)

# test conversion of audio waveform into vocos mel spectra

audio_batch = next(iter(dataloader)).to(device)
audio_batch_mels = vocos.feature_extractor(audio_batch.unsqueeze(1))

print("audio_batch s ", audio_batch.shape)
print("audio_batch_mels s ", audio_batch_mels.shape)

"""
Create Models
"""

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


# test encoder
audio_batch = next(iter(dataloader)).to(device)
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

# test decoder
audio_decoder_in = audio_encoder_out
audio_decoder_out = decoder(audio_decoder_in)

audio_batch_mels.shape
audio_decoder_out.shape

audio_features = torch.cat([audio_batch_mels[:, 0, :, :-audio_mel_count_vae], audio_decoder_out[:, 0, :, :]], dim=2)
audio_batch = vocos.decode(audio_features)

print("audio_decoder_in s ", audio_decoder_in.shape)
print("audio_decoder_out s ", audio_decoder_out.shape)
print("audio_features s ", audio_features.shape)
print("audio_batch s ", audio_batch.shape)

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

"""
Training
"""

def calc_ae_beta_values():
    
    ae_beta_values = []

    for e in range(epochs):
        
        cycle_step = e % ae_beta_cycle_duration
        
        #print("cycle_step ", cycle_step)

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

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# KL Divergence

def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl
   
def variational_loss2(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #alternative: mean squared distance from ideal mu=0 and std=1:
    vl=torch.mean(mu.pow(2)+(1-std).pow(2))
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


def ae_train_step(y_wave):
    
    #print("y_wave s ", y_wave.shape)
    
    batch_size = y_wave.shape[0]

    y_mels = vocos.feature_extractor(y_wave)
    
    #print("y_mels s ", y_mels.shape)
    
    # regroup mels
    # from: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
    # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    y_mels_regrouped = y_mels.reshape(batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae)   
    
    #print("y_mels_regrouped s ", y_mels_regrouped.shape)
    
    # permute tensor so that audio_mel_group_count can be combined with batch size
    # from: batch_size,1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    # to: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    y_mels_regrouped = y_mels_regrouped.permute(0, 3, 1, 2, 4) 
    
    #print("y_mels_regrouped 2 s ", y_mels_regrouped.shape)
    
    # reshape tensor to combine batch_size and audio_vae_mels_per_vocos_mels
    # from: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    # to: batch_size x audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    y_mels_regrouped = y_mels_regrouped.reshape(-1, 1, audio_mel_filter_count, audio_mel_count_vae)
    
    #print("y_mels_regrouped 3 s ", y_mels_regrouped.shape)
    
    # encode mels 
    audio_encoder_out_mu, audio_encoder_out_std = encoder(y_mels_regrouped)
    
    #print("audio_encoder_out_mu s ", audio_encoder_out_mu.shape)

    mu = audio_encoder_out_mu
    std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
    decoder_input = encoder.reparameterize(mu, std)
    
    #print("decoder_input s ", decoder_input.shape)
    
    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
 
    yhat_mels_regrouped = decoder(decoder_input)
    
    #print("yhat_mels_regrouped s ", yhat_mels_regrouped.shape)
    
    # ae mel rec loss
    _ae_mel_rec_loss = ae_mel_loss(y_mels_regrouped, yhat_mels_regrouped)
    
    # convert the mel grouping back to original
    # from: batch_size x audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    # to: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae
    yhat_mels = yhat_mels_regrouped.reshape(batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae) 
    
    #print("yhat_mels s ", yhat_mels.shape)
    
    # from: batch_size, audio_vae_mels_per_vocos_mels, 1, audio_mel_filter_count, audio_mel_count_vae        
    # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    yhat_mels = yhat_mels.permute(0, 2, 3, 1, 4) 
    
    #print("yhat_mels 2 s ", yhat_mels.shape)
    
    # from: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
    # to: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
    yhat_mels = yhat_mels.reshape(batch_size, audio_mel_filter_count, audio_mel_count_vocos) 
    
    #print("yhat_mels 3 s ", yhat_mels.shape)
    
    yhat_wave = vocos.decode(yhat_mels)
    
    #print("yhat_wave s ", yhat_wave.shape)
    
    # ae perc rec loss
    _ae_perc_rec_loss = ae_perc_loss(y_wave, yhat_wave.unsqueeze(1))

    _total_loss = 0.0
    _total_loss += _ae_mel_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_perc_rec_loss * ae_rec_loss_scale * 0.5
    _total_loss += _ae_kld_loss * ae_beta

    # Backpropagation
    ae_optimizer.zero_grad()
    _total_loss.backward()
    
    #torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.01)
    #torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.01)

    ae_optimizer.step()
    
    return _total_loss, _ae_mel_rec_loss, _ae_perc_rec_loss, _ae_kld_loss

# ae_train_step

audio_batch = next(iter(dataloader)).to(device)
test_loss = ae_train_step(audio_batch)

def train(dataloader, epochs):
    
    global ae_beta
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae mel"] = []
    loss_history["ae perc"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_beta = ae_beta_values[epoch]
        
        #print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        ae_train_loss_per_epoch = []
        ae_mel_loss_per_epoch = []
        ae_perc_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for train_batch in dataloader:
            train_batch = train_batch.unsqueeze(1).to(device)
            
            _ae_loss, _ae_mel_loss, _ae_perc_loss, _ae_kld_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_mel_loss = _ae_mel_loss.detach().cpu().numpy()
            _ae_perc_loss = _ae_perc_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_mel_loss_per_epoch.append(_ae_mel_loss)
            ae_perc_loss_per_epoch.append(_ae_perc_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_mel_loss_per_epoch = np.mean(np.array(ae_mel_loss_per_epoch))
        ae_perc_loss_per_epoch = np.mean(np.array(ae_perc_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae mel"].append(ae_mel_loss_per_epoch)
        loss_history["ae perc"].append(ae_perc_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} mel {:01.4f} perc {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_mel_loss_per_epoch, ae_perc_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        ae_scheduler.step()
        
    return loss_history

# fit model
loss_history = train(dataloader, epochs)

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
torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))


"""
Inference
"""

def create_ref_audio_window(waveform_window, file_name):

    torchaudio.save("{}".format(file_name), waveform_window, audio_sample_rate)

def create_voc_audio_window(waveform_window, file_name):

    with torch.no_grad():
        audio_features = vocos.feature_extractor(waveform_window.to(device))
        waveform_window_voc = vocos.decode(audio_features)
    
    torchaudio.save("{}".format(file_name), waveform_window_voc.detach().cpu(), audio_sample_rate)

def create_pred_audio_window(waveform_window, file_name):
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        
        y_wave = waveform_window.to(device)
        
        batch_size = 1
        
        print("y_wave s ", y_wave.shape)
        
        y_mels = vocos.feature_extractor(y_wave)
        
        print("y_mels s ", y_mels.shape)
        
        # regroup mels
        # from: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
        # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
        y_mels_regrouped = y_mels.reshape(1, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae)   
        
        print("y_mels_regrouped s ", y_mels_regrouped.shape)
        
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
        
        yhat_mels_regrouped = decoder(decoder_input)
        
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
        
        yhat_wave = vocos.decode(yhat_mels.squeeze(1))
        
    torchaudio.save("{}".format(file_name), yhat_wave.detach().cpu(), audio_sample_rate)

    encoder.train()
    decoder.train()
    
test_waveform, _ = torchaudio.load("E:/Data/audio/Diane/48khz/4d69949b.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")
test_waveform_sample_index = audio_sample_rate * 10
test_waveform_window = test_waveform[:, test_waveform_sample_index:test_waveform_sample_index+audio_window_length_vocos]

create_ref_audio_window(test_waveform_window, "results/audio/audio_window_orig.wav")
create_voc_audio_window(test_waveform_window, "results/audio/audio_window_voc.wav")
create_pred_audio_window(test_waveform_window, "results/audio/audio_window_pred_epoch_{}.wav".format(epochs))

def create_ref_audio(waveform, file_name):

    torchaudio.save("{}".format(file_name), waveform, audio_sample_rate)
    
    #print("waveform s ", waveform.shape)

def create_voc_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_window_length_vocos // 2
    audio_window_env = torch.hann_window(audio_window_length_vocos)
    
    audio_window_count = int(waveform_length - audio_window_length_vocos) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    

    for i in range(audio_window_count):
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_window_length_vocos
        
        target_audio = waveform[:, window_start:window_end]
        
        #print("i ", i, " target_audio s ", target_audio.shape)
        
        with torch.no_grad():
            audio_features = vocos.feature_extractor(target_audio.to(device))
            voc_audio = vocos.decode(audio_features).detach().cpu()

        #print("voc_audio s ", voc_audio.shape)
        #print("grain_env s ", grain_env.shape)

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_window_length_vocos] += voc_audio[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

def create_pred_audio(waveform, file_name):
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        
        batch_size = 1
    
        waveform_length = waveform.shape[1]
        audio_window_offset = audio_window_length_vocos // 2
        audio_window_env = torch.hann_window(audio_window_length_vocos)
        
        audio_window_count = int(waveform_length - audio_window_length_vocos) // audio_window_offset
        pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
        
        #print("pred_audio_sequence s ", pred_audio_sequence.shape)
        
        for i in range(audio_window_count):
            
            window_start = i * audio_window_offset
            window_end = window_start + audio_window_length_vocos
            
            waveform_window = waveform[:, window_start:window_end]
            
            y_wave = waveform_window.to(device)
            
            y_mels = vocos.feature_extractor(y_wave)
            
            # regroup mels
            # from: batch_size, 1, audio_mel_filter_count, audio_mel_count_vocos
            # to: batch_size, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae
            y_mels_regrouped = y_mels.reshape(1, 1, audio_mel_filter_count, audio_vae_mels_per_vocos_mels, audio_mel_count_vae)   
            
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
            
            yhat_mels_regrouped = decoder(decoder_input)
            
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
            
            yhat_wave = vocos.decode(yhat_mels.squeeze(1))
            
            yhat_wave = yhat_wave.detach().cpu()
            
            pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_window_length_vocos] += yhat_wave[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

    encoder.train()
    decoder.train()

def create_pred_audio2(waveform, file_name):
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        
        batch_size = 1
    
        waveform_length = waveform.shape[1]
        audio_window_offset = audio_window_length_vae // 2
        audio_window_env = torch.hann_window(audio_window_length_vae)
        
        audio_window_count = int(waveform_length - audio_window_length_vocos) // audio_window_offset
        pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
        
        #print("pred_audio_sequence s ", pred_audio_sequence.shape)
        
        # covert first waveform of length audio_window_length vocos into mel spectra
        
        waveform_window = waveform[:, 0:audio_window_length_vocos]
        y_wave_vocos = waveform_window.to(device)
        yhat_mels_vocos = vocos.feature_extractor(y_wave_vocos)
        
        for i in range(audio_window_count):
            
            window_start = i * audio_window_offset
            window_end = window_start + audio_window_length_vocos
            waveform_window = waveform[:, window_start:window_end]
            
            y_wave_vocos = waveform_window.to(device)
            
            #print("y_wave_vocos s ", y_wave_vocos.shape)
            
            # get last mel spectra for vae to encode and decode
            
            y_mels_vocos = vocos.feature_extractor(y_wave_vocos)
            
            #print("y_mels_vocos s ", y_mels_vocos.shape)
            
            y_mels_vae = y_mels_vocos[:, :, -audio_mel_count_vae:]
            
            #print("y_mels_vae s ", y_mels_vae.shape)
            
            # encode mels 
            audio_encoder_in = y_mels_vae.unsqueeze(1)
            
            #print("audio_encoder_in s ", audio_encoder_in.shape)
            
            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            mu = audio_encoder_out_mu
            std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            
            # decode back into mels
            audio_decoder_in = encoder.reparameterize(mu, std)
            
            #print("audio_decoder_in s ", audio_decoder_in.shape)
            
            audio_decoder_out = decoder(audio_decoder_in)
            
            #print("audio_decoder_out s ", audio_decoder_out.shape)
            
            yhat_mels_vae = audio_decoder_out.squeeze(1)
            
            # pop first mels from yhat_mels_vocos and append new yhat_mels_vae
            
            #print("yhat_mels_vae s ", yhat_mels_vae.shape)
            #print("yhat_mels_vocos[:, :, audio_mel_count_vae:] s ", yhat_mels_vocos[:, :, audio_mel_count_vae:].shape)
            
            yhat_mels_vocos = torch.cat([yhat_mels_vocos[:, :, audio_mel_count_vae:], yhat_mels_vae], dim=2)
            
            #print("yhat_mels_vocos s ", yhat_mels_vocos.shape)

            # convert yhat_mels_vocos into waveform
            yhat_wave_vocos = vocos.decode(yhat_mels_vocos.squeeze(1))
            
            # take only last section of length audio_window_length_vae of the waveform
            yhat_wave_vae = yhat_wave_vocos[:, -audio_window_length_vae:]
            
            yhat_wave_vae = yhat_wave_vae.detach().cpu()
            
            pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_window_length_vae] += yhat_wave_vae[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

    encoder.train()
    decoder.train()

#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take1__double_Bind_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take1__double_Bind_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take2_Hibr_II_HQ_audio_crop_48khz.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")

test_waveform, _ = torchaudio.load("E:/Data/audio/Diane/48khz/4d69949b.wav")
#test_waveform, _ = torchaudio.load("E:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav")

test_start_times_sec = [ 20, 120, 240 ]
test_duration_sec = 20

for test_start_time_sec in test_start_times_sec:
    start_time_frames = test_start_time_sec * audio_sample_rate
    end_time_frames = start_time_frames + test_duration_sec * audio_sample_rate

    create_ref_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_ref_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)))
    create_voc_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_voc_{}-{}.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec)))
    create_pred_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_pred_{}-{}_epoch_{}_.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec), epochs))
    create_pred_audio2(test_waveform[:, start_time_frames:end_time_frames], "results/audio/audio_pred_{}-{}_epoch_{}_2.wav".format(test_start_time_sec, (test_start_time_sec + test_duration_sec), epochs))


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
            
            #print("i ", i)
            
            audio_encoder_in = y_mels[i, ...].unsqueeze(0)
            
            #print("audio_encoder_in s ", audio_encoder_in.shape)
            
            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            mu = audio_encoder_out_mu
            std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            audio_encoder_out = encoder.reparameterize(mu, std)
            
            #print("audio_encoder_out s ", audio_encoder_out.shape)

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
            
            #print("yhat_mels s ", yhat_mels.shape)
            
            yhat_mels = yhat_mels.reshape(audio_mel_filter_count, audio_mel_count_vae).detach()
            
            #print("yhat_mels 2 s ", yhat_mels.shape)
            
            yhat_mels_all.append(yhat_mels)
            
        yhat_mels_all = torch.cat(yhat_mels_all, dim=1).unsqueeze(0)
        
        #print("yhat_mels_all s ", yhat_mels_all.shape)
         
        yhat_audio = vocos.decode(yhat_mels_all).detach().cpu()
        
        #print("yhat_audio s ", yhat_audio.shape)

    
    torchaudio.save("{}".format(file_name), torch.reshape(yhat_audio, (1, -1)), audio_sample_rate)

    
    decoder.train()
    

# reconstruct original waveform

test_start_time_sec = 20
test_duration_sec = 20
start_time_frames = test_start_time_sec * audio_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_sample_rate
    
latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
decode_audio_encodings(latent_vectors, "results/audio/rec_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))


# random walk
# somewhat broken, needs fixing

test_start_time_sec = 20
test_duration_sec = 20

start_time_frames = test_start_time_sec * audio_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_sample_rate

audio_window_offset = audio_window_length_vae // 2

latent_vectors = encode_audio(test_waveform[:, start_time_frames:start_time_frames + audio_window_length_vocos + audio_window_offset])
latent_vectors = latent_vectors[:1]

audio_window_count = int(test_duration_sec * audio_sample_rate - audio_window_length_vae) // audio_window_offset - 1


for window_index in range(audio_window_count):
    random_step = np.random.random((latent_dim)).astype(np.float32) * 0.1
    latent_vectors.append(latent_vectors[window_index] + random_step)

decode_audio_encodings(latent_vectors, "results/audio/randwalk_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))


# sequence offset following

test_start_time_sec = 20
test_duration_sec = 20

start_time_frames = test_start_time_sec * audio_sample_rate
end_time_frames = start_time_frames + test_duration_sec * audio_sample_rate

latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])

offset_encodings = []

for index in range(len(latent_vectors)):
    sin_value = np.sin(index / (len(latent_vectors) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 1.0
    offset_encoding = latent_vectors[index] + offset
    offset_encodings.append(offset_encoding)
    
decode_audio_encodings(offset_encodings, "results/audio/offset_audio_epochs_{}_audio_{}-{}.wav".format(epochs, test_start_time_sec, test_duration_sec))


# interpolate two original sequences

test1_start_time_sec = 20
test2_start_time_sec = 60
test_duration_sec = 20

start1_time_frames = test1_start_time_sec * audio_sample_rate
end1_time_frames = start1_time_frames + test_duration_sec * audio_sample_rate

start2_time_frames = test2_start_time_sec * audio_sample_rate
end2_time_frames = start2_time_frames + test_duration_sec * audio_sample_rate

latent_vectors_1 = encode_audio(test_waveform[:, start1_time_frames:end1_time_frames])
latent_vectors_2 = encode_audio(test_waveform[:, start2_time_frames:end2_time_frames])


mix_encodings = []

for index in range(len(latent_vectors_1)):
    mix_factor = index / (len(latent_vectors_1) - 1)
    mix_encoding = latent_vectors_1[index] * (1.0 - mix_factor) + latent_vectors_2[index] * mix_factor
    mix_encodings.append(mix_encoding)

decode_audio_encodings(mix_encodings, "results/audio/mix_audio_epochs_{}_audio1_{}-{}_audio2_{}-{}.wav".format(epochs, test1_start_time_sec, test1_start_time_sec + test_duration_sec, test2_start_time_sec, test2_start_time_sec + test_duration_sec))




# latent space visualisation

"""
Create Dataset
"""


"""
Load Data - Audio
"""

audio_all_data = []


for audio_data_file in audio_files:   
    
    audio_data, _ = torchaudio.load(audio_file_path + audio_data_file)
    
    print("audio_data 2 s ", audio_data.shape)
    
    audio_all_data.append(audio_data)

audio_dataset = []


audio_waveform_start_offset = audio_window_length_vocos - audio_window_length_vae


for sI in range(len(audio_all_data)):
    
    with torch.no_grad():
        
        mocap_data = mocap_all_data[sI]["motion"]["rot_local"].reshape(-1, mocap_pose_dim)
        audio_data = audio_all_data[sI][0]
    
        print(sI)
        print("mocap_data s ", mocap_data.shape)
        print("audio_data s ", audio_data.shape)
        
        mocap_frame_count = mocap_data.shape[0]

        for mfI in range(mocap_frame_start_offset, mocap_frame_count - motion_vae_mocap_length * 3, mocap_frame_incr):
            
            print("mfI ", mfI, " out of ", (mocap_frame_count - motion_vae_mocap_length))
            
            # mocap sequence part
            
            # get mocap sequence
            mocap_excerpt_start = mfI
            mocap_excerpt_end = mfI + motion_vae_mocap_length
            
            #print("mocap_excerpt_start ", mocap_excerpt_start)
            #print("mocap_excerpt_end ", mocap_excerpt_end)
            
            mocap_excerpt = mocap_data[mocap_excerpt_start:mocap_excerpt_end, :]
            mocap_excerpt = torch.from_numpy(mocap_excerpt).unsqueeze(0).to(torch.float32).to(device)
            
            #print("mfI ", mfI, " mocap_excerpt s ", mocap_excerpt.shape)
            
            # normalise mocap excerpt
            mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
            
            mocap_dataset.append(mocap_excerpt_norm.cpu())
            
            # audio sequence part
            asI = mfI * audio_waveform_length_per_mocap_frame
            audio_waveform_start = asI - audio_waveform_start_offset
            audio_waveform_end = audio_waveform_start + audio_vocos_waveform_length
            
            #print("audio_waveform_start ", audio_waveform_start)
            #print("audio_waveform_end ", audio_waveform_end)
            
            audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
            
            #print("mfI ", mfI, " audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            
            #print("mfI ", mfI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
            
            # normalise audio mels excerpt
            audio_mels_excerpt_norm = (audio_mels_excerpt - audio_mean) / (audio_std + 1e-8)
            
            audio_dataset.append(audio_mels_excerpt_norm.cpu())

mocap_dataset = torch.cat(mocap_dataset, dim=0)
audio_dataset = torch.cat(audio_dataset, dim=0)

print("mocap_dataset s ", mocap_dataset.shape)
print("audio_dataset s ", audio_dataset.shape)