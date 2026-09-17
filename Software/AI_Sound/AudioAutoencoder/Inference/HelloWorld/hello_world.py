# -------------------------------------------------------------------------------------------------
# Loads an CNN-based variational audio autoencoder
# And demonstrates different experiments in inference mode
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

from vocos import Vocos


# -------------------------------------------------------------------------------------------------
# Compute Unit
# -------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

# -------------------------------------------------------------------------------------------------
# Audio Settings
# -------------------------------------------------------------------------------------------------

audio_data_path = "data/audio/"
audio_data_files = ["Take1__double_Bind_HQ_audio_crop_48khz.wav",
                    "Take2_Hibr_II_HQ_audio_crop_48khz.wav"]
audio_sample_rate = 48000 # numer of audio samples per sec

audio_window_length_vocos = 65280 # 256 mel frames worth of audio
audio_window_length_vae = 1792 # 8 mel frames worth of audio
audio_mel_count_vocos = None # will be calculated
audio_mel_count_vae = None

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------

save_audio_path = os.path.join("results/audio/")

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

encoder_weights_file = "data/models/vae_cnn_Stocos_ld32/encoder_weights_epoch_400"
decoder_weights_file = "data/models/vae_cnn_Stocos_ld32/decoder_weights_epoch_400"

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
decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

# -------------------------------------------------------------------------------------------------
# Load Test Waveforms
# -------------------------------------------------------------------------------------------------

audio_test_waveforms = [] 

for audio_data_file in audio_data_files:
    waveform, _ = torchaudio.load(audio_data_path + audio_data_file)
    audio_test_waveforms.append(waveform)

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

test_audio_file_index = 0
test_audio_ranges_sec = [ [ 10.0, 20.0] ]

for test_audio_range_sec in test_audio_ranges_sec:

    start_time_frames = int(test_audio_range_sec[0] * audio_sample_rate)
    end_time_frames = int(test_audio_range_sec[1] * audio_sample_rate)

    create_ref_audio(audio_test_waveforms[test_audio_file_index][:, start_time_frames:end_time_frames], "{}audio_orig_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio_file_index], test_audio_range_sec[0], test_audio_range_sec[1]))
    create_voc_audio(audio_test_waveforms[test_audio_file_index][:, start_time_frames:end_time_frames], "{}audio_voc_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio_file_index], test_audio_range_sec[0], test_audio_range_sec[1]))

# -------------------------------------------------------------------------------------------------
# Reconstruct Original Waveform
# -------------------------------------------------------------------------------------------------

test_audio_file_index = 0
test_audio_ranges_sec = [ [ 10.0, 20.0] ]

for test_audio_range_sec in test_audio_ranges_sec:

    start_time_frames = int(test_audio_range_sec[0] * audio_sample_rate)
    end_time_frames = int(test_audio_range_sec[1] * audio_sample_rate)
            
    latent_vectors = encode_audio(audio_test_waveforms[test_audio_file_index][:, start_time_frames:end_time_frames])
    decode_audio_encodings(latent_vectors, "{}audio_rec_audio_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio_file_index], test_audio_range_sec[0], test_audio_range_sec[1]))

# -------------------------------------------------------------------------------------------------
# Random Walk
# -------------------------------------------------------------------------------------------------

test_audio_file_index = 0
test_audio_ranges_sec = [ [ 10.0, 20.0] ]
random_walk_step_size = 1.0

for test_audio_range_sec in test_audio_ranges_sec:

    start_time_frames = int(test_audio_range_sec[0] * audio_sample_rate)
    end_time_frames = int(test_audio_range_sec[1] * audio_sample_rate)

    latent_vectors = encode_audio(audio_test_waveforms[test_audio_file_index][:, start_time_frames:end_time_frames])

    for i in range(1, len(latent_vectors)):
        random_step = np.random.randn(latent_dim).astype(np.float32)
        random_step = random_step / np.linalg.norm(random_step) * random_walk_step_size
        latent_vectors[i] = latent_vectors[i-1] + random_step

    decode_audio_encodings(latent_vectors, "{}audio_randwalk_audio_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio_file_index], test_audio_range_sec[0], test_audio_range_sec[1]))

# -------------------------------------------------------------------------------------------------
# Offset Following
# -------------------------------------------------------------------------------------------------

test_audio_file_index = 0
test_audio_ranges_sec = [ [ 10.0, 20.0] ]
offset_sizes = [ 0.0, 4.0 ]

num_offsets = len(offset_sizes)

for test_audio_range_sec in test_audio_ranges_sec:

    start_time_frames = int(test_audio_range_sec[0] * audio_sample_rate)
    end_time_frames = int(test_audio_range_sec[1] * audio_sample_rate)

    latent_vectors = encode_audio(audio_test_waveforms[test_audio_file_index][:, start_time_frames:end_time_frames])
    offset_encodings = []

    for i in range(1, len(latent_vectors)):

        if num_offsets == 1:
            offset_size = offset_sizes[0]
        else:
            pos = i / (len(latent_vectors) - 1) * (num_offsets - 1)
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, num_offsets - 1)
            frac = pos - idx_low

            offset_size = offset_sizes[idx_low] * (1.0 - frac) + offset_sizes[idx_high] * frac

        offset = np.ones(shape=(latent_dim), dtype=np.float32) * offset_size
        offset_encoding = latent_vectors[i] + offset
        offset_encodings.append(offset_encoding)

    decode_audio_encodings(offset_encodings, "{}audio_offset_audio_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio_file_index], test_audio_range_sec[0], test_audio_range_sec[1]))

# -------------------------------------------------------------------------------------------------
# Latent Interpolation
# -------------------------------------------------------------------------------------------------

test_audio1_file_index = 0
test_audio1_ranges_sec = [ [ 10.0, 20.0] ]
test_audio2_file_index = 1
test_audio2_ranges_sec = [ [ 10.0, 20.0] ]
mix_factors = [ 0.0, 1.0, 2.0, -2.0 ]

num_mix_factors = len(mix_factors)
mix_encodings = []

for test_audio1_range_sec, test_audio2_range_sec in zip(test_audio1_ranges_sec, test_audio2_ranges_sec):

    audio1_start_time_frames = int(test_audio1_range_sec[0] * audio_sample_rate)
    audio1_end_time_frames = int(test_audio1_range_sec[1] * audio_sample_rate)

    audio2_start_time_frames = int(test_audio2_range_sec[0] * audio_sample_rate)
    audio2_end_time_frames = int(test_audio2_range_sec[1] * audio_sample_rate)

    latent_vectors_1 = encode_audio(audio_test_waveforms[test_audio1_file_index][:, audio1_start_time_frames:audio1_end_time_frames])
    latent_vectors_2 = encode_audio(audio_test_waveforms[test_audio2_file_index][:, audio2_start_time_frames:audio2_end_time_frames])


    for i in range(len(latent_vectors_1)):

        if num_mix_factors == 1:
            mix_factor = mix_factors[0]
        else:
            pos = i / (len(latent_vectors_1) - 1) * (num_mix_factors - 1)
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, num_mix_factors - 1)
            frac = pos - idx_low

            mix_factor = mix_factors[idx_low] * (1.0 - frac) + mix_factors[idx_high] * frac

        mix_encoding = latent_vectors_1[i] * (1.0 - mix_factor) + latent_vectors_2[i] * mix_factor
        mix_encodings.append(mix_encoding)


    decode_audio_encodings(mix_encodings, "{}audio_mix_audio_{}_{}-{}_and_{}_{}-{}.wav".format(save_audio_path, audio_data_files[test_audio1_file_index], test_audio1_range_sec[0], test_audio1_range_sec[1], audio_data_files[test_audio2_file_index], test_audio2_range_sec[0], test_audio2_range_sec[1]))