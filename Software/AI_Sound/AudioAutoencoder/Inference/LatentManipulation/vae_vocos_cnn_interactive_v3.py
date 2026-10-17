"""
Audio Autoencoder for Vocos 48khz - interactive real-time version (rolling mel buffer)
---------------------------------------------------------------------------------------
- Per source, each latent vector is computed from exactly ONE 8-frame mel block. The block's mels
  are cut from a slightly longer waveform slice (a few STFT context frames on each side, never
  passed to the autoencoder) so they look like training mels instead of reflect-padded edges.
- The two latents are mixed (per-dimension mix/offset, read at every step) and decoded by the
  autoencoder decoder into 8-frame mel blocks.
- Generated mel blocks are appended to a rolling buffer of the last N generated mel frames.
- Vocos decodes the WHOLE buffer; only a segment away from the right edge (lookahead) is output.
- Output segments are contiguous in time: no overlap-add, no phase cancellation.
"""
import os
import sys
import numpy as np
import threading
import queue

import torch
from torch import nn
import torchaudio

from vocos import Vocos
import sounddevice as sd

from pythonosc import dispatcher
from pythonosc import osc_server

from PyQt5 import QtWidgets, QtCore

"""
Device Settings
"""

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

"""
Audio Settings
"""
# Define a list of audio files to load
audio_file_paths = [
    "data/audio/Take1__double_Bind_HQ_audio_crop_48khz.wav",
    "data/audio/Take2_Hibr_II_HQ_audio_crop_48khz.wav"
]

audio_sample_rate = 48000
audio_channel_count = 1
default_audio_output_device = 0  # Will be overridden by GUI if needed

# audio callback block size (small -> low output latency, large -> more robust)
playback_block_size = 1024
playback_latency = 'high'

# Rolling mel buffer / Vocos settings
vocos_hop = 256                  # samples per mel frame (must match Vocos)
vae_window_samples = 1792        # samples worth of one VAE block (training: audio_window_length_vae)
buffer_frames = 128              # generated mel frames kept and fed to Vocos on every decode
lookahead_frames = 16            # right-edge frames held back; latency ~ lookahead_frames*hop samples
blocks_per_step = 2              # VAE blocks generated per Vocos call (1 = lowest latency, most compute)
max_audio_queue_length = 4       # number of generated segments buffered ahead
use_mean_encoding = True         # True: use mu only; False: sample like in training
sampling_noise_scale = 1.0       # only used if use_mean_encoding is False

"""
Autoencoder Settings
"""
latent_dim = 32
ae_conv_channel_counts = [ 16, 32, 64, 128 ]
ae_conv_kernel_size = (5, 3)
ae_dense_layer_sizes = [ 512 ]

ae_encoder_weights_file = "data/models/vae_cnn_Stocos_ld32/encoder_weights_epoch_400.pt"
ae_decoder_weights_file = "data/models/vae_cnn_Stocos_ld32/decoder_weights_epoch_400.pt"

"""
OSC Control Settings
"""
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9005

"""
Load Audio
"""
loaded_audio_waveforms = []
audio_durations_sec = []

for filepath in audio_file_paths:
    assert os.path.exists(filepath), f"Audio file not found: {filepath}"
    waveform, sr = torchaudio.load(filepath)
    # Ensure matching sample rate if necessary, here assuming 48kHz
    samples = waveform[0].to(device)
    loaded_audio_waveforms.append(samples)
    audio_durations_sec.append(samples.shape[0] / audio_sample_rate)


"""
Global Synthesis State
"""
file_index_1 = 0
file_index_2 = min(1, len(loaded_audio_waveforms) - 1)

audio_source_start_frame_index_1 = 0
audio_source_end_frame_index_1 = loaded_audio_waveforms[file_index_1].shape[0]
audio_source_frame_index_1 = 0

audio_source_start_frame_index_2 = 0
audio_source_end_frame_index_2 = loaded_audio_waveforms[file_index_2].shape[0]
audio_source_frame_index_2 = 0

audio_encoding_mix_factors = torch.zeros((latent_dim), dtype=torch.float32).to(device)
audio_encoding_offset_factors = torch.zeros((latent_dim), dtype=torch.float32).to(device)

"""
Create Models
"""
print("Loading Vocos Model...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
with torch.no_grad():
    f_vae = vocos.feature_extractor(torch.rand(size=(1, vae_window_samples), device=device))
    silent_mels = vocos.feature_extractor(torch.zeros(size=(1, 16 * vocos_hop), device=device))[..., 4:5]  # (1,F,1)
mel_filters = f_vae.shape[1]
mel_count = f_vae.shape[-1]
block_samples = mel_count * vocos_hop

try:
    stft_n_fft = int(vocos.feature_extractor.mel_spec.n_fft)
except Exception:
    stft_n_fft = 2048
    print("Could not read n_fft from Vocos feature extractor, assuming", stft_n_fft)
stft_context_frames = int(np.ceil((stft_n_fft // 2) / vocos_hop)) + 1
slice_frames = mel_count + 2 * stft_context_frames
slice_samples = (slice_frames - 1) * vocos_hop

step_frames = blocks_per_step * mel_count
step_samples = step_frames * vocos_hop
assert buffer_frames % mel_count == 0, "buffer_frames must be a multiple of the VAE block length"
assert lookahead_frames + step_frames < buffer_frames, "buffer too small for step + lookahead"
with torch.no_grad():
    t_slice = vocos.feature_extractor(torch.rand(size=(1, slice_samples), device=device)).shape[-1]
    assert t_slice == slice_frames, f"unexpected mel frame count {t_slice} vs {slice_frames}"
    t_dec = vocos.decode(vocos.feature_extractor(torch.rand(size=(1, (buffer_frames - 1) * vocos_hop), device=device))).shape[-1]
    assert t_dec == (buffer_frames - 1) * vocos_hop, "vocos decode length must be (frames-1)*hop"
print(f"VAE block: {mel_count} frames, STFT context: {stft_context_frames} frames, "
      f"step: {step_samples} samples ({step_samples / audio_sample_rate * 1000:.0f} ms), "
      f"lookahead: {lookahead_frames * vocos_hop / audio_sample_rate * 1000:.0f} ms")

# zero-padded copies of the waveforms so slices at file borders are always valid
pad_samples = slice_samples + step_samples
padded_audio_waveforms = [torch.nn.functional.pad(w, (pad_samples, pad_samples)) for w in loaded_audio_waveforms]

class Encoder(nn.Module):
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.ModuleList()
        stride = ((conv_kernel_size[0] - 1) // 2, (conv_kernel_size[1] - 1) // 2)
        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        for layer_index in range(1, len(conv_channel_counts)):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        self.dense_layers = nn.ModuleList()
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y

        self.dense_layers.append(nn.Linear(dense_layer_input_size, dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        for layer_index in range(1, len(dense_layer_sizes)):
            self.dense_layers.append(nn.Linear(dense_layer_sizes[layer_index-1], dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        self.fc_mu = nn.Linear(dense_layer_sizes[-1], latent_dim)
        self.fc_std = nn.Linear(dense_layer_sizes[-1], latent_dim)

    def forward(self, x):
        for layer in self.conv_layers: x = layer(x)
        x = self.flatten(x)
        for layer in self.dense_layers: x = layer(x)
        return self.fc_mu(x), self.fc_std(x)
    
    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)

encoder = Encoder(latent_dim, mel_count, mel_filters, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)
if os.path.exists(ae_encoder_weights_file):
    encoder.load_state_dict(torch.load(ae_encoder_weights_file, map_location=device))
encoder.eval()

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
        
        #padding = stride
        #output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        same_padding = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)

        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            #self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            # first upsampling stage: no bias, no activation directly after, to avoid seeding a
            # DC-offset spectral peak that later layers would otherwise replicate across the band
            use_bias = layer_index != 1
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=1, padding=same_padding, bias=use_bias))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        #self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        self.conv_layers.append(nn.Upsample(scale_factor=stride, mode="nearest"))
        self.conv_layers.append(nn.Conv2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=1, padding=same_padding))

    def forward(self, x):
        
        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        
        x = self.unflatten(x)

        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)
    
        return x

decoder = Decoder(latent_dim, mel_count, mel_filters, list(reversed(ae_conv_channel_counts)), ae_conv_kernel_size, list(reversed(ae_dense_layer_sizes))).to(device)
if os.path.exists(ae_decoder_weights_file):
    decoder.load_state_dict(torch.load(ae_decoder_weights_file, map_location=device))
decoder.eval()

"""
Audio Synthesis (rolling mel buffer)
"""
mel_buffer = silent_mels.expand(1, mel_filters, buffer_frames).clone()


def gather_blocks(source_index, start_sample, block_count):
    """(block_count, slice_samples) slices; slice b starts one STFT context before block b."""
    w = padded_audio_waveforms[source_index]
    first = start_sample - stft_context_frames * vocos_hop + pad_samples
    starts = first + torch.arange(block_count, device=w.device) * block_samples
    idx = starts[:, None] + torch.arange(slice_samples, device=w.device)[None, :]
    return w[idx]


def encode_blocks(slices):
    """slices (B, slice_samples) -> codes (B, latent_dim); only the central 8-frame mel block is encoded."""
    mels = vocos.feature_extractor(slices)                                      # (B, F, slice_frames)
    mels = mels[..., stft_context_frames:stft_context_frames + mel_count]       # (B, F, 8)
    mu, std = encoder(mels.unsqueeze(1))
    if use_mean_encoding:
        return mu
    std = torch.nn.functional.softplus(std) + 1e-6
    return mu + sampling_noise_scale * std * torch.randn_like(std)


@torch.no_grad()
def synthesize_audio():
    """Generates one contiguous segment of step_samples samples through the rolling mel buffer."""
    global mel_buffer, audio_source_frame_index_1, audio_source_frame_index_2

    k = blocks_per_step
    pos_1 = (audio_source_frame_index_1 // vocos_hop) * vocos_hop
    pos_2 = (audio_source_frame_index_2 // vocos_hop) * vocos_hop

    slices = torch.cat([gather_blocks(file_index_1, pos_1, k), gather_blocks(file_index_2, pos_2, k)], dim=0)
    z = encode_blocks(slices)
    z_1, z_2 = z[:k], z[k:]

    mix = audio_encoding_mix_factors
    decoder_in = z_1 * (1.0 - mix) + z_2 * mix + audio_encoding_offset_factors   # (k, latent_dim)

    new_mels = decoder(decoder_in).squeeze(1)                                    # (k, F, 8)
    new_mels = new_mels.permute(1, 0, 2).reshape(1, mel_filters, step_frames)    # (1, F, k*8)

    mel_buffer = torch.cat([mel_buffer[..., step_frames:], new_mels], dim=-1)    # rolling buffer
    wave = vocos.decode(mel_buffer).reshape(-1)                                  # (buffer_frames-1)*hop samples

    # frame f is centred at sample f*hop; output the newest step that is lookahead_frames from the right edge
    seg_end = (buffer_frames - lookahead_frames) * vocos_hop
    segment = wave[seg_end - step_samples:seg_end]

    audio_source_frame_index_1 += step_samples
    if audio_source_frame_index_1 + step_samples + slice_samples > audio_source_end_frame_index_1:
        audio_source_frame_index_1 = audio_source_start_frame_index_1

    audio_source_frame_index_2 += step_samples
    if audio_source_frame_index_2 + step_samples + slice_samples > audio_source_end_frame_index_2:
        audio_source_frame_index_2 = audio_source_start_frame_index_2

    return segment

"""
Audio Threading
"""
import time
audio_queue = queue.Queue(maxsize=max_audio_queue_length)
pending_audio = np.zeros(0, dtype=np.float32)
producer_running = True


def producer_thread():
    n_done, t_sum = 0, 0.0
    while producer_running:
        if not audio_queue.full():
            try:
                t0 = time.time()
                seg = synthesize_audio().cpu().numpy().astype(np.float32)
                audio_queue.put(seg)
                t_sum += time.time() - t0
                n_done += 1
                if n_done % 100 == 0:
                    print(f"avg synthesis time {t_sum / n_done * 1000:.1f} ms per {step_samples / audio_sample_rate * 1000:.0f} ms of audio")
            except Exception as e:
                print("Synthesis error:", e)
                sd.sleep(100)
        else:
            sd.sleep(5)


def audio_callback(out_data, frames, time_info, status):
    global pending_audio
    if status:
        print(status)
    while pending_audio.shape[0] < frames:
        try:
            pending_audio = np.concatenate([pending_audio, audio_queue.get_nowait()])
        except queue.Empty:
            break
    n = min(frames, pending_audio.shape[0])
    out_data[:] = 0.0
    out_data[:n, 0] = pending_audio[:n]
    pending_audio = pending_audio[n:]

class AudioStreamController:
    def __init__(self):
        self.stream = None
        self.is_playing = False
        self.device_id = default_audio_output_device
        # Start the background producer
        threading.Thread(target=producer_thread, daemon=True).start()

    def set_device(self, device_id):
        self.device_id = device_id
        if self.is_playing:
            self.stop()
            self.start()

    def start(self):
        if not self.is_playing:
            try:
                self.stream = sd.OutputStream(
                    samplerate=audio_sample_rate,
                    device=self.device_id,
                    channels=audio_channel_count,
                    callback=audio_callback,
                    blocksize=playback_block_size,
                    latency=playback_latency
                )
                self.stream.start()
                self.is_playing = True
                print(f"Audio started on device {self.device_id}")
            except Exception as e:
                print(f"Error starting audio stream: {e}")

    def stop(self):
        if self.is_playing and self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            print("Audio stopped")

audio_controller = AudioStreamController()

"""
PyQt5 GUI
"""
class AudioAutoencoderGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Audio Autoencoder Direct Dimension Control")
        self.setGeometry(100, 100, 800, 800)
        main_layout = QtWidgets.QVBoxLayout()

        # Audio Output Device Selection
        device_group = QtWidgets.QGroupBox("Audio Device Setup")
        device_layout = QtWidgets.QFormLayout()
        self.device_combo = QtWidgets.QComboBox()
        
        self.device_mapping = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0:
                self.device_mapping.append(i)
                self.device_combo.addItem(f"{i}: {dev['name']}")
                if i == default_audio_output_device:
                    self.device_combo.setCurrentIndex(len(self.device_mapping) - 1)
                    
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        device_layout.addRow("Output Device:", self.device_combo)
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)

        max_file_idx = max(0, len(loaded_audio_waveforms) - 1)

        # File 1 & File 2 Controls Layout
        files_layout = QtWidgets.QHBoxLayout()

        # Sequence 1
        seq1_group = QtWidgets.QGroupBox("Audio File 1 Controls")
        seq1_layout = QtWidgets.QFormLayout()
        
        self.seq1_idx_box = QtWidgets.QSpinBox()
        self.seq1_idx_box.setRange(0, max_file_idx)
        self.seq1_idx_box.setValue(file_index_1)
        self.seq1_idx_box.valueChanged.connect(self.on_seq1_idx_changed)
        seq1_layout.addRow("File Index:", self.seq1_idx_box)

        self.seq1_start_box = QtWidgets.QDoubleSpinBox()
        self.seq1_start_box.setDecimals(3)
        self.seq1_start_box.setSingleStep(0.1)
        self.seq1_end_box = QtWidgets.QDoubleSpinBox()
        self.seq1_end_box.setDecimals(3)
        self.seq1_end_box.setSingleStep(0.1)

        self.seq1_start_box.valueChanged.connect(self.update_seq1_ranges)
        self.seq1_end_box.valueChanged.connect(self.update_seq1_ranges)
        
        seq1_layout.addRow("Start Time (s):", self.seq1_start_box)
        seq1_layout.addRow("End Time (s):", self.seq1_end_box)
        seq1_group.setLayout(seq1_layout)

        # Sequence 2
        seq2_group = QtWidgets.QGroupBox("Audio File 2 Controls")
        seq2_layout = QtWidgets.QFormLayout()
        
        self.seq2_idx_box = QtWidgets.QSpinBox()
        self.seq2_idx_box.setRange(0, max_file_idx)
        self.seq2_idx_box.setValue(file_index_2)
        self.seq2_idx_box.valueChanged.connect(self.on_seq2_idx_changed)
        seq2_layout.addRow("File Index:", self.seq2_idx_box)

        self.seq2_start_box = QtWidgets.QDoubleSpinBox()
        self.seq2_start_box.setDecimals(3)
        self.seq2_start_box.setSingleStep(0.1)
        self.seq2_end_box = QtWidgets.QDoubleSpinBox()
        self.seq2_end_box.setDecimals(3)
        self.seq2_end_box.setSingleStep(0.1)

        self.seq2_start_box.valueChanged.connect(self.update_seq2_ranges)
        self.seq2_end_box.valueChanged.connect(self.update_seq2_ranges)
        
        seq2_layout.addRow("Start Time (s):", self.seq2_start_box)
        seq2_layout.addRow("End Time (s):", self.seq2_end_box)
        seq2_group.setLayout(seq2_layout)

        files_layout.addWidget(seq1_group)
        files_layout.addWidget(seq2_group)
        main_layout.addLayout(files_layout)

        # Scrollable area for latent dimensions
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QHBoxLayout(scroll_widget)

        # Encoding Mix
        mix_group = QtWidgets.QGroupBox("Encoding Mix")
        mix_layout = QtWidgets.QFormLayout()
        self.mix_master_box = QtWidgets.QDoubleSpinBox()
        self.mix_master_box.setRange(-1e6, 1e6)
        self.mix_master_box.setSingleStep(0.1)
        self.mix_master_box.valueChanged.connect(self.on_mix_master_changed)
        mix_layout.addRow("Master Mix:", self.mix_master_box)

        self.mix_boxes = []
        for d in range(latent_dim):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self.apply_mix_offsets)
            self.mix_boxes.append(box)
            mix_layout.addRow(f"Dim {d}:", box)
        mix_group.setLayout(mix_layout)

        # Encoding Offset
        offset_group = QtWidgets.QGroupBox("Encoding Offset")
        offset_layout = QtWidgets.QFormLayout()
        self.offset_master_box = QtWidgets.QDoubleSpinBox()
        self.offset_master_box.setRange(-1e6, 1e6)
        self.offset_master_box.setSingleStep(0.1)
        self.offset_master_box.valueChanged.connect(self.on_offset_master_changed)
        offset_layout.addRow("Master Offset:", self.offset_master_box)

        self.offset_boxes = []
        for d in range(latent_dim):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self.apply_mix_offsets)
            self.offset_boxes.append(box)
            offset_layout.addRow(f"Dim {d}:", box)
        offset_group.setLayout(offset_layout)

        scroll_layout.addWidget(mix_group)
        scroll_layout.addWidget(offset_group)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Playback Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Playback")
        self.start_btn.clicked.connect(self.start_playback)
        self.stop_btn = QtWidgets.QPushButton("Stop Playback")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.exit_app)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.exit_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        # Initialize File Bounds
        self.init_seq1_bounds()
        self.init_seq2_bounds()

    # --- Callbacks ---
    def on_device_changed(self, idx):
        if idx >= 0 and idx < len(self.device_mapping):
            real_device_id = self.device_mapping[idx]
            audio_controller.set_device(real_device_id)

    def init_seq1_bounds(self):
        duration = audio_durations_sec[file_index_1]
        self.seq1_start_box.blockSignals(True)
        self.seq1_end_box.blockSignals(True)
        self.seq1_start_box.setRange(0.0, duration)
        self.seq1_end_box.setRange(0.0, duration)
        self.seq1_start_box.setValue(0.0)
        self.seq1_end_box.setValue(duration)
        self.seq1_start_box.blockSignals(False)
        self.seq1_end_box.blockSignals(False)
        self.update_seq1_ranges()

    def init_seq2_bounds(self):
        duration = audio_durations_sec[file_index_2]
        self.seq2_start_box.blockSignals(True)
        self.seq2_end_box.blockSignals(True)
        self.seq2_start_box.setRange(0.0, duration)
        self.seq2_end_box.setRange(0.0, duration)
        self.seq2_start_box.setValue(0.0)
        self.seq2_end_box.setValue(duration)
        self.seq2_start_box.blockSignals(False)
        self.seq2_end_box.blockSignals(False)
        self.update_seq2_ranges()

    def update_seq1_ranges(self):
        global audio_source_start_frame_index_1, audio_source_end_frame_index_1, audio_source_frame_index_1
        start_val = self.seq1_start_box.value()
        end_val = self.seq1_end_box.value()
        duration = audio_durations_sec[file_index_1]

        self.seq1_start_box.setRange(0.0, end_val)
        self.seq1_end_box.setRange(start_val, duration)

        audio_source_start_frame_index_1 = int(start_val * audio_sample_rate)
        audio_source_end_frame_index_1 = int(end_val * audio_sample_rate)
        if audio_source_frame_index_1 < audio_source_start_frame_index_1 or audio_source_frame_index_1 > audio_source_end_frame_index_1:
            audio_source_frame_index_1 = audio_source_start_frame_index_1

    def update_seq2_ranges(self):
        global audio_source_start_frame_index_2, audio_source_end_frame_index_2, audio_source_frame_index_2
        start_val = self.seq2_start_box.value()
        end_val = self.seq2_end_box.value()
        duration = audio_durations_sec[file_index_2]

        self.seq2_start_box.setRange(0.0, end_val)
        self.seq2_end_box.setRange(start_val, duration)

        audio_source_start_frame_index_2 = int(start_val * audio_sample_rate)
        audio_source_end_frame_index_2 = int(end_val * audio_sample_rate)
        if audio_source_frame_index_2 < audio_source_start_frame_index_2 or audio_source_frame_index_2 > audio_source_end_frame_index_2:
            audio_source_frame_index_2 = audio_source_start_frame_index_2

    def on_seq1_idx_changed(self, val):
        global file_index_1, audio_source_frame_index_1
        file_index_1 = val
        audio_source_frame_index_1 = 0
        self.init_seq1_bounds()

    def on_seq2_idx_changed(self, val):
        global file_index_2, audio_source_frame_index_2
        file_index_2 = val
        audio_source_frame_index_2 = 0
        self.init_seq2_bounds()

    def on_mix_master_changed(self, val):
        for box in self.mix_boxes:
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.apply_mix_offsets()

    def on_offset_master_changed(self, val):
        for box in self.offset_boxes:
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.apply_mix_offsets()

    def apply_mix_offsets(self):
        for i in range(latent_dim):
            audio_encoding_mix_factors[i] = self.mix_boxes[i].value()
            audio_encoding_offset_factors[i] = self.offset_boxes[i].value()

    def start_playback(self):
        audio_controller.start()

    def stop_playback(self):
        audio_controller.stop()

    def exit_app(self):
        global producer_running
        self.stop_playback()
        producer_running = False
        self.close()

"""
OSC Server Mappings
"""
def osc_setIndex1(address, *args):
    global file_index_1
    if args[0] < len(loaded_audio_waveforms):
        file_index_1 = int(args[0])

def osc_setIndex2(address, *args):
    global file_index_2
    if args[0] < len(loaded_audio_waveforms):
        file_index_2 = int(args[0])
     
def osc_setRange1(address, *args):
    global audio_source_start_frame_index_1, audio_source_end_frame_index_1
    audio_source_start_frame_index_1 = int(args[0])
    audio_source_end_frame_index_1 = int(args[0] + args[1])

def osc_setRange2(address, *args):
    global audio_source_start_frame_index_2, audio_source_end_frame_index_2
    audio_source_start_frame_index_2 = int(args[0])
    audio_source_end_frame_index_2 = int(args[0] + args[1])

def osc_setEncodingMix(address, *args):
    for i in range(min(len(args), latent_dim)):
        audio_encoding_mix_factors[i] = args[i]

def osc_setEncodingOffset(address, *args):
    for i in range(min(len(args), latent_dim)):
        audio_encoding_offset_factors[i] = args[i]

osc_handler = dispatcher.Dispatcher()
osc_handler.map("/audio/sampleindex1", osc_setIndex1)
osc_handler.map("/audio/sampleindex2", osc_setIndex2)
osc_handler.map("/audio/samplerange1", osc_setRange1)
osc_handler.map("/audio/samplerange2", osc_setRange2)
osc_handler.map("/synth/encodingmix", osc_setEncodingMix)
osc_handler.map("/synth/encodingoffset", osc_setEncodingOffset)

"""
Execution
"""

def main():
    # Start OSC
    osc_server_instance = osc_server.ThreadingOSCUDPServer((osc_receive_ip, osc_receive_port), osc_handler)
    osc_thread = threading.Thread(target=osc_server_instance.serve_forever, daemon=True)
    osc_thread.start()

    # Start PyQt5 App
    app = QtWidgets.QApplication(sys.argv)
    gui = AudioAutoencoderGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()