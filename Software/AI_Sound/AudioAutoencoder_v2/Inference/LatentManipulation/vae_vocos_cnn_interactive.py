"""
Real-time latent mixing with the fully convolutional VAE (audio_vae_vocos_cnn_v5.py) + fine-tuned Vocos
------------------------------------------------------------------------------------------------------
Revised for glitch-free real-time playback:
- Queue is deeper than the number of chunks consumed per audio callback (previously 4 < 8 -> silence gaps).
- Smaller audio blocks, clean hop-based overlap-add in the callback.
- Latent encodings are cached (they do not depend on mix/offset), and unused branches are skipped.
- Mix/offset live in CPU numpy arrays (no per-scalar GPU writes from GUI/OSC threads).
- Optional fp16 autocast for the decoder, startup warm-up with timing report, underrun counter.
"""

import os
import sys
import time
import numpy as np
import threading
import queue

import torch
from torch import nn
import torch.nn.functional as F
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

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

use_fp16_decoder = False   # try True on CUDA/MPS; check audio quality (Vocos itself stays fp32)
force_fft_on_cpu = None # force the CPU fallback on/off

"""
Audio Settings
"""

audio_file_paths = [
    "data/audio/Take1__double_Bind_HQ_audio_crop_48khz.wav",
    "data/audio/Take2_Hibr_II_HQ_audio_crop_48khz.wav"
]

audio_sample_rate = 48000
audio_channel_count = 1
default_audio_output_device = 0

# window = hop * (4k - 1) so that the mel frame count is divisible by time_downsample
mel_hop = 256
window_mel_frames = 64                                  # 64 frames -> 16128 samples (~336 ms)
window_size = mel_hop * (window_mel_frames - 1)
gen_buffer_size = window_size
window_overlap = 8             # windows overlap by (1 - 1/window_overlap): 2 = 50 %, 4 = 75 %, 8 = 87.5 %
window_power = 2               # window shape: 1 = Hann, 2 = Hann^2 (suppresses weak window edges)
assert window_overlap >= 2 and window_size % window_overlap == 0
window_offset = window_size // window_overlap           # hop between windows (42 ms at overlap 8)

chunks_per_callback = window_overlap                    # audio block = one window length
play_buffer_size = window_offset * chunks_per_callback
max_audio_queue_length = 3 * chunks_per_callback        # about 1 s of audio ahead
playback_latency = 0.5                                  # seconds; raise if you still hear underruns
assert max_audio_queue_length > chunks_per_callback

"""
Autoencoder Settings (must match audio_vae_vocos_cnn_v5.py)
"""

latent_dim = 32
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]         # (freq, time)
bottleneck_width = 512
time_downsample = 4
mel_floor = -11.5
sample_posterior = False

weights_dir = "data/models/vae_cnn_Stocos_ld32/"
weights_epoch = 400
ae_encoder_weights_file = os.path.join(weights_dir, f"encoder_weights_epoch_{weights_epoch}.pt")
ae_decoder_weights_file = os.path.join(weights_dir, f"decoder_weights_epoch_{weights_epoch}.pt")
ae_vocos_weights_file = os.path.join(weights_dir, f"vocos_weights_epoch_{weights_epoch}.pt")

"""
OSC Control Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9005

"""
Check CPU Fallback for MacOS
"""

def device_supports_fft(dev):
    """Probe the FFT ops used by the Vocos feature extractor and ISTFT head on the given device."""
    if dev == 'cpu':
        return True
    try:
        n_fft, hop = 1024, 256
        x = torch.rand(1, 8192, device=dev)
        win = torch.hann_window(n_fft, device=dev)
        spec = torch.stft(x, n_fft, hop_length=hop, win_length=n_fft, window=win,
                          center=True, return_complex=True)                       # _fft_r2c
        _ = torch.fft.irfft(spec, n_fft, dim=1, norm="backward")                  # _fft_c2r
        _ = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft, window=win)
        if dev == 'mps':
            torch.mps.synchronize()
        elif dev == 'cuda':
            torch.cuda.synchronize()
        return True
    except (NotImplementedError, RuntimeError) as e:
        print(f"FFT probe failed on {dev}: {str(e).splitlines()[0]}")
        return False


if force_fft_on_cpu is None:
    fft_on_cpu = not device_supports_fft(device)
else:
    fft_on_cpu = bool(force_fft_on_cpu)

if fft_on_cpu:
    print(f"FFT ops not available on {device}: mel extraction and Vocos ISTFT head will run on CPU "
          f"(everything else stays on {device}).")
else:
    print(f"FFT ops available on {device}: running everything on {device}.")

# Old MPS builds mishandle sliced views of device tensors (windows repeat -> "hammering" loop).
# Keep the source waveforms on the CPU in that case and copy only the excerpt to the device.
force_waveforms_on_cpu = None     # None = auto (MPS or FFT fallback), True/False = force
if force_waveforms_on_cpu is None:
    waveforms_on_cpu = (device == 'mps') or fft_on_cpu
else:
    waveforms_on_cpu = bool(force_waveforms_on_cpu)
print(f"Source waveforms kept on {'CPU' if waveforms_on_cpu else device}.")

"""
Load Audio
"""

loaded_audio_waveforms = []
audio_durations_sec = []

for filepath in audio_file_paths:
    assert os.path.exists(filepath), f"Audio file not found: {filepath}"
    waveform, sr = torchaudio.load(filepath)
    if sr != audio_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, audio_sample_rate)
    samples = waveform[0].contiguous()
    if not waveforms_on_cpu:
        samples = samples.to(device)
    loaded_audio_waveforms.append(samples)
    audio_durations_sec.append(samples.shape[0] / audio_sample_rate)

hann_window = (torch.hann_window(window_size, periodic=True) ** window_power).to(device)
_w = hann_window.cpu().numpy().astype(np.float64)
hop_norm = _w.reshape(window_overlap, window_offset).sum(axis=0)    # sum over overlapping windows per hop slot
assert hop_norm.min() > 1e-3
inv_hop_norm = (1.0 / hop_norm).astype(np.float32)

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

# Master copies live on the CPU; GUI/OSC threads only touch these numpy arrays.
audio_encoding_mix_factors = np.zeros(latent_dim, dtype=np.float32)
audio_encoding_offset_factors = np.zeros(latent_dim, dtype=np.float32)

"""
Create Models
"""

print("Loading Vocos Model...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
for p in vocos.parameters():
    p.requires_grad = False

# Only if the device lacks FFT kernels: keep the FFT-dependent submodules on the CPU.
if fft_on_cpu:
    vocos.feature_extractor.to('cpu')
    vocos.head.to('cpu')

def get_mels(wave):
    if fft_on_cpu:
        m = vocos.feature_extractor(wave.cpu()).to(device)
    else:
        m = vocos.feature_extractor(wave.to(device))
    if m.ndim == 4:
        m = m[:, 0]
    return m.contiguous() if waveforms_on_cpu else m

def vocos_decode(mel):
    """mel: (1,F,T) on `device` -> waveform on `device`."""
    if fft_on_cpu:
        h = vocos.backbone(mel)                       # on device
        return vocos.head(h.cpu()).to(device)         # linear + ISTFT on CPU
    return vocos.decode(mel)

with torch.no_grad():
    _m = get_mels(torch.rand(1, window_size, device=device))
    mel_filters, mel_count = _m.shape[1], _m.shape[2]
    assert mel_count % time_downsample == 0, \
        f"mel frame count {mel_count} must be divisible by {time_downsample}; use window = hop*(4k-1)"
    assert vocos_decode(_m).shape[-1] == window_size, "Vocos output length differs from window_size"
print(f"mel frames {mel_count}, mel bins {mel_filters}, window {window_size} samples")

def gn(ch):
    return nn.GroupNorm(8, ch)

class ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(gn(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 gn(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))
    def forward(self, x):
        return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, latent_channels, mel_filter_count, channels, strides, width):
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        layers, c_in = [], channels[0]
        for c_out, s in zip(channels, strides):
            layers += [ResBlock2d(c_in), gn(c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, stride=s, padding=1)]
            c_in = c_out
        self.down = nn.Sequential(*layers)
        f_red = int(np.prod([s[0] for s in strides]))
        self.flat_ch = channels[-1] * (mel_filter_count // f_red)
        self.head = nn.Sequential(nn.Conv1d(self.flat_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, 2 * latent_channels, 3, padding=1))
    def forward(self, x):  # x: (B,1,F,T)
        h = self.down(self.stem(x))
        h = h.flatten(1, 2)
        mu, raw = self.head(h).chunk(2, dim=1)
        std = F.softplus(raw) + 1e-4
        return mu, std
    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)

class Decoder(nn.Module):
    def __init__(self, latent_channels, mel_filter_count, channels, strides, width):
        super().__init__()
        rc, rs = channels[::-1], strides[::-1]
        f_red = int(np.prod([s[0] for s in strides]))
        self.c0, self.f0 = rc[0], mel_filter_count // f_red
        self.inp = nn.Sequential(nn.Conv1d(latent_channels, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv1d(width, self.c0 * self.f0, 3, padding=1))
        layers = []
        for i, s in enumerate(rs):
            c_in = rc[i]
            c_out = rc[i + 1] if i + 1 < len(rc) else rc[-1]
            layers += [ResBlock2d(c_in), gn(c_in), nn.SiLU(), nn.Upsample(scale_factor=s, mode="nearest"),
                       nn.Conv2d(c_in, c_out, 3, padding=1)]
        self.up = nn.Sequential(*layers)
        self.out = nn.Sequential(gn(rc[-1]), nn.SiLU(), nn.Conv2d(rc[-1], 1, 3, padding=1))
    def forward(self, z):  # z: (B,Cz,T')
        h = self.inp(z)
        h = h.view(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))  # (B,1,F,T)

def load_required(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

encoder = Encoder(latent_dim, mel_filters, conv_channel_counts, conv_strides, bottleneck_width).to(device)
encoder.load_state_dict(load_required(ae_encoder_weights_file))
encoder.eval()

decoder = Decoder(latent_dim, mel_filters, conv_channel_counts, conv_strides, bottleneck_width).to(device)
decoder.load_state_dict(load_required(ae_decoder_weights_file))
decoder.eval()

_vw = load_required(ae_vocos_weights_file)
vocos.backbone.load_state_dict(_vw["backbone"])
vocos.head.load_state_dict(_vw["head"])
vocos.eval()

with torch.no_grad():
    _x = get_mels(torch.rand(1, window_size, device=device)).clamp(min=mel_floor).unsqueeze(1)
    _mu, _ = encoder(_x)
    assert _mu.shape[1] == latent_dim and _mu.shape[-1] == mel_count // time_downsample
    assert decoder(_mu).shape == _x.shape
print("latent shape per window:", tuple(_mu.shape))

"""
Audio Synthesis Loop
"""

def get_excerpt(samples, start):
    x = samples[start:start + window_size]
    if waveforms_on_cpu:
        x = x.clone()                                   # own storage, no offset view
    if x.shape[0] < window_size:
        x = F.pad(x, (0, window_size - x.shape[0]))
    x = x.unsqueeze(0)
    return x.contiguous() if waveforms_on_cpu else x

@torch.inference_mode()
def encode_excerpt(waveform):
    mels = get_mels(waveform).clamp(min=mel_floor).unsqueeze(1)  # (1,1,F,T)
    mu, std = encoder(mels)
    return Encoder.reparameterize(mu, std) if sample_posterior else mu  # (1,Cz,T')

# Encodings depend only on (file, start position), not on mix/offset -> cache them.
# Only the producer thread touches this dict.
encoding_cache = {}
ENCODING_CACHE_MAX = 20000 # 4096

def cached_encoding(file_idx, start):
    if sample_posterior:  # stochastic: caching would freeze the noise
        return encode_excerpt(get_excerpt(loaded_audio_waveforms[file_idx], start))
    key = (file_idx, start)
    enc = encoding_cache.get(key)
    if enc is None:
        if len(encoding_cache) >= ENCODING_CACHE_MAX:
            encoding_cache.clear()
        enc = encode_excerpt(get_excerpt(loaded_audio_waveforms[file_idx], start))
        encoding_cache[key] = enc
    return enc

def advance_index(idx, start, end):
    idx += window_offset
    if idx >= end - window_size:
        idx = start
    return idx

@torch.inference_mode()
def synthesize_audio():
    global audio_source_frame_index_1, audio_source_frame_index_2

    # snapshot control values (CPU numpy, cheap)
    mix_np = audio_encoding_mix_factors.copy()
    off_np = audio_encoding_offset_factors.copy()
    f1, f2 = file_index_1, file_index_2
    i1, i2 = audio_source_frame_index_1, audio_source_frame_index_2

    if np.all(mix_np == 0.0):
        decoder_in = cached_encoding(f1, i1)
    elif np.all(mix_np == 1.0):
        decoder_in = cached_encoding(f2, i2)
    else:
        mix = torch.from_numpy(mix_np).to(device).view(1, latent_dim, 1)
        decoder_in = cached_encoding(f1, i1) * (1.0 - mix) + cached_encoding(f2, i2) * mix

    offset = torch.from_numpy(off_np).to(device).view(1, latent_dim, 1)
    decoder_in = decoder_in + offset                      # (1,Cz,T')

    if use_fp16_decoder and device in ('cuda', 'mps'):
        with torch.autocast(device_type=device, dtype=torch.float16):
            gen_mels = decoder(decoder_in)[:, 0]
        gen_mels = gen_mels.float()
    else:
        gen_mels = decoder(decoder_in)[:, 0]              # (1,F,T)

    gen_waveform = vocos_decode(gen_mels).reshape(-1)[:window_size]
    gen_waveform = gen_waveform * hann_window             # window once, here

    audio_source_frame_index_1 = advance_index(i1, audio_source_start_frame_index_1, audio_source_end_frame_index_1)
    audio_source_frame_index_2 = advance_index(i2, audio_source_start_frame_index_2, audio_source_end_frame_index_2)

    return gen_waveform

"""
Audio Threading
"""
audio_queue = queue.Queue(maxsize=max_audio_queue_length)
overlap_accum = np.zeros(window_size, dtype=np.float32)
silence_chunk = np.zeros(window_size, dtype=np.float32)
producer_running = True
underrun_count = 0
step_times = []

def producer_thread():
    global step_times
    while producer_running:
        if audio_queue.full():
            time.sleep(0.005)
            continue
        try:
            t0 = time.perf_counter()
            wave = synthesize_audio().float().cpu().numpy()   # .cpu() syncs the device
            dt = time.perf_counter() - t0
            audio_queue.put(wave)
            step_times.append(dt)
            if len(step_times) >= 20:
                avg_ms = 1000.0 * float(np.mean(step_times))
                budget_ms = 1000.0 * window_offset / audio_sample_rate
                print(f"avg step {avg_ms:.1f} ms (budget {budget_ms:.1f} ms), "
                      f"queue {audio_queue.qsize()}/{max_audio_queue_length}, underruns {underrun_count}")
                step_times = []
        except Exception as e:
            print(f"Producer error: {e}")
            time.sleep(0.1)

def audio_callback(out_data, frames, time_info, status):
    # Each queue item is a windowed chunk of window_size samples, one new chunk per hop.
    # The accumulator holds the sum of all chunks that overlap the current hop slot.
    global underrun_count
    if status:
        print(f"Audio status: {status}")
    out = out_data[:, 0]
    out[:] = 0.0
    hop = window_offset
    cursor = 0
    while cursor + hop <= frames:
        try:
            chunk = audio_queue.get_nowait()
        except queue.Empty:
            chunk = silence_chunk
            underrun_count += 1
        overlap_accum[:] += chunk
        out[cursor:cursor + hop] = overlap_accum[:hop] * inv_hop_norm
        overlap_accum[:-hop] = overlap_accum[hop:]
        overlap_accum[-hop:] = 0.0
        cursor += hop

class AudioStreamController:
    def __init__(self):
        self.stream = None
        self.is_playing = False
        self.device_id = default_audio_output_device
        self.producer = None

    def start_producer(self):
        if self.producer is None:
            self.producer = threading.Thread(target=producer_thread, daemon=True)
            self.producer.start()

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
                    dtype='float32',
                    callback=audio_callback,
                    blocksize=play_buffer_size,
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

def warm_up(n=4):
    """Run a few synthesis steps (compiles kernels, fills cache) and report timing."""
    global audio_source_frame_index_1, audio_source_frame_index_2
    s1, s2 = audio_source_frame_index_1, audio_source_frame_index_2
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        synthesize_audio().cpu()
        times.append(time.perf_counter() - t0)
    audio_source_frame_index_1, audio_source_frame_index_2 = s1, s2
    budget_ms = 1000.0 * window_offset / audio_sample_rate
    print(f"warm-up step times (ms): {[round(1000 * t, 1) for t in times]}  | real-time budget {budget_ms:.1f} ms")

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
        files_layout = QtWidgets.QHBoxLayout()

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

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QHBoxLayout(scroll_widget)

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

        self.init_seq1_bounds()
        self.init_seq2_bounds()

    def on_device_changed(self, idx):
        if 0 <= idx < len(self.device_mapping):
            audio_controller.set_device(self.device_mapping[idx])

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
        audio_encoding_mix_factors[:] = [b.value() for b in self.mix_boxes]
        audio_encoding_offset_factors[:] = [b.value() for b in self.offset_boxes]

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
    if 0 <= args[0] < len(loaded_audio_waveforms):
        file_index_1 = int(args[0])

def osc_setIndex2(address, *args):
    global file_index_2
    if 0 <= args[0] < len(loaded_audio_waveforms):
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
    n = min(len(args), latent_dim)
    audio_encoding_mix_factors[:n] = np.asarray(args[:n], dtype=np.float32)

def osc_setEncodingOffset(address, *args):
    n = min(len(args), latent_dim)
    audio_encoding_offset_factors[:n] = np.asarray(args[:n], dtype=np.float32)

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
    osc_server_instance = osc_server.ThreadingOSCUDPServer((osc_receive_ip, osc_receive_port), osc_handler)
    threading.Thread(target=osc_server_instance.serve_forever, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    gui = AudioAutoencoderGUI()
    warm_up()                          # after the GUI has set the file ranges
    audio_controller.start_producer()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()