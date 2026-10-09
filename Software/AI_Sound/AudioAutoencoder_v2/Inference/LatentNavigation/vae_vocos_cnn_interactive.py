"""
Latent-map (t-SNE) real-time synthesis with the fully convolutional VAE (audio_vae_vocos_cnn_v5.py)
+ fine-tuned Vocos, 48 kHz.  -- per-time-step latent map, continuous trajectory playback
------------------------------------------------------------------------------------------------
- Each map point is ONE latent vector (one latent time step, 1024 samples = 21.3 ms of audio).
- Drawn points form a path. For every synthesis window the 16 latent steps are sampled from the
  path by linear interpolation at a playhead that advances continuously (path wraps around).
- 'Point duration' sets how long each drawn point lasts (ms): the temporal development is
  defined by the drawing, not by fixed windows.
- Conditional macOS/MPS handling:
  * A startup probe checks whether the device supports the FFT ops used by Vocos (stft/irfft/istft).
    Only if not (e.g. MPS with old PyTorch) do the mel extractor and the Vocos ISTFT head run on CPU.
  * On MPS (or when the FFT fallback is active) the source waveform is kept on the CPU, because
    old MPS builds mishandle sliced views of device tensors (repeating windows -> "hammering").
  Windows/CUDA runs entirely on the device as before.
"""

import os
import sys
import time
import threading
import queue
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
from vocos import Vocos
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

"""
Compute Device
"""
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device (torch {torch.__version__})')

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

use_fp16_decoder = False       # try True on CUDA/MPS if step time exceeds the budget; check quality

force_fft_on_cpu = None           # None = auto-detect by probing; True/False = force
force_waveforms_on_cpu = None     # None = auto (MPS or FFT fallback); True/False = force

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

if force_waveforms_on_cpu is None:
    waveforms_on_cpu = (device == 'mps') or fft_on_cpu
else:
    waveforms_on_cpu = bool(force_waveforms_on_cpu)

if fft_on_cpu:
    print(f"FFT ops not available on {device}: mel extraction and Vocos ISTFT head run on CPU "
          f"(everything else stays on {device}).")
else:
    print(f"FFT ops available on {device}: running everything on {device}.")
print(f"Source waveform kept on {'CPU' if waveforms_on_cpu else device}.")

"""
Audio Settings
"""
audio_file = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
audio_sample_rate = 48000
audio_channels = 1
audio_output_device = 0

mel_hop = 256
window_mel_frames = 64                                  # -> 16128 samples (~336 ms)
window_size = mel_hop * (window_mel_frames - 1)
window_offset = window_size // 2                        # playback hop (50 % overlap)

time_downsample = 4
latent_step_samples = mel_hop * time_downsample         # 1024 samples per latent step

chunks_per_callback = 2
play_buffer_size = window_offset * chunks_per_callback
max_audio_queue_length = 6
playback_latency = 0.4
assert max_audio_queue_length > chunks_per_callback

"""
Map Settings
"""
latent_edge_margin = 2          # skip this many latent steps at each window edge (padding effects)
max_map_points = 8000           # random subsample cap for t-SNE speed
tsne_iterations = 1000
n_neighbors_default = 4
encode_batch_size = 16
default_point_duration_ms = 100.0   # matches the 100 ms drawing timer

"""
VAE Model Settings (must match audio_vae_vocos_cnn_v5.py)
"""
latent_dim = 32
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]
bottleneck_width = 512
mel_floor = -11.5

weights_dir = "data/models/vae_cnn_Gutenberg_ld32/"     # adjust to your v5 training run
weights_epoch = 400
ae_encoder_weights_file = os.path.join(weights_dir, f"encoder_weights_epoch_{weights_epoch}.pt")
ae_decoder_weights_file = os.path.join(weights_dir, f"decoder_weights_epoch_{weights_epoch}.pt")
ae_vocos_weights_file = os.path.join(weights_dir, f"vocos_weights_epoch_{weights_epoch}.pt")

"""
Load Audio
"""
assert os.path.exists(audio_file), f"Audio file not found: {audio_file}"
audio_waveform, _sr = torchaudio.load(audio_file)
if _sr != audio_sample_rate:
    audio_waveform = torchaudio.functional.resample(audio_waveform, _sr, audio_sample_rate)
audio_source_samples = audio_waveform[0].contiguous()
if not waveforms_on_cpu:
    audio_source_samples = audio_source_samples.to(device)
hann_window = torch.hann_window(window_size, periodic=True).to(device)

"""
Vocoder
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

"""
v5 Models
"""
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
    def forward(self, x):
        h = self.down(self.stem(x))
        h = h.flatten(1, 2)
        mu, raw = self.head(h).chunk(2, dim=1)
        std = F.softplus(raw) + 1e-4
        return mu, std

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
    def forward(self, z):
        h = self.inp(z)
        h = h.view(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))

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

# load_state_dict copies in place, so each submodule keeps its own device
_vw = load_required(ae_vocos_weights_file)
vocos.backbone.load_state_dict(_vw["backbone"])
vocos.head.load_state_dict(_vw["head"])
vocos.eval()

with torch.no_grad():
    _x = get_mels(torch.rand(1, window_size, device=device)).clamp(min=mel_floor).unsqueeze(1)
    _mu, _ = encoder(_x)
    latent_steps = _mu.shape[-1]
    assert _mu.shape[1] == latent_dim and latent_steps == mel_count // time_downsample
    assert decoder(_mu).shape == _x.shape
print(f"latent per window: {latent_dim} x {latent_steps} steps")
assert latent_steps > 2 * latent_edge_margin

"""
Per-time-step latent map
"""
@torch.inference_mode()
def encode_batch(wave_batch):
    mels = get_mels(wave_batch).clamp(min=mel_floor).unsqueeze(1)
    mu, _ = encoder(mels)
    return mu                                            # (B,Cz,T')

print("Generating latents for mapping...")
inner_steps = latent_steps - 2 * latent_edge_margin
excerpt_hop = inner_steps * latent_step_samples          # inner regions tile the audio
excerpt_starts = list(range(0, audio_source_samples.shape[0] - window_size + 1, excerpt_hop))
assert len(excerpt_starts) >= 2, "audio file too short for a latent map"

_vecs = []
for b in range(0, len(excerpt_starts), encode_batch_size):
    batch = torch.stack([audio_source_samples[s:s + window_size].clone()
                         for s in excerpt_starts[b:b + encode_batch_size]], 0).contiguous()
    mu = encode_batch(batch)[:, :, latent_edge_margin:latent_steps - latent_edge_margin]   # (B,Cz,inner)
    _vecs.append(mu.permute(0, 2, 1).reshape(-1, latent_dim).cpu())
audio_encodings = torch.cat(_vecs, dim=0).numpy().astype(np.float32)     # (M,Cz)

if audio_encodings.shape[0] > max_map_points:
    sel = np.random.default_rng(0).choice(audio_encodings.shape[0], max_map_points, replace=False)
    audio_encodings = audio_encodings[np.sort(sel)]
print(f"{audio_encodings.shape[0]} latent vectors in map")

print("Calculating t-SNE projection...")
tsne = TSNE(n_components=2, perplexity=min(30, audio_encodings.shape[0] - 1),
            max_iter=tsne_iterations, verbose=1)
Z_tsne = tsne.fit_transform(audio_encodings)

n_neighbors = min(n_neighbors_default, Z_tsne.shape[0])
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
knn.fit(Z_tsne)

def calc_distance_based_averaged_encoding(point2D):
    """Distance-weighted average latent vector (Cz,) for a 2D point of shape (1,2)."""
    _, indices = knn.kneighbors(point2D)
    idx = indices[0]
    d = np.linalg.norm(Z_tsne[idx] - point2D, axis=1)
    max_d = np.max(d)
    if max_d < 1e-6:
        return audio_encodings[idx[0]].copy()
    weights = 1.0 - d / max_d
    if weights.sum() < 1e-8:
        return audio_encodings[idx[0]].copy()
    return np.average(audio_encodings[idx], weights=weights, axis=0).astype(np.float32)

"""
Interactive latent path (shared between GUI and producer threads)
"""
inter_audio_encodings = []        # list of np.float32 (Cz,) vectors, in drawing order
encoding_lock = threading.Lock()
point_duration_ms = default_point_duration_ms
play_pos = 0.0                    # playhead in units of path points (producer thread only)
playing_index = -1                # path index currently audible (set in the audio callback)

@torch.inference_mode()
def decode_latents(z_np):
    z = torch.from_numpy(np.ascontiguousarray(z_np)).to(device).unsqueeze(0)    # (1,Cz,T')
    if use_fp16_decoder and device in ('cuda', 'mps'):
        with torch.autocast(device_type=device, dtype=torch.float16):
            gen_mels = decoder(z)[:, 0]
        gen_mels = gen_mels.float()
    else:
        gen_mels = decoder(z)[:, 0]
    wave = vocos_decode(gen_mels).reshape(-1)[:window_size] * hann_window
    return wave.float().cpu().numpy()

silence_chunk = np.zeros(window_size, dtype=np.float32)
_step_idx = np.arange(latent_steps, dtype=np.float64) + 0.5

def synthesize_audio():
    """Sample the path at the playhead -> latent sequence -> one Hann-windowed audio chunk."""
    global play_pos
    with encoding_lock:
        n = len(inter_audio_encodings)
        if n == 0:
            return silence_chunk, -1
        P = np.stack(inter_audio_encodings, axis=0)                  # (n,Cz)

    dur_samples = max(point_duration_ms, 1.0) * audio_sample_rate / 1000.0
    pts_per_step = latent_step_samples / dur_samples                 # path points per latent step

    pos = (play_pos % n) + _step_idx * pts_per_step                  # (T',) in points
    if n == 1:
        Z = np.repeat(P[:1], latent_steps, axis=0)
        center_idx = 0
    else:
        pos = pos % n
        i0 = np.floor(pos).astype(np.int64)
        frac = (pos - i0).astype(np.float32)[:, None]
        i1 = (i0 + 1) % n
        Z = P[i0] * (1.0 - frac) + P[i1] * frac                      # (T',Cz), wraps last->first
        center_idx = int(i0[latent_steps // 2])

    play_pos = (play_pos + window_offset / dur_samples) % n
    return decode_latents(Z.T), center_idx                           # Z.T: (Cz,T')

"""
Audio Streaming
"""
audio_queue = queue.Queue(maxsize=max_audio_queue_length)
overlap_tail = np.zeros(window_offset, dtype=np.float32)
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
            item = synthesize_audio()
            if item[1] >= 0:
                step_times.append(time.perf_counter() - t0)
            audio_queue.put(item)
            if len(step_times) >= 20:
                budget = 1000.0 * window_offset / audio_sample_rate
                print(f"avg step {1000*np.mean(step_times):.1f} ms (budget {budget:.1f} ms), "
                      f"queue {audio_queue.qsize()}/{max_audio_queue_length}, underruns {underrun_count}")
                step_times = []
        except Exception as e:
            print(f"Producer error: {e}")
            time.sleep(0.1)

def audio_callback(out_data, frames, time_info, status):
    global underrun_count, playing_index
    if status:
        print(f"Audio status: {status}")
    out = out_data[:, 0]
    cursor, hop = 0, window_offset
    while cursor < frames:
        n = min(hop, frames - cursor)
        try:
            chunk, idx = audio_queue.get_nowait()
            playing_index = idx
        except queue.Empty:
            chunk = silence_chunk
            underrun_count += 1
        out[cursor:cursor + n] = overlap_tail[:n] + chunk[:n]
        if n == hop:
            overlap_tail[:] = chunk[hop:]
        cursor += n

class AudioStreamController:
    def __init__(self):
        self.stream = None
        self.is_playing = False
        self.device_id = audio_output_device
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
                    channels=audio_channels,
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

def warm_up():
    z = np.repeat(audio_encodings[:1], latent_steps, axis=0).T
    t0 = time.perf_counter(); decode_latents(z)
    t1 = time.perf_counter(); decode_latents(z)
    t2 = time.perf_counter()
    budget = 1000.0 * window_offset / audio_sample_rate
    print(f"decode warm-up {1000*(t1-t0):.1f} ms, steady {1000*(t2-t1):.1f} ms (budget per hop {budget:.1f} ms)")

"""
Interactive scatter Qt window
"""
class ScatterPlotApp(QtWidgets.QMainWindow):
    """
    - Left button press/drag: add points (one latent vector each, kNN-averaged).
    - Right button press/drag: remove points near the cursor.
    - Middle button: pan.  'C' key / Clear button: remove all points.
    - 'Point duration': how long each drawn point lasts in the audio (path is interpolated).
    """
    def __init__(self, points2D):
        super().__init__()
        self.setWindowTitle("Latent Mapping Autoencoder (v5, per-step latents)")
        self.resize(800, 700)

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.graph_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.graph_widget)

        self.scatter = pg.ScatterPlotItem(x=points2D[:, 0], y=points2D[:, 1], pen=pg.mkPen(None),
                                          brush=pg.mkBrush(100, 100, 255, 120), size=6)
        self.graph_widget.addItem(self.scatter)

        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.graph_widget.addItem(self.click_scatter)

        self.play_scatter = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(0, 255, 0, 220))
        self.graph_widget.addItem(self.play_scatter)

        self.controls_layout = QtWidgets.QHBoxLayout()
        self.device_layout = QtWidgets.QHBoxLayout()
        self.device_label = QtWidgets.QLabel("Output Device:")
        self.device_combo = QtWidgets.QComboBox()

        self.device_mapping = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0:
                self.device_mapping.append(i)
                self.device_combo.addItem(f"{i}: {dev['name']}")
                if i == audio_output_device:
                    self.device_combo.setCurrentIndex(len(self.device_mapping) - 1)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)

        self.device_layout.addWidget(self.device_label)
        self.device_layout.addWidget(self.device_combo)
        self.controls_layout.addLayout(self.device_layout)

        self.duration_label = QtWidgets.QLabel("Point duration (ms):")
        self.duration_box = QtWidgets.QDoubleSpinBox()
        self.duration_box.setRange(10.0, 5000.0)
        self.duration_box.setSingleStep(10.0)
        self.duration_box.setValue(default_point_duration_ms)
        self.duration_box.valueChanged.connect(self.on_duration_changed)
        self.controls_layout.addWidget(self.duration_label)
        self.controls_layout.addWidget(self.duration_box)

        self.start_btn = QtWidgets.QPushButton("Start Playback")
        self.start_btn.clicked.connect(self.start_playback)
        self.stop_btn = QtWidgets.QPushButton("Stop Playback")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clearInteractiveEncodings)
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.exit_app)

        self.controls_layout.addWidget(self.start_btn)
        self.controls_layout.addWidget(self.stop_btn)
        self.controls_layout.addWidget(self.clear_btn)
        self.controls_layout.addWidget(self.exit_btn)
        self.main_layout.addLayout(self.controls_layout)

        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.right_button_pressed = False
        self.last_mouse_pos = None

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.continuous_add_point)

        self.playpoint_timer = QtCore.QTimer()
        self.playpoint_timer.setInterval(25)
        self.playpoint_timer.timeout.connect(self.update_play_point)
        self.playpoint_timer.start()

        self.graph_widget.scene().installEventFilter(self)

    def on_device_changed(self, idx):
        if 0 <= idx < len(self.device_mapping):
            audio_controller.set_device(self.device_mapping[idx])

    def on_duration_changed(self, val):
        global point_duration_ms
        point_duration_ms = float(val)

    def start_playback(self):
        audio_controller.start()

    def stop_playback(self):
        audio_controller.stop()

    def exit_app(self):
        global producer_running
        self.stop_playback()
        producer_running = False
        self.close()

    def addInteractiveEncoding(self, mouse_point):
        z = calc_distance_based_averaged_encoding(np.array([[mouse_point.x(), mouse_point.y()]]))
        with encoding_lock:
            inter_audio_encodings.append(z)

    def removeInteractiveEncodingNear(self, mouse_point, radius=0.5):
        mp_x, mp_y = mouse_point.x(), mouse_point.y()
        to_remove = [i for i, p in enumerate(self.click_points)
                     if ((p['pos'][0] - mp_x) ** 2 + (p['pos'][1] - mp_y) ** 2) ** 0.5 < radius]
        with encoding_lock:
            for idx in reversed(to_remove):
                self.click_points.pop(idx)
                inter_audio_encodings.pop(idx)
        self.click_scatter.setData([p['pos'][0] for p in self.click_points],
                                   [p['pos'][1] for p in self.click_points])

    def clearInteractiveEncodings(self):
        global play_pos
        with encoding_lock:
            self.click_points.clear()
            inter_audio_encodings.clear()
            play_pos = 0.0
        self.click_scatter.setData([], [])
        self.play_scatter.setData([], [])

    def eventFilter(self, source, event):
        if source == self.graph_widget.scene():
            if event.type() == QtCore.QEvent.GraphicsSceneMousePress:
                if event.button() == QtCore.Qt.LeftButton:
                    self.left_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    self.add_point_at(event.scenePos())
                    self.timer.start()
                    return True
                elif event.button() == QtCore.Qt.MiddleButton:
                    self.middle_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    return True
                elif event.button() == QtCore.Qt.RightButton:
                    self.right_button_pressed = True
                    self.last_mouse_pos = event.scenePos()
                    return True
            elif event.type() == QtCore.QEvent.GraphicsSceneMouseRelease:
                if event.button() == QtCore.Qt.LeftButton:
                    self.left_button_pressed = False
                    self.timer.stop()
                    return True
                elif event.button() == QtCore.Qt.MiddleButton:
                    self.middle_button_pressed = False
                    return True
                elif event.button() == QtCore.Qt.RightButton:
                    self.right_button_pressed = False
                    return True
            elif event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
                if self.left_button_pressed:
                    self.last_mouse_pos = event.scenePos()
                    return True
                elif self.middle_button_pressed:
                    if self.last_mouse_pos is not None:
                        diff = event.scenePos() - self.last_mouse_pos
                        vb = self.graph_widget.plotItem.vb
                        vb.translateBy(x=-diff.x(), y=diff.y())
                        self.last_mouse_pos = event.scenePos()
                    return True
                elif self.right_button_pressed:
                    vb = self.graph_widget.plotItem.vb
                    self.removeInteractiveEncodingNear(vb.mapSceneToView(event.scenePos()))
                    self.last_mouse_pos = event.scenePos()
                    return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_C:
            self.clearInteractiveEncodings()
        super().keyPressEvent(event)

    def add_point_at(self, scene_pos):
        if self.graph_widget.sceneBoundingRect().contains(scene_pos):
            vb = self.graph_widget.plotItem.vb
            mouse_point = vb.mapSceneToView(scene_pos)
            self.addInteractiveEncoding(mouse_point)
            self.click_points.append({'pos': (mouse_point.x(), mouse_point.y())})
            self.click_scatter.setData([p['pos'][0] for p in self.click_points],
                                       [p['pos'][1] for p in self.click_points])

    def continuous_add_point(self):
        if self.left_button_pressed and self.last_mouse_pos is not None:
            self.add_point_at(self.last_mouse_pos)

    def update_play_point(self):
        idx = playing_index
        if 0 <= idx < len(self.click_points):
            p = self.click_points[idx]
            self.play_scatter.setData([p['pos'][0]], [p['pos'][1]])

"""
Entrypoint
"""
def main():
    warm_up()
    audio_controller.start_producer()
    app = QtWidgets.QApplication(sys.argv)
    win = ScatterPlotApp(Z_tsne)
    win.show()
    print("Qt window shown; entering event loop")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()