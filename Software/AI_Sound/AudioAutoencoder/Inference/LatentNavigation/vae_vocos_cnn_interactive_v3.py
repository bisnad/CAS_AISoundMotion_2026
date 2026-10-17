"""
Audio Autoencoder for Vocos 48khz - latent space painting (rolling mel buffer version)
------------------------------------------------------------------------------------------
- Loads audio, VAE models, and vocoder.
- Encodes 8-frame mel blocks (cut from a longer STFT, so edges are not reflect-padded) and
  maps their latents to 2D with t-SNE.
- GUI: paint points in the 2D map; each point's encoding is a distance-weighted kNN average.
- Real-time synthesis: the painted encodings are cycled; each is decoded to an 8-frame mel block
  that is appended to a rolling buffer of generated mels. Vocos decodes the WHOLE buffer and only
  the segment ending `lookahead_frames` before the right edge is output (contiguous, no overlap-add).
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import torch
from torch import nn
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
print(f'Using {device} device')

"""
Audio Settings
"""
audio_file = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"

audio_sample_rate = 48000
audio_channels = 1
audio_window_length_vae = 1792   # samples worth of one VAE block (8 mel frames)
audio_output_device = 0

# audio callback block size (small -> low output latency, large -> more robust)
playback_block_size = 1024
playback_latency = 'high'

# Rolling mel buffer / Vocos settings
vocos_hop = 256                  # samples per mel frame (must match Vocos)
buffer_frames = 128              # generated mel frames kept and fed to Vocos on every decode
lookahead_frames = 16            # right-edge frames held back; latency ~ lookahead_frames*hop samples
blocks_per_step = 2              # latents decoded per Vocos call (1 = lowest latency, most compute)
max_audio_queue_length = 4       # number of generated segments buffered ahead
use_mean_encoding = True         # True: use mu only for the map; False: sample like in training

"""
VAE Model Settings
"""
latent_dim = 32
ae_conv_channel_counts = [16, 32, 64, 128]
ae_conv_kernel_size = (5, 3)
ae_dense_layer_sizes = [512]
ae_encoder_weights_file = "data/models/vae_cnn_Gutenberg_ld32/encoder_weights_epoch_400.pt"
ae_decoder_weights_file = "data/models/vae_cnn_Gutenberg_ld32/decoder_weights_epoch_400.pt"

"""
Load Audio
"""
assert os.path.exists(audio_file), f"Audio file not found: {audio_file}"
audio_waveform, _ = torchaudio.load(audio_file)

"""
Vocoder Model
"""
print("Loading Vocos Model...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
for param in vocos.parameters():
    param.requires_grad = False

with torch.no_grad():
    f_vae = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vae), device=device))
    # reflect padding needs an input longer than n_fft/2 (1024), hence 16 hops
    silent_mels = vocos.feature_extractor(torch.zeros(size=(1, 16 * vocos_hop), device=device))[..., 4:5]  # (1,F,1)
audio_mel_filter_count = f_vae.shape[1]
audio_mel_count_vae = f_vae.shape[-1]
block_samples = audio_mel_count_vae * vocos_hop

try:
    stft_n_fft = int(vocos.feature_extractor.mel_spec.n_fft)
except Exception:
    stft_n_fft = 2048
    print("Could not read n_fft from Vocos feature extractor, assuming", stft_n_fft)
stft_context_frames = int(np.ceil((stft_n_fft // 2) / vocos_hop)) + 1
slice_frames = audio_mel_count_vae + 2 * stft_context_frames
slice_samples = (slice_frames - 1) * vocos_hop

step_frames = blocks_per_step * audio_mel_count_vae
step_samples = step_frames * vocos_hop
assert buffer_frames % audio_mel_count_vae == 0, "buffer_frames must be a multiple of the VAE block length"
assert lookahead_frames + step_frames < buffer_frames, "buffer too small for step + lookahead"
with torch.no_grad():
    t_slice = vocos.feature_extractor(torch.rand(size=(1, slice_samples), device=device)).shape[-1]
    assert t_slice == slice_frames, f"unexpected mel frame count {t_slice} vs {slice_frames}"
    t_dec = vocos.decode(vocos.feature_extractor(torch.rand(size=(1, (buffer_frames - 1) * vocos_hop), device=device))).shape[-1]
    assert t_dec == (buffer_frames - 1) * vocos_hop, "vocos decode length must be (frames-1)*hop"
print(f"VAE block: {audio_mel_count_vae} frames, STFT context: {stft_context_frames} frames, "
      f"step: {step_samples} samples ({step_samples / audio_sample_rate * 1000:.0f} ms), "
      f"lookahead: {lookahead_frames * vocos_hop / audio_sample_rate * 1000:.0f} ms")

# zero-padded waveform so that slices at the file borders are always valid
pad_samples = slice_samples + step_samples
padded_waveform = torch.nn.functional.pad(audio_waveform[0].to(device), (pad_samples, pad_samples))

"""
Create Models
"""
class Encoder(nn.Module):
    def __init__(self, latent_dim, mel_count, mel_filters,
                 conv_channels, conv_kernel, dense_sizes):
        super().__init__()
        stride = ((conv_kernel[0] - 1) // 2, (conv_kernel[1] - 1) // 2)
        padding = stride

        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, conv_kernel, stride=stride, padding=padding),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_ch)
            ]
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        last_x = int(mel_filters // np.power(stride[0], len(conv_channels)))
        last_y = int(mel_count // np.power(stride[1], len(conv_channels)))
        dense_in = conv_channels[-1] * last_x * last_y

        dense = []
        in_size = dense_in
        for size in dense_sizes:
            dense += [nn.Linear(in_size, size), nn.ReLU()]
            in_size = size
        self.dense_layers = nn.Sequential(*dense)

        self.fc_mu = nn.Linear(dense_sizes[-1], latent_dim)
        self.fc_std = nn.Linear(dense_sizes[-1], latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        return mu, std

    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)


class Decoder(nn.Module):
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes

        self.dense_layers = nn.ModuleList()
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)

        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())

        for layer_index in range(1, len(dense_layer_sizes)):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())

        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y

        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)

        self.conv_layers = nn.ModuleList()
        same_padding = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)

        for layer_index in range(1, len(conv_channel_counts)):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            # first upsampling stage: no bias, to avoid seeding a DC-offset spectral peak
            use_bias = layer_index != 1
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=1, padding=same_padding, bias=use_bias))
            self.conv_layers.append(nn.LeakyReLU(0.2))

        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.Upsample(scale_factor=stride, mode="nearest"))
        self.conv_layers.append(nn.Conv2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=1, padding=same_padding))

    def forward(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        x = self.unflatten(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x


encoder = Encoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)
decoder = Decoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, list(reversed(ae_conv_channel_counts)), ae_conv_kernel_size, list(reversed(ae_dense_layer_sizes))).to(device)

if os.path.exists(ae_encoder_weights_file):
    encoder.load_state_dict(torch.load(ae_encoder_weights_file, map_location=device))
else:
    print("WARNING: encoder weights not found:", ae_encoder_weights_file)
if os.path.exists(ae_decoder_weights_file):
    decoder.load_state_dict(torch.load(ae_decoder_weights_file, map_location=device))
else:
    print("WARNING: decoder weights not found:", ae_decoder_weights_file)

encoder.eval()
decoder.eval()

"""
Latent encodings for the 2D map
"""
@torch.no_grad()
def encode_blocks_at(start_samples):
    """Encodes the 8-frame mel block that starts at each sample position (mels cut from a longer STFT)."""
    starts = torch.as_tensor(start_samples, dtype=torch.long, device=device)
    first = starts - stft_context_frames * vocos_hop + pad_samples
    idx = first[:, None] + torch.arange(slice_samples, device=device)[None, :]
    mels = vocos.feature_extractor(padded_waveform[idx])                                       # (B, F, slice_frames)
    mels = mels[..., stft_context_frames:stft_context_frames + audio_mel_count_vae]            # (B, F, 8)
    mu, std = encoder(mels.unsqueeze(1))
    if use_mean_encoding:
        return mu
    std = torch.nn.functional.softplus(std) + 1e-6
    return Encoder.reparameterize(mu, std)


print("Generating latents for mapping...")
audio_excerpt_frame_offset = 10000
audio_excerpt_starts = list(range(0, audio_waveform.shape[1] - audio_window_length_vae, audio_excerpt_frame_offset))
batch_size = 32

audio_encodings = []
for eI in range(0, len(audio_excerpt_starts), batch_size):
    audio_encodings.append(encode_blocks_at(audio_excerpt_starts[eI:eI + batch_size]).detach().cpu())
audio_encodings = torch.cat(audio_encodings, dim=0).numpy()

print("Calculating t-SNE projection...")
tsne = TSNE(n_components=2, max_iter=5000, verbose=1)
Z_tsne = tsne.fit_transform(audio_encodings)

# kNN for latent averaging based on mouse clicks
n_neighbors = 4
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
knn.fit(Z_tsne)


def calc_distance_based_averaged_encoding(point2D):
    """Returns distance-weighted averaged encoding for a given 2D point."""
    _, indices = knn.kneighbors(point2D)
    nearest_positions = Z_tsne[indices[0]]
    nearest_encodings = audio_encodings[indices[0]]
    nearest_2D_distances = np.linalg.norm(nearest_positions - point2D, axis=1)
    max_2D_distance = np.max(nearest_2D_distances)
    if max_2D_distance < 1e-6:
        return nearest_encodings[0]
    norm_nearest_2D_distances = nearest_2D_distances / max_2D_distance
    weights = (1.0 - norm_nearest_2D_distances)
    if weights.sum() < 1e-9:
        return nearest_encodings[0]
    return np.average(nearest_encodings, weights=weights, axis=0)


"""
Audio Synthesis (rolling mel buffer)
"""
inter_audio_encodings = []
inter_audio_encoding_index = 0
mel_buffer = silent_mels.expand(1, audio_mel_filter_count, buffer_frames).clone()


@torch.no_grad()
def synthesize_audio():
    """Decodes the next blocks_per_step painted encodings; returns one contiguous segment of step_samples samples."""
    global mel_buffer, inter_audio_encoding_index

    encs = list(inter_audio_encodings)   # snapshot, the GUI thread modifies the list
    if len(encs) == 0:
        mel_buffer = silent_mels.expand(1, audio_mel_filter_count, buffer_frames).clone()
        return torch.zeros(step_samples, device=device)

    idx = inter_audio_encoding_index
    chosen = []
    for _ in range(blocks_per_step):
        idx = (idx + 1) % len(encs)
        chosen.append(encs[idx])
    inter_audio_encoding_index = idx
    latents = torch.cat(chosen, dim=0)                                            # (k, latent_dim)

    new_mels = decoder(latents).squeeze(1)                                        # (k, F, 8)
    new_mels = new_mels.permute(1, 0, 2).reshape(1, audio_mel_filter_count, step_frames)

    mel_buffer = torch.cat([mel_buffer[..., step_frames:], new_mels], dim=-1)     # rolling buffer
    wave = vocos.decode(mel_buffer).reshape(-1)                                   # (buffer_frames-1)*hop samples

    # frame f is centred at sample f*hop; output the newest step that is lookahead_frames from the right edge
    seg_end = (buffer_frames - lookahead_frames) * vocos_hop
    return wave[seg_end - step_samples:seg_end]


"""
Audio Streaming Controller
"""
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
    """Manages audio playback, including starting, stopping, and swapping devices."""
    def __init__(self):
        self.stream = None
        self.is_playing = False
        self.device_id = audio_output_device
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
                    channels=audio_channels,
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
Interactive Scatter Qt Window
"""
class ScatterPlotApp(QtWidgets.QMainWindow):
    """
    Interactive scatter plot GUI for latent encodings.
    - Left button press/drag: Add points and encodings.
    - Right button press/drag: Remove points and corresponding encodings near cursor.
    - Middle button: pan plot.
    - 'C' key: clear all points/encodings.
    """
    def __init__(self, points2D, inter_audio_encodings):
        super().__init__()
        self.setWindowTitle("Latent Mapping Autoencoder")
        self.resize(800, 700)

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.graph_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.graph_widget)

        self.scatter = pg.ScatterPlotItem(
            x=points2D[:, 0],
            y=points2D[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10
        )
        self.graph_widget.addItem(self.scatter)

        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.graph_widget.addItem(self.click_scatter)

        self.play_point = []
        self.play_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(0, 255, 0, 200))
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
        self.click_encodings = inter_audio_encodings

    def on_device_changed(self, idx):
        if idx >= 0 and idx < len(self.device_mapping):
            audio_controller.set_device(self.device_mapping[idx])

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
        """Add an encoding for the clicked position."""
        encoding = calc_distance_based_averaged_encoding(np.array([[mouse_point.x(), mouse_point.y()]]))
        encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(device)
        self.click_encodings.append(encoding)

    def removeInteractiveEncodingNear(self, mouse_point, radius=0.5):
        """Remove points and encodings within a radius of the mouse cursor."""
        to_remove_indices = []
        mp_x, mp_y = mouse_point.x(), mouse_point.y()
        for i, p in enumerate(self.click_points):
            dx = p['pos'][0] - mp_x
            dy = p['pos'][1] - mp_y
            if (dx*dx + dy*dy)**0.5 < radius:
                to_remove_indices.append(i)
        for idx in reversed(to_remove_indices):
            self.click_points.pop(idx)
            self.click_encodings.pop(idx)
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )

    def clearInteractiveEncodings(self):
        """Clear all interactive points and encodings."""
        self.click_points.clear()
        self.click_scatter.setData([], [])
        self.click_encodings.clear()

    def eventFilter(self, source, event):
        """Handle mouse events for the plot."""
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
                    mouse_point = vb.mapSceneToView(event.scenePos())
                    self.removeInteractiveEncodingNear(mouse_point)
                    self.last_mouse_pos = event.scenePos()
                    return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        """Clear points/encodings on 'C' key."""
        if event.key() == QtCore.Qt.Key_C:
            self.clearInteractiveEncodings()
        super().keyPressEvent(event)

    def add_point_at(self, scene_pos):
        """Add a scatter point and encoding at mouse position."""
        if self.graph_widget.sceneBoundingRect().contains(scene_pos):
            vb = self.graph_widget.plotItem.vb
            mouse_point = vb.mapSceneToView(scene_pos)
            self.addInteractiveEncoding(mouse_point)
            x, y = mouse_point.x(), mouse_point.y()
            self.click_points.append({'pos': (x, y)})
            self.click_scatter.setData(
                [p['pos'][0] for p in self.click_points],
                [p['pos'][1] for p in self.click_points]
            )

    def continuous_add_point(self):
        """Timer-driven continuous interactive addition while dragging."""
        if self.left_button_pressed and self.last_mouse_pos is not None:
            self.add_point_at(self.last_mouse_pos)

    def update_play_point(self):
        idx = inter_audio_encoding_index
        if idx >= len(self.click_points):
            return
        play_p = self.click_points[idx]
        self.play_scatter.setData([play_p['pos'][0]], [play_p['pos'][1]])


"""
App Entrypoint
"""
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = ScatterPlotApp(Z_tsne, inter_audio_encodings)
    main.show()
    print("Qt window shown; entering event loop")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
