"""
Audio Autoencoder for Vocos 48khz
The encoder and decoder architectures employ convolutional neural networks
------------------------------------------
- Loads audio, VAE models, and vocoder.
- Uses t-SNE to map latent encodings for visualization.
- PyQt/pyqtgraph GUI allows users to interactively select and generate new encodings via mouse.
- Real-time audio synthesis using the currently selected encoding(s).
- Audio streaming and GUI are run concurrently in separate threads.
"""

import os
import math
import numpy as np
import threading
import queue
import torch
from torch import nn
import torchaudio
from torchaudio.functional import highpass_biquad
import sounddevice as sd
from vocos import Vocos
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys

"""
Settings
"""

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

# audio settings
audio_file = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"

audio_sample_rate = 48000 # numer of audio samples per sec
audio_channels = 1

audio_window_length_vocos = 29696 # 117 mel frames worth of audio
audio_window_length_vae = 1792 # 8 mel frames worth of audio
audio_mel_count_vocos = None # will be calculated
audio_mel_count_vae = None
audio_mel_count_vae_per_vocos = None
audio_window_offset = 960

audio_output_device = 7

"""
VAE Model Settings
"""

# model settings
latent_dim = 32
ae_conv_channel_counts = [ 16, 32, 64, 128 ]
ae_conv_kernel_size = (5, 3)
ae_dense_layer_sizes = [ 512 ]
ae_encoder_weights_file = "data/models/vae_cnn_Gutenberg_ld32/encoder_weights_epoch_400"
ae_decoder_weights_file = "data/models/vae_cnn_Gutenberg_ld32/decoder_weights_epoch_400"


# automated settings
window_size = audio_window_length_vae
window_offset = window_size // 2
play_buffer_size = audio_window_length_vae * 16
playback_latency = 1.0
max_audio_queue_length = 32

"""
Load Audio
"""

# ==== Audio Loading ====
assert os.path.exists(audio_file), f"Audio file not found: {audio_file}"

audio_waveform, _ = torchaudio.load(audio_file)
audio_source_samples = audio_waveform[0].to(device)
audio_source_frame_index = 0
hann_window = torch.from_numpy(np.hanning(window_size)).float().to(device) # Move to device once

"""
Vocoder Model
"""

print("Loading Vocos Model...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)

# freeze model parameters
for param in vocos.parameters():
    param.requires_grad = False

vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vocos), dtype=torch.float32).to(device))
audio_mel_count_vocos = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vae), dtype=torch.float32).to(device))
audio_mel_count_vae = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

audio_mel_count_vae_per_vocos = audio_mel_count_vocos // audio_mel_count_vae

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

# ==== Instantiate and load weights ====
encoder = Encoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)
decoder = Decoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, list(reversed(ae_conv_channel_counts)), ae_conv_kernel_size, list(reversed(ae_dense_layer_sizes))).to(device)

if os.path.exists(ae_encoder_weights_file):
    encoder.load_state_dict(torch.load(ae_encoder_weights_file, map_location=device))
if os.path.exists(ae_decoder_weights_file):
    decoder.load_state_dict(torch.load(ae_decoder_weights_file, map_location=device))

encoder.eval()
decoder.eval()

# ==== Audio excerpt generation for 2D mapping ====
print("Generating latents for mapping...")
audio_excerpt_start_frame = 0
audio_excerpt_end_frame = audio_waveform.shape[1]
audio_excerpt_frame_offset = 10000
audio_excerpts = []
for fI in range(audio_excerpt_start_frame, audio_excerpt_end_frame - audio_window_length_vae, audio_excerpt_frame_offset):
    audio_excerpt = audio_waveform[0, fI:fI + audio_window_length_vae]
    audio_excerpts.append(audio_excerpt)
audio_excerpts = torch.stack(audio_excerpts, dim=0)
batch_size = 32

# ==== Generate latent encodings and 2D projections ====
with torch.no_grad():
    audio_encodings = []
    for eI in range(0, audio_excerpts.shape[0] - batch_size, batch_size):
        batch = audio_excerpts[eI:eI+batch_size].to(device)
        mels = vocos.feature_extractor(batch)
        mu, std = encoder(mels.unsqueeze(1))
        std = torch.nn.functional.softplus(std) + 1e-6
        encoded_batch = Encoder.reparameterize(mu, std).detach().cpu()
        audio_encodings.append(encoded_batch)
    audio_encodings = torch.cat(audio_encodings, dim=0).numpy()

print("Calculating t-SNE projection...")
tsne = TSNE(n_components=2, max_iter=5000, verbose=1)
Z_tsne = tsne.fit_transform(audio_encodings)

# ==== kNN for latent averaging based on mouse clicks ====
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
    # prevent division by zero
    if max_2D_distance < 1e-6:
        return nearest_encodings[0]
    norm_nearest_2D_distances = nearest_2D_distances / max_2D_distance
    weights = (1.0 - norm_nearest_2D_distances)
    return np.average(nearest_encodings, weights=weights, axis=0)

# ========== AUDIO CALLBACKS ==========
inter_audio_encodings = []
inter_audio_encoding_index = 0
inter_vocos_mels = torch.zeros((1, 128, 117)).to(device)

@torch.no_grad()
def encode_audio(waveform):
    assert waveform.shape[1] == audio_window_length_vocos, "mismatch between waveform length and audio_window_length_vocos"
    y_wave_vocos = waveform.to(device)
    y_mels_vocos = vocos.feature_extractor(y_wave_vocos)
    encodings = []
    
    for i in range(audio_mel_count_vae_per_vocos):
        y_mels_vae = y_mels_vocos[:, :, i * audio_mel_count_vae:i * audio_mel_count_vae + audio_mel_count_vae]
        mu, std = encoder(y_mels_vae.unsqueeze(1))
        std = torch.nn.functional.softplus(std) + 1e-6
        encoding = Encoder.reparameterize(mu, std)
        encodings.append(encoding)
        
    encodings = torch.cat(encodings, dim=0)
    return encodings

@torch.no_grad()
def decode_audio(latent):
    global inter_vocos_mels
    yhat_mels_vae = decoder(latent).squeeze(1)
    inter_vocos_mels = torch.cat([inter_vocos_mels[:, :, yhat_mels_vae.shape[-1]:], yhat_mels_vae.detach() ], dim=2 )
    yhat_waveform_vocos  = vocos.decode(inter_vocos_mels)
    yhat_waveform_vae = yhat_waveform_vocos[:, -window_size: ]
    return yhat_waveform_vae

@torch.no_grad()
def synthesize_audio():
    """Decode the next latent encoding from the current interactive list."""
    global inter_audio_encoding_index
    
    if len(inter_audio_encodings) == 0:
        return torch.zeros(window_size)

    inter_audio_encoding_index += 1
    if inter_audio_encoding_index >= len(inter_audio_encodings):
        inter_audio_encoding_index = 0
        
    latent = inter_audio_encodings[inter_audio_encoding_index]
    gen_waveform = decode_audio(latent)

    return gen_waveform[0]

# ========== AUDIO STREAMING CONTROLLER ==========
audio_queue = queue.Queue(maxsize=max_audio_queue_length)
last_chunk = np.zeros(window_size, dtype=np.float32)
producer_running = True

def producer_thread():
    """Continuously generates audio and fills the output buffer queue."""
    while producer_running:
        if not audio_queue.full():
            gen_waveform = synthesize_audio()
            audio_queue.put(gen_waveform.cpu().numpy())
        else:
            sd.sleep(5)

def audio_callback(out_data, frames, time_info, status):
    """sounddevice stream callback function."""
    global last_chunk
    output = np.zeros((frames, audio_channels), dtype=np.float32)
    
    cursor = 0
    overlap_len = window_size // 2
    copy_len = min(overlap_len, frames)
    output[cursor:cursor+copy_len, 0] += last_chunk[overlap_len:overlap_len+copy_len]
    samples_needed = frames
    
    while samples_needed > 0:
        try:
            chunk = audio_queue.get_nowait()
            chunk = (chunk * hann_window.cpu().numpy())
        except queue.Empty:
            chunk = np.zeros(window_size, dtype=np.float32)
            
        chunk_copy_size = min(window_size, samples_needed)
        output[cursor:cursor+chunk_copy_size, 0] += chunk[:chunk_copy_size]
        cursor += window_size // 2
        samples_needed = frames - cursor
        last_chunk[:] = chunk[:]
    out_data[:] = output

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

# ========== INTERACTIVE SCATTER QT WINDOW ==========
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
        
        # Container and main vertical layout
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)

        # --- Scatter Plot Graph (Top element) ---
        self.graph_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.graph_widget)

        # Main scatter item (fixed background points)
        self.scatter = pg.ScatterPlotItem(
            x=points2D[:, 0],
            y=points2D[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10
        )
        self.graph_widget.addItem(self.scatter)

        # Interactive points and encodings (red points are user-selected)
        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.graph_widget.addItem(self.click_scatter)
        
        # Audio playback point
        self.play_point = []
        self.play_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(0, 255, 0, 200))
        self.graph_widget.addItem(self.play_scatter)

        # --- Playback and Settings Controls (Bottom elements) ---
        self.controls_layout = QtWidgets.QHBoxLayout()

        # Device selection layout
        self.device_layout = QtWidgets.QHBoxLayout()
        self.device_label = QtWidgets.QLabel("Output Device:")
        self.device_combo = QtWidgets.QComboBox()
        
        self.device_mapping = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] > 0:
                self.device_mapping.append(i)
                self.device_combo.addItem(f"{i}: {dev['name']}")
                if i == audio_output_device:
                    # Set the combo box to the currently selected default
                    self.device_combo.setCurrentIndex(len(self.device_mapping) - 1)
                    
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        
        self.device_layout.addWidget(self.device_label)
        self.device_layout.addWidget(self.device_combo)
        self.controls_layout.addLayout(self.device_layout)
        
        # Control Buttons
        self.start_btn = QtWidgets.QPushButton("Start Playback")
        self.start_btn.clicked.connect(self.start_playback)
        self.stop_btn = QtWidgets.QPushButton("Stop Playback")
        self.stop_btn.clicked.connect(self.stop_playback)
        
        # Clear button inserted between Stop and Exit
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clearInteractiveEncodings)
        
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.exit_app)
        
        self.controls_layout.addWidget(self.start_btn)
        self.controls_layout.addWidget(self.stop_btn)
        self.controls_layout.addWidget(self.clear_btn)
        self.controls_layout.addWidget(self.exit_btn)
        
        self.main_layout.addLayout(self.controls_layout)

        # Interaction state
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.right_button_pressed = False
        self.last_mouse_pos = None

        # Timer for continuous add (while dragging left button)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)  # ms
        self.timer.timeout.connect(self.continuous_add_point)
        
        # Timer for updating play point
        self.playpoint_timer = QtCore.QTimer()
        self.playpoint_timer.setInterval(25)  # ms
        self.playpoint_timer.timeout.connect(self.update_play_point)
        self.playpoint_timer.start()

        # Register event filter for mouse events
        self.graph_widget.scene().installEventFilter(self)
        self.click_encodings = inter_audio_encodings

    # --- Callbacks for the newly added GUI elements ---
    def on_device_changed(self, idx):
        if idx >= 0 and idx < len(self.device_mapping):
            real_device_id = self.device_mapping[idx]
            audio_controller.set_device(real_device_id)

    def start_playback(self):
        audio_controller.start()

    def stop_playback(self):
        audio_controller.stop()

    def exit_app(self):
        global producer_running
        self.stop_playback()
        producer_running = False
        self.close()

    # --- Drawing and Graphing methods ---
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
            dist = (dx*dx + dy*dy)**0.5
            if dist < radius:
                to_remove_indices.append(i)
        # Remove from lists in reverse order for safety
        for idx in reversed(to_remove_indices):
            self.click_points.pop(idx)
            self.click_encodings.pop(idx)
        # Update plot
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
                    # Remove points near mouse position on drag
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
        global inter_audio_encoding_index
        
        if inter_audio_encoding_index >= len(self.click_points):
            return
        
        play_p = self.click_points[inter_audio_encoding_index]

        self.play_scatter.setData( [ play_p['pos'][0] ], [ play_p['pos'][1] ] )

# ========== APP ENTRYPOINT ==========
def main():
    """Create and run the Qt application."""
    app = QtWidgets.QApplication(sys.argv)
    main = ScatterPlotApp(Z_tsne, inter_audio_encodings)
    main.show()
    print("Qt window shown; entering event loop")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()