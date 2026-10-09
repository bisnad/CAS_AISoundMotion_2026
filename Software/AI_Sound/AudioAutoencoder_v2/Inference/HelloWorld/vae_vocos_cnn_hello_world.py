"""
Audio Autoencoder - Inference - Introductory Code Example
------------------------------------------------------------------------------------
Audio Generation Pipeline:
  waveform --Vocos mel--> mel (128 x 256 per window) --Encoder--> latent (32 x 64 per window)
  --manipulate--> --Decoder--> mel --fine-tuned Vocos--> waveform
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

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
print(f'Using {device} device (torch {torch.__version__})')

# -------------------------------------------------------------------------------------------------
# Conditional macOS / MPS handling
# -------------------------------------------------------------------------------------------------

force_fft_on_cpu = None           # run FFT on cpu, None = auto-detect by probing; True/False = force
force_safe_copies = None          # make tensors contiguous, None = auto (MPS or FFT fallback); True/False = force

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

if force_safe_copies is None:
    safe_copies = (device == 'mps') or fft_on_cpu
else:
    safe_copies = bool(force_safe_copies)

if fft_on_cpu:
    print(f"FFT ops not available on {device}: mel extraction and Vocos ISTFT heads run on CPU "
          f"(everything else stays on {device}).")
else:
    print(f"FFT ops available on {device}: running everything on {device}.")

if force_safe_copies:
    print("Employ Safe Slice Copies")
else:
    print("No Safe Slice Copies Necessary")

def safe(x):
    """Independent contiguous copy of a (possibly sliced) tensor on MPS; identity otherwise."""
    return x.clone().contiguous() if safe_copies else x

# -------------------------------------------------------------------------------------------------
# Audio Settings
# -------------------------------------------------------------------------------------------------

audio_data_path = "data/audio/"
audio_data_files = ["Take1__double_Bind_HQ_audio_crop_48khz.wav",
                    "Take2_Hibr_II_HQ_audio_crop_48khz.wav"]

audio_sample_rate = 48000
audio_window_length_vocos = 65280  # number of samples per window, corresponds to 256 mel frames
mel_floor = -11.5                          # ~ln(1e-5); log-mels are clamped to this value (silence)

# -------------------------------------------------------------------------------------------------
# Save Paths
# -------------------------------------------------------------------------------------------------

save_audio_path = os.path.join("results/audio/")
os.makedirs(save_audio_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

# model (must match the training script)
latent_channels = 32                       # latent channels per latent frame (time is downsampled 4x)
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]   # (freq, time); freq 128 -> 8, time /4
bottleneck_width = 512                     # number of channels in the hidden layers of the last 1D convolution stack
time_downsample = 4                        # product of the time strides, keep consistent with conv_strides

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

epochs = 400                               # epoch of the loaded weights (also used in output file names)
weights_tag = "data/models/vae_cnn_Stocos_ld32/{}_weights_epoch_400.pt"

# -------------------------------------------------------------------------------------------------
# Fix Seeds
# -------------------------------------------------------------------------------------------------

seed = 42

def set_all_seeds(seed):
    # Python's built-in RNG
    random.seed(seed); 
    # NumPy RNG
    np.random.seed(seed); 
    # PyTorch RNG (CPU)
    torch.manual_seed(seed)
    # PyTorch RNG (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

set_all_seeds(seed)
rng = np.random.default_rng(seed) # get a standard normal distribution

# -------------------------------------------------------------------------------------------------
# Vocoder
# -------------------------------------------------------------------------------------------------

# 1) pretrained Vocos: only used for the "plain Vocos" reference audio
vocos_pretrained = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
# 2) Vocos whose backbone/head were fine-tuned together with the autoencoder
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vw = torch.load(weights_tag.format("vocos"), map_location=device)
vocos.backbone.load_state_dict(vw["backbone"]); vocos.head.load_state_dict(vw["head"])

for m in (vocos_pretrained, vocos):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

# Only if the device lacks FFT kernels: keep the FFT-dependent submodules on the CPU.
if fft_on_cpu:
    for m in (vocos_pretrained, vocos):
        m.feature_extractor.to('cpu')
        m.head.to('cpu')

def get_mels(wave, vocoder=vocos):
    # wave: (B, N) -> log-mels (B, n_mels, T) on the compute device
    if fft_on_cpu:
        m = vocoder.feature_extractor(wave.cpu()).to(device)
    else:
        m = vocoder.feature_extractor(wave.to(device))
    if m.ndim == 4:
        m = m[:, 0]
    return m.contiguous() if safe_copies else m

def vocos_decode(mel, vocoder=vocos):
    # mel (B,F,T) on the compute device -> waveform (B,N) on the compute device
    if fft_on_cpu:
        h = vocoder.backbone(mel)                      # on device
        return vocoder.head(h.cpu()).to(device)        # linear + ISTFT on CPU
    return vocoder.decode(mel)

with torch.no_grad():
    _m = get_mels(torch.rand(1, audio_window_length_vocos, device=device))
    audio_mel_filter_count, audio_mel_count_vocos = _m.shape[1], _m.shape[2]
    assert vocos_decode(_m).shape[-1] == audio_window_length_vocos
audio_hop = audio_window_length_vocos // (audio_mel_count_vocos - 1)
assert audio_mel_count_vocos % time_downsample == 0, "mel frame count must be divisible by time_downsample"
print("mel frames", audio_mel_count_vocos, "mel bins", audio_mel_filter_count, "hop", audio_hop)

# windowing constants (derived from the probed mel frame count)
latent_step_samples = audio_hop * time_downsample                       # 1024 samples per latent step
latent_steps_per_window = audio_mel_count_vocos // time_downsample      # 64
window_hop_steps = latent_steps_per_window // 2                         # 32 steps: windows overlap by 50 %
window_hop = window_hop_steps * latent_step_samples                     # 32768 samples
edge_steps = window_hop_steps // 2                                      # 16 steps at each window edge are discarded

# -------------------------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------------------------

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
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1) # entry layer
        layers, c_in = [], channels[0]
        for c_out, s in zip(channels, strides):
            layers += [ResBlock2d(c_in), gn(c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, stride=s, padding=1)]
            c_in = c_out
        self.down = nn.Sequential(*layers) # downsampling trunk
        f_red = int(np.prod([s[0] for s in strides]))
        self.flat_ch = channels[-1] * (mel_filter_count // f_red)
        self.head = nn.Sequential(nn.Conv1d(self.flat_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  nn.Conv1d(width, 2 * latent_channels, 3, padding=1)) # projection head
    def forward(self, x):                      # x: (B,1,F,T)
        h = self.down(self.stem(x))            # (B,C,F',T')
        h = h.flatten(1, 2)                    # (B,C*F',T')
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
    def forward(self, z):                      # z: (B,Cz,T')
        h = self.inp(z)
        h = h.view(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))            # (B,1,F,T)

encoder = Encoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
decoder = Decoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
encoder.load_state_dict(torch.load(weights_tag.format("encoder"), map_location=device))
decoder.load_state_dict(torch.load(weights_tag.format("decoder"), map_location=device))
encoder.eval()
decoder.eval()

with torch.no_grad():
    _x = _m.clamp(min=mel_floor).unsqueeze(1)
    _mu, _std = encoder(_x)
    _y = decoder(_mu)
    assert _mu.shape[1:] == (latent_channels, latent_steps_per_window), f"unexpected latent shape {_mu.shape}"
    assert _y.shape == _x.shape, "decoder output shape differs from mel shape"
print("mel", tuple(_x.shape), "latent", tuple(_mu.shape), "decoded", tuple(_y.shape))

# -------------------------------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Load Test Audio (mono, 48 kHz)
# -------------------------------------------------------------------------------------------------

audio_test_starts_sec = [[10.0], [10.0]]
audio_test_duration_sec = 10.0

def load_mono(path):
    w, sr = torchaudio.load(path)
    if sr != audio_sample_rate:
        w = torchaudio.functional.resample(w, sr, audio_sample_rate)
    return w[0].contiguous()         

audio_test_waveforms = {i: load_mono(audio_data_path + audio_data_files[i])
                        for i in range(len(audio_data_files)) if len(audio_test_starts_sec[i]) > 0}
audio_test_file_indices = list(audio_test_waveforms.keys())

# -------------------------------------------------------------------------------------------------
# Windowing helpers
# -------------------------------------------------------------------------------------------------

def pad_for_windows(waveform, window, hop):
    """Zero-pad at the end so that an integer number of windows (at the given hop) covers the signal."""
    n = waveform.shape[0]
    total = window if n <= window else window + int(np.ceil((n - window) / hop)) * hop
    return F.pad(waveform, (0, total - n)), (total - window) // hop + 1

def overlap_add(chunks, window, hop, num_samples):
    """Hann-windowed overlap-add, normalised by the summed windows so the gain is exactly one."""
    total = (len(chunks) - 1) * hop + window
    out = torch.zeros(total)
    norm = torch.zeros(total)
    env = torch.hann_window(window, periodic=True)
    for i, chunk in enumerate(chunks):
        out[i * hop:i * hop + window] += chunk * env
        norm[i * hop:i * hop + window] += env
    return (out / norm.clamp(min=1e-3))[:num_samples]

def save_audio(waveform, file_name):
    torchaudio.save(file_name, waveform.reshape(1, -1).cpu(), audio_sample_rate)

def create_ref_audio(waveform, file_name):
    save_audio(waveform, file_name)

@torch.inference_mode()
def create_voc_audio(waveform, file_name):
    """Plain (pretrained) Vocos resynthesis: mel -> waveform, no autoencoder."""
    off = audio_window_length_vocos // 2
    padded, window_count = pad_for_windows(waveform, audio_window_length_vocos, off)
    chunks = []
    for i in range(window_count):
        seg = safe(padded[i * off:i * off + audio_window_length_vocos].unsqueeze(0)).to(device)
        mels = get_mels(seg, vocos_pretrained)
        chunks.append(vocos_decode(mels, vocos_pretrained)[0].cpu())
    save_audio(overlap_add(chunks, audio_window_length_vocos, off, waveform.shape[0]), file_name)

@torch.inference_mode()
def encode_audio(waveform, sample=False):
    """
    Waveform (N,) -> latent timeline z of shape (latent_channels, Tz) as a numpy array.
    Windows overlap by 50 %; of each window only the centre half of the latent steps is kept
    (first window: also its start, last window: also its end), so z has no window-edge artefacts.
    sample=False: use the posterior mean (deterministic).
    """
    padded, window_count = pad_for_windows(waveform, audio_window_length_vocos, window_hop)
    pieces = []
    for w in range(window_count):
        seg = safe(padded[w * window_hop:w * window_hop + audio_window_length_vocos].unsqueeze(0)).to(device)
        mels = get_mels(seg).clamp(min=mel_floor)                                # (1,F,256)
        mu, std = encoder(mels.unsqueeze(1))
        z = Encoder.reparameterize(mu, std) if sample else mu                    # (1,Cz,64)
        lo = 0 if w == 0 else edge_steps
        hi = latent_steps_per_window if w == window_count - 1 else latent_steps_per_window - edge_steps
        pieces.append(z[0, :, lo:hi].cpu().numpy())
    return np.concatenate(pieces, axis=1)                                        # (Cz, Tz)

@torch.inference_mode()
def decode_audio_encodings(z, num_samples, file_name):
    """Latent timeline z (latent_channels, Tz) -> waveform file. Windows of 64 steps at a hop of 32 steps."""
    Tz = z.shape[1]
    if Tz < latent_steps_per_window:
        extra = latent_steps_per_window - Tz
    else:
        extra = (-(Tz - latent_steps_per_window)) % window_hop_steps
    if extra > 0:
        z = np.pad(z, ((0, 0), (0, extra)), mode="edge")                         # repeat last step
    window_count = (z.shape[1] - latent_steps_per_window) // window_hop_steps + 1

    chunks = []
    for w in range(window_count):
        st = w * window_hop_steps
        zw = torch.from_numpy(np.ascontiguousarray(z[:, st:st + latent_steps_per_window])).float().unsqueeze(0).to(device)
        mels = decoder(zw)[:, 0]                                                 # (1,F,256)
        chunks.append(vocos_decode(mels)[0].cpu())

    save_audio(overlap_add(chunks, audio_window_length_vocos, window_hop, num_samples), file_name)

# -------------------------------------------------------------------------------------------------
# Original, Vocos and Reconstructed Waveform
# -------------------------------------------------------------------------------------------------

for fi in audio_test_file_indices:
    wf = audio_test_waveforms[fi]
    tag = os.path.splitext(audio_data_files[fi])[0]
    for s in audio_test_starts_sec[fi]:
        a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
        seg, name = wf[a:b], f"{tag}_{s}-{s + audio_test_duration_sec}"

        create_ref_audio(seg, f"{save_audio_path}audio_ref_{name}.wav")
        create_voc_audio(seg, f"{save_audio_path}audio_voc_{name}.wav")

        z = encode_audio(seg)                                                    # deterministic (mu)
        print("latent timeline:", z.shape, "= (channels, steps of 21.3 ms)")
        decode_audio_encodings(z, seg.shape[0], f"{save_audio_path}rec_audio_epochs_{epochs}_{name}.wav")

# -------------------------------------------------------------------------------------------------
# Random Walk
# -------------------------------------------------------------------------------------------------

random_walk_step_size = 0.10            # latent distance per 21.3 ms step; tune to your latent scale

for fi in audio_test_file_indices:
    wf = audio_test_waveforms[fi]
    tag = os.path.splitext(audio_data_files[fi])[0]
    for s in audio_test_starts_sec[fi]:
        a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
        seg, name = wf[a:b], f"{tag}_{s}-{s + audio_test_duration_sec}"

        z = encode_audio(seg)
        steps = rng.standard_normal((latent_channels, z.shape[1] - 1)).astype(np.float32)
        steps = steps / np.linalg.norm(steps, axis=0, keepdims=True) * random_walk_step_size   # fixed-length steps
        z_walk = z.copy()
        z_walk[:, 1:] = z[:, :1] + np.cumsum(steps, axis=1)       # start at the first encoded latent

        decode_audio_encodings(z_walk, seg.shape[0], f"{save_audio_path}randwalk_audio_{name}.wav")

# -------------------------------------------------------------------------------------------------
# Offset Following
# -------------------------------------------------------------------------------------------------

offset_sizes = [0.0, -4.0]               # offset ramps linearly through these values over the excerpt

for fi in audio_test_file_indices:
    wf = audio_test_waveforms[fi]
    tag = os.path.splitext(audio_data_files[fi])[0]
    for s in audio_test_starts_sec[fi]:
        a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
        seg, name = wf[a:b], f"{tag}_{s}-{s + audio_test_duration_sec}"

        z = encode_audio(seg)
        positions = np.linspace(0.0, 1.0, z.shape[1])
        offset_curve = np.interp(positions, np.linspace(0.0, 1.0, len(offset_sizes)), offset_sizes).astype(np.float32)
        z_offset = z + offset_curve[None, :]                      # same offset added to every latent channel

        decode_audio_encodings(z_offset, seg.shape[0], f"{save_audio_path}offset_audio_{name}.wav")

# -------------------------------------------------------------------------------------------------
# Latent Interpolation (between the test excerpts of file 0 and file 1)
# -------------------------------------------------------------------------------------------------

mix_file_indices = [0, 1]
mix_factors = [0.0, 1.0, 4.0, -4.0]     # 0: audio 1, 1: audio 2, outside [0,1]: extrapolation

tags = [os.path.splitext(audio_data_files[fi])[0] for fi in mix_file_indices]
for s1, s2 in zip(audio_test_starts_sec[mix_file_indices[0]], audio_test_starts_sec[mix_file_indices[1]]):
    segs, zs = [], []
    for fi, s in zip(mix_file_indices, (s1, s2)):
        a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
        segs.append(audio_test_waveforms[fi][a:b])
        zs.append(encode_audio(segs[-1]))

    n = min(zs[0].shape[1], zs[1].shape[1])
    positions = np.linspace(0.0, 1.0, n)
    mix = np.interp(positions, np.linspace(0.0, 1.0, len(mix_factors)), mix_factors).astype(np.float32)
    zm = zs[0][:, :n] * (1.0 - mix[None, :]) + zs[1][:, :n] * mix[None, :]

    decode_audio_encodings(zm, min(segs[0].shape[0], segs[1].shape[0]),
                           f"{save_audio_path}mix_audio_{tags[0]}_{s1}_{tags[1]}_{s2}.wav")