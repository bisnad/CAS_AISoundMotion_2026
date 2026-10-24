"""
Live Audio VAE: train and listen at the same time
(v3: sophisticated ResNet VAE with causal decoder, configurable loss registry, frozen Vocos)
-----------------------------------------------------------------------------------------
- Trains the CNN VAE on Vocos 48 kHz log-mels in a background thread (step-based).
- The encoder processes groups of audio_mel_count_vae mel frames independently; the causal decoder
  decodes the whole latent timeline; a frozen Vocos turns the mels into audio.
- A second thread runs an eval-mode copy of the model in real time, synced every N steps.
- All training data selection, loss weights and loss configurations can be changed while training.

Losses (each has its own weight; weight 0 = disabled and not computed):
  vae_mel    Vocos-mel distance (L1/L2) on the VAE input/output mels
  mr_stft    Perceptual multi-resolution mel-STFT (auraloss)
  mel_mr     Multi-resolution log-mel spectrogram loss
  lin_mr     Multi-resolution log linear-magnitude spectrogram loss
  mfcc       MFCC loss (optional deltas)
  loudness   Frame loudness loss (A-weighted STFT or torchaudio ITU BS.1770)
  flatness   Spectral flatness loss (dB, optionally per band)
  centroid   Spectral centroid loss
  lw_centroid Loudness-weighted spectral centroid loss
  temporal   Temporal centroid loss (per segment)
Total = rec_loss_scale * sum_k(weight_k * loss_k) + beta * KL
"""

import os
import sys
import time
import queue
import threading
import traceback
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import auraloss
from vocos import Vocos

from PyQt5 import QtWidgets, QtCore

# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

audio_data_path = "data/audio/"
audio_data_files = [
    "Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
]

audio_sample_rate = 48000

# User-specified: mel frames per Vocos window (training excerpt) and per encoder group
audio_mel_count_vocos = 256
audio_mel_count_vae = 8
mel_floor = -11.5                          # log-mels are clamped to this value (silence)

# User-specified: mel frames per live-inference window (multiple of audio_mel_count_vae).
# Longer windows give the causal decoder more context, but add latency.
inference_mel_count = 32 # 256

# Computed automatically (see Vocos section)
audio_window_length_vocos = None
audio_window_length_vae = None
inference_window_length = None
inference_hop = None

inference_batch_windows = 4
max_queued_blocks = 24
default_audio_output_device = None

# Model (same as the offline training script)
latent_channels = 32
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]   # (freq, time)
bottleneck_width = 512
time_downsample = 4                        # product of the time strides

grad_clip = 1.0

save_path = "results/live_vae"
os.makedirs(save_path, exist_ok=True)
autosave_interval_steps = 5000

initial_encoder_weights = None
initial_decoder_weights = None


@dataclass
class TrainSettings:
    rec_loss_scale: float = 5.0        # global multiplier on the sum of all weighted reconstruction losses
    min_beta: float = 0.0
    max_beta: float = 0.01
    beta_cyclic: bool = False
    beta_cycle_steps: int = 2000
    beta_min_const_steps: int = 400
    beta_max_const_steps: int = 400
    learning_rate: float = 5e-4
    batch_size: int = 32
    sync_interval: int = 5
    paused: bool = False


@dataclass
class InferenceSettings:
    file_index: int = 0
    start_sec: float = 0.0
    end_sec: float = 1e9
    use_mean: bool = False
    latent_noise: float = 0.0
    gain: float = 1.0


train_settings = TrainSettings()
inf_settings = InferenceSettings()

# -------------------------------------------------------------------------------------------------
# Loss specifications (drives both the computation and the GUI)
# -------------------------------------------------------------------------------------------------

FFT_CHOICES = ["256", "512", "1024", "2048", "4096"]

LOSS_SPECS = [
    dict(key="vae_mel", title="VAE log-mel distance (Vocos mels)", weight=0.5, params=[
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="mr_stft", title="Perceptual multi-res mel-STFT (auraloss)", weight=0.5, params=[
        dict(key="n_bins", label="Mel bins", type="int", default=128, min=16, max=512),
        dict(key="perceptual", label="Perceptual weighting", type="bool", default=True),
    ]),
    dict(key="mel_mr", title="Multi-resolution log-mel spectrogram", weight=0.0, params=[
        dict(key="n_res", label="Resolutions", type="int", default=3, min=1, max=5),
        dict(key="min_fft", label="Smallest FFT size", type="choice", default="512", choices=FFT_CHOICES),
        dict(key="n_mels", label="Mel bands", type="int", default=80, min=8, max=256),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="lin_mr", title="Multi-resolution log linear spectrogram", weight=0.5, params=[
        dict(key="n_res", label="Resolutions", type="int", default=3, min=1, max=5),
        dict(key="min_fft", label="Smallest FFT size", type="choice", default="512", choices=FFT_CHOICES),
        dict(key="n_bands", label="Pooled bands (0 = all bins)", type="int", default=0, min=0, max=2048),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="mfcc", title="MFCC", weight=0.0, params=[
        dict(key="n_mfcc", label="MFCC coefficients", type="int", default=20, min=2, max=128),
        dict(key="n_mels", label="Mel bands", type="int", default=64, min=8, max=256),
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="skip_c0", label="Skip c0 (overall level)", type="bool", default=True),
        dict(key="deltas", label="Also match deltas", type="bool", default=False),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="loudness", title="Loudness", weight=0.0, params=[
        dict(key="mode", label="Mode", type="choice", default="A-weighted STFT (fast)",
             choices=["A-weighted STFT (fast)", "torchaudio BS.1770 (experimental)"]),
        dict(key="n_fft", label="FFT size (A-weighted mode)", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="db_div", label="dB normalization", type="float", default=20.0, min=1.0, max=200.0, step=1.0),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="flatness", title="Spectral flatness", weight=0.5, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="n_bands", label="Log-spaced bands", type="int", default=8, min=1, max=64),
        dict(key="db_div", label="dB normalization", type="float", default=10.0, min=1.0, max=200.0, step=1.0),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="centroid", title="Spectral centroid", weight=0.0, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="log2", label="Compare in log2(Hz)", type="bool", default=True),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="lw_centroid", title="Loudness-weighted spectral centroid", weight=0.0, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="exponent", label="Weight exponent (0 = unweighted)", type="float", default=1.0, min=0.0, max=4.0, step=0.1),
        dict(key="log2", label="Compare in log2(Hz)", type="bool", default=True),
    ]),
    dict(key="temporal", title="Temporal centroid", weight=0.0, params=[
        dict(key="frame", label="Envelope frame (samples)", type="choice", default="512", choices=["128", "256", "512", "1024"]),
        dict(key="segments", label="Segments per window", type="int", default=16, min=1, max=128),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
]

loss_cfg = {}
for _s in LOSS_SPECS:
    loss_cfg[_s["key"]] = {"weight": _s["weight"], **{p["key"]: p["default"] for p in _s["params"]}}

WAVEFORM_LOSS_KEYS = [s["key"] for s in LOSS_SPECS if s["key"] != "vae_mel"]

# -------------------------------------------------------------------------------------------------
# Vocos (frozen)
# -------------------------------------------------------------------------------------------------

print("Loading Vocos...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
for p in vocos.parameters():
    p.requires_grad = False


def get_mels(wave):
    # wave (B, N) -> clamped log-mels (B, n_mels, T)
    m = vocos.feature_extractor(wave)
    if m.ndim == 4:
        m = m[:, 0]
    return m.clamp(min=mel_floor)


def find_vocos_hop():
    hop = getattr(getattr(vocos.feature_extractor, "mel_spec", None), "hop_length", None)
    if hop is None:
        n = 480000
        with torch.no_grad():
            t = vocos.feature_extractor(torch.rand(1, n, device=device)).shape[-1]
        hop = int(round(n / (t - 1)))
    return int(hop)


audio_hop = find_vocos_hop()

assert audio_mel_count_vocos % audio_mel_count_vae == 0, "audio_mel_count_vocos must be a multiple of audio_mel_count_vae"
assert audio_mel_count_vae % time_downsample == 0, "audio_mel_count_vae must be divisible by time_downsample"
assert inference_mel_count % audio_mel_count_vae == 0, "inference_mel_count must be a multiple of audio_mel_count_vae"

# centered STFT frames: n_frames = n_samples // hop + 1
audio_window_length_vocos = (audio_mel_count_vocos - 1) * audio_hop
audio_window_length_vae = (audio_mel_count_vae - 1) * audio_hop
inference_window_length = (inference_mel_count - 1) * audio_hop
inference_hop = inference_window_length // 2

with torch.no_grad():
    for n, cnt in ((audio_window_length_vocos, audio_mel_count_vocos),
                   (audio_window_length_vae, audio_mel_count_vae),
                   (inference_window_length, inference_mel_count)):
        m = vocos.feature_extractor(torch.rand(1, n, device=device))
        assert m.shape[-1] == cnt, f"{n} samples give {m.shape[-1]} mel frames, expected {cnt}"
        assert vocos.decode(m).shape[-1] == n, "Vocos decode length mismatch"
    mel_filter_count = m.shape[1]

audio_vae_mels_per_vocos_mels = audio_mel_count_vocos // audio_mel_count_vae
latent_steps_per_group = audio_mel_count_vae // time_downsample
print(f"hop {audio_hop} | mel bins {mel_filter_count} | vocos window {audio_window_length_vocos} samples | "
      f"vae window {audio_window_length_vae} | inference window {inference_window_length} "
      f"({inference_window_length / audio_sample_rate * 1000:.0f} ms)")


# -------------------------------------------------------------------------------------------------
# Grouping helpers (the encoder sees groups of audio_mel_count_vae mel frames, independently)
# -------------------------------------------------------------------------------------------------

def group_mels(mels):
    # (B, n_mels, T) -> (B*G, 1, n_mels, audio_mel_count_vae)
    B, n_mels, T = mels.shape
    G = T // audio_mel_count_vae
    return mels.reshape(B, n_mels, G, audio_mel_count_vae).permute(0, 2, 1, 3).reshape(B * G, 1, n_mels, audio_mel_count_vae)


def ungroup_latents(z, B):
    # (B*G, Cz, latent_steps_per_group) -> (B, Cz, Tz)
    BG, Cz, Tzg = z.shape
    G = BG // B
    return z.reshape(B, G, Cz, Tzg).permute(0, 2, 1, 3).reshape(B, Cz, G * Tzg)


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
    def forward(self, x):                      # x: (B,1,F,T) -> mu, std (B,Cz,T')
        h = self.down(self.stem(x)).flatten(1, 2)
        mu, raw = self.head(h).chunk(2, dim=1)
        return mu, F.softplus(raw) + 1e-4      # std is already positive
    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)


class FrameGroupNorm(nn.Module):
    """GroupNorm computed independently for every time step: required for causal streaming."""
    def __init__(self, groups, ch):
        super().__init__()
        self.gn = nn.GroupNorm(groups, ch)
    def forward(self, x):
        if x.ndim == 3:                                        # (B,C,T)
            B, C, T = x.shape
            y = self.gn(x.permute(0, 2, 1).reshape(B * T, C, 1))
            return y.reshape(B, T, C).permute(0, 2, 1)
        B, C, Fq, T = x.shape                                  # (B,C,F,T)
        y = self.gn(x.permute(0, 3, 1, 2).reshape(B * T, C, Fq))
        return y.reshape(B, T, C, Fq).permute(0, 2, 3, 1)


def fgn(ch):
    return FrameGroupNorm(8, ch)


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c_in, c_out, k)
        self.streaming = False
        self.cache = None
    def forward(self, x):
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], self.k - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k - 1, 0))
        return self.conv(x)


class CausalConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_f=3, k_t=3):
        super().__init__()
        self.k_f, self.k_t = k_f, k_t
        self.conv = nn.Conv2d(c_in, c_out, (k_f, k_t))
        self.streaming = False
        self.cache = None
    def forward(self, x):
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.k_t - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k_t - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k_t - 1, 0))
        x = F.pad(x, (0, 0, self.k_f // 2, self.k_f // 2))
        return self.conv(x)


class CausalResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(fgn(ch), nn.SiLU(), CausalConv2d(ch, ch),
                                 fgn(ch), nn.SiLU(), CausalConv2d(ch, ch))
    def forward(self, x):
        return x + self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels, mel_filter_count, channels, strides, width):
        super().__init__()
        rc, rs = channels[::-1], strides[::-1]
        f_red = int(np.prod([s[0] for s in strides]))
        self.c0, self.f0 = rc[0], mel_filter_count // f_red
        self.inp = nn.Sequential(CausalConv1d(latent_channels, width), fgn(width), nn.SiLU(),
                                 CausalConv1d(width, width), fgn(width), nn.SiLU(),
                                 CausalConv1d(width, self.c0 * self.f0))
        layers = []
        for i, s in enumerate(rs):
            c_in = rc[i]
            c_out = rc[i + 1] if i + 1 < len(rc) else rc[-1]
            layers += [CausalResBlock2d(c_in), fgn(c_in), nn.SiLU(), nn.Upsample(scale_factor=s, mode="nearest"),
                       CausalConv2d(c_in, c_out)]
        self.up = nn.Sequential(*layers)
        self.out = nn.Sequential(fgn(rc[-1]), nn.SiLU(), CausalConv2d(rc[-1], 1))
    def forward(self, z):                      # z: (B,Cz,T') -> (B,1,F,T)
        h = self.inp(z)
        h = h.reshape(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))


def build_models():
    enc = Encoder(latent_channels, mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
    dec = Decoder(latent_channels, mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
    return enc, dec


def kl_loss(mu, std):
    return -0.5 * torch.mean(1 + 2 * torch.log(std) - mu.pow(2) - std.pow(2))

# -------------------------------------------------------------------------------------------------
# Loss implementations
# -------------------------------------------------------------------------------------------------

_cache = {}


def cached(key, factory):
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def hann(n):
    return cached(("hann", n), lambda: torch.hann_window(n).to(device))


def mel_fb(n_fft, n_mels):
    return cached(("fb", n_fft, n_mels), lambda: torchaudio.functional.melscale_fbanks(
        n_fft // 2 + 1, 0.0, audio_sample_rate / 2, n_mels, audio_sample_rate,
        norm="slaney", mel_scale="slaney").to(device))


def freq_axis(n_fft):
    return cached(("freqs", n_fft), lambda: torch.fft.rfftfreq(n_fft, 1.0 / audio_sample_rate).to(device))


def a_weight_gain(n_fft):
    def make():
        f2 = freq_axis(n_fft) ** 2
        ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
        return ra * 10 ** (2.0 / 20)
    return cached(("again", n_fft), make)


def band_edges(n_bins, n_bands):
    def make():
        e = np.unique(np.round(np.geomspace(1, n_bins, n_bands + 1)).astype(int))
        return e if len(e) > 1 else np.array([1, n_bins])
    return cached(("edges", n_bins, n_bands), make)


def get_mfcc(n_mfcc, n_mels, n_fft):
    n_mfcc = min(n_mfcc, n_mels)
    return cached(("mfcc", n_mfcc, n_mels, n_fft), lambda: torchaudio.transforms.MFCC(
        sample_rate=audio_sample_rate, n_mfcc=n_mfcc, log_mels=True,
        melkwargs=dict(n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels)).to(device))


def get_mr_stft(n_bins, perceptual):
    return cached(("mrstft", n_bins, perceptual), lambda: auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192], hop_sizes=[256, 512, 2048], win_lengths=[1024, 2048, 8192],
        scale="mel", n_bins=n_bins, sample_rate=audio_sample_rate, perceptual_weighting=perceptual).to(device))


def dist(a, b, kind):
    return (a - b).abs().mean() if kind == "L1" else F.mse_loss(a, b)


class LossContext:
    """Holds target / reconstruction waveforms (B, N) and mels (B, F, T); caches STFT magnitudes per step."""
    def __init__(self, y, yhat, y_mels, yhat_mels):
        self.y, self.yhat = y, yhat
        self.y_mels, self.yhat_mels = y_mels, yhat_mels
        self._mags = {}

    def mags(self, n_fft):
        if n_fft not in self._mags:
            B = self.y.shape[0]
            x = torch.cat([self.y, self.yhat], dim=0)
            S = torch.stft(x, n_fft, hop_length=n_fft // 4, win_length=n_fft, window=hann(n_fft),
                           center=True, return_complex=True)
            mag = torch.sqrt(torch.view_as_real(S).pow(2).sum(-1) + 1e-9)      # (2B, F, T)
            self._mags[n_fft] = (mag[:B], mag[B:])
        return self._mags[n_fft]


def loss_vae_mel(ctx, c):
    return dist(ctx.yhat_mels, ctx.y_mels, c["dist"])


def loss_mr_stft(ctx, c):
    return get_mr_stft(int(c["n_bins"]), bool(c["perceptual"]))(ctx.yhat.unsqueeze(1), ctx.y.unsqueeze(1))


def loss_mel_mr(ctx, c):
    total, n = 0.0, int(c["n_res"])
    for i in range(n):
        n_fft = int(c["min_fft"]) * 2 ** i
        my, mh = ctx.mags(n_fft)
        fb = mel_fb(n_fft, int(c["n_mels"]))
        ly = torch.log(torch.einsum('bft,fm->bmt', my, fb) + 1e-5)
        lh = torch.log(torch.einsum('bft,fm->bmt', mh, fb) + 1e-5)
        total = total + dist(lh, ly, c["dist"])
    return total / n


def loss_lin_mr(ctx, c):
    total, n, bands = 0.0, int(c["n_res"]), int(c["n_bands"])
    for i in range(n):
        n_fft = int(c["min_fft"]) * 2 ** i
        my, mh = ctx.mags(n_fft)
        if bands > 0:
            my = F.adaptive_avg_pool1d(my.permute(0, 2, 1), bands)
            mh = F.adaptive_avg_pool1d(mh.permute(0, 2, 1), bands)
        total = total + dist(torch.log(mh + 1e-5), torch.log(my + 1e-5), c["dist"])
    return total / n


def loss_mfcc(ctx, c):
    tf = get_mfcc(int(c["n_mfcc"]), int(c["n_mels"]), int(c["n_fft"]))
    cy, ch = tf(ctx.y), tf(ctx.yhat)                         # (B, n_mfcc, T)
    if c["skip_c0"]:
        cy, ch = cy[:, 1:], ch[:, 1:]
    loss = dist(ch, cy, c["dist"])
    if c["deltas"]:
        d = torchaudio.functional.compute_deltas
        loss = loss + dist(d(ch), d(cy), c["dist"])
    return loss


def loss_loudness(ctx, c):
    if c["mode"].startswith("torchaudio"):
        def lo(x):
            v = torchaudio.functional.loudness(x.unsqueeze(1), audio_sample_rate)
            return torch.nan_to_num(v, nan=-70.0, neginf=-70.0, posinf=0.0)
        return dist(lo(ctx.yhat), lo(ctx.y), c["dist"]) / c["db_div"]
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    g = a_weight_gain(n_fft)[None, :, None]

    def lo(m):
        return 10.0 * torch.log10((m * g).pow(2).mean(1) + 1e-6)     # (B, T) dB
    return dist(lo(mh), lo(my), c["dist"]) / c["db_div"]


def loss_flatness(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    edges = band_edges(my.shape[1], int(c["n_bands"]))

    def flat_db(m):
        p = m.pow(2) + 1e-8
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            band = p[:, a:b, :]
            out.append((torch.log(band).mean(1) - torch.log(band.mean(1))) * 4.342944819)   # 10 / ln(10)
        return torch.stack(out, dim=1)                                # (B, bands, T)
    return dist(flat_db(mh), flat_db(my), c["dist"]) / c["db_div"]


def _centroid(m, n_fft, use_log2):
    f = freq_axis(n_fft)[None, :, None]
    cen = (f * m).sum(1) / (m.sum(1) + 1e-6)                          # (B, T) Hz
    return torch.log2(cen + 1.0) if use_log2 else cen / 1000.0


def loss_centroid(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    return dist(_centroid(mh, n_fft, c["log2"]), _centroid(my, n_fft, c["log2"]), c["dist"])


def loss_lw_centroid(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    cy, ch = _centroid(my, n_fft, c["log2"]), _centroid(mh, n_fft, c["log2"])
    w = (torch.sqrt(my.pow(2).sum(1)) + 1e-6).pow(c["exponent"]).detach()      # target frame amplitude
    w = w / w.sum(1, keepdim=True)
    return (w * (ch - cy).abs()).sum(1).mean()


def loss_temporal(ctx, c):
    frame, S = int(c["frame"]), int(c["segments"])
    hop = frame // 2

    def tc(x):
        env = torch.sqrt(x.unfold(-1, frame, hop).pow(2).mean(-1) + 1e-8)     # (B, T)
        T = (env.shape[1] // S) * S
        env = env[:, :T].reshape(env.shape[0], S, T // S)
        t = torch.linspace(0, 1, T // S, device=x.device)
        return (env * t).sum(-1) / (env.sum(-1) + 1e-8)                       # (B, S) in [0, 1]
    return dist(tc(ctx.yhat), tc(ctx.y), c["dist"])


LOSS_FUNCS = dict(vae_mel=loss_vae_mel, mr_stft=loss_mr_stft, mel_mr=loss_mel_mr, lin_mr=loss_lin_mr,
                  mfcc=loss_mfcc, loudness=loss_loudness, flatness=loss_flatness, centroid=loss_centroid,
                  lw_centroid=loss_lw_centroid, temporal=loss_temporal)

# -------------------------------------------------------------------------------------------------
# Audio data store (live-editable)
# -------------------------------------------------------------------------------------------------

class AudioStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.names, self.waves = [], []
        self.enabled, self.start, self.end = [], [], []

    def add_file(self, path):
        wave, sr = torchaudio.load(path)
        if sr != audio_sample_rate:
            wave = torchaudio.functional.resample(wave, sr, audio_sample_rate)
        wave = wave[0].to(device)
        with self.lock:
            self.names.append(os.path.basename(path))
            self.waves.append(wave)
            self.enabled.append(True)
            self.start.append(0)
            self.end.append(wave.shape[0])
        print(f"Loaded {path}: {wave.shape[0] / audio_sample_rate:.1f} s")

    def sample_batch(self, n):
        W = audio_window_length_vocos
        with self.lock:
            usable = np.array([(e - s - W + 1) if (en and e - s >= W) else 0
                               for en, s, e in zip(self.enabled, self.start, self.end)], dtype=np.float64)
            if usable.sum() <= 0:
                return None
            probs = usable / usable.sum()
            files = np.random.choice(len(usable), size=n, p=probs)
            out = []
            for f in files:
                st = self.start[f] + np.random.randint(0, int(usable[f]))
                out.append(self.waves[f][st:st + W])
        return torch.stack(out).unsqueeze(1)


store = AudioStore()

# -------------------------------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------------------------------

class Trainer:
    def __init__(self):
        self.model_lock = threading.Lock()
        self.enc, self.dec = build_models()
        self.inf_enc, self.inf_dec = build_models()
        if initial_encoder_weights and initial_decoder_weights:
            self.enc.load_state_dict(torch.load(initial_encoder_weights, map_location=device))
            self.dec.load_state_dict(torch.load(initial_decoder_weights, map_location=device))
        self._make_optimizer()
        self.sync_inference_models()
        self.inf_enc.eval()
        self.inf_dec.eval()
        self.step = 0
        self.running = True
        self.reset_req = False
        self.save_req = False
        self.load_req = None
        self.stats = dict(total=0.0, kld=0.0, beta=0.0, sps=0.0)
        self.loss_ema = {s["key"]: 0.0 for s in LOSS_SPECS}
        self.status_msg = ""

    def _make_optimizer(self):
        self.params = list(self.enc.parameters()) + list(self.dec.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=train_settings.learning_rate,
                                     betas=(0.8, 0.99), weight_decay=1e-4)

    @torch.no_grad()
    def sync_inference_models(self):
        with self.model_lock:
            self.inf_enc.load_state_dict(self.enc.state_dict())
            self.inf_dec.load_state_dict(self.dec.state_dict())

    def compute_beta(self, step):
        s = train_settings
        if not s.beta_cyclic:
            return s.max_beta
        c = step % max(1, s.beta_cycle_steps)
        if c < s.beta_min_const_steps:
            return s.min_beta
        if c > s.beta_cycle_steps - s.beta_max_const_steps:
            return s.max_beta
        ramp = max(1, s.beta_cycle_steps - s.beta_min_const_steps - s.beta_max_const_steps)
        return s.min_beta + (s.max_beta - s.min_beta) * (c - s.beta_min_const_steps) / ramp

    def train_step(self, y_wave, beta):
        s = train_settings
        B = y_wave.shape[0]
        cfgs = {k: dict(v) for k, v in loss_cfg.items()}               # snapshot for this step

        y = y_wave.squeeze(1)
        y_mels = get_mels(y)                                           # (B, F, audio_mel_count_vocos)
        mu, std = self.enc(group_mels(y_mels))                         # encoder works on groups of mel frames
        z = Encoder.reparameterize(mu, std)
        kld = kl_loss(mu, std)

        yhat_mels = self.dec(ungroup_latents(z, B))[:, 0]              # causal decoder over the whole timeline

        need_wave = any(cfgs[k]["weight"] != 0 for k in WAVEFORM_LOSS_KEYS)
        yhat_wave = vocos.decode(yhat_mels) if need_wave else None     # frozen Vocos, differentiable
        ctx = LossContext(y if need_wave else None, yhat_wave, y_mels, yhat_mels)

        rec = 0.0
        values = {}
        for spec in LOSS_SPECS:
            k = spec["key"]
            w = cfgs[k]["weight"]
            if w == 0:
                continue
            val = LOSS_FUNCS[k](ctx, cfgs[k])
            if not torch.isfinite(val):
                self.status_msg = f"Non-finite loss in '{k}', step skipped"
                self.opt.zero_grad()
                return None
            values[k] = val.detach()
            rec = rec + w * val

        total = s.rec_loss_scale * rec + beta * kld
        if not torch.is_tensor(total):
            total = beta * kld
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
        self.opt.step()
        return total.item(), kld.item(), {k: v.item() for k, v in values.items()}

    def save_checkpoint(self):
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        torch.save(self.enc.state_dict(), os.path.join(save_path, "weights", f"encoder_weights_step_{self.step}.pt"))
        torch.save(self.dec.state_dict(), os.path.join(save_path, "weights", f"decoder_weights_step_{self.step}.pt"))
        self.status_msg = f"Saved checkpoint at step {self.step}"

    def run(self):
        ema = 0.98
        last_t, last_step = time.time(), 0
        while self.running:
            try:
                if self.reset_req:
                    self.reset_req = False
                    self.enc, self.dec = build_models()
                    self._make_optimizer()
                    self.step = 0
                    self.sync_inference_models()
                    self.status_msg = "Model reset"
                if self.load_req:
                    enc_path, self.load_req = self.load_req, None
                    dec_path = os.path.join(os.path.dirname(enc_path),
                                            os.path.basename(enc_path).replace("encoder", "decoder"))
                    self.enc.load_state_dict(torch.load(enc_path, map_location=device))
                    self.dec.load_state_dict(torch.load(dec_path, map_location=device))
                    self.sync_inference_models()
                    self.status_msg = f"Loaded {os.path.basename(enc_path)}"
                if self.save_req:
                    self.save_req = False
                    self.save_checkpoint()

                if train_settings.paused:
                    time.sleep(0.05)
                    continue
                batch = store.sample_batch(train_settings.batch_size)
                if batch is None:
                    self.status_msg = "No usable training data (enable a file / widen its range)"
                    time.sleep(0.2)
                    continue

                for g in self.opt.param_groups:
                    g['lr'] = train_settings.learning_rate
                beta = self.compute_beta(self.step)
                result = self.train_step(batch, beta)
                if result is None:
                    time.sleep(0.05)
                    continue
                total, kld, values = result
                self.step += 1

                st = self.stats
                if self.step == 1:
                    st.update(total=total, kld=kld)
                st["total"] = ema * st["total"] + (1 - ema) * total
                st["kld"] = ema * st["kld"] + (1 - ema) * kld
                st["beta"] = beta
                for k in self.loss_ema:
                    if k in values:
                        prev = self.loss_ema[k]
                        self.loss_ema[k] = values[k] if prev == 0.0 else ema * prev + (1 - ema) * values[k]
                    else:
                        self.loss_ema[k] = 0.0

                if self.step % max(1, train_settings.sync_interval) == 0:
                    self.sync_inference_models()
                if autosave_interval_steps and self.step % autosave_interval_steps == 0:
                    self.save_checkpoint()

                now = time.time()
                if now - last_t > 1.0:
                    st["sps"] = (self.step - last_step) / (now - last_t)
                    last_t, last_step = now, self.step
            except Exception:
                traceback.print_exc()
                self.status_msg = "Training error, see console"
                time.sleep(0.5)

# -------------------------------------------------------------------------------------------------
# Real-time inference
# -------------------------------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self, trainer):
        self.trainer = trainer
        self.q = queue.Queue()
        self.running = True
        self.pos = 0
        self.leftover = np.zeros(0, dtype=np.float32)
        self.stream = None
        self.device_id = default_audio_output_device
        self.hann = torch.hann_window(inference_window_length, periodic=True).to(device)
        self.tail = np.zeros(inference_hop, dtype=np.float32)

    def _next_window(self):
        W = inference_window_length
        s = inf_settings
        idx = min(max(0, s.file_index), len(store.waves) - 1)
        wave = store.waves[idx]
        st = int(s.start_sec * audio_sample_rate)
        en = min(int(s.end_sec * audio_sample_rate), wave.shape[0])
        st = min(st, max(0, en - 1))
        if self.pos < st or self.pos + W > en:
            self.pos = st
        seg = wave[self.pos:self.pos + W]
        self.pos += inference_hop
        if seg.shape[0] < W:
            seg = F.pad(seg, (0, W - seg.shape[0]))
        return seg

    @torch.no_grad()
    def producer(self):
        K = inference_batch_windows
        while self.running:
            if self.q.qsize() > max_queued_blocks - K or len(store.waves) == 0:
                time.sleep(0.005)
                continue
            try:
                s = inf_settings
                x = torch.stack([self._next_window() for _ in range(K)])          # (K, W)
                with self.trainer.model_lock:
                    mels = get_mels(x)                                            # (K, F, inference_mel_count)
                    mu, std = self.trainer.inf_enc(group_mels(mels))
                    z = mu if s.use_mean else Encoder.reparameterize(mu, std)
                    z = ungroup_latents(z, K)                                     # (K, Cz, Tz)
                    if s.latent_noise > 0:
                        z = z + torch.randn_like(z) * s.latent_noise
                    gen = self.trainer.inf_dec(z)[:, 0]                           # (K, F, inference_mel_count)
                    wav = vocos.decode(gen)
                wav = (wav * self.hann * s.gain).cpu().numpy()
                for w in wav:
                    block = w[:inference_hop] + self.tail
                    self.tail = w[inference_hop:].copy()
                    self.q.put(np.clip(block, -1.0, 1.0).astype(np.float32))
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)

    def callback(self, outdata, frames, time_info, status):
        n = 0
        while n < frames:
            if self.leftover.shape[0] == 0:
                try:
                    self.leftover = self.q.get_nowait()
                except queue.Empty:
                    break
            take = min(frames - n, self.leftover.shape[0])
            outdata[n:n + take, 0] = self.leftover[:take]
            self.leftover = self.leftover[take:]
            n += take
        if n < frames:
            outdata[n:, 0] = 0.0

    def start(self):
        if self.stream is not None:
            return
        try:
            self.stream = sd.OutputStream(samplerate=audio_sample_rate, device=self.device_id, channels=1,
                                          dtype='float32', blocksize=1024, latency=0.15,
                                          callback=self.callback)
            self.stream.start()
        except Exception as e:
            print("Audio start error:", e)
            self.stream = None

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def set_device(self, device_id):
        self.device_id = device_id
        if self.stream is not None:
            self.stop()
            self.start()

# -------------------------------------------------------------------------------------------------
# GUI
# -------------------------------------------------------------------------------------------------

def dspin(value, lo, hi, step, decimals=3):
    b = QtWidgets.QDoubleSpinBox()
    b.setDecimals(decimals)
    b.setRange(lo, hi)
    b.setSingleStep(step)
    b.setValue(value)
    return b


class LiveGUI(QtWidgets.QWidget):
    def __init__(self, trainer, engine):
        super().__init__()
        self.trainer, self.engine = trainer, engine
        self.loss_labels = {}
        self.setWindowTitle("Live Audio VAE: train and listen")
        self.resize(950, 850)
        self.build_ui()
        self.refresh_files()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(250)

    def build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs)
        tabs.addTab(self.build_train_tab(), "Data and training")
        tabs.addTab(self.build_loss_tab(), "Losses")
        tabs.addTab(self.build_inference_tab(), "Live inference")
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setStyleSheet("font-family: monospace;")
        root.addWidget(self.stats_label)

    # --- tab 1
    def build_train_tab(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(tab)
        ts = train_settings

        data_box = QtWidgets.QGroupBox("Training data")
        dl = QtWidgets.QVBoxLayout(data_box)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemChanged.connect(self.on_file_toggled)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        dl.addWidget(self.file_list)
        row = QtWidgets.QHBoxLayout()
        self.range_start = dspin(0, 0, 1e5, 1.0)
        self.range_end = dspin(0, 0, 1e5, 1.0)
        self.range_start.valueChanged.connect(self.on_range_changed)
        self.range_end.valueChanged.connect(self.on_range_changed)
        add_btn = QtWidgets.QPushButton("Add audio files...")
        add_btn.clicked.connect(self.add_files)
        row.addWidget(QtWidgets.QLabel("Range of selected file, start (s):"))
        row.addWidget(self.range_start)
        row.addWidget(QtWidgets.QLabel("end (s):"))
        row.addWidget(self.range_end)
        row.addWidget(add_btn)
        dl.addLayout(row)
        root.addWidget(data_box)

        tp = QtWidgets.QGroupBox("Training parameters (live)")
        form = QtWidgets.QFormLayout(tp)
        self.w_rec = dspin(ts.rec_loss_scale, 0, 1e4, 0.5, 3)
        self.w_minb = dspin(ts.min_beta, 0, 100, 0.01, 5)
        self.w_maxb = dspin(ts.max_beta, 0, 100, 0.01, 5)
        self.w_cyc = QtWidgets.QCheckBox("Cyclic beta schedule (min -> max)")
        self.w_cyc.setChecked(ts.beta_cyclic)
        self.w_cyc_len = QtWidgets.QSpinBox(); self.w_cyc_len.setRange(10, 10**7); self.w_cyc_len.setValue(ts.beta_cycle_steps)
        self.w_cyc_min = QtWidgets.QSpinBox(); self.w_cyc_min.setRange(0, 10**7); self.w_cyc_min.setValue(ts.beta_min_const_steps)
        self.w_cyc_max = QtWidgets.QSpinBox(); self.w_cyc_max.setRange(0, 10**7); self.w_cyc_max.setValue(ts.beta_max_const_steps)
        self.w_lr = dspin(ts.learning_rate, 1e-7, 1.0, 1e-4, 7)
        self.w_bs = QtWidgets.QSpinBox(); self.w_bs.setRange(2, 512); self.w_bs.setValue(ts.batch_size)
        self.w_sync = QtWidgets.QSpinBox(); self.w_sync.setRange(1, 1000); self.w_sync.setValue(ts.sync_interval)
        form.addRow("Global rec loss scale:", self.w_rec)
        form.addRow("Min beta:", self.w_minb)
        form.addRow("Max beta:", self.w_maxb)
        form.addRow(self.w_cyc)
        form.addRow("Cycle length (steps):", self.w_cyc_len)
        form.addRow("Min-const steps:", self.w_cyc_min)
        form.addRow("Max-const steps:", self.w_cyc_max)
        form.addRow("Learning rate:", self.w_lr)
        form.addRow("Batch size:", self.w_bs)
        form.addRow("Sync to inference every N steps:", self.w_sync)
        for w in (self.w_rec, self.w_minb, self.w_maxb, self.w_lr, self.w_bs, self.w_sync,
                  self.w_cyc_len, self.w_cyc_min, self.w_cyc_max):
            w.valueChanged.connect(self.apply_train_settings)
        self.w_cyc.stateChanged.connect(self.apply_train_settings)
        root.addWidget(tp)

        ctl = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Pause training")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.on_pause)
        reset_btn = QtWidgets.QPushButton("Reset model")
        reset_btn.clicked.connect(lambda: setattr(self.trainer, "reset_req", True))
        save_btn = QtWidgets.QPushButton("Save checkpoint")
        save_btn.clicked.connect(lambda: setattr(self.trainer, "save_req", True))
        load_btn = QtWidgets.QPushButton("Load checkpoint...")
        load_btn.clicked.connect(self.load_checkpoint)
        for b in (self.pause_btn, reset_btn, save_btn, load_btn):
            ctl.addWidget(b)
        root.addLayout(ctl)
        root.addStretch(1)
        return tab

    # --- tab 2: dynamic loss widgets
    def build_loss_tab(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        for spec in LOSS_SPECS:
            lay.addWidget(self.build_loss_group(spec))
        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def build_loss_group(self, spec):
        key = spec["key"]
        box = QtWidgets.QGroupBox(spec["title"])
        form = QtWidgets.QFormLayout(box)

        w = dspin(loss_cfg[key]["weight"], 0, 1e4, 0.1, 4)
        w.valueChanged.connect(lambda v, k=key: loss_cfg[k].__setitem__("weight", v))
        form.addRow("Weight (0 = off):", w)

        for p in spec["params"]:
            pk, t = p["key"], p["type"]
            if t == "int":
                widget = QtWidgets.QSpinBox()
                widget.setRange(p["min"], p["max"])
                widget.setValue(p["default"])
                widget.valueChanged.connect(lambda v, k=key, q=pk: loss_cfg[k].__setitem__(q, int(v)))
            elif t == "float":
                widget = dspin(p["default"], p["min"], p["max"], p.get("step", 0.1), 3)
                widget.valueChanged.connect(lambda v, k=key, q=pk: loss_cfg[k].__setitem__(q, float(v)))
            elif t == "bool":
                widget = QtWidgets.QCheckBox()
                widget.setChecked(p["default"])
                widget.stateChanged.connect(lambda v, k=key, q=pk: loss_cfg[k].__setitem__(q, v != 0))
            else:
                widget = QtWidgets.QComboBox()
                widget.addItems(p["choices"])
                widget.setCurrentText(p["default"])
                widget.currentTextChanged.connect(lambda v, k=key, q=pk: loss_cfg[k].__setitem__(q, v))
            form.addRow(p["label"] + ":", widget)

        lbl = QtWidgets.QLabel("value: off")
        lbl.setStyleSheet("font-family: monospace;")
        form.addRow("Running value:", lbl)
        self.loss_labels[key] = lbl
        return box

    # --- tab 3
    def build_inference_tab(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(tab)
        ib = QtWidgets.QGroupBox("Live inference")
        il = QtWidgets.QFormLayout(ib)
        self.dev_combo = QtWidgets.QComboBox()
        self.dev_map = []
        default_out = sd.default.device[1] if default_audio_output_device is None else default_audio_output_device
        for i, d in enumerate(sd.query_devices()):
            if d['max_output_channels'] > 0:
                self.dev_map.append(i)
                self.dev_combo.addItem(f"{i}: {d['name']}")
                if i == default_out:
                    self.dev_combo.setCurrentIndex(len(self.dev_map) - 1)
        self.engine.device_id = self.dev_map[self.dev_combo.currentIndex()] if self.dev_map else None
        self.dev_combo.currentIndexChanged.connect(lambda i: self.engine.set_device(self.dev_map[i]))
        self.inf_file = QtWidgets.QComboBox()
        self.inf_file.currentIndexChanged.connect(self.on_inf_file)
        self.inf_start = dspin(0, 0, 1e5, 1.0)
        self.inf_end = dspin(0, 0, 1e5, 1.0)
        self.inf_start.valueChanged.connect(self.apply_inf_settings)
        self.inf_end.valueChanged.connect(self.apply_inf_settings)
        self.inf_mean = QtWidgets.QCheckBox("Use latent mean (no sampling)")
        self.inf_mean.stateChanged.connect(self.apply_inf_settings)
        self.inf_noise = dspin(0.0, 0, 100, 0.05, 3)
        self.inf_noise.valueChanged.connect(self.apply_inf_settings)
        self.inf_gain = dspin(1.0, 0, 10, 0.1, 2)
        self.inf_gain.valueChanged.connect(self.apply_inf_settings)
        il.addRow("Output device:", self.dev_combo)
        il.addRow("Source file:", self.inf_file)
        il.addRow("Source start (s):", self.inf_start)
        il.addRow("Source end (s):", self.inf_end)
        il.addRow(self.inf_mean)
        il.addRow("Latent noise:", self.inf_noise)
        il.addRow("Gain:", self.inf_gain)
        pb = QtWidgets.QHBoxLayout()
        start_btn = QtWidgets.QPushButton("Start audio"); start_btn.clicked.connect(self.engine.start)
        stop_btn = QtWidgets.QPushButton("Stop audio"); stop_btn.clicked.connect(self.engine.stop)
        pb.addWidget(start_btn); pb.addWidget(stop_btn)
        il.addRow(pb)
        root.addWidget(ib)
        root.addStretch(1)
        return tab

    # --- training data callbacks
    def refresh_files(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for name, en in zip(store.names, store.enabled):
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if en else QtCore.Qt.Unchecked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self.inf_file.blockSignals(True)
        self.inf_file.clear()
        self.inf_file.addItems(store.names)
        self.inf_file.setCurrentIndex(min(inf_settings.file_index, max(0, len(store.names) - 1)))
        self.inf_file.blockSignals(False)
        if store.names:
            self.file_list.setCurrentRow(0)
            self.on_inf_file(self.inf_file.currentIndex())

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add audio", audio_data_path, "Audio (*.wav *.flac *.mp3)")
        for p in paths:
            store.add_file(p)
        if paths:
            self.refresh_files()

    def on_file_toggled(self, item):
        store.enabled[self.file_list.row(item)] = item.checkState() == QtCore.Qt.Checked

    def on_file_selected(self, r):
        if r < 0 or r >= len(store.waves):
            return
        dur = store.waves[r].shape[0] / audio_sample_rate
        for b in (self.range_start, self.range_end):
            b.blockSignals(True)
            b.setRange(0, dur)
        self.range_start.setValue(store.start[r] / audio_sample_rate)
        self.range_end.setValue(store.end[r] / audio_sample_rate)
        for b in (self.range_start, self.range_end):
            b.blockSignals(False)

    def on_range_changed(self):
        r = self.file_list.currentRow()
        if r < 0:
            return
        s, e = self.range_start.value(), self.range_end.value()
        if e <= s:
            return
        with store.lock:
            store.start[r] = int(s * audio_sample_rate)
            store.end[r] = int(e * audio_sample_rate)

    def apply_train_settings(self, *_):
        ts = train_settings
        ts.rec_loss_scale = self.w_rec.value()
        ts.min_beta = self.w_minb.value()
        ts.max_beta = self.w_maxb.value()
        ts.beta_cyclic = self.w_cyc.isChecked()
        ts.beta_cycle_steps = self.w_cyc_len.value()
        ts.beta_min_const_steps = self.w_cyc_min.value()
        ts.beta_max_const_steps = self.w_cyc_max.value()
        ts.learning_rate = self.w_lr.value()
        ts.batch_size = self.w_bs.value()
        ts.sync_interval = self.w_sync.value()

    def on_pause(self, checked):
        train_settings.paused = checked
        self.pause_btn.setText("Resume training" if checked else "Pause training")

    def load_checkpoint(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select encoder weights (decoder is derived by name)", save_path)
        if p:
            self.trainer.load_req = p

    def on_inf_file(self, idx):
        if idx < 0 or idx >= len(store.waves):
            return
        dur = store.waves[idx].shape[0] / audio_sample_rate
        for b in (self.inf_start, self.inf_end):
            b.blockSignals(True)
            b.setRange(0, dur)
        self.inf_start.setValue(0.0)
        self.inf_end.setValue(dur)
        for b in (self.inf_start, self.inf_end):
            b.blockSignals(False)
        inf_settings.file_index = idx
        self.apply_inf_settings()

    def apply_inf_settings(self, *_):
        s = inf_settings
        s.file_index = max(0, self.inf_file.currentIndex())
        s.start_sec = self.inf_start.value()
        s.end_sec = self.inf_end.value()
        s.use_mean = self.inf_mean.isChecked()
        s.latent_noise = self.inf_noise.value()
        s.gain = self.inf_gain.value()

    def update_stats(self):
        st = self.trainer.stats
        parts = []
        for spec in LOSS_SPECS:
            k = spec["key"]
            v = self.trainer.loss_ema.get(k, 0.0)
            active = loss_cfg[k]["weight"] != 0
            self.loss_labels[k].setText(f"{v:.5f}  (weighted {v * loss_cfg[k]['weight']:.5f})" if active else "off")
            if active:
                parts.append(f"{k} {v:.4f}")
        self.stats_label.setText(
            f"step {self.trainer.step}  ({st['sps']:.1f} it/s)  beta {st['beta']:.4f}  total {st['total']:.4f}  kld {st['kld']:.4f}\n"
            f"{'  '.join(parts)}\n"
            f"audio queue {self.engine.q.qsize()}/{max_queued_blocks}   {self.trainer.status_msg}")

    def closeEvent(self, event):
        self.trainer.running = False
        self.engine.running = False
        self.engine.stop()
        event.accept()

# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    for f in audio_data_files:
        store.add_file(os.path.join(audio_data_path, f))

    trainer = Trainer()
    engine = InferenceEngine(trainer)
    threading.Thread(target=trainer.run, daemon=True).start()
    threading.Thread(target=engine.producer, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    gui = LiveGUI(trainer, engine)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()