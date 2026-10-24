# -------------------------------------------------------------------------------------------------
# Audio Autoencoder - Training (1792-sample encoder window, causal streaming decoder, frozen Vocos)
# -------------------------------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random, os, time, csv, math
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.manifold import TSNE
from vocos import Vocos
import auraloss


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

force_fft_on_cpu = None           # None = auto-detect by probing; True/False = force
force_safe_copies = None          # None = auto (MPS or FFT fallback); True/False = force


def device_supports_fft(dev):
    """Probe the FFT ops (forward and backward) on the given device."""
    if dev == 'cpu':
        return True
    try:
        n_fft, hop = 1024, 256
        x = torch.rand(1, 8192, device=dev, requires_grad=True)
        win = torch.hann_window(n_fft, device=dev)
        spec = torch.stft(x, n_fft, hop_length=hop, win_length=n_fft, window=win,
                          center=True, return_complex=True)
        _ = torch.fft.irfft(spec, n_fft, dim=1, norm="backward")
        _ = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft, window=win)
        spec.abs().sum().backward()
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
    print(f"FFT ops not available on {device}: mel extraction and Vocos ISTFT head run on CPU "
          f"(everything else stays on {device}).")
else:
    print(f"FFT ops available on {device}: running everything on {device}.")

print("Employ Safe Slice Copies" if safe_copies else "No Safe Slice Copies Necessary")


def safe(x):
    """Independent contiguous copy of a (possibly sliced) tensor on MPS; identity otherwise."""
    return x.clone().contiguous() if safe_copies else x


if device == 'cuda':
    torch.backends.cudnn.benchmark = True


# -------------------------------------------------------------------------------------------------
# Audio Settings
# -------------------------------------------------------------------------------------------------

audio_data_path = "data/audio/"
#audio_data_files = ["Take1__double_Bind_HQ_audio_crop_48khz.wav", "Take2_Hibr_II_HQ_audio_crop_48khz.wav"]
#audio_valid_ranges = [[-1.0, -1.0], [-1.0, -1.0]]
audio_data_files = ["Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"]
audio_valid_ranges = [[-1.0, -1.0]]

audio_sample_rate = 48000
audio_mel_count_vocos = 256                # mel frames per Vocos window (training excerpt length)
audio_mel_count_vae = 8                    # mel frames per encoder group; must divide audio_mel_count_vocos
audio_window_length_vocos = None # Computed automatically from the mel counts and the Vocos hop size
audio_window_length_vae = None # Computed automatically from the mel counts and the Vocos hop size

mel_floor = -11.5                          # ~ln(1e-5); log-mels are clamped to this value (silence)

# -------------------------------------------------------------------------------------------------
# Save Paths
# -------------------------------------------------------------------------------------------------

save_path = "results/vae_cnn_conv_Gutenberg_ld32_simple"
save_weights_path = os.path.join(save_path, "weights/")
save_history_path = os.path.join(save_path, "histories/")
save_audio_path = os.path.join(save_path, "audio/")
for p in (save_weights_path, save_history_path, save_audio_path):
    os.makedirs(p, exist_ok=True)


# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

latent_channels = 32                       # latent channels per latent frame (time is downsampled 4x)
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]   # (freq, time); freq 128 -> 8, time /4
bottleneck_width = 512
time_downsample = 4                        # product of the time strides, keep consistent with conv_strides


# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

gain_aug_db = 6.0                          # random gain augmentation, uniform in +/- this many dB
batch_size = 32                            # excerpts per training batch
epochs = 400                               # number of training epochs

excerpts_per_epoch = 1000                  # random training excerpts drawn per epoch
test_excerpt_count = 128                   # fixed excerpts drawn from the held-out test region
test_fraction = 0.1                        # last 10% of each file is held out (time-separated)

lr_gen = 5e-4                              # AdamW learning rate of encoder and decoder
lr_final_factor = 0.01                     # cosine schedule decays the learning rate to this fraction
grad_clip = 1.0                            # max gradient norm

w_mel = 2.5                                # weight of the L1 loss between original and reconstructed log-mels
w_stft = 2.5                               # weight of the multi-resolution mel STFT loss on the waveform

# beta cycle
ae_beta_cycle_duration = 100               # length of one KL-weight (beta) cycle in epochs
ae_beta_min_const_duration = 20            # epochs at the start of each cycle with beta held at its minimum
ae_beta_max_const_duration = 20            # epochs at the end of each cycle with beta held at its maximum
ae_min_beta = 0.0                          # smallest KL weight (no KL pressure)
ae_max_beta = 0.01                         # largest KL weight

save_weights = True
load_weights = False
model_save_interval = 50
weights_tag = "results/vae_cnn_conv_Take2_Hibr_v2/weights/{}_weights_epoch_400.pt"

#audio_test_starts_sec = [[5, 50, 100, 140, 160], [5, 50, 100, 140, 160]]
audio_test_starts_sec = [[5, 50, 100, 140, 160]]
audio_test_duration_sec = 20
perturbation_sizes = [0.0, 0.1, 0.2, 0.5, 1.0]


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_all_seeds(42)


# -------------------------------------------------------------------------------------------------
# Vocoder (frozen, used only for mel extraction and mel -> waveform)
# -------------------------------------------------------------------------------------------------

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
for p in vocos.parameters():
    p.requires_grad = False

# Only if the device lacks FFT kernels: keep the FFT-dependent submodules on the CPU.
if fft_on_cpu:
    vocos.feature_extractor.to('cpu')
    vocos.head.to('cpu')


def get_mels(wave):
    # wave: (B, N) -> log-mels (B, n_mels, T) on the compute device
    if fft_on_cpu:
        m = vocos.feature_extractor(wave.cpu()).to(device)
    else:
        m = vocos.feature_extractor(wave.to(device))
    if m.ndim == 4:
        m = m[:, 0]
    return m.contiguous() if safe_copies else m


def vocos_decode(mel):
    # mel (B,F,T) on the compute device -> waveform (B,N) on the compute device
    # (Vocos is frozen, but gradients still flow through it back to the decoder)
    if fft_on_cpu:
        h = vocos.backbone(mel)
        return vocos.head(h.cpu()).to(device)
    return vocos.decode(mel)


def find_vocos_hop():
    # hop size in samples; read from the mel front end, otherwise estimated by probing
    hop = getattr(getattr(vocos.feature_extractor, "mel_spec", None), "hop_length", None)
    if hop is None:
        n = 480000
        with torch.no_grad():
            t = get_mels(torch.rand(1, n, device=device)).shape[-1]
        hop = int(round(n / (t - 1)))
    return int(hop)


# ---- derive the window lengths from the user-specified mel counts ----------------------------------

audio_hop = find_vocos_hop()

assert audio_mel_count_vocos % audio_mel_count_vae == 0, \
    "audio_mel_count_vocos must be an integer multiple of audio_mel_count_vae"
assert audio_mel_count_vae % time_downsample == 0, \
    "audio_mel_count_vae must be divisible by time_downsample"

# Vocos uses centered STFT frames: n_frames = n_samples // hop + 1  ->  n_samples = (n_frames - 1) * hop
audio_window_length_vocos = (audio_mel_count_vocos - 1) * audio_hop
audio_window_length_vae = (audio_mel_count_vae - 1) * audio_hop

# verify the derivation against the real feature extractor and vocoder
with torch.no_grad():
    _m = get_mels(torch.rand(1, audio_window_length_vocos, device=device))
    audio_mel_filter_count = _m.shape[1]
    assert _m.shape[2] == audio_mel_count_vocos, \
        f"Vocos window gives {_m.shape[2]} mel frames, expected {audio_mel_count_vocos}"
    assert vocos_decode(_m).shape[-1] == audio_window_length_vocos, "Vocos decode length mismatch"
    _mv = get_mels(torch.rand(1, audio_window_length_vae, device=device))
    assert _mv.shape[2] == audio_mel_count_vae, \
        f"VAE window gives {_mv.shape[2]} mel frames, expected {audio_mel_count_vae}"

audio_vae_mels_per_vocos_mels = audio_mel_count_vocos // audio_mel_count_vae     # groups per window
latent_steps_per_group = audio_mel_count_vae // time_downsample                  # latent steps per group

print(f"hop {audio_hop} samples | mel bins {audio_mel_filter_count}")
print(f"vocos: {audio_mel_count_vocos} mel frames -> {audio_window_length_vocos} samples "
      f"({audio_window_length_vocos / audio_sample_rate:.3f} s)")
print(f"vae:   {audio_mel_count_vae} mel frames -> {audio_window_length_vae} samples "
      f"({audio_window_length_vae / audio_sample_rate * 1000:.1f} ms)")
print(f"groups per vocos window {audio_vae_mels_per_vocos_mels} | latent steps per group {latent_steps_per_group}")


# -------------------------------------------------------------------------------------------------
# Grouping helpers (the encoder sees groups of audio_mel_count_vae mel frames, independently)
# -------------------------------------------------------------------------------------------------

def group_mels(mels):
    # (B, n_mels, T) -> (B*G, 1, n_mels, audio_mel_count_vae)
    B, n_mels, T = mels.shape
    assert T % audio_mel_count_vae == 0, "mel frame count must be a multiple of audio_mel_count_vae"
    G = T // audio_mel_count_vae
    return mels.reshape(B, n_mels, G, audio_mel_count_vae).permute(0, 2, 1, 3).reshape(B * G, 1, n_mels, audio_mel_count_vae)


def ungroup_mels(x, B):
    BG, n_mels, Tg = x.shape
    G = BG // B
    return x.reshape(B, G, n_mels, Tg).permute(0, 2, 1, 3).reshape(B, n_mels, G * Tg)


def group_latents(z):
    B, Cz, Tz = z.shape
    assert Tz % latent_steps_per_group == 0, "latent step count must be a multiple of latent_steps_per_group"
    G = Tz // latent_steps_per_group
    return z.reshape(B, Cz, G, latent_steps_per_group).permute(0, 2, 1, 3).reshape(B * G, Cz, latent_steps_per_group)


def ungroup_latents(z, B):
    BG, Cz, Tzg = z.shape
    G = BG // B
    return z.reshape(B, G, Cz, Tzg).permute(0, 2, 1, 3).reshape(B, Cz, G * Tzg)


# -------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------

def load_mono(path):
    w, sr = torchaudio.load(path)
    if sr != audio_sample_rate:
        w = torchaudio.functional.resample(w, sr, audio_sample_rate)
    return w[0].contiguous()


audio_all_data = []
for f, r in zip(audio_data_files, audio_valid_ranges):
    w = load_mono(audio_data_path + f)
    if r[0] >= 0 and r[1] > 0:
        w = w[int(r[0] * audio_sample_rate):int(r[1] * audio_sample_rate)].contiguous()
    audio_all_data.append(w)
    print(f, "duration [s]:", w.shape[0] / audio_sample_rate)


class ExcerptDataset(Dataset):
    def __init__(self, waves, regions, length, n_items, fixed, gain_db=0.0, seed=0):
        self.waves, self.regions, self.length, self.n_items = waves, regions, length, n_items
        self.fixed, self.gain_db = fixed, gain_db
        usable = np.array([max(0, e - s - length) for s, e in regions], dtype=np.float64)
        assert usable.sum() > 0, "region too short for the requested window"
        self.probs = usable / usable.sum()
        if fixed:
            rng = np.random.RandomState(seed)
            self.items = [self._draw(rng) for _ in range(n_items)]

    def _draw(self, rng):
        fi = int(rng.choice(len(self.waves), p=self.probs))
        s, e = self.regions[fi]
        return fi, int(rng.randint(s, e - self.length + 1))

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        fi, st = self.items[idx] if self.fixed else self._draw(np.random)
        x = self.waves[fi][st:st + self.length].clone()
        if self.gain_db > 0:
            x = x * 10 ** (random.uniform(-self.gain_db, self.gain_db) / 20)
        return x


train_regions = [(0, int(w.shape[0] * (1 - test_fraction))) for w in audio_all_data]
test_regions = [(int(w.shape[0] * (1 - test_fraction)), w.shape[0]) for w in audio_all_data]
train_dataset = ExcerptDataset(audio_all_data, train_regions, audio_window_length_vocos, excerpts_per_epoch, False, gain_aug_db)
test_dataset = ExcerptDataset(audio_all_data, test_regions, audio_window_length_vocos, test_excerpt_count, True, 0.0, seed=1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


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
    def forward(self, x):                      # x: (B,1,F,T)
        h = self.down(self.stem(x))            # (B,C,F',T')
        h = h.flatten(1, 2)                    # (B,C*F',T')
        mu, raw = self.head(h).chunk(2, dim=1)
        std = F.softplus(raw) + 1e-4
        return mu, std
    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)


# ---- causal building blocks for the streaming decoder -------------------------------------------

class FrameGroupNorm(nn.Module):
    """GroupNorm computed independently for every time step (statistics over channel group and frequency only),
    so the output at time t never depends on other time steps: required for causal streaming."""
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
    """Conv1d over time that only looks at the past. streaming=False: zero initial state (offline / training).
    streaming=True: the last k-1 input frames are cached between calls (chunk-wise inference)."""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c_in, c_out, k)
        self.streaming = False
        self.cache = None
    def forward(self, x):                                      # (B,C,T)
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], self.k - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k - 1, 0))
        return self.conv(x)


class CausalConv2d(nn.Module):
    """Conv2d with symmetric padding over frequency and causal (past-only) padding over time, optional cache."""
    def __init__(self, c_in, c_out, k_f=3, k_t=3):
        super().__init__()
        self.k_f, self.k_t = k_f, k_t
        self.conv = nn.Conv2d(c_in, c_out, (k_f, k_t))
        self.streaming = False
        self.cache = None
    def forward(self, x):                                      # (B,C,F,T)
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.k_t - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k_t - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k_t - 1, 0))
        x = F.pad(x, (0, 0, self.k_f // 2, self.k_f // 2))     # frequency padding
        return self.conv(x)


class CausalResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(fgn(ch), nn.SiLU(), CausalConv2d(ch, ch),
                                 fgn(ch), nn.SiLU(), CausalConv2d(ch, ch))
    def forward(self, x):
        return x + self.net(x)


class Decoder(nn.Module):                                      # causal, streaming-capable
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
    def forward(self, z):                      # z: (B,Cz,T')
        h = self.inp(z)
        h = h.reshape(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))            # (B,1,F,T)
    def _stateful(self):
        return [m for m in self.modules() if isinstance(m, (CausalConv1d, CausalConv2d))]
    def set_streaming(self, flag):             # switch between offline (zero state) and cached-state mode
        for m in self._stateful():
            m.streaming = flag
            m.cache = None
    def reset_state(self):                     # clear all caches (start of a new stream)
        for m in self._stateful():
            m.cache = None


encoder = Encoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
decoder = Decoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)


with torch.no_grad():
    _x = group_mels(get_mels(next(iter(train_loader)).to(device)).clamp(min=mel_floor))
    _B = _x.shape[0] // audio_vae_mels_per_vocos_mels
    _mu, _std = encoder(_x)
    _y = decoder(ungroup_latents(_mu, _B))
    print("mel", tuple(_x.shape), "latent", tuple(_mu.shape), "decoded", tuple(_y.shape))
    assert _mu.shape[-1] == latent_steps_per_group, "unexpected latent steps per group"
    assert _y.shape == (_B, 1, audio_mel_filter_count, audio_mel_count_vocos), "unexpected decoder output shape"
print("params enc/dec:", [sum(p.numel() for p in m.parameters()) for m in (encoder, decoder)])


@torch.no_grad()
def check_decoder(steps=64, t0=20, chunk_sizes=(1, 2, 7), tol=1e-2):
    # verifies (1) chunk-wise streaming with cached state == offline decoding, (2) the decoder is causal
    decoder.eval()
    z = torch.randn(1, latent_channels, steps, device=device)
    decoder.set_streaming(False)
    full = decoder(z)
    scale = full.abs().max().item()
    for cs in chunk_sizes:
        decoder.set_streaming(True)
        try:
            outs = [decoder(z[..., i:i + cs]) for i in range(0, steps, cs)]
        finally:
            decoder.set_streaming(False)
        err = (full - torch.cat(outs, dim=-1)).abs().max().item() / scale
        print(f"decoder streaming check, chunk of {cs} latent step(s): max relative error {err:.2e}")
        assert err < tol, "streaming output differs from offline output"
    z2 = z.clone()
    z2[..., t0] += 1.0
    diff = (decoder(z2) - full).abs().amax(dim=(0, 1, 2))
    changed = (diff > 1e-6 * scale).nonzero().flatten()
    assert changed.min().item() >= t0 * time_downsample, "decoder output depends on future latent steps"
    print(f"decoder causality check: latent step {t0} influences output frames "
          f"{changed.min().item()}..{changed.max().item()} "
          f"(past context of {changed.max().item() - t0 * time_downsample + 1} frames "
          f"= {(changed.max().item() - t0 * time_downsample + 1) * audio_hop / audio_sample_rate * 1000:.0f} ms)")


check_decoder()


if load_weights:
    encoder.load_state_dict(torch.load(weights_tag.format("encoder"), map_location=device))
    decoder.load_state_dict(torch.load(weights_tag.format("decoder"), map_location=device))


# -------------------------------------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------------------------------------

class MelMRSTFT(nn.Module):
    # (fft, hop, n_mel_bins): bins must be small enough that no mel filter is empty
    def __init__(self, cfgs, sample_rate):
        super().__init__()
        self.losses = nn.ModuleList([
            auraloss.freq.STFTLoss(fft_size=n, hop_size=h, win_length=n,
                                   scale="mel", n_bins=b, sample_rate=sample_rate,
                                   perceptual_weighting=True)
            for n, h, b in cfgs])
    def forward(self, x, y):
        return sum(l(x, y) for l in self.losses) / len(self.losses)


# With the FFT fallback the loss module lives on the CPU; inputs are moved there and the scalar back.
_stft_loss_module = MelMRSTFT([(512, 128, 32), (1024, 256, 96), (2048, 512, 128), (8192, 2048, 128)],
                              audio_sample_rate).to('cpu' if fft_on_cpu else device)


def stft_loss(x, y):
    if fft_on_cpu:
        return _stft_loss_module(x.cpu(), y.cpu()).to(device)
    return _stft_loss_module(x, y)


def kl_loss(mu, std):
    return -0.5 * torch.mean(1 + 2 * torch.log(std) - mu.pow(2) - std.pow(2))


def calc_ae_beta_values():
    vals = []
    for e in range(epochs):
        c = e % ae_beta_cycle_duration
        if c < ae_beta_min_const_duration:
            vals.append(ae_min_beta)
        elif c > ae_beta_cycle_duration - ae_beta_max_const_duration:
            vals.append(ae_max_beta)
        else:
            t = (c - ae_beta_min_const_duration) / (ae_beta_cycle_duration - ae_beta_min_const_duration - ae_beta_max_const_duration)
            vals.append(ae_min_beta + (ae_max_beta - ae_min_beta) * t)
    return vals
ae_beta_values = calc_ae_beta_values()


# -------------------------------------------------------------------------------------------------
# Optimiser
# -------------------------------------------------------------------------------------------------

ae_params = list(encoder.parameters()) + list(decoder.parameters())
opt = torch.optim.AdamW(ae_params, lr=lr_gen, betas=(0.8, 0.99), weight_decay=1e-4)
cos_fn = lambda e: lr_final_factor + (1 - lr_final_factor) * 0.5 * (1 + math.cos(math.pi * e / epochs))
sched = torch.optim.lr_scheduler.LambdaLR(opt, cos_fn)


# -------------------------------------------------------------------------------------------------
# Train / test steps
# -------------------------------------------------------------------------------------------------

def forward_vae(y, sample):
    B = y.shape[0]
    y_mels = get_mels(y).clamp(min=mel_floor)                                   # (B,F,256): full vocos window
    mu, std = encoder(group_mels(y_mels))                                       # groups of 8 frames, (B*G,Cz,2)
    z = Encoder.reparameterize(mu, std) if sample else mu
    yhat_mels = decoder(ungroup_latents(z, B))[:, 0]                            # causal decoder over whole timeline
    yhat = vocos_decode(yhat_mels)                                              # frozen Vocos
    return y_mels, yhat_mels, yhat, mu, std


def train_step(y, beta):
    y_mels, yhat_mels, yhat, mu, std = forward_vae(y, sample=True)
    mel_l = F.l1_loss(yhat_mels, y_mels)
    stft_l = stft_loss(yhat.unsqueeze(1), y.unsqueeze(1))
    kl_l = kl_loss(mu, std)
    total = w_mel * mel_l + w_stft * stft_l + beta * kl_l

    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(ae_params, grad_clip)
    opt.step()
    return [x.detach().item() for x in (total, mel_l, stft_l, kl_l)]


@torch.no_grad()
def test_step(y):
    y_mels, yhat_mels, yhat, mu, std = forward_vae(y, sample=False)
    mel_l = F.l1_loss(yhat_mels, y_mels)
    stft_l = stft_loss(yhat.unsqueeze(1), y.unsqueeze(1))
    return (w_mel * mel_l + w_stft * stft_l).item(), mel_l.item(), stft_l.item()


def train():
    keys = ["train", "test", "mel", "stft", "kld", "test_mel", "test_stft", "beta", "lr"]
    hist = {k: [] for k in keys}
    for epoch in range(epochs):
        t0 = time.time()
        beta = ae_beta_values[epoch]
        encoder.train(); decoder.train()
        tr = np.mean([train_step(b.to(device), beta) for b in train_loader], axis=0)
        encoder.eval(); decoder.eval()
        te = np.mean([test_step(b.to(device)) for b in test_loader], axis=0)
        for k, v in zip(keys, [tr[0], te[0], tr[1], tr[2], tr[3], te[1], te[2], beta, opt.param_groups[0]["lr"]]):
            hist[k].append(float(v))
        if save_weights and epoch % model_save_interval == 0:
            save_all(epoch)
        sched.step()
        print(f"epoch {epoch+1} train {tr[0]:.4f} test {te[0]:.4f} | mel {tr[1]:.3f} stft {tr[2]:.3f} kl {tr[3]:.4f} "
              f"| test mel {te[1]:.3f} stft {te[2]:.3f} | {time.time()-t0:.1f}s")
    return hist


def save_all(tag):
    torch.save(encoder.state_dict(), f"{save_weights_path}encoder_weights_epoch_{tag}.pt")
    torch.save(decoder.state_dict(), f"{save_weights_path}decoder_weights_epoch_{tag}.pt")


def plot_training_history(h, file_name):
    keys = [k for k in h if k not in ("beta", "lr")] + ["beta", "lr"]
    fig, axs = plt.subplots(3, 3, figsize=(14, 10))
    for ax, k in zip(axs.flat, keys):
        ax.plot(np.arange(1, len(h[k]) + 1), h[k]); ax.set_title(k); ax.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(file_name, dpi=150); plt.close()


def save_loss_as_csv(h, file_name):
    with open(file_name, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(h.keys()), lineterminator="\n"); w.writeheader()
        for i in range(len(h["train"])):
            w.writerow({k: h[k][i] for k in h})


# -------------------------------------------------------------------------------------------------
# Inference helpers (deterministic: posterior mean)
# -------------------------------------------------------------------------------------------------

audio_test_waveforms = {i: load_mono(audio_data_path + audio_data_files[i]).unsqueeze(0)
                        for i in range(len(audio_data_files)) if len(audio_test_starts_sec[i]) > 0}
audio_test_file_indices = list(audio_test_waveforms.keys())


@torch.no_grad()
def encode_audio(waveform, sample=False):
    # waveform: (1,N) -> latent timeline (1, Cz, Tz), original mel frame count T
    encoder.eval()
    mels = get_mels(safe(waveform).to(device)).clamp(min=mel_floor)
    T = mels.shape[-1]
    pad = (-T) % audio_mel_count_vae
    if pad:
        mels = F.pad(mels, (0, pad), mode="replicate")
    mu, std = encoder(group_mels(mels))
    z = Encoder.reparameterize(mu, std) if sample else mu
    return ungroup_latents(z, mels.shape[0]), T


@torch.no_grad()
def decode_latents(z, T, file_name=None):
    # offline decoding of a whole latent timeline (zero initial decoder state)
    decoder.eval()
    decoder.set_streaming(False)
    mels = safe(decoder(z.to(device))[:, 0, :, :T])
    wave = vocos_decode(mels).detach().cpu()
    if file_name is not None:
        torchaudio.save(file_name, wave.reshape(1, -1), audio_sample_rate)
    return wave


@torch.no_grad()
def decode_latents_streaming(z, T, chunk_steps=1, file_name=None):
    # simulates real-time use of the decoder: latent chunks are fed one after the other with cached state.
    # Vocos is still run offline on the stitched mels.
    decoder.eval()
    decoder.set_streaming(True)
    try:
        mel_chunks = [decoder(z[..., i:i + chunk_steps].to(device))[:, 0] for i in range(0, z.shape[-1], chunk_steps)]
    finally:
        decoder.set_streaming(False)
    mels = safe(torch.cat(mel_chunks, dim=-1)[:, :, :T])
    wave = vocos_decode(mels).detach().cpu()
    if file_name is not None:
        torchaudio.save(file_name, wave.reshape(1, -1), audio_sample_rate)
    return wave


def create_ref_audio(waveform, file_name):
    torchaudio.save(file_name, waveform, audio_sample_rate)


@torch.no_grad()
def create_voc_audio(waveform, file_name):
    # plain Vocos resynthesis of the real mels (upper quality bound of the pipeline)
    L, off = audio_window_length_vocos, audio_window_length_vocos // 2
    env = torch.hann_window(L)
    n = waveform.shape[1]
    out = torch.zeros(n)
    for i in range((n - L) // off + 1):
        seg = safe(waveform[:, i * off:i * off + L]).to(device)
        out[i * off:i * off + L] += vocos_decode(get_mels(seg)).detach().cpu()[0] * env
    torchaudio.save(file_name, out.reshape(1, -1), audio_sample_rate)


def create_2d_latent_space_representation(waveform, window_offset, max_windows=2000):
    n = waveform.shape[1]
    count = min((n - audio_window_length_vocos) // window_offset, max_windows)
    enc = []
    for i in range(count):
        seg = waveform[:, i * window_offset:i * window_offset + audio_window_length_vocos]
        z, _ = encode_audio(seg)
        enc.append(safe(z[:, :, z.shape[-1] // 2]).cpu())      # central latent frame (avoids edge effects)
    return torch.cat(enc, 0).numpy()


def distinct_hsv_colors(n, s=0.8, v=0.9):
    phi, h, cols = 0.618033988749895, 0.0, []
    for _ in range(n):
        cols.append((h % 1.0, s, v)); h += phi
    return cols


def create_audio_space_image(Z, ranges, file_name):
    cols = [hsv_to_rgb(c) for c in distinct_hsv_colors(len(ranges))]
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1], s=0.1, c="grey", alpha=0.2)
    for i, r in enumerate(ranges):
        ax.scatter(Z[r[0]:r[1], 0], Z[r[0]:r[1], 1], marker="o", facecolors="none", s=10.0, linewidths=0.4, edgecolors=cols[i], alpha=0.5)
    ax.set_xlabel("$c_1$"); ax.set_ylabel("$c_2$")
    fig.savefig(file_name, dpi=600); plt.close()


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    if save_weights:
        history = train()
        save_loss_as_csv(history, f"{save_history_path}history_{epochs}.csv")
        plot_training_history(history, f"{save_history_path}history_{epochs}.png")
        save_all(epochs)

        offset = audio_window_length_vocos // 2
        all_enc, ranges, run = [], [], 0
        for fi in audio_test_file_indices:
            enc = create_2d_latent_space_representation(audio_test_waveforms[fi], offset)
            for s in audio_test_starts_sec[fi]:
                ranges.append([run + int(s * audio_sample_rate) // offset,
                               run + int((s + audio_test_duration_sec) * audio_sample_rate) // offset])
            all_enc.append(enc); run += enc.shape[0]
        Z = TSNE(n_components=2, max_iter=5000, verbose=1).fit_transform(np.concatenate(all_enc, 0))
        create_audio_space_image(Z, ranges, f"{save_path}/audio_space_plot_epoch_{epochs}.png")

    for fi in audio_test_file_indices:
        wf = audio_test_waveforms[fi]
        tag = os.path.splitext(audio_data_files[fi])[0]
        for s in audio_test_starts_sec[fi]:
            a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
            seg, name = wf[:, a:b], f"{tag}_{s}-{s + audio_test_duration_sec}"

            create_ref_audio(seg, f"{save_audio_path}audio_ref_{name}.wav")
            create_voc_audio(seg, f"{save_audio_path}audio_voc_{name}.wav")

            z, T = encode_audio(seg)                                   # deterministic (mu)
            decode_latents(z, T, f"{save_audio_path}rec_audio_epochs_{epochs}_{name}.wav")
            decode_latents_streaming(z, T, 1, f"{save_audio_path}stream_audio_epochs_{epochs}_{name}.wav")

            # perturbation: noise level increases over time in len(perturbation_sizes) steps
            Tz = z.shape[-1]
            step = max(1, Tz // len(perturbation_sizes))
            sigma = torch.tensor([perturbation_sizes[min(i // step, len(perturbation_sizes) - 1)] for i in range(Tz)], device=z.device)
            zp = z + torch.randn_like(z) * sigma.view(1, 1, -1)
            decode_latents(zp, T, f"{save_audio_path}perturb_audio_{name}.wav")

        # interpolation between consecutive test excerpts
        starts = audio_test_starts_sec[fi]
        for i in range(len(starts) - 1):
            zs = []
            for s in (starts[i], starts[i + 1]):
                a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
                z, T = encode_audio(wf[:, a:b]); zs.append(z)
            n = min(zs[0].shape[-1], zs[1].shape[-1])
            mix = torch.linspace(0, 1, n, device=zs[0].device).view(1, 1, -1)
            zm = zs[0][..., :n] * (1 - mix) + zs[1][..., :n] * mix
            decode_latents(zm, min(T, n * time_downsample),
                           f"{save_audio_path}mix_audio_{tag}_{starts[i]}_{starts[i+1]}.wav")