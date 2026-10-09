# -------------------------------------------------------------------------------------------------
# Audio Autoencoder - Training
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
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
                          center=True, return_complex=True)                       # _fft_r2c
        _ = torch.fft.irfft(spec, n_fft, dim=1, norm="backward")                  # _fft_c2r
        _ = torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft, window=win)
        spec.abs().sum().backward()                                               # backward pass
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

if device == 'cuda':
    # speed optimisation for CNNs on NVIDIA GPUs
    torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------------------------------
# Audio Settings
# -------------------------------------------------------------------------------------------------

audio_data_path = "data/audio/"
audio_data_files = ["Take1__double_Bind_HQ_audio_crop_48khz.wav", "Take2_Hibr_II_HQ_audio_crop_48khz.wav"]
audio_valid_ranges = [[-1.0, -1.0], [-1.0, -1.0]]

audio_sample_rate = 48000
audio_window_length_vocos = 65280          # 256 mel frames
mel_floor = -11.5                          # ~ln(1e-5); log-mels are clamped to this value (silence)

# -------------------------------------------------------------------------------------------------
# Save Paths
# -------------------------------------------------------------------------------------------------

save_path = "results/vae_cnn_Stocos_ld32_test"
save_weights_path = os.path.join(save_path, "weights/")
save_history_path = os.path.join(save_path, "histories/")
save_audio_path = os.path.join(save_path, "audio/")
for p in (save_weights_path, save_history_path, save_audio_path):
    os.makedirs(p, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

# autoencoder
latent_channels = 32                       # latent channels per latent frame (time is downsampled 4x)
conv_channel_counts = [32, 64, 128, 256]
conv_strides = [(2, 1), (2, 2), (2, 1), (2, 2)]   # (freq, time); freq 128 -> 8, time /4
bottleneck_width = 512                     # number of channels in the hidden layers of the last 1D convolution stack
time_downsample = 4                        # product of the time strides, keep consistent with conv_strides

# multi-resolution spectral discriminator
disc_channels = 16
disc_crop = 24576 #  length in samples of the audio snippet for the discriminator
disc_resolutions = [(512, 128), (1024, 256), (2048, 512)] # (n_fft, hop) pairs for different spectral resolutions

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

gain_aug_db = 6.0                          # random gain augmentation, uniform in +/- this many dB
batch_size = 32                            # number of audio excerpts per training batch
epochs = 400                               # number of training epochs

# training / test split
test_fraction = 0.1                        # last 10% of each file is held out (time-separated)
excerpts_per_epoch = 1000                  # number of  excerpts drawn from training region per epoch
test_excerpt_count = 128                   # number of excerpts drawn from the held-out test region per epoch

# learning rates
lr_gen = 5e-4                              # learning rate of encoder and decoder
lr_vocos = 1e-5                            # learning rate for fine-tuning the Vocos backbone and head
lr_disc = 2e-4                             # learning rate of the discriminator
lr_final_factor = 0.01                     # cosine schedule decays the learning rates to this fraction
grad_clip = 1.0                            # max gradient norm for the generator side (encoder, decoder, Vocos)

# loss weights
w_mel = 2.5                                # weight of the L1 loss between original and reconstructed log-mels
w_stft = 2.5                               # weight of the multi-resolution mel STFT loss on the waveform
w_anchor = 1.0                             # vocoder anchoring on real mels
anchor_batch = 8                           # max. number of batch items used for the anchor loss
w_adv = 1.0                                # weight of the adversarial (generator) loss
w_fm = 2.0                                 # weight of the adversarial (discriminator) loss

vocos_unfreeze_epoch = 40                  # start fine-tuning Vocos
gan_start_epoch = 60                       # start adversarial training

# beta cycle
ae_beta_cycle_duration = 100
ae_beta_min_const_duration = 20
ae_beta_max_const_duration = 20
ae_min_beta = 0.0
ae_max_beta = 0.01

save_weights = True
load_weights = False
model_save_interval = 50
weights_tag = "results/vae_cnn_Stocos_ld32/weights/{}_weights_epoch_400"

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

# -------------------------------------------------------------------------------------------------
# Vocoder
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
    # mel (B,F,T) on the compute device -> waveform (B,N) on the compute device (differentiable)
    if fft_on_cpu:
        h = vocos.backbone(mel)                        # on device
        return vocos.head(h.cpu()).to(device)          # linear + ISTFT on CPU; grads flow back
    return vocos.decode(mel)

with torch.no_grad():
    _m = get_mels(torch.rand(1, audio_window_length_vocos, device=device))
    audio_mel_filter_count, audio_mel_count_vocos = _m.shape[1], _m.shape[2]
    assert vocos_decode(_m).shape[-1] == audio_window_length_vocos
audio_hop = audio_window_length_vocos // (audio_mel_count_vocos - 1)
assert audio_mel_count_vocos % time_downsample == 0, "mel frame count must be divisible by time_downsample"
print("mel frames", audio_mel_count_vocos, "mel bins", audio_mel_filter_count, "hop", audio_hop)

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

# Autoencoder

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

# Discriminator

class SpecDiscriminator(nn.Module):
    def __init__(self, n_fft, hop, ch):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))
        sn = nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            sn(nn.Conv2d(1, ch, (3, 9), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            sn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)))])
        self.out = sn(nn.Conv2d(ch, 1, (3, 3), padding=1))
    def forward(self, x):                      # x: (B,N)
        if fft_on_cpu:                         # STFT on CPU (differentiable), rest on device
            s = torch.stft(x.cpu(), self.n_fft, self.hop, window=self.window.cpu(), return_complex=True).abs().to(x.device)
        else:
            s = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True).abs()
        h = torch.log(s.clamp(min=1e-5)).unsqueeze(1)
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            feats.append(h)
        return self.out(h), feats

class MultiResDiscriminator(nn.Module):
    def __init__(self, resolutions, ch):
        super().__init__()
        self.discs = nn.ModuleList([SpecDiscriminator(n, h, ch) for n, h in resolutions])
    def forward(self, x):
        outs, feats = [], []
        for d in self.discs:
            o, f = d(x)
            outs.append(o); feats.append(f)
        return outs, feats

encoder = Encoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
decoder = Decoder(latent_channels, audio_mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
disc = MultiResDiscriminator(disc_resolutions, disc_channels).to(device)

with torch.no_grad():
    _x = get_mels(next(iter(train_loader)).to(device)).clamp(min=mel_floor).unsqueeze(1)
    _mu, _std = encoder(_x)
    _y = decoder(_mu)
    print("mel", tuple(_x.shape), "latent", tuple(_mu.shape), "decoded", tuple(_y.shape))
    assert _y.shape == _x.shape
print("params enc/dec/disc:", [sum(p.numel() for p in m.parameters()) for m in (encoder, decoder, disc)])

if load_weights:
    encoder.load_state_dict(torch.load(weights_tag.format("encoder"), map_location=device))
    decoder.load_state_dict(torch.load(weights_tag.format("decoder"), map_location=device))
    vw = torch.load(weights_tag.format("vocos"), map_location=device)
    vocos.backbone.load_state_dict(vw["backbone"]); vocos.head.load_state_dict(vw["head"])

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

# perceptual loss
def stft_loss(x, y):
    if fft_on_cpu:
        return _stft_loss_module(x.cpu(), y.cpu()).to(device)
    return _stft_loss_module(x, y)

# kl divergence loss
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
# Optimisers
# -------------------------------------------------------------------------------------------------

vocos_params = list(vocos.backbone.parameters()) + list(vocos.head.parameters())
# parameters live on two devices when the FFT fallback is active -> disable the fused/foreach paths
_opt_kwargs = {"foreach": False} if fft_on_cpu else {}
gen_opt = torch.optim.AdamW([
    {"params": list(encoder.parameters()) + list(decoder.parameters()), "lr": lr_gen},
    {"params": vocos_params, "lr": lr_vocos}], betas=(0.8, 0.99), weight_decay=1e-4, **_opt_kwargs)
disc_opt = torch.optim.AdamW(disc.parameters(), lr=lr_disc, betas=(0.8, 0.99))
cos_fn = lambda e: lr_final_factor + (1 - lr_final_factor) * 0.5 * (1 + math.cos(math.pi * e / epochs))
gen_sched = torch.optim.lr_scheduler.LambdaLR(gen_opt, cos_fn)
disc_sched = torch.optim.lr_scheduler.LambdaLR(disc_opt, cos_fn)

def set_vocos_trainable(flag):
    for p in vocos_params:
        p.requires_grad = flag

# -------------------------------------------------------------------------------------------------
# Train / test steps
# -------------------------------------------------------------------------------------------------

def forward_vae(y, sample):
    y_mels = get_mels(y).clamp(min=mel_floor)
    mu, std = encoder(y_mels.unsqueeze(1))
    z = Encoder.reparameterize(mu, std) if sample else mu
    yhat_mels = decoder(z)[:, 0]
    yhat = vocos_decode(yhat_mels)
    return y_mels, yhat_mels, yhat, mu, std

def train_step(y, beta, use_gan):
    y_mels, yhat_mels, yhat, mu, std = forward_vae(y, sample=True)
    mel_l = F.l1_loss(yhat_mels, y_mels)
    stft_l = stft_loss(yhat.unsqueeze(1), y.unsqueeze(1))
    kl_l = kl_loss(mu, std)
    total = w_mel * mel_l + w_stft * stft_l + beta * kl_l

    if next(iter(vocos_params)).requires_grad:
        k = min(anchor_batch, y.shape[0])
        y_anchor = vocos_decode(get_mels(safe(y[:k])))
        anchor_l = stft_loss(y_anchor.unsqueeze(1), y[:k].unsqueeze(1))
        total = total + w_anchor * anchor_l

    adv_l = fm_l = d_l = torch.zeros((), device=y.device)
    if use_gan:
        st = random.randint(0, y.shape[-1] - disc_crop)
        real_c, fake_c = safe(y[:, st:st + disc_crop]), safe(yhat[:, st:st + disc_crop])
        disc_opt.zero_grad()
        o_r, _ = disc(real_c)
        o_f, _ = disc(fake_c.detach())
        d_l = sum(((a - 1) ** 2).mean() + (b ** 2).mean() for a, b in zip(o_r, o_f)) / len(o_r)
        d_l.backward()
        disc_opt.step()

        with torch.no_grad():
            _, f_r = disc(real_c)
        o_f, f_f = disc(fake_c)
        adv_l = sum(((b - 1) ** 2).mean() for b in o_f) / len(o_f)
        fm_l = sum(F.l1_loss(a, b.detach()) for fr, ff in zip(f_r, f_f) for a, b in zip(ff, fr)) / sum(len(f) for f in f_r)
        total = total + w_adv * adv_l + w_fm * fm_l

    gen_opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + vocos_params, grad_clip)
    gen_opt.step()
    return [x.detach().item() for x in (total, mel_l, stft_l, kl_l, adv_l, fm_l, d_l)]

@torch.no_grad()
def test_step(y):
    y_mels, yhat_mels, yhat, mu, std = forward_vae(y, sample=False)
    mel_l = F.l1_loss(yhat_mels, y_mels)
    stft_l = stft_loss(yhat.unsqueeze(1), y.unsqueeze(1))
    return (w_mel * mel_l + w_stft * stft_l).item(), mel_l.item(), stft_l.item()

def train():
    keys = ["train", "test", "mel", "stft", "kld", "adv", "fm", "disc", "test_mel", "test_stft", "beta", "lr"]
    hist = {k: [] for k in keys}
    set_vocos_trainable(False)
    for epoch in range(epochs):
        t0 = time.time()
        if epoch == vocos_unfreeze_epoch:
            set_vocos_trainable(True)
        use_gan = epoch >= gan_start_epoch
        beta = ae_beta_values[epoch]
        encoder.train(); decoder.train(); disc.train()
        tr = np.mean([train_step(b.to(device), beta, use_gan) for b in train_loader], axis=0)
        encoder.eval(); decoder.eval()
        te = np.mean([test_step(b.to(device)) for b in test_loader], axis=0)
        for k, v in zip(keys, [tr[0], te[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], te[1], te[2], beta, gen_opt.param_groups[0]["lr"]]):
            hist[k].append(float(v))
        if save_weights and epoch % model_save_interval == 0:
            save_all(epoch)
        gen_sched.step(); disc_sched.step()
        print(f"epoch {epoch+1} train {tr[0]:.4f} test {te[0]:.4f} | mel {tr[1]:.3f} stft {tr[2]:.3f} kl {tr[3]:.4f} "
              f"adv {tr[4]:.3f} fm {tr[5]:.3f} D {tr[6]:.3f} | test mel {te[1]:.3f} stft {te[2]:.3f} | {time.time()-t0:.1f}s")
    return hist

def save_all(tag):
    torch.save(encoder.state_dict(), f"{save_weights_path}encoder_weights_epoch_{tag}")
    torch.save(decoder.state_dict(), f"{save_weights_path}decoder_weights_epoch_{tag}")
    torch.save({"backbone": vocos.backbone.state_dict(), "head": vocos.head.state_dict()}, f"{save_weights_path}vocos_weights_epoch_{tag}")
    torch.save(disc.state_dict(), f"{save_weights_path}disc_weights_epoch_{tag}")

def plot_training_history(h, file_name):
    keys = [k for k in h if k not in ("beta", "lr")] + ["beta", "lr"]
    fig, axs = plt.subplots(4, 3, figsize=(16, 14))
    for ax, k in zip(axs.flat, keys):
        ax.plot(np.arange(1, len(h[k]) + 1), h[k]); ax.set_title(k); ax.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(file_name, dpi=150); plt.close()

def save_loss_as_csv(h, file_name):
    with open(file_name, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(h.keys()), lineterminator="\n"); w.writeheader()
        for i in range(len(h["train"])):
            w.writerow({k: h[k][i] for k in h})

# -------------------------------------------------------------------------------------------------
# Inference 
# -------------------------------------------------------------------------------------------------

audio_test_starts_sec = [[5, 50, 100, 140, 160], [5, 50, 100, 140, 160]]
audio_test_duration_sec = 20
perturbation_sizes = [0.0, 0.1, 0.2, 0.5, 1.0]

audio_test_waveforms = {i: load_mono(audio_data_path + audio_data_files[i]).unsqueeze(0)
                        for i in range(len(audio_data_files)) if len(audio_test_starts_sec[i]) > 0}
audio_test_file_indices = list(audio_test_waveforms.keys())

@torch.no_grad()
def encode_audio(waveform, sample=False):
    # waveform: (1,N) -> latents (1, Cz, T'), original mel frame count
    encoder.eval()
    mels = get_mels(safe(waveform).to(device)).clamp(min=mel_floor)
    T = mels.shape[-1]
    pad = (-T) % time_downsample
    if pad:
        mels = F.pad(mels, (0, pad), mode="replicate")
    mu, std = encoder(mels.unsqueeze(1))
    z = Encoder.reparameterize(mu, std) if sample else mu
    return z, T

@torch.no_grad()
def decode_latents(z, T, file_name=None):
    decoder.eval()
    mels = safe(decoder(z.to(device))[:, 0, :, :T])
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

            # perturbation: noise level increases over time in len(perturbation_sizes) steps
            Tz = z.shape[-1]
            step = max(1, Tz // len(perturbation_sizes))
            sigma = torch.tensor([perturbation_sizes[min(i // step, len(perturbation_sizes) - 1)] for i in range(Tz)], device=z.device)
            zp = z + torch.randn_like(z) * sigma.view(1, 1, -1)
            decode_latents(zp, T, f"{save_audio_path}perturb_audio_{name}.wav")

        # interpolation between consecutive test excerpts
        starts = audio_test_starts_sec[fi]
        for i in range(len(starts) - 1):
            segs, zs = [], []
            for s in (starts[i], starts[i + 1]):
                a, b = int(s * audio_sample_rate), int((s + audio_test_duration_sec) * audio_sample_rate)
                z, T = encode_audio(wf[:, a:b]); zs.append(z)
            n = min(zs[0].shape[-1], zs[1].shape[-1])
            mix = torch.linspace(0, 1, n, device=zs[0].device).view(1, 1, -1)
            zm = zs[0][..., :n] * (1 - mix) + zs[1][..., :n] * mix
            decode_latents(zm, min(T, n * time_downsample),
                           f"{save_audio_path}mix_audio_{tag}_{starts[i]}_{starts[i+1]}.wav")