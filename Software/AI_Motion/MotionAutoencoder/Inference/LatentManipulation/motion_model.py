import torch
from torch import nn
import torch.nn.functional as nnF

config = {
    "vae_input_dim": 138,
    "vae_latent_dim": 16,
    "vae_conv_channel_counts": [128, 128, 128],
    "vae_conv_kernel_sizes": [3, 3, 3, 4],
    "vae_conv_strides": [2, 2, 2, 2],
    "vae_conv_dilations": [1, 2, 4, 1],
    "vae_window_length": 64,
    "device": "cuda",
    "vae_weights_path": "",
}


def resolve_device(device_str):
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation)

    def forward(self, x):
        # Use 'replicate' to stretch the first frame backwards instead of dropping to 0
        x_padded = nnF.pad(x, (self.pad, 0), mode='replicate')
        return self.conv(x_padded)

class MotionEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels, channel_counts, kernel_sizes, strides, dilations):
        super().__init__()
        layers = []
        current_channels = in_channels

        for i in range(len(kernel_sizes)):
            out_channels = channel_counts[i] if i < len(channel_counts) else latent_channels * 2
            layers.append(CausalConv1d(current_channels, out_channels, kernel_size=kernel_sizes[i], stride=strides[i], dilation=dilations[i]))
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
            current_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MotionDecoder(nn.Module):
    def __init__(self, out_channels, latent_channels, channel_counts, kernel_sizes, strides, dilations):
        super().__init__()
        layers = []
        current_channels = latent_channels

        for i in range(len(kernel_sizes)):
            if strides[i] > 1:
                layers.append(nn.Upsample(scale_factor=strides[i], mode="nearest"))
            
            next_channels = channel_counts[i] if i < len(channel_counts) else out_channels
            layers.append(CausalConv1d(current_channels, next_channels, kernel_size=kernel_sizes[i], stride=1, dilation=dilations[i]))
            
            if i < len(kernel_sizes) - 1:
                layers.append(nn.GELU())
            current_channels = next_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CausalMotionVAE(nn.Module):
    def __init__(self, input_dim=63, latent_dim=16, conv_channel_counts=[128, 128, 128], conv_kernel_sizes=[3, 3, 3, 4], conv_strides=[1, 1, 1, 2], conv_dilations=[1, 2, 4, 1]):
        super().__init__()

        self.input_dim = input_dim
        self.encoder = MotionEncoder(in_channels=input_dim, latent_channels=latent_dim, channel_counts=conv_channel_counts, kernel_sizes=conv_kernel_sizes, strides=conv_strides, dilations=conv_dilations)
        
        rev_channels = list(reversed(conv_channel_counts))
        rev_kernels = list(reversed(conv_kernel_sizes))
        rev_strides = list(reversed(conv_strides))
        rev_dilations = list(reversed(conv_dilations))
        
        self.decoder = MotionDecoder(out_channels=input_dim, latent_channels=latent_dim, channel_counts=rev_channels, kernel_sizes=rev_kernels, strides=rev_strides, dilations=rev_dilations)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "model_state_dict" in loaded_obj:
            return loaded_obj["model_state_dict"]
        if "state_dict" in loaded_obj:
            return loaded_obj["state_dict"]
    return loaded_obj


def load_model_weights(model, weights_path, device):
    if weights_path is None:
        return model

    if isinstance(weights_path, (list, tuple)):
        weights_path = weights_path[0] if len(weights_path) > 0 else ""

    if weights_path == "":
        return model

    loaded_obj = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(loaded_obj)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def infer_latent_shape(model, seq_window_length, device):
    dummy = torch.zeros(1, model.input_dim, seq_window_length, device=device)
    mu, _ = model.encode(dummy)
    return tuple(mu.shape[1:])


def createModels(config_override=None):
    cfg = dict(config)
    if config_override is not None:
        cfg.update(config_override)

    device = resolve_device(cfg["device"])

    vae = CausalMotionVAE(
        input_dim=cfg["vae_input_dim"],
        latent_dim=cfg["vae_latent_dim"],
        conv_channel_counts=cfg["vae_conv_channel_counts"],
        conv_kernel_sizes=cfg["vae_conv_kernel_sizes"],
        conv_strides=cfg["vae_conv_strides"],
        conv_dilations=cfg["vae_conv_dilations"],
    ).to(device)

    vae = load_model_weights(vae, cfg.get("vae_weights_path", ""), device)
    vae.eval()

    window_length = cfg.get("vae_window_length", 64)
    vae.window_length = window_length
    vae.latent_shape = infer_latent_shape(vae, window_length, device)
    vae.latent_steps = vae.latent_shape[-1]

    return vae