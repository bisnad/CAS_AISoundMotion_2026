import sys, types, time, os, torch, unloop
import audiotools as at
from contextlib import nullcontext
import model_cache  # Import the cache

def _gpus_arg_default():
    return None

def _peak_preprocess(self, sig: at.AudioSignal):
    peak = sig.audio_data.abs().max()
    if torch.is_tensor(peak):
        peak = peak.item()
    if peak > 0:
        sig.audio_data = sig.audio_data * (0.99 / peak)
    return sig

# Setup compatibility
for mod_name in (
    "pytorch_lightning.utilities.argparse_utils",
    "lightning.pytorch.utilities.argparse_utils",
):
    fake = types.ModuleType(mod_name)
    fake._gpus_arg_default = _gpus_arg_default
    sys.modules[mod_name] = fake

sys.modules["__main__"]._gpus_arg_default = _gpus_arg_default

# Device detection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_float32_matmul_precision("high")

# Load and prepare interface
interface = unloop.interface.Interface.default()
interface.float()

if hasattr(interface, "to") and callable(getattr(interface, "to")):
    interface.to(device)

interface._preprocess = _peak_preprocess.__get__(interface, type(interface))

# Store in shared cache
model_cache.set_interface(interface, device)

print("Model preloaded successfully!")
