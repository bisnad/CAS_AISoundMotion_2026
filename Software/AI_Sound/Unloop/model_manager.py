import os, sys, types, torch, unloop
import audiotools as at

CACHE_FILE = 'unloop_cache.pth'

def _gpus_arg_default():
    return None

def _peak_preprocess(self, sig: at.AudioSignal):
    peak = sig.audio_data.abs().max()
    if torch.is_tensor(peak):
        peak = peak.item()
    if peak > 0:
        sig.audio_data = sig.audio_data * (0.99 / peak)
    return sig

def setup_compatibility():
    """Setup PyTorch Lightning compatibility"""
    for mod_name in (
        "pytorch_lightning.utilities.argparse_utils",
        "lightning.pytorch.utilities.argparse_utils",
    ):
        fake = types.ModuleType(mod_name)
        fake._gpus_arg_default = _gpus_arg_default
        sys.modules[mod_name] = fake
    
    sys.modules["__main__"]._gpus_arg_default = _gpus_arg_default

def get_device():
    """Get the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_or_create_interface():
    """Load cached interface or create new one"""
    device = get_device()
    
    if os.path.exists(CACHE_FILE):
        print("Loading cached model...")
        try:
            # Recreate interface structure
            interface = unloop.interface.Interface.default()
            
            # Load cached states
            checkpoint = torch.load(CACHE_FILE, map_location='cpu')
            interface.coarse.load_state_dict(checkpoint['coarse'])
            if interface.c2f and checkpoint.get('c2f'):
                interface.c2f.load_state_dict(checkpoint['c2f'])
            interface.codec.load_state_dict(checkpoint['codec'])
            
            # Setup interface
            interface.float()
            if hasattr(interface, "to") and callable(getattr(interface, "to")):
                interface.to(device)
            interface._preprocess = _peak_preprocess.__get__(interface, type(interface))
            
            print(f"Cached model loaded on {device}")
            return interface, device
            
        except Exception as e:
            print(f"Failed to load cache: {e}. Creating new model...")
            os.remove(CACHE_FILE)  # Remove corrupted cache
    
    # Create new interface
    print("Creating new model (this will take a while)...")
    setup_compatibility()
    torch.set_float32_matmul_precision("high")
    
    interface = unloop.interface.Interface.default()
    interface.float()
    
    if hasattr(interface, "to") and callable(getattr(interface, "to")):
        interface.to(device)
    
    interface._preprocess = _peak_preprocess.__get__(interface, type(interface))
    
    # Cache the model
    try:
        torch.save({
            'coarse': interface.coarse.state_dict(),
            'c2f': interface.c2f.state_dict() if interface.c2f else None,
            'codec': interface.codec.state_dict(),
        }, CACHE_FILE)
        print(f"Model cached to {CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not cache model: {e}")
    
    return interface, device

def unload_model():
    """Remove the cached model"""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Model cache cleared")
    else:
        print("No cached model found")
