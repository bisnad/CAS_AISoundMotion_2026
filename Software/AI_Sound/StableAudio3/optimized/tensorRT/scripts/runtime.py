"""
Runtime support: tokenizer + distribution-shift constructors.

Everything model-dependent (tokenizer, T5Gemma engine, DiT engines, decoder
engines) ships from HuggingFace under models/. The DiT engines bundle their
per-DiT conditioner tensors (padding_embedding + seconds_total Linear) as
graph Constants, and DistributionShift's schedule constants are SA3-canonical
and live as defaults on the class.
"""

import math
import os
import torch

# runtime.py lives in scripts/; models/ lives one level up at the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_REPO, "models")
# ARCH_DIR is set by sa3_trt.py before calling load() (it knows the GPU arch).
ARCH_DIR = MODELS_DIR  # placeholder; overwritten by sa3_trt.py
IO_CHANNELS = 256
SAMPLE_RATE = 44100
DOWNSAMPLING_RATIO = 4096


class FastTokenizer:
    """Tokenizer using HuggingFace tokenizers lib directly (no transformers import)."""
    def __init__(self, tokenizer_json_path, max_length=256, pad_token_id=0):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_json_path)
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(self, text, return_tensors="pt", max_length=None, padding="max_length", truncation=True):
        max_len = max_length or self.max_length
        enc = self.tokenizer.encode(text)
        ids = enc.ids[:max_len]
        mask = [1] * len(ids)
        if padding == "max_length":
            pad_len = max_len - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}


def load_tokenizer():
    """Load tokenizer using fast tokenizers lib (no HF transformers import).

    tokenizer.json is bundled in the repo next to this module (it's arch-
    agnostic — same T5Gemma vocab whether you're on sm_90, sm_100, sm_120
    etc.). Falls back to the legacy per-arch location if the bundled file
    is somehow missing (e.g. older checkouts).
    """
    bundled = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    legacy  = os.path.join(ARCH_DIR, "t5gemma", "tokenizer.json")
    path = bundled if os.path.exists(bundled) else legacy
    return FastTokenizer(path)


class DistributionShift:
    """FluxDistributionShift — schedule warp used by SA3 models (rf_denoiser).

    Without this warp the model is conditioned on the wrong t at each step and
    final latent magnitudes blow up ~60×. Defaults match HF SA3 model configs
    (all variants share base_shift=0.5, max_shift=1.15, min/max_length=256/4096).
    """
    def __init__(self, base_shift=0.5, max_shift=1.15, min_length=256, max_length=4096):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.min_length = min_length
        self.max_length = max_length

    def shift(self, t, seq_len):
        sl = min(max(int(seq_len), self.min_length), self.max_length)
        mu = -(self.base_shift + (self.max_shift - self.base_shift) *
               (sl - self.min_length) / (self.max_length - self.min_length))
        return 1 - math.exp(mu) / (math.exp(mu) + (1.0 / (1.0 - t) - 1.0))


def load():
    """Load the runtime pieces needed for TRT inference: tokenizer and distribution
    shift. Neither requires the full PyTorch stable_audio_tools model.
    """
    return {"tokenizer": load_tokenizer(), "dist_shift": DistributionShift()}


if __name__ == "__main__":
    state = load()
    print(f"Loaded: {list(state.keys())}")
