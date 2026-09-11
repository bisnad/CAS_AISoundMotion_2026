"""
stable_audio_3_core.py

Everything genuinely specific to Stable Audio 3 -- model loading, audio
I/O, and the four generation techniques (text-to-audio, init_audio,
inpainting/continuation, text-embedding interpolation) -- with ZERO
dependency on any GUI toolkit. This module only imports torch, torchaudio,
and stable_audio_3, so it can be read, tested, and reused exactly like
hello_stable_audio_3.py, independent of whichever GUI (or no GUI at all)
calls into it.

Every GUI tool built on top of Stable Audio 3 in this project imports
from this module rather than duplicating this logic -- the GUI layer's
only job is building widgets, wiring signals, and calling these functions
on background threads.

Sections:
  - Model loading and variant resolution
  - Audio I/O helpers
  - Text-to-audio / init_audio generation
  - Inpainting / continuation generation
  - Text-embedding interpolation generation

CORRECTNESS NOTES carried forward from debugging across this project
(kept here, not scattered across GUI files, since this is the one place
each technique's real logic lives):
  - init_audio / inpaint_audio require the source waveform resampled to
    the model's sample rate and channel-coerced to match io_channels --
    load_wav_matching_model() does this once, correctly, for every caller.
  - Post-trained checkpoints ("small-music", "small-sfx", "medium") do
    not rely on classifier-free guidance at inference; cfg_scale=7.0 was
    confirmed to cause severe clipping specifically on the inpainting
    path for both "small" and "medium". "-base" checkpoints are the
    paper's own validated CFG regime (50 steps, cfg_scale=7). cfg_scale
    is therefore always OPTIONAL (None by default) in every generation
    function below -- callers decide whether to pass a value.
  - Text-embedding interpolation reaches past the public generate(prompt=)
    API by patching conditioner.forward (NOT conditioner.__call__, since
    Python's implicit dispatch looks up __call__ on the type, not the
    instance) and always restores the original forward in a finally
    block, so a failure mid-generation can never leave the model
    permanently patched.
  - torch._dynamo.disable wraps the final generate() call inside
    interpolation specifically, since monkeypatching forward() at runtime
    invalidates torch.compile's guard assumptions for "medium" (whose
    SAME-L decoder uses torch.compile), causing repeated failed
    recompilation attempts and log spam without this.
"""

import torch
import torchaudio
import torch._dynamo

from stable_audio_3 import StableAudioModel


# ============================================================================
# Model loading and variant resolution
# ============================================================================

def detect_default_device():
    """Picks the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_variant(size, variant, small_domain):
    """Maps three independent choices to the actual from_pretrained() ID.

    size: "small" or "medium"
    variant: "post-trained" (8-step ping-pong, CFG not needed/validated)
             or "base" (~50-step Euler, CFG meaningful)
    small_domain: "music" or "sfx" -- only used when size == "small",
                  since medium handles both domains in one checkpoint.
    """
    if size == "medium":
        return "medium-base" if variant == "base" else "medium"
    base_name = "small-music" if small_domain == "music" else "small-sfx"
    return f"{base_name}-base" if variant == "base" else base_name


def load_model(model_variant, device=None):
    """Loads a Stable Audio 3 checkpoint and returns everything callers
    need: the model, its inner text/duration conditioner (needed for
    interpolation), the sample rate, and the channel count."""
    if device is None:
        device = detect_default_device()
    model = StableAudioModel.from_pretrained(model_variant, device=device)
    conditioner = model.model.conditioner
    sample_rate = model.model_config["sample_rate"]
    io_channels = model.model_config.get("io_channels", 2)
    return model, conditioner, sample_rate, io_channels, device


# ============================================================================
# Audio I/O helpers
# ============================================================================

def load_wav_matching_model(path, sample_rate, io_channels):
    """Loads a .wav file, resampling to the model's sample rate and
    coercing channel count to match io_channels. Required before using
    any loaded audio as init_audio or inpaint_audio -- skipping this
    causes the model to misinterpret the audio's duration/pitch/channel
    layout."""
    audio, orig_sr = torchaudio.load(path)
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)
        audio = resampler(audio)
    if audio.shape[0] == 1 and io_channels == 2:
        audio = audio.repeat(io_channels, 1)
    elif audio.shape[0] > io_channels:
        audio = audio[:io_channels]
    return audio


def save_wav(path, audio_tensor, sample_rate):
    """Thin wrapper kept for symmetry with load_wav_matching_model --
    always saves with the MODEL's sample rate, never a source file's."""
    torchaudio.save(path, audio_tensor.detach().cpu(), sample_rate)


# ============================================================================
# Text-to-audio / init_audio generation
# ============================================================================

def generate_text_to_audio(
    model, prompt, duration, steps, seed,
    negative_prompt=None, cfg_scale=None,
    init_audio_tensor=None, init_audio_sample_rate=None, init_noise_level=None,
):
    """Plain text-to-audio, optionally starting from an existing waveform
    (init_audio) instead of pure noise. cfg_scale is optional and only
    forwarded if explicitly provided -- see module docstring."""
    kwargs = dict(prompt=prompt, duration=duration, steps=steps, seed=seed)
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if cfg_scale is not None:
        kwargs["cfg_scale"] = cfg_scale
    if init_audio_tensor is not None:
        kwargs["init_audio"] = (init_audio_sample_rate, init_audio_tensor)
        kwargs["init_noise_level"] = init_noise_level if init_noise_level is not None else 0.4

    with torch.no_grad():
        audio = model.generate(**kwargs)
    return audio[0].detach().cpu()


# ============================================================================
# Inpainting / continuation generation
# ============================================================================

def generate_inpaint_or_continue(
    model, device, source_audio_tensor, source_sample_rate, regions,
    prompt, duration, steps, seed,
    negative_prompt=None, cfg_scale=None,
):
    """Runs one or more inpainting passes over `source_audio_tensor`.

    regions: list of (mask_start_sec, mask_end_sec) tuples, applied
    SEQUENTIALLY -- each pass's output becomes the next pass's source.
    This matches the only demonstrated generate() signature (a single
    inpaint_mask_start_seconds/inpaint_mask_end_seconds pair per call);
    if the installed API accepts multiple mask spans in one call, this
    could be simplified to a single joint pass instead.

    Continuation is the same mechanism with a single region:
    [kept_from_seconds, new_total_duration].
    """
    base_kwargs = dict(prompt=prompt, duration=duration, steps=steps, seed=seed)
    if negative_prompt:
        base_kwargs["negative_prompt"] = negative_prompt
    if cfg_scale is not None:
        base_kwargs["cfg_scale"] = cfg_scale

    current_audio = source_audio_tensor
    for mask_start, mask_end in regions:
        kwargs = dict(base_kwargs)
        kwargs["inpaint_audio"] = (source_sample_rate, current_audio.to(device))
        kwargs["inpaint_mask_start_seconds"] = mask_start
        kwargs["inpaint_mask_end_seconds"] = mask_end

        with torch.no_grad():
            result = model.generate(**kwargs)
        current_audio = result[0].detach()

    return current_audio.cpu()


# ============================================================================
# Text-embedding interpolation generation
# ============================================================================

def encode_prompt_to_embedding(conditioner, prompt_text, duration_sec, device):
    """Encodes a single prompt via conditioner.forward() directly -- the
    real entry point; nn.Module.__call__ delegates to this internally."""
    batch_metadata = [{
        "prompt": prompt_text,
        "seconds_start": 0,
        "seconds_total": duration_sec,
    }]
    with torch.no_grad():
        conditioning_tensors = conditioner.forward(batch_metadata, device)
    prompt_emb, prompt_mask = conditioning_tensors["prompt"]
    return prompt_emb, prompt_mask


def mask_aware_interpolate(emb_a, mask_a, emb_b, mask_b, interp_factor):
    """Blends two prompt embeddings only where BOTH are real content;
    positions real in only one embedding are kept untouched rather than
    blended against unrelated learned padding. Returns the interpolated
    embedding, mask, and a stats dict for diagnostics."""
    mask_a_bool = mask_a.bool()
    mask_b_bool = mask_b.bool()
    both_real = mask_a_bool & mask_b_bool
    only_a = mask_a_bool & (~mask_b_bool)
    only_b = mask_b_bool & (~mask_a_bool)
    neither = (~mask_a_bool) & (~mask_b_bool)

    emb_interp = torch.zeros_like(emb_a)
    emb_interp = torch.where(
        both_real.unsqueeze(-1),
        emb_a * (1.0 - interp_factor) + emb_b * interp_factor,
        emb_interp,
    )
    emb_interp = torch.where(only_a.unsqueeze(-1), emb_a, emb_interp)
    emb_interp = torch.where(only_b.unsqueeze(-1), emb_b, emb_interp)
    emb_interp = torch.where(neither.unsqueeze(-1), emb_a, emb_interp)

    mask_interp = (mask_a_bool | mask_b_bool).to(mask_a.dtype)

    stats = {
        "blended": int(both_real.sum().item()),
        "only_a": int(only_a.sum().item()),
        "only_b": int(only_b.sum().item()),
        "padding": int(neither.sum().item()),
    }
    return emb_interp, mask_interp, stats


@torch._dynamo.disable
def _generate_with_patched_conditioner(model, gen_kwargs):
    """Isolated so torch._dynamo never attempts to trace/compile through
    this call. Monkeypatching conditioner.forward at runtime invalidates
    torch.compile's guard assumptions for any graph region that included
    it -- relevant specifically for "medium", whose SAME-L decoder uses
    torch.compile (per the paper); without this, dynamo repeatedly
    attempts and fails to recompile on every interpolated generation."""
    with torch.no_grad():
        return model.generate(**gen_kwargs)


def generate_interpolated(
    model, conditioner, device, prompt_a, prompt_b, interp_factor,
    duration, steps, seed,
    negative_prompt=None, cfg_scale=None,
):
    """Runs the full interpolation pipeline: encode A, encode B,
    mask-aware blend, encode seconds_total, monkeypatch
    conditioner.forward, generate, and ALWAYS restore the original
    forward in a finally block -- regardless of success or failure.
    Leaving the model patched would cause every SUBSEQUENT generate()
    call on this model to silently reuse this stale interpolated
    conditioning instead of properly encoding whatever prompt is
    actually requested.

    Returns (audio_tensor, stats_dict).
    """
    original_forward = conditioner.forward
    try:
        emb_a, mask_a = encode_prompt_to_embedding(conditioner, prompt_a, duration, device)
        emb_b, mask_b = encode_prompt_to_embedding(conditioner, prompt_b, duration, device)

        if emb_a.shape != emb_b.shape:
            raise ValueError(
                f"Prompt A and B produced different embedding shapes "
                f"({emb_a.shape} vs {emb_b.shape}) -- interpolation requires matching shapes."
            )

        emb_interp, mask_interp, stats = mask_aware_interpolate(emb_a, mask_a, emb_b, mask_b, interp_factor)

        batch_metadata_for_duration = [{
            "prompt": prompt_a,
            "seconds_start": 0,
            "seconds_total": duration,
        }]
        with torch.no_grad():
            full_conditioning_reference = original_forward(batch_metadata_for_duration, device)
        seconds_total_tensors = full_conditioning_reference["seconds_total"]

        conditioning_tensors_interp = {
            "prompt": (emb_interp, mask_interp),
            "seconds_total": seconds_total_tensors,
        }

        def patched_forward(batch_metadata, device_arg):
            return conditioning_tensors_interp

        conditioner.forward = patched_forward

        gen_kwargs = dict(
            prompt="unused -- forward() is patched, this string is never encoded",
            duration=duration,
            steps=steps,
            seed=seed,
        )
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt
        if cfg_scale is not None:
            gen_kwargs["cfg_scale"] = cfg_scale

        gen_audio = _generate_with_patched_conditioner(model, gen_kwargs)
        return gen_audio[0].detach().cpu(), stats

    finally:
        conditioner.forward = original_forward
