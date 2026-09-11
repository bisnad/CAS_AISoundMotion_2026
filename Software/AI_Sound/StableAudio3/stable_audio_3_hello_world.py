# ----------------------------------------------------------------------------
# minimal, code-only tour of four core Stable Audio 3
# capabilities: text 2 audio, inpainting, continuation, prompt interpolation
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

import torch
import torchaudio

from stable_audio_3 import StableAudioModel

# ----------------------------------------------------------------------------
# Select Compute Device
# ----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Load Pre-trained Stable Audio 3 Model
# ----------------------------------------------------------------------------

model = StableAudioModel.from_pretrained("small-sfx", device=device)

# ----------------------------------------------------------------------------
# Check Model Configuration
# ----------------------------------------------------------------------------

sample_rate = model.model_config["sample_rate"] 
io_channels = model.model_config.get("io_channels", 2)

# ----------------------------------------------------------------------------
# Load Audio Helper Function
# ----------------------------------------------------------------------------

def load_audio(path):
    """Loads a .wav file, resampling to the model's sample rate and
    coercing channel count to match io_channels -- required before using
    any loaded audio as init_audio/inpaint_audio, so the model interprets
    its duration/pitch/channel layout correctly."""
    audio, orig_sr = torchaudio.load(path)
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)
        audio = resampler(audio)
    if audio.shape[0] == 1 and io_channels == 2:
        audio = audio.repeat(io_channels, 1)
    elif audio.shape[0] > io_channels:
        audio = audio[:io_channels]
    return audio

# ----------------------------------------------------------------------------
# SECTION 1: Basic Text to Audio
#
# The principle: give the model a text prompt describing a sound, and it
# generates a new audio clip matching that description from scratch. This
# is the simplest possible use of the model -- one prompt in, one audio
# file out.
# ----------------------------------------------------------------------------

prompt = "footsteps on gravel"
negative_prompt = "low-quality, distorted, noisy"
duration = 5.0
steps = 8
seed = 42

text_to_audio_result = model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    duration=duration,
    steps=steps,
    seed=seed,
)

torchaudio.save("stable_audio_3_text_to_audio.wav", text_to_audio_result[0].detach().cpu(), sample_rate)


# ----------------------------------------------------------------------------
# SECTION 2: Text to Audio with Initial Audio
#
# The principle: instead of starting purely from random noise, start from
# an EXISTING audio clip that has been partially noised (controlled by
# init_noise_level), then denoise it under a text prompt. The lower
# init_noise_level is, the more the result resembles the original audio;
# the higher it is, the closer the result gets to plain text-to-audio.
# ----------------------------------------------------------------------------

audio_file = "input/hitting_metal.wav"
initial_noise_level = 0.4
prompt = "footsteps on gravel"
negative_prompt = "low-quality, distorted, noisy"
duration = 5.0
steps = 8
seed = 42

init_audio_waveform = load_audio(audio_file)

initial_audio_result = model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    duration=duration,
    steps=steps,
    init_audio=(sample_rate, init_audio_waveform),
    init_noise_level=initial_noise_level,
    seed=seed,
)

torchaudio.save("stable_audio_3_initial_audio.wav", initial_audio_result[0].detach().cpu(), sample_rate)


# ----------------------------------------------------------------------------
# SECTION 3: INPAINTING
#
# THE PRINCIPLE: give the model an existing audio clip plus a text prompt,
# and mark a TIME SPAN inside that clip to be regenerated. Everything
# outside that span is kept byte-for-byte identical; everything inside it
# is replaced with new audio guided by the prompt. Think of it like
# "select a region in a waveform editor and re-render just that region."
# ----------------------------------------------------------------------------

audio_file = "input/hitting_metal.wav"
inpaint_mask_start_sec = 1.0
inpaint_mask_end_sec = 4.0
prompt = "footsteps on gravel"
negative_prompt = "low-quality, distorted, noisy"
steps = 8
seed = 42

inpaint_source_waveform = load_audio(audio_file)
inpaint_source_duration = inpaint_source_waveform.shape[1] / sample_rate

if inpaint_mask_end_sec > inpaint_source_duration:
    raise ValueError(
        f"inpaint_mask_end_sec ({inpaint_mask_end_sec}s) exceeds the source "
        f"clip's actual duration ({inpaint_source_duration:.2f}s)."
    )

inpainting_result = model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    duration=inpaint_source_duration,
    steps=steps,
    inpaint_audio=(sample_rate, inpaint_source_waveform),
    inpaint_mask_start_seconds=inpaint_mask_start_sec,
    inpaint_mask_end_seconds=inpaint_mask_end_sec,
    seed=seed,
)

torchaudio.save("stable_audio_3_inpainting.wav", inpainting_result[0].detach().cpu(), sample_rate)


# ----------------------------------------------------------------------------
# SECTION 4: CONTINUATION
#
# THE PRINCIPLE: give the model an existing audio clip plus a text prompt,
# and ask it to extend the clip PAST its own ending, out to a longer total
# duration. This is a special case of inpainting where the masked
# (to-be-generated) region starts exactly where the source clip ends (or
# wherever you choose to keep audio up to) and runs to the new, longer
# duration -- everything before that point is kept, only the newly
# appended tail is generated.
# ----------------------------------------------------------------------------

audio_file = "input/hitting_metal.wav"
audio_continuation_start = 5.0
duration = 30.0
prompt = "footsteps on gravel"
negative_prompt = "low-quality, distorted, noisy"
steps = 8
seed = 42

continuation_source_waveform = load_audio(audio_file)
continuation_source_duration = continuation_source_waveform.shape[1] / sample_rate

if audio_continuation_start > continuation_source_duration:
    raise ValueError(
        f"audio_continuation_start ({audio_continuation_start}s) exceeds the source "
        f"clip's actual duration ({continuation_source_duration:.2f}s)."
    )
if duration <= continuation_source_duration:
    raise ValueError(
        f"duration ({duration}s) must be greater than the source clip's "
        f"actual duration ({continuation_source_duration:.2f}s), or there is nothing to continue."
    )

continuation_result = model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    duration=duration,
    steps=steps,
    inpaint_audio=(sample_rate, continuation_source_waveform),
    inpaint_mask_start_seconds=audio_continuation_start,
    inpaint_mask_end_seconds=duration,
    seed=seed,
)

torchaudio.save("stable_audio_3_continuation.wav", continuation_result[0].detach().cpu(), sample_rate)


# ----------------------------------------------------------------------------
# SECTION 5: TEXT EMBEDDING INTERPOLATION
#
# THE PRINCIPLE: instead of choosing one text prompt, blend the internal
# EMBEDDINGS of two prompts together (e.g. 50% "a violin sound", 50% "a
# frog sound") and generate audio from that blended embedding. This lets
# you explore the space "between" two ideas rather than only the two
# ideas themselves.
# ----------------------------------------------------------------------------

prompt_a = "a violin sound"
prompt_b = "a frog sound"
negative_prompt = "low-quality, distorted, noisy"
interpolation_factor = 0.5
duration = 8.0
steps = 8
seed = 42

conditioner = model.model.conditioner

def encode_prompt(prompt_text, duration_sec):
    batch = [{"prompt": prompt_text, "seconds_start": 0, "seconds_total": duration_sec}]
    with torch.no_grad():
        tensors = conditioner.forward(batch, device)
    return tensors["prompt"]  # (embedding, mask)

emb_a, mask_a = encode_prompt(prompt_a, duration)
emb_b, mask_b = encode_prompt(prompt_b, duration)

# Blend only where BOTH prompts have real (non-padding) content at a given
# position. Prompts of different lengths get padded to the same fixed
# length with a learned padding embedding -- blending real content against
# that unrelated padding (a naive position-by-position average) produces
# meaningless embeddings. Where only one prompt has real content, keep it
# untouched instead of blending it against padding.
mask_a_bool, mask_b_bool = mask_a.bool(), mask_b.bool()
both_real = mask_a_bool & mask_b_bool
only_b = mask_b_bool & ~mask_a_bool

emb_interp = torch.where(
    both_real.unsqueeze(-1), emb_a * (1 - interpolation_factor) + emb_b * interpolation_factor, emb_a
)
emb_interp = torch.where(only_b.unsqueeze(-1), emb_b, emb_interp)
mask_interp = (mask_a_bool | mask_b_bool).to(mask_a.dtype)

# get_conditioning_inputs() expects BOTH "prompt" and "seconds_total" in
# the conditioning dict (the model conditions on duration too) -- so our
# replacement must supply both keys, not just the one we're overriding.
seconds_total_tensors = conditioner.forward(
    [{"prompt": prompt_a, "seconds_start": 0, "seconds_total": duration}], device
)["seconds_total"]
conditioning_override = {"prompt": (emb_interp, mask_interp), "seconds_total": seconds_total_tensors}

# Patch .forward, NOT .__call__: Python looks up __call__ on the CLASS,
# not the instance, so assigning conditioner.__call__ = ... would silently
# do nothing. forward is a plain method, so an instance-level override
# works correctly here.
original_conditioner_forward = conditioner.forward
conditioner.forward = lambda batch, dev: conditioning_override

try:
    interpolation_result = model.generate(
        prompt="unused, forward() is patched",
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        seed=seed,
    )
finally:
    # ALWAYS restore, even if generate() raised -- otherwise every later
    # call on this model would silently keep reusing this stale blended
    # embedding instead of properly encoding whatever prompt is requested.
    conditioner.forward = original_conditioner_forward

torchaudio.save("stable_audio_3_interpolation.wav", interpolation_result[0].detach().cpu(), sample_rate)


