import sys, types, time, os, torch
import audiotools as at
from contextlib import nullcontext
import model_cache  # Import the cache

# Check if interface is preloaded
if not model_cache.has_interface():
    print("Error: No preloaded interface found. Run preload_model.py first.")
    sys.exit(1)

interface, device = model_cache.get_interface()
print(f"Using preloaded interface on device: {device}")

if len(sys.argv) != 2:
    print("Usage: python process_audio.py <audio_file_path>")
    sys.exit(1)

audio_path = sys.argv[1]

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

signal = at.AudioSignal(audio_path)

if signal.num_channels > 1:
    signal.audio_data = signal.audio_data[:, 0:1, :]
signal.sample_rate = 44100

amp_ctx = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()

with torch.no_grad(), amp_ctx:
    codes = interface.encode(signal).to(device, dtype=torch.long)
    
    mask = interface.build_mask(
        codes, signal,
        periodic_prompt=7,
        upper_codebook_mask=3,
    ).to(device, dtype=torch.long)

    output_tokens = interface.vamp(
        codes, mask,
        return_mask=False,
        temperature=1.0,
        typical_filtering=True,
    )

    output_signal = interface.decode(output_tokens)
    output_signal.audio_data = output_signal.audio_data[:, 0:1, :]

if hasattr(output_signal, "audio_data"):
    if isinstance(output_signal.audio_data, (list, tuple)):
        output_signal.audio_data = [ch.cpu() for ch in output_signal.audio_data]
    else:
        output_signal.audio_data = output_signal.audio_data.cpu()

out_path = "output.wav"
output_signal.write(out_path)
print(f"Output saved: {out_path}")
