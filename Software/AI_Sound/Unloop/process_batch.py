import sys, types, time, os, torch, unloop
import audiotools as at
from contextlib import nullcontext

# ... [all the setup code from run_from_term.py] ...

# Load model once
interface = unloop.interface.Interface.default()
interface.float()
if hasattr(interface, "to"):
    interface.to(device)
interface._preprocess = _peak_preprocess.__get__(interface, type(interface))

# Process multiple audio files
audio_files = sys.argv[1:]  # Accept multiple files

for audio_path in audio_files:
    if not os.path.exists(audio_path):
        print(f"Skipping {audio_path} - file not found")
        continue
    
    print(f"Processing {audio_path}...")
    
    signal = at.AudioSignal(audio_path)
    # ... [processing code] ...
    
    out_path = f"output_{os.path.basename(audio_path)}"
    output_signal.write(out_path)
    print(f"Saved: {out_path}")
