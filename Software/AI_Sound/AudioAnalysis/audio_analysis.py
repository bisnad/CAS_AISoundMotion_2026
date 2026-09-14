"""
Things that are unclear at the moment:

Heavy/windowed descriptors (chroma_cqt, chroma_cens, chroma_vqt, tempo,
tempogram, fourier_tempogram, tempogram_ratio) are computed on a longer
rolling window (heavy_window_seconds) at a reduced rate (every
heavy_update_every ticks). They are OFF by default (see "enabled_defaults"
below) since they were the main source of "pausing" at 48kHz - turn them on
selectively if needed.

PERFORMANCE: audio_pipeline.py computes a single shared STFT per tick and
reuses it across chroma_stft / mel_spectrogram / mfcc / spectral_centroid /
spectral_bandwidth / spectral_contrast / spectral_rolloff. Every descriptor
is gated by pipeline.enabled[name]; disabled descriptors are not computed
at all. audio_receiver.py's file playback uses a real-time
sounddevice.OutputStream CALLBACK backed by a small ring buffer, fully
decoupled from analysis/OSC/GUI timing.

GUI CONTROL SEMANTICS (audio_gui.py):
- The "start"/"stop" buttons now control BOTH the analysis/OSC update loop
  AND the audio source/transport (file playback or microphone capture) -
  there is no separate call needed to start the receiver.
- HIGHLIGHTING a descriptor in the list controls what is shown in the
  canvas. This is completely independent of its checkbox.
- The CHECKBOX next to each descriptor controls ONLY whether that
  descriptor is sent via OSC.
- Computation (pipeline.enabled) is the union of "checked for OSC" OR
  "currently highlighted for display" - so you can preview an unchecked
  (not-sent) descriptor by highlighting it, and it will be computed on
  demand for as long as it stays highlighted.

All analysis window sizes (n_fft, hop_length, frame_length, buffer size) are
specified in SECONDS everywhere in this tool, and converted to samples using
the audio source's actual sample rate. Loading a new file or switching to
the microphone changes the effective sample rate; the GUI already calls
pipeline.reconfigure() automatically whenever that happens.
"""


"""
imports
"""

import sys

import analysis as aa
import audio_receiver
import audio_sender
import audio_pipeline
import audio_gui

"""
Audio Source Settings
"""

# choose one of: "file" or "mic"
AUDIO_MODE = "file"
AUDIO_FILE_PATH = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"   # only used when AUDIO_MODE == "file"
INPUT_DEVICE = None                       # None = default input device, only used when AUDIO_MODE == "mic"
OUTPUT_DEVICE = None                      # None = default output device, used for file playback
AUDIO_SAMPLE_RATE = 48000                 # fallback / mic-mode rate; file mode uses the file's own rate

"""
Audio Receiver (source + transport/player)
"""

audio_receiver.config["mode"] = AUDIO_MODE
audio_receiver.config["file_path"] = AUDIO_FILE_PATH
audio_receiver.config["loop"] = True
audio_receiver.config["input_device"] = INPUT_DEVICE
audio_receiver.config["output_device"] = OUTPUT_DEVICE
audio_receiver.config["sample_rate"] = AUDIO_SAMPLE_RATE
audio_receiver.config["channels"] = 1
audio_receiver.config["buffer_seconds"] = 4.0             # rolling analysis buffer length, in seconds
audio_receiver.config["block_seconds"] = 1024 / float(AUDIO_SAMPLE_RATE)  # analysis-buffer refill granularity
audio_receiver.config["output_ring_seconds"] = 1.0        # playback ring buffer depth - increase if clicks persist

audio_source = audio_receiver.AudioReceiver(audio_receiver.config)

print("[audio_analysis] source sample rate: {} Hz, buffer: {:.2f}s ({} samples)".format(
    audio_source.get_sample_rate(), audio_source.get_buffer_seconds(), audio_source.buffer_size
))

"""
OSC Sender
"""

audio_sender.config["ip"] = "127.0.0.1"
audio_sender.config["port"] = 9009

osc_sender = audio_sender.OscSender(audio_sender.config)

"""
Data Pipeline
"""

audio_pipeline_config = {
    "fps": 30,
    "n_fft_seconds": 2048 / 22050.0,     # ~93ms analysis window, sample-rate independent
    "hop_seconds": 512 / 22050.0,        # ~23ms hop, sample-rate independent
    "frame_seconds": 2048 / 22050.0,     # per-tick frame length fed to cheap descriptors
    "heavy_window_seconds": 3.0,         # must be <= audio_receiver.config["buffer_seconds"]
    "heavy_update_every": 10,
    # recommended-for-48kHz defaults: cheap/shared-STFT descriptors on,
    # expensive ones (mfcc, mel_spectrogram, spectral_contrast,
    # poly_features, tonnetz, chroma_stft) and all heavy/windowed
    # descriptors off. Toggle any of these live from the GUI checklist, or
    # preview a disabled one by highlighting it in the list.
    "enabled_defaults": {
        "rms": True,
        "zero_crossing_rate": True,
        "chroma_stft": False,
        "tonnetz": False,
        "mel_spectrogram": False,
        "mfcc": False,
        "spectral_centroid": True,
        "spectral_bandwidth": True,
        "spectral_contrast": False,
        "spectral_flatness": True,
        "spectral_rolloff": True,
        "poly_features": False,
        "onset_strength": True,
        "chroma_cqt": False,
        "chroma_cens": False,
        "chroma_vqt": False,
        "tempo": False,
        "tempogram": False,
        "fourier_tempogram": False,
        "tempogram_ratio": False,
    },
}

pipeline = audio_pipeline.AudioPipeline(audio_source, audio_pipeline_config)

"""
GUI
"""

from PyQt5 import QtWidgets

audio_gui.config["pipeline"] = pipeline
audio_gui.config["sender"] = osc_sender
audio_gui.config["receiver"] = audio_source

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui(audio_gui.config)

"""
Start Application

NOTE: the audio source/transport is now started/stopped exclusively via the
GUI's start/stop buttons (AudioGui.start()/stop() call
self.receiver.start()/stop() directly) - it is intentionally NOT started
here, so there is a single source of truth for transport state.
"""

gui.show()
app.exec_()
