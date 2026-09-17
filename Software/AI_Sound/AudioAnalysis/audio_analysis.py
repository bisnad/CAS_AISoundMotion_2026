"""
Audio Anaysis Application
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

from PyQt5 import QtWidgets

"""
Audio Source Settings
"""

AUDIO_MODE = "file" # choose one of: "file" or "mic"
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
    "n_fft_seconds": 2048 / 22050.0,
    "hop_seconds": 512 / 22050.0,
    "frame_seconds": 2048 / 22050.0,
    "heavy_window_seconds": 3.0,
    "heavy_update_every": 10,
    "enabled_defaults": {
        "rms": True,
        "zero_crossing_rate": False,
        "chroma_stft": False,
        "tonnetz": False,
        "mel_spectrogram": False,
        "mfcc": False,
        "spectral_centroid": False,
        "spectral_bandwidth": False,
        "spectral_contrast": False,
        "spectral_flatness": False,
        "spectral_rolloff": False,
        "poly_features": False,
        "onset_strength": False,
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

audio_gui.config["pipeline"] = pipeline
audio_gui.config["sender"] = osc_sender
audio_gui.config["receiver"] = audio_source

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui(audio_gui.config)

"""
Start Application
"""

gui.show()
app.exec_()
