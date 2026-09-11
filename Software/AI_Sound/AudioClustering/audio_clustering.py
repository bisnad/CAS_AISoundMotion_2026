"""
Things that are unclear at the moment:

- chroma_cqt / chroma_cens / chroma_vqt / tempo / tempogram_ratio remain
  commented out by default (as in the original script) since they are the
  slowest to compute across potentially thousands of excerpts; tempogram
  is kept enabled (with an explicit, short win_length - see analysis.py)
  since the original script already computed it.

Revisions vs. the original audio_clustering.py:

1. Uses analysis.py from the AudioAnalysis tool (superset of the original -
   same functions plus zero_crossing_rate, tonnetz, poly_features,
   onset_strength, and safer chroma_cqt/tempogram parameters).
2. Uses audio_receiver.AudioReceiver (from AudioAnalysis) to load audio
   files via its get_loaded_samples() accessor - see
   audio_receiver_patch_note.md for the one small addition needed in
   audio_receiver.py - instead of a bare librosa.load() call, so
   file-loading behavior (sample-rate handling, mono-mixing) is consistent
   across both tools.
3. Real-time playback goes through audio_synthesis.AudioSynthesis, which
   uses sounddevice (matching audio_receiver.py) instead of pyaudio.
4. audio_model.Clustering now z-score normalizes each feature before
   k-means and exposes cluster sizes/feature names for the GUI, plus
   set_data() for swapping in a newly loaded file's excerpts/features.
5. audio_gui.AudioGui is a real control panel (feature selection, cluster
   method/count, normalize toggle, device selection, cluster list,
   start/stop, and now a "load audio file..." button).
6. NEW: excerpt extraction + feature computation + clustering is now
   wrapped in build_clustering_data(), callable both at startup and again
   whenever a new file is loaded from the GUI. Because this can take
   several seconds to tens of seconds (feature extraction across
   potentially thousands of excerpts), the GUI runs it on a background
   QThread (see audio_gui.py's _LoadFileWorker) so the window does not
   freeze, and disables playback/controls with a status message while
   processing.
"""


"""
imports
"""

import sys
import numpy as np

import analysis as aa
import audio_receiver
import audio_model
import audio_synthesis
import audio_gui
import audio_control

"""
Audio Settings
"""

AUDIO_FILE_PATH = "E:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
AUDIO_SAMPLE_RATE = 48000       # fallback only - the file's own sample rate always wins
AUDIO_EXCERPT_LENGTH_MS = 100   # excerpt length, in milliseconds
AUDIO_EXCERPT_OFFSET_MS = 90    # hop between excerpt starts, in milliseconds
AUDIO_ANALYSIS_SECONDS = 60.0   # how much of a loaded file to analyze/cluster
OUTPUT_DEVICE = None            # None = default output device


def load_waveform(file_path, fallback_sample_rate, analysis_seconds):
    """
    Loads a file via audio_receiver.AudioReceiver (consistent sample-rate
    handling / mono-mixing with the rest of these tools) and returns
    (waveform, sample_rate), trimmed to at most analysis_seconds.
    """
    audio_receiver.config["mode"] = "file"
    audio_receiver.config["file_path"] = file_path
    audio_receiver.config["loop"] = False
    audio_receiver.config["sample_rate"] = fallback_sample_rate
    audio_receiver.config["channels"] = 1

    loader = audio_receiver.AudioReceiver(audio_receiver.config)
    sample_rate = loader.get_sample_rate()

    full_waveform = loader.get_loaded_samples()
    analysis_samples = min(int(analysis_seconds * sample_rate), full_waveform.shape[0])
    waveform = full_waveform[:analysis_samples].copy()

    return waveform, sample_rate


def build_excerpts(waveform, sample_rate, excerpt_length_ms, excerpt_offset_ms):
    """
    Slices a waveform into overlapping excerpts of excerpt_length_ms,
    starting every excerpt_offset_ms.
    """
    excerpt_length_sc = int(excerpt_length_ms / 1000 * sample_rate)
    excerpt_offset_sc = int(excerpt_offset_ms / 1000 * sample_rate)

    excerpts = []
    for sI in range(0, waveform.shape[0] - excerpt_length_sc, excerpt_offset_sc):
        excerpts.append(waveform[sI:sI + excerpt_length_sc])

    return np.stack(excerpts, axis=0)


def build_features(audio_excerpts, sample_rate):
    """
    Computes the full audio_features dict for a set of excerpts. Kept as
    its own function so it can be called identically at startup and after
    loading a new file.
    """
    audio_features = {}
    audio_features["waveform"] = audio_excerpts
    audio_features["root mean square"] = aa.rms(audio_excerpts)
    audio_features["chroma stft"] = aa.chroma_stft(audio_excerpts, sample_rate)
    # chroma_cqt / chroma_cens / chroma_vqt are slow across many excerpts -
    # disabled by default, same as the original script.
    # audio_features["chroma cqt"] = aa.chroma_cqt(audio_excerpts, sample_rate)
    # audio_features["chroma cens"] = aa.chroma_cens(audio_excerpts, sample_rate)
    # audio_features["chroma vqt"] = aa.chroma_vqt(audio_excerpts, sample_rate)
    audio_features["mel spectrogram"] = aa.mel_spectrogram(audio_excerpts, sample_rate)
    audio_features["mfcc"] = aa.mfcc(audio_excerpts, sample_rate)
    audio_features["spectral centroid"] = aa.spectral_centroid(audio_excerpts, sample_rate)
    audio_features["spectral bandwidth"] = aa.spectral_bandwidth(audio_excerpts, sample_rate)
    audio_features["spectral contrast"] = aa.spectral_contrast(audio_excerpts, sample_rate)
    audio_features["spectral flatness"] = aa.spectral_flatness(audio_excerpts)
    audio_features["spectral rolloff"] = aa.spectral_rolloff(audio_excerpts, sample_rate)
    audio_features["zero crossing rate"] = aa.zero_crossing_rate(audio_excerpts)
    # tempo / tempogram ratio disabled by default (slow, and tempo needs
    # more context than a 100ms excerpt can meaningfully provide);
    # tempogram kept on (short, explicit win_length safe for 100ms
    # excerpts) to match the original script's enabled set.
    # audio_features["tempo"] = aa.tempo(audio_excerpts, sample_rate)
    audio_features["tempogram"] = aa.tempogram(audio_excerpts, sample_rate, win_length=16)
    # audio_features["tempogram ratio"] = aa.tempogram_ratio(audio_excerpts, sample_rate)

    return audio_features


def build_clustering_data(file_path, excerpt_length_ms=AUDIO_EXCERPT_LENGTH_MS,
                           excerpt_offset_ms=AUDIO_EXCERPT_OFFSET_MS,
                           analysis_seconds=AUDIO_ANALYSIS_SECONDS,
                           fallback_sample_rate=AUDIO_SAMPLE_RATE):
    """
    Full pipeline: load file -> slice excerpts -> compute features.
    Returns (audio_excerpts, audio_features, sample_rate). This is the
    function audio_gui.py's background load-file worker calls.
    """
    waveform, sample_rate = load_waveform(file_path, fallback_sample_rate, analysis_seconds)
    audio_excerpts = build_excerpts(waveform, sample_rate, excerpt_length_ms, excerpt_offset_ms)
    audio_features = build_features(audio_excerpts, sample_rate)

    return audio_excerpts, audio_features, sample_rate


"""
Initial Load
"""

audio_excerpts, audio_features, audio_sample_rate = build_clustering_data(AUDIO_FILE_PATH)

print("[audio_clustering] loaded '{}' at {} Hz, {} excerpts of {}ms (offset {}ms)".format(
    AUDIO_FILE_PATH, audio_sample_rate, audio_excerpts.shape[0], AUDIO_EXCERPT_LENGTH_MS, AUDIO_EXCERPT_OFFSET_MS
))

"""
Create Clustering Model
"""

audio_model.config = {
    "audio_excerpts": audio_excerpts,
    "audio_features": audio_features,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170,
    "normalize": True,
}

clustering = audio_model.createModel(audio_model.config)

"""
Setup Audio Synthesis
"""

audio_synthesis.config = {
    "model": clustering,
    "audio_excerpts": audio_excerpts,
    "audio_sample_rate": audio_sample_rate,
    "audio_excerpt_length": AUDIO_EXCERPT_LENGTH_MS,
    "audio_excerpt_offset": AUDIO_EXCERPT_OFFSET_MS,
    "output_device": OUTPUT_DEVICE,
}

synthesis = audio_synthesis.AudioSynthesis(audio_synthesis.config)
synthesis.setClusterLabel(1)

"""
OSC Control
"""

audio_control.config["synthesis"] = synthesis
audio_control.config["model"] = clustering
audio_control.config["ip"] = "127.0.0.1"
audio_control.config["port"] = 9002

osc_control = audio_control.AudioControl(audio_control.config)

"""
GUI
"""

from PyQt5 import QtWidgets

audio_gui.config["model"] = clustering
audio_gui.config["synthesis"] = synthesis
audio_gui.config["control"] = osc_control
# passed through so the GUI's "load audio file..." button can run the same
# extraction/feature pipeline used at startup, on a background thread
audio_gui.config["build_clustering_data"] = build_clustering_data
audio_gui.config["excerpt_length_ms"] = AUDIO_EXCERPT_LENGTH_MS
audio_gui.config["excerpt_offset_ms"] = AUDIO_EXCERPT_OFFSET_MS
audio_gui.config["analysis_seconds"] = AUDIO_ANALYSIS_SECONDS

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui(audio_gui.config)


def closeEvent():
    synthesis.stop()
    osc_control.stop()
    QtWidgets.QApplication.quit()


app.lastWindowClosed.connect(closeEvent)

"""
Start Application

NOTE: playback is started via the GUI's start button
(AudioGui.start() -> synthesis.start()), matching the AudioAnalysis tool's
convention that start/stop buttons are the single source of truth for
audio transport state - it is intentionally not auto-started here.
"""

osc_control.start()
gui.show()
app.exec_()
