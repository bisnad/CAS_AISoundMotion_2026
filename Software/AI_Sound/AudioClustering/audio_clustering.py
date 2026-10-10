"""
audio_clustering.py - entry point of the GUI app.

Excerpt extraction and feature computation now live in clustering_pipeline.py;
the GUI lets the user change the excerpt length / overlap (ms) and choose and
combine audio features, re-running the pipeline on a background thread.
"""

import sys

import clustering_pipeline
import audio_model
import audio_synthesis
import audio_gui
import audio_control

"""
Audio Settings
"""

AUDIO_FILE_PATH = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
AUDIO_SAMPLE_RATE = 48000          # fallback only - the file's own sample rate always wins
AUDIO_EXCERPT_LENGTH_MS = 100      # initial excerpt length, in milliseconds (editable in the GUI)
AUDIO_EXCERPT_OVERLAP_MS = 10      # initial overlap between excerpts, in milliseconds (editable in the GUI)
AUDIO_EXCERPT_OFFSET_MS = AUDIO_EXCERPT_LENGTH_MS - AUDIO_EXCERPT_OVERLAP_MS
AUDIO_ANALYSIS_SECONDS = 60.0
AUDIO_FEATURES = ["root mean square", "mfcc"]   # initially selected features
OUTPUT_DEVICE = None

"""
Initial Load
"""

pipeline = clustering_pipeline.ClusteringPipeline(AUDIO_SAMPLE_RATE, AUDIO_ANALYSIS_SECONDS)
audio_excerpts, audio_features, audio_sample_rate = pipeline.prepare(
    AUDIO_FILE_PATH, AUDIO_EXCERPT_LENGTH_MS, AUDIO_EXCERPT_OFFSET_MS, AUDIO_FEATURES)

print("[audio_clustering] loaded '{}' at {} Hz, {} excerpts of {}ms (offset {}ms)".format(
    AUDIO_FILE_PATH, audio_sample_rate, audio_excerpts.shape[0], AUDIO_EXCERPT_LENGTH_MS, AUDIO_EXCERPT_OFFSET_MS))

"""
Clustering Model
"""

audio_model.config = {
    "audio_excerpts": audio_excerpts,
    "audio_features": audio_features,
    "selected_features": AUDIO_FEATURES,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170,
    "normalize": True,
    "equal_feature_weight": True,
    "feature_provider": pipeline.ensure_features,
}

clustering = audio_model.createModel(audio_model.config)

"""
Audio Synthesis
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
audio_gui.config["pipeline"] = pipeline
audio_gui.config["excerpt_length_ms"] = AUDIO_EXCERPT_LENGTH_MS
audio_gui.config["excerpt_offset_ms"] = AUDIO_EXCERPT_OFFSET_MS

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui(audio_gui.config)


def closeEvent():
    synthesis.stop()
    osc_control.stop()
    QtWidgets.QApplication.quit()


app.lastWindowClosed.connect(closeEvent)

"""
Start Application (playback is started with the GUI's start button)
"""

osc_control.start()
gui.show()
app.exec_()
