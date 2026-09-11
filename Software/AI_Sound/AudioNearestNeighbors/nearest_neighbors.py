"""
Things that are unclear at the moment:

- If the search finishes (model.is_done() becomes True) while the GUI's
  generation thread is still running, the thread simply stops calling
  step() and idles until "stop" is pressed - playback continues normally
  through whatever was generated. This matches the original script's
  behavior of producing one complete, finite sequence per run (there is no
  "infinite" generation mode).
- Changing the selected audio features while a generation is in progress
  resets the search from scratch (matching how changing the clustering
  feature resets clustering in the AudioClustering tool) - partially
  generated audio from before the change is discarded, and both the
  generation loop and playback are stopped and restarted automatically by
  the GUI.

Revisions vs. the original nearest_neighbors.py:

1. Uses analysis.py from the AudioAnalysis tool.
2. FIXED TWO CORRECTNESS BUGS in the original greedy nearest-neighbor
   search: a crash on the final iteration, and silent index corruption
   during the run caused by not remapping the "current" excerpt index
   after removing an excerpt from the pool (see nn_model.py's module
   docstring for full details and how this was verified).
3. Per-dimension feature normalization (StandardScaler) instead of a
   single global mean/std across the whole feature matrix, matching the
   AudioClustering tool's normalization fix.
4. REAL-TIME CAPABLE: the search is now stepped incrementally
   (nn_model.NearestNeighborModel.step()) rather than run to completion in
   one blocking loop, and nn_synthesis.NNSynthesis plays back the growing
   output waveform live via a sounddevice.OutputStream callback, so the
   user hears the generated recording build up as it happens instead of
   only being able to listen after the whole (previously crash-prone)
   offline run finished.
5. GUI (previously nonexistent - the original was a plain script) now
   provides: audio folder selection (background-threaded, like
   AudioClustering's load-file workflow), a feature checklist, start/stop
   for generation+playback, a progress bar, and a "save audio file..."
   button that writes everything generated so far to disk at any point -
   the user does not have to wait for the full search to finish before
   saving, and can save partial results.
"""


"""
imports
"""

import sys

import analysis as aa
import nn_model
import nn_synthesis
import audio_gui

"""
Settings
"""

AUDIO_FOLDER_PATH = "data/audio/"
AUDIO_FILE_EXTENSIONS = ["wav", "aiff", "aif"]
AUDIO_SAMPLE_RATE = 48000
AUDIO_EXCERPT_SECONDS = 5.0
AUDIO_EXCERPT_OFFSET_SECONDS = 2.5
AUDIO_FEATURE_NAMES = ["root mean square", "mfcc"]
OUTPUT_DEVICE = None
STEP_INTERVAL_SECONDS = 0.05   # how often (in seconds) the background thread calls model.step()

"""
Create Model

NOTE: unlike AudioClustering (which loads and analyzes a file immediately
at startup), the excerpt pool here starts EMPTY - the user loads an audio
folder from the GUI ("load audio folder..."), since scanning and analyzing
a whole folder of files can take much longer than analyzing a single file
and should not block application startup.
"""

nn_model.config = {
    "audio_folder_path": AUDIO_FOLDER_PATH,
    "audio_file_extensions": AUDIO_FILE_EXTENSIONS,
    "audio_sample_rate": AUDIO_SAMPLE_RATE,
    "audio_excerpt_seconds": AUDIO_EXCERPT_SECONDS,
    "audio_excerpt_offset_seconds": AUDIO_EXCERPT_OFFSET_SECONDS,
    "audio_feature_names": AUDIO_FEATURE_NAMES,
}

model = nn_model.NearestNeighborModel(nn_model.config)

"""
Setup Synthesis
"""

nn_synthesis.config = {
    "model": model,
    "output_device": OUTPUT_DEVICE,
}

synthesis = nn_synthesis.NNSynthesis(nn_synthesis.config)

"""
GUI
"""

from PyQt5 import QtWidgets

audio_gui.config["model"] = model
audio_gui.config["synthesis"] = synthesis
audio_gui.config["step_interval_seconds"] = STEP_INTERVAL_SECONDS

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui(audio_gui.config)


def closeEvent():
    gui.stop()
    QtWidgets.QApplication.quit()


app.lastWindowClosed.connect(closeEvent)

"""
Start Application

NOTE: generation and playback are started via the GUI's start button, not
automatically here - matching the AudioAnalysis/AudioClustering tools'
convention that start/stop buttons are the single source of truth for
audio transport state. The user must also load an audio folder before
pressing start, since the excerpt pool is empty at startup.
"""

gui.show()
app.exec_()
