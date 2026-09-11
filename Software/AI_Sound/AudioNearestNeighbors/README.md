# Audio Nearest Neighbors

A real-time-observable audio recombination tool: it loads every audio file in a folder, slices them into short overlapping excerpts, computes audio descriptors for each excerpt using [librosa](https://librosa.org/), and builds a brand-new recording by repeatedly jumping to the closest-sounding excerpt still available — starting from the first excerpt, always moving to its nearest unused neighbor, and never revisiting an excerpt once it has been used. The result is a single continuous recording that resequences an entire folder of audio according to sonic similarity rather than its original order.

## Summary

This tool answers the question *"what does this whole folder of sounds become if you keep chaining together whichever unused excerpt sounds most similar to the one you just heard?"* Every audio file in a folder is sliced into short excerpts and turned into numeric descriptors. Starting from the very first excerpt, the tool greedily walks to the nearest still-available excerpt, cross-fades it into a growing output recording, removes the excerpt it just came from, and repeats — continuing until every excerpt in the entire folder has been consumed exactly once. Unlike the original offline version of this tool, generation happens incrementally and can be listened to live as it builds, and the growing recording can be saved to disk at any point, not only once the whole folder has been fully processed.

## Functionality

### Loading a folder and extracting excerpts
The **load audio folder...** button scans a folder (recursively) for audio files matching a set of extensions, loads each one, and slices them into fixed-length overlapping excerpts (length and offset/hop are both configurable directly in the GUI, in seconds). This runs on a background thread with a progress bar, since loading and analyzing an entire folder of files can take anywhere from several seconds to a couple of minutes depending on how much audio is involved.

### Audio descriptors
For every excerpt, a set of audio descriptors is computed: RMS energy, zero-crossing rate, chroma (STFT-based), mel spectrogram, MFCCs, spectral centroid, bandwidth, contrast, flatness, rolloff, and tempogram. A checklist lets you choose any combination of these descriptors as the basis for the nearest-neighbor search; each selected descriptor is z-score normalized (per dimension, across all excerpts) before being concatenated into a single distance vector, so that no single descriptor's raw scale dominates the similarity comparison. Changing the feature selection restarts the search from scratch.

### Excerpt length and offset
Two controls — **excerpt length** and **excerpt offset**, both in seconds — determine how each source file is sliced. The offset is normally shorter than the length, producing overlapping excerpts that the cross-fade envelope needs to blend smoothly; setting the offset equal to the length removes overlap entirely. Because these values determine how the underlying excerpt pool was built in the first place, changing either one and pressing **apply excerpt timing** re-slices and re-analyzes the entire currently loaded folder from scratch (on the same kind of background thread as the initial load), rather than taking effect immediately.

### The search itself
Starting from the very first excerpt in the pool, the tool repeatedly finds whichever remaining excerpt has the smallest distance (in the chosen, normalized feature space) to the current one, cross-fades that excerpt into the output recording, and removes the excerpt it just moved on from — so every excerpt is used exactly once and the walk always continues into unexplored territory. The search runs incrementally, one step at a time, on a background thread, rather than all at once.

### Real-time playback
As the search proceeds, the growing output recording plays back live through a selected audio output device — you hear the recombined audio being built in real time rather than having to wait for the entire folder to be processed first. If the search is still catching up to real-time (e.g. an expensive combination of descriptors on a very large folder), playback simply reports "waiting for more audio..." rather than stalling or glitching. **Start** begins (or resumes) both the search and playback together; **stop** pauses both.

### Saving the result
The **save audio file...** button writes everything generated so far to a WAV file, at any point — you do not need to wait for the search to finish processing the entire folder before saving; a partially generated recording can be saved just as easily as a complete one.

## Editable Source Code Sections

The entry-point script exposes a block of top-level settings intended to be edited directly, since they configure the tool's startup state rather than something adjusted interactively at runtime.

```python
AUDIO_FOLDER_PATH = "data/audio/"
AUDIO_FILE_EXTENSIONS = ["wav", "aiff", "aif"]
AUDIO_SAMPLE_RATE = 48000
AUDIO_EXCERPT_SECONDS = 5.0
AUDIO_EXCERPT_OFFSET_SECONDS = 2.5
AUDIO_FEATURE_NAMES = ["root mean square", "mfcc"]
OUTPUT_DEVICE = None
STEP_INTERVAL_SECONDS = 0.05
```

- **`AUDIO_FOLDER_PATH`**: the folder scanned automatically if "load audio folder..." is used without having changed the folder path yet; primarily relevant as a convenient default location for the file-picker dialog to reflect. Has no effect once a different folder has been picked from the GUI.
- **`AUDIO_FILE_EXTENSIONS`**: which file extensions are treated as audio and included when scanning a folder. Add extensions here (without the leading dot, lowercase) to support additional formats librosa can read (e.g. `"mp3"`, `"flac"`, `"ogg"`).
- **`AUDIO_SAMPLE_RATE`**: the sample rate every loaded file is resampled to. All files in a folder are analyzed and combined at this single rate regardless of their original sample rates, since the nearest-neighbor search and cross-fading require every excerpt to share a common sample rate.
- **`AUDIO_EXCERPT_SECONDS`** / **`AUDIO_EXCERPT_OFFSET_SECONDS`**: the startup values for excerpt length and offset, in seconds. These can also be changed live from the GUI's excerpt timing controls; the values here only apply to the very first load. Shorter excerpts produce a larger, more fine-grained pool with faster per-excerpt feature computation but a shorter available context per excerpt (relevant for descriptors like tempogram); longer excerpts do the opposite.
- **`AUDIO_FEATURE_NAMES`**: which descriptors are selected by default in the feature checklist when the tool starts. Any combination of the descriptor names computed in `nn_model.py`'s feature-computation step can be listed here.
- **`OUTPUT_DEVICE`**: the audio output device index used for real-time playback. Leave as `None` to use the system default, or set a specific device index (see the GUI's output-device dropdown, or `sounddevice.query_devices()`, for available indices).
- **`STEP_INTERVAL_SECONDS`**: how often, in seconds, the background generation thread performs one nearest-neighbor search step. Smaller values make the search progress faster (more steps per second) at a slightly higher constant CPU cost from the thread's wake-up/sleep cycle; larger values slow the search down, which can be useful for deliberately pacing generation to stay comfortably ahead of (or roughly in sync with) real-time playback.

A second editable section, inside the feature-computation method, controls exactly which descriptors are computed and available for the nearest-neighbor search:

```python
self.audio_features["root mean square"] = aa.rms(wf)
self.audio_features["zero crossing rate"] = aa.zero_crossing_rate(wf)
self.audio_features["chroma stft"] = aa.chroma_stft(wf, sr)
self.audio_features["mel spectrogram"] = aa.mel_spectrogram(wf, sr)
self.audio_features["mfcc"] = aa.mfcc(wf, sr)
...
self.audio_features["tempogram"] = aa.tempogram(wf, sr, win_length=32)
# chroma_cqt / chroma_cens / chroma_vqt / tempo / tempogram_ratio are
# left out of the default set...
```

- **Adding a new line** in the same pattern (`self.audio_features["your label"] = aa.some_function(wf, ...)`) makes any other function from `analysis.py` available in the feature checklist, as long as it returns one row of numeric values per excerpt. The commented-out descriptors (`chroma_cqt`, `chroma_cens`, `chroma_vqt`, `tempo`, `tempogram_ratio`) were left out of the default set because they are significantly slower to compute across many excerpts than the others, and `tempo` in particular needs more temporal context than a short excerpt reliably provides — enabling any of them will noticeably increase the time taken every time a folder is loaded or excerpt timing is re-applied.
- **Removing a line** removes that descriptor from the checklist; if it was part of the currently selected search features when removed, those features are dropped from the active search on the next reset.
