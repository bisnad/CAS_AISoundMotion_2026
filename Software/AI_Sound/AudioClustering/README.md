# Audio Clustering

A real-time-playable audio clustering tool: it slices an audio file into short overlapping excerpts, computes a range of audio descriptors (timbral, spectral, and rhythmic) for each excerpt using [librosa](https://librosa.org/), groups the excerpts into clusters using k-means (or mini-batch k-means), and lets you listen to any cluster as a continuously looping, cross-faded audio stream. Cluster membership, the descriptor used for clustering, and playback can all be controlled from the GUI or remotely via OSC.

## Summary

This tool answers the question *"which parts of this recording sound similar to each other?"* by turning audio into numeric descriptors, clustering those descriptors, and letting you audition each resulting group as sound. Load an audio file, choose which audio feature to cluster on (timbre, pitch, energy, rhythm, etc.), and the tool automatically groups all excerpts of the file into clusters. Selecting a cluster in the GUI reconstructs and loops a playable audio stream made only from the excerpts in that cluster, so you can hear what that group of sounds actually sounds like. A new audio file can be loaded at any time and is re-analyzed and re-clustered in the background without restarting the application. An OSC server also exposes cluster selection, feature selection, cluster count, and clustering method to external software, making the tool usable as a live sound-exploration or composition component in a larger patch (Max/MSP, Pure Data, TouchDesigner, etc.).

## Functionality

### Excerpt extraction and feature computation
On load, the audio file is sliced into short, overlapping excerpts (length and hop/offset are configurable, in milliseconds). For every excerpt, a set of audio descriptors is computed: RMS energy, chroma (STFT-based), mel spectrogram, MFCCs, spectral centroid, bandwidth, contrast, flatness, rolloff, zero-crossing rate, and tempogram. Each descriptor is stored as its own feature matrix (one row per excerpt), and any one of them can be selected as the basis for clustering.

### Clustering
Clustering is performed with k-means or mini-batch k-means (scikit-learn). Before fitting, each feature is z-score normalized across excerpts by default, since raw descriptor magnitudes differ by orders of magnitude (e.g. mel spectrogram energy vs. a spectral centroid in Hz) and unnormalized clustering would be dominated by whichever descriptor happens to have the largest scale. Normalization can be turned off from the GUI to compare behavior. The clustering method, cluster count, and normalization setting can all be changed live; changing any of them re-clusters immediately using the currently selected feature.

### Cluster playback
Selecting a cluster reconstructs a single continuous waveform from every excerpt assigned to that cluster, overlap-added with a cross-fade envelope derived from the excerpt length and hop, then loops that waveform through a real-time audio output stream. Because excerpts within a cluster are not necessarily contiguous in the original recording, the resulting playback is a resequenced, looping collage of everything the tool judged to sound alike. The audio output device can be selected from a dropdown, and playback is started/stopped explicitly with dedicated buttons.

### Loading a new audio file
The "load audio file..." button opens a file picker, then re-runs excerpt extraction, feature computation, and clustering for the newly selected file on a background thread, so the interface stays responsive during analysis (which can take anywhere from a few seconds to a bit longer depending on file length and excerpt count). A status message reports progress and the resulting excerpt count once finished. Cluster count, clustering method, normalization, and the currently selected feature are preserved across the file swap; only the underlying excerpts and computed features are replaced. Playback is stopped automatically before loading and must be restarted manually afterward.

### Remote (OSC) control
An OSC server runs alongside the GUI and accepts the following messages, letting an external application drive the tool live:

| OSC address | Argument | Effect |
|---|---|---|
| `/synth/clusterlabel` | integer | Selects and plays the given cluster label |
| `/synth/audiofeature` | string | Switches the descriptor used for clustering, then re-clusters |
| `/synth/clustercount` | integer | Changes the number of clusters, then re-clusters |
| `/synth/clustermethod` | string (`kmeans` or `minibatch_kmeans`) | Changes the clustering algorithm, then re-clusters |

The GUI's cluster list and status stay in sync with changes made via OSC (refreshed on a short timer), so the two control paths can be mixed freely.

## Editable Source Code Sections

The entry-point script exposes a block of top-level settings intended to be edited directly, rather than through the GUI, since they define the initial state of the tool rather than something meant to change interactively at runtime.

```python
AUDIO_FILE_PATH = "D:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
AUDIO_SAMPLE_RATE = 48000
AUDIO_EXCERPT_LENGTH_MS = 100
AUDIO_EXCERPT_OFFSET_MS = 90
AUDIO_ANALYSIS_SECONDS = 60.0
OUTPUT_DEVICE = None
```

- **`AUDIO_FILE_PATH`**: the audio file loaded automatically when the tool starts. Change this to point at a different file; it has no effect on files subsequently loaded via the GUI's "load audio file..." button, which always uses whatever file the user picks.
- **`AUDIO_SAMPLE_RATE`**: a fallback sample rate, only used if a loaded file's own sample rate cannot be determined. In practice, the file's actual sample rate always takes precedence — this value does not resample anything and rarely needs changing.
- **`AUDIO_EXCERPT_LENGTH_MS`**: the duration of each analyzed excerpt, in milliseconds. Larger values give each excerpt more context (useful for descriptors like tempogram or spectral contrast) but produce fewer, coarser excerpts and slower per-excerpt feature computation; smaller values give finer-grained, more numerous excerpts at higher computational cost.
- **`AUDIO_EXCERPT_OFFSET_MS`**: the hop between the start of consecutive excerpts, in milliseconds. Smaller than `AUDIO_EXCERPT_LENGTH_MS` (as in the default 90 ms offset for a 100 ms excerpt) creates overlapping excerpts, which the cross-fade envelope in playback relies on for smooth-sounding cluster reconstruction. Setting the offset equal to the length removes overlap entirely; setting it larger creates gaps in coverage.
- **`AUDIO_ANALYSIS_SECONDS`**: how much of a loaded file (from its start) is analyzed and clustered. Increasing this analyzes more of the file at the cost of longer feature-computation time and more excerpts to cluster; it does not affect files shorter than this value, which are analyzed in full.
- **`OUTPUT_DEVICE`**: the audio output device index used for cluster playback. Leave as `None` to use the system default, or set it to a specific device index (see the GUI's output-device dropdown, or `sounddevice.query_devices()`, for available indices) to hard-code a particular output device at startup.

A second editable section, inside the feature-computation function, controls exactly which descriptors are computed and available for clustering:

```python
audio_features["root mean square"] = aa.rms(audio_excerpts)
audio_features["chroma stft"] = aa.chroma_stft(audio_excerpts, sample_rate)
# audio_features["chroma cqt"] = aa.chroma_cqt(audio_excerpts, sample_rate)
# audio_features["chroma cens"] = aa.chroma_cens(audio_excerpts, sample_rate)
# audio_features["chroma vqt"] = aa.chroma_vqt(audio_excerpts, sample_rate)
audio_features["mel spectrogram"] = aa.mel_spectrogram(audio_excerpts, sample_rate)
audio_features["mfcc"] = aa.mfcc(audio_excerpts, sample_rate)
...
# audio_features["tempo"] = aa.tempo(audio_excerpts, sample_rate)
audio_features["tempogram"] = aa.tempogram(audio_excerpts, sample_rate, win_length=16)
# audio_features["tempogram ratio"] = aa.tempogram_ratio(audio_excerpts, sample_rate)
```

- **Uncommenting a line** adds that descriptor to the audio feature dropdown, making it selectable as a clustering basis both in the GUI and via the `/synth/audiofeature` OSC message. The commented-out descriptors (`chroma_cqt`, `chroma_cens`, `chroma_vqt`, `tempo`, `tempogram_ratio`) are disabled by default because they are significantly slower to compute across many excerpts than the others, so enabling them will noticeably increase both startup time and the time taken each time a new file is loaded.
- **Commenting out an enabled line** removes that descriptor from availability; if it was the currently selected clustering feature when removed, the tool falls back to the first remaining available descriptor.
- **Adding a new line** in the same pattern (`audio_features["your label"] = aa.some_function(audio_excerpts, ...)`) makes any other function from `analysis.py` available as a clustering descriptor, as long as it returns one row of numeric values per excerpt.
