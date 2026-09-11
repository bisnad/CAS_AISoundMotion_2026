# Audio Analysis

A real-time audio analysis tool that extracts a wide range of audio descriptors on the fly — from either a playing audio file or a live microphone input — and streams them out as [OSC](https://ccrma.stanford.edu/groups/osc/) messages for use in other software (Max/MSP, Pure Data, TouchDesigner, game engines, etc.). It mirrors the structure of the companion real-time motion-capture analysis tool, but for audio: a source feeding a rolling buffer, a pipeline computing descriptors every tick, an OSC sender, and a GUI for control and live visualization.

## Summary

This tool continuously analyzes audio — either an audio file being played back through your speakers or sound coming from a microphone — and turns it into a stream of numeric descriptors (loudness, spectral shape, pitch content, rhythm, and more) at a configurable frame rate. Each descriptor can be individually enabled, previewed live in a bar-chart display, and sent out as an OSC message to drive another application in real time. Playback is fully controllable: you can scrub the play head, define a playback region within the file, loop it, and load a different file at any time, all while descriptors keep streaming out.

## Functionality

### Audio source: file or microphone
The tool can analyze audio from either a loaded file or a live microphone input, toggled with a single checkbox. In file mode, the tool both plays the file through a selected output device and analyzes exactly what is being played, so what you hear and what is measured are always the same signal. In microphone mode, it analyzes whatever the selected input device is capturing, with no file-related controls active.

### Playback transport (file mode)
Three sliders control playback of the loaded file, all in seconds: the **play head** (current playback position), the **region start**, and the **region end**. Only the audio within the region start/end range is played; playback loops back to the region start when it reaches the region end if the **loop** toggle is enabled, and stops there otherwise. The play head slider continuously reflects the actual playback position while playing, and can be dragged to jump to a new position at any time. A **load audio file...** button opens a file picker, swaps in the new file, and resets the transport sliders to match its duration.

### Descriptor computation
On every analysis tick, the tool pulls the most recent audio from a rolling buffer and computes a configurable set of descriptors using [librosa](https://librosa.org/): RMS energy, zero-crossing rate, chroma (STFT-based), tonnetz, mel spectrogram, MFCCs, spectral centroid, bandwidth, contrast, flatness, and rolloff, polynomial (spectral tilt) features, and onset strength — plus a set of slower, more context-hungry descriptors (constant-Q and CENS/VQT chroma variants, tempo, tempogram, Fourier tempogram, tempogram ratio) computed less frequently on a longer rolling window rather than every tick, since they need more temporal context than a single short frame provides.

### Enabling and previewing descriptors
Every descriptor has a checkbox in a list. The checkbox controls **only** whether that descriptor is sent via OSC. Separately, **highlighting** a descriptor in the list (regardless of its checkbox state) controls what is drawn in the live bar-chart display. Whichever is true — checked for OSC, or currently highlighted for viewing, or both — the tool computes that descriptor; unchecking and un-highlighting a descriptor stops computing it entirely, which is what keeps CPU load manageable when only a handful of descriptors are actually needed at once.

### OSC output
Enabled descriptors are sent as OSC messages (e.g. `/audio/0/rms`, `/audio/0/spectral_centroid`, `/audio/0/mfcc`) to a configurable IP address and port, with an on/off toggle for the OSC sender itself. The analysis frame rate (how often descriptors are recomputed and sent) is adjustable from the GUI.

### Device selection
Both the audio input device (for microphone mode) and the audio output device (for file playback) can be chosen from dropdown menus populated from the system's available audio devices.

## Editable Source Code Sections

The entry-point script exposes a block of top-level settings intended to be edited directly, since they configure the tool's startup state rather than something adjusted interactively at runtime.

```python
AUDIO_MODE = "file"
AUDIO_FILE_PATH = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz.wav"
INPUT_DEVICE = None
OUTPUT_DEVICE = None
AUDIO_SAMPLE_RATE = 48000
```

- **`AUDIO_MODE`**: `"file"` or `"mic"` — which audio source is active when the tool starts. This can also be changed live from the GUI's "use microphone" toggle; the setting here only determines the initial state.
- **`AUDIO_FILE_PATH`**: the audio file loaded automatically at startup when `AUDIO_MODE` is `"file"`. Has no effect on files subsequently loaded via the GUI's "load audio file..." button.
- **`INPUT_DEVICE`** / **`OUTPUT_DEVICE`**: device indices used at startup for microphone capture and file playback, respectively. Leave as `None` to use the system default device; both can also be changed live from the GUI's device dropdowns.
- **`AUDIO_SAMPLE_RATE`**: a fallback sample rate, used only for microphone mode or if a loaded file's own sample rate cannot be determined. A loaded file's actual sample rate always takes precedence over this value.

A second editable section configures the analysis pipeline's timing and which descriptors are computed by default:

```python
audio_pipeline_config = {
    "fps": 30,
    "n_fft_seconds": 2048 / 22050.0,
    "hop_seconds": 512 / 22050.0,
    "frame_seconds": 2048 / 22050.0,
    "heavy_window_seconds": 3.0,
    "heavy_update_every": 10,
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
```

- **`fps`**: how many times per second descriptors are recomputed and sent via OSC. Higher values give more temporally precise output at the cost of more CPU work per second; lowering it is one of the most effective ways to reduce CPU load if descriptor computation or playback starts to lag.
- **`n_fft_seconds`, `hop_seconds`, `frame_seconds`**: analysis window sizes for the per-tick ("cheap") descriptors, expressed in **seconds** rather than raw sample counts so they scale automatically to whatever sample rate the current audio source actually uses (e.g. loading a 48kHz file after starting in mic mode at 44.1kHz does not silently shrink these windows). Increasing them gives descriptors more context per frame (useful for lower-frequency content) at the cost of coarser time resolution and slightly more compute per frame.
- **`heavy_window_seconds`**: the length of the rolling window used for the slower, context-hungry descriptors (chroma_cqt, chroma_cens, chroma_vqt, tempo, tempogram, fourier_tempogram, tempogram_ratio). This must stay less than or equal to the audio receiver's buffer length (`buffer_seconds`, set in the receiver's own config, not shown above) or these descriptors will be computed on a truncated, less accurate window.
- **`heavy_update_every`**: how many analysis ticks pass between recomputations of the heavy descriptors (e.g. `10` means once every 10 ticks). Raising this reduces the CPU cost of heavy descriptors further at the cost of less frequently updated values; lowering it towards `1` computes them every tick, which is rarely necessary given how slowly most of these descriptors change.
- **`enabled_defaults`**: which descriptors are computed (and therefore available to check for OSC output or highlight for viewing) when the tool starts. Descriptors set to `False` here are not computed at all until enabled from the GUI checklist or by highlighting them — this is the main lever for keeping CPU usage low by default, since the descriptors listed as `False` (mfcc, mel_spectrogram, spectral_contrast, poly_features, tonnetz, chroma_stft, and all the heavy/windowed descriptors) are the most computationally expensive ones. Setting any of these to `True` makes that descriptor active immediately at startup instead of requiring a manual toggle in the GUI.
