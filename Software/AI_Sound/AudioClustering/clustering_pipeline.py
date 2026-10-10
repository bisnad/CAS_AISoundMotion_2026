"""
clustering_pipeline.py

Owns everything between "audio file on disk" and "feature matrices ready for
clustering":

- loads a file (through audio_receiver, as before) and trims it to
  analysis_seconds
- slices it into excerpts of excerpt_length_ms, starting every
  excerpt_offset_ms (offset = length - overlap)
- computes ONLY the audio features that were requested, and caches them, so
  ticking an extra feature computes just that feature, and changing only the
  clustering settings recomputes nothing
- changing the file or the excerpt length/offset invalidates the cache

Thread safety: all public methods take an RLock, so the GUI worker thread and
the OSC thread can both call into it.
"""

import threading
import numpy as np

import analysis as aa
import audio_receiver

N_FFT = 2048        # analysis.py default; excerpts must contain at least this many samples
MAX_EXCERPTS = 50000

# name -> (function(excerpts, sample_rate), is_slow)
FEATURES = {
    "root mean square":   (lambda x, sr: aa.rms(x), False),
    "zero crossing rate": (lambda x, sr: aa.zero_crossing_rate(x), False),
    "spectral centroid":  (lambda x, sr: aa.spectral_centroid(x, sr), False),
    "spectral bandwidth": (lambda x, sr: aa.spectral_bandwidth(x, sr), False),
    "spectral contrast":  (lambda x, sr: aa.spectral_contrast(x, sr), False),
    "spectral flatness":  (lambda x, sr: aa.spectral_flatness(x), False),
    "spectral rolloff":   (lambda x, sr: aa.spectral_rolloff(x, sr), False),
    "poly features":      (lambda x, sr: aa.poly_features(x, sr), False),
    "mel spectrogram":    (lambda x, sr: aa.mel_spectrogram(x, sr), False),
    "mfcc":               (lambda x, sr: aa.mfcc(x, sr), False),
    "chroma stft":        (lambda x, sr: aa.chroma_stft(x, sr), False),
    "onset strength":     (lambda x, sr: aa.onset_strength(x, sr), False),
    "tempogram":          (lambda x, sr: aa.tempogram(x, sr, win_length=16), False),
    "fourier tempogram":  (lambda x, sr: aa.fourier_tempogram(x, sr, win_length=16), True),
    "tempogram ratio":    (lambda x, sr: aa.tempogram_ratio(x, sr, win_length=16), True),
    "tonnetz":            (lambda x, sr: aa.tonnetz(x, sr), True),
    "chroma cqt":         (lambda x, sr: aa.chroma_cqt(x, sr), True),
    "chroma cens":        (lambda x, sr: aa.chroma_cens(x, sr), True),
    "chroma vqt":         (lambda x, sr: aa.chroma_vqt(x, sr), True),
    "tempo":              (lambda x, sr: aa.tempo(x, sr), True),
}

DEFAULT_FEATURES = ["root mean square", "mfcc"]


def get_all_feature_names():
    return list(FEATURES.keys())


def is_slow_feature(name):
    return FEATURES[name][1]


def min_excerpt_length_ms(sample_rate):
    return N_FFT / sample_rate * 1000.0


def load_waveform(file_path, fallback_sample_rate, analysis_seconds):
    audio_receiver.config["mode"] = "file"
    audio_receiver.config["file_path"] = file_path
    audio_receiver.config["loop"] = False
    audio_receiver.config["sample_rate"] = fallback_sample_rate
    audio_receiver.config["channels"] = 1

    loader = audio_receiver.AudioReceiver(audio_receiver.config)
    sample_rate = loader.get_sample_rate()

    full_waveform = loader.get_loaded_samples()
    analysis_samples = min(int(analysis_seconds * sample_rate), full_waveform.shape[0])
    return full_waveform[:analysis_samples].copy(), sample_rate


def ms_to_samples(ms, sample_rate):
    return int(round(ms / 1000.0 * sample_rate))


def count_excerpts(duration_ms, length_ms, offset_ms):
    if offset_ms <= 0 or duration_ms < length_ms:
        return 0
    return int((duration_ms - length_ms) // offset_ms) + 1


def build_excerpts(waveform, sample_rate, excerpt_length_ms, excerpt_offset_ms):
    """Slices waveform into excerpts of length_ms, one starting every offset_ms."""
    length_sc = ms_to_samples(excerpt_length_ms, sample_rate)
    offset_sc = ms_to_samples(excerpt_offset_ms, sample_rate)

    if offset_sc < 1:
        raise ValueError("excerpt offset is shorter than one sample")
    if length_sc < N_FFT:
        raise ValueError(
            "excerpt length must be at least {:.1f} ms at {} Hz (the analysis uses a {}-sample FFT)".format(
                min_excerpt_length_ms(sample_rate), sample_rate, N_FFT))
    if waveform.shape[0] < length_sc:
        raise ValueError("audio ({:.0f} ms) is shorter than one excerpt ({} ms)".format(
            waveform.shape[0] / sample_rate * 1000.0, excerpt_length_ms))

    starts = np.arange(0, waveform.shape[0] - length_sc + 1, offset_sc)
    if starts.shape[0] > MAX_EXCERPTS:
        raise ValueError(
            "{} excerpts would be created (limit {}). Increase the excerpt length or reduce the overlap.".format(
                starts.shape[0], MAX_EXCERPTS))

    return np.stack([waveform[s:s + length_sc] for s in starts], axis=0)


class ClusteringPipeline():

    def __init__(self, fallback_sample_rate=48000, analysis_seconds=60.0):
        self.fallback_sample_rate = fallback_sample_rate
        self.analysis_seconds = analysis_seconds

        self.file_path = None
        self.waveform = None
        self.sample_rate = fallback_sample_rate

        self.excerpt_length_ms = None
        self.excerpt_offset_ms = None
        self.excerpts = None
        self.features = {}

        self._lock = threading.RLock()

    def get_duration_ms(self):
        with self._lock:
            if self.waveform is None:
                return 0.0
            return self.waveform.shape[0] / self.sample_rate * 1000.0

    def get_sample_rate(self):
        return self.sample_rate

    def _compute_missing(self, features, excerpts, sample_rate, feature_names):
        for name in feature_names:
            if name in features:
                continue
            if name not in FEATURES:
                raise ValueError("unknown audio feature '{}'".format(name))
            try:
                features[name] = FEATURES[name][0](excerpts, sample_rate)
            except Exception as e:
                raise RuntimeError("feature '{}' could not be computed for {} ms excerpts: {}".format(
                    name, self.excerpt_length_ms if excerpts is self.excerpts else "these", e))

    def prepare(self, file_path, excerpt_length_ms, excerpt_offset_ms, feature_names):
        """
        Brings the pipeline to the requested state and returns
        (audio_excerpts, audio_features, sample_rate).

        file_path=None keeps the current file. State is only committed once
        everything succeeded, so a failed attempt leaves the previous
        (working) excerpts/features untouched.
        """
        feature_names = [n for n in feature_names if n != "waveform"]

        with self._lock:
            waveform, sample_rate, path = self.waveform, self.sample_rate, self.file_path
            reloaded = False
            if file_path is not None:
                waveform, sample_rate = load_waveform(file_path, self.fallback_sample_rate, self.analysis_seconds)
                path = file_path
                reloaded = True
            if waveform is None:
                raise ValueError("no audio file loaded")

            unchanged = (not reloaded and self.excerpts is not None
                         and excerpt_length_ms == self.excerpt_length_ms
                         and excerpt_offset_ms == self.excerpt_offset_ms)

            if unchanged:
                excerpts, features = self.excerpts, self.features
            else:
                excerpts = build_excerpts(waveform, sample_rate, excerpt_length_ms, excerpt_offset_ms)
                features = {"waveform": excerpts}

            old_params = (self.excerpt_length_ms, self.excerpt_offset_ms)
            self.excerpt_length_ms, self.excerpt_offset_ms = excerpt_length_ms, excerpt_offset_ms
            try:
                self._compute_missing(features, excerpts, sample_rate, feature_names)
            except Exception:
                self.excerpt_length_ms, self.excerpt_offset_ms = old_params
                raise

            self.waveform, self.sample_rate, self.file_path = waveform, sample_rate, path
            self.excerpts, self.features = excerpts, features
            return excerpts, features, sample_rate

    def ensure_features(self, feature_names):
        """Computes any requested feature not yet cached (used by OSC) and returns the shared feature dict."""
        with self._lock:
            self._compute_missing(self.features, self.excerpts, self.sample_rate,
                                  [n for n in feature_names if n != "waveform"])
            return self.features
