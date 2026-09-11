import numpy as np
import analysis as aa


class AudioPipeline():
    """
    Mirrors motion_pipeline.MotionPipeline: on every update() call it pulls
    the latest audio samples from the receiver's rolling buffer, computes a
    set of "cheap" per-hop descriptors, and periodically (on a longer
    rolling window) computes "heavy" descriptors that need more temporal
    context (tempo, tempogram, CQT/CENS/VQT chroma variants).

    PERFORMANCE: two independent optimizations are applied compared to the
    naive "call every librosa function separately" approach, which was too
    slow to keep up with 48kHz audio at interactive frame rates:

    1. Shared STFT: chroma_stft, mel_spectrogram, mfcc, spectral_centroid,
       spectral_bandwidth, spectral_contrast and spectral_rolloff all
       accept a precomputed power spectrogram (S). This pipeline computes
       that spectrogram ONCE per tick via analysis.stft_power() and passes
       it to every descriptor that can use it, instead of each descriptor
       recomputing its own STFT independently.

    2. Enable flags (self.enabled): every descriptor - cheap or heavy - is
       gated by a boolean in self.enabled. Disabled descriptors are not
       computed at all (not just excluded from OSC sending), so turning
       off expensive ones (mfcc, mel_spectrogram, spectral_contrast,
       poly_features, tonnetz, chroma_stft) directly reduces per-tick CPU
       load. audio_gui.py's checklist items double as these enable flags -
       checking an item both enables computation AND OSC sending for it.

    RECOMMENDATION for 48kHz real-time use: keep rms, zero_crossing_rate,
    spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness
    and onset_strength enabled (all cheap, and/or share the one STFT); disable
    mfcc, mel_spectrogram, spectral_contrast, poly_features, tonnetz and
    chroma_stft first if still CPU-bound, in roughly that order of cost.
    Heavy descriptors (chroma_cqt/cens/vqt, tempo, tempogram family) already
    run at a reduced rate (heavy_update_every) and can be disabled entirely
    via self.enabled if they still cause periodic stalls.
    """

    def __init__(self, audioReceiver, audio_config):

        self.audio_config = audio_config
        self.receiver = audioReceiver

        self.fps = audio_config["fps"]
        self.updateInterval = 1.0 / self.fps

        self._tick_count = 0

        # smoothing factors (mirrors posSmoothFactor etc. in mocap pipeline)
        self.rmsSmoothFactor = 0.9
        self.centroidSmoothFactor = 0.9
        self.onsetSmoothFactor = 0.9

        # per-descriptor enable flags - disabled descriptors are skipped
        # entirely (not computed), which is what actually saves CPU time.
        # audio_gui.py keeps this dict in sync with its checklist.
        default_enabled = audio_config.get("enabled_defaults", None)
        self.enabled = {
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
        }
        if default_enabled:
            self.enabled.update(default_enabled)

        # cheap descriptors (initialized to zero-shaped arrays, populated on first update)
        self.rms = np.zeros(1)
        self.rmsSmooth = np.zeros(1)
        self.zero_crossing_rate = np.zeros(1)
        self.chroma_stft = np.zeros(12)
        self.tonnetz = np.zeros(6)
        self.mel_spectrogram = np.zeros(1)
        self.mfcc = np.zeros(1)
        self.spectral_centroid = np.zeros(1)
        self.spectral_centroidSmooth = np.zeros(1)
        self.spectral_bandwidth = np.zeros(1)
        self.spectral_contrast = np.zeros(1)
        self.spectral_flatness = np.zeros(1)
        self.spectral_rolloff = np.zeros(1)
        self.poly_features = np.zeros(1)
        self.onset_strength = np.zeros(1)
        self.onset_strengthSmooth = np.zeros(1)

        # heavy / windowed descriptors
        self.chroma_cqt = np.zeros(12)
        self.chroma_cens = np.zeros(12)
        self.chroma_vqt = np.zeros(12)
        self.tempo = np.zeros(1)
        self.tempogram = np.zeros(1)
        self.fourier_tempogram = np.zeros(1)
        self.tempogram_ratio = np.zeros(1)

        # ring buffer of onset-strength values, in case downstream code
        # wants windowed aggregation (mirrors ringSize pattern in mocap pipeline)
        self.ringSize = 25
        self.onsetRing = np.zeros(self.ringSize)

        self.reconfigure()

    def reconfigure(self):
        """
        (Re)derives all sample-rate-dependent window sizes from the
        receiver's CURRENT sample rate. Call this after loading a new audio
        file or switching source mode (file/mic), since the sample rate may
        have changed.
        """

        audio_config = self.audio_config
        self.sample_rate = self.receiver.get_sample_rate()

        n_fft_seconds = audio_config.get("n_fft_seconds", 2048 / 22050.0)
        hop_seconds = audio_config.get("hop_seconds", 512 / 22050.0)
        frame_seconds = audio_config.get("frame_seconds", n_fft_seconds)

        self.n_fft = self._next_pow2(int(round(n_fft_seconds * self.sample_rate)))
        self.hop_length = max(int(round(hop_seconds * self.sample_rate)), 1)
        self.frame_length = max(int(round(frame_seconds * self.sample_rate)), self.n_fft)

        self.heavy_window_seconds = audio_config.get("heavy_window_seconds", 3.0)
        self.heavy_update_every = audio_config.get("heavy_update_every", 10)  # in ticks

        self.heavy_window_samples = int(self.heavy_window_seconds * self.sample_rate)

        heavy_hop_seconds = audio_config.get("heavy_hop_seconds", hop_seconds)
        self.heavy_hop_length = max(int(round(heavy_hop_seconds * self.sample_rate)), 1)

        self.chroma_cqt_octaves = audio_config.get("chroma_cqt_octaves", 5)

        self.tempogram_win_length = self._safe_tempogram_win_length(
            self.heavy_window_samples, self.heavy_hop_length
        )

        buffer_seconds = self.receiver.get_buffer_seconds() if hasattr(self.receiver, "get_buffer_seconds") else None
        if buffer_seconds is not None and buffer_seconds < self.heavy_window_seconds:
            print(
                "[audio_pipeline] warning: receiver buffer ({:.2f}s) is shorter than "
                "heavy_window_seconds ({:.2f}s) - heavy descriptors will be computed "
                "on a truncated window and may be less accurate.".format(
                    buffer_seconds, self.heavy_window_seconds
                )
            )

    @staticmethod
    def _next_pow2(n):
        n = max(int(n), 1)
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _safe_tempogram_win_length(window_samples, hop_length, max_default=384, min_default=16, margin_divisor=4):
        """
        Derives a win_length (in onset-envelope frames) for
        tempogram/fourier_tempogram/tempogram_ratio that is guaranteed to be
        comfortably smaller than the number of onset frames the given
        window/hop combination will actually produce, then snaps it down to
        a power of 2.
        """
        available_frames = max(window_samples // hop_length, 1)
        target = available_frames // margin_divisor
        target = max(min(target, max_default), min_default)

        p = 1
        while p * 2 <= target:
            p *= 2
        return max(p, 2)

    def setUpdateInterval(self, updateInterval):
        self.updateInterval = updateInterval

    def set_enabled(self, name, value):
        if name in self.enabled:
            self.enabled[name] = bool(value)

    def update(self):

        frame = self.receiver.get_latest(self.frame_length)
        frame = frame[np.newaxis, :]  # shape [1, samples], batch dim of 1

        sr = self.sample_rate
        en = self.enabled

        effective_n_fft = min(self.n_fft, frame.shape[-1]) if frame.shape[-1] > 0 else self.n_fft
        effective_n_fft = self._next_pow2(effective_n_fft) if effective_n_fft >= 1 else self.n_fft
        effective_n_fft = min(effective_n_fft, self.n_fft)
        effective_n_fft = max(effective_n_fft, 2)

        # compute the shared power spectrogram once if any STFT-based
        # descriptor is enabled, and reuse it across all of them
        needs_stft = en["chroma_stft"] or en["mel_spectrogram"] or en["mfcc"] or \
            en["spectral_centroid"] or en["spectral_bandwidth"] or en["spectral_contrast"] or en["spectral_rolloff"]

        S_list = None
        if needs_stft:
            S_list = aa.stft_power(frame, n_fft=effective_n_fft, hop_length=self.hop_length)

        if en["rms"]:
            _rms = aa.rms(frame, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            self.rmsSmooth = self.rmsSmooth * self.rmsSmoothFactor + _rms * (1.0 - self.rmsSmoothFactor)
            self.rms = _rms

        if en["zero_crossing_rate"]:
            self.zero_crossing_rate = aa.zero_crossing_rate(frame, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        if en["chroma_stft"]:
            self.chroma_stft = aa.chroma_stft(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["tonnetz"]:
            self.tonnetz = aa.tonnetz(frame, sr, hop_length=self.hop_length, n_fft=effective_n_fft)[0]

        if en["mel_spectrogram"]:
            self.mel_spectrogram = aa.mel_spectrogram(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["mfcc"]:
            self.mfcc = aa.mfcc(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["spectral_centroid"]:
            _centroid = aa.spectral_centroid(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]
            self.spectral_centroidSmooth = self.spectral_centroidSmooth * self.centroidSmoothFactor + _centroid * (1.0 - self.centroidSmoothFactor)
            self.spectral_centroid = _centroid

        if en["spectral_bandwidth"]:
            self.spectral_bandwidth = aa.spectral_bandwidth(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["spectral_contrast"]:
            self.spectral_contrast = aa.spectral_contrast(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["spectral_flatness"]:
            self.spectral_flatness = aa.spectral_flatness(frame, n_fft=effective_n_fft, hop_length=self.hop_length)[0]

        if en["spectral_rolloff"]:
            self.spectral_rolloff = aa.spectral_rolloff(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["poly_features"]:
            self.poly_features = aa.poly_features(frame, sr, order=1, n_fft=effective_n_fft, hop_length=self.hop_length, S_list=S_list)[0]

        if en["onset_strength"]:
            _onset = aa.onset_strength(frame, sr, n_fft=effective_n_fft, hop_length=self.hop_length)[0]
            self.onset_strengthSmooth = self.onset_strengthSmooth * self.onsetSmoothFactor + _onset * (1.0 - self.onsetSmoothFactor)
            self.onset_strength = _onset

            self.onsetRing = np.roll(self.onsetRing, shift=1, axis=0)
            self.onsetRing[0] = np.mean(_onset) if _onset.size > 0 else 0.0

        self._tick_count += 1
        if self._tick_count % self.heavy_update_every == 0:
            self._update_heavy()

    def _update_heavy(self):

        en = self.enabled
        if not (en["chroma_cqt"] or en["chroma_cens"] or en["chroma_vqt"] or
                en["tempo"] or en["tempogram"] or en["fourier_tempogram"] or en["tempogram_ratio"]):
            return

        window = self.receiver.get_latest(self.heavy_window_samples)
        window = window[np.newaxis, :]

        sr = self.sample_rate

        if en["chroma_cqt"]:
            try:
                self.chroma_cqt = aa.chroma_cqt(
                    window, sr, hop_length=self.heavy_hop_length, n_octaves=self.chroma_cqt_octaves
                )[0]
            except Exception:
                pass

        if en["chroma_cens"]:
            try:
                self.chroma_cens = aa.chroma_cens(
                    window, sr, hop_length=self.heavy_hop_length, n_octaves=self.chroma_cqt_octaves
                )[0]
            except Exception:
                pass

        if en["chroma_vqt"]:
            try:
                self.chroma_vqt = aa.chroma_vqt(window, sr, hop_length=self.heavy_hop_length)[0]
            except Exception:
                pass

        if en["tempo"]:
            try:
                self.tempo = aa.tempo(window, sr, hop_length=self.heavy_hop_length)[0]
            except Exception:
                pass

        if en["tempogram"]:
            try:
                self.tempogram = aa.tempogram(
                    window, sr, hop_length=self.heavy_hop_length, win_length=self.tempogram_win_length
                )[0]
            except Exception:
                pass

        if en["fourier_tempogram"]:
            try:
                self.fourier_tempogram = aa.fourier_tempogram(
                    window, sr, hop_length=self.heavy_hop_length, win_length=self.tempogram_win_length
                )[0]
            except Exception:
                pass

        if en["tempogram_ratio"]:
            try:
                self.tempogram_ratio = aa.tempogram_ratio(
                    window, sr, hop_length=self.heavy_hop_length, win_length=self.tempogram_win_length
                )[0]
            except Exception:
                pass
