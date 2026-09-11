import numpy as np
import librosa

"""
Real-time-friendly audio descriptor functions.

PERFORMANCE NOTE: several of librosa's spectral descriptors (chroma_stft,
melspectrogram, spectral_centroid, spectral_bandwidth, spectral_contrast,
spectral_rolloff) all internally start by computing the same STFT magnitude
spectrogram from y. Calling them independently - as the original version of
this module did - recomputes that STFT once per descriptor, which is the
single biggest CPU cost in the real-time pipeline at 48kHz. Every function
below that supports it now accepts an optional precomputed power
spectrogram S (shape [n_fft//2+1, frames]); audio_pipeline.py computes S
once per tick via stft_power() and passes it into all of these, cutting
several redundant STFTs down to one. Functions are still fully usable with
just y/sr (S=None) for standalone/offline use.
"""


def stft_power(audio_excerpts, n_fft=2048, hop_length=512, center=False):
    """
    Compute the power spectrogram (|STFT|^2) once, to be shared across
    chroma_stft, mel_spectrogram, spectral_centroid, spectral_bandwidth,
    spectral_contrast and spectral_rolloff for a given frame. Returns a
    list of per-excerpt spectrograms (not stacked, since frame counts are
    typically small and consistent within a batch of size 1).
    """
    spectrograms = []
    for audio_excerpt in audio_excerpts:
        D = librosa.stft(y=audio_excerpt, n_fft=n_fft, hop_length=hop_length, center=center)
        S = np.abs(D) ** 2
        spectrograms.append(S)
    return spectrograms


def rms(audio_excerpts, frame_length=2048, hop_length=512, center=False):

    """
    Compute root-mean-square (RMS) value for each frame from the audio samples
    """

    root_mean_square = []

    for audio_excerpt in audio_excerpts:

        rms_ = librosa.feature.rms(y=audio_excerpt, frame_length=frame_length, hop_length=hop_length, center=center)
        rms_ = rms_.flatten()

        root_mean_square.append(rms_)

    root_mean_square = np.stack(root_mean_square, axis=0)

    return root_mean_square


def zero_crossing_rate(audio_excerpts, frame_length=2048, hop_length=512, center=False):

    """
    Compute the zero-crossing rate of an audio time series.
    Cheap proxy for noisiness / percussiveness.
    """

    zcr = []

    for audio_excerpt in audio_excerpts:

        zcr_ = librosa.feature.zero_crossing_rate(y=audio_excerpt, frame_length=frame_length, hop_length=hop_length, center=center)
        zcr_ = zcr_.flatten()

        zcr.append(zcr_)

    zcr = np.stack(zcr, axis=0)

    return zcr


def chroma_stft(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    """
    Compute a chromagram from a waveform, or from a precomputed power
    spectrogram (S_list, one per excerpt) to avoid recomputing the STFT.
    """

    chroma_stft = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            stft = librosa.feature.chroma_stft(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            stft = librosa.feature.chroma_stft(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        stft = stft.flatten()

        chroma_stft.append(stft)

    chroma_stft = np.stack(chroma_stft, axis=0)

    return chroma_stft


def chroma_cqt(audio_excerpts, audio_sample_rate, hop_length=512, fmin=None, n_octaves=5, bins_per_octave=12):

    """
    Constant-Q chromagram (slow to calculate). n_octaves is reduced from
    librosa's default (7) to 5 by default so the internal CQT filterbank
    does not require more samples than a short real-time buffer provides.
    """

    chroma_cqt = []

    for audio_excerpt in audio_excerpts:

        cqt = librosa.feature.chroma_cqt(
            y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length,
            fmin=fmin, n_octaves=n_octaves, bins_per_octave=bins_per_octave
        )
        cqt = cqt.flatten()

        chroma_cqt.append(cqt)

    chroma_cqt = np.stack(chroma_cqt, axis=0)

    return chroma_cqt


def chroma_cens(audio_excerpts, audio_sample_rate, hop_length=512, fmin=None, n_octaves=5, bins_per_octave=12):

    """
    Compute the chroma variant "Chroma Energy Normalized" (CENS) (slow to
    calculate). n_octaves reduced from librosa's default for the same
    reason as chroma_cqt above.
    """

    chroma_cens = []

    for audio_excerpt in audio_excerpts:

        cens = librosa.feature.chroma_cens(
            y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length,
            fmin=fmin, n_octaves=n_octaves, bins_per_octave=bins_per_octave
        )
        cens = cens.flatten()

        chroma_cens.append(cens)

    chroma_cens = np.stack(chroma_cens, axis=0)

    return chroma_cens


def chroma_vqt(audio_excerpts, audio_sample_rate, hop_length=512):

    # Variable-Q chromagram (slow to calculate)

    chroma_vqt = []

    for audio_excerpt in audio_excerpts:

        vqt = librosa.feature.chroma_vqt(y=audio_excerpt, sr=audio_sample_rate, intervals="ji5", hop_length=hop_length)
        vqt = vqt.flatten()

        chroma_vqt.append(vqt)

    chroma_vqt = np.stack(chroma_vqt, axis=0)

    return chroma_vqt


def tonnetz(audio_excerpts, audio_sample_rate, chroma=None, hop_length=512, n_fft=2048):

    """
    Compute the tonal centroid features (tonnetz). If no precomputed chroma
    is supplied, chroma is computed here explicitly (rather than letting
    librosa fall back to its own internal chroma_cqt call with defaults).
    Expensive relative to its usefulness in a real-time OSC context -
    consider disabling first if CPU-bound.
    """

    tonnetz_list = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if chroma is not None:
            chroma_i = chroma[i]
        else:
            chroma_i = librosa.feature.chroma_stft(
                y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=False
            )

        tnz = librosa.feature.tonnetz(y=audio_excerpt, sr=audio_sample_rate, chroma=chroma_i)
        tnz = tnz.flatten()

        tonnetz_list.append(tnz)

    tonnetz_arr = np.stack(tonnetz_list, axis=0)

    return tonnetz_arr


def mel_spectrogram(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    """
    Compute a mel-scaled spectrogram, optionally reusing a precomputed
    power spectrogram (S_list) instead of recomputing the STFT.
    """

    mel_spectrogram = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            mel = librosa.feature.melspectrogram(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            mel = librosa.feature.melspectrogram(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        mel = mel.flatten()

        mel_spectrogram.append(mel)

    mel_spectrogram = np.stack(mel_spectrogram, axis=0)

    return mel_spectrogram


def mfcc(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    """
    Mel-frequency cepstral coefficients (MFCCs). If S_list (precomputed
    power spectrograms) is given, they are converted to log-mel and passed
    to librosa.feature.mfcc via its S kwarg to avoid recomputing the STFT.
    Still one of the more expensive per-tick descriptors due to the DCT
    step - consider disabling first if CPU-bound.
    """

    mfcc = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            mel = librosa.feature.melspectrogram(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
            log_mel = librosa.power_to_db(mel)
            mfcc_ = librosa.feature.mfcc(S=log_mel, sr=audio_sample_rate)
        else:
            mfcc_ = librosa.feature.mfcc(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        mfcc_ = mfcc_.flatten()

        mfcc.append(mfcc_)

    mfcc = np.stack(mfcc, axis=0)

    return mfcc


def spectral_centroid(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    # Compute the spectral centroid, optionally reusing a precomputed power spectrogram.

    spectral_centroid = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            cent = librosa.feature.spectral_centroid(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            cent = librosa.feature.spectral_centroid(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        cent = cent.flatten()

        spectral_centroid.append(cent)

    spectral_centroid = np.stack(spectral_centroid, axis=0)

    return spectral_centroid


def spectral_bandwidth(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    # Compute p'th-order spectral bandwidth, optionally reusing a precomputed power spectrogram.

    spectral_bandwidth = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            bandwidth = librosa.feature.spectral_bandwidth(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        bandwidth = bandwidth.flatten()

        spectral_bandwidth.append(bandwidth)

    spectral_bandwidth = np.stack(spectral_bandwidth, axis=0)

    return spectral_bandwidth


def spectral_contrast(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    """
    Compute spectral contrast, optionally reusing a precomputed power
    spectrogram. Still relatively expensive (per-octave-band statistics) -
    consider disabling first if CPU-bound.
    """

    spectral_contrast = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            contrast = librosa.feature.spectral_contrast(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            contrast = librosa.feature.spectral_contrast(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        contrast = contrast.flatten()

        spectral_contrast.append(contrast)

    spectral_contrast = np.stack(spectral_contrast, axis=0)

    return spectral_contrast


def spectral_flatness(audio_excerpts, n_fft=2048, hop_length=512, center=False):

    # Compute spectral flatness (needs a magnitude, not power, spectrogram - kept independent)

    spectral_flatness = []

    for audio_excerpt in audio_excerpts:

        flatness = librosa.feature.spectral_flatness(y=audio_excerpt, n_fft=n_fft, hop_length=hop_length, center=center)
        flatness = flatness.flatten()

        spectral_flatness.append(flatness)

    spectral_flatness = np.stack(spectral_flatness, axis=0)

    return spectral_flatness


def spectral_rolloff(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False, S_list=None):

    # Compute roll-off frequency, optionally reusing a precomputed power spectrogram.

    spectral_rolloff = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            rolloff = librosa.feature.spectral_rolloff(S=S_list[i], sr=audio_sample_rate, n_fft=n_fft)
        else:
            rolloff = librosa.feature.spectral_rolloff(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        rolloff = rolloff.flatten()

        spectral_rolloff.append(rolloff)

    spectral_rolloff = np.stack(spectral_rolloff, axis=0)

    return spectral_rolloff


def poly_features(audio_excerpts, audio_sample_rate, order=1, n_fft=2048, hop_length=512, center=False, S_list=None):

    """
    Compute polynomial features (spectral tilt / slope). Fitting the
    polynomial itself (np.polyfit under the hood) is one of the more
    expensive per-frame operations - consider disabling first if CPU-bound.
    """

    poly = []

    for i, audio_excerpt in enumerate(audio_excerpts):

        if S_list is not None:
            poly_ = librosa.feature.poly_features(S=S_list[i], sr=audio_sample_rate, order=order, n_fft=n_fft)
        else:
            poly_ = librosa.feature.poly_features(y=audio_excerpt, sr=audio_sample_rate, order=order, n_fft=n_fft, hop_length=hop_length, center=center)
        poly_ = poly_.flatten()

        poly.append(poly_)

    poly = np.stack(poly, axis=0)

    return poly


def onset_strength(audio_excerpts, audio_sample_rate, n_fft=2048, hop_length=512, center=False):

    """
    Compute a spectral flux onset strength envelope (novelty curve).
    """

    onset_env = []

    for audio_excerpt in audio_excerpts:

        env = librosa.onset.onset_strength(y=audio_excerpt, sr=audio_sample_rate, n_fft=n_fft, hop_length=hop_length, center=center)
        env = env.flatten()

        onset_env.append(env)

    onset_env = np.stack(onset_env, axis=0)

    return onset_env


def tempo(audio_excerpts, audio_sample_rate, hop_length=512):

    # Estimate the tempo (beats per minute)

    tempo = []

    for audio_excerpt in audio_excerpts:

        tempo_ = librosa.feature.tempo(y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length)
        tempo_ = tempo_.flatten()

        tempo.append(tempo_)

    tempo = np.stack(tempo, axis=0)

    return tempo


def tempogram(audio_excerpts, audio_sample_rate, hop_length=512, win_length=64):

    """
    Compute the tempogram: local autocorrelation of the onset strength
    envelope. win_length is in onset-envelope FRAMES (see module docstring
    in audio_pipeline.py for sizing details).
    """

    tempogram = []

    for audio_excerpt in audio_excerpts:

        tempogram_ = librosa.feature.tempogram(y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length, win_length=win_length)
        tempogram_ = tempogram_.flatten()

        tempogram.append(tempogram_)

    tempogram = np.stack(tempogram, axis=0)

    return tempogram


def fourier_tempogram(audio_excerpts, audio_sample_rate, hop_length=512, win_length=64):

    """
    Compute the Fourier tempogram: the short-time Fourier transform of the
    onset strength envelope. See tempogram() docstring re: win_length units.
    """

    fourier_tempogram = []

    for audio_excerpt in audio_excerpts:

        tempogram = librosa.feature.fourier_tempogram(y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length, win_length=win_length)
        tempogram = tempogram.flatten()

        fourier_tempogram.append(tempogram)

    fourier_tempogram = np.stack(fourier_tempogram, axis=0)
    fourier_tempogram = np.stack([np.real(fourier_tempogram), np.imag(fourier_tempogram)], axis=-1)
    fourier_tempogram = fourier_tempogram.reshape(audio_excerpts.shape[0], -1)

    return fourier_tempogram


def tempogram_ratio(audio_excerpts, audio_sample_rate, hop_length=512, win_length=64):

    """
    Tempogram ratio features, also known as spectral rhythm patterns.
    See tempogram() docstring re: win_length units.
    """

    tempogram_ratio = []

    for audio_excerpt in audio_excerpts:

        ratio = librosa.feature.tempogram_ratio(y=audio_excerpt, sr=audio_sample_rate, hop_length=hop_length, win_length=win_length)
        ratio = ratio.flatten()

        tempogram_ratio.append(ratio)

    tempogram_ratio = np.stack(tempogram_ratio, axis=0)

    return tempogram_ratio
