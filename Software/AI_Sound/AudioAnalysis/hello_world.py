# ------------------------
# Imports
# ------------------------

import sys
import numpy as np
import scipy, sklearn
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd

# ----------------------------
# Audio File Settings
# ----------------------------

audio_file_path = "data/audio/hitting_metal.wav"
audio_sample_rate = 48000
audio_excerpt_sec = [ 0.0, 2.0 ]

# ----------------------------
# Load Audio
# ----------------------------

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate, mono=True)

if isinstance(audio_excerpt_sec, list) and len(audio_excerpt_sec) == 2: 
    audio_waveform = audio_waveform[ int(audio_excerpt_sec[0] * audio_sample_rate) : int(audio_excerpt_sec[1] * audio_sample_rate) ]

# ----------------------------
# Play Audio
# ----------------------------

sd.play(audio_waveform, samplerate=audio_sample_rate)

# ----------------------------
# Plot Audio Waveform
# ----------------------------

plt.title('Waveform')
plt.plot(audio_waveform)
plt.show()

# ----------------------------
# Extract Audio Buffer
# ----------------------------

audio_buffer_size = 1024
audio_buffer = audio_waveform[:audio_buffer_size]

plt.title('Audio Buffer')
plt.plot(audio_buffer)
plt.show()

# ----------------------------
# Create Amplitude Envelope
# ----------------------------

audio_amplitude_envelope = np.hanning(audio_buffer_size)

plt.title('Amplitude Envelope')
plt.plot(audio_amplitude_envelope)
plt.show()

# ----------------------------
# Create Windowed Audio Buffer
# ----------------------------

audio_buffer_windowed = audio_buffer * audio_amplitude_envelope

plt.title('Audio Buffer Windowed')
plt.plot(audio_buffer_windowed)
plt.show()

# ----------------------------
# Calculate Fourier Transform
# ----------------------------

stft_frame = librosa.stft(
    y=audio_buffer_windowed,
    n_fft=len(audio_buffer_windowed),
    hop_length=len(audio_buffer_windowed),
    win_length=len(audio_buffer_windowed),
    center=False
)

# magnitude of spectrum
plt.title('Audio Spectrum Magnitude')
plt.plot(np.abs(stft_frame))
plt.show()

# phase of spectrum
plt.title('Audio Spectrum Phase')
plt.plot(np.angle(stft_frame))
plt.show()

# ----------------------------
# Calculate Audio Features
# ----------------------------

def plot_waveform_and_feature(title, audio_waveform, audio_feature, audio_sample_rate, hop_length=512):
    audio_waveform = np.asarray(audio_waveform)
    audio_feature = np.asarray(audio_feature)

    duration = len(audio_waveform) / audio_sample_rate
    time_wave = np.linspace(0, duration, num=len(audio_waveform))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 7), sharex="col",
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": [1, 0.03], "wspace": 0.02}
    )
    (ax_wave, ax_wave_spacer), (ax_feat, ax_cbar) = axes
    ax_wave_spacer.axis("off")
    fig.suptitle(title)

    ax_wave.plot(time_wave, audio_waveform, linewidth=0.6, color="steelblue")
    ax_wave.set(ylabel="Amplitude")

    if audio_feature.ndim == 1:
        time_feat = np.arange(len(audio_feature)) * hop_length / audio_sample_rate
        ax_feat.plot(time_feat, audio_feature, color="darkorange")
        ax_feat.set(xlabel="Time (s)", ylabel="Feature value")
        ax_cbar.axis("off")
        feat_duration = time_feat[-1] if len(time_feat) else duration
    else:
        n_frames = audio_feature.shape[1]
        feat_duration = n_frames * hop_length / audio_sample_rate
        img = ax_feat.imshow(
            audio_feature, origin="lower", aspect="auto", cmap="magma",
            extent=[0, feat_duration, 0, audio_feature.shape[0]]
        )
        ax_feat.set(xlabel="Time (s)", ylabel="Bin")
        fig.colorbar(img, cax=ax_cbar)

    # Use the shorter of the two spans so neither axis shows empty background
    shared_max = min(duration, feat_duration)
    ax_wave.set_xlim(0, shared_max)

    fig.tight_layout()
    plt.show()

# root-mean-square (RMS)

frame_length = 2048
hop_length=512
center=False

rms = librosa.feature.rms(y=audio_waveform, frame_length=frame_length, hop_length=hop_length, center=center)
rms = rms.flatten()

plot_waveform_and_feature("root mean square", audio_waveform, rms, audio_sample_rate, hop_length)

# power spectrum

frame_length=2048
hop_length=512
center=False

stft = librosa.stft(y=audio_waveform, n_fft=frame_length, hop_length=hop_length, center=center)
power_spectrum = np.abs(stft) ** 2
power_spectrum = librosa.power_to_db(power_spectrum, ref=np.max)

plot_waveform_and_feature("power spectrogram", audio_waveform, power_spectrum, audio_sample_rate, hop_length)

# zero crossing rate

frame_length = 2048
hop_length=512
center=False

zcr = librosa.feature.zero_crossing_rate(y=audio_waveform, frame_length=frame_length, hop_length=hop_length, center=center)
zcr = zcr.flatten()

plot_waveform_and_feature("zero crossing rate", audio_waveform, zcr, audio_sample_rate, hop_length)

# chroma stft

frame_length = 2048
hop_length=512
center=False

chroma_stft = librosa.feature.chroma_stft(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)

plot_waveform_and_feature("chroma stft", audio_waveform, chroma_stft, audio_sample_rate, hop_length)

# chroma cqt

frame_length = 2048
hop_length=512
fmin=None
n_octaves=5
bins_per_octave=12

cqt = librosa.feature.chroma_cqt(y=audio_waveform, sr=audio_sample_rate, hop_length=hop_length, fmin=fmin, n_octaves=n_octaves, bins_per_octave=bins_per_octave)

plot_waveform_and_feature("chroma cqt", audio_waveform, cqt, audio_sample_rate, hop_length)

# chroma cens

frame_length = 2048
hop_length=512
fmin=None
n_octaves=5
bins_per_octave=12

cens = librosa.feature.chroma_cens(y=audio_waveform, sr=audio_sample_rate, hop_length=hop_length,fmin=fmin, n_octaves=n_octaves, bins_per_octave=bins_per_octave)

plot_waveform_and_feature("chroma cens", audio_waveform, cens, audio_sample_rate, hop_length)

# chroma vqt

frame_length = 2048
hop_length=512

vqt = librosa.feature.chroma_vqt(y=audio_waveform, sr=audio_sample_rate, intervals="ji5", hop_length=hop_length)

plot_waveform_and_feature("chroma vqt", audio_waveform, vqt, audio_sample_rate, hop_length)

# tonal centroid features

frame_length = 2048
hop_length=512
center=False

chroma = librosa.feature.chroma_stft(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
tnz = librosa.feature.tonnetz(y=audio_waveform, sr=audio_sample_rate, chroma=chroma)

plot_waveform_and_feature("tonal centroid features", audio_waveform, tnz, audio_sample_rate, hop_length)

# mel spectrogram

frame_length = 2048
hop_length=512
center=False

mel = librosa.feature.melspectrogram(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)

plot_waveform_and_feature("mel spectrogram", audio_waveform, mel, audio_sample_rate, hop_length)

# mel-frequency cepstral coefficients

frame_length = 2048
hop_length=512
center=False

mfcc = librosa.feature.mfcc(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)

plot_waveform_and_feature("mel-frequency cepstral coefficients", audio_waveform, mfcc, audio_sample_rate, hop_length)

# spectral centroid

frame_length = 2048
hop_length=512
center=False

spectral_centroid = librosa.feature.spectral_centroid(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
spectral_centroid = spectral_centroid.flatten()

plot_waveform_and_feature("spectral centroid", audio_waveform, spectral_centroid, audio_sample_rate, hop_length)

# spectral bandwidth

frame_length = 2048
hop_length=512
center=False

spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
spectral_bandwidth = spectral_bandwidth.flatten()

plot_waveform_and_feature("spectral bandwidth", audio_waveform, spectral_bandwidth, audio_sample_rate, hop_length)

# spectral contrast

frame_length = 2048
hop_length=512
center=False

spectral_contrast = librosa.feature.spectral_contrast(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
spectral_contrast = spectral_contrast.flatten()

plot_waveform_and_feature("spectral contrast", audio_waveform, spectral_contrast, audio_sample_rate, hop_length)

# spectral flatness

frame_length = 2048
hop_length=512
center=False

spectral_flatness = librosa.feature.spectral_flatness(y=audio_waveform, n_fft=frame_length, hop_length=hop_length, center=center)
spectral_flatness = spectral_flatness.flatten()

plot_waveform_and_feature("spectral flatness", audio_waveform, spectral_flatness, audio_sample_rate, hop_length)

# spectral rolloff

frame_length = 2048
hop_length=512
center=False

spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
spectral_rolloff = spectral_rolloff.flatten()

plot_waveform_and_feature("spectral rolloff", audio_waveform, spectral_rolloff, audio_sample_rate, hop_length)

# poly features

frame_length = 2048
hop_length=512
order=1
center=False

poly_features = librosa.feature.poly_features(y=audio_waveform, sr=audio_sample_rate, order=order, n_fft=frame_length, hop_length=hop_length, center=center)
poly_features = poly_features.flatten()

plot_waveform_and_feature("poly features", audio_waveform, poly_features, audio_sample_rate, hop_length)

# onset strength

frame_length = 2048
hop_length=512
center=False

onset_strength = librosa.onset.onset_strength(y=audio_waveform, sr=audio_sample_rate, n_fft=frame_length, hop_length=hop_length, center=center)
onset_strength = onset_strength.flatten()

plot_waveform_and_feature("onset strength", audio_waveform, onset_strength, audio_sample_rate, hop_length)

# tempogram

frame_length = 2048
hop_length=512

tempogram = librosa.feature.tempogram(y=audio_waveform, sr=audio_sample_rate, hop_length=hop_length, win_length=frame_length)
tempogram = tempogram.flatten()

plot_waveform_and_feature("temtempogrampo", audio_waveform, tempogram, audio_sample_rate, hop_length)

# fourier tempogram

frame_length = 2048
hop_length=512

fourier_tempogram = librosa.feature.fourier_tempogram(y=audio_waveform, sr=audio_sample_rate, hop_length=hop_length, win_length=frame_length)
fourier_tempogram = fourier_tempogram.flatten()

plot_waveform_and_feature("fourier tempogram", audio_waveform, fourier_tempogram, audio_sample_rate, hop_length)

# tempogram ratio

frame_length = 2048
hop_length=512

tempogram_ratio = librosa.feature.tempogram_ratio(y=audio_waveform, sr=audio_sample_rate, hop_length=hop_length, win_length=frame_length)

plot_waveform_and_feature("tempogram ratio", audio_waveform, tempogram_ratio, audio_sample_rate, hop_length)