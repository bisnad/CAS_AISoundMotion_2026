# -------------------------------------------------------------------------------------------------
# Audio Representation
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torchaudio as ta
import sounddevice as sd
import librosa
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------------------------------
# Audio File Settings
# -------------------------------------------------------------------------------------------------

audio_file_path = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
audio_sample_rate = 48000
audio_range_sec = [ 29.0, 36.0 ]

# -------------------------------------------------------------------------------------------------
# Load Audio File
# -------------------------------------------------------------------------------------------------

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate)

print("audio_waveform s ", audio_waveform.shape)

# 1D to 2D array
if len(audio_waveform.shape) == 1:
    audio_waveform = np.expand_dims(audio_waveform, 0)

print("audio_waveform s ", audio_waveform.shape)

# Audio Excerpt 
if None not in audio_range_sec:
    audio_waveform = audio_waveform[:, int(audio_range_sec[0] * audio_sample_rate) : int(audio_range_sec[1] * audio_sample_rate) ]

print("audio_waveform s ", audio_waveform.shape)

# -------------------------------------------------------------------------------------------------
# Play Audio
# -------------------------------------------------------------------------------------------------

sd.play(audio_waveform.T.astype(np.float32), audio_sample_rate)

sd.stop()

# -------------------------------------------------------------------------------------------------
# Plot Audio Waveform
# -------------------------------------------------------------------------------------------------

plt.title('Waveform')
plt.plot(audio_waveform[0])
plt.show()

# -------------------------------------------------------------------------------------------------
# Create Audio Buffer
# -------------------------------------------------------------------------------------------------

audio_buffer_size = 1024

audio_buffer = audio_waveform[0, :audio_buffer_size]

plt.title('Audio Buffer')
plt.plot(audio_buffer)
plt.show()

# -------------------------------------------------------------------------------------------------
# Create Amplitude Envelope
# -------------------------------------------------------------------------------------------------

audio_window = np.hanning(audio_buffer_size)

plt.title('Hanning Window')
plt.plot(audio_window)
plt.show()

# -------------------------------------------------------------------------------------------------
# Create Windowed Audio Buffer
# -------------------------------------------------------------------------------------------------

audio_buffer_windowed = audio_buffer * audio_window

plt.title('Audio Buffer Windowed')
plt.plot(audio_buffer_windowed)
plt.show()

# -------------------------------------------------------------------------------------------------
# Calculate Audio Spectrum
# -------------------------------------------------------------------------------------------------

audio_buffer_windowed = torch.from_numpy(audio_buffer_windowed)

audio_spectrum = torch.fft.fft(audio_buffer_windowed)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# magnitude of spectrum
ax1.set_title('Audio Spectrum Magnitude')
ax1.plot(audio_spectrum[:audio_buffer_size//2].abs().numpy())

# phase of spectrum
ax2.set_title('Audio Spectrum Phase')
ax2.plot(audio_spectrum[:audio_buffer_size//2].angle().numpy())

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------
# Reconstruct Audio Buffer from Spectrum
# -------------------------------------------------------------------------------------------------

audio_buffer_rec = torch.fft.ifft(audio_spectrum).real

plt.title('Reconstructed Audio Buffer')
plt.plot(audio_buffer_rec.numpy())
plt.show()

# -------------------------------------------------------------------------------------------------
# Calculate Audio Spectrogram
# see: [https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html](https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html)
# -------------------------------------------------------------------------------------------------

nFFT = 1024

audio_spectrogram = ta.transforms.Spectrogram(n_fft=nFFT)(torch.from_numpy(audio_waveform))

plt.figure(figsize=(10, 4))
plt.imshow(audio_spectrogram.squeeze().log2().numpy(), aspect='auto', origin='lower')
plt.title('Audio Spectrogram (log2)')
plt.xlabel('Frame')
plt.ylabel('FFT bins')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------
# Calculate MEL Spectrogram
# -------------------------------------------------------------------------------------------------

nFFT = 1024
nMels = 128

# Create MelSpectrogram transform
mel_transform = ta.transforms.MelSpectrogram(
    sample_rate=audio_sample_rate,
    n_fft=nFFT,
    hop_length=nFFT // 2,
    n_mels=nMels
)

# Compute Mel spectrogram (shape: [channels, n_mels, time])
mel_spec = mel_transform(torch.from_numpy(audio_waveform))

plt.figure(figsize=(10, 4))
plt.imshow(mel_spec.squeeze().log2().numpy(), aspect='auto', origin='lower')
plt.title('Mel Spectrogram (log2)')
plt.xlabel('Frame')
plt.ylabel('Mel bins')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
