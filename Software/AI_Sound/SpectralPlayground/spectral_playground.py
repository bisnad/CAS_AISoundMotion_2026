# -------------------------------------------------------------------------------------------------
# Spectral Playground
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torchaudio as ta
import sounddevice as sd

# -------------------------------------------------------------------------------------------------
# Audio File Settings
# -------------------------------------------------------------------------------------------------

audio_file_path = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
audio_sample_rate = 48000
audio_range_sec = [ 29.0, 36.0 ]
audio_buffer_size = 512

# -------------------------------------------------------------------------------------------------
# Load Audio File
# -------------------------------------------------------------------------------------------------

audio_waveform, _ = ta.load(audio_file_path)
audio_waveform = audio_waveform[:, int(audio_range_sec[0] * audio_sample_rate) : int(audio_range_sec[1] * audio_sample_rate) ]

audio_channel_count = audio_waveform.shape[0]
audio_sample_count = audio_waveform.shape[1]

print("audio_channel_count ", audio_channel_count)
print("audio_sample_count ", audio_sample_count)

# -------------------------------------------------------------------------------------------------
# Create Audio Playback Buffers
# -------------------------------------------------------------------------------------------------

audio_window = torch.from_numpy(np.hanning(audio_buffer_size))
audio_ring_buffer = torch.zeros((audio_buffer_size * 2), dtype=torch.float32)

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - Regular Windowed Audio Playback 
# -------------------------------------------------------------------------------------------------

audio_sample_index = 0

def audio_callback(out_data, frame_count, time_info, status):

    global audio_ring_buffer
    global audio_sample_index
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer * audio_window
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]

audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - FFT and IFFT Audio Playback (with an additional conversion from complex values to polar coordinates and back)
# -------------------------------------------------------------------------------------------------

audio_sample_index = 0

# regular fft and ifft (with an additional conversion from complex values to polar coordinates and back)
def audio_callback(out_data, frame_count, time_info, status):
    
    global audio_ring_buffer
    global audio_sample_index
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        audio_buffer_windowed = audio_buffer * audio_window
    
        audio_spectrum = torch.fft.fft(audio_buffer_windowed.reshape(1, -1))
        audio_spectrum_mag = audio_spectrum.abs()
        audio_spectrum_phase = audio_spectrum.angle()
        audio_spectrum2 = torch.polar(audio_spectrum_mag, audio_spectrum_phase)
        
        audio_buffer_rec = torch.fft.ifft(audio_spectrum2)[0].real
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer_rec
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]

audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - Audio Playback with Magnitude Threshold
# -------------------------------------------------------------------------------------------------

mag_threshold = 5.0
audio_sample_index = 0

# discard phase information
def audio_callback(out_data, frame_count, time_info, status):
    
    global audio_ring_buffer
    global audio_sample_index
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        audio_buffer_windowed = audio_buffer * audio_window
    
        audio_spectrum = torch.fft.fft(audio_buffer_windowed.reshape(1, -1))
        audio_spectrum_mag = audio_spectrum.abs()
        audio_spectrum_mag[torch.logical_and(audio_spectrum_mag>=0, audio_spectrum_mag<=mag_threshold)] = 0.0
        audio_spectrum_phase = audio_spectrum.angle()
        audio_spectrum2 = torch.polar(audio_spectrum_mag, audio_spectrum_phase)

        audio_buffer_rec = torch.fft.ifft(audio_spectrum2)[0].real
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer_rec
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]

audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - Discard Phase Information
# -------------------------------------------------------------------------------------------------

audio_sample_index = 0

# discard phase information
def audio_callback(out_data, frame_count, time_info, status):
    
    global audio_ring_buffer
    global audio_sample_index
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        audio_buffer_windowed = audio_buffer * audio_window
    
        audio_spectrum = torch.fft.fft(audio_buffer_windowed.reshape(1, -1))
        audio_spectrum_mag = audio_spectrum.abs()
        audio_spectrum_phase = torch.zeros_like(audio_spectrum.angle())
        audio_spectrum2 = torch.polar(audio_spectrum_mag, audio_spectrum_phase)

        audio_buffer_rec = torch.fft.ifft(audio_spectrum2)[0].real
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer_rec
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]

audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - Swap Frequency Bins
# -------------------------------------------------------------------------------------------------

rng = np.random.default_rng()
freq_bin_remap = rng.choice(audio_buffer_size//2, size=audio_buffer_size//2, replace=False)
freq_bin_remap = np.concatenate((freq_bin_remap, freq_bin_remap[::-1]))

audio_sample_index = 0

def audio_callback(out_data, frame_count, time_info, status):
    
    global audio_ring_buffer
    global audio_sample_index
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        audio_buffer_windowed = audio_buffer * audio_window
    
        audio_spectrum = torch.fft.fft(audio_buffer_windowed.reshape(1, -1))
        audio_spectrum2 = audio_spectrum[:, freq_bin_remap]

        audio_buffer_rec = torch.fft.ifft(audio_spectrum2)[0].real
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer_rec
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]

audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()

# -------------------------------------------------------------------------------------------------
# Real-Time Audio - Freeze Spectrum
# -------------------------------------------------------------------------------------------------

audio_spectrum = None
audio_spectrum_freeze_factor = 0.9
audio_sample_index = 0

def audio_callback(out_data, frame_count, time_info, status):
    
    global audio_ring_buffer
    global audio_sample_index
    global audio_spectrum
    
    for i in range(2):
        
        audio_buffer = audio_waveform[0, audio_sample_index:audio_sample_index + audio_buffer_size]
        audio_buffer_windowed = audio_buffer * audio_window
    
        new_audio_spectrum = torch.fft.fft(audio_buffer_windowed.reshape(1, -1))
        
        if audio_spectrum is None:
            audio_spectrum = new_audio_spectrum
        else:
            old_audio_spectrum_mag = audio_spectrum.abs()
            new_audio_spectrum_mag = new_audio_spectrum.abs()
            new_audio_spectrum_phase = new_audio_spectrum.angle()
            
            audio_spectrum_mag = old_audio_spectrum_mag * audio_spectrum_freeze_factor + new_audio_spectrum_mag * (1.0 - audio_spectrum_freeze_factor)

            audio_spectrum = torch.polar(audio_spectrum_mag, new_audio_spectrum_phase)

        audio_buffer_rec = torch.fft.ifft(audio_spectrum)[0].real
        
        audio_ring_buffer = torch.roll(audio_ring_buffer, -audio_buffer_size // 2)
        audio_ring_buffer[-audio_buffer_size//2:] = 0.0
        audio_ring_buffer[-audio_buffer_size:] += audio_buffer_rec
        
        audio_sample_index += audio_buffer_size // 2
        
        # loop
        if audio_sample_index >= audio_sample_count - audio_buffer_size:
            audio_sample_index = 0
            
    out_data[:, 0] = audio_ring_buffer[:audio_buffer_size]


audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, blocksize=audio_buffer_size, channels=audio_channel_count,
    callback=audio_callback)

audio_stream.start()

audio_stream.stop()
