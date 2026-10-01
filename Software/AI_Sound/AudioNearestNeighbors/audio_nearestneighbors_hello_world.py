"""
Audio Nearest Neighbors - Introductory Code Example
"""

"""
Imports
"""

import os
import numpy as np
import librosa
import torch
import torchaudio
import analysis as aa
from matplotlib import pyplot as plt
import wave
import time

"""
Audio Settings
"""

audio_file_path = "data/audio/Night_and_Day_by_Virginia_Woolf_48khz_excerpt.wav"
audio_sample_rate = 48000
audio_channel_count = 1
audio_range_sec = [ 10.0, 70.0 ]
audio_excerpt_length = 1.0 # in secs
audio_excerpt_offset = 0.5 # in secs
#audio_excerpt_length = 0.05 # in secs
#audio_excerpt_offset = 0.025 # in secs

audio_feature_names = ["root mean square", "mfcc"]

"""
Load Audio
"""

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate)

if audio_channel_count > 1:
    audio_waveform = audio_waveform[0]

audio_waveform = audio_waveform[int(audio_range_sec[0] * audio_sample_rate):int(audio_range_sec[1] * audio_sample_rate)]
audio_waveform_sc = audio_waveform.shape[0]

"""
Create Audio Excerpts
"""

audio_excerpts = []

audio_excerpt_length_sc = int(audio_excerpt_length * audio_sample_rate)
audio_excerpt_offset_sc = int(audio_excerpt_offset * audio_sample_rate)

for sI in range(0, audio_waveform_sc - audio_excerpt_length_sc, audio_excerpt_offset_sc):
    
    audio_excerpt = audio_waveform[sI:sI + audio_excerpt_length_sc]
    audio_excerpts.append(audio_excerpt)
    
audio_waveforms = np.stack(audio_excerpts, axis=0)
        
"""
Calculate Audio Features
"""

audio_features = {}

audio_features["waveform"] = audio_waveforms
if "root mean square" in audio_feature_names:
    audio_features["root mean square"] = aa.rms(audio_waveforms)
if "chroma stft" in audio_feature_names:
    audio_features["chroma stft"] = aa.chroma_stft(audio_waveforms, audio_sample_rate)
if "chroma cqt" in audio_feature_names:
    audio_features["chroma cqt"] = aa.chroma_cqt(audio_waveforms, audio_sample_rate)
if "chroma cens" in audio_feature_names:
    audio_features["chroma cens"] = aa.chroma_cens(audio_waveforms, audio_sample_rate)
if "chroma vqt" in audio_feature_names:
    audio_features["chroma vqt"] = aa.chroma_vqt(audio_waveforms, audio_sample_rate)
if "mel spectrogram" in audio_feature_names:
    audio_features["mel spectrogram"] = aa.mel_spectrogram(audio_waveforms, audio_sample_rate)
if "mfcc" in audio_feature_names:
    audio_features["mfcc"] = aa.mfcc(audio_waveforms, audio_sample_rate)
if "spectral centroid" in audio_feature_names:
    audio_features["spectral centroid"] = aa.spectral_centroid(audio_waveforms, audio_sample_rate)
if "spectral bandwidth" in audio_feature_names:
    audio_features["spectral bandwidth"] = aa.spectral_bandwidth(audio_waveforms, audio_sample_rate)
if "spectral contrast" in audio_feature_names:
    audio_features["spectral contrast"] = aa.spectral_contrast(audio_waveforms, audio_sample_rate)
if "spectral flatness" in audio_feature_names:
    audio_features["spectral flatness"] = aa.spectral_flatness(audio_waveforms)
if "spectral rolloff" in audio_feature_names:
    audio_features["spectral rolloff"] = aa.spectral_rolloff(audio_waveforms, audio_sample_rate)
if "tempo" in audio_feature_names:
    audio_features["tempo"] = aa.tempo(audio_waveforms, audio_sample_rate)
if "tempogram" in audio_feature_names:
    audio_features["tempogram"] = aa.tempogram(audio_waveforms, audio_sample_rate)
if "tempogram ratio" in audio_feature_names:
    audio_features["tempogram ratio"] = aa.tempogram_ratio(audio_waveforms, audio_sample_rate)

"""
Normalise Audio Features
"""

for audio_feature_name in list(audio_features.keys()):
    
    #print(audio_feature_name)
    
    audio_feature = audio_features[audio_feature_name]
    
    audio_feature_mean = np.mean(audio_feature)
    audio_feature_std = np.std(audio_feature)
    
    audio_feature_norm = (audio_feature - audio_feature_mean) / audio_feature_std
    
    #print("audio_feature_norm s ", audio_feature_norm.shape)

    audio_features[audio_feature_name + " norm"] = audio_feature_norm
    
"""
Find Nearest Neighbors
"""

# gather all waveforms and audio features
audio_features_proc = []
for audio_feature_name in audio_feature_names:
    audio_norm_feature_name = audio_feature_name + " norm"
    audio_feature = audio_features[audio_norm_feature_name]
    #print("name ", audio_norm_feature_name, " shape ", audio_feature.shape)
    audio_features_proc.append(audio_feature)
    
audio_features_proc = np.concatenate(audio_features_proc, axis=1)
audio_waveforms_proc = np.copy(audio_waveforms)

# select first audio feature to begin search with
nn_current_index = 0
nn_current_waveform = audio_waveforms_proc[nn_current_index]
nn_current_feature = audio_features_proc[nn_current_index]
nn_current_feature = np.expand_dims(nn_current_feature, 0)

# prepare empty waveform to copy waveforms corresponding to nearest features into
nn_element_count = audio_features_proc.shape[0]
gen_waveform_sample_count = 2 * audio_excerpt_length_sc + (nn_element_count - 1) * audio_excerpt_offset_sc
gen_waveform = np.zeros(gen_waveform_sample_count)

# create amplitude enevelope for blending audio waveforms into gen_waveform
audio_sample_overlap = audio_excerpt_length_sc - audio_excerpt_offset_sc
hann_window = torch.hann_window(audio_sample_overlap * 2).numpy()

amplitude_envelope = np.ones([audio_excerpt_length_sc])
amplitude_envelope[:audio_sample_overlap] *= hann_window[:audio_sample_overlap]
amplitude_envelope[-audio_sample_overlap:] *= hann_window[audio_sample_overlap:]

# add first waveform to gen waveform
gen_waveform[:audio_excerpt_length_sc] += nn_current_waveform * amplitude_envelope

# iterate through all neighbors
sI = audio_excerpt_length_sc -  audio_sample_overlap # sample index for waveform insertion in gen waveform

while nn_element_count > 1:
    
    print("nn_element_count ", nn_element_count)

    # search nearest element
    nn_distances = np.linalg.norm(audio_features_proc - nn_current_feature, axis=1)
    k = 2
    nn_indices = nn_distances.argsort()[:k]

    # replace current element with nearest element
    nn_previous_index = nn_current_index
    nn_current_index = nn_indices[1]
    nn_current_waveform = audio_waveforms_proc[nn_current_index]
    nn_current_feature = audio_features_proc[nn_current_index]
    nn_current_feature = np.expand_dims(nn_current_feature, 0)
    
    # blend waveform corresponding to current element into gen waveform
    gen_waveform[sI:sI + audio_excerpt_length_sc] += nn_current_waveform * amplitude_envelope
    
    # remove previous element (via boolean mask - equivalent to, but clearer
    # than, the original's three-way slice/concatenate special-casing)
    keep_mask = np.ones(audio_waveforms_proc.shape[0], dtype=bool)
    keep_mask[nn_previous_index] = False
    audio_waveforms_proc = np.copy(audio_waveforms_proc[keep_mask])
    audio_features_proc = np.copy(audio_features_proc[keep_mask])

    # remap nn_current_index into the just-shrunk array: everything that used
    # to sit after nn_previous_index has moved one slot to the left
    if nn_current_index > nn_previous_index:
        nn_current_index -= 1

    nn_element_count -= 1
    sI += audio_excerpt_length_sc - audio_sample_overlap

"""
Save Generated Waveform
"""
os.makedirs("results/audio", exist_ok=True)

gen_waveform = torch.tensor(gen_waveform)
gen_waveform = gen_waveform.unsqueeze(0)
torchaudio.save("results/audio/gen_waveform.wav", gen_waveform, audio_sample_rate)
