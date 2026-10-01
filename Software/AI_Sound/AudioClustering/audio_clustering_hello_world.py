"""
Audio Clustering - Introductory Code Example
"""

"""
Imports
"""

import os
import numpy as np
import librosa
import torch
import torchaudio
import soundfile as sf
import analysis as aa
from matplotlib import pyplot as plt
import wave
import time

from sklearn.cluster import KMeans

"""
Audio Settings
"""

audio_file_path = "data/audio"
audio_file_extensions = ["wav", "aiff", "aif"] 
audio_sample_rate = 48000
audio_sample_count = audio_sample_rate // 1
audio_sample_offset = audio_sample_count // 2

audio_feature_names = ["root mean square", "mfcc"]

"""
Load Audio
"""

audio_waveforms_full = []

for root, _, fnames in sorted(os.walk(audio_file_path, followlinks=True)):
    for fname in sorted(fnames):
        
        path = root + "/" + fname
        
        print("path ", path)
        
        audio_waveform, _ = librosa.load(path, sr=audio_sample_rate)

        print("waveform s ", audio_waveform.shape)
        
        audio_waveforms_full.append(audio_waveform)

"""
Create Audio Excerpts
"""

audio_waveform_excerpts = []

for waveform_full in audio_waveforms_full:
    waveform_sample_count = waveform_full.shape[0]
    for sI in range(0, waveform_sample_count - audio_sample_count, audio_sample_offset):
        waveform_except = waveform_full[sI:sI + audio_sample_count]
        audio_waveform_excerpts.append(waveform_except)
        
audio_waveforms = np.stack(audio_waveform_excerpts, axis=0)
        
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
Combine Audio Features
"""

audio_features_proc = []
for audio_feature_name in audio_feature_names:
    audio_norm_feature_name = audio_feature_name + " norm"
    audio_feature = audio_features[audio_norm_feature_name]
    #print("name ", audio_norm_feature_name, " shape ", audio_feature.shape)
    audio_features_proc.append(audio_feature)
    
audio_features_proc = np.concatenate(audio_features_proc, axis=1)

"""
KMeans Clustering
"""

cluster_count = 10
random_state = 170

km = KMeans(n_clusters=cluster_count, n_init= "auto", random_state = random_state)
labels =  km.fit_predict(audio_features_proc)

"""
Export Concatenated Audio Per Cluster
"""

audio_amp_envelope = np.hanning(audio_sample_count)

def concatenate_excerpts_with_crossfade(excerpt_list, hop, window):
    """
    Overlap-adds a list of equal-length excerpts (in order) into one
    continuous waveform, applying `window` to each excerpt and hopping
    forward by `hop` samples between excerpts. Returns the normalized
    concatenated waveform.
    """
    excerpt_len = excerpt_list[0].shape[0]
    n_excerpts = len(excerpt_list)
    total_len = excerpt_len + hop * (n_excerpts - 1)

    buffer = np.zeros(total_len, dtype=np.float64)
    window_sum = np.zeros(total_len, dtype=np.float64)

    for i, excerpt in enumerate(excerpt_list):
        start = i * hop
        buffer[start:start + excerpt_len] += excerpt * window
        window_sum[start:start + excerpt_len] += window

    # avoid divide-by-zero in any (shouldn't normally occur) zero-weight gaps
    window_sum[window_sum < 1e-8] = 1.0

    return (buffer / window_sum).astype(np.float32)

os.makedirs("results/audio", exist_ok=True)

for cluster_index in range(cluster_count):
    # keep excerpts in their original (temporal/file) order within the cluster,
    # so the concatenated result plays back as a coherent sequence rather than
    # in a randomly shuffled order
    cluster_excerpt_indices = np.where(labels == cluster_index)[0]

    if cluster_excerpt_indices.shape[0] == 0:
        continue

    cluster_excerpts = [audio_waveforms[i] for i in cluster_excerpt_indices]

    concatenated_waveform = concatenate_excerpts_with_crossfade(
        cluster_excerpts, audio_sample_offset, audio_amp_envelope
    )

    audio_file_name = "results/audio/audio_cluster_{:05d}_excerpts_{:05d}.wav".format(
        cluster_index, len(cluster_excerpts)
    )
    sf.write(audio_file_name, concatenated_waveform, audio_sample_rate)

    print("wrote ", audio_file_name, " (", len(cluster_excerpts), " excerpts, ",
          concatenated_waveform.shape[0] / audio_sample_rate, " s)")
