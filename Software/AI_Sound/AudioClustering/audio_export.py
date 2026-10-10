"""
audio_export.py

Exports the content of every cluster as one audio file, like the simple
clustering script did: for each cluster, its excerpts (kept in their original
temporal order) are overlap-added into one continuous waveform and written as

    audio_cluster_<label, 5 digits>_excerpts_<excerpt count, 5 digits>.wav

The waveform is exactly what you hear when the cluster is selected for
playback (AudioSynthesis._create_cluster_audio: crossfade envelope derived from
excerpt length and overlap, peak-normalized only if it would clip).
"""

import os
import numpy as np
import soundfile as sf


def export_clusters(model, synthesis, folder, subtype="PCM_24", progress=None):
    """
    Writes one WAV per non-empty cluster into folder. Returns a list of
    (file_path, excerpt_count, duration_seconds). progress(done, total) is
    called after each cluster if given.
    """
    os.makedirs(folder, exist_ok=True)

    sizes = model.get_cluster_sizes()
    labels = sorted(sizes.keys())
    sample_rate = int(synthesis.audio_sample_rate)

    written = []
    for done, label in enumerate(labels, start=1):
        cluster_audio = synthesis._create_cluster_audio(label)

        if cluster_audio is not None and cluster_audio.shape[0] > 0:
            file_path = os.path.join(
                folder, "audio_cluster_{:05d}_excerpts_{:05d}.wav".format(label, sizes[label]))
            sf.write(file_path, cluster_audio.astype(np.float32), sample_rate, subtype=subtype)
            written.append((file_path, sizes[label], cluster_audio.shape[0] / sample_rate))

        if progress is not None:
            progress(done, len(labels))

    return written
