import threading
import numpy as np
import sounddevice as sd

"""
AudioSynthesis reconstructs a continuous, loopable waveform for the
currently selected cluster by overlap-adding all audio excerpts belonging
to that cluster (crossfaded with a trapezoidal envelope derived from the
excerpt length/offset), then plays that waveform back in real time.

Revisions vs. the original audio_synthesis.py:

1. Playback engine switched from pyaudio to sounddevice, matching
   audio_receiver.py from the AudioAnalysis tool. Uses a real-time
   sd.OutputStream CALLBACK (PortAudio's own thread) rather than a manual
   buffer passed to a pyaudio stream_callback - functionally similar, but
   this keeps both tools on one consistent audio backend/dependency and
   exposes the same list_output_devices()/set_output_device() interface.

2. update() (renamed internally as _fill_output, called from the stream
   callback) is now guarded by a lock, since cluster_audio can be replaced
   at any time by setClusterLabel()/selectAudioFeature() from the GUI or
   OSC thread while the audio callback is concurrently reading from it -
   the original had a race condition here (no synchronization between the
   audio thread and cluster-audio rebuilding).

3. Rebuilding cluster_audio (_create_cluster_audio) can be relatively slow
   for large clusters; it now happens outside the lock and the result is
   swapped in atomically, so playback is not blocked while the new cluster
   waveform is being assembled - only briefly stalled for the pointer swap.

4. Fixed a looping bug for the case where a cluster's waveform is SHORTER
   than one audio callback buffer (buffer_length >= cluster_length): the
   original always tiled starting from sample 0 on every callback,
   restarting the loop phase each time and producing an audible
   click/discontinuity at every buffer boundary. The tile now starts from
   the current play_sample_index and advances it properly, so playback
   loops seamlessly regardless of how the cluster length compares to the
   callback buffer size.
"""

config = {
    "model": None,
    "audio_excerpts": None,
    "audio_sample_rate": 48000,
    "audio_excerpt_length": 1000,
    "audio_excerpt_offset": 900,
    "output_device": None,
}


class AudioSynthesis():

    def __init__(self, config):
        self.model = config["model"]
        self.audio_sample_rate = config["audio_sample_rate"]
        self.audio_excerpt_length = config["audio_excerpt_length"]
        self.audio_excerpt_offset = config["audio_excerpt_offset"]
        self.output_device = config.get("output_device", None)

        self.audio_excerpt_length_sc = int(self.audio_excerpt_length / 1000 * self.audio_sample_rate)
        self.audio_excerpt_offset_sc = int(self.audio_excerpt_offset / 1000 * self.audio_sample_rate)

        self._create_envelope()

        self.cluster_label = 0
        self.cluster_audio = None

        self.play_sample_index = 0

        self._audio_lock = threading.Lock()
        self._output_stream = None
        self._running = threading.Event()

    def _create_envelope(self):

        overlap_sc = self.audio_excerpt_length_sc - self.audio_excerpt_offset_sc

        if overlap_sc <= 0:
            # no overlap between consecutive excerpts - use a flat envelope
            self.audio_window_envelope = np.ones(self.audio_excerpt_length_sc, dtype=np.float32)
            return

        env_part1 = np.linspace(0.0, 1.0, overlap_sc)
        env_part2 = np.ones(max(self.audio_excerpt_length_sc - 2 * overlap_sc, 0), dtype=np.float32)
        env_part3 = np.linspace(1.0, 0.0, overlap_sc)

        self.audio_window_envelope = np.concatenate((env_part1, env_part2, env_part3)).astype(np.float32)

    def _create_cluster_audio(self, label):
        """
        Builds the overlap-added waveform for the given cluster label.
        Returns None if the cluster has no excerpts. Does not touch
        self.cluster_audio directly - the caller is responsible for
        atomically swapping it in under the lock.
        """
        cluster_audio_excerpts = self.model.get_cluster_audio_excerpts(label)

        if cluster_audio_excerpts is None or cluster_audio_excerpts.shape[0] == 0:
            return None

        excerpt_count = cluster_audio_excerpts.shape[0]

        audio_cluster_sc = self.audio_excerpt_length_sc + self.audio_excerpt_offset_sc * (excerpt_count - 1)
        cluster_audio = np.zeros(audio_cluster_sc, dtype=np.float32)

        insert_index = 0
        for audio_excerpt in cluster_audio_excerpts:
            cluster_audio[insert_index:insert_index + self.audio_excerpt_length_sc] += (
                audio_excerpt * self.audio_window_envelope
            )
            insert_index += self.audio_excerpt_offset_sc

        # normalize to avoid clipping from overlapping envelope gains
        peak = np.max(np.abs(cluster_audio)) if cluster_audio.size > 0 else 0.0
        if peak > 1.0:
            cluster_audio = cluster_audio / peak

        return cluster_audio

    def setClusterLabel(self, label):

        label_count = self.model.get_label_count()
        if label_count == 0:
            return

        label = int(max(0, min(label, label_count - 1)))

        new_cluster_audio = self._create_cluster_audio(label)

        with self._audio_lock:
            self.cluster_label = label
            self.cluster_audio = new_cluster_audio
            self.play_sample_index = 0

    def selectAudioFeature(self, feature_name):

        self.model.select_audio_feature(feature_name)
        self.setClusterLabel(0)

    def get_cluster_label(self):
        return self.cluster_label

    # ---------------- real-time playback ----------------

    def _fill_output(self, audio_buffer):
        """
        Fills audio_buffer (1D float32 array) with the next samples of the
        current cluster's waveform, looping seamlessly regardless of
        whether the cluster is longer or shorter than the buffer. Leaves
        the buffer as silence if no cluster audio is available yet.
        """
        with self._audio_lock:
            cluster_audio = self.cluster_audio
            play_index = self.play_sample_index

            if cluster_audio is None:
                audio_buffer[:] = 0.0
                return

            buffer_length = audio_buffer.shape[0]
            cluster_length = cluster_audio.shape[0]

            if buffer_length >= cluster_length:
                # cluster shorter than (or equal to) the callback buffer -
                # tile it starting from the CURRENT play_index so playback
                # continues seamlessly across callback boundaries instead
                # of restarting the loop phase every callback
                reps = (buffer_length + play_index) // cluster_length + 2
                tiled = np.tile(cluster_audio, reps)
                audio_buffer[:] = tiled[play_index:play_index + buffer_length]
                self.play_sample_index = (play_index + buffer_length) % cluster_length
                return

            if play_index + buffer_length <= cluster_length:
                audio_buffer[:] = cluster_audio[play_index:play_index + buffer_length]
                self.play_sample_index = play_index + buffer_length
            else:
                part1_length = cluster_length - play_index
                audio_buffer[:part1_length] = cluster_audio[play_index:cluster_length]

                part2_length = buffer_length - part1_length
                audio_buffer[part1_length:] = cluster_audio[0:part2_length]

                self.play_sample_index = part2_length

    def _output_callback(self, outdata, frames, time_info, status):
        buf = np.zeros(frames, dtype=np.float32)
        self._fill_output(buf)
        outdata[:, 0] = buf

    def set_output_device(self, device_id):
        self.output_device = device_id
        was_running = self._running.is_set()
        if was_running:
            self.stop()
            self.start()

    @staticmethod
    def list_output_devices():
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]

    def start(self):
        if self._running.is_set():
            return

        self._running.set()
        self._output_stream = sd.OutputStream(
            samplerate=self.audio_sample_rate,
            channels=1,
            device=self.output_device,
            dtype="float32",
            callback=self._output_callback,
        )
        self._output_stream.start()

    def stop(self):
        self._running.clear()
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
