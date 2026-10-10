import threading
import numpy as np
import sounddevice as sd

"""
AudioSynthesis reconstructs a continuous, loopable waveform for the currently
selected cluster by overlap-adding all excerpts of that cluster (crossfaded
with an envelope derived from excerpt length/offset) and plays it back in
real time through a sounddevice OutputStream callback.

Revisions in this version:

1. Excerpt length/offset (ms) and the sample rate can change at run time:
   set_excerpt_params(length_ms, offset_ms, sample_rate) recomputes the sample
   counts and the envelope, and restarts the output stream if the sample rate
   changed while playing (previously the stream kept running at the old rate
   after a new file with a different rate was loaded).
2. The envelope now works for any overlap. Before, an overlap larger than half
   the excerpt length produced an envelope longer than the excerpt (shape
   mismatch). Fade length is now min(overlap, length // 2): linear fades with
   an optional flat middle; with no overlap the envelope is flat.
3. selectAudioFeatures(names) selects a COMBINATION of features;
   selectAudioFeature(name) is kept and selects a single one.
4. _create_cluster_audio uses the excerpt length of the actual excerpt array and
   returns None on a mismatch, so a race between data swap and parameter
   update cannot crash the GUI/OSC thread.
"""

config = {
    "model": None,
    "audio_excerpts": None,
    "audio_sample_rate": 48000,
    "audio_excerpt_length": 100,
    "audio_excerpt_offset": 90,
    "output_device": None,
}


class AudioSynthesis():

    def __init__(self, config):
        self.model = config["model"]
        self.output_device = config.get("output_device", None)

        self._audio_lock = threading.Lock()
        self._output_stream = None
        self._running = threading.Event()

        self.cluster_label = 0
        self.cluster_audio = None
        self.play_sample_index = 0

        self._apply_params(config["audio_excerpt_length"], config["audio_excerpt_offset"],
                           config["audio_sample_rate"])

    # ---------------- excerpt parameters ----------------

    def _apply_params(self, length_ms, offset_ms, sample_rate):
        self.audio_excerpt_length = length_ms
        self.audio_excerpt_offset = offset_ms
        self.audio_sample_rate = sample_rate

        self.audio_excerpt_length_sc = max(1, int(round(length_ms / 1000 * sample_rate)))
        self.audio_excerpt_offset_sc = max(1, int(round(offset_ms / 1000 * sample_rate)))

        self._create_envelope()

    def set_excerpt_params(self, length_ms, offset_ms, sample_rate=None):
        """Call after the model has received excerpts of the new length."""
        if sample_rate is None:
            sample_rate = self.audio_sample_rate

        rate_changed = (sample_rate != self.audio_sample_rate)
        was_running = self._running.is_set()
        if rate_changed and was_running:
            self.stop()

        self._apply_params(length_ms, offset_ms, sample_rate)

        if rate_changed and was_running:
            self.start()

    def _create_envelope(self):
        length_sc = self.audio_excerpt_length_sc
        overlap_sc = length_sc - self.audio_excerpt_offset_sc

        if overlap_sc <= 0:
            self.audio_window_envelope = np.ones(length_sc, dtype=np.float32)
            return

        fade_sc = max(1, min(overlap_sc, length_sc // 2))
        fade_in = np.linspace(0.0, 1.0, fade_sc, dtype=np.float32)
        fade_out = fade_in[::-1]
        flat = np.ones(length_sc - 2 * fade_sc, dtype=np.float32)

        self.audio_window_envelope = np.concatenate((fade_in, flat, fade_out)).astype(np.float32)

    # ---------------- cluster audio ----------------

    def _create_cluster_audio(self, label):
        """Overlap-adds the excerpts of a cluster. Returns None if empty/inconsistent."""
        cluster_audio_excerpts = self.model.get_cluster_audio_excerpts(label)

        if cluster_audio_excerpts is None or cluster_audio_excerpts.shape[0] == 0:
            return None

        length_sc = cluster_audio_excerpts.shape[1]
        offset_sc = self.audio_excerpt_offset_sc
        envelope = self.audio_window_envelope

        if envelope.shape[0] != length_sc:
            return None

        excerpt_count = cluster_audio_excerpts.shape[0]
        cluster_audio = np.zeros(length_sc + offset_sc * (excerpt_count - 1), dtype=np.float32)

        insert_index = 0
        for audio_excerpt in cluster_audio_excerpts:
            cluster_audio[insert_index:insert_index + length_sc] += audio_excerpt * envelope
            insert_index += offset_sc

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

    def selectAudioFeatures(self, feature_names):
        """Cluster on a combination of features."""
        self.model.set_selected_features(feature_names)
        self.setClusterLabel(0)

    def selectAudioFeature(self, feature_name):
        self.selectAudioFeatures([feature_name])

    def get_cluster_label(self):
        return self.cluster_label

    # ---------------- real-time playback ----------------

    def _fill_output(self, audio_buffer):
        with self._audio_lock:
            cluster_audio = self.cluster_audio
            play_index = self.play_sample_index

        if cluster_audio is None:
            audio_buffer[:] = 0.0
            return

        buffer_length = audio_buffer.shape[0]
        cluster_length = cluster_audio.shape[0]
        play_index = play_index % cluster_length

        if buffer_length >= cluster_length:
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
