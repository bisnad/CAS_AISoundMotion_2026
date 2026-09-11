import threading
import numpy as np
import soundfile as sf
import sounddevice as sd

"""
NNSynthesis plays back the growing output waveform from a
NearestNeighborModel in real time, as it is being generated - the play
cursor advances through whatever has been written so far and never reads
past what nn_model.step() has already produced, so the user hears the
nearest-neighbor chain build up live rather than waiting for the entire
sequence to finish (which is how the original, offline script worked).

Uses a real-time sounddevice.OutputStream CALLBACK (matching
audio_receiver.py / audio_synthesis.py from the AudioAnalysis /
AudioClustering tools), reading directly from the model's output buffer
under a lock. If playback catches up to the point generation has reached
(i.e. the model is producing excerpts slower than real time - possible for
expensive feature/search configurations), the callback outputs silence for
the remainder of that block rather than reading uninitialized data or
stalling, and simply continues once more audio has been generated.
"""

config = {
    "model": None,
    "output_device": None,
}


class NNSynthesis():

    def __init__(self, config):
        self.model = config["model"]
        self.output_device = config.get("output_device", None)

        self._play_cursor = 0
        self._lock = threading.Lock()

        self._output_stream = None
        self._running = threading.Event()

    def reset_playback(self):
        with self._lock:
            self._play_cursor = 0

    def get_output_callback(self, outdata, frames, time_info, status):
        with self._lock:
            available = self.model.get_output_waveform()
            available_length = available.shape[0]

            end_cursor = self._play_cursor + frames

            buf = np.zeros(frames, dtype=np.float32)

            if self._play_cursor < available_length:
                copy_end = min(end_cursor, available_length)
                copy_length = copy_end - self._play_cursor
                buf[:copy_length] = available[self._play_cursor:copy_end]
                self._play_cursor += copy_length
            # if play_cursor has caught up to what's been generated, the
            # remainder of buf stays silent (zeros) until more is produced

        outdata[:, 0] = buf

    def get_progress_seconds(self):
        with self._lock:
            return self._play_cursor / float(self.model.audio_sample_rate)

    def is_caught_up(self):
        """
        True if playback has reached the end of everything generated so
        far (useful for a GUI to show a "waiting for more audio..." state
        distinct from normal playback).
        """
        with self._lock:
            return self._play_cursor >= self.model.get_output_waveform().shape[0]

    @staticmethod
    def list_output_devices():
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]

    def set_output_device(self, device_id):
        self.output_device = device_id
        was_running = self._running.is_set()
        if was_running:
            self.stop()
            self.start()

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._output_stream = sd.OutputStream(
            samplerate=self.model.audio_sample_rate,
            channels=1,
            device=self.output_device,
            dtype="float32",
            callback=self.get_output_callback,
        )
        self._output_stream.start()

    def stop(self):
        self._running.clear()
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

    def save_to_file(self, file_path):
        """
        Saves everything generated so far (not necessarily the full,
        finished sequence) to a WAV file at the given path.
        """
        waveform = self.model.get_output_waveform()
        peak = np.max(np.abs(waveform)) if waveform.size > 0 else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        sf.write(file_path, waveform, self.model.audio_sample_rate)
