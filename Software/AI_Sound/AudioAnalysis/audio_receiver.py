import threading
import time
import numpy as np
import soundfile as sf
import sounddevice as sd

"""
AudioReceiver plays the role of motion_receiver.MotionReceiver for the audio
pipeline: it owns a pre-allocated numpy ring buffer ("data") that is
continuously filled with the most recent audio samples, either from a file
(played back in real time) or from a live microphone input stream. The
pipeline reads from this buffer the same way MotionPipeline reads from
oscReceiver.data.

PERFORMANCE NOTE (playback clicks): audio output now uses a real-time
sounddevice.OutputStream CALLBACK, not a blocking stream.write() call from
a plain Python thread. A blocking write() driven by time.sleep() timing is
subject to GIL contention and OS scheduling jitter from the rest of the
Python process (analysis, OSC sending, Qt event loop) - any delay there
directly starves the output stream and causes audible clicks/underruns.
With a callback, PortAudio's own dedicated, precisely-timed thread pulls
audio data on demand from a small thread-safe ring buffer; a separate
"filler" thread keeps that ring buffer topped up from the file. This
decouples playback timing completely from analysis/OSC/GUI timing.

NOTE on sample-rate handling: when mode == "file", the file's own sample
rate always wins (no resampling is performed). buffer_seconds / block_seconds
are expressed in SECONDS and converted to samples once the true sample rate
is known.
"""

config = {
    "mode": "file",             # "file" or "mic"
    "file_path": None,          # path to audio file (mode == "file")
    "loop": True,                # loop playback within [region_start, region_end]
    "input_device": None,        # input device index (mode == "mic"), None = default
    "output_device": None,       # output device index for file playback, None = default
    "sample_rate": 22050,        # only used as fallback / for mode == "mic"
    "channels": 1,
    "buffer_seconds": 4.0,       # length of rolling analysis buffer, in SECONDS
    "block_seconds": 1024 / 22050.0,  # analysis-buffer refill granularity, in SECONDS
    "output_ring_seconds": 1.0,  # size of the playback ring buffer feeding the OutputStream callback
}


class _OutputRingBuffer:
    """
    Small single-producer/single-consumer ring buffer of audio samples used
    to decouple the file-reading/filler thread from the real-time
    OutputStream callback. The callback only ever reads; the filler thread
    only ever writes - both under a lock, but critically the callback's
    lock hold time is minimal (a numpy slice copy), so it does not block
    on any librosa/analysis work happening elsewhere in the process.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.filled = 0
        self.lock = threading.Lock()

    def write(self, samples):
        n = samples.shape[0]
        with self.lock:
            free = self.capacity - self.filled
            n = min(n, free)
            if n <= 0:
                return 0
            end = self.write_pos + n
            if end <= self.capacity:
                self.buf[self.write_pos:end] = samples[:n]
            else:
                first = self.capacity - self.write_pos
                self.buf[self.write_pos:] = samples[:first]
                self.buf[:end - self.capacity] = samples[first:n]
            self.write_pos = end % self.capacity
            self.filled += n
            return n

    def read(self, n):
        with self.lock:
            available = min(n, self.filled)
            out = np.zeros(n, dtype=np.float32)
            if available <= 0:
                return out
            end = self.read_pos + available
            if end <= self.capacity:
                out[:available] = self.buf[self.read_pos:end]
            else:
                first = self.capacity - self.read_pos
                out[:first] = self.buf[self.read_pos:]
                out[first:available] = self.buf[:end - self.capacity]
            self.read_pos = end % self.capacity
            self.filled -= available
            return out

    def available_space(self):
        with self.lock:
            return self.capacity - self.filled

    def clear(self):
        with self.lock:
            self.write_pos = 0
            self.read_pos = 0
            self.filled = 0


class AudioReceiver():

    def __init__(self, config):

        self.mode = config["mode"]
        self.file_path = config["file_path"]
        self.loop = config.get("loop", True)
        self.input_device = config.get("input_device", None)
        self.output_device = config.get("output_device", None)
        self.sample_rate = config["sample_rate"]
        self.channels = config["channels"]

        self._lock = threading.Lock()
        self._running = threading.Event()

        self._input_stream = None
        self._output_stream = None
        self._filler_thread = None
        self._sf_handle = None

        self._file_samples = None
        self._file_length = 0

        self._play_head = 0
        self._region_start = 0
        self._region_end = 0
        self._transport_lock = threading.Lock()

        self.buffer_seconds_cfg = config.get("buffer_seconds", 4.0)
        self.block_seconds_cfg = config.get("block_seconds", 1024 / 22050.0)
        self.output_ring_seconds_cfg = config.get("output_ring_seconds", 1.0)

        self.data = [np.zeros((1,), dtype=np.float32)]
        self._output_ring = None

        if self.mode == "file":
            if self.file_path is not None:
                self.load_file(self.file_path)
        elif self.mode == "mic":
            self._configure_buffers()

    # ---------------- device enumeration ----------------

    @staticmethod
    def list_input_devices():
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]

    @staticmethod
    def list_output_devices():
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]

    def set_input_device(self, device_id):
        self.input_device = device_id
        if self.mode == "mic" and self._running.is_set():
            self.stop()
            self.start()

    def set_output_device(self, device_id):
        self.output_device = device_id
        was_running = self._running.is_set()
        if self.mode == "file" and was_running:
            self.stop()
            self.start()

    # ---------------- file loading ----------------

    def load_file(self, file_path):
        was_running = self._running.is_set() and self.mode == "file"
        if was_running:
            self.stop()

        self._sf_handle = sf.SoundFile(file_path)
        self.sample_rate = self._sf_handle.samplerate
        samples = self._sf_handle.read(dtype="float32", always_2d=True)
        self._sf_handle.close()
        self._sf_handle = None

        mono = np.mean(samples, axis=-1).astype(np.float32)

        with self._transport_lock:
            self._file_samples = mono
            self._file_length = mono.shape[0]
            self._play_head = 0
            self._region_start = 0
            self._region_end = self._file_length

        self.file_path = file_path
        self.mode = "file"

        self._configure_buffers()

        if was_running:
            self.start()

    def _configure_buffers(self):
        self.buffer_size = max(int(round(self.buffer_seconds_cfg * self.sample_rate)), 1)
        self.block_size = max(int(round(self.block_seconds_cfg * self.sample_rate)), 1)
        self.output_ring_size = max(int(round(self.output_ring_seconds_cfg * self.sample_rate)), self.block_size * 4)
        with self._lock:
            self.data = [np.zeros((self.buffer_size,), dtype=np.float32)]
        self._output_ring = _OutputRingBuffer(self.output_ring_size)

    def get_sample_rate(self):
        return self.sample_rate

    def get_buffer_seconds(self):
        return self.buffer_size / float(self.sample_rate)

    def get_file_duration(self):
        if self._file_samples is None:
            return 0.0
        return self._file_length / float(self.sample_rate)

    # ---------------- transport control ----------------

    def set_play_head(self, seconds):
        with self._transport_lock:
            sample = int(round(seconds * self.sample_rate))
            sample = max(0, min(sample, self._file_length))
            self._play_head = sample
        if self._output_ring is not None:
            self._output_ring.clear()

    def get_play_head(self):
        with self._transport_lock:
            return self._play_head / float(self.sample_rate)

    def set_region(self, start_seconds, end_seconds):
        with self._transport_lock:
            start_sample = max(0, min(int(round(start_seconds * self.sample_rate)), self._file_length))
            end_sample = max(0, min(int(round(end_seconds * self.sample_rate)), self._file_length))
            if end_sample <= start_sample:
                end_sample = min(start_sample + 1, self._file_length)
            self._region_start = start_sample
            self._region_end = end_sample
            if self._play_head < start_sample or self._play_head > end_sample:
                self._play_head = start_sample

    def get_region(self):
        with self._transport_lock:
            return (self._region_start / float(self.sample_rate),
                    self._region_end / float(self.sample_rate))

    def set_loop(self, loop):
        self.loop = loop

    def get_loop(self):
        return self.loop

    # ---------------- internal sample pushing (analysis buffer) ----------------

    def _push_samples(self, samples):

        samples = np.asarray(samples, dtype=np.float32)

        if samples.ndim > 1:
            samples = np.mean(samples, axis=-1)

        n = samples.shape[0]

        with self._lock:
            buf = self.data[0]
            if n >= self.buffer_size:
                buf[:] = samples[-self.buffer_size:]
            else:
                buf[:-n] = buf[n:]
                buf[-n:] = samples

    def _mic_callback(self, indata, frames, time_info, status):
        self._push_samples(indata[:, 0] if indata.ndim > 1 else indata)

    def _next_file_block(self, n):
        """
        Pull the next n samples from the loaded file according to the
        current play head / region / loop settings, advancing the play
        head. Returns (samples, still_playing).
        """
        with self._transport_lock:
            if self._file_samples is None or self._file_length == 0:
                return np.zeros(n, dtype=np.float32), False

            out = np.zeros(n, dtype=np.float32)
            cursor = 0
            head = self._play_head
            start = self._region_start
            end = self._region_end
            still_playing = True

            while cursor < n:
                if head >= end:
                    if self.loop:
                        head = start
                    else:
                        still_playing = False
                        break

                available = end - head
                take = min(available, n - cursor)
                out[cursor:cursor + take] = self._file_samples[head:head + take]
                head += take
                cursor += take

            self._play_head = head
            return out, still_playing

    # ---------------- real-time output callback ----------------

    def _output_callback(self, outdata, frames, time_info, status):
        """
        Runs on PortAudio's dedicated real-time thread. Only reads from the
        pre-filled ring buffer (a fast, minimal-lock-time numpy copy) - it
        never touches the file, disk I/O, or analysis code, so it cannot be
        stalled by librosa/OSC/GUI work happening on other threads.
        """
        chunk = self._output_ring.read(frames)
        outdata[:, 0] = chunk

    def _filler_loop(self):
        """
        Runs on a plain Python thread. Keeps the output ring buffer topped
        up by reading ahead from the file and also feeds the same samples
        into the analysis rolling buffer. Being a non-realtime thread, any
        jitter here only affects how far ahead the ring buffer is filled -
        not the actual audio output timing, since the callback just reads
        whatever is already buffered.
        """
        fill_chunk = max(self.block_size, 1)

        while self._running.is_set():

            space = self._output_ring.available_space()

            if space >= fill_chunk:
                mono, still_playing = self._next_file_block(fill_chunk)
                self._push_samples(mono)
                self._output_ring.write(mono)

                if not still_playing:
                    self._running.clear()
                    break
            else:
                time.sleep(0.005)

    def start(self):

        self._running.set()

        if self.mode == "mic":
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                device=self.input_device,
                dtype="float32",
                callback=self._mic_callback,
            )
            self._input_stream.start()
        elif self.mode == "file":
            self._output_ring.clear()

            self._output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=self.output_device,
                dtype="float32",
                blocksize=0,
                callback=self._output_callback,
            )

            self._filler_thread = threading.Thread(target=self._filler_loop, daemon=True)
            self._filler_thread.start()

            # give the filler thread a head start so the ring buffer has
            # some data before the output stream starts pulling from it
            time.sleep(min(self.output_ring_seconds_cfg * 0.25, 0.2))

            self._output_stream.start()
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

    def stop(self):

        self._running.clear()

        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

        if self._filler_thread is not None:
            self._filler_thread.join(timeout=1.0)
            self._filler_thread = None

        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

    def set_mode(self, mode):
        was_running = self._running.is_set()
        if was_running:
            self.stop()

        self.mode = mode

        if mode == "mic" and self.sample_rate is not None:
            self._configure_buffers()

        if was_running:
            self.start()

    def get_latest(self, n_samples):
        """
        Return the most recent n_samples from the rolling buffer. If
        n_samples exceeds the buffer size, the full buffer is returned
        (i.e. the caller may get fewer samples than requested).
        """
        with self._lock:
            buf = self.data[0]
            if n_samples >= self.buffer_size:
                return buf.copy()
            return buf[-n_samples:].copy()
