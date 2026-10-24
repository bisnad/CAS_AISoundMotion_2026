"""
Live Audio VAE: train, listen, optionally record, and replay an experiment
(v22: portable GUI configurations store paths relative to the main experiment
 folder (the directory containing this script), not the JSON folder; loading
 remains compatible with the config-directory paths written by v21;
 v20: resilient config restoration, bounded status labels, atomic JSON saving,
 session recording OFF by default;
(v16: event-log interval replay with passive listening and optional live training;
 v15: automatic startup logging and full snapshots after GUI configuration loads;
 v14 features: persistent bookmarks with optional notes, sparse metrics;
 tabbed GUI with Common tab and save/restore of the GUI configuration (incl. audio files and their
training toggles), direct beta factor, vispy loss-contribution log, splitting applied to the training
dataset only (playback = whole files), click-free playback (crossfaded segments, soft limiter, fades on
underruns), primary loss / LR sliders with adjustable ranges, ConvLoRA, selectable optimizers, numbered audio chunks,
 deterministic full-window batches, causal streaming decoder, and frozen Vocos)
-----------------------------------------------------------------------------------------
- Trains the CNN VAE on Vocos 48 kHz log-mels in a background thread (step-based).
- The encoder processes groups of audio_mel_count_vae mel frames independently; the causal decoder
  decodes the whole latent timeline; a frozen Vocos turns the mels into audio.
- A second thread runs an eval-mode copy of the model in real time, synced every N steps.
- All training data selection, loss weights and loss configurations can be changed while training.
- Audio can be split into independently selectable _1, _2, ... entries instead of cropped.
- Full-network training and ConvLoRA encoder+decoder / decoder-only tuning are available.
- Optimizers: AdamW, Adam, SGD with momentum, RMSprop, Adagrad, and L-BFGS.
- Changing LoRA scope/rank merges currently active adapters into the base (no output jump).
- Primary controls highlight the five default-active loss terms and logarithmic learning rate.
- All numeric GUI fields are draggable sliders with exact input and editable min/max ranges.

Losses (each has its own weight; weight 0 = disabled and not computed):
  vae_mel    Vocos-mel distance (L1/L2) on the VAE input/output mels
  mr_stft    Perceptual multi-resolution mel-STFT (auraloss)
  mel_mr     Multi-resolution log-mel spectrogram loss
  lin_mr     Multi-resolution log linear-magnitude spectrogram loss
  mfcc       MFCC loss (optional deltas)
  loudness   Frame loudness loss (A-weighted STFT or torchaudio ITU BS.1770)
  flatness   Spectral flatness loss (dB, optionally per band)
  centroid   Spectral centroid loss
  lw_centroid Loudness-weighted spectral centroid loss
  temporal   Temporal centroid loss (per segment)
  bandwidth  Spectral bandwidth loss
  contrast   Spectral contrast loss (octave bands, peak vs valley dB)
  rolloff    Soft spectral rolloff loss
  onset      Onset-strength envelope loss
  rms        RMS loss (dB or linear)
  zcr        Soft zero-crossing-rate loss
  chroma     Chroma (STFT) loss
  tonnetz    Tonnetz loss
  poly       Spectral polynomial-fit coefficient loss
  kl         VAE latent KL divergence (scaled only by the beta factor)
Total = rec_loss_scale * sum_k(weight_k * loss_k) + beta * KL
"""

import os
import sys
import time
import queue
import threading
import traceback
import json
import math
import ntpath
import tempfile
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import uuid

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import auraloss
from vocos import Vocos

from PyQt5 import QtWidgets, QtCore, QtGui

from vispy import app as vispy_app, scene
vispy_app.use_app("pyqt5")


# -------------------------------------------------------------------------------------------------
# Compact experiment logger (never called from the real-time audio callback)
# -------------------------------------------------------------------------------------------------

class ExperimentLogger:
    """Append-only JSONL sessions. A single writer thread avoids storing histories in RAM.

    Recording is OFF by default and starts when the user selects Record session.
    Each ON period starts a new, independently interpretable session, including a complete
    initial configuration and dataset snapshot. Only semantic
    GUI actions and low-frequency training metrics are persisted; no audio or tensors.
    """
    FORMAT_VERSION = 1
    QUEUE_CAPACITY = 8192

    def __init__(self, root):
        self.root = os.path.join(root, "sessions")
        self.lock = threading.RLock()
        self.enabled = False
        self.session_dir = None
        self.start_monotonic = 0.0
        self.seq = 0
        self.bookmarks = 0
        self.metrics_written = 0
        self.events_written = 0
        self.dropped_metrics = 0
        self.generation = 0
        self.last_control = {}
        self.writer_error = None
        self.trainer = None
        self.engine = None
        self._queue = None
        self._thread = None
        self._last_metric_time = 0.0
        self._last_metric_step = 0
        self._thread_local = threading.local()  # replay UI actions must not masquerade as new human actions

    @staticmethod
    def _write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)

    @staticmethod
    def _worker(q, folder, owner):
        try:
            with open(os.path.join(folder, "events.jsonl"), "a", encoding="utf-8", buffering=65536) as events, \
                 open(os.path.join(folder, "metrics.jsonl"), "a", encoding="utf-8", buffering=65536) as metrics:
                last_flush = time.monotonic()
                while True:
                    try:
                        kind, record = q.get(timeout=0.8)
                    except queue.Empty:
                        events.flush()
                        metrics.flush()
                        last_flush = time.monotonic()
                        continue
                    if kind == "stop":
                        break
                    target = events if kind == "event" else metrics
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                            allow_nan=False) + "\n")
                    if kind == "event":
                        owner.events_written += 1
                    else:
                        owner.metrics_written += 1
                    if time.monotonic() - last_flush >= 1.0 or record.get("type") in (
                            "bookmark.added", "checkpoint.saved", "session.ended"):
                        events.flush()
                        metrics.flush()
                        last_flush = time.monotonic()
                events.flush()
                metrics.flush()
        except Exception as exc:
            owner.writer_error = f"Experiment logger I/O error: {exc}"
            traceback.print_exc()

    def start(self, initial_state, trigger="user_enabled"):
        """Start a fresh log with a complete configuration snapshot in the event stream."""
        with self.lock:
            if self.enabled:
                return self.session_dir
            now = datetime.now().astimezone()
            label = now.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            folder = os.path.join(self.root, label)
            os.makedirs(folder, exist_ok=False)
            try:
                with open(__file__, "rb") as f:
                    source_hash = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                source_hash = None
            manifest = {
                "format": "live_audio_vae_experiment_session",
                "version": self.FORMAT_VERSION,
                "session_id": label,
                "started_local": now.isoformat(),
                "started_utc": now.astimezone(timezone.utc).isoformat(),
                "application": os.path.basename(__file__),
                "source_sha256": source_hash,
                "data_policy": "No raw audio or weight tensors; audio sources identified by basename/size/mtime (notes/errors may contain user text)",
                "replay_limit": "Model weights, optimizer/RNG state and exact sampled windows are not archived by this logger",
            }
            self._write_json(os.path.join(folder, "manifest.json"), manifest)
            self._write_json(os.path.join(folder, "initial_state.json"), initial_state)
            self.session_dir = folder
            self.start_monotonic = time.monotonic()
            self.seq = 0
            self.bookmarks = 0
            self.metrics_written = 0
            self.events_written = 0
            self.dropped_metrics = 0
            self.generation = 0
            self.last_control = {}
            self.writer_error = None
            self._last_metric_time = 0.0
            self._last_metric_step = -1
            self._queue = queue.Queue(maxsize=self.QUEUE_CAPACITY)
            self._thread = threading.Thread(
                target=self._worker, args=(self._queue, folder, self),
                daemon=True, name="audio-experiment-log-writer")
            self.enabled = True
            self._thread.start()
            # The session starts with the complete effective GUI/training/data state.
            # Keep both records ahead of training-thread events. initial_state.json
            # remains useful, while the JSONL snapshot makes events.jsonl standalone.
            self.event("session.started", "system" if trigger == "application_started" else "user", {
                "initial_step": initial_state.get("model", {}).get("step", 0),
                "trigger": trigger})
            self.event("config.snapshot", "system", {
                "trigger": trigger, "state": initial_state})
        return folder

    def _now(self):
        return int((time.monotonic() - self.start_monotonic) * 1000)

    @contextmanager
    def mute_gui_events(self):
        """Mute only this (GUI) thread while replay applies already-recorded actions.

        Trainer-thread events and training measurements remain visible in the new session.
        """
        old = getattr(self._thread_local, "muted", False)
        self._thread_local.muted = True
        try:
            yield
        finally:
            self._thread_local.muted = old

    def event(self, typ, origin="user", data=None, cause_seq=None):
        if getattr(self._thread_local, "muted", False):
            return None
        with self.lock:
            if not self.enabled:
                return None
            self.seq += 1
            rec = {"v": self.FORMAT_VERSION, "seq": self.seq,
                   "t_ms": self._now(),
                   "step": int(self.trainer.step) if self.trainer else 0,
                   "generation": self.generation,
                   "type": typ, "origin": origin, "data": data or {}}
            if cause_seq is not None:
                rec["cause_seq"] = int(cause_seq)
            if typ == "control.set":
                uid = rec["data"].get("id")
                if uid:
                    self.last_control[uid] = self.seq
            if typ == "bookmark.added":
                self.bookmarks += 1
            # The single FIFO queue preserves event/metric order. Important decisions
            # are never dropped; the writer normally drains much faster than UI events.
            self._queue.put(("event", rec))
            return self.seq

    def last_change(self, uid):
        with self.lock:
            return self.last_control.get(uid)

    def metric(self, data, force=False):
        with self.lock:
            if not self.enabled:
                return False
            now = time.monotonic()
            step = int(self.trainer.step) if self.trainer else 0
            # Usually 30s apart; allow one after 100 steps and at least 5s.
            if not force and (now - self._last_metric_time < 30.0 and
                              (step - self._last_metric_step < 100 or now - self._last_metric_time < 5.0)):
                return False
            rec = {"v": self.FORMAT_VERSION, "t_ms": self._now(),
                   "step": step, "generation": self.generation, **data}
            try:
                self._queue.put_nowait(("metric", rec))
            except queue.Full:
                self.dropped_metrics += 1
                return False
            self._last_metric_time = now
            self._last_metric_step = step
            return True

    def new_generation(self, reason, data=None):
        with self.lock:
            if not self.enabled:
                return
            self.generation += 1
        self.event("model.generation_started", "trainer", {"reason": reason, **(data or {})})

    def stop(self, reason="user_disabled"):
        with self.lock:
            if not self.enabled:
                return
            self.event("session.ended", "user" if reason == "user_disabled" else "system",
                       {"reason": reason, "dropped_metrics": self.dropped_metrics})
            self.enabled = False
            q, thread = self._queue, self._thread
        q.put(("stop", None))
        thread.join(timeout=10.0)
        if thread.is_alive():
            self.writer_error = "Logger writer did not finish; inspect session files"

# -------------------------------------------------------------------------------------------------
# Read-only experiment event replay planning (does not load audio, run Qt, or modify model weights)
# -------------------------------------------------------------------------------------------------

REPLAY_TYPES = {
    "config.snapshot", "config.loaded", "control.set", "ui.slider_range_changed",
    "ui.common_visibility", "ui.tab_selected", "ui.training_entry_selected",
    "data.sources_imported", "data.entry_split", "data.entry_enabled",
    "playback.started", "playback.stopped", "playback.source_changed",
    "training.paused", "training.resumed", "model.reset_requested",
    "lora.merge_requested", "bookmark.added", "checkpoint.load_requested",
}


class ReplaySession:
    """Read only actionable events in chronological order; metrics stay on disk.

    Works with v14/v15/v16 events.jsonl. A full state at session start is required.
    No model weights or captured audio are inferred or synthesized from the JSONL.
    """
    MAX_EVENTS = 200000

    def __init__(self, path, events, initial_state, duration_ms, warnings):
        self.path = os.path.abspath(path)
        self.events = events
        self.initial_state = initial_state
        self.duration_ms = int(duration_ms)
        self.warnings = warnings
        self.bookmarks = [e for e in events if e["type"] == "bookmark.added"]

    @classmethod
    def load(cls, path):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, "events.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        events, initial, warnings = [], None, []
        last_time = 0
        prev_time = -1
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    warnings.append(f"Invalid JSONL record at line {line_number}; ignored")
                    continue
                if not isinstance(e, dict):
                    continue
                try:
                    t = int(e.get("t_ms", 0))
                except (TypeError, ValueError, OverflowError):
                    warnings.append(f"Bad timestamp at line {line_number}; ignored")
                    continue
                if t < 0:
                    continue
                last_time = max(t, last_time)
                typ = e.get("type", "")
                if typ not in REPLAY_TYPES:
                    continue
                e["t_ms"] = t
                if typ == "config.snapshot" and initial is None:
                    initial = e.get("data", {}).get("state")
                if t < prev_time:
                    warnings.append("Event timestamps were out of order; replay sorts by time and sequence")
                prev_time = t
                events.append(e)
                if len(events) > cls.MAX_EVENTS:
                    raise ValueError(f"Event log exceeds {cls.MAX_EVENTS:,} replay events")
        if initial is None:
            fallback = os.path.join(os.path.dirname(path), "initial_state.json")
            if os.path.isfile(fallback):
                with open(fallback, encoding="utf-8") as f:
                    initial = json.load(f)
        if not isinstance(initial, dict) or not isinstance(initial.get("gui"), dict):
            # Old v14 recordings may have only changed controls and no complete
            # initial snapshot. Allow an explicit best-effort replay rather than
            # fabricating the missing original training state.
            initial = None
            warnings.append("No complete initial snapshot: using the current GUI state as baseline; "
                            "pre-log values and model state cannot be recovered")
        # The initial snapshot is applied explicitly, not again as an event.
        events = [e for e in events if e.get("type") != "config.snapshot"]
        events.sort(key=lambda e: (e["t_ms"], int(e.get("seq", 0))))
        return cls(path, events, initial, last_time, list(dict.fromkeys(warnings)))

    def events_between(self, start_ms, end_ms):
        return [e for e in self.events if start_ms <= e["t_ms"] <= end_ms]

    def state_before(self, start_ms):
        """Return events for instant fast-forward, never including the start boundary."""
        return [e for e in self.events if e["t_ms"] < start_ms]

    def missing_sources(self, locator, until_ms=None):
        """Check all full snapshots before mutating the user's running application."""
        problems = []
        states = [self.initial_state]
        states.extend(e.get("data", {}).get("effective_state") for e in self.events
                      if e.get("type") == "config.loaded" and
                      (until_ms is None or e["t_ms"] <= until_ms))
        # v14 emitted a dataset and changed-elements diff on config.loaded,
        # but did not embed a complete effective_state.
        states.extend({"gui": {"dataset": e.get("data", {}).get("dataset", {})}}
                      for e in self.events
                      if e.get("type") == "config.loaded" and
                      (until_ms is None or e["t_ms"] <= until_ms) and
                      isinstance(e.get("data", {}).get("dataset"), dict))
        for e in self.events:
            if until_ms is not None and e["t_ms"] > until_ms:
                continue
            if e.get("type") == "data.sources_imported":
                for desc in e.get("data", {}).get("sources", []):
                    if locator(desc) is None:
                        problems.append(desc.get("name", "unnamed audio"))
        for state in states:
            if not isinstance(state, dict):
                continue
            sources = state.get("gui", {}).get("dataset", {}).get("sources", [])
            for desc in sources:
                if locator(desc) is None:
                    problems.append(desc.get("name", str(desc)) if isinstance(desc, dict) else str(desc))
        return sorted(set(problems))


HISTORY_CAPACITY = 100000   # training steps kept for the Log tab (ring buffer)

# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using {device} device')

# This is the main experiment folder: the directory that contains this GUI script.
# Put the script at the project root alongside data/ and results/ to make
# configurations portable by copying the whole experiment folder.
# Override explicitly with AUDIO_VAE_EXPERIMENT_ROOT when keeping the script
# in a subfolder. The override is local; no absolute root path is saved in JSON.
EXPERIMENT_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get("AUDIO_VAE_EXPERIMENT_ROOT", os.path.dirname(os.path.realpath(__file__)))))
audio_data_path = os.path.join(EXPERIMENT_ROOT, "data", "audio")
audio_data_files = [
    "SSB00050001.wav",
]
# 0 loads entire startup files; a positive value creates _1, _2, ... segments.
default_audio_chunk_seconds = 0.0

audio_sample_rate = 48000

# User-specified: mel frames per Vocos window (training excerpt) and per encoder group
audio_mel_count_vocos = 256
audio_mel_count_vae = 8
mel_floor = -11.5                          # log-mels are clamped to this value (silence)

# Live inference with a rolling mel buffer:
# - every step encodes `inference_blocks_per_step` blocks of audio_mel_count_vae mel frames (one latent group each),
# - the causal decoder turns them into mel frames that are appended to a rolling buffer,
# - Vocos decodes the WHOLE buffer, and only a segment `inference_lookahead_frames` away from the right edge is played.
inference_buffer_frames = 128          # generated mel frames kept and fed to Vocos at every step (multiple of audio_mel_count_vae)
inference_lookahead_frames = 32        # right-edge frames held back; >= Vocos receptive field (~27 frames). Latency ~ lookahead * hop samples
inference_crossfade_samples = 1024     # even; crossfade between consecutive Vocos runs (must be <= inference_step_samples // 2)
inference_blocks_per_step = 2          # encoder groups per step (1 = lowest latency, highest compute)
decoder_streaming = True               # True: causal decoder with conv caches (cheap). False: re-decode a latent history each step
inference_decoder_context_steps = 24   # latent steps of history when decoder_streaming is False (receptive field is ~14 steps)

# Computed automatically (see Vocos section)
audio_window_length_vocos = None
audio_window_length_vae = None

max_queued_blocks = 4                  # generated segments buffered ahead (each is blocks_per_step * 8 frames long)
default_audio_output_device = None

# Model (same as the offline training script)
latent_channels = 32
conv_channel_counts = [32, 256]
conv_strides = [(4, 2), (4, 2)]   # (freq, time)
bottleneck_width = 256
time_downsample = 4                        # product of the time strides

grad_clip = 1.0

save_path = os.path.join(EXPERIMENT_ROOT, "results", "live_vae")
os.makedirs(save_path, exist_ok=True)
autosave_interval_steps = 5000

initial_encoder_weights = None
initial_decoder_weights = None


@dataclass
class TrainSettings:
    rec_loss_scale: float = 5.0        # multiplier on the sum of all weighted reconstruction losses
    beta: float = 0.01                 # KL factor: total = rec_loss_scale * rec + beta * KL
    learning_rate: float = 5e-4
    batch_size: int = 32
    sync_interval: int = 5
    optimizer: str = "AdamW"
    batch_mode: str = "Random windows"
    lbfgs_max_iter: int = 4
    lbfgs_history_size: int = 10
    lbfgs_hold_steps: int = 10  # hold stochastic batches fixed to make curvature estimates meaningful
    train_mode: str = "Full network"
    lora_rank: int = 8
    lora_alpha: float = 8.0
    paused: bool = False


@dataclass
class InferenceSettings:
    file_index: int = 0
    start_sec: float = 0.0
    end_sec: float = 1e9
    use_mean: bool = False
    latent_noise: float = 0.0
    gain: float = 1.0


train_settings = TrainSettings()
inf_settings = InferenceSettings()

# -------------------------------------------------------------------------------------------------
# Loss specifications (drives both the computation and the GUI)
# -------------------------------------------------------------------------------------------------

FFT_CHOICES = ["256", "512", "1024", "2048", "4096"]

LOSS_SPECS = [
    dict(key="vae_mel", title="VAE log-mel distance (Vocos mels)", weight=0.5, params=[
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="mr_stft", title="Perceptual multi-res mel-STFT (auraloss)", weight=0.5, params=[
        dict(key="n_bins", label="Mel bins", type="int", default=128, min=16, max=512),
        dict(key="perceptual", label="Perceptual weighting", type="bool", default=True),
    ]),
    dict(key="mel_mr", title="Multi-resolution log-mel spectrogram", weight=0.0, params=[
        dict(key="n_res", label="Resolutions", type="int", default=3, min=1, max=5),
        dict(key="min_fft", label="Smallest FFT size", type="choice", default="512", choices=FFT_CHOICES),
        dict(key="n_mels", label="Mel bands", type="int", default=80, min=8, max=256),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="lin_mr", title="Multi-resolution log linear spectrogram", weight=0.5, params=[
        dict(key="n_res", label="Resolutions", type="int", default=3, min=1, max=5),
        dict(key="min_fft", label="Smallest FFT size", type="choice", default="512", choices=FFT_CHOICES),
        dict(key="n_bands", label="Pooled bands (0 = all bins)", type="int", default=0, min=0, max=2048),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="mfcc", title="MFCC", weight=0.0, params=[
        dict(key="n_mfcc", label="MFCC coefficients", type="int", default=20, min=2, max=128),
        dict(key="n_mels", label="Mel bands", type="int", default=64, min=8, max=256),
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="skip_c0", label="Skip c0 (overall level)", type="bool", default=True),
        dict(key="deltas", label="Also match deltas", type="bool", default=False),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="loudness", title="Loudness", weight=0.0, params=[
        dict(key="mode", label="Mode", type="choice", default="A-weighted STFT (fast)",
             choices=["A-weighted STFT (fast)", "torchaudio BS.1770 (experimental)"]),
        dict(key="n_fft", label="FFT size (A-weighted mode)", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="db_div", label="dB normalization", type="float", default=20.0, min=1.0, max=200.0, step=1.0),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="flatness", title="Spectral flatness", weight=0.5, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="n_bands", label="Log-spaced bands", type="int", default=8, min=1, max=64),
        dict(key="db_div", label="dB normalization", type="float", default=10.0, min=1.0, max=200.0, step=1.0),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="centroid", title="Spectral centroid", weight=0.0, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="log2", label="Compare in log2(Hz)", type="bool", default=True),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
    dict(key="lw_centroid", title="Loudness-weighted spectral centroid", weight=0.0, params=[
        dict(key="n_fft", label="FFT size", type="choice", default="2048", choices=FFT_CHOICES),
        dict(key="exponent", label="Weight exponent (0 = unweighted)", type="float", default=1.0, min=0.0, max=4.0, step=0.1),
        dict(key="log2", label="Compare in log2(Hz)", type="bool", default=True),
    ]),
    dict(key="temporal", title="Temporal centroid", weight=0.0, params=[
        dict(key="frame", label="Envelope frame (samples)", type="choice", default="512", choices=["128", "256", "512", "1024"]),
        dict(key="segments", label="Segments per window", type="int", default=16, min=1, max=128),
        dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"]),
    ]),
]

# --- Additional descriptor losses (differentiable re-implementations of librosa features) ---
_EXP = dict(key="exponent", label="Loudness weighting exponent (0 = off)", type="float",
            default=0.0, min=0.0, max=4.0, step=0.1)
_DIST = dict(key="dist", label="Distance", type="choice", default="L1", choices=["L1", "L2"])
_NFFT = lambda d="2048": dict(key="n_fft", label="FFT size", type="choice", default=d, choices=FFT_CHOICES)
_FRAME = dict(key="frame", label="Frame (samples)", type="choice", default="1024",
              choices=["256", "512", "1024", "2048"])

LOSS_SPECS += [
    dict(key="bandwidth", title="Spectral bandwidth", weight=0.0, params=[_NFFT(), _EXP, _DIST]),
    dict(key="contrast", title="Spectral contrast", weight=0.0, params=[
        _NFFT(),
        dict(key="n_bands", label="Octave bands", type="int", default=6, min=1, max=6),
        dict(key="fmin", label="Lowest band edge (Hz)", type="float", default=200.0, min=20.0, max=2000.0, step=10.0),
        dict(key="quantile", label="Peak/valley quantile", type="float", default=0.02, min=0.005, max=0.5, step=0.005),
        dict(key="db_div", label="dB normalization", type="float", default=30.0, min=1.0, max=200.0, step=1.0),
        _EXP, _DIST]),
    dict(key="rolloff", title="Spectral rolloff (soft)", weight=0.0, params=[
        _NFFT(),
        dict(key="percent", label="Energy fraction", type="float", default=0.85, min=0.05, max=0.99, step=0.05),
        dict(key="sharpness", label="Softness (higher = sharper)", type="float", default=100.0, min=5.0, max=1000.0, step=10.0),
        dict(key="log2", label="Compare in log2(Hz)", type="bool", default=True),
        _EXP, _DIST]),
    dict(key="onset", title="Onset strength", weight=0.0, params=[
        _NFFT(),
        dict(key="n_mels", label="Mel bands", type="int", default=128, min=8, max=256),
        dict(key="lag", label="Frame lag", type="int", default=1, min=1, max=8),
        dict(key="db_div", label="dB normalization", type="float", default=10.0, min=1.0, max=200.0, step=1.0),
        _DIST]),
    dict(key="rms", title="RMS", weight=0.0, params=[
        _FRAME,
        dict(key="in_db", label="Compare in dB", type="bool", default=True),
        dict(key="db_div", label="dB normalization", type="float", default=20.0, min=1.0, max=200.0, step=1.0),
        _DIST]),
    dict(key="zcr", title="Zero-crossing rate (soft)", weight=0.0, params=[
        _FRAME,
        dict(key="sharpness", label="Sign softness (higher = sharper)", type="float", default=50.0, min=1.0, max=1000.0, step=5.0),
        _DIST]),
    dict(key="chroma", title="Chroma STFT", weight=0.0, params=[
        _NFFT("4096"),
        dict(key="sigma", label="Pitch-class width (semitones)", type="float", default=0.5, min=0.1, max=2.0, step=0.1),
        dict(_EXP, default=1.0), _DIST]),
    dict(key="tonnetz", title="Tonnetz", weight=0.0, params=[
        _NFFT("4096"),
        dict(key="sigma", label="Pitch-class width (semitones)", type="float", default=0.5, min=0.1, max=2.0, step=0.1),
        dict(_EXP, default=1.0), _DIST]),
    dict(key="poly", title="Poly features (spectral polynomial fit)", weight=0.0, params=[
        _NFFT(),
        dict(key="order", label="Polynomial order", type="int", default=1, min=0, max=4),
        dict(key="fit", label="Fit on", type="choice", default="log magnitude",
             choices=["log magnitude", "linear magnitude"]),
        _EXP, _DIST]),
]

loss_cfg = {}
for _s in LOSS_SPECS:
    loss_cfg[_s["key"]] = {"weight": _s["weight"], **{p["key"]: p["default"] for p in _s["params"]}}

WAVEFORM_LOSS_KEYS = [s["key"] for s in LOSS_SPECS if s["key"] != "vae_mel"]

# These four correspond to the default-active reconstruction loss terms (KL is set via beta).
# This is a practical quick-access selection, not a universal ranking of audio losses.
PRIMARY_LOSS_KEYS = ("vae_mel", "mr_stft", "lin_mr", "flatness")
PRIMARY_LOSS_LABELS = {
    "vae_mel": "VAE log-mel reconstruction",
    "mr_stft": "Perceptual multi-res STFT",
    "lin_mr": "Linear multi-res spectrum",
    "flatness": "Spectral flatness",
}

# -------------------------------------------------------------------------------------------------
# Vocos (frozen)
# -------------------------------------------------------------------------------------------------

print("Loading Vocos...")
vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
for p in vocos.parameters():
    p.requires_grad = False


def get_mels(wave):
    # wave (B, N) -> clamped log-mels (B, n_mels, T)
    m = vocos.feature_extractor(wave)
    if m.ndim == 4:
        m = m[:, 0]
    return m.clamp(min=mel_floor)


def vocos_decode_diff(mels):
    # Vocos.decode() is decorated with @torch.inference_mode(), which silently cuts the autograd graph.
    # Same computation (backbone -> head), but differentiable w.r.t. the input mels.
    return vocos.head(vocos.backbone(mels))


def find_vocos_hop():
    hop = getattr(getattr(vocos.feature_extractor, "mel_spec", None), "hop_length", None)
    if hop is None:
        n = 480000
        with torch.no_grad():
            t = vocos.feature_extractor(torch.rand(1, n, device=device)).shape[-1]
        hop = int(round(n / (t - 1)))
    return int(hop)


audio_hop = find_vocos_hop()

assert audio_mel_count_vocos % audio_mel_count_vae == 0, "audio_mel_count_vocos must be a multiple of audio_mel_count_vae"
assert audio_mel_count_vae % time_downsample == 0, "audio_mel_count_vae must be divisible by time_downsample"
assert inference_buffer_frames % audio_mel_count_vae == 0, "inference_buffer_frames must be a multiple of audio_mel_count_vae"

# centered STFT frames: n_frames = n_samples // hop + 1
audio_window_length_vocos = (audio_mel_count_vocos - 1) * audio_hop
audio_window_length_vae = (audio_mel_count_vae - 1) * audio_hop
inference_buffer_window_length = (inference_buffer_frames - 1) * audio_hop
inference_step_frames = inference_blocks_per_step * audio_mel_count_vae
inference_step_samples = inference_step_frames * audio_hop
block_samples = audio_mel_count_vae * audio_hop
assert inference_lookahead_frames + inference_step_frames < inference_buffer_frames, "buffer too small for step + lookahead"
assert inference_crossfade_samples % 2 == 0 and 2 * inference_crossfade_samples <= inference_step_samples, \
    "inference_crossfade_samples must be even and at most half of inference_step_samples"
FADE_SAMPLES = 128                     # ramp length used when audio stops or resumes (underruns)

# encoder input blocks are cut from a slightly longer slice so that the STFT edges see real audio
try:
    stft_n_fft = int(vocos.feature_extractor.mel_spec.n_fft)
except Exception:
    stft_n_fft = 2048
    print("Could not read n_fft from the Vocos feature extractor, assuming", stft_n_fft)
stft_context_frames = int(np.ceil((stft_n_fft // 2) / audio_hop)) + 1
slice_frames = audio_mel_count_vae + 2 * stft_context_frames
slice_samples = (slice_frames - 1) * audio_hop

with torch.no_grad():
    for n, cnt in ((audio_window_length_vocos, audio_mel_count_vocos),
                   (audio_window_length_vae, audio_mel_count_vae),
                   (inference_buffer_window_length, inference_buffer_frames),
                   (slice_samples, slice_frames)):
        m = vocos.feature_extractor(torch.rand(1, n, device=device))
        assert m.shape[-1] == cnt, f"{n} samples give {m.shape[-1]} mel frames, expected {cnt}"
        assert vocos.decode(m).shape[-1] == n, "Vocos decode length mismatch"
    mel_filter_count = m.shape[1]

audio_vae_mels_per_vocos_mels = audio_mel_count_vocos // audio_mel_count_vae
latent_steps_per_group = audio_mel_count_vae // time_downsample
print(f"hop {audio_hop} | mel bins {mel_filter_count} | vocos window {audio_window_length_vocos} samples | "
      f"vae window {audio_window_length_vae} | stft context {stft_context_frames} frames | "
      f"step {inference_step_samples} samples ({inference_step_samples / audio_sample_rate * 1000:.0f} ms) | "
      f"lookahead {inference_lookahead_frames * audio_hop / audio_sample_rate * 1000:.0f} ms")


# -------------------------------------------------------------------------------------------------
# Grouping helpers (the encoder sees groups of audio_mel_count_vae mel frames, independently)
# -------------------------------------------------------------------------------------------------

def group_mels(mels):
    # (B, n_mels, T) -> (B*G, 1, n_mels, audio_mel_count_vae)
    B, n_mels, T = mels.shape
    G = T // audio_mel_count_vae
    return mels.reshape(B, n_mels, G, audio_mel_count_vae).permute(0, 2, 1, 3).reshape(B * G, 1, n_mels, audio_mel_count_vae)


def ungroup_latents(z, B):
    # (B*G, Cz, latent_steps_per_group) -> (B, Cz, Tz)
    BG, Cz, Tzg = z.shape
    G = BG // B
    return z.reshape(B, G, Cz, Tzg).permute(0, 2, 1, 3).reshape(B, Cz, G * Tzg)


# -------------------------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------------------------

def gn(ch):
    return nn.GroupNorm(8, ch)


class ConvLoRAMixin:
    """LoCon-style kernel LoRA: flatten each Conv1D/2D kernel to (out, in*k...).

    This is a true low-rank convolutional *weight update*, not an adapter after
    the convolution. It preserves kernel stride, padding, and causal cache logic.
    The zero-initialized B factor keeps the original function on activation.
    """
    def _init_lora(self, rank=8, alpha=8.0):
        if self.groups != 1:
            raise ValueError("ConvLoRA currently requires groups=1")
        self.lora_enabled = False
        self.lora_rank = 0
        self.lora_alpha = float(alpha)
        self.configure_lora(rank, alpha)

    def configure_lora(self, rank, alpha):
        rank = int(rank)
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if rank != self.lora_rank:
            # Do not resize live adapters without merging them first.
            shape_in = self.weight[0].numel()
            self.lora_A = nn.Parameter(self.weight.new_empty(rank, shape_in))
            self.lora_B = nn.Parameter(self.weight.new_zeros(self.out_channels, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_rank = rank
        self.lora_alpha = float(alpha)

    def lora_delta(self):
        return ((self.lora_B @ self.lora_A).reshape_as(self.weight)
                * (self.lora_alpha / self.lora_rank))

    @torch.no_grad()
    def merge_lora(self):
        if self.lora_enabled:
            self.weight.add_(self.lora_delta())
            self.lora_B.zero_()

    def forward(self, x):
        if self.lora_enabled:
            # _conv_forward respects all the original conv padding/stride options.
            return self._conv_forward(x, self.weight + self.lora_delta(), self.bias)
        return self._conv_forward(x, self.weight, self.bias)


class LoRAConv1d(ConvLoRAMixin, nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_lora()


class LoRAConv2d(ConvLoRAMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_lora()


def lora_layers(model):
    return (m for m in model.modules() if isinstance(m, ConvLoRAMixin))


def set_lora_trainability(enc, dec, mode, rank, alpha, merge_existing=False):
    """Preserve accumulated learning before changing trainable parameter scope."""
    if mode not in ("Full network", "ConvLoRA: encoder + decoder", "ConvLoRA: decoder only"):
        raise ValueError(f"Unknown training mode: {mode}")
    if merge_existing:
        for model in (enc, dec):
            for layer in lora_layers(model):
                layer.merge_lora()
    for model, is_decoder in ((enc, False), (dec, True)):
        use_lora = mode == "ConvLoRA: encoder + decoder" or (is_decoder and mode == "ConvLoRA: decoder only")
        for layer in lora_layers(model):
            layer.configure_lora(rank, alpha)
            layer.lora_enabled = use_lora
        for name, param in model.named_parameters():
            is_adapter = name.endswith((".lora_A", ".lora_B"))
            param.requires_grad_(use_lora and is_adapter if mode != "Full network" else not is_adapter)


class ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(gn(ch), nn.SiLU(), LoRAConv2d(ch, ch, 3, padding=1),
                                 gn(ch), nn.SiLU(), LoRAConv2d(ch, ch, 3, padding=1))
    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, latent_channels, mel_filter_count, channels, strides, width):
        super().__init__()
        self.stem = LoRAConv2d(1, channels[0], 3, padding=1)
        layers, c_in = [], channels[0]
        for c_out, s in zip(channels, strides):
            layers += [ResBlock2d(c_in), gn(c_in), nn.SiLU(), LoRAConv2d(c_in, c_out, 3, stride=s, padding=1)]
            c_in = c_out
        self.down = nn.Sequential(*layers)
        f_red = int(np.prod([s[0] for s in strides]))
        self.flat_ch = channels[-1] * (mel_filter_count // f_red)
        self.head = nn.Sequential(LoRAConv1d(self.flat_ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  LoRAConv1d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                  LoRAConv1d(width, 2 * latent_channels, 3, padding=1))
    def forward(self, x):                      # x: (B,1,F,T) -> mu, std (B,Cz,T')
        h = self.down(self.stem(x)).flatten(1, 2)
        mu, raw = self.head(h).chunk(2, dim=1)
        return mu, F.softplus(raw) + 1e-4      # std is already positive
    @staticmethod
    def reparameterize(mu, std):
        return mu + std * torch.randn_like(std)


class FrameGroupNorm(nn.Module):
    """GroupNorm computed independently for every time step: required for causal streaming."""
    def __init__(self, groups, ch):
        super().__init__()
        self.gn = nn.GroupNorm(groups, ch)
    def forward(self, x):
        if x.ndim == 3:                                        # (B,C,T)
            B, C, T = x.shape
            y = self.gn(x.permute(0, 2, 1).reshape(B * T, C, 1))
            return y.reshape(B, T, C).permute(0, 2, 1)
        B, C, Fq, T = x.shape                                  # (B,C,F,T)
        y = self.gn(x.permute(0, 3, 1, 2).reshape(B * T, C, Fq))
        return y.reshape(B, T, C, Fq).permute(0, 2, 3, 1)


def fgn(ch):
    return FrameGroupNorm(8, ch)


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = LoRAConv1d(c_in, c_out, k)
        self.streaming = False
        self.cache = None
    def forward(self, x):
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], self.k - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k - 1, 0))
        return self.conv(x)


class CausalConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_f=3, k_t=3):
        super().__init__()
        self.k_f, self.k_t = k_f, k_t
        self.conv = LoRAConv2d(c_in, c_out, (k_f, k_t))
        self.streaming = False
        self.cache = None
    def forward(self, x):
        if self.streaming:
            if self.cache is None or self.cache.shape[0] != x.shape[0]:
                self.cache = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.k_t - 1)
            x = torch.cat([self.cache, x], dim=-1)
            self.cache = x[..., -(self.k_t - 1):].detach().clone()
        else:
            x = F.pad(x, (self.k_t - 1, 0))
        x = F.pad(x, (0, 0, self.k_f // 2, self.k_f // 2))
        return self.conv(x)


class CausalResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(fgn(ch), nn.SiLU(), CausalConv2d(ch, ch),
                                 fgn(ch), nn.SiLU(), CausalConv2d(ch, ch))
    def forward(self, x):
        return x + self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels, mel_filter_count, channels, strides, width):
        super().__init__()
        rc, rs = channels[::-1], strides[::-1]
        f_red = int(np.prod([s[0] for s in strides]))
        self.c0, self.f0 = rc[0], mel_filter_count // f_red
        self.inp = nn.Sequential(CausalConv1d(latent_channels, width), fgn(width), nn.SiLU(),
                                 CausalConv1d(width, width), fgn(width), nn.SiLU(),
                                 CausalConv1d(width, self.c0 * self.f0))
        layers = []
        for i, s in enumerate(rs):
            c_in = rc[i]
            c_out = rc[i + 1] if i + 1 < len(rc) else rc[-1]
            layers += [CausalResBlock2d(c_in), fgn(c_in), nn.SiLU(), nn.Upsample(scale_factor=s, mode="nearest"),
                       CausalConv2d(c_in, c_out)]
        self.up = nn.Sequential(*layers)
        self.out = nn.Sequential(fgn(rc[-1]), nn.SiLU(), CausalConv2d(rc[-1], 1))
    def forward(self, z):                      # z: (B,Cz,T') -> (B,1,F,T)
        h = self.inp(z)
        h = h.reshape(h.shape[0], self.c0, self.f0, h.shape[-1])
        return self.out(self.up(h))


def build_models():
    enc = Encoder(latent_channels, mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
    dec = Decoder(latent_channels, mel_filter_count, conv_channel_counts, conv_strides, bottleneck_width).to(device)
    return enc, dec


def kl_loss(mu, std):
    return -0.5 * torch.mean(1 + 2 * torch.log(std) - mu.pow(2) - std.pow(2))

# -------------------------------------------------------------------------------------------------
# Loss implementations
# -------------------------------------------------------------------------------------------------

_cache = {}


def cached(key, factory):
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


def hann(n):
    return cached(("hann", n), lambda: torch.hann_window(n).to(device))


def mel_fb(n_fft, n_mels):
    return cached(("fb", n_fft, n_mels), lambda: torchaudio.functional.melscale_fbanks(
        n_fft // 2 + 1, 0.0, audio_sample_rate / 2, n_mels, audio_sample_rate,
        norm="slaney", mel_scale="slaney").to(device))


def freq_axis(n_fft):
    return cached(("freqs", n_fft), lambda: torch.fft.rfftfreq(n_fft, 1.0 / audio_sample_rate).to(device))


def a_weight_gain(n_fft):
    def make():
        f2 = freq_axis(n_fft) ** 2
        ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
        return ra * 10 ** (2.0 / 20)
    return cached(("again", n_fft), make)


def band_edges(n_bins, n_bands):
    def make():
        e = np.unique(np.round(np.geomspace(1, n_bins, n_bands + 1)).astype(int))
        return e if len(e) > 1 else np.array([1, n_bins])
    return cached(("edges", n_bins, n_bands), make)


def get_mfcc(n_mfcc, n_mels, n_fft):
    n_mfcc = min(n_mfcc, n_mels)
    return cached(("mfcc", n_mfcc, n_mels, n_fft), lambda: torchaudio.transforms.MFCC(
        sample_rate=audio_sample_rate, n_mfcc=n_mfcc, log_mels=True,
        melkwargs=dict(n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels)).to(device))


def get_mr_stft(n_bins, perceptual):
    return cached(("mrstft", n_bins, perceptual), lambda: auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192], hop_sizes=[256, 512, 2048], win_lengths=[1024, 2048, 8192],
        scale="mel", n_bins=n_bins, sample_rate=audio_sample_rate, perceptual_weighting=perceptual).to(device))


def dist(a, b, kind):
    return (a - b).abs().mean() if kind == "L1" else F.mse_loss(a, b)


class LossContext:
    """Holds target / reconstruction waveforms (B, N) and mels (B, F, T); caches STFT magnitudes per step."""
    def __init__(self, y, yhat, y_mels, yhat_mels):
        self.y, self.yhat = y, yhat
        self.y_mels, self.yhat_mels = y_mels, yhat_mels
        self._mags = {}

    def mags(self, n_fft):
        if n_fft not in self._mags:
            B = self.y.shape[0]
            x = torch.cat([self.y, self.yhat], dim=0)
            S = torch.stft(x, n_fft, hop_length=n_fft // 4, win_length=n_fft, window=hann(n_fft),
                           center=True, return_complex=True)
            mag = torch.sqrt(torch.view_as_real(S).pow(2).sum(-1) + 1e-9)      # (2B, F, T)
            self._mags[n_fft] = (mag[:B], mag[B:])
        return self._mags[n_fft]


def loss_vae_mel(ctx, c):
    return dist(ctx.yhat_mels, ctx.y_mels, c["dist"])


def loss_mr_stft(ctx, c):
    return get_mr_stft(int(c["n_bins"]), bool(c["perceptual"]))(ctx.yhat.unsqueeze(1), ctx.y.unsqueeze(1))


def loss_mel_mr(ctx, c):
    total, n = 0.0, int(c["n_res"])
    for i in range(n):
        n_fft = int(c["min_fft"]) * 2 ** i
        my, mh = ctx.mags(n_fft)
        fb = mel_fb(n_fft, int(c["n_mels"]))
        ly = torch.log(torch.einsum('bft,fm->bmt', my, fb) + 1e-5)
        lh = torch.log(torch.einsum('bft,fm->bmt', mh, fb) + 1e-5)
        total = total + dist(lh, ly, c["dist"])
    return total / n


def loss_lin_mr(ctx, c):
    total, n, bands = 0.0, int(c["n_res"]), int(c["n_bands"])
    for i in range(n):
        n_fft = int(c["min_fft"]) * 2 ** i
        my, mh = ctx.mags(n_fft)
        if bands > 0:
            my = F.adaptive_avg_pool1d(my.permute(0, 2, 1), bands)
            mh = F.adaptive_avg_pool1d(mh.permute(0, 2, 1), bands)
        total = total + dist(torch.log(mh + 1e-5), torch.log(my + 1e-5), c["dist"])
    return total / n


def loss_mfcc(ctx, c):
    tf = get_mfcc(int(c["n_mfcc"]), int(c["n_mels"]), int(c["n_fft"]))
    cy, ch = tf(ctx.y), tf(ctx.yhat)                         # (B, n_mfcc, T)
    if c["skip_c0"]:
        cy, ch = cy[:, 1:], ch[:, 1:]
    loss = dist(ch, cy, c["dist"])
    if c["deltas"]:
        d = torchaudio.functional.compute_deltas
        loss = loss + dist(d(ch), d(cy), c["dist"])
    return loss


def loss_loudness(ctx, c):
    if c["mode"].startswith("torchaudio"):
        def lo(x):
            v = torchaudio.functional.loudness(x.unsqueeze(1), audio_sample_rate)
            return torch.nan_to_num(v, nan=-70.0, neginf=-70.0, posinf=0.0)
        return dist(lo(ctx.yhat), lo(ctx.y), c["dist"]) / c["db_div"]
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    g = a_weight_gain(n_fft)[None, :, None]

    def lo(m):
        return 10.0 * torch.log10((m * g).pow(2).mean(1) + 1e-6)     # (B, T) dB
    return dist(lo(mh), lo(my), c["dist"]) / c["db_div"]


def loss_flatness(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    edges = band_edges(my.shape[1], int(c["n_bands"]))

    def flat_db(m):
        p = m.pow(2) + 1e-8
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            band = p[:, a:b, :]
            out.append((torch.log(band).mean(1) - torch.log(band.mean(1))) * 4.342944819)   # 10 / ln(10)
        return torch.stack(out, dim=1)                                # (B, bands, T)
    return dist(flat_db(mh), flat_db(my), c["dist"]) / c["db_div"]


def _centroid(m, n_fft, use_log2):
    f = freq_axis(n_fft)[None, :, None]
    cen = (f * m).sum(1) / (m.sum(1) + 1e-6)                          # (B, T) Hz
    return torch.log2(cen + 1.0) if use_log2 else cen / 1000.0


def loss_centroid(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    return dist(_centroid(mh, n_fft, c["log2"]), _centroid(my, n_fft, c["log2"]), c["dist"])


def loss_lw_centroid(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    cy, ch = _centroid(my, n_fft, c["log2"]), _centroid(mh, n_fft, c["log2"])
    w = (torch.sqrt(my.pow(2).sum(1)) + 1e-6).pow(c["exponent"]).detach()      # target frame amplitude
    w = w / w.sum(1, keepdim=True)
    return (w * (ch - cy).abs()).sum(1).mean()


def loss_temporal(ctx, c):
    frame, S = int(c["frame"]), int(c["segments"])
    hop = frame // 2

    def tc(x):
        env = torch.sqrt(x.unfold(-1, frame, hop).pow(2).mean(-1) + 1e-8)     # (B, T)
        T = (env.shape[1] // S) * S
        env = env[:, :T].reshape(env.shape[0], S, T // S)
        t = torch.linspace(0, 1, T // S, device=x.device)
        return (env * t).sum(-1) / (env.sum(-1) + 1e-8)                       # (B, S) in [0, 1]
    return dist(tc(ctx.yhat), tc(ctx.y), c["dist"])


# --- Additional descriptor losses ---

def _ferr(a, b, kind):
    d = a - b
    return d.abs() if kind == "L1" else d * d


def _reduce_frames(e, my, exponent):
    # e: (B, ..., T) elementwise error -> scalar, optionally weighted by target frame amplitude
    e = e.reshape(e.shape[0], -1, e.shape[-1]).mean(1)                 # (B, T)
    if exponent <= 0:
        return e.mean()
    w = (torch.sqrt(my.pow(2).sum(1)) + 1e-6).pow(exponent).detach()
    w = w / w.sum(1, keepdim=True)
    return (w * e).sum(1).mean()


def _frames(x, frame):
    xp = F.pad(x.unsqueeze(1), (frame // 2, frame // 2), mode="reflect").squeeze(1)
    return xp.unfold(-1, frame, max(1, frame // 4))                    # (B, T, frame)


def loss_bandwidth(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    f = freq_axis(n_fft)[None, :, None]

    def bw(m):
        p = m / (m.sum(1, keepdim=True) + 1e-6)
        cen = (p * f).sum(1, keepdim=True)
        return torch.sqrt((p * (f - cen) ** 2).sum(1) + 1.0) / 1000.0   # kHz
    return _reduce_frames(_ferr(bw(mh), bw(my), c["dist"]), my, c["exponent"])


def _contrast_bands(n_fft, n_bands, fmin):
    def make():
        f = freq_axis(n_fft).cpu().numpy()
        nb = max(1, min(n_bands, int(np.floor(np.log2((audio_sample_rate / 2) / fmin)))))
        edges = np.concatenate([[0.0], fmin * 2.0 ** np.arange(nb + 1)])
        idx = np.searchsorted(f, edges)
        idx[-1] = len(f)                                               # last band reaches Nyquist
        return [(int(a), int(b)) for a, b in zip(idx[:-1], idx[1:]) if b > a]
    return cached(("cbands", n_fft, n_bands, float(fmin)), make)


def loss_contrast(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    bands = _contrast_bands(n_fft, int(c["n_bands"]), float(c["fmin"]))
    q = float(c["quantile"])

    def contrast(m):
        out = []
        for a, b in bands:
            s, _ = m[:, a:b, :].sort(dim=1)                            # gradients flow to selected bins
            k = max(1, int(round(q * (b - a))))
            out.append(10 * torch.log10(s[:, -k:].mean(1) + 1e-6) - 10 * torch.log10(s[:, :k].mean(1) + 1e-6))
        return torch.stack(out, 1) / c["db_div"]                       # (B, bands, T)
    return _reduce_frames(_ferr(contrast(mh), contrast(my), c["dist"]), my, c["exponent"])


def loss_rolloff(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    f = freq_axis(n_fft)[None, :, None]
    k, pct = float(c["sharpness"]), float(c["percent"])

    def ro(m):
        cum = m.cumsum(1) / (m.sum(1, keepdim=True) + 1e-6)
        ge = torch.sigmoid(k * (cum - pct))                            # soft "reached pct"
        first = ge - F.pad(ge, (0, 0, 1, 0))[:, :-1]                   # soft first-crossing distribution
        hz = (first * f).sum(1) / (first.sum(1) + 1e-6)
        return torch.log2(hz + 1.0) if c["log2"] else hz / 1000.0
    return _reduce_frames(_ferr(ro(mh), ro(my), c["dist"]), my, c["exponent"])


def loss_onset(ctx, c):
    n_fft, lag = int(c["n_fft"]), int(c["lag"])
    my, mh = ctx.mags(n_fft)
    fb = mel_fb(n_fft, int(c["n_mels"]))

    def onset(m):
        db = 10 * torch.log10(torch.einsum('bft,fm->bmt', m.pow(2), fb) + 1e-8)
        return F.relu(db[..., lag:] - db[..., :-lag]).mean(1) / c["db_div"]
    return dist(onset(mh), onset(my), c["dist"])


def loss_rms(ctx, c):
    frame = int(c["frame"])

    def rms(x):
        ms = _frames(x, frame).pow(2).mean(-1)
        return 10 * torch.log10(ms + 1e-8) / c["db_div"] if c["in_db"] else torch.sqrt(ms + 1e-8)
    return dist(rms(ctx.yhat), rms(ctx.y), c["dist"])


def loss_zcr(ctx, c):
    frame, k = int(c["frame"]), float(c["sharpness"])

    def zcr(x):
        s = torch.tanh(k * _frames(x, frame))
        return (0.5 * (1 - s[..., 1:] * s[..., :-1])).mean(-1)
    return dist(zcr(ctx.yhat), zcr(ctx.y), c["dist"])


def chroma_fb(n_fft, sigma):
    def make():
        fr = freq_axis(n_fft)
        f = fr.clone(); f[0] = 1.0
        pc = (69 + 12 * torch.log2(f / 440.0)) % 12
        cls = torch.arange(12, device=fr.device, dtype=fr.dtype)[:, None]
        d = (pc[None] - cls).abs()
        d = torch.minimum(d, 12 - d)
        fb = torch.exp(-0.5 * (d / sigma) ** 2)
        fb[:, (fr < 55.0) | (fr > 5000.0)] = 0.0
        return fb
    return cached(("chroma", n_fft, float(sigma)), make)


def _chroma_raw(m, n_fft, sigma):
    return torch.einsum('cf,bft->bct', chroma_fb(n_fft, float(sigma)), m.pow(2))


def loss_chroma(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)

    def ch(m):
        x = _chroma_raw(m, n_fft, c["sigma"])
        return x / torch.sqrt((x ** 2).sum(1, keepdim=True) + 1e-12)
    return _reduce_frames(_ferr(ch(mh), ch(my), c["dist"]), my, c["exponent"])


def tonnetz_phi():
    def make():
        scale = torch.tensor([7 / 6, 7 / 6, 3 / 2, 3 / 2, 2 / 3, 2 / 3], device=device)
        V = scale[:, None] * torch.arange(12.0, device=device)[None]
        V[::2] -= 0.5
        R = torch.tensor([1, 1, 1, 1, 0.5, 0.5], device=device)
        return R[:, None] * torch.cos(np.pi * V)                       # (6, 12), same as librosa
    return cached(("tonnetz",), make)


def loss_tonnetz(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    phi = tonnetz_phi()

    def tn(m):
        x = _chroma_raw(m, n_fft, c["sigma"])
        x = x / (x.abs().sum(1, keepdim=True) + 1e-6)                  # L1 norm as in librosa
        return torch.einsum('dc,bct->bdt', phi, x)                     # (B, 6, T)
    return _reduce_frames(_ferr(tn(mh), tn(my), c["dist"]), my, c["exponent"])


def poly_pinv(n_fft, order):
    def make():
        x = torch.linspace(-1, 1, n_fft // 2 + 1)
        P = [torch.ones_like(x), x]
        for n in range(1, order):
            P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))   # Legendre basis (well conditioned)
        V = torch.stack(P[:order + 1], 1)
        return torch.linalg.pinv(V).to(device)                           # (order+1, F), solved on CPU
    return cached(("poly", n_fft, order), make)


def loss_poly(ctx, c):
    n_fft = int(c["n_fft"])
    my, mh = ctx.mags(n_fft)
    pinv = poly_pinv(n_fft, int(c["order"]))
    use_log = c["fit"].startswith("log")

    def coef(m):
        return torch.einsum('of,bft->bot', pinv, torch.log(m + 1e-5) if use_log else m)
    return _reduce_frames(_ferr(coef(mh), coef(my), c["dist"]), my, c["exponent"])


LOSS_FUNCS = dict(vae_mel=loss_vae_mel, mr_stft=loss_mr_stft, mel_mr=loss_mel_mr, lin_mr=loss_lin_mr,
                  mfcc=loss_mfcc, loudness=loss_loudness, flatness=loss_flatness, centroid=loss_centroid,
                  lw_centroid=loss_lw_centroid, temporal=loss_temporal)

LOSS_FUNCS.update(bandwidth=loss_bandwidth, contrast=loss_contrast, rolloff=loss_rolloff,
                  onset=loss_onset, rms=loss_rms, zcr=loss_zcr, chroma=loss_chroma,
                  tonnetz=loss_tonnetz, poly=loss_poly)

# -------------------------------------------------------------------------------------------------
# Audio data store (live-editable)
# -------------------------------------------------------------------------------------------------


def portable_audio_basename(path):
    """Extract a filename from either Windows or POSIX paths on either platform."""
    return ntpath.basename(str(path).rstrip("/\\"))


def portable_relative_audio_path(source_path, base_dir):
    """Serialize source_path relative to base_dir using universal '/' separators.

    For normal GUI configurations, base_dir is EXPERIMENT_ROOT, regardless of
    where the JSON file is saved. Native absolute paths remain in memory.
    Files on different Windows volumes cannot be relativized and are saved by
    filename only, to be located with the existing audio-folder picker.
    """
    if not base_dir:
        raise ValueError("A base directory is required for relative audio paths")
    source = os.path.abspath(os.path.expanduser(os.fspath(source_path)))
    base = os.path.abspath(os.path.expanduser(os.fspath(base_dir)))
    try:
        relative = os.path.relpath(source, start=base)
    except ValueError:  # different drives or shares on Windows
        relative = portable_audio_basename(source)
    return relative.replace("\\", "/")


def portable_audio_search_roots(config_dir=None, audio_root=None, experiment_root=None):
    """Ordered, bounded search paths; never recursively scan a disk."""
    roots = []

    def add(directory):
        if directory:
            directory = os.path.abspath(os.path.expanduser(str(directory)))
            if directory not in roots:
                roots.append(directory)

    add(audio_root)   # the folder chosen by the user, if anything is missing
    add(config_dir)  # legacy v21 configs and filename-only fallbacks
    if config_dir:
        parent = os.path.abspath(config_dir)
        for _ in range(3):
            add(os.path.join(parent, "data", "audio"))
            add(os.path.join(parent, "audio"))
            parent = os.path.dirname(parent)
    project = EXPERIMENT_ROOT if experiment_root is None else experiment_root
    add(project)
    add(os.path.join(project, "data", "audio"))
    add(os.path.join(project, "audio"))
    add(audio_data_path)
    return roots

class AudioStore:
    """Whole source files (playback) and numbered independent segments (training dataset)."""
    MAX_FULL_WINDOWS = 64  # avoid accidental huge full-batch backward passes

    def __init__(self):
        self.lock = threading.Lock()
        self.src_paths, self.src_names, self.src_waves = [], [], []   # whole source files: live inference only
        self.names, self.waves = [], []             # training entries (segments): dataset only
        self.refs = []                              # per training entry: (source index, start sample in source)
        self.enabled, self.start, self.end = [], [], []
        self.version = 0
        self._full_cache = None
        self.last_restore_info = {"requested": 0, "loaded": 0, "preserved_existing": False}

    def _source_candidates(self, saved_path, *, config_dir=None, audio_root=None,
                           relative_to="config_directory", experiment_root=None):
        """Resolve saved audio by its explicitly declared base directory.

        v22 'experiment_root': relative to the directory containing the GUI
        script, independent of JSON location and the current working directory.
        v21 'config_directory': relative to the saved JSON's location.
        v20 and earlier: native absolute paths or safe filename relocation.
        """
        saved = (str(saved_path.get("path") or saved_path.get("name") or "")
                 if isinstance(saved_path, dict) else str(saved_path))
        name = portable_audio_basename(saved)
        if not name or name in (".", ".."):
            return []
        result, seen = [], set()

        def add(candidate):
            if not candidate:
                return
            normalized = os.path.abspath(os.path.expanduser(os.fspath(candidate)))
            key = os.path.normcase(normalized)
            if key not in seen:
                seen.add(key)
                result.append(normalized)

        relative = saved.replace("\\", "/").replace("/", os.sep)
        native_absolute = os.path.isabs(saved)
        absolute_on_either_platform = native_absolute or ntpath.isabs(saved)
        project = EXPERIMENT_ROOT if experiment_root is None else experiment_root
        anchor = project if relative_to == "experiment_root" else config_dir

        if not absolute_on_either_platform:
            if anchor:
                add(os.path.join(anchor, relative))  # authoritative path first
            if audio_root:
                add(os.path.join(audio_root, relative))
            if anchor is None and audio_root is None:
                add(saved)  # only for non-config callers such as replay
        elif native_absolute:
            add(saved)  # older absolute-path configs on the original OS

        roots = portable_audio_search_roots(config_dir, audio_root, project)
        for root in roots:
            if not absolute_on_either_platform:
                add(os.path.join(root, relative))
            add(os.path.join(root, name))  # fallback for relocated sources
        with self.lock:
            existing = tuple(self.src_paths)
        for current in existing:
            if portable_audio_basename(current).casefold() == name.casefold():
                add(current)
        for root in roots:
            if not os.path.isdir(root):
                continue
            try:
                with os.scandir(root) as items:
                    for item in items:
                        if item.name.casefold() == name.casefold() and item.is_file():
                            add(item.path)
                            break
            except OSError:
                pass
        return result

    def unresolved_sources(self, layout, *, config_dir=None, audio_root=None,
                           experiment_root=None):
        """Non-destructive lookup before optionally showing an audio-folder picker."""
        missing = []
        relative_to = layout.get("relative_to", "config_directory")
        for saved in layout.get("sources", []):
            if not any(os.path.isfile(p) for p in self._source_candidates(
                    saved, config_dir=config_dir, audio_root=audio_root,
                    relative_to=relative_to, experiment_root=experiment_root)):
                missing.append(portable_audio_basename(
                    saved.get("path") or saved.get("name") or "" if isinstance(saved, dict) else saved))
        return missing

    @staticmethod
    def _load_wave(path):
        wave, sr = torchaudio.load(path)
        if sr != audio_sample_rate:
            wave = torchaudio.functional.resample(wave, sr, audio_sample_rate)
        wave = wave[0].to(device)
        if wave.numel() == 0:
            raise ValueError(f"Empty audio file: {path}")
        return wave

    @staticmethod
    def _segments(name, wave, chunk_seconds):
        """Returns [(name, view, offset_in_wave)]; views share storage, so splitting costs no extra memory."""
        length = int(round(float(chunk_seconds) * audio_sample_rate))
        if length <= 0 or wave.shape[0] <= length:
            return [(name, wave, 0)]
        stem, ext = os.path.splitext(name)
        return [(f"{stem}_{index}{ext}", wave[pos:pos + length], pos)
                for index, pos in enumerate(range(0, wave.shape[0], length), start=1)]

    def _insert_entries(self, index, entries, is_enabled=True, replace_count=0):
        """entries: [(name, wave, source_index, offset_in_source)]"""
        names = [e[0] for e in entries]
        waves = [e[1] for e in entries]
        count = len(entries)
        self.names[index:index + replace_count] = names
        self.waves[index:index + replace_count] = waves
        self.refs[index:index + replace_count] = [(e[2], e[3]) for e in entries]
        self.enabled[index:index + replace_count] = [bool(is_enabled)] * count
        self.start[index:index + replace_count] = [0] * count
        self.end[index:index + replace_count] = [int(wav.shape[0]) for wav in waves]
        self.version += 1
        self._full_cache = None

    def add_file(self, path, chunk_seconds=0.0, enabled=False):
        wave = self._load_wave(path)
        segments = self._segments(os.path.basename(path), wave, chunk_seconds)
        with self.lock:
            src_index = len(self.src_names)
            self.src_paths.append(os.path.abspath(path))
            self.src_names.append(os.path.basename(path))   # playback keeps the whole file
            self.src_waves.append(wave)
            entries = [(n, w, src_index, off) for n, w, off in segments]
            self._insert_entries(len(self.names), entries, is_enabled=enabled)
        print(f"Loaded {path}: {wave.shape[0] / audio_sample_rate:.1f} s -> {len(entries)} training entries")
        return len(entries)

    def split_entry(self, index, chunk_seconds):
        """Replace a selected training entry with its numbered chunks, without cutting it off."""
        with self.lock:
            if not (0 <= index < len(self.names)):
                return 0
            segments = self._segments(self.names[index], self.waves[index], chunk_seconds)
            if len(segments) <= 1:
                return 0
            src_index, base = self.refs[index]
            entries = [(n, w, src_index, base + off) for n, w, off in segments]
            self._insert_entries(index, entries, self.enabled[index], replace_count=1)
            return len(entries)

    def export_layout(self, *, config_dir=None, experiment_root=None):
        """Export the dataset with native paths for logs, portable paths for JSON.

        GUI config saving supplies experiment_root, generating v22 references
        independent of the JSON's directory. With no base, keep the original
        native-path representation used by session logging and replay.
        config_dir is retained to support legacy callers in tests and tools.
        """
        with self.lock:
            sources = list(self.src_paths)
            entries = [dict(name=n, source=int(r[0]), offset=int(r[1]), length=int(w.shape[0]),
                            enabled=bool(on))
                       for n, w, r, on in zip(self.names, self.waves, self.refs, self.enabled)]
        if experiment_root is None and config_dir is None:
            return {"sources": sources, "entries": entries}
        base_dir = experiment_root if experiment_root is not None else config_dir
        relative_to = "experiment_root" if experiment_root is not None else "config_directory"
        portable = [portable_relative_audio_path(p, base_dir) for p in sources]
        # A Windows path on another drive/share cannot have a relative reference.
        base_drive = os.path.splitdrive(os.path.abspath(base_dir))[0].casefold()
        basename_only = [portable_audio_basename(p) for p in sources
                         if os.path.splitdrive(os.path.abspath(p))[0].casefold() != base_drive]
        layout = {"path_format": "posix_relative", "relative_to": relative_to,
                  "sources": portable, "entries": entries}
        if basename_only:
            layout["basename_only_sources"] = basename_only
        return layout

    def restore_layout(self, layout, *, config_dir=None, audio_root=None,
                       experiment_root=None):
        """Restore portable sources and segmented training entries.

        Returns missing *filenames* (never platform-specific full paths).
        If every listed source is missing, keep the user's current dataset
        instead of emptying it. Other GUI settings can still be restored.
        """
        if not isinstance(layout, dict):
            raise ValueError("Dataset configuration must be a JSON object")
        sources = layout.get("sources", [])
        entries = layout.get("entries", [])
        if not isinstance(sources, list) or not isinstance(entries, list):
            raise ValueError("Invalid dataset sources/entries in GUI configuration")
        loaded, missing = {}, []
        relative_to = layout.get("relative_to", "config_directory")
        for i, saved in enumerate(sources):
            candidates = self._source_candidates(
                saved, config_dir=config_dir, audio_root=audio_root,
                relative_to=relative_to, experiment_root=experiment_root)
            name = portable_audio_basename(
                saved.get("path") or saved.get("name") or "" if isinstance(saved, dict) else saved)
            found = False
            load_error = None
            for candidate in candidates:
                if not os.path.isfile(candidate):
                    continue
                try:
                    loaded[i] = (os.path.abspath(candidate), self._load_wave(candidate))
                    found = True
                    break
                except Exception as exc:
                    load_error = exc
            if not found:
                missing.append(name or f"source_{i}")
                if load_error is not None:
                    print(f"Could not decode {name}: {load_error}")
        self.last_restore_info = {"requested": len(sources), "loaded": len(loaded),
                                  "preserved_existing": bool(sources and not loaded)}
        if sources and not loaded:
            return missing  # do NOT wipe a working dataset on a different computer
        order = sorted(loaded)
        remap = {old: new for new, old in enumerate(order)}
        new_paths = [loaded[i][0] for i in order]
        new_waves = [loaded[i][1] for i in order]
        names, waves, refs, enabled, starts, ends = [], [], [], [], [], []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            try:
                old_src = int(ent["source"])
                if old_src not in remap:
                    continue
                new_src = remap[old_src]
                wave = new_waves[new_src]
                if wave.numel() == 0:
                    continue
                offset = min(max(0, int(ent.get("offset", 0))), wave.shape[0] - 1)
                length = max(0, int(ent.get("length", wave.shape[0])))
                part = wave[offset:offset + length]
                if part.numel() == 0:
                    continue
                names.append(str(ent.get("name") or portable_audio_basename(new_paths[new_src])))
                waves.append(part)
                refs.append((new_src, offset))
                enabled.append(bool(ent.get("enabled", False)))
                starts.append(0)
                ends.append(int(part.shape[0]))
            except (KeyError, ValueError, TypeError, OverflowError) as exc:
                print(f"Skipped invalid training entry while restoring config: {exc}")
        # Commit the new dataset only after all sources and entries have been processed.
        with self.lock:
            self.src_paths = new_paths
            self.src_names = [os.path.basename(p) for p in new_paths]
            self.src_waves = new_waves
            self.names, self.waves, self.refs = names, waves, refs
            self.enabled, self.start, self.end = enabled, starts, ends
            self.version += 1
            self._full_cache = None
        return missing

    def set_enabled(self, index, value):
        with self.lock:
            if 0 <= index < len(self.enabled):
                self.enabled[index] = bool(value)
                self.version += 1
                self._full_cache = None

    def _snapshot(self):
        with self.lock:
            return [(w, s, e) for w, on, s, e in zip(self.waves, self.enabled, self.start, self.end) if on and e > s]

    def sample_batch(self, n):
        W = audio_window_length_vocos
        with self.lock:
            lengths = [max(0, e-s) for s, e in zip(self.start, self.end)]
            usable = np.array([max(1, L - W + 1) if enabled and L > 0 else 0
                               for enabled, L in zip(self.enabled, lengths)], dtype=np.float64)
            if usable.sum() <= 0:
                return None
            files = np.random.choice(len(usable), size=n, p=usable / usable.sum())
            windows = []
            for f in files:
                usable_length = lengths[f]
                offset = np.random.randint(0, usable_length - W + 1) if usable_length >= W else 0
                start = self.start[f] + offset
                window = self.waves[f][start:min(start + W, self.end[f])]
                if window.numel() < W:  # short final chunks remain independently trainable
                    window = F.pad(window, (0, W - window.numel()))
                windows.append(window)
        return torch.stack(windows).unsqueeze(1)

    def full_batch(self):
        """Deterministic non-overlapping windows plus final overlap to cover tails."""
        with self.lock:
            if self._full_cache is not None and self._full_cache[0] == self.version:
                return self._full_cache[1]
            W = audio_window_length_vocos
            specs = []
            for f, (enabled, start, end) in enumerate(zip(self.enabled, self.start, self.end)):
                if not enabled or end <= start:
                    continue
                if end - start < W:
                    positions = [start]
                else:
                    positions = list(range(start, end - W + 1, W))
                    if positions[-1] != end - W:
                        positions.append(end - W)
                specs.extend((f, pos) for pos in positions)
                if len(specs) > self.MAX_FULL_WINDOWS:
                    raise ValueError(f"Full batch has more than {self.MAX_FULL_WINDOWS} audio windows; "
                                     "use Random windows or fewer/shorter enabled entries")
            if not specs:
                return None
            windows = []
            for f, pos in specs:
                window = self.waves[f][pos:min(pos + W, self.end[f])]
                windows.append(F.pad(window, (0, W - window.numel())) if window.numel() < W else window)
            batch = torch.stack(windows).unsqueeze(1)
            self._full_cache = (self.version, batch)
            return batch


store = AudioStore()

# -------------------------------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------------------------------

class Trainer:
    def __init__(self):
        self.model_lock = threading.Lock()
        self.experiment_logger = None    # attached by main() after construction
        self.optimizer_revision = 0
        self.inference_sync_revision = 0
        self.inference_sync_step = 0
        self.last_loss_snapshot = None
        self.last_loss_step = None
        self._loss_suspended = False
        self._last_logged_error = None
        self._pending_checkpoint_seq = None
        self._pending_merge_seq = None
        self.enc, self.dec = build_models()
        self.inf_enc, self.inf_dec = build_models()
        self._active_lora_settings = None
        self._active_optimizer_settings = None
        self._active_batch_mode = None
        self._lbfgs_batch = None
        self._lbfgs_data_version = -1
        self._lbfgs_batch_steps = 0
        self._lbfgs_epsilon = None
        self.gui_config_revision = 0
        self.step = 0
        self.running = True
        self.reset_req = False
        self.merge_lora_req = False
        self.save_req = False
        self.load_req = None
        self.stats = dict(total=0.0, kld=0.0, beta=0.0, sps=0.0, evals=0)
        self.loss_ema = {spec["key"]: 0.0 for spec in LOSS_SPECS}
        self.status_msg = ""
        self.hist_keys = [s["key"] for s in LOSS_SPECS] + ["kl", "total"]
        self.hist_row = {k: i for i, k in enumerate(self.hist_keys)}
        self.hist = np.full((len(self.hist_keys), HISTORY_CAPACITY), np.nan, dtype=np.float32)
        self.hist_step = 0
        self._apply_configuration(merge_existing=False)
        if initial_encoder_weights and initial_decoder_weights:
            self._restore_pair(initial_encoder_weights, initial_decoder_weights)
        self.sync_inference_models()
        self.inf_enc.eval()
        self.inf_dec.eval()
        for model in (self.inf_enc, self.inf_dec):
            for param in model.parameters():
                param.requires_grad_(False)

    def _make_optimizer(self):
        s = train_settings
        self.params = [p for m in (self.enc, self.dec) for p in m.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("No trainable parameters selected")
        name = s.optimizer
        lr = s.learning_rate
        if name == "AdamW":
            self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)
        elif name == "Adam":
            self.opt = torch.optim.Adam(self.params, lr=lr, betas=(0.9, 0.999))
        elif name == "SGD + momentum":
            self.opt = torch.optim.SGD(self.params, lr=lr, momentum=0.9, nesterov=True)
        elif name == "RMSprop":
            self.opt = torch.optim.RMSprop(self.params, lr=lr, alpha=0.99)
        elif name == "Adagrad":
            self.opt = torch.optim.Adagrad(self.params, lr=lr)
        elif name == "L-BFGS":
            self.opt = torch.optim.LBFGS(
                self.params, lr=lr, max_iter=s.lbfgs_max_iter,
                history_size=s.lbfgs_history_size, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Unknown optimizer {name}")
        self.active_parameters = sum(p.numel() for p in self.params)
        self.total_parameters = sum(p.numel() for m in (self.enc, self.dec) for p in m.parameters())
        self.optimizer_revision += 1

    def _clear_lbfgs_data(self):
        self._lbfgs_batch = None
        self._lbfgs_data_version = -1
        self._lbfgs_batch_steps = 0
        self._lbfgs_epsilon = None

    def _apply_configuration(self, merge_existing=True):
        """Apply GUI changes on the trainer thread, never from a Qt signal handler."""
        s = train_settings
        lora_sig = (s.train_mode, int(s.lora_rank), float(s.lora_alpha))
        optimizer_sig = (s.optimizer, int(s.lbfgs_max_iter), int(s.lbfgs_history_size))
        previous_lora = self._active_lora_settings
        previous_optimizer = self._active_optimizer_settings
        previous_batch = self._active_batch_mode
        mode_changed = lora_sig != previous_lora
        optimizer_changed = optimizer_sig != previous_optimizer
        batch_mode_changed = s.batch_mode != previous_batch
        if mode_changed:
            set_lora_trainability(self.enc, self.dec, *lora_sig,
                                  merge_existing=merge_existing and self._active_lora_settings is not None)
            self._active_lora_settings = lora_sig
        if mode_changed or optimizer_changed:
            self._make_optimizer()
            self._active_optimizer_settings = optimizer_sig
            self._clear_lbfgs_data()
            self.status_msg = (f"{s.optimizer} | {s.train_mode} | "
                               f"trainable {self.active_parameters:,}/{self.total_parameters:,}")
            self.sync_inference_models()
        if batch_mode_changed:
            self._clear_lbfgs_data()
            if s.optimizer == "L-BFGS" and not (mode_changed or optimizer_changed):
                self._make_optimizer()  # new objective => reset old curvature history
            self._active_batch_mode = s.batch_mode
        lg = self.experiment_logger
        if lg is not None and lg.enabled:
            if mode_changed:
                changed_uid = next((uid for uid, before, after in (
                    ("training.train_mode", previous_lora[0] if previous_lora else None, lora_sig[0]),
                    ("training.lora_rank", previous_lora[1] if previous_lora else None, lora_sig[1]),
                    ("training.lora_alpha", previous_lora[2] if previous_lora else None, lora_sig[2]))
                    if before != after), "training.train_mode")
                cause = lg.last_change(changed_uid)
                if merge_existing and previous_lora and previous_lora[0] != "Full network":
                    lg.event("trainer.lora_merged", "trainer", {
                        "reason": "adaptation_configuration_changed", "previous": list(previous_lora),
                        "new": list(lora_sig)}, cause_seq=cause)
                lg.event("trainer.train_mode_applied", "trainer", {
                    "mode": lora_sig[0], "rank": lora_sig[1], "alpha": lora_sig[2],
                    "trainable_parameters": self.active_parameters,
                    "total_parameters": self.total_parameters}, cause_seq=cause)
            if mode_changed or optimizer_changed or (batch_mode_changed and s.optimizer == "L-BFGS"):
                cause = (lg.last_change("training.optimizer") if optimizer_changed else
                         lg.last_change("training.train_mode") if mode_changed else
                         lg.last_change("training.batch_mode"))
                lg.event("trainer.optimizer_recreated", "trainer", {
                    "optimizer": s.optimizer, "learning_rate": float(s.learning_rate),
                    "reason": "optimizer_changed" if optimizer_changed else
                              "train_mode_changed" if mode_changed else "batch_mode_changed",
                    "optimizer_revision": self.optimizer_revision}, cause_seq=cause)
            if batch_mode_changed:
                lg.event("trainer.batch_mode_applied", "trainer", {
                    "old": previous_batch, "new": s.batch_mode})

    @torch.no_grad()
    def sync_inference_models(self):
        with self.model_lock:
            for src_model, dest_model in ((self.enc, self.inf_enc), (self.dec, self.inf_dec)):
                for src, dst in zip(lora_layers(src_model), lora_layers(dest_model)):
                    if dst.lora_rank != src.lora_rank or dst.lora_alpha != src.lora_alpha:
                        dst.configure_lora(src.lora_rank, src.lora_alpha)
                    dst.lora_enabled = src.lora_enabled
                dest_model.load_state_dict(src_model.state_dict())
                for p in dest_model.parameters():
                    p.requires_grad_(False)
            self.inference_sync_revision += 1
            self.inference_sync_step = int(self.step)

    def _clear_history(self):
        self.hist[:] = np.nan
        self.hist_step = self.step

    def _record_history(self, total, kld, values, beta):
        """Store each active term's weighted contribution to the total loss for this step (Log tab)."""
        i = self.step % HISTORY_CAPACITY
        scale = float(train_settings.rec_loss_scale)
        col = self.hist[:, i]
        col[:] = np.nan
        for k, v in values.items():
            col[self.hist_row[k]] = scale * loss_cfg[k]["weight"] * v
        if beta != 0:
            col[self.hist_row["kl"]] = beta * kld
        col[self.hist_row["total"]] = total
        self.hist_step = self.step + 1

    def compute_beta(self, step):
        return float(train_settings.beta)

    def train_step(self, y_wave, beta):
        """Use a fixed VAE noise sample across all L-BFGS line-search closures."""
        s = train_settings
        B = y_wave.shape[0]
        cfgs = {k: dict(v) for k, v in loss_cfg.items()}
        rec_scale = float(s.rec_loss_scale)
        y = y_wave.squeeze(1)
        with torch.no_grad():
            y_mels = get_mels(y).detach()
        need_wave = any(cfgs[k]["weight"] != 0 for k in WAVEFORM_LOSS_KEYS)
        is_lbfgs = isinstance(self.opt, torch.optim.LBFGS)
        epsilon = self._lbfgs_epsilon if is_lbfgs else None
        latest = {}
        n_evals = 0

        def closure():
            nonlocal epsilon, n_evals
            self.opt.zero_grad(set_to_none=True)
            mu, std = self.enc(group_mels(y_mels))
            if epsilon is None or epsilon.shape != std.shape:
                epsilon = torch.randn_like(std)
            z = mu + std * epsilon
            kld = kl_loss(mu, std)
            yhat_mels = self.dec(ungroup_latents(z, B))[:, 0]
            yhat_wave = vocos_decode_diff(yhat_mels) if need_wave else None
            if need_wave and not yhat_wave.requires_grad:
                raise RuntimeError("Vocos output lacks gradient for waveform loss")
            ctx = LossContext(y if need_wave else None, yhat_wave, y_mels, yhat_mels)
            rec = 0.0
            values = {}
            for spec in LOSS_SPECS:
                key = spec["key"]
                weight = cfgs[key]["weight"]
                if weight == 0:
                    continue
                val = LOSS_FUNCS[key](ctx, cfgs[key])
                if not torch.isfinite(val).all():
                    raise FloatingPointError(f"Non-finite {key} loss")
                values[key] = val.detach()
                rec = rec + weight * val
            total = rec_scale * rec + float(beta) * kld
            if not torch.isfinite(total).all():
                raise FloatingPointError("Non-finite total loss")
            total.backward()
            if is_lbfgs:
                # Clipping alters the true gradient, confusing L-BFGS line search.
                if any(not torch.isfinite(p.grad).all() for p in self.params if p.grad is not None):
                    raise FloatingPointError("Non-finite gradient")
            else:
                torch.nn.utils.clip_grad_norm_(self.params, grad_clip, error_if_nonfinite=True)
            n_evals += 1
            latest["value"] = (total.detach(), kld.detach(), values)
            return total

        if is_lbfgs:
            # Optimizers with line search can leave candidate weights in place on error.
            before = [p.detach().clone() for p in self.params]
            try:
                self.opt.step(closure)
            except Exception:
                with torch.no_grad():
                    for p, old in zip(self.params, before):
                        p.copy_(old)
                self._make_optimizer()
                raise
            self._lbfgs_epsilon = epsilon.detach()
        else:
            closure()
            self.opt.step()
        self.stats["evals"] += n_evals
        total, kld, values = latest["value"]
        return total.item(), kld.item(), {k: v.item() for k, v in values.items()}

    def _training_metadata(self):
        return dict(step=self.step, optimizer=train_settings.optimizer,
                    train_mode=train_settings.train_mode,
                    lora_rank=train_settings.lora_rank,
                    lora_alpha=train_settings.lora_alpha,
                    learning_rate=train_settings.learning_rate,
                    batch_mode=train_settings.batch_mode,
                    batch_size=train_settings.batch_size,
                    lbfgs_max_iter=train_settings.lbfgs_max_iter,
                    lbfgs_history_size=train_settings.lbfgs_history_size,
                    lbfgs_hold_steps=train_settings.lbfgs_hold_steps)

    def save_checkpoint(self, trigger="manual"):
        folder = os.path.join(save_path, "weights")
        os.makedirs(folder, exist_ok=True)
        torch.save(self.enc.state_dict(), os.path.join(folder, f"encoder_weights_step_{self.step}.pt"))
        torch.save(self.dec.state_dict(), os.path.join(folder, f"decoder_weights_step_{self.step}.pt"))
        with open(os.path.join(folder, f"train_settings_step_{self.step}.json"), "w", encoding="utf-8") as f:
            json.dump(self._training_metadata(), f, indent=2)
        self.status_msg = f"Saved weights and LoRA settings at step {self.step}"
        lg = self.experiment_logger
        if lg is not None:
            lg.event("checkpoint.saved", "trainer", {
                "step": int(self.step), "trigger": trigger,
                "encoder": os.path.basename(f"encoder_weights_step_{self.step}.pt"),
                "decoder": os.path.basename(f"decoder_weights_step_{self.step}.pt"),
                "metadata": f"train_settings_step_{self.step}.json",
                "optimizer_state_saved": False, "rng_state_saved": False},
                cause_seq=self._pending_checkpoint_seq if trigger == "manual" else None)
        self._pending_checkpoint_seq = None

    def _restore_pair(self, enc_path, dec_path):
        enc_sd = torch.load(enc_path, map_location=device, weights_only=True)
        dec_sd = torch.load(dec_path, map_location=device, weights_only=True)
        basename = os.path.basename(enc_path)
        suffix = basename.split("encoder_weights_step_", 1)[-1]
        meta_path = os.path.join(os.path.dirname(enc_path), "train_settings_step_" + suffix.replace(".pt", ".json"))
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        adapter_key = next((k for k in enc_sd if k.endswith(".lora_A")), None)
        checkpoint_rank = int(enc_sd[adapter_key].shape[0]) if adapter_key else train_settings.lora_rank
        # Old v4 checkpoints have no adapter parameters; load in full mode.
        train_settings.lora_rank = int(metadata.get("lora_rank", checkpoint_rank))
        train_settings.lora_alpha = float(metadata.get("lora_alpha", train_settings.lora_rank))
        train_settings.train_mode = metadata.get("train_mode", "Full network")
        train_settings.optimizer = metadata.get("optimizer", train_settings.optimizer)
        for field in ("learning_rate", "batch_mode", "batch_size", "lbfgs_max_iter",
                      "lbfgs_history_size", "lbfgs_hold_steps"):
            if field in metadata:
                default = getattr(train_settings, field)
                setattr(train_settings, field, type(default)(metadata[field]))
        self._active_lora_settings = None
        self._active_optimizer_settings = None
        self._apply_configuration(merge_existing=False)
        for model, state in ((self.enc, enc_sd), (self.dec, dec_sd)):
            # Missing adapter keys in old v4 checkpoints must not retain stale adapters.
            for layer in lora_layers(model):
                with torch.no_grad():
                    layer.lora_B.zero_()
            bad = model.load_state_dict(state, strict=False)
            if bad.unexpected_keys or any(not k.endswith((".lora_A", ".lora_B")) for k in bad.missing_keys):
                raise RuntimeError(f"Checkpoint structure mismatch: {bad}")
        try:
            inferred_step = int(suffix.removesuffix(".pt"))
        except ValueError:
            inferred_step = self.step
        self.step = int(metadata.get("step", inferred_step))
        self._make_optimizer()
        self._clear_lbfgs_data()
        self.sync_inference_models()
        self._clear_history()
        self.last_loss_snapshot = None
        self.last_loss_step = None
        self.gui_config_revision += 1
        self.status_msg = f"Loaded {os.path.basename(enc_path)} ({train_settings.train_mode})"
        lg = self.experiment_logger
        if lg is not None:
            lg.new_generation("checkpoint_loaded", {"checkpoint": os.path.basename(enc_path),
                                                     "restored_step": int(self.step)})
            lg.event("checkpoint.loaded", "trainer", {
                "encoder": os.path.basename(enc_path), "decoder": os.path.basename(dec_path),
                "step": int(self.step), "train_mode": train_settings.train_mode,
                "optimizer_state_restored": False})

    def _training_batch(self):
        s = train_settings
        if s.optimizer != "L-BFGS":
            return store.full_batch() if s.batch_mode.startswith("Full") else store.sample_batch(s.batch_size)
        v = store.version
        hold = max(1, s.lbfgs_hold_steps)
        needs_new = (self._lbfgs_batch is None or self._lbfgs_data_version != v or
                     (s.batch_mode == "Random windows" and self._lbfgs_batch_steps >= hold))
        if needs_new:
            self._lbfgs_batch = store.full_batch() if s.batch_mode.startswith("Full") else store.sample_batch(s.batch_size)
            self._lbfgs_data_version = v
            self._lbfgs_batch_steps = 0
            self._lbfgs_epsilon = None
            self._make_optimizer()  # reset Hessian history when objective/data changes
        self._lbfgs_batch_steps += 1
        return self._lbfgs_batch

    def run(self):
        ema = 0.98
        last_t, last_step = time.time(), 0
        while self.running:
            try:
                if self.reset_req:
                    self.reset_req = False
                    self.enc, self.dec = build_models()
                    self._active_lora_settings = None
                    self._active_optimizer_settings = None
                    self._apply_configuration(merge_existing=False)
                    self.step = 0
                    self._clear_history()
                    self.sync_inference_models()
                    self.last_loss_snapshot = None
                    self.last_loss_step = None
                    if self.experiment_logger is not None:
                        self.experiment_logger.new_generation("model_reset", {"reset_to_step": 0})
                    self.status_msg = "Model reset (current optimizer and adaptation mode)"
                if self.load_req:
                    enc_path, self.load_req = self.load_req, None
                    dec_path = os.path.join(os.path.dirname(enc_path),
                                            os.path.basename(enc_path).replace("encoder", "decoder"))
                    self._restore_pair(enc_path, dec_path)
                if self.merge_lora_req:
                    self.merge_lora_req = False
                    for model in (self.enc, self.dec):
                        for layer in lora_layers(model):
                            layer.merge_lora()
                    self._make_optimizer()
                    self._clear_lbfgs_data()
                    self.sync_inference_models()
                    self.status_msg = "LoRA merged into CNN weights; fresh zero-output adapters"
                    if self.experiment_logger is not None:
                        self.experiment_logger.event("trainer.lora_merged", "trainer", {
                            "reason": "manual_merge", "mode": train_settings.train_mode,
                            "rank": train_settings.lora_rank, "optimizer_revision": self.optimizer_revision},
                            cause_seq=self._pending_merge_seq)
                    self._pending_merge_seq = None
                self._apply_configuration()
                if self.save_req:
                    self.save_req = False
                    self.save_checkpoint(trigger="manual")
                if train_settings.paused:
                    time.sleep(0.05)
                    continue
                # If every effective loss term is off, do not run the optimizer:
                # momentum and AdamW weight decay could still change the weights.
                rec_enabled = (train_settings.rec_loss_scale != 0 and any(
                    c["weight"] != 0 for c in loss_cfg.values()))
                kl_enabled = train_settings.beta != 0
                if not rec_enabled and not kl_enabled:
                    if not self._loss_suspended:
                        self._loss_suspended = True
                        if self.experiment_logger is not None:
                            self.experiment_logger.event("training.suspended", "trainer", {
                                "reason": "all_loss_weights_zero"})
                    self.stats["sps"] = 0.0
                    self.stats["total"] = 0.0
                    self.stats["beta"] = self.compute_beta(self.step)
                    self.status_msg = "All loss weights disabled: optimizer updates suspended"
                    time.sleep(0.05)
                    continue
                if self._loss_suspended:
                    self._loss_suspended = False
                    if self.experiment_logger is not None:
                        self.experiment_logger.event("training.resumed", "trainer", {"reason": "loss_enabled"})
                if self.status_msg == "All loss weights disabled: optimizer updates suspended":
                    self.status_msg = "Loss active: optimizer updates resumed"
                try:
                    batch = self._training_batch()
                except ValueError as exc:
                    self.status_msg = str(exc)
                    time.sleep(0.2)
                    continue
                if batch is None:
                    self.status_msg = "No enabled training audio"
                    time.sleep(0.2)
                    continue
                for group in self.opt.param_groups:
                    group['lr'] = train_settings.learning_rate
                beta = self.compute_beta(self.step)
                total, kld, values = self.train_step(batch, beta)
                self._record_history(total, kld, values, beta)
                self.step += 1
                self.last_loss_snapshot = {
                    "total": float(total), "kl_raw": float(kld), "beta": float(beta),
                    "losses_raw": dict(values), "measurement": "last_training_batch_pre_update"}
                self.last_loss_step = int(self.step)
                st = self.stats
                if self.step == 1:
                    st.update(total=total, kld=kld)
                st["total"] = ema * st["total"] + (1 - ema) * total
                st["kld"] = ema * st["kld"] + (1 - ema) * kld
                st["beta"] = beta
                for k in self.loss_ema:
                    if k in values:
                        prev = self.loss_ema[k]
                        self.loss_ema[k] = values[k] if prev == 0.0 else ema * prev + (1 - ema) * values[k]
                    else:
                        self.loss_ema[k] = 0.0
                if self.step % max(1, train_settings.sync_interval) == 0:
                    self.sync_inference_models()
                if autosave_interval_steps and self.step % autosave_interval_steps == 0:
                    self.save_checkpoint(trigger="autosave")
                if self.experiment_logger is not None and self.experiment_logger.enabled:
                    self.experiment_logger.metric(self.logging_metrics())
                now = time.time()
                if now - last_t > 1.0:
                    st["sps"] = (self.step - last_step) / (now - last_t)
                    last_t, last_step = now, self.step
            except Exception as exc:
                traceback.print_exc()
                self.status_msg = f"Training error: {exc} (see console)"
                signature = (type(exc).__name__, str(exc))
                if signature != self._last_logged_error:
                    self._last_logged_error = signature
                    if self.experiment_logger is not None:
                        self.experiment_logger.event("trainer.error", "trainer", {
                            "exception": signature[0], "message": signature[1][:1000]})
                time.sleep(0.5)

    def logging_metrics(self):
        """Small CPU-only snapshot, not a new model evaluation or a plotted history dump."""
        last = self.last_loss_snapshot or {}
        with store.lock:
            enabled_count = sum(bool(ok) for ok in store.enabled)
            data_revision = int(store.version)
        return {
            "measurement": "last_training_batch_pre_update",
            "last_loss_step": self.last_loss_step,
            "optimizer_updates_suspended": bool(self._loss_suspended or train_settings.paused),
            "total": round(float(last["total"]), 7) if last else None,
            "kl_raw": round(float(last["kl_raw"]), 7) if last else None,
            "beta": float(train_settings.beta),
            "losses_raw": {k: round(float(v), 7) for k, v in last.get("losses_raw", {}).items()},
            "optimizer": train_settings.optimizer,
            "optimizer_revision": self.optimizer_revision,
            "optimizer_evaluations_total": int(self.stats["evals"]),
            "train_mode": train_settings.train_mode,
            "learning_rate": float(train_settings.learning_rate),
            "updates_per_second_ema": round(float(self.stats["sps"]), 3),
            "enabled_entry_count": int(enabled_count),
            "data_revision": data_revision,
            "inference_sync_step": int(self.inference_sync_step),
            "inference_sync_revision": int(self.inference_sync_revision),
            "playback_underruns": int(getattr(getattr(self.experiment_logger, "engine", None), "underruns", 0)),
            "device_underflows": int(getattr(getattr(self.experiment_logger, "engine", None), "device_underflows", 0)),
        }

# -------------------------------------------------------------------------------------------------
# Real-time inference
# -------------------------------------------------------------------------------------------------

class InferenceEngine:
    """Rolling-mel-buffer inference: encode 8-frame blocks, decode causally, Vocos on the whole buffer."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.q = queue.Queue()
        self.running = True
        self.pos = 0
        self.leftover = np.zeros(0, dtype=np.float32)
        self.stream = None
        self.device_id = default_audio_output_device
        self.state_lock = threading.Lock()
        self.mel_buffer = None
        self.z_hist = None
        self.prev_tail = None        # tail of the previous Vocos run, crossfaded into the next one
        self.starved = True          # True while the output has no audio (start / after an underrun)
        self.last_sample = 0.0
        self.underruns = 0
        self.device_underflows = 0
        self._playback_start_time = None
        self._playback_start_underruns = 0
        self._playback_start_underflows = 0
        self.reset_state()

    def reset_state(self):
        with self.state_lock:
            self.pos = 0
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.leftover = np.zeros(0, dtype=np.float32)
            self.prev_tail = None
            self.starved = True
            self.mel_buffer = torch.full((1, mel_filter_count, inference_buffer_frames), mel_floor, device=device)
            self.z_hist = None
            with self.trainer.model_lock:
                for m in self.trainer.inf_dec.modules():
                    if hasattr(m, "streaming"):
                        m.streaming = decoder_streaming
                        m.cache = None

    @torch.no_grad()
    def _generate_step(self):
        """Generates one contiguous audio segment of inference_step_samples samples."""
        s = inf_settings
        k = inference_blocks_per_step
        idx = min(max(0, s.file_index), len(store.src_waves) - 1)
        wave = store.src_waves[idx]
        n = wave.shape[0]
        st = (int(s.start_sec * audio_sample_rate) // audio_hop) * audio_hop
        en = min(int(s.end_sec * audio_sample_rate), n)
        st = min(st, max(0, en - 1))
        need = k * block_samples + stft_context_frames * audio_hop
        if self.pos < st or self.pos + need > en:
            self.pos = st

        # slice b starts one STFT context before block b; zero outside the file
        starts = self.pos - stft_context_frames * audio_hop + torch.arange(k, device=device) * block_samples
        ix = starts[:, None] + torch.arange(slice_samples, device=device)[None, :]
        valid = (ix >= 0) & (ix < n)
        x = wave[ix.clamp(0, n - 1)] * valid
        self.pos += inference_step_samples

        # only the central audio_mel_count_vae frames of each slice reach the encoder
        mels = get_mels(x)[..., stft_context_frames:stft_context_frames + audio_mel_count_vae]    # (k, F, 8)
        with self.trainer.model_lock:
            mu, std = self.trainer.inf_enc(mels.unsqueeze(1))
            z = mu if s.use_mean else Encoder.reparameterize(mu, std)
            z = ungroup_latents(z, 1)                                                              # (1, Cz, k*latent_steps_per_group)
            if s.latent_noise > 0:
                z = z + torch.randn_like(z) * s.latent_noise
            if decoder_streaming:
                new_mels = self.trainer.inf_dec(z)[:, 0]                                           # (1, F, step_frames)
            else:
                self.z_hist = z if self.z_hist is None else torch.cat([self.z_hist, z], dim=-1)[..., -inference_decoder_context_steps:]
                new_mels = self.trainer.inf_dec(self.z_hist)[:, 0][..., -inference_step_frames:]

        self.mel_buffer = torch.cat([self.mel_buffer[..., inference_step_frames:], new_mels], dim=-1)
        wav = vocos.decode(self.mel_buffer).reshape(-1)                                            # (buffer_frames-1)*hop samples

        # frame f is centred at sample f*hop: play the newest step that lies lookahead_frames before the right edge
        seg_end = (inference_buffer_frames - inference_lookahead_frames) * audio_hop
        ov = inference_crossfade_samples
        half = ov // 2
        # Every step is a fresh Vocos run on the whole buffer. Run t supplies [a - half, a + step + half); the first
        # and last `ov` samples overlap with the neighbouring runs and are crossfaded instead of hard-joined.
        ext = wav[seg_end - inference_step_samples - half: seg_end + half] * s.gain
        head, mid, tail = ext[:ov], ext[ov:-ov], ext[-ov:]
        fade = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, ov, device=device))
        if self.prev_tail is None:
            blended = head * fade                                   # start from silence
        else:
            blended = self.prev_tail * (1 - fade) + head * fade     # old run -> new run over the same samples
        self.prev_tail = tail
        seg = torch.cat([blended, mid])                             # exactly inference_step_samples long
        knee = 0.9                                                  # soft limiter instead of hard clipping
        a_ = seg.abs()
        seg = torch.sign(seg) * (a_.clamp(max=knee) + (1 - knee) * torch.tanh((a_ - knee).clamp(min=0) / (1 - knee)))
        return seg.cpu().numpy().astype(np.float32)

    def producer(self):
        n_done, t_sum = 0, 0.0
        while self.running:
            if self.q.qsize() >= max_queued_blocks or len(store.src_waves) == 0:
                time.sleep(0.005)
                continue
            try:
                t0 = time.time()
                with self.state_lock:
                    seg = self._generate_step()
                self.q.put(seg)
                t_sum += time.time() - t0
                n_done += 1
                if n_done % 100 == 0:
                    print(f"inference: avg {t_sum / n_done * 1000:.1f} ms per {inference_step_samples / audio_sample_rate * 1000:.0f} ms of audio")
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)

    def callback(self, outdata, frames, time_info, status):
        if status.output_underflow:
            self.device_underflows += 1
        n = 0
        while n < frames:
            if self.leftover.shape[0] == 0:
                try:
                    self.leftover = self.q.get_nowait()
                except queue.Empty:
                    break
                if self.starved:                       # audio resumes after a gap: fade in, no hard edge
                    m = min(FADE_SAMPLES, self.leftover.shape[0])
                    self.leftover[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)
                    self.starved = False
            take = min(frames - n, self.leftover.shape[0])
            outdata[n:n + take, 0] = self.leftover[:take]
            self.leftover = self.leftover[take:]
            n += take
        if n > 0:
            self.last_sample = float(outdata[n - 1, 0])
        if n < frames:
            if not self.starved:                       # real underrun: ramp to zero instead of cutting
                self.underruns += 1
                self.starved = True
                m = min(frames - n, FADE_SAMPLES)
                outdata[n:n + m, 0] = self.last_sample * np.linspace(1.0, 0.0, m, dtype=np.float32)
                outdata[n + m:, 0] = 0.0
            else:
                outdata[n:, 0] = 0.0

    def start(self):
        if self.stream is not None:
            return
        try:
            self.reset_state()
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.leftover = np.zeros(0, dtype=np.float32)
            self.stream = sd.OutputStream(samplerate=audio_sample_rate, device=self.device_id, channels=1,
                                          dtype='float32', blocksize=1024, latency=0.15,
                                          callback=self.callback)
            self.stream.start()
            self._playback_start_time = time.monotonic()
            self._playback_start_underruns = self.underruns
            self._playback_start_underflows = self.device_underflows
            lg = self.trainer.experiment_logger
            if lg is not None:
                index = int(inf_settings.file_index)
                lg.event("playback.started", "audio", {
                    "source_index": index,
                    "source_name": store.src_names[index] if 0 <= index < len(store.src_names) else None,
                    "range_seconds": [float(inf_settings.start_sec), float(inf_settings.end_sec)],
                    "mean_latent": bool(inf_settings.use_mean),
                    "noise": float(inf_settings.latent_noise),
                    "gain": float(inf_settings.gain),
                    "output_device_id": self.device_id,
                    "inference_sync_step": self.trainer.inference_sync_step,
                    "inference_sync_revision": self.trainer.inference_sync_revision})
        except Exception as e:
            print("Audio start error:", e)
            self.stream = None
            if self.trainer.experiment_logger is not None:
                self.trainer.experiment_logger.event("playback.error", "audio", {"message": str(e)[:1000]})

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            lg = self.trainer.experiment_logger
            if lg is not None:
                index = int(inf_settings.file_index)
                lg.event("playback.stopped", "audio", {
                    "source_index": index,
                    "source_name": store.src_names[index] if 0 <= index < len(store.src_names) else None,
                    "duration_s": round(time.monotonic() - self._playback_start_time, 3)
                                  if self._playback_start_time is not None else None,
                    "underruns_delta": int(self.underruns - self._playback_start_underruns),
                    "device_underflows_delta": int(self.device_underflows - self._playback_start_underflows)})
            self._playback_start_time = None

    def set_device(self, device_id):
        self.device_id = device_id
        if self.stream is not None:
            self.stop()
            self.start()

# -------------------------------------------------------------------------------------------------
# GUI
# -------------------------------------------------------------------------------------------------

# One widget for numeric settings: draggable slider, exact value editor, and
# user-editable visible range. The original numeric constraints remain hard
# bounds for safety; the slider's visible min/max can be changed within them.
SLIDER_RESOLUTION = 2000


def _slider_position_to_value(pos, lower, upper, logarithmic=False, integer=False):
    fraction = min(1.0, max(0.0, float(pos) / SLIDER_RESOLUTION))
    if logarithmic:
        result = math.exp(math.log(lower) + fraction * (math.log(upper) - math.log(lower)))
    else:
        result = lower + fraction * (upper - lower)
    return int(round(result)) if integer else result


def _value_to_slider_position(value, lower, upper, logarithmic=False):
    if logarithmic:
        frac = (math.log(max(value, lower)) - math.log(lower)) / (math.log(upper) - math.log(lower))
    else:
        frac = (value - lower) / (upper - lower)
    return int(round(SLIDER_RESOLUTION * min(1.0, max(0.0, frac))))


class NumericRangeSlider(QtWidgets.QWidget):
    """QSlider + exact spin input + expandable, editable min/max range.

    `.value()`, `.setValue()`, `.setRange()`, `.valueChanged`, `.blockSignals()`
    behave like the numeric spin boxes used in older GUI versions. A log scale
    is supported (used for the learning-rate slider).
    """
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, value, lower, upper, step=1.0, decimals=3,
                 integer=False, logarithmic=False, hard_lower=None, hard_upper=None):
        super().__init__()
        self.integer = bool(integer)
        self.logarithmic = bool(logarithmic)
        self.decimals = 0 if self.integer else int(decimals)
        self.step = max(1 if self.integer else 10**(-self.decimals), step)
        self.hard_lower = float(lower if hard_lower is None else hard_lower)
        self.hard_upper = float(upper if hard_upper is None else hard_upper)
        if self.logarithmic and self.hard_lower <= 0:
            raise ValueError("Logarithmic slider requires strictly positive minimum")
        if self.hard_upper <= self.hard_lower:
            raise ValueError("Numeric slider requires a nonempty hard range")
        self.lower = max(self.hard_lower, float(lower))
        self.upper = min(self.hard_upper, float(upper))
        if self.upper <= self.lower:
            self.upper = min(self.hard_upper, self.lower + self.step)
        self._value = float(value)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 1, 0, 1)
        outer.setSpacing(1)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, SLIDER_RESOLUTION)
        self.slider.setMinimumWidth(130)
        self.slider.setToolTip("Logarithmic scale" if self.logarithmic else "Drag to adjust value")
        row.addWidget(self.slider, 1)
        self.editor = self._make_spin()
        self.editor.setRange(
            int(round(self.hard_lower)) if self.integer else self.hard_lower,
            int(round(self.hard_upper)) if self.integer else self.hard_upper)
        self.editor.setSingleStep(self.step)
        self.editor.setMinimumWidth(112 if self.decimals >= 6 else 85)
        self.editor.setToolTip("Exact numeric value; typing beyond the visible slider range expands its range")
        row.addWidget(self.editor)
        self.range_toggle = QtWidgets.QPushButton("Range...")
        self.range_toggle.setCheckable(True)
        self.range_toggle.setMaximumWidth(74)
        self.range_toggle.setToolTip("Edit the minimum and maximum used by this slider")
        row.addWidget(self.range_toggle)
        outer.addLayout(row)

        self.range_panel = QtWidgets.QWidget()
        range_row = QtWidgets.QHBoxLayout(self.range_panel)
        range_row.setContentsMargins(2, 2, 2, 2)
        range_row.setSpacing(6)
        range_row.addWidget(QtWidgets.QLabel("Slider min:"))
        self.min_editor = self._make_spin()
        self.min_editor.setRange(
            int(round(self.hard_lower)) if self.integer else self.hard_lower,
            int(round(self.hard_upper)) if self.integer else self.hard_upper)
        self.min_editor.setSingleStep(self.step)
        self.min_editor.setValue(int(round(self.lower)) if self.integer else self.lower)
        range_row.addWidget(self.min_editor)
        range_row.addWidget(QtWidgets.QLabel("Slider max:"))
        self.max_editor = self._make_spin()
        self.max_editor.setRange(
            int(round(self.hard_lower)) if self.integer else self.hard_lower,
            int(round(self.hard_upper)) if self.integer else self.hard_upper)
        self.max_editor.setSingleStep(self.step)
        self.max_editor.setValue(int(round(self.upper)) if self.integer else self.upper)
        range_row.addWidget(self.max_editor)
        if self.logarithmic:
            range_row.addWidget(QtWidgets.QLabel("log scale"))
        range_row.addStretch(1)
        self.range_panel.setVisible(False)
        outer.addWidget(self.range_panel)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.editor.valueChanged.connect(self._on_exact_changed)
        self.min_editor.valueChanged.connect(self._on_range_edited)
        self.max_editor.valueChanged.connect(self._on_range_edited)
        self.range_toggle.toggled.connect(self.range_panel.setVisible)
        self.setValue(value)

    def _make_spin(self):
        if self.integer:
            return QtWidgets.QSpinBox()
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(self.decimals)
        return spin

    def _sync_editors(self):
        self.slider.blockSignals(True)
        self.slider.setValue(_value_to_slider_position(
            self._value, self.lower, self.upper, self.logarithmic))
        self.slider.blockSignals(False)
        self.editor.blockSignals(True)
        self.editor.setValue(int(round(self._value)) if self.integer else self._value)
        self.editor.blockSignals(False)

    def _set_visible_bounds(self, lower, upper, clamp_current=False):
        lower = max(self.hard_lower, float(lower))
        upper = min(self.hard_upper, float(upper))
        min_gap = 1 if self.integer else 10**(-self.decimals)
        if upper - lower < min_gap * 0.5:
            return False
        if self.logarithmic and lower <= 0:
            return False
        self.lower, self.upper = lower, upper
        for spin, value in ((self.min_editor, lower), (self.max_editor, upper)):
            spin.blockSignals(True)
            spin.setValue(int(round(value)) if self.integer else value)
            spin.blockSignals(False)
        if clamp_current:
            self.setValue(max(lower, min(upper, self._value)))
        else:
            self._sync_editors()
        return True

    def _on_range_edited(self, *_):
        new_lower, new_upper = self.min_editor.value(), self.max_editor.value()
        if new_upper > new_lower and (not self.logarithmic or new_lower > 0):
            self._set_visible_bounds(new_lower, new_upper, clamp_current=True)
        # An incomplete range edit remains visible until the user enters a valid bound.

    def _on_slider_changed(self, position):
        self.setValue(_slider_position_to_value(
            position, self.lower, self.upper, self.logarithmic, self.integer))

    def _on_exact_changed(self, value):
        self.setValue(value)

    def value(self):
        return int(round(self._value)) if self.integer else float(self._value)

    def setValue(self, value):
        new = min(self.hard_upper, max(self.hard_lower, float(value)))
        if self.integer:
            new = float(int(round(new)))
        else:
            # Keep the actual training parameter equal to the visible number.
            new = round(new, self.decimals)
        if new < self.lower or new > self.upper:
            # Programmatic changes (e.g. selecting L-BFGS with lr=1) and exact
            # numeric entry remain possible without silently clamping to the
            # slider's current display range.
            self._set_visible_bounds(min(self.lower, new), max(self.upper, new))
        changed = not math.isclose(new, self._value, rel_tol=0.0, abs_tol=1e-12)
        self._value = new
        self._sync_editors()
        if changed:
            self.valueChanged.emit(float(new))

    def setRange(self, lower, upper):
        # Used for dynamic audio lengths; unlike the Range... editor, this
        # also resets the numeric widget's current visible limits.
        lower = float(lower)
        upper = float(upper)
        if upper <= lower:
            upper = lower + max(self.step, 10**(-self.decimals))
        lower = max(self.hard_lower, lower)
        upper = min(self.hard_upper, upper)
        self._set_visible_bounds(lower, upper, clamp_current=True)


def dspin(value, lo, hi, step, decimals=3, *, visible_min=None, visible_max=None,
          logarithmic=False):
    return NumericRangeSlider(
        value,
        lo if visible_min is None else visible_min,
        hi if visible_max is None else visible_max,
        step=step, decimals=decimals, logarithmic=logarithmic,
        hard_lower=lo, hard_upper=hi)


def ispin(value, lo, hi, *, visible_min=None, visible_max=None, step=1):
    return NumericRangeSlider(
        value,
        lo if visible_min is None else visible_min,
        hi if visible_max is None else visible_max,
        step=step, integer=True,
        hard_lower=lo, hard_upper=hi)


# -------------------------------------------------------------------------------------------------
# GUI helpers: element registry (common-tab mirroring), blue group style, vispy loss log
# -------------------------------------------------------------------------------------------------

GROUP_STYLE = ("QGroupBox { border: 2px solid #3985bd; border-radius: 7px; margin-top: 9px; "
               "padding-top: 6px; } "
               "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0px 4px; }")

GUI_CATEGORIES = [("data", "Data selection"), ("inference", "Live inference"),
                  ("training", "Live training"), ("losses", "Loss functions"), ("log", "Training log"),
                  ("session", "Session recording"), ("replay", "Session replay")]


def _get_value(w):
    if isinstance(w, NumericRangeSlider):
        return w.value()
    if isinstance(w, QtWidgets.QCheckBox):
        return w.isChecked()
    return w.currentText()


def _set_value(w, v):
    if isinstance(w, NumericRangeSlider):
        w.setValue(float(v))
    elif isinstance(w, QtWidgets.QCheckBox):
        w.setChecked(bool(v))
    else:
        w.setCurrentText(str(v))


def _value_signal(w):
    if isinstance(w, NumericRangeSlider):
        return w.valueChanged
    if isinstance(w, QtWidgets.QCheckBox):
        return w.toggled
    return w.currentTextChanged


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def combo_factory(items, current):
    def make():
        c = QtWidgets.QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        return c
    return make


def checkbox_factory(checked):
    def make():
        c = QtWidgets.QCheckBox()
        c.setChecked(bool(checked))
        return c
    return make


class GuiElement:
    """kind 'value': slider / checkbox / combo with a value.  kind 'widget': composite widget (file list, plot, buttons)."""
    def __init__(self, uid, category, label, common_label, factory, primary, order,
                 items_fn=None, kind="value", update_fn=None):
        self.uid, self.category, self.label, self.common_label = uid, category, label, common_label
        self.factory, self.primary, self.order, self.items_fn = factory, primary, order, items_fn
        self.kind, self.update_fn = kind, update_fn
        self.mirror = None
        self.mirror_row = None
        self.toggle = None



def _copy_aux_widget(source, destination, *, preserve_active_edit=False):
    """Synchronize a mirrored composite-panel widget without emitting GUI actions.

    Unlike ordinary scalar settings, session/replay panels contain buttons,
    labels, edit boxes, and transient playback state.  Copy only when different
    to avoid disrupting the cursor and partly entered spinbox values.
    """
    previous = destination.blockSignals(True)
    try:
        if isinstance(source, QtWidgets.QDoubleSpinBox):
            if (destination.minimum() != source.minimum() or
                    destination.maximum() != source.maximum()):
                destination.setRange(source.minimum(), source.maximum())
            if destination.value() != source.value():
                destination.setValue(source.value())
        elif isinstance(source, QtWidgets.QComboBox):
            if destination.currentIndex() != source.currentIndex():
                destination.setCurrentIndex(source.currentIndex())
        elif isinstance(source, QtWidgets.QCheckBox):
            if destination.isChecked() != source.isChecked():
                destination.setChecked(source.isChecked())
        elif isinstance(source, QtWidgets.QLineEdit):
            if not (preserve_active_edit and destination.hasFocus()):
                if destination.text() != source.text():
                    destination.setText(source.text())
        elif isinstance(source, QtWidgets.QProgressBar):
            if (destination.minimum(), destination.maximum()) != (source.minimum(), source.maximum()):
                destination.setRange(source.minimum(), source.maximum())
            if destination.value() != source.value():
                destination.setValue(source.value())
        elif isinstance(source, QtWidgets.QAbstractButton):
            if destination.text() != source.text():
                destination.setText(source.text())
        elif isinstance(source, QtWidgets.QLabel):
            if destination.text() != source.text():
                destination.setText(source.text())
        if destination.isEnabled() != source.isEnabled():
            destination.setEnabled(source.isEnabled())
        if destination.toolTip() != source.toolTip():
            destination.setToolTip(source.toolTip())
    finally:
        destination.blockSignals(previous)


class ExperimentSessionMirror(QtWidgets.QGroupBox):
    """Independent Common-tab controls driving the one original session logger."""
    def __init__(self, gui):
        super().__init__("Experiment session logging and bookmarks")
        self.gui = gui
        outer = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.record_toggle = QtWidgets.QCheckBox("Record session")
        self.record_toggle.toggled.connect(gui.record_toggle.setChecked)
        row.addWidget(self.record_toggle)
        self.bookmark_tag = QtWidgets.QComboBox()
        self.bookmark_tag.addItems(["Moment", "Observation", "Hypothesis", "Comparison", "Problem"])
        self.bookmark_tag.currentIndexChanged.connect(gui.bookmark_tag.setCurrentIndex)
        row.addWidget(self.bookmark_tag)
        self.bookmark_note = QtWidgets.QLineEdit()
        self.bookmark_note.setPlaceholderText("Optional note about what you hear / are testing")
        self.bookmark_note.textEdited.connect(gui.bookmark_note.setText)
        self.bookmark_note.returnPressed.connect(gui.add_bookmark)
        row.addWidget(self.bookmark_note, 1)
        self.bookmark_btn = QtWidgets.QPushButton("Bookmark")
        self.bookmark_btn.clicked.connect(gui.add_bookmark)
        row.addWidget(self.bookmark_btn)
        self.folder_btn = QtWidgets.QPushButton("Open log folder")
        self.folder_btn.clicked.connect(gui.open_session_folder)
        row.addWidget(self.folder_btn)
        outer.addLayout(row)
        self.logging_status = QtWidgets.QLabel()
        self.logging_status.setWordWrap(True)
        self.logging_status.setMinimumWidth(0)
        self.logging_status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        outer.addWidget(self.logging_status)
        self.sync_from_primary()

    def sync_from_primary(self):
        gui = self.gui
        for name in ("record_toggle", "bookmark_tag", "bookmark_note",
                     "bookmark_btn", "folder_btn", "logging_status"):
            _copy_aux_widget(getattr(gui, name), getattr(self, name),
                             preserve_active_edit=(name == "bookmark_note"))


class ExperimentReplayMirror(QtWidgets.QGroupBox):
    """Fully interactive Common-tab copy of the existing replay panel.

    Every edit is delegated to the original widgets/handlers. No second replay
    timer or second ReplaySession is created, so replay and recording do not
    diverge when the user switches between tabs.
    """
    def __init__(self, gui):
        super().__init__("Replay a recorded experiment")
        self.gui = gui
        outer = QtWidgets.QVBoxLayout(self)

        first = QtWidgets.QHBoxLayout()
        self.replay_load_btn = QtWidgets.QPushButton("Load event log...")
        self.replay_load_btn.clicked.connect(gui.load_replay_log)
        first.addWidget(self.replay_load_btn)
        self.replay_audio_btn = QtWidgets.QPushButton("Audio folder...")
        self.replay_audio_btn.clicked.connect(gui.choose_replay_audio_folder)
        first.addWidget(self.replay_audio_btn)
        self.replay_file_label = QtWidgets.QLabel()
        self.replay_file_label.setWordWrap(True)
        self.replay_file_label.setMinimumWidth(0)
        self.replay_file_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        first.addWidget(self.replay_file_label, 1)
        outer.addLayout(first)

        interval = QtWidgets.QHBoxLayout()
        self.replay_from = QtWidgets.QDoubleSpinBox()
        self.replay_to = QtWidgets.QDoubleSpinBox()
        for w, primary in ((self.replay_from, gui.replay_from),
                           (self.replay_to, gui.replay_to)):
            w.setDecimals(primary.decimals())
            w.setSuffix(primary.suffix())
            w.setSingleStep(primary.singleStep())
            w.valueChanged.connect(primary.setValue)
        interval.addWidget(QtWidgets.QLabel("From:"))
        interval.addWidget(self.replay_from)
        interval.addWidget(QtWidgets.QLabel("To:"))
        interval.addWidget(self.replay_to)
        self.replay_speed = QtWidgets.QComboBox()
        self.replay_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.replay_speed.currentIndexChanged.connect(gui.replay_speed.setCurrentIndex)
        interval.addWidget(QtWidgets.QLabel("Speed:"))
        interval.addWidget(self.replay_speed)
        self.replay_audio_mode = QtWidgets.QComboBox()
        self.replay_audio_mode.addItems(["Follow recorded play/stop", "Always listen"])
        self.replay_audio_mode.currentIndexChanged.connect(gui.replay_audio_mode.setCurrentIndex)
        interval.addWidget(self.replay_audio_mode)
        outer.addLayout(interval)

        options = QtWidgets.QHBoxLayout()
        self.replay_train = QtWidgets.QCheckBox("Train while replaying")
        self.replay_train.toggled.connect(gui.replay_train.setChecked)
        options.addWidget(self.replay_train)
        self.replay_reset = QtWidgets.QCheckBox("Allow model reset / LoRA merge")
        self.replay_reset.toggled.connect(gui.replay_reset.setChecked)
        options.addWidget(self.replay_reset)
        self.replay_go = QtWidgets.QPushButton("Start replay")
        self.replay_go.clicked.connect(gui.start_replay)
        options.addWidget(self.replay_go)
        self.replay_pause_btn = QtWidgets.QPushButton("Pause replay")
        self.replay_pause_btn.clicked.connect(gui.toggle_replay_pause)
        options.addWidget(self.replay_pause_btn)
        self.replay_stop_btn = QtWidgets.QPushButton("Stop replay")
        self.replay_stop_btn.clicked.connect(lambda: gui._finish_replay("user_stopped"))
        options.addWidget(self.replay_stop_btn)
        outer.addLayout(options)

        self.replay_progress = QtWidgets.QProgressBar()
        outer.addWidget(self.replay_progress)
        self.replay_status = QtWidgets.QLabel()
        self.replay_status.setWordWrap(True)
        self.replay_status.setMinimumWidth(0)
        self.replay_status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        outer.addWidget(self.replay_status)
        self.replay_bookmark_label = QtWidgets.QLabel()
        self.replay_bookmark_label.setWordWrap(True)
        outer.addWidget(self.replay_bookmark_label)
        self.sync_from_primary()

    def sync_from_primary(self):
        gui = self.gui
        for name in ("replay_load_btn", "replay_audio_btn", "replay_file_label",
                     "replay_from", "replay_to", "replay_speed", "replay_audio_mode",
                     "replay_train", "replay_reset", "replay_go", "replay_pause_btn",
                     "replay_stop_btn", "replay_progress", "replay_status",
                     "replay_bookmark_label"):
            _copy_aux_widget(getattr(gui, name), getattr(self, name))


class AudioFilesPanel(QtWidgets.QWidget):
    """Checkable list of training entries plus the 'Add audio files...' button (can exist twice: tab + Common)."""
    toggled = QtCore.pyqtSignal(int, bool)
    selected = QtCore.pyqtSignal(int)
    add_requested = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(200)
        self.list.itemChanged.connect(
            lambda item: self.toggled.emit(self.list.row(item), item.checkState() == QtCore.Qt.Checked))
        self.list.currentRowChanged.connect(self.selected)
        self.add_btn = QtWidgets.QPushButton("Add audio files...")
        self.add_btn.clicked.connect(lambda *_: self.add_requested.emit())
        lay.addWidget(self.list)
        lay.addWidget(self.add_btn)

    def set_entries(self, entries, current_row=None):
        self.list.blockSignals(True)
        self.list.clear()
        for name, enabled in entries:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if enabled else QtCore.Qt.Unchecked)
            self.list.addItem(item)
        if current_row is not None and 0 <= current_row < len(entries):
            self.list.setCurrentRow(current_row)
        self.list.blockSignals(False)

    def set_checked(self, row, on):
        item = self.list.item(row)
        if item is not None:
            self.list.blockSignals(True)
            item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
            self.list.blockSignals(False)

    def set_current(self, row):
        self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)


LOSS_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
               "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
               "#800000", "#aaffc3", "#808000", "#ffd8b1", "#5c7cfa", "#a9a9a9"]


def _downsample(steps, vals, max_points):
    """Block-average a long history to at most max_points samples (NaN-aware)."""
    n = len(vals)
    if n <= max_points:
        return steps, vals
    b = int(np.ceil(n / max_points))
    m = (n // b) * b
    v, s = vals[n - m:].reshape(-1, b), steps[n - m:].reshape(-1, b)
    ok = np.isfinite(v)
    cnt = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, v, 0.0).sum(1) / cnt
    return s.mean(1), mean


class LossHistoryPlot(QtWidgets.QWidget):
    """vispy line plot of every active loss term's weighted contribution over a sliding step window."""
    MAX_POINTS = 1500

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.keys = trainer.hist_keys
        self.colors = {k: LOSS_COLORS[i % len(LOSS_COLORS)] for i, k in enumerate(self.keys)}
        self.colors["kl"], self.colors["total"] = "#ffffff", "#808080"
        self.setMinimumHeight(300)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="#14181c", show=False)
        grid = self.canvas.central_widget.add_grid(margin=0, spacing=0)
        self.y_axis = scene.AxisWidget(orientation="left", axis_label="weighted contribution (log10 if log y)")
        self.y_axis.width_max = 95
        self.x_axis = scene.AxisWidget(orientation="bottom", axis_label="training step")
        self.x_axis.height_max = 50
        grid.add_widget(self.y_axis, row=0, col=0)
        grid.add_widget(self.x_axis, row=1, col=1)
        self.view = grid.add_view(row=0, col=1, border_color="#555555")
        self.view.camera = scene.PanZoomCamera(rect=(0, 0, 1000, 1))
        self.view.camera.interactive = False
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.lines = {}
        for k in self.keys:
            line = scene.visuals.Line(pos=np.zeros((2, 2), np.float32), color=self.colors[k],
                                      width=1.5, method="gl", parent=self.view.scene)
            line.visible = False
            self.lines[k] = line

        lay.addWidget(self.canvas.native, 1)
        self.legend = QtWidgets.QLabel("")
        self.legend.setTextFormat(QtCore.Qt.RichText)
        self.legend.setStyleSheet("font-family: monospace;")
        self.legend.setAlignment(QtCore.Qt.AlignTop)
        legend_box = QtWidgets.QGroupBox("Active terms (latest)")
        QtWidgets.QVBoxLayout(legend_box).addWidget(self.legend)
        legend_box.setFixedWidth(280)
        lay.addWidget(legend_box)

    def update_plot(self, window, show_total, log_y):
        tr = self.trainer
        end = int(tr.hist_step)
        n = int(min(window, end, HISTORY_CAPACITY))
        if n < 2:
            return
        steps = np.arange(end - n, end)
        data = tr.hist[:, steps % HISTORY_CAPACITY]          # (K, n) in one gather
        drawn, rows = [], []
        for k in self.keys:
            vals = data[tr.hist_row[k]]
            if (k == "total" and not show_total) or not np.isfinite(vals).any():
                self.lines[k].visible = False
                continue
            s, v = _downsample(steps, vals, self.MAX_POINTS)
            m = np.isfinite(v)
            if m.sum() < 2:
                self.lines[k].visible = False
                continue
            s, v = s[m], v[m]
            if log_y:
                v = np.log10(np.maximum(v, 1e-8))
            self.lines[k].set_data(pos=np.column_stack([s, v]).astype(np.float32))
            self.lines[k].visible = True
            drawn.append(v)
            last = vals[np.isfinite(vals)][-1]
            rows.append(f'<span style="color:{self.colors[k]}">&#9632;</span> {k:<11}{last:9.5f}')
        if drawn:
            lo = min(float(v.min()) for v in drawn)
            hi = max(float(v.max()) for v in drawn)
            y0, y1 = (lo - 0.1, hi + 0.1) if log_y else (0.0, max(hi * 1.05, 1e-9))
            self.view.camera.set_range(x=(end - n, end - 1), y=(y0, y1), margin=0)
        self.legend.setText("<br>".join(rows))
        self.canvas.update()


class LiveGUI(QtWidgets.QWidget):
    def __init__(self, trainer, engine, experiment_logger):
        super().__init__()
        self.setStyleSheet(GROUP_STYLE)          # blue boundary around every group box
        self.trainer, self.engine = trainer, engine
        self.experiment_logger = experiment_logger
        self._control_shadow = {}
        self._range_shadow = {}
        self._pending_controls = {}
        self._pending_ranges = {}
        self._source_override = None
        self._automatic_control = None
        self._selected_file_shadow = None
        self._suppress_file_selection_log = False
        self.loss_labels = {}
        self.elements = {}
        self.common_forms, self.common_groups = {}, {}
        self._src_snapshot = []                  # names of the whole source files shown in the inference list
        self._loading = False
        self._ready = False
        self._last_gui_config_revision = trainer.gui_config_revision
        self._logging_start_trigger = None
        self.replay_session = None
        self.replay_audio_root = os.path.abspath(audio_data_path)
        self.replay_timer = QtCore.QTimer(self)
        self.replay_timer.setInterval(25)
        self.replay_timer.timeout.connect(self._replay_tick)
        self._replay_running = False
        self._replay_paused = False
        self._replay_events = []
        self._replay_index = 0
        self._replay_start_ms = 0
        self._replay_end_ms = 0
        self._replay_clock = 0.0
        self._replay_elapsed_ms = 0
        self._replay_speed_factor = 1.0
        self._replay_preexisting_pause = False
        self._replay_playback_wanted = False
        self._replay_training_paused = False
        self._replay_warnings = []
        self.setWindowTitle("Live Audio VAE v20: train, listen, and replay sessions")
        self.resize(1180, 900)
        self.build_ui()
        self._reset_control_shadows()
        self.refresh_files()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(250)
        self.event_flush_timer = QtCore.QTimer(self)
        self.event_flush_timer.timeout.connect(self._flush_due_controls)
        self.event_flush_timer.start(150)
        self._update_logging_status()

    # ---------------------------------------------------------------- element registry
    def _make_toggle(self, elem):
        toggle = QtWidgets.QCheckBox("common")
        toggle.setToolTip("Show this control also in the Common tab")
        toggle.toggled.connect(lambda on, e=elem: self._set_common(e, on))
        elem.toggle = toggle
        return toggle

    def add_element(self, form, category, uid, label, factory, on_change=None,
                    common=False, common_label=None, items_fn=None):
        """Create a setting with a 'common' toggle on its left; the toggle mirrors it into the Common tab."""
        w = factory()
        order = sum(1 for e in self.elements.values() if e.category == category)
        elem = GuiElement(uid, category, label, common_label or label, factory, w, order, items_fn)
        self.elements[uid] = elem
        if on_change is not None:
            _value_signal(w).connect(on_change)
        _value_signal(w).connect(lambda *_, e=elem: self._primary_changed(e))
        self._attach_numeric_logging(elem, w, mirror=False)

        toggle = self._make_toggle(elem)
        left = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(toggle)
        lay.addWidget(QtWidgets.QLabel(label + ":"))
        form.addRow(left, w)
        if common:
            toggle.setChecked(True)
        return w

    def add_widget_element(self, layout, category, uid, label, factory, update_fn=None, common=False):
        """Composite widget (file list, plot, buttons) with a 'common' toggle row above it; the mirror is a second instance."""
        w = factory()
        order = sum(1 for e in self.elements.values() if e.category == category)
        elem = GuiElement(uid, category, label, label, factory, w, order, kind="widget", update_fn=update_fn)
        self.elements[uid] = elem
        toggle = self._make_toggle(elem)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(toggle)
        row.addWidget(QtWidgets.QLabel(label))
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(w)
        if common:
            toggle.setChecked(True)
        return w

    def _register_common_toolbar(self, panel, layout, category, uid, label, factory):
        """Register the original panel's Common toggle in its OWN tab only.

        Its Common-tab representation is a copy of the panel contents, never a
        copy of this checkbox. The shared element registry owns visibility.
        """
        order = sum(1 for e in self.elements.values() if e.category == category)
        elem = GuiElement(uid, category, label, label, factory, panel,
                          order, kind="widget", update_fn=lambda mirror: mirror.sync_from_primary())
        self.elements[uid] = elem
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._make_toggle(elem))
        row.addWidget(QtWidgets.QLabel("Show this panel in the Common tab"))
        row.addStretch(1)
        layout.insertLayout(0, row)
        return elem

    def _sync_aux_panels(self):
        """Refresh mirrors of stateful composite panels (progress, buttons, notes)."""
        for uid in ("session.panel", "replay.panel"):
            e = self.elements.get(uid)
            if e is not None and e.mirror is not None:
                e.mirror.sync_from_primary()

    def _set_common(self, e, on):
        form = self.common_forms[e.category]
        if on and e.mirror is None:
            m = e.factory()
            e.mirror = m
            pos = sum(1 for o in self.elements.values()
                      if o.category == e.category and o.mirror is not None and o.order < e.order)
            if e.kind == "widget":
                container = QtWidgets.QWidget()
                cl = QtWidgets.QVBoxLayout(container)
                cl.setContentsMargins(0, 0, 0, 0)
                cl.addWidget(QtWidgets.QLabel(e.common_label + ":"))
                cl.addWidget(m)
                e.mirror_row = container
                if e.update_fn is not None:
                    e.update_fn(m)
                form.insertRow(pos, container)
            else:
                self._sync_mirror(e, bounds=True)
                _value_signal(m).connect(lambda *_, el=e: self._mirror_changed(el))
                self._attach_numeric_logging(e, m, mirror=True)
                form.insertRow(pos, e.common_label + ":", m)
        elif not on and e.mirror is not None:
            if e.kind == "widget":
                form.removeRow(e.mirror_row)
                e.mirror_row = None
            else:
                form.removeRow(e.mirror)       # removes and deletes the row (label + widget)
            e.mirror = None
        self.common_groups[e.category].setVisible(form.rowCount() > 0)
        if self._ready and not self._loading:
            self.experiment_logger.event("ui.common_visibility", "user", {
                "id": e.uid, "visible": bool(on)})

    def _sync_mirror(self, e, bounds=False):
        m = e.mirror
        if m is None or e.kind == "widget":
            return
        m.blockSignals(True)
        try:
            if isinstance(m, QtWidgets.QComboBox) and e.items_fn is not None:
                m.clear()
                m.addItems(e.items_fn())
            if bounds and isinstance(m, NumericRangeSlider):
                m._set_visible_bounds(e.primary.lower, e.primary.upper)
            _set_value(m, _get_value(e.primary))
        finally:
            m.blockSignals(False)

    def _primary_changed(self, e):
        if e.mirror is not None and not _same(_get_value(e.mirror), _get_value(e.primary)):
            self._sync_mirror(e)
        self._capture_control_change(e)

    def _mirror_changed(self, e):
        if e.mirror is not None and not _same(_get_value(e.mirror), _get_value(e.primary)):
            self._source_override = "Common"
            try:
                _set_value(e.primary, _get_value(e.mirror))   # fires primary handlers
            finally:
                self._source_override = None

    def refresh_mirrors(self, *uids):
        for uid in uids:
            self._sync_mirror(self.elements[uid], bounds=True)

    def set_element_silently(self, uid, value):
        e = self.elements[uid]
        e.primary.blockSignals(True)
        _set_value(e.primary, value)
        e.primary.blockSignals(False)
        self._sync_mirror(e)

    # ---------------------------------------------------------------- experiment event recording
    def _reset_control_shadows(self):
        """Replace logging shadows after a bulk restore or new logging session."""
        self._pending_controls.clear()
        self._pending_ranges.clear()
        self._control_shadow = {uid: _get_value(e.primary) for uid, e in self.elements.items()
                                if e.kind == "value"}
        self._range_shadow = {uid: [float(e.primary.lower), float(e.primary.upper)]
                              for uid, e in self.elements.items()
                              if isinstance(e.primary, NumericRangeSlider)}

    def _attach_numeric_logging(self, e, w, mirror=False):
        if not isinstance(w, NumericRangeSlider):
            return
        w.slider.sliderReleased.connect(lambda uid=e.uid: self._flush_control(uid))
        w.editor.editingFinished.connect(lambda uid=e.uid: self._flush_control(uid))
        for editor in (w.min_editor, w.max_editor):
            editor.valueChanged.connect(lambda *_, elem=e, widget=w, is_mirror=mirror:
                                        self._on_slider_range_changed(elem, widget, is_mirror))
            editor.editingFinished.connect(lambda uid=e.uid: self._flush_range(uid))

    def _control_tab(self):
        if self._source_override is not None:
            return self._source_override
        if not hasattr(self, "tabs"):
            return None
        i = self.tabs.currentIndex()
        return self.tabs.tabText(i) if i >= 0 else None

    def _capture_control_change(self, e, immediate=False):
        if e.kind != "value":
            return None
        uid = e.uid
        value = _get_value(e.primary)
        old = self._control_shadow.get(uid, value)
        self._control_shadow[uid] = value
        if not self._ready or self._loading or _same(old, value):
            return None
        if not self.experiment_logger.enabled:
            return None
        auto = self._automatic_control
        via = (auto.get("via", "automatic") if auto is not None else
               "slider_or_exact_input" if isinstance(e.primary, NumericRangeSlider) else
               "checkbox" if isinstance(e.primary, QtWidgets.QCheckBox) else "combobox")
        origin = auto.get("origin", "gui") if auto else "user"
        data = {"id": uid, "old": old, "new": value, "via": via,
                "tab": self._control_tab()}
        if uid == "inference.file":
            data["source_index"] = self.inf_file.currentIndex()
        cause = auto.get("cause_seq") if auto else None
        if isinstance(e.primary, NumericRangeSlider) and not immediate and not auto:
            pending = self._pending_controls.get(uid)
            if pending is None:
                self._pending_controls[uid] = {"data": data, "origin": origin,
                                                "cause": cause, "last": time.monotonic(),
                                                "start_step": int(self.trainer.step)}
            else:
                pending["data"]["new"] = value
                pending["last"] = time.monotonic()
            return None
        self._flush_control(uid)
        return self.experiment_logger.event("control.set", origin, data, cause_seq=cause)

    def _flush_control(self, uid):
        pending = self._pending_controls.pop(uid, None)
        if pending is None or not self.experiment_logger.enabled:
            return None
        data = pending["data"]
        if _same(data["old"], data["new"]):
            return None
        data["start_step"] = pending["start_step"]
        data["end_step"] = int(self.trainer.step)
        return self.experiment_logger.event("control.set", pending["origin"], data,
                                            cause_seq=pending["cause"])

    def _on_slider_range_changed(self, e, w, is_mirror):
        if is_mirror and e.mirror is w:
            # A display-range edit in Common must be reproducible in the primary tab.
            e.primary._set_visible_bounds(w.lower, w.upper)
        uid = e.uid
        new = [float(e.primary.lower), float(e.primary.upper)]
        old = self._range_shadow.get(uid, new)
        self._range_shadow[uid] = new
        if not self._ready or self._loading or old == new or not self.experiment_logger.enabled:
            return
        pending = self._pending_ranges.get(uid)
        if pending is None:
            self._pending_ranges[uid] = {"old": old, "new": new,
                                         "tab": "Common" if is_mirror else self._control_tab(),
                                         "last": time.monotonic()}
        else:
            pending["new"] = new
            pending["last"] = time.monotonic()

    def _flush_range(self, uid):
        pending = self._pending_ranges.pop(uid, None)
        if pending is None or not self.experiment_logger.enabled:
            return
        if pending["old"] != pending["new"]:
            self.experiment_logger.event("ui.slider_range_changed", "user", {
                "id": uid, "old": pending["old"], "new": pending["new"],
                "tab": pending["tab"]})

    def _flush_due_controls(self):
        now = time.monotonic()
        for uid, item in list(self._pending_controls.items()):
            if now - item["last"] >= 0.45:
                self._flush_control(uid)
        for uid, item in list(self._pending_ranges.items()):
            if now - item["last"] >= 0.45:
                self._flush_range(uid)

    def _flush_all_changes(self):
        for uid in list(self._pending_controls):
            self._flush_control(uid)
        for uid in list(self._pending_ranges):
            self._flush_range(uid)

    def _mapped_tab_index(self, index):
        """Map old session/replay tab indices to the single v19 Session log tab."""
        try:
            index = int(index)
        except (TypeError, ValueError):
            return -1
        if index == 7:  # v18 Replay tab, now merged into Session log
            index = 6
        return index if 0 <= index < self.tabs.count() else -1

    def _on_tab_changed(self, index):
        if self._ready and not self._loading and index >= 0:
            self.experiment_logger.event("ui.tab_selected", "user", {
                "tab": self.tabs.tabText(index), "index": int(index)})

    @staticmethod
    def _source_description(index, path):
        desc = {"id": f"source_{index}", "name": os.path.basename(path)}
        try:
            stat = os.stat(path)
            desc["size_bytes"] = int(stat.st_size)
            desc["mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            desc["file_exists"] = False
        return desc

    def _log_entry_description(self, index):
        with store.lock:
            if not 0 <= index < len(store.names):
                return None
            src, offset = store.refs[index]
            return {"index": int(index), "name": store.names[index],
                    "source_id": f"source_{src}", "offset_samples": int(offset),
                    "length_samples": int(store.waves[index].shape[0]),
                    "enabled": bool(store.enabled[index])}

    def _logging_initial_state(self):
        gui = self.collect_gui_config()
        # Replace absolute file paths with portable source IDs and metadata.
        layout = gui.get("dataset", {})
        layout["sources"] = [self._source_description(i, path)
                             for i, path in enumerate(layout.get("sources", []))]
        with store.lock:
            for i, source in enumerate(layout["sources"]):
                if i < len(store.src_waves):
                    source["decoded_samples_48k"] = int(store.src_waves[i].numel())
        with store.lock:
            enabled = [i for i, on in enumerate(store.enabled) if on]
        return {
            "format": "live_audio_vae_experiment_initial_state", "version": 1,
            "gui": gui, "training": asdict(train_settings),
            "inference": asdict(inf_settings),
            "loss_configuration": {k: dict(v) for k, v in loss_cfg.items()},
            "model": {"step": int(self.trainer.step),
                      "inference_sync_step": int(self.trainer.inference_sync_step),
                      "train_mode": train_settings.train_mode,
                      "optimizer": train_settings.optimizer,
                      "optimizer_state_archived": False,
                      "weights_archived": False},
            "data": {"enabled_entry_indices": enabled, "sample_rate": audio_sample_rate},
            "playback": {"running_at_start": self.engine.stream is not None,
                         "source_index": int(inf_settings.file_index),
                         "underruns": int(self.engine.underruns),
                         "device_underflows": int(self.engine.device_underflows)},
            "environment": {"device": device, "torch": torch.__version__,
                            "vocos_checkpoint": "kittn/vocos-mel-48khz-alpha1"},
        }

    def build_logging_toolbar(self, root):
        box = QtWidgets.QGroupBox("Experiment session logging and bookmarks")
        outer = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.record_toggle = QtWidgets.QCheckBox("Record session")
        self.record_toggle.setToolTip("Recording is OFF at startup. Check to start a session; uncheck to stop it.")
        self.record_toggle.toggled.connect(self._toggle_recording)
        row.addWidget(self.record_toggle)
        self.bookmark_tag = QtWidgets.QComboBox()
        self.bookmark_tag.addItems(["Moment", "Observation", "Hypothesis", "Comparison", "Problem"])
        row.addWidget(self.bookmark_tag)
        self.bookmark_note = QtWidgets.QLineEdit()
        self.bookmark_note.setPlaceholderText("Optional note about what you hear / are testing")
        self.bookmark_note.setToolTip("Saved with timestamp, training step, playback source and current model settings")
        self.bookmark_note.returnPressed.connect(self.add_bookmark)
        row.addWidget(self.bookmark_note, 1)
        self.bookmark_btn = QtWidgets.QPushButton("Bookmark")
        self.bookmark_btn.setEnabled(False)
        self.bookmark_btn.clicked.connect(self.add_bookmark)
        row.addWidget(self.bookmark_btn)
        self.folder_btn = QtWidgets.QPushButton("Open log folder")
        self.folder_btn.setEnabled(False)
        self.folder_btn.clicked.connect(self.open_session_folder)
        row.addWidget(self.folder_btn)
        outer.addLayout(row)
        self.logging_status = QtWidgets.QLabel("Logging OFF. No experiment actions are being recorded.")
        self.logging_status.setWordWrap(True)
        self.logging_status.setMinimumWidth(0)
        self.logging_status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        outer.addWidget(self.logging_status)
        self._register_common_toolbar(
            box, outer, "session", "session.panel", "Experiment session and bookmarks",
            lambda: ExperimentSessionMirror(self))
        root.addWidget(box)

    def start_logging_on_launch(self):
        """Optional explicit startup recording for automated experiments (not default)."""
        self._logging_start_trigger = "application_started"
        try:
            self.record_toggle.setChecked(True)
        finally:
            self._logging_start_trigger = None

    def _toggle_recording(self, on):
        if on:
            try:
                self._reset_control_shadows()
                trigger = self._logging_start_trigger or "user_enabled"
                folder = self.experiment_logger.start(self._logging_initial_state(), trigger=trigger)
                if self.engine.stream is not None:
                    self.experiment_logger.event("playback.already_running", "audio", {
                        "source_index": int(inf_settings.file_index),
                        "source_name": store.src_names[inf_settings.file_index]
                                       if 0 <= inf_settings.file_index < len(store.src_names) else None})
                self.experiment_logger.metric(self.trainer.logging_metrics(), force=True)
                self.logging_status.setToolTip(folder)
            except Exception as exc:
                traceback.print_exc()
                self.record_toggle.blockSignals(True)
                self.record_toggle.setChecked(False)
                self.record_toggle.blockSignals(False)
                QtWidgets.QMessageBox.warning(self, "Could not start logger", str(exc))
        else:
            self._flush_all_changes()
            self.experiment_logger.metric(self.trainer.logging_metrics(), force=True)
            self.experiment_logger.stop("user_disabled")
        self._update_logging_status()

    def _update_logging_status(self):
        lg = self.experiment_logger
        if not hasattr(self, "logging_status"):
            return
        self.bookmark_btn.setEnabled(lg.enabled)
        self.folder_btn.setEnabled(bool(lg.session_dir))
        if lg.writer_error:
            self.logging_status.setText(lg.writer_error)
        elif lg.enabled:
            self.logging_status.setText(
                f"RECORDING | {os.path.basename(lg.session_dir)} | "
                f"{lg.events_written} events, {lg.metrics_written} metrics, "
                f"{lg.bookmarks} bookmarks | metrics dropped: {lg.dropped_metrics}")
        elif lg.session_dir:
            self.logging_status.setText(
                f"Logging OFF | last session: {lg.session_dir} | "
                f"{lg.bookmarks} bookmarks")
        else:
            self.logging_status.setText("Logging OFF. No experiment actions are being recorded.")

    def open_session_folder(self):
        path = self.experiment_logger.session_dir
        if path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(path)))

    # ---------------------------------------------------------------- experiment replay
    def build_replay_toolbar(self, root):
        box = QtWidgets.QGroupBox("Replay a recorded experiment (actions + live listening)")
        outer = QtWidgets.QVBoxLayout(box)
        first = QtWidgets.QHBoxLayout()
        self.replay_load_btn = QtWidgets.QPushButton("Load event log...")
        self.replay_load_btn.clicked.connect(self.load_replay_log)
        first.addWidget(self.replay_load_btn)
        self.replay_audio_btn = QtWidgets.QPushButton("Audio folder...")
        self.replay_audio_btn.setToolTip("Choose the folder containing the original audio recordings")
        self.replay_audio_btn.clicked.connect(self.choose_replay_audio_folder)
        first.addWidget(self.replay_audio_btn)
        self.replay_file_label = QtWidgets.QLabel("No replay log loaded")
        self.replay_file_label.setToolTip("Select a recorded session's events.jsonl")
        self.replay_file_label.setWordWrap(True)
        self.replay_file_label.setMinimumWidth(0)
        self.replay_file_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        first.addWidget(self.replay_file_label, 1)
        outer.addLayout(first)

        interval = QtWidgets.QHBoxLayout()
        self.replay_from = QtWidgets.QDoubleSpinBox()
        self.replay_to = QtWidgets.QDoubleSpinBox()
        for w in (self.replay_from, self.replay_to):
            w.setDecimals(1)
            w.setRange(0.0, 0.0)
            w.setSuffix(" s")
            w.setSingleStep(1.0)
        interval.addWidget(QtWidgets.QLabel("From:"))
        interval.addWidget(self.replay_from)
        interval.addWidget(QtWidgets.QLabel("To:"))
        interval.addWidget(self.replay_to)
        self.replay_speed = QtWidgets.QComboBox()
        self.replay_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.replay_speed.setCurrentText("1x")
        interval.addWidget(QtWidgets.QLabel("Speed:"))
        interval.addWidget(self.replay_speed)
        self.replay_audio_mode = QtWidgets.QComboBox()
        self.replay_audio_mode.addItems(["Follow recorded play/stop", "Always listen"])
        self.replay_audio_mode.setToolTip("Follow recorded listening, or force audio on throughout the selected interval")
        interval.addWidget(self.replay_audio_mode)
        outer.addLayout(interval)

        options = QtWidgets.QHBoxLayout()
        self.replay_train = QtWidgets.QCheckBox("Train while replaying")
        self.replay_train.setChecked(True)
        self.replay_train.setToolTip("Run optimization while action timing is replayed; results are not bit-identical")
        options.addWidget(self.replay_train)
        self.replay_reset = QtWidgets.QCheckBox("Allow model reset / LoRA merge")
        self.replay_reset.setChecked(False)
        self.replay_reset.setToolTip("Potentially destructive actions; disabled by default. Load starting weights manually first.")
        options.addWidget(self.replay_reset)
        self.replay_go = QtWidgets.QPushButton("Start replay")
        self.replay_go.clicked.connect(self.start_replay)
        self.replay_go.setEnabled(False)
        options.addWidget(self.replay_go)
        self.replay_pause_btn = QtWidgets.QPushButton("Pause replay")
        self.replay_pause_btn.setEnabled(False)
        self.replay_pause_btn.clicked.connect(self.toggle_replay_pause)
        options.addWidget(self.replay_pause_btn)
        self.replay_stop_btn = QtWidgets.QPushButton("Stop replay")
        self.replay_stop_btn.setEnabled(False)
        self.replay_stop_btn.clicked.connect(lambda: self._finish_replay("user_stopped"))
        options.addWidget(self.replay_stop_btn)
        outer.addLayout(options)
        self.replay_progress = QtWidgets.QProgressBar()
        self.replay_progress.setRange(0, 1000)
        self.replay_progress.setValue(0)
        outer.addWidget(self.replay_progress)
        self.replay_status = QtWidgets.QLabel(
            "Event replay uses the CURRENT model weights (or a checkpoint you load manually). "
            "Logs do not contain original model/optimizer states or audio. Training will sound different "
            "unless the starting conditions match. A replay starting mid-session restarts the "
            "selected listening passage, not the exact historic playhead position.")
        self.replay_status.setWordWrap(True)
        self.replay_status.setMinimumWidth(0)
        self.replay_status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        outer.addWidget(self.replay_status)
        self.replay_bookmark_label = QtWidgets.QLabel("Recorded bookmark: none yet")
        self.replay_bookmark_label.setWordWrap(True)
        outer.addWidget(self.replay_bookmark_label)
        self._register_common_toolbar(
            box, outer, "replay", "replay.panel", "Experiment replay controls",
            lambda: ExperimentReplayMirror(self))
        root.addWidget(box)

    def load_replay_log(self):
        if self._replay_running:
            return
        default = (self.experiment_logger.session_dir or
                   os.path.join(save_path, "sessions"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select experiment events.jsonl", default, "JSON Lines (*.jsonl)")
        if not path:
            return
        try:
            replay = ReplaySession.load(path)
        except (OSError, ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid replay log", str(exc))
            return
        self.replay_session = replay
        duration = replay.duration_ms / 1000.0
        for w in (self.replay_from, self.replay_to):
            w.setRange(0.0, max(0.1, duration))
        self.replay_from.setValue(0.0)
        self.replay_to.setValue(duration)
        self.replay_file_label.setText(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))
        self.replay_file_label.setToolTip(path)
        self.replay_go.setEnabled(True)
        self.replay_progress.setValue(0)
        self.replay_status.setText(
            f"Loaded {len(replay.events)} replayable events, {len(replay.bookmarks)} bookmarks, "
            f"{duration:.1f} s. Select a time range and press Start replay. "
            "Original audio must be available; original model weights are NOT in this log. " +
            ("WARNING: this older log lacks an initial snapshot." if replay.initial_state is None else ""))

    def choose_replay_audio_folder(self):
        if self._replay_running:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Find the original audio recordings", self.replay_audio_root)
        if path:
            self.replay_audio_root = os.path.abspath(path)
            self.replay_audio_btn.setToolTip(self.replay_audio_root)
            self.replay_status.setText(f"Replay audio folder: {self.replay_audio_root}")

    def _replay_locate_source(self, desc):
        """Resolve a log's portable source descriptor, without trusting unrecorded absolute paths."""
        if isinstance(desc, str):
            name, expected = portable_audio_basename(desc), None
            # Older logs can carry paths: accept only their basename under known folders.
        elif isinstance(desc, dict):
            name = portable_audio_basename(desc.get("name", ""))
            expected = desc.get("size_bytes")
        else:
            return None
        if not name or name in (".", ".."):
            return None
        candidates = [os.path.join(self.replay_audio_root, name),
                      os.path.join(os.path.abspath(audio_data_path), name)]
        with store.lock:
            candidates.extend(p for p in store.src_paths if os.path.basename(p) == name)
        seen = set()
        for path in candidates:
            path = os.path.abspath(path)
            if path in seen:
                continue
            seen.add(path)
            if not os.path.isfile(path):
                continue
            if expected is not None:
                try:
                    if os.stat(path).st_size != int(expected):
                        continue
                except (OSError, TypeError, ValueError):
                    continue
            return path
        return None

    def _replay_resolve_layout(self, layout):
        resolved = dict(layout)
        sources = []
        for desc in layout.get("sources", []):
            path = self._replay_locate_source(desc)
            if path is None:
                raise FileNotFoundError("Missing or different audio source: " +
                                        (desc.get("name", "?") if isinstance(desc, dict) else str(desc)) +
                                        ". Select its original folder using Audio folder...")
            sources.append(path)
        resolved["sources"] = sources
        return resolved

    def _replay_apply_snapshot(self, state):
        """Load a *complete effective* GUI+dataset state, never a checkpoint or model step."""
        cfg = state.get("gui", {})
        layout = cfg.get("dataset")
        if not isinstance(cfg.get("elements"), dict):
            raise ValueError("Replay snapshot has no GUI elements")
        if not isinstance(layout, dict):
            raise ValueError("Replay snapshot has no dataset")
        resolved = self._replay_resolve_layout(layout)
        with self.engine.state_lock:
            missing = store.restore_layout(resolved)
            if missing:
                raise FileNotFoundError("Audio not found: " + ", ".join(missing))
            self.engine.pos = 0
        self._src_snapshot = []
        self.refresh_files(selected_index=0)
        # Apply source before source start/end; record includes sliders' display ranges.
        elements = cfg["elements"]
        ordered = ["inference.file", "inference.start", "inference.end"]
        ordered += [uid for uid in elements if uid not in ordered]
        for uid in ordered:
            d = elements.get(uid)
            e = self.elements.get(uid)
            if e is None or not isinstance(d, dict):
                continue
            try:
                if e.kind == "value":
                    if "range" in d and isinstance(e.primary, NumericRangeSlider):
                        e.primary._set_visible_bounds(*d["range"])
                    if "value" in d:
                        _set_value(e.primary, d["value"])
                e.toggle.setChecked(bool(d.get("common", False)))
                self._sync_mirror(e, bounds=True)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                self._replay_warnings.append(f"Control {uid} could not be restored: {exc}")
        index = self._mapped_tab_index(cfg.get("active_tab", 0))
        if index >= 0:
            self.tabs.setCurrentIndex(index)
        self.apply_train_settings()
        self.apply_inf_settings()
        self._reset_control_shadows()
        # Neither the stored model step nor its optimizer/RNG state can be recreated here.
        self._replay_training_paused = bool(state.get("training", {}).get("paused", False))
        self._replay_playback_wanted = bool(state.get("playback", {}).get("running_at_start", False))

    def _replay_find_entry(self, description):
        if not isinstance(description, dict):
            return None
        index = description.get("index")
        with store.lock:
            if isinstance(index, int) and 0 <= index < len(store.names):
                sidx, offset = store.refs[index]
                if store.names[index] == description.get("name") and (
                        description.get("offset_samples") is None or
                        int(offset) == description.get("offset_samples")):
                    return index
            for i, name in enumerate(store.names):
                if name != description.get("name"):
                    continue
                source_idx, offset = store.refs[i]
                if int(offset) == description.get("offset_samples", offset):
                    return i
        return None

    def _replay_apply_event(self, e, pre_roll=False):
        """Execute one previously recorded event on the GUI thread. Never re-emit user events."""
        typ, data = e.get("type"), e.get("data", {})
        if not isinstance(data, dict):
            return
        if typ == "control.set":
            uid = data.get("id")
            element = self.elements.get(uid)
            if element is not None and element.kind == "value" and "new" in data:
                _set_value(element.primary, data["new"])
                self._sync_mirror(element)
        elif typ == "ui.slider_range_changed":
            element = self.elements.get(data.get("id"))
            bounds = data.get("new")
            if element and isinstance(element.primary, NumericRangeSlider) and isinstance(bounds, list) and len(bounds) == 2:
                element.primary._set_visible_bounds(bounds[0], bounds[1])
                self._sync_mirror(element, bounds=True)
        elif typ == "ui.common_visibility":
            element = self.elements.get(data.get("id"))
            if element:
                element.toggle.setChecked(bool(data.get("visible")))
        elif typ == "ui.tab_selected":
            index = self._mapped_tab_index(data.get("index", -1))
            if index >= 0:
                self.tabs.setCurrentIndex(index)
        elif typ == "ui.training_entry_selected":
            idx = self._replay_find_entry(data.get("entry"))
            if idx is not None:
                for panel in self.audio_panels():
                    panel.set_current(idx)
                self.on_file_selected(idx)
        elif typ == "data.entry_enabled":
            idx = self._replay_find_entry(data.get("entry"))
            if idx is None:
                self._replay_warnings.append(f"Missing training entry for toggle: {data.get('entry')}")
            else:
                self.on_panel_toggled(idx, bool(data.get("enabled")), self.file_panel)
                self.file_panel.set_checked(idx, bool(data.get("enabled")))
        elif typ == "data.entry_split":
            idx = self._replay_find_entry(data.get("original"))
            if idx is None:
                self._replay_warnings.append("Could not reproduce split: missing original entry")
            else:
                count = store.split_entry(idx, float(data.get("chunk_seconds", 0)))
                if count:
                    self.refresh_files(selected_index=idx)
        elif typ == "data.sources_imported":
            for desc in data.get("sources", []):
                # Skip if this specific import already exists (e.g. after a loaded config snapshot).
                existing = any(os.path.basename(p) == desc.get("name") for p in store.src_paths)
                if existing:
                    continue
                path = self._replay_locate_source(desc)
                if path is None:
                    self._replay_warnings.append(f"Missing imported audio: {desc.get('name')}")
                    continue
                store.add_file(path, chunk_seconds=float(desc.get("chunk_seconds", 0)), enabled=False)
            self.refresh_files()
        elif typ == "config.loaded":
            state = data.get("effective_state")
            if isinstance(state, dict):
                # Full v15+ effective state is the authoritative reconstruction.
                # Do not let optimization run through a potentially expensive
                # dataset reload or on partially applied settings.
                if not pre_roll:
                    self.on_pause(True)
                self._replay_apply_snapshot(state)
                if not pre_roll:
                    self.on_pause(True if not self.replay_train.isChecked() else self._replay_training_paused)
            elif isinstance(data.get("dataset"), dict):
                # v14 recorded the resulting dataset and changed GUI elements,
                # but not the complete unchanged configuration. Restore what is
                # known, explicitly retaining the current values for the rest.
                resolved = self._replay_resolve_layout(data["dataset"])
                with self.engine.state_lock:
                    missing = store.restore_layout(resolved)
                    if missing:
                        raise FileNotFoundError("Missing legacy log sources: " + ", ".join(missing))
                self._src_snapshot = []
                self.refresh_files(selected_index=0)
                diff = data.get("changed_elements", {})
                ordered = ["inference.file", "inference.start", "inference.end"]
                ordered += [uid for uid in diff if uid not in ordered]
                for uid in ordered:
                    elem = self.elements.get(uid)
                    target = diff.get(uid, {}).get("new")
                    if elem is None or not isinstance(target, dict):
                        continue
                    if elem.kind == "value" and "value" in target:
                        if isinstance(elem.primary, NumericRangeSlider) and "range" in target:
                            elem.primary._set_visible_bounds(*target["range"])
                        _set_value(elem.primary, target["value"])
                    if "common" in target:
                        elem.toggle.setChecked(bool(target["common"]))
                    self._sync_mirror(elem, bounds=True)
                self.apply_train_settings()
                self.apply_inf_settings()
                index = self._mapped_tab_index(data.get("current_active_tab", -1))
                if index >= 0:
                    self.tabs.setCurrentIndex(index)
                self._replay_warnings.append(
                    "Legacy config.loaded restored dataset and known changes only; unlogged controls are uncertain")
            else:
                self._replay_warnings.append("Old config.loaded lacks dataset or effective state; skipped")
        elif typ == "playback.source_changed":
            name = data.get("source_name")
            if name in store.src_names:
                self.inf_file.setCurrentIndex(store.src_names.index(name))
        elif typ == "playback.started":
            self._replay_playback_wanted = True
            name = data.get("source_name")
            if name in store.src_names:
                self.inf_file.setCurrentIndex(store.src_names.index(name))
            if "range_seconds" in data and isinstance(data["range_seconds"], list):
                self.inf_start.setValue(data["range_seconds"][0])
                self.inf_end.setValue(data["range_seconds"][1])
            self.inf_mean.setChecked(bool(data.get("mean_latent", self.inf_mean.isChecked())))
            self.inf_noise.setValue(float(data.get("noise", self.inf_noise.value())))
            self.inf_gain.setValue(float(data.get("gain", self.inf_gain.value())))
            self.apply_inf_settings()
            if not pre_roll and self.replay_audio_mode.currentIndex() == 0:
                self.engine.start()
        elif typ == "playback.stopped":
            self._replay_playback_wanted = False
            if not pre_roll and self.replay_audio_mode.currentIndex() == 0:
                self.engine.stop()
        elif typ in ("training.paused", "training.resumed"):
            if data.get("reason", "pause_button") == "pause_button":
                self._replay_training_paused = typ == "training.paused"
                if not pre_roll and self.replay_train.isChecked():
                    self.on_pause(self._replay_training_paused)
        elif typ == "model.reset_requested":
            if self.replay_reset.isChecked() and not pre_roll:
                self.reset_model()
            else:
                self._replay_warnings.append("Skipped historical model reset; original weights cannot be recovered")
        elif typ == "lora.merge_requested":
            if self.replay_reset.isChecked() and not pre_roll:
                self.request_merge_lora()
            else:
                self._replay_warnings.append("Skipped historical LoRA merge")
        elif typ == "checkpoint.load_requested":
            # Logs contain basename only; never guess a path or silently load an unrelated checkpoint.
            self._replay_warnings.append("Checkpoint load in log: load matching checkpoint manually")
        elif typ == "bookmark.added":
            if not pre_roll:
                label = data.get("label", "Moment")
                note = data.get("note", "")
                self.replay_bookmark_label.setText(
                    f"Recorded bookmark at {e['t_ms']/1000:.1f} s: {label} — {note}")

    def start_replay(self):
        replay = self.replay_session
        if replay is None or self._replay_running:
            return
        start_ms = int(round(self.replay_from.value() * 1000))
        end_ms = int(round(self.replay_to.value() * 1000))
        if end_ms <= start_ms:
            QtWidgets.QMessageBox.warning(self, "Invalid replay period", "End must be after start")
            return
        missing = replay.missing_sources(self._replay_locate_source, until_ms=end_ms)
        if missing:
            QtWidgets.QMessageBox.warning(self, "Original audio missing",
                "Select the folder containing the original recordings (matching file sizes):\n\n" +
                "\n".join(missing[:16]) + ("\n..." if len(missing) > 16 else ""))
            return
        self._flush_all_changes()
        self.replay_timer.stop()
        self._replay_warnings = list(replay.warnings)
        self._replay_events = replay.events_between(start_ms, end_ms)
        self._replay_index = 0
        self._replay_start_ms = start_ms
        self._replay_end_ms = end_ms
        self._replay_elapsed_ms = start_ms
        self._replay_speed_factor = float(self.replay_speed.currentText().removesuffix("x"))
        self._replay_running = True
        self._replay_paused = False
        self._replay_preexisting_pause = bool(train_settings.paused)
        # Stop audio and training while reconstructing settings at the interval start.
        with self.experiment_logger.mute_gui_events():
            self.engine.stop()
            self.on_pause(True)
        try:
            self._loading = True
            with self.experiment_logger.mute_gui_events():
                if replay.initial_state is not None:
                    self._replay_apply_snapshot(replay.initial_state)
                else:
                    self._replay_training_paused = self._replay_preexisting_pause
                    self._replay_playback_wanted = False
                for e in replay.state_before(start_ms):
                    self._replay_apply_event(e, pre_roll=True)
                # Reconstructed GUI state is reached instantly; model weights/steps cannot be fast-forwarded.
                self.on_pause(True if not self.replay_train.isChecked() else self._replay_training_paused)
                if self.replay_audio_mode.currentIndex() == 1 or self._replay_playback_wanted:
                    self.engine.start()
        except Exception as exc:
            traceback.print_exc()
            self._loading = False
            self._finish_replay("error")
            QtWidgets.QMessageBox.warning(self, "Could not start replay", str(exc))
            return
        finally:
            self._loading = False
        self._reset_control_shadows()
        self._replay_clock = time.monotonic()
        self.replay_go.setEnabled(False)
        self.replay_load_btn.setEnabled(False)
        self.replay_audio_btn.setEnabled(False)
        self.replay_pause_btn.setEnabled(True)
        self.replay_pause_btn.setText("Pause replay")
        self.replay_stop_btn.setEnabled(True)
        for widget in (self.replay_speed, self.replay_audio_mode, self.replay_train, self.replay_reset,
                       self.replay_from, self.replay_to):
            widget.setEnabled(False)
        self.experiment_logger.event("replay.started", "user", {
            "source_log": os.path.basename(replay.path), "period_ms": [start_ms, end_ms],
            "speed": self.replay_speed.currentText(), "train_during_replay": self.replay_train.isChecked(),
            "audio_mode": self.replay_audio_mode.currentText(), "model_weights_from_log": False})
        self.replay_timer.start()
        self.replay_bookmark_label.setText("Recorded bookmark: none yet")
        self.replay_status.setText(f"REPLAYING {start_ms/1000:.1f}–{end_ms/1000:.1f} s | "
                                   f"{len(self._replay_events)} events | current model, not original weights")

    def _replay_tick(self):
        if not self._replay_running or self._replay_paused:
            return
        speed = self._replay_speed_factor
        self._replay_elapsed_ms = min(
            self._replay_end_ms, int(self._replay_start_ms +
                                     (time.monotonic() - self._replay_clock) * speed * 1000))
        try:
            # Bounded per Qt tick: large logs do not block the UI or real-time audio.
            limit = 60
            count = 0
            while self._replay_index < len(self._replay_events) and count < limit:
                event = self._replay_events[self._replay_index]
                if event["t_ms"] > self._replay_elapsed_ms:
                    break
                self._loading = True
                try:
                    with self.experiment_logger.mute_gui_events():
                        self._replay_apply_event(event)
                finally:
                    self._loading = False
                self._replay_index += 1
                count += 1
        except Exception as exc:
            self._replay_warnings.append(str(exc))
            traceback.print_exc()
            self._finish_replay("error")
            return
        fraction = (self._replay_elapsed_ms - self._replay_start_ms) / max(
            1, self._replay_end_ms - self._replay_start_ms)
        self.replay_progress.setValue(round(fraction * 1000))
        if self._replay_elapsed_ms >= self._replay_end_ms and self._replay_index >= len(self._replay_events):
            self._finish_replay("completed")
        elif count:
            self.replay_status.setText(
                f"REPLAYING {self._replay_elapsed_ms/1000:.1f}/{self._replay_end_ms/1000:.1f} s "
                f"| {self._replay_index}/{len(self._replay_events)} events | "
                f"step now {self.trainer.step} (original timeline may differ)")

    def toggle_replay_pause(self):
        if not self._replay_running:
            return
        with self.experiment_logger.mute_gui_events():
            if not self._replay_paused:
                self._replay_paused = True
                self.replay_timer.stop()
                self._replay_pause_wall = time.monotonic()
                self.engine.stop()
                self.on_pause(True)
                self.replay_pause_btn.setText("Resume replay")
                self.replay_status.setText("Replay paused. Audio restarts its selected passage on resume.")
            else:
                pause_duration = time.monotonic() - self._replay_pause_wall
                self._replay_clock += pause_duration
                self._replay_paused = False
                if self.replay_train.isChecked():
                    self.on_pause(self._replay_training_paused)
                if self.replay_audio_mode.currentIndex() == 1 or self._replay_playback_wanted:
                    self.engine.start()
                self.replay_pause_btn.setText("Pause replay")
                self.replay_timer.start()

    def _finish_replay(self, reason):
        if not self._replay_running:
            return
        self.replay_timer.stop()
        self._replay_running = False
        self._replay_paused = False
        with self.experiment_logger.mute_gui_events():
            self.engine.stop()
            self.on_pause(True)  # explicit safety: end with model stopped, settings left at replayed values
        self._reset_control_shadows()
        self.replay_go.setEnabled(self.replay_session is not None)
        self.replay_load_btn.setEnabled(True)
        self.replay_audio_btn.setEnabled(True)
        self.replay_pause_btn.setEnabled(False)
        self.replay_pause_btn.setText("Pause replay")
        self.replay_stop_btn.setEnabled(False)
        for widget in (self.replay_speed, self.replay_audio_mode, self.replay_train, self.replay_reset,
                       self.replay_from, self.replay_to):
            widget.setEnabled(True)
        if reason == "completed":
            self.replay_progress.setValue(1000)
        warning_count = len(self._replay_warnings)
        self.replay_status.setText(
            f"Replay {reason}. Applied {self._replay_index}/{len(self._replay_events)} events. "
            "Playback stopped and training paused. Original model state was not reconstructed." +
            (f" {warning_count} warning(s): " + "; ".join(self._replay_warnings[:3]) if warning_count else ""))
        self.experiment_logger.event("replay.ended", "user" if reason == "user_stopped" else "system", {
            "reason": reason, "source_log": os.path.basename(self.replay_session.path),
            "processed_events": self._replay_index, "warnings": self._replay_warnings[:20]})

    @staticmethod
    def _index_ranges(indices):
        """Run-length representation of sorted enabled entry indices, inclusive ranges."""
        ranges = []
        for value in indices:
            if ranges and value == ranges[-1][1] + 1:
                ranges[-1][1] = int(value)
            else:
                ranges.append([int(value), int(value)])
        return ranges

    def add_bookmark(self):
        lg = self.experiment_logger
        if not lg.enabled:
            return
        self._flush_all_changes()
        idx = int(inf_settings.file_index)
        with store.lock:
            enabled = [i for i, on in enumerate(store.enabled) if on]
            src_name = store.src_names[idx] if 0 <= idx < len(store.src_names) else None
        data = {
            "label": self.bookmark_tag.currentText(),
            "note": self.bookmark_note.text().strip(),
            "source_id": f"source_{idx}" if src_name is not None else None,
            "source_name": src_name,
            "playback_running": self.engine.stream is not None,
            "source_range_s": [float(inf_settings.start_sec), float(inf_settings.end_sec)],
            "source_position_estimate_s": round(self.engine.pos / audio_sample_rate, 2),
            "note_on_position": "inference producer position, not exact audible playhead",
            "use_latent_mean": bool(inf_settings.use_mean),
            "latent_noise": float(inf_settings.latent_noise),
            "gain": float(inf_settings.gain),
            "active_training_entry_ranges_inclusive": self._index_ranges(enabled),
            "optimizer": train_settings.optimizer,
            "training_mode": train_settings.train_mode,
            "learning_rate": float(train_settings.learning_rate),
            "beta": float(train_settings.beta),
            "inference_model_step": int(self.trainer.inference_sync_step),
            "loss_total_latest": (self.trainer.last_loss_snapshot or {}).get("total"),
            "last_loss_measurement_step": self.trainer.last_loss_step,
            "underruns": int(self.engine.underruns),
        }
        lg.event("bookmark.added", "observer", data)
        lg.metric(self.trainer.logging_metrics(), force=True)
        self.bookmark_note.clear()
        self._update_logging_status()

    # ---------------------------------------------------------------- layout
    def build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)
        common_tab = self.build_common_tab()      # must exist before elements are registered
        data_tab = self.build_data_tab()
        inference_tab = self.build_inference_tab()
        training_tab = self.build_training_tab()
        loss_tab = self.build_loss_tab()
        self.log_tab = self.build_log_tab()
        # Both session panels share one scrollable tab. Each retains its own
        # independent Common checkbox in the original panel, and neither
        # checkbox is copied to the Common tab.
        session_tab, session_root = self._scroll_tab()
        self.build_logging_toolbar(session_root)
        self.build_replay_toolbar(session_root)
        session_root.addStretch(1)
        for widget, title in ((common_tab, "Common"), (data_tab, "Data selection"),
                              (inference_tab, "Live inference"), (training_tab, "Live training"),
                              (loss_tab, "Loss functions"), (self.log_tab, "Training log"),
                              (session_tab, "Session log")):
            self.tabs.addTab(widget, title)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumWidth(0)
        self.stats_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.stats_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.stats_label.setStyleSheet("font-family: monospace;")
        root.addWidget(self.stats_label)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._ready = True
        self.apply_train_settings()
        self.apply_inf_settings()
        self.log_timer = QtCore.QTimer(self)
        self.log_timer.timeout.connect(self.update_log_plot)
        self.log_timer.start(50)          # 20 fps, drawn only for visible plots

    @staticmethod
    def _scroll_tab():
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        outer.addWidget(scroll)
        scroll.setWidget(content)
        return tab, root

    # ---------------------------------------------------------------- Common tab
    def build_common_tab(self):
        tab, root = self._scroll_tab()
        cfg_box = QtWidgets.QGroupBox("GUI configuration")
        bar = QtWidgets.QHBoxLayout(cfg_box)
        save_btn = QtWidgets.QPushButton("Save GUI configuration...")
        save_btn.clicked.connect(self.save_gui_config)
        load_btn = QtWidgets.QPushButton("Load GUI configuration...")
        load_btn.clicked.connect(self.load_gui_config)
        bar.addWidget(save_btn)
        bar.addWidget(load_btn)
        bar.addStretch(1)
        root.addWidget(cfg_box)
        for key, title in GUI_CATEGORIES:
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            group.setVisible(False)
            self.common_groups[key], self.common_forms[key] = group, form
            root.addWidget(group)
        root.addStretch(1)
        return tab

    # ---------------------------------------------------------------- Data selection tab
    def make_file_panel(self):
        p = AudioFilesPanel()
        p.toggled.connect(lambda row, on, src=p: self.on_panel_toggled(row, on, src))
        p.selected.connect(lambda row, src=p: self.on_panel_selected(row, src))
        p.add_requested.connect(self.add_files)
        return p

    def audio_panels(self):
        m = self.elements["data.files"].mirror
        return [self.file_panel] + ([m] if m is not None else [])

    def fill_panel(self, panel):
        with store.lock:
            entries = list(zip(store.names, store.enabled))
        panel.set_entries(entries, self.file_panel.list.currentRow())

    def build_data_tab(self):
        tab, root = self._scroll_tab()
        box = QtWidgets.QGroupBox("Training data: files and independent segments")
        dl = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(
            "Newly added audio is NOT part of the training set: tick an entry to include it. "
            "Splitting into segments only affects the training dataset; live inference always plays "
            "the whole source file.")
        hint.setWordWrap(True)
        dl.addWidget(hint)
        self.file_panel = self.add_widget_element(
            dl, "data", "data.files", "Audio files (training entries)",
            self.make_file_panel, update_fn=self.fill_panel)
        self.file_list = self.file_panel.list

        form = QtWidgets.QFormLayout()
        self.chunk_on_load = self.add_element(
            form, "data", "data.chunk_on_load", "Training segments for new files (s; 0 = whole)",
            lambda: dspin(default_audio_chunk_seconds, 0, 100000, 1.0, 2, visible_max=60.0))
        self.chunk_on_load.setToolTip("0 keeps each new file whole in the dataset; >0 creates independent "
                                      "_1, _2, ... training entries (playback is never split)")
        self.range_start = self.add_element(
            form, "data", "data.range_start", "Selected entry: start (s)",
            lambda: dspin(0, 0, 1e5, 1.0, 2, visible_max=60), on_change=self.on_range_changed)
        self.range_end = self.add_element(
            form, "data", "data.range_end", "Selected entry: end (s)",
            lambda: dspin(0, 0, 1e5, 1.0, 2, visible_max=60), on_change=self.on_range_changed)
        dl.addLayout(form)
        self.split_btn = QtWidgets.QPushButton("Split selected into _1, _2, ...")
        self.split_btn.clicked.connect(self.split_selected)
        dl.addWidget(self.split_btn)
        self.range_info = QtWidgets.QLabel(
            "Select an entry, set a width via end - start, then click Split. Nothing is truncated.")
        self.range_info.setWordWrap(True)
        dl.addWidget(self.range_info)
        root.addWidget(box)
        root.addStretch(1)
        return tab

    def refresh_files(self, selected_index=None, inference_index=None):
        selected = self.file_list.currentRow() if selected_index is None else selected_index
        idx = inf_settings.file_index if inference_index is None else inference_index
        with store.lock:
            entries = list(zip(store.names, store.enabled))
            sources = list(store.src_names)
        row = max(0, min(selected, len(entries) - 1)) if entries else None
        self._suppress_file_selection_log = True
        try:
            for p in self.audio_panels():
                p.set_entries(entries, row)
            if entries:
                self.on_file_selected(row)
        finally:
            self._suppress_file_selection_log = False
        self._selected_file_shadow = row
        # The inference list holds the WHOLE source files and only changes when files are added.
        if sources != self._src_snapshot:
            old, self._src_snapshot = self._src_snapshot, sources
            idx = max(0, min(idx, len(sources) - 1)) if sources else 0
            self.inf_file.blockSignals(True)
            self.inf_file.clear()
            self.inf_file.addItems(sources)
            if sources:
                self.inf_file.setCurrentIndex(idx)
            self.inf_file.blockSignals(False)
            self._sync_mirror(self.elements["inference.file"])
            if sources and (not old or old[:idx + 1] != sources[:idx + 1]):
                self.on_inf_file(idx)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add audio", audio_data_path, "Audio (*.wav *.flac *.mp3)")
        if not paths:
            return
        added = 0
        imported = []
        for path in paths:
            try:
                # New audio is NOT part of the training set until its toggle is switched on.
                with store.lock:
                    previous_sources = len(store.src_paths)
                    previous_entries = len(store.names)
                number = store.add_file(path, chunk_seconds=self.chunk_on_load.value(), enabled=False)
                added += number
                info = self._source_description(previous_sources, path)
                info["new_entry_index_range"] = [previous_entries, previous_entries + number - 1]
                with store.lock:
                    info["decoded_samples_48k"] = int(store.src_waves[previous_sources].numel())
                info["chunk_seconds"] = float(self.chunk_on_load.value())
                info["training_enabled"] = False
                imported.append(info)
            except Exception as exc:
                self.experiment_logger.event("data.import_error", "system", {
                    "name": os.path.basename(path), "error": str(exc)[:600]})
                QtWidgets.QMessageBox.warning(self, "Could not load audio", f"{path}\n{exc}")
        if imported:
            self._flush_all_changes()
            self.experiment_logger.event("data.sources_imported", "user", {"sources": imported})
        self.refresh_files()
        self.range_info.setText(f"Added {added} training entries (not used for training; tick them to include).")

    def split_selected(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
        width = self.range_end.value() - self.range_start.value()
        if width <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid segment", "Set end > start to define a positive chunk length.")
            return
        original = self._log_entry_description(row)
        count = store.split_entry(row, width)       # touches only training, never playback
        if count:
            self._flush_all_changes()
            self.experiment_logger.event("data.entry_split", "user", {
                "original": original, "chunk_seconds": float(width),
                "result_count": int(count), "first_result_index": int(row),
                "result_names": [store.names[i] for i in range(row, row + count)]
                                if count <= 100 else None,
                "naming_rule": "_1, _2, ...; final partial chunk retained",
            })
        if not count:
            self.range_info.setText("Selected item is already shorter than (or equal to) the chosen chunk length.")
            return
        self.refresh_files(selected_index=row)
        self.range_info.setText(f"Created {count} independent numbered training segments "
                                "(including the last partial chunk). Live inference still plays the whole file.")

    def on_panel_toggled(self, row, on, src):
        before = self._log_entry_description(row)
        if before is None:
            return
        store.set_enabled(row, on)
        if bool(before["enabled"]) != bool(on) and not self._loading:
            self._flush_all_changes()
            self.experiment_logger.event("data.entry_enabled", "user", {
                "entry": before, "old": before["enabled"], "enabled": bool(on)})
        for p in self.audio_panels():
            if p is not src:
                p.set_checked(row, on)

    def on_panel_selected(self, row, src):
        for p in self.audio_panels():
            if p is not src:
                p.set_current(row)
        self.on_file_selected(row)
        if (row >= 0 and not self._loading and not self._suppress_file_selection_log
                and row != self._selected_file_shadow):
            self.experiment_logger.event("ui.training_entry_selected", "user", {
                "entry": self._log_entry_description(row)})
        self._selected_file_shadow = row

    def on_file_selected(self, r):
        if r < 0 or r >= len(store.waves):
            return
        dur = store.waves[r].shape[0] / audio_sample_rate
        for w in (self.range_start, self.range_end):
            w.blockSignals(True)
            w.setRange(0, dur)
        self.range_start.setValue(0.0)
        self.range_end.setValue(dur)
        for w in (self.range_start, self.range_end):
            w.blockSignals(False)
        self.refresh_mirrors("data.range_start", "data.range_end")
        self.on_range_changed()

    def on_range_changed(self, *_):
        # These fields do NOT truncate or update training windows.
        width = self.range_end.value() - self.range_start.value()
        self.range_info.setText(
            f"Chunk width: {max(0, width):.2f} s. Click Split to create _1, _2, ... from the ENTIRE selected entry. "
            "The last chunk may be shorter; all chunks can be trained independently.")

    # ---------------------------------------------------------------- Live inference tab
    def make_audio_buttons(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        start_btn = QtWidgets.QPushButton("Start audio")
        start_btn.clicked.connect(lambda *_: self.engine.start())
        stop_btn = QtWidgets.QPushButton("Stop audio")
        stop_btn.clicked.connect(lambda *_: self.engine.stop())
        lay.addWidget(start_btn)
        lay.addWidget(stop_btn)
        return w

    def build_inference_tab(self):
        tab, root = self._scroll_tab()
        ib = QtWidgets.QGroupBox("Live inference")
        il = QtWidgets.QFormLayout(ib)

        self.dev_map, dev_names = [], []
        default_out = sd.default.device[1] if default_audio_output_device is None else default_audio_output_device
        default_text = None
        for i, d in enumerate(sd.query_devices()):
            if d['max_output_channels'] > 0:
                self.dev_map.append(i)
                dev_names.append(f"{i}: {d['name']}")
                if i == default_out:
                    default_text = dev_names[-1]
        self.dev_combo = self.add_element(
            il, "inference", "inference.device", "Output device",
            combo_factory(dev_names, default_text or (dev_names[0] if dev_names else "")),
            on_change=lambda *_: self.on_device_changed())
        self.engine.device_id = self.dev_map[self.dev_combo.currentIndex()] if self.dev_map else None

        self.inf_file = self.add_element(
            il, "inference", "inference.file", "Source file (whole file)",
            lambda: QtWidgets.QComboBox(),
            on_change=lambda *_: self.on_inf_file(self.inf_file.currentIndex()),
            items_fn=lambda: list(store.src_names))
        self.inf_start = self.add_element(
            il, "inference", "inference.start", "Source start (s)",
            lambda: dspin(0, 0, 1e5, 1.0, visible_max=60), on_change=self.apply_inf_settings)
        self.inf_end = self.add_element(
            il, "inference", "inference.end", "Source end (s)",
            lambda: dspin(0, 0, 1e5, 1.0, visible_max=60), on_change=self.apply_inf_settings)
        self.inf_mean = self.add_element(
            il, "inference", "inference.use_mean", "Use latent mean (no sampling)",
            checkbox_factory(False), on_change=self.apply_inf_settings)
        self.inf_noise = self.add_element(
            il, "inference", "inference.noise", "Latent noise",
            lambda: dspin(0.0, 0, 100, 0.05, 3, visible_max=2.0), on_change=self.apply_inf_settings)
        self.inf_gain = self.add_element(
            il, "inference", "inference.gain", "Gain",
            lambda: dspin(1.0, 0, 10, 0.1, 2, visible_max=2.0), on_change=self.apply_inf_settings)

        audio_wrap = QtWidgets.QVBoxLayout()
        self.audio_buttons = self.add_widget_element(
            audio_wrap, "inference", "inference.audio_buttons", "Audio output", self.make_audio_buttons)
        il.addRow(audio_wrap)
        root.addWidget(ib)
        root.addStretch(1)
        return tab

    def on_device_changed(self):
        i = self.dev_combo.currentIndex()
        if 0 <= i < len(self.dev_map):
            self.engine.set_device(self.dev_map[i])

    def on_inf_file(self, idx):
        if idx < 0 or idx >= len(store.src_waves):
            return
        previous_index = int(inf_settings.file_index)
        if self._ready and not self._loading and idx != previous_index:
            self._flush_all_changes()
            self._capture_control_change(self.elements["inference.file"], immediate=True)
        dur = store.src_waves[idx].shape[0] / audio_sample_rate      # whole source file, never split
        for b in (self.inf_start, self.inf_end):
            b.blockSignals(True)
            b.setRange(0, dur)
        self.inf_start.setValue(0.0)
        self.inf_end.setValue(dur)
        for b in (self.inf_start, self.inf_end):
            b.blockSignals(False)
        self.refresh_mirrors("inference.start", "inference.end")
        inf_settings.file_index = idx
        self.apply_inf_settings()
        self.engine.reset_state()
        if self._ready and not self._loading and previous_index != idx:
            self.experiment_logger.event("playback.source_changed", "audio", {
                "source_index": int(idx), "source_name": store.src_names[idx],
                "range_reset_to_s": [0.0, float(dur)],
                "inference_decoder_state_reset": True,
                "playback_running": self.engine.stream is not None},
                cause_seq=self.experiment_logger.last_change("inference.file"))

    def apply_inf_settings(self, *_):
        if not self._ready:
            return
        s = inf_settings
        s.file_index = max(0, self.inf_file.currentIndex())
        s.start_sec = self.inf_start.value()
        s.end_sec = self.inf_end.value()
        s.use_mean = self.inf_mean.isChecked()
        s.latent_noise = self.inf_noise.value()
        s.gain = self.inf_gain.value()

    # ---------------------------------------------------------------- Live training tab
    def make_control_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        pause = QtWidgets.QPushButton("Resume training" if train_settings.paused else "Pause training")
        pause.setCheckable(True)
        pause.setChecked(train_settings.paused)
        pause.toggled.connect(self.on_pause)
        reset_btn = QtWidgets.QPushButton("Reset model")
        reset_btn.clicked.connect(lambda *_: self.reset_model())
        save_btn = QtWidgets.QPushButton("Save checkpoint")
        save_btn.clicked.connect(lambda *_: self.request_save_checkpoint())
        load_btn = QtWidgets.QPushButton("Load checkpoint...")
        load_btn.clicked.connect(lambda *_: self.load_checkpoint())
        for b in (pause, reset_btn, save_btn, load_btn):
            lay.addWidget(b)
        w.pause_btn = pause
        return w

    def control_panels(self):
        m = self.elements["training.controls"].mirror
        return [self.control_panel] + ([m] if m is not None else [])

    def build_training_tab(self):
        tab, root = self._scroll_tab()
        ts = train_settings

        tp = QtWidgets.QGroupBox("Training parameters (live)")
        form = QtWidgets.QFormLayout(tp)
        A = self.add_element
        self.w_rec = A(form, "training", "training.rec_loss_scale", "Rec loss scale",
                       lambda: dspin(ts.rec_loss_scale, 0, 1e4, 0.5, 3, visible_max=15.0),
                       on_change=self.apply_train_settings)
        self.w_beta = A(form, "training", "training.beta", "Beta (KL factor)",
                        lambda: dspin(ts.beta, 0, 100, 0.001, 5, visible_max=0.10),
                        on_change=self.apply_train_settings, common=True)
        self.w_lr = A(form, "training", "training.learning_rate", "Learning rate (log)",
                      lambda: dspin(ts.learning_rate, 1e-8, 10.0, 1e-5, 9,
                                    visible_min=1e-6, visible_max=1e-2, logarithmic=True),
                      on_change=self.apply_train_settings, common=True)
        self.w_optimizer = A(form, "training", "training.optimizer", "Optimizer",
                             combo_factory(["AdamW", "Adam", "SGD + momentum", "RMSprop", "Adagrad", "L-BFGS"],
                                           ts.optimizer),
                             on_change=self.on_optimizer_changed)
        self.w_batch_mode = A(form, "training", "training.batch_mode", "Sampling",
                              combo_factory(["Random windows", "Full dataset windows"], ts.batch_mode),
                              on_change=self.apply_train_settings)
        self.w_bs = A(form, "training", "training.batch_size", "Random batch size",
                      lambda: ispin(ts.batch_size, 1, 512, visible_max=128), on_change=self.apply_train_settings)
        self.w_lbfgs_iters = A(form, "training", "training.lbfgs_iters", "L-BFGS internal iterations",
                               lambda: ispin(ts.lbfgs_max_iter, 1, 50), on_change=self.apply_train_settings)
        self.w_lbfgs_history = A(form, "training", "training.lbfgs_history", "L-BFGS history length",
                                 lambda: ispin(ts.lbfgs_history_size, 1, 100), on_change=self.apply_train_settings)
        self.w_lbfgs_hold = A(form, "training", "training.lbfgs_hold", "L-BFGS reuse random batch for N steps",
                              lambda: ispin(ts.lbfgs_hold_steps, 1, 1000, visible_max=100),
                              on_change=self.apply_train_settings)
        self.w_sync = A(form, "training", "training.sync_interval", "Sync to inference every N steps",
                        lambda: ispin(ts.sync_interval, 1, 1000, visible_max=50),
                        on_change=self.apply_train_settings)
        root.addWidget(tp)

        lora_box = QtWidgets.QGroupBox("Trainable parameters (full training or ConvLoRA fine-tuning)")
        lf = QtWidgets.QFormLayout(lora_box)
        self.w_train_mode = A(lf, "training", "training.train_mode", "Trainable weights",
                              combo_factory(["Full network", "ConvLoRA: encoder + decoder",
                                             "ConvLoRA: decoder only"], ts.train_mode),
                              on_change=self.apply_train_settings)
        self.w_lora_rank = A(lf, "training", "training.lora_rank", "ConvLoRA rank",
                             lambda: ispin(ts.lora_rank, 1, 128, visible_max=32),
                             on_change=self.apply_train_settings)
        self.w_lora_alpha = A(lf, "training", "training.lora_alpha", "ConvLoRA scaling alpha",
                              lambda: dspin(ts.lora_alpha, 0.1, 1024.0, 1.0, 2, visible_max=32),
                              on_change=self.apply_train_settings)
        lora_help = QtWidgets.QLabel(
            "ConvLoRA factorizes Conv1D/2D kernels (including causal decoder filters). "
            "Train Full network first; switching modes/rank merges existing active LoRA updates "
            "into the base weights to preserve audible output. Adapters start with zero contribution.")
        lora_help.setWordWrap(True)
        lf.addRow(lora_help)
        self.merge_btn = QtWidgets.QPushButton("Merge LoRA into base and start fresh adapters")
        self.merge_btn.clicked.connect(self.request_merge_lora)
        lf.addRow(self.merge_btn)
        root.addWidget(lora_box)

        note_box = QtWidgets.QGroupBox("Notes")
        nl = QtWidgets.QVBoxLayout(note_box)
        note = QtWidgets.QLabel(
            "Total loss = rec loss scale x sum(weight x loss) + beta x KL. "
            "Full dataset sampling uses deterministic nonoverlapping windows (up to 64 including "
            "tail coverage). L-BFGS re-evaluates the same batch and VAE noise during its line search. "
            "Use a smaller full dataset or Random windows if you run out of GPU memory.")
        note.setWordWrap(True)
        nl.addWidget(note)
        root.addWidget(note_box)

        ctl_box = QtWidgets.QGroupBox("Training control")
        cl = QtWidgets.QVBoxLayout(ctl_box)
        self.control_panel = self.add_widget_element(
            cl, "training", "training.controls", "Pause / reset / checkpoint buttons", self.make_control_panel)
        root.addWidget(ctl_box)
        root.addStretch(1)
        return tab

    def apply_train_settings(self, *_):
        if not self._ready:
            return
        ts = train_settings
        ts.rec_loss_scale = self.w_rec.value()
        ts.beta = self.w_beta.value()
        ts.learning_rate = self.w_lr.value()
        ts.batch_size = self.w_bs.value()
        ts.sync_interval = self.w_sync.value()
        ts.optimizer = self.w_optimizer.currentText()
        ts.batch_mode = self.w_batch_mode.currentText()
        ts.lbfgs_max_iter = self.w_lbfgs_iters.value()
        ts.lbfgs_history_size = self.w_lbfgs_history.value()
        ts.lbfgs_hold_steps = self.w_lbfgs_hold.value()
        ts.train_mode = self.w_train_mode.currentText()
        ts.lora_rank = self.w_lora_rank.value()
        ts.lora_alpha = self.w_lora_alpha.value()
        using_lbfgs = ts.optimizer == "L-BFGS"
        for widget in (self.w_lbfgs_iters, self.w_lbfgs_history, self.w_lbfgs_hold):
            widget.setEnabled(using_lbfgs)
        self.w_bs.setEnabled(ts.batch_mode == "Random windows")
        self.merge_btn.setEnabled(ts.train_mode != "Full network")

    def on_optimizer_changed(self, name):
        if not self._ready:
            return
        cause = None
        if not self._loading:
            self._flush_all_changes()
            # Log the explicit optimizer decision BEFORE its automatic learning-rate side effect.
            cause = self._capture_control_change(self.elements["training.optimizer"], immediate=True)
            preferred_lr = {"AdamW": 5e-4, "Adam": 5e-4, "SGD + momentum": 1e-3,
                            "RMSprop": 1e-4, "Adagrad": 1e-2, "L-BFGS": 1.0}
            self._automatic_control = {"origin": "gui", "via": "automatic_optimizer_default",
                                       "cause_seq": cause}
            try:
                self.w_lr.setValue(preferred_lr.get(name, 5e-4))
            finally:
                self._automatic_control = None
        self.apply_train_settings()

    def reset_model(self):
        self._flush_all_changes()
        self.experiment_logger.event("model.reset_requested", "user", {
            "previous_step": int(self.trainer.step)})
        # LoRA on a freshly randomized frozen CNN is not useful. Restart from
        # full-network training, then enable LoRA after learning the first song.
        self.w_train_mode.setCurrentText("Full network")
        self.trainer.reset_req = True

    def request_save_checkpoint(self):
        self._flush_all_changes()
        self.trainer._pending_checkpoint_seq = self.experiment_logger.event(
            "checkpoint.requested", "user", {"step": int(self.trainer.step)})
        self.trainer.save_req = True

    def request_merge_lora(self):
        self._flush_all_changes()
        self.trainer._pending_merge_seq = self.experiment_logger.event("lora.merge_requested", "user", {
            "mode": train_settings.train_mode, "rank": train_settings.lora_rank})
        self.trainer.merge_lora_req = True

    def on_pause(self, checked):
        if bool(train_settings.paused) != bool(checked):
            self._flush_all_changes()
            self.experiment_logger.event("training.paused" if checked else "training.resumed",
                                         "user", {"reason": "pause_button"})
        train_settings.paused = checked
        for panel in self.control_panels():       # keep the tab and Common-tab copies in sync
            btn = panel.pause_btn
            if btn.isChecked() != checked:
                btn.blockSignals(True)
                btn.setChecked(checked)
                btn.blockSignals(False)
            btn.setText("Resume training" if checked else "Pause training")

    def load_checkpoint(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select encoder weights (decoder is derived by name)", save_path)
        if p:
            self._flush_all_changes()
            self.experiment_logger.event("checkpoint.load_requested", "user", {
                "encoder": os.path.basename(p)})
            self.trainer.load_req = p

    # ---------------------------------------------------------------- Loss functions tab
    def build_loss_tab(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        info_box = QtWidgets.QGroupBox("Overview")
        label = QtWidgets.QLabel(
            "PRIMARY LOSSES - the four reconstruction terms active by default; their weights are also "
            "available in the Common tab. KL is controlled only by beta (Live training tab).")
        label.setWordWrap(True)
        QtWidgets.QVBoxLayout(info_box).addWidget(label)
        lay.addWidget(info_box)
        for spec in LOSS_SPECS:
            if spec["key"] in PRIMARY_LOSS_KEYS:
                lay.addWidget(self.build_loss_group(spec))
        other_label = QtWidgets.QLabel("OTHER OPTIONAL LOSSES")
        other_label.setStyleSheet("font-weight: 600; margin-top: 12px;")
        lay.addWidget(other_label)
        for spec in LOSS_SPECS:
            if spec["key"] not in PRIMARY_LOSS_KEYS:
                lay.addWidget(self.build_loss_group(spec))
        lay.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def add_loss_weight(self, form, key, common_title):
        return self.add_element(
            form, "losses", f"losses.{key}.weight", "Weight (0 = off)",
            lambda: dspin(loss_cfg[key]["weight"], 0, 1e4, 0.05, 4, visible_max=2.0),
            on_change=lambda v, k=key: loss_cfg[k].__setitem__("weight", float(v)),
            common=key in PRIMARY_LOSS_KEYS, common_label=f"{common_title}: weight")

    def build_loss_group(self, spec):
        key = spec["key"]
        primary = key in PRIMARY_LOSS_KEYS
        box = QtWidgets.QGroupBox(("PRIMARY | " if primary else "") + spec["title"])
        form = QtWidgets.QFormLayout(box)
        short = PRIMARY_LOSS_LABELS.get(key, spec["title"])
        self.add_loss_weight(form, key, short)

        for p in spec["params"]:
            pk, t = p["key"], p["type"]
            if t == "int":
                factory, cast = (lambda p=p: ispin(p["default"], p["min"], p["max"])), int
            elif t == "float":
                factory, cast = (lambda p=p: dspin(p["default"], p["min"], p["max"], p.get("step", 0.1), 3)), float
            elif t == "bool":
                factory, cast = checkbox_factory(p["default"]), bool
            else:
                factory, cast = combo_factory(p["choices"], p["default"]), str
            self.add_element(
                form, "losses", f"losses.{key}.{pk}", p["label"], factory,
                on_change=lambda v, k=key, q=pk, c=cast: loss_cfg[k].__setitem__(q, c(v)),
                common_label=f"{short} - {p['label']}")

        lbl = QtWidgets.QLabel("value: off")
        lbl.setStyleSheet("font-family: monospace;")
        form.addRow("Running value:", lbl)
        self.loss_labels[key] = lbl
        return box

    # ---------------------------------------------------------------- Log tab
    def build_log_tab(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(tab)
        box = QtWidgets.QGroupBox("Graph settings")
        form = QtWidgets.QFormLayout(box)
        self.log_window = self.add_element(
            form, "log", "log.window_steps", "Window width (training steps)",
            lambda: NumericRangeSlider(2000, 10, HISTORY_CAPACITY, step=1, integer=True, logarithmic=True,
                                       hard_lower=10, hard_upper=HISTORY_CAPACITY))
        self.log_total = self.add_element(form, "log", "log.show_total", "Show total loss",
                                          checkbox_factory(False))
        self.log_logy = self.add_element(form, "log", "log.log_y", "Log y-axis",
                                         checkbox_factory(False))
        root.addWidget(box)
        plot_box = QtWidgets.QGroupBox(
            "Weighted loss contributions (rec loss scale x weight x loss; KL = beta x KL)")
        pl = QtWidgets.QVBoxLayout(plot_box)
        self.log_plot = self.add_widget_element(
            pl, "log", "log.plot", "Loss timeline", lambda: LossHistoryPlot(self.trainer))
        root.addWidget(plot_box, 1)
        note = QtWidgets.QLabel(
            "Experiment logging is controlled in the Session log tab; its recording panel can be shown "
            "in Common by checking 'common' in that panel. Recording is OFF by default. "
            "When enabled, it writes events.jsonl and sparse metrics.jsonl to a new "
            "results/live_vae/sessions/... folder. Add bookmarks while listening; "
            "no audio or model weights are copied into the log.")
        note.setWordWrap(True)
        root.addWidget(note)
        return tab

    def update_log_plot(self):
        plots = [self.log_plot]
        mirror = self.elements["log.plot"].mirror
        if mirror is not None:
            plots.append(mirror)
        for p in plots:
            if p.isVisible():          # drawn only while its tab is shown
                p.update_plot(self.log_window.value(), self.log_total.isChecked(), self.log_logy.isChecked())

    # ---------------------------------------------------------------- save / restore configuration
    def collect_gui_config(self, *, config_dir=None):
        """Save paths anchored to EXPERIMENT_ROOT when creating a GUI JSON.

        The no-argument form is also used for in-memory GUI state snapshots;
        those keep the logger's existing native-path representation.
        """
        elements = {}
        for uid, e in self.elements.items():
            d = {"common": e.mirror is not None}
            if e.kind == "value":
                d["value"] = _get_value(e.primary)
                if isinstance(e.primary, NumericRangeSlider):
                    d["range"] = [e.primary.lower, e.primary.upper]
            elements[uid] = d
        dataset = (store.export_layout(experiment_root=EXPERIMENT_ROOT)
                   if config_dir is not None else store.export_layout())
        return {"format": "live_audio_vae_gui_config", "version": 4,
                "elements": elements,
                "dataset": dataset,
                "active_tab": self.tabs.currentIndex()}

    def save_gui_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GUI configuration", os.path.join(save_path, "gui_config.json"), "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        tmp_path = None
        try:
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)
            # JSON file location is intentionally irrelevant: saved audio paths
            # are relative to EXPERIMENT_ROOT, not to the JSON's directory.
            portable_config = self.collect_gui_config(config_dir=folder)
            # An atomic replacement avoids a truncated config if the process is
            # interrupted while saving (also works with the Windows file dialog).
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                             prefix=".audio_vae_config_", dir=folder,
                                             delete=False) as out:
                tmp_path = out.name
                json.dump(portable_config, out, indent=2, ensure_ascii=False,
                          allow_nan=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp_path, path)
            filename_only = portable_config["dataset"].get("basename_only_sources", [])
            self.trainer.status_msg = (f"GUI configuration saved: {os.path.basename(path)} "
                                       "(paths relative to experiment root)" +
                                       (f" | {len(filename_only)} source(s) require relinking"
                                        if filename_only else ""))
            if filename_only:
                QtWidgets.QMessageBox.information(
                    self, "Portable configuration saved",
                    f"{len(filename_only)} audio source(s) are on a different Windows drive or share "
                    "than the main experiment folder. They were saved by filename only, "
                    "not as absolute paths. Move them into data/audio under the experiment "
                    "folder (or select their folder when loading).")
            self._flush_all_changes()
            self.experiment_logger.event("config.saved", "user", {
                "filename": os.path.basename(path), "config_version": 4,
                "file_size_bytes": int(os.stat(path).st_size),
                "contains_weights": False})
        except Exception as exc:
            traceback.print_exc()
            self.trainer.status_msg = "Could not save GUI configuration (see error dialog)"
            QtWidgets.QMessageBox.warning(self, "Could not save configuration",
                                          f"{type(exc).__name__}: {str(exc)[:900]}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def load_gui_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load GUI configuration", save_path, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict) or cfg.get("format") != "live_audio_vae_gui_config":
                raise ValueError("Not an Audio VAE GUI configuration JSON")
            if not isinstance(cfg.get("elements", {}), dict):
                raise ValueError("Invalid control values in configuration")
        except Exception as exc:
            self.trainer.status_msg = "Could not read GUI configuration (see error dialog)"
            QtWidgets.QMessageBox.warning(self, "Could not read configuration",
                                          f"{type(exc).__name__}: {str(exc)[:900]}")
            return
        self._flush_all_changes()
        before_state = self._logging_initial_state() if self.experiment_logger.enabled else None
        self._loading = True
        skipped, skipped_devices, missing = [], [], []
        preserved_audio = False
        # v22 POSIX paths are relative to EXPERIMENT_ROOT. The dataset's
        # 'relative_to' field retains v21 config-directory and v20 path support.
        config_dir = os.path.dirname(os.path.abspath(path))
        audio_root = None
        try:
            if isinstance(cfg.get("dataset"), dict):
                preflight = store.unresolved_sources(cfg["dataset"], config_dir=config_dir)
                if preflight:
                    preview = ", ".join(preflight[:4])
                    if len(preflight) > 4:
                        preview += f", and {len(preflight) - 4} more"
                    choice = QtWidgets.QMessageBox.question(
                        self, "Locate audio files",
                        f"{len(preflight)} audio file(s) were not found in the saved paths "
                        f"or local audio directories.\n{preview}\n\n"
                        "Choose a folder with these files? You may also skip this "
                        "and load the other GUI settings.",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes)
                    if choice == QtWidgets.QMessageBox.Yes:
                        audio_root = QtWidgets.QFileDialog.getExistingDirectory(
                            self, "Select folder containing the original audio files",
                            os.path.abspath(audio_data_path)) or None

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                if "dataset" in cfg:
                    try:
                        with self.engine.state_lock:
                            missing = store.restore_layout(cfg["dataset"],
                                config_dir=config_dir, audio_root=audio_root)
                        preserved_audio = bool(store.last_restore_info["preserved_existing"])
                        self._src_snapshot = []  # rebuild inference source list
                    except Exception as exc:
                        traceback.print_exc()
                        skipped.append("dataset: " + str(exc)[:100])
                elif isinstance(cfg.get("files"), dict):  # old config format
                    with store.lock:
                        names = list(store.names)
                    for i, name in enumerate(names):
                        if name in cfg["files"]:
                            store.set_enabled(i, cfg["files"][name])
                self.refresh_files(selected_index=0)
                # Resolve file source before start/end values (regardless of
                # the order in which controls appear in old JSON files).
                saved_controls = cfg.get("elements", {})
                preferred = ["inference.device", "inference.file", "inference.start", "inference.end"]
                ordered = preferred + [uid for uid in saved_controls if uid not in preferred]
                for uid in ordered:
                    d = saved_controls.get(uid)
                    e = self.elements.get(uid)
                    if e is None or not isinstance(d, dict):
                        continue
                    try:
                        if e.kind == "value" and "value" in d:
                            if "range" in d and isinstance(e.primary, NumericRangeSlider):
                                bounds = d["range"]
                                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                                    e.primary._set_visible_bounds(bounds[0], bounds[1])
                            if isinstance(e.primary, QtWidgets.QComboBox):
                                requested = str(d["value"])
                                if e.primary.findText(requested) < 0:
                                    if uid == "inference.device":
                                        skipped_devices.append(requested)
                                    else:
                                        skipped.append(uid)
                                else:
                                    _set_value(e.primary, requested)
                            else:
                                _set_value(e.primary, d["value"])
                        e.toggle.setChecked(bool(d.get("common", False)))
                        self._sync_mirror(e, bounds=True)
                    except Exception as exc:
                        print(f"Could not restore GUI control {uid}: {exc}")
                        skipped.append(uid)
                restored_tab = self._mapped_tab_index(cfg.get("active_tab", 0))
                if restored_tab >= 0:
                    self.tabs.setCurrentIndex(restored_tab)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        finally:
            self._loading = False

        self.apply_train_settings()
        self.apply_inf_settings()
        notes = []
        if missing:
            notes.append(f"{len(missing)} audio file(s) not found")
        if preserved_audio:
            notes.append("existing audio retained")
        if skipped_devices:
            notes.append("unavailable output device: local default kept")
        if skipped:
            notes.append(f"{len(skipped)} setting(s) skipped")
        self.trainer.status_msg = (
            f"GUI configuration loaded: {os.path.basename(path)}" +
            (" | " + "; ".join(notes) if notes else ""))
        self._reset_control_shadows()
        if before_state is not None:
            after_state = self._logging_initial_state()
            previous = before_state["gui"]["elements"]
            current = after_state["gui"]["elements"]
            changes = {uid: {"old": previous.get(uid), "new": entry}
                       for uid, entry in current.items() if previous.get(uid) != entry}
            self.experiment_logger.event("config.loaded", "user", {
                "filename": os.path.basename(path), "changed_elements": changes,
                "effective_state": after_state,
                "previous_source_names": [s["name"] for s in before_state["gui"]["dataset"]["sources"]],
                "current_active_tab": after_state["gui"]["active_tab"],
                "missing_sources": missing, "skipped_controls": skipped,
                "unavailable_audio_devices": skipped_devices,
                "audio_dataset_preserved": preserved_audio,
                "note": "Complete effective GUI configuration; model/optimizer weights not restored"})
            self.experiment_logger.metric(self.trainer.logging_metrics(), force=True)
        if missing or skipped or skipped_devices:
            summary = "The remaining available GUI settings were restored."
            if preserved_audio:
                summary += "\nThe existing training data was kept because no saved audio files could be found."
            if missing:
                summary += "\nMissing audio files (filenames only): " + ", ".join(missing[:12])
                if len(missing) > 12:
                    summary += f" and {len(missing)-12} more"
            if skipped_devices:
                summary += "\nSaved audio device is unavailable on this computer; the local selection was kept."
            if skipped:
                summary += f"\n{len(skipped)} GUI control(s) could not be restored."
            QtWidgets.QMessageBox.information(self, "Configuration loaded with warnings", summary)

    # ---------------------------------------------------------------- periodic update
    def update_stats(self):
        self._sync_aux_panels()
        if self._last_gui_config_revision != self.trainer.gui_config_revision:
            ts = train_settings
            for uid, value in (("training.optimizer", ts.optimizer), ("training.train_mode", ts.train_mode),
                               ("training.lora_rank", ts.lora_rank), ("training.lora_alpha", ts.lora_alpha),
                               ("training.learning_rate", ts.learning_rate), ("training.batch_mode", ts.batch_mode),
                               ("training.batch_size", ts.batch_size), ("training.lbfgs_iters", ts.lbfgs_max_iter),
                               ("training.lbfgs_history", ts.lbfgs_history_size),
                               ("training.lbfgs_hold", ts.lbfgs_hold_steps)):
                self.set_element_silently(uid, value)
            self._last_gui_config_revision = self.trainer.gui_config_revision
            self.apply_train_settings()
        self._update_logging_status()
        st = self.trainer.stats
        parts = []
        for spec in LOSS_SPECS:
            k = spec["key"]
            v = self.trainer.loss_ema.get(k, 0.0)
            active = loss_cfg[k]["weight"] != 0
            self.loss_labels[k].setText(f"{v:.5f}  (weighted {v * loss_cfg[k]['weight']:.5f})" if active else "off")
            if active:
                parts.append(f"{k} {v:.4f}")
        self.stats_label.setText(
            f"step {self.trainer.step} ({st['sps']:.1f} updates/s, {st['evals']} loss evaluations) "
            f"beta {st['beta']:.4f} total {st['total']:.4f} kld {st['kld']:.4f} "
            f"(KL contribution {st['beta'] * st['kld']:.5f})\n"
            f"{'  '.join(parts)}\n"
            f"audio queue {self.engine.q.qsize()}/{max_queued_blocks}  underruns {self.engine.underruns} "
            f"(device {self.engine.device_underflows})   {self.trainer.status_msg}")

    def closeEvent(self, event):
        if self._replay_running:
            self._finish_replay("window_closed")
        self._flush_all_changes()
        self.trainer.running = False
        self.engine.running = False
        self.engine.stop()
        self.experiment_logger.metric(self.trainer.logging_metrics(), force=True)
        self.experiment_logger.stop("window_closed")
        event.accept()


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main():
    for f in audio_data_files:
        path = os.path.join(audio_data_path, f)
        try:
            if os.path.isfile(path):
                store.add_file(path, chunk_seconds=default_audio_chunk_seconds, enabled=True)
            else:
                print(f"Startup audio not found; skipped {f}. Load audio via the GUI.")
        except Exception as exc:
            print(f"Could not load startup audio {f}: {exc}")

    trainer = Trainer()
    engine = InferenceEngine(trainer)
    experiment_logger = ExperimentLogger(save_path)
    experiment_logger.trainer = trainer
    experiment_logger.engine = engine
    trainer.experiment_logger = experiment_logger
    # Initialize Qt and log the actual initial configuration before training
    # starts; otherwise many steps can elapse before the first event.
    app = QtWidgets.QApplication(sys.argv)
    gui = LiveGUI(trainer, engine, experiment_logger)
    # Session recording is opt-in; user can start it from Session log or Common.
    threading.Thread(target=trainer.run, daemon=True).start()
    threading.Thread(target=engine.producer, daemon=True).start()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()