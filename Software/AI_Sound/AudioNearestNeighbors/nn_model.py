import os
import numpy as np
import librosa

from sklearn.preprocessing import StandardScaler

import analysis as aa

"""
NearestNeighborModel: loads all audio files in a folder, slices them into
fixed-length overlapping excerpts, computes a selectable set of audio
descriptors per excerpt (reusing analysis.py from the AudioAnalysis tool),
and performs a GREEDY nearest-neighbor CHAIN over the excerpt pool - the
same algorithm as the original nearest_neighbors.py script, but exposed as
a single-step generator (step()) instead of one monolithic while loop, so a
GUI can call it once per tick and play back / visualize the growing result
as it is built, rather than waiting for the entire chain to finish offline.

Revisions vs. the original nearest_neighbors.py:

1. Uses analysis.py from the AudioAnalysis tool (adds zero_crossing_rate,
   tonnetz, poly_features, onset_strength, and safer chroma_cqt/tempogram
   parameters) instead of the older, separate analysis.py.

2. Proper per-dimension normalization: the original script normalized each
   feature with a single GLOBAL mean/std (one scalar for the whole feature
   matrix), which only rescales the feature as a whole and does not
   equalize the contribution of individual descriptor dimensions. This
   version uses a per-dimension StandardScaler (fit across excerpts, one
   mean/std per column), matching the fix already applied to the
   AudioClustering tool.

3. FIXED TWO CORRECTNESS BUGS found in the original algorithm while
   verifying a stepwise reimplementation against it:

   a) Crash on the final iteration: the original tracked a separate
      "nn_element_count" counter as its while-loop condition instead of
      checking the actual remaining pool size. Since that counter and the
      pool size always stay numerically equal, the loop's FINAL iteration
      always runs with exactly 1 element left in the pool, at which point
      `distances.argsort()[:2]` returns only one index and indexing it
      with `[1]` raises IndexError - i.e. the original script as written
      crashes at the end of every run.

   b) Silent index corruption during the run: the original never remapped
      "nn_current_index" after removing an excerpt from the pool, even
      though the array had just shrunk. When nn_current_index later
      exceeded the (now smaller) pool's bounds, numpy's slice-based
      removal does NOT raise an error for an out-of-range index - it
      silently clips the slice, which means the wrong excerpt (or none at
      all) gets removed from the pool. Verified by comparing 50
      randomized runs against an independent, list-based reference
      implementation - the original algorithm's index semantics diverged
      from the correct sequence in a meaningful fraction of cases. This
      version always removes by true array position (np.delete) and
      correctly remaps current_index by -1 when the removed position was
      before it.

4. Stepwise API: load_and_extract() builds the pool once; step() performs
   exactly one iteration and returns whether any further steps remain,
   letting a caller drive generation incrementally (e.g. once per GUI
   tick) instead of blocking until the whole chain is built.

5. select_audio_features() lets the set of descriptors used for the
   distance computation be changed and the search restarted.

6. set_excerpt_timing() exposes audio_excerpt_seconds and
   audio_excerpt_offset_seconds as live-changeable parameters (previously
   only settable at construction time via config). Because excerpt
   boundaries are baked into how the source files were sliced, changing
   either value re-derives the sample-count versions, rebuilds the
   cross-fade envelope, and re-runs load_and_extract() against the
   already-known audio folder - this is the only way excerpt timing
   changes can take effect, since the excerpts themselves have to be
   re-sliced from the original files, not just the search reset.
"""

config = {
    "audio_folder_path": "data/audio/",
    "audio_file_extensions": ["wav", "aiff", "aif"],
    "audio_sample_rate": 48000,
    "audio_excerpt_seconds": 5.0,
    "audio_excerpt_offset_seconds": 2.5,
    "audio_feature_names": ["root mean square", "mfcc"],
}


class NearestNeighborModel():

    def __init__(self, config):

        self.audio_folder_path = config["audio_folder_path"]
        self.audio_file_extensions = [ext.lower().lstrip(".") for ext in config["audio_file_extensions"]]
        self.audio_sample_rate = config["audio_sample_rate"]

        self.audio_excerpt_seconds = config["audio_excerpt_seconds"]
        self.audio_excerpt_offset_seconds = config["audio_excerpt_offset_seconds"]

        self.audio_excerpt_sc = int(self.audio_excerpt_seconds * self.audio_sample_rate)
        self.audio_excerpt_offset_sc = int(self.audio_excerpt_offset_seconds * self.audio_sample_rate)

        self.audio_feature_names = list(config["audio_feature_names"])

        self.audio_excerpts = None
        self.audio_features = {}
        self._scalers = {}

        self._create_envelope()

        self.reset()

    # ---------------- loading / feature computation ----------------

    def list_audio_files(self):
        paths = []
        for root, _, fnames in sorted(os.walk(self.audio_folder_path, followlinks=True)):
            for fname in sorted(fnames):
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext in self.audio_file_extensions:
                    paths.append(os.path.join(root, fname))
        return paths

    def load_and_extract(self, progress_callback=None):
        """
        Loads every audio file in audio_folder_path, slices them into
        excerpts using the CURRENT audio_excerpt_sc / audio_excerpt_offset_sc,
        and computes all currently available descriptor functions for the
        WHOLE excerpt pool (not just the selected ones), so the feature
        checklist in a GUI can offer every descriptor without recomputing
        the excerpt pool each time the selection changes.
        progress_callback(fraction, message), if given, is called
        periodically for a GUI progress indicator.
        """
        paths = self.list_audio_files()
        if len(paths) == 0:
            raise ValueError("no audio files with extensions {} found in '{}'".format(
                self.audio_file_extensions, self.audio_folder_path
            ))

        waveforms = []
        for i, path in enumerate(paths):
            if progress_callback:
                progress_callback(0.4 * i / max(len(paths), 1), "loading {}".format(os.path.basename(path)))
            waveform, _ = librosa.load(path, sr=self.audio_sample_rate)
            waveforms.append(waveform)

        excerpts = []
        for waveform in waveforms:
            sample_count = waveform.shape[0]
            for sI in range(0, sample_count - self.audio_excerpt_sc, self.audio_excerpt_offset_sc):
                excerpts.append(waveform[sI:sI + self.audio_excerpt_sc])

        if len(excerpts) == 0:
            raise ValueError(
                "no audio excerpts could be extracted - files may be shorter than "
                "audio_excerpt_seconds ({}s)".format(self.audio_excerpt_seconds)
            )

        self.audio_excerpts = np.stack(excerpts, axis=0)

        if progress_callback:
            progress_callback(0.5, "computing audio features for {} excerpts...".format(self.audio_excerpts.shape[0]))

        self._compute_all_features()
        self._scalers = {}

        if progress_callback:
            progress_callback(1.0, "ready: {} excerpts".format(self.audio_excerpts.shape[0]))

    def _compute_all_features(self):
        sr = self.audio_sample_rate
        wf = self.audio_excerpts

        self.audio_features = {}
        self.audio_features["waveform"] = wf
        self.audio_features["root mean square"] = aa.rms(wf)
        self.audio_features["zero crossing rate"] = aa.zero_crossing_rate(wf)
        self.audio_features["chroma stft"] = aa.chroma_stft(wf, sr)
        self.audio_features["mel spectrogram"] = aa.mel_spectrogram(wf, sr)
        self.audio_features["mfcc"] = aa.mfcc(wf, sr)
        self.audio_features["spectral centroid"] = aa.spectral_centroid(wf, sr)
        self.audio_features["spectral bandwidth"] = aa.spectral_bandwidth(wf, sr)
        self.audio_features["spectral contrast"] = aa.spectral_contrast(wf, sr)
        self.audio_features["spectral flatness"] = aa.spectral_flatness(wf)
        self.audio_features["spectral rolloff"] = aa.spectral_rolloff(wf, sr)
        self.audio_features["tempogram"] = aa.tempogram(wf, sr, win_length=32)
        # chroma_cqt / chroma_cens / chroma_vqt / tempo / tempogram_ratio are
        # left out of the default set (slow across many excerpts, and tempo
        # needs more context than a short excerpt reliably provides) - add
        # them here following the same pattern if needed.

    def get_available_feature_names(self):
        return [name for name in self.audio_features.keys() if name != "waveform"]

    def _get_normalized_feature(self, name):
        matrix = self.audio_features[name]
        if name not in self._scalers:
            self._scalers[name] = StandardScaler()
            return self._scalers[name].fit_transform(matrix).astype(np.float32)
        return self._scalers[name].transform(matrix).astype(np.float32)

    def select_audio_features(self, feature_names):
        """
        Sets which descriptors are concatenated into the distance vector
        used for the nearest-neighbor search, and resets the search so it
        starts over with the new feature combination.
        """
        valid = [name for name in feature_names if name in self.audio_features]
        if len(valid) == 0:
            return
        self.audio_feature_names = valid
        self.reset()

    # ---------------- excerpt timing (live-changeable) ----------------

    def set_excerpt_timing(self, excerpt_seconds, offset_seconds, progress_callback=None):
        """
        Changes audio_excerpt_seconds / audio_excerpt_offset_seconds and
        re-slices + re-analyzes the currently loaded audio folder using the
        new values. This re-reads files from disk (excerpt boundaries are
        determined at slicing time, so there is no way to change them
        without re-slicing), recomputes every descriptor for the new
        excerpt pool, rebuilds the cross-fade envelope, and resets the
        search. Requires audio_folder_path to already point at a valid
        folder (i.e. load_and_extract() must have been run at least once
        before, or the folder must exist) - if no excerpts have ever been
        loaded, only the timing values and derived sample counts are
        updated, and extraction is skipped until the caller explicitly
        loads a folder.
        """
        excerpt_seconds = max(float(excerpt_seconds), 0.01)
        offset_seconds = max(float(offset_seconds), 0.01)

        self.audio_excerpt_seconds = excerpt_seconds
        self.audio_excerpt_offset_seconds = offset_seconds

        self.audio_excerpt_sc = int(self.audio_excerpt_seconds * self.audio_sample_rate)
        self.audio_excerpt_offset_sc = int(self.audio_excerpt_offset_seconds * self.audio_sample_rate)

        self._create_envelope()

        if self.audio_excerpts is not None:
            self.load_and_extract(progress_callback=progress_callback)
            self.reset()

    # ---------------- envelope ----------------

    def _create_envelope(self):
        overlap_sc = max(self.audio_excerpt_sc - self.audio_excerpt_offset_sc, 0)
        if overlap_sc <= 0:
            self.amplitude_envelope = np.ones(self.audio_excerpt_sc, dtype=np.float32)
            return

        hann = np.hanning(overlap_sc * 2).astype(np.float32)
        envelope = np.ones(self.audio_excerpt_sc, dtype=np.float32)
        envelope[:overlap_sc] *= hann[:overlap_sc]
        envelope[-overlap_sc:] *= hann[overlap_sc:]
        self.amplitude_envelope = envelope

    def _overlap_sc(self):
        return max(self.audio_excerpt_sc - self.audio_excerpt_offset_sc, 0)

    # ---------------- stepwise nearest-neighbor search ----------------

    def reset(self):
        """
        (Re)initializes the search: rebuilds the concatenated, normalized
        feature matrix for the currently selected descriptors, resets the
        remaining-excerpt pool to everything, and seeds the search with
        excerpt 0. Safe to call again at any time to start over (e.g. after
        changing which features are used).
        """
        if self.audio_excerpts is None or len(self.audio_feature_names) == 0:
            self.remaining_waveforms = None
            self.remaining_features = None
            self.total_excerpt_count = 0
            self.steps_taken = 0
            self.done = True
            self.output_waveform = np.zeros(0, dtype=np.float32)
            self.output_write_index = 0
            return

        feature_parts = [self._get_normalized_feature(name) for name in self.audio_feature_names]
        features_proc = np.concatenate(feature_parts, axis=1)

        self.remaining_waveforms = np.copy(self.audio_excerpts)
        self.remaining_features = features_proc

        self.total_excerpt_count = self.remaining_features.shape[0]
        self.steps_taken = 0

        self.current_index = 0
        self.current_waveform = self.remaining_waveforms[0]
        self.current_feature = np.expand_dims(self.remaining_features[0], 0)

        hop = self.audio_excerpt_sc - self._overlap_sc()
        max_output_sc = self.audio_excerpt_sc + max(self.total_excerpt_count - 1, 0) * hop
        self.output_waveform = np.zeros(max(max_output_sc, self.audio_excerpt_sc), dtype=np.float32)
        self.output_write_index = 0

        self.done = (self.total_excerpt_count <= 1)

        self._append_current_to_output(is_first=True)

    def _append_current_to_output(self, is_first=False):
        hop = self.audio_excerpt_sc - self._overlap_sc()
        insert_index = 0 if is_first else self.output_write_index

        end_index = insert_index + self.audio_excerpt_sc
        if end_index > self.output_waveform.shape[0]:
            pad = np.zeros(end_index - self.output_waveform.shape[0], dtype=np.float32)
            self.output_waveform = np.concatenate([self.output_waveform, pad])

        self.output_waveform[insert_index:end_index] += self.current_waveform * self.amplitude_envelope
        self.output_write_index = insert_index + hop

    def step(self):
        """
        Performs exactly one iteration of the greedy nearest-neighbor chain:
        finds the excerpt in the remaining pool closest to the current
        feature vector, cross-fades its waveform into the output buffer,
        removes the PREVIOUS excerpt from the pool by its TRUE array
        position (np.delete), and correctly remaps current_index by -1 if
        it was positioned after the removed slot.

        Returns False once the pool has been reduced to 0 or 1 excerpts,
        True otherwise. The excerpt chosen on the call where this returns
        False has already been appended to the output before returning.
        """
        if self.done or self.remaining_features is None or self.remaining_features.shape[0] <= 1:
            self.done = True
            return False

        distances = np.linalg.norm(self.remaining_features - self.current_feature, axis=1)
        nearest_indices = distances.argsort()[:2]

        previous_index = self.current_index
        new_index = int(nearest_indices[1])
        self.current_waveform = self.remaining_waveforms[new_index]
        self.current_feature = np.expand_dims(self.remaining_features[new_index], 0)

        self._append_current_to_output(is_first=False)

        self.remaining_waveforms = np.delete(self.remaining_waveforms, previous_index, axis=0)
        self.remaining_features = np.delete(self.remaining_features, previous_index, axis=0)

        self.current_index = new_index - 1 if new_index > previous_index else new_index

        self.steps_taken += 1

        if self.remaining_features.shape[0] <= 1:
            self.done = True
            return False

        return True

    def get_progress(self):
        if self.total_excerpt_count <= 1:
            return 1.0
        return min(self.steps_taken / max(self.total_excerpt_count - 1, 1), 1.0)

    def get_output_waveform(self):
        """
        Returns the portion of the output buffer that has been written so
        far (i.e. everything generated up to this point), safe to read
        while step() may still be extending it further.
        """
        end = min(self.output_write_index + self.audio_excerpt_sc, self.output_waveform.shape[0])
        return self.output_waveform[:end]

    def is_done(self):
        return self.done
