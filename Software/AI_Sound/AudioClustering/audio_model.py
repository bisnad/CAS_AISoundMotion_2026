import threading
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

"""
audio_model.py

Revisions vs. the previous version:

1. Several audio features can be combined. The model keeps a LIST of selected
   feature names (set_selected_features). For clustering, each selected feature
   (an N x D matrix) is
     a) z-scored per column with StandardScaler (if normalize is on),
     b) optionally divided by sqrt(D) ("equal feature weight"), so that
        every feature contributes the same total variance. Without this a
        feature with many dimensions (mel spectrogram, tempogram) would
        drown out one with few (RMS, spectral centroid),
     c) concatenated column-wise into a single matrix.
2. Features can be missing from audio_features and be computed lazily through
   an optional feature_provider (ClusteringPipeline.ensure_features), which is
   what lets an OSC message select a feature that was not computed yet.
3. cluster_count is clamped to the number of excerpts, and NaN/inf values in a
   feature are replaced by 0 so one bad frame cannot make KMeans fail.
4. select_audio_feature(name) is kept for backward compatibility (it selects
   exactly that single feature).
"""

config = {
    "audio_excerpts": None,
    "audio_features": None,
    "selected_features": None,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170,
    "normalize": True,
    "equal_feature_weight": True,
}


class Clustering():
    def __init__(self, audio_excerpts, audio_features, normalize=True,
                 selected_features=None, equal_feature_weight=True, feature_provider=None):

        self.audio_excerpts = audio_excerpts
        self.audio_features = audio_features
        self.normalize = normalize
        self.equal_feature_weight = equal_feature_weight
        self.feature_provider = feature_provider

        self.feature_names = []
        self.set_selected_features_silent(selected_features)

        self.cluster_method = "kmeans"
        self.cluster_labels = None

        self.cluster_count = 20
        self.random_state = 170

        self._scaler_cache = {}
        self._lock = threading.RLock()

    # ---------------- feature selection ----------------

    def get_feature_names(self):
        """All features currently available (computed), without the raw waveform."""
        return [name for name in self.audio_features.keys() if name != "waveform"]

    def get_selected_features(self):
        return list(self.feature_names)

    def get_current_feature(self):
        return self.feature_names[0] if self.feature_names else None

    def _default_features(self):
        names = self.get_feature_names()
        return names[:1]

    def set_selected_features_silent(self, names):
        """Stores a valid selection without re-clustering."""
        if names is None:
            names = []
        if isinstance(names, str):
            names = [names]

        names = [n for n in dict.fromkeys(names) if n != "waveform"]  # unique, ordered

        missing = [n for n in names if n not in self.audio_features]
        if missing and self.feature_provider is not None:
            self.feature_provider(names)

        names = [n for n in names if n in self.audio_features]
        self.feature_names = names if names else self._default_features()

    def set_selected_features(self, names):
        with self._lock:
            self.set_selected_features_silent(names)
            self.create_clusters()

    def select_audio_feature(self, feature_name):
        """Backward compatible: select exactly one feature."""
        self.set_selected_features([feature_name])

    def set_equal_feature_weight(self, enabled):
        self.equal_feature_weight = bool(enabled)
        self.create_clusters()

    def get_feature_dimensions(self):
        """{feature name: dimensionality} for the selected features."""
        return {n: int(np.prod(self.audio_features[n].shape[1:])) for n in self.feature_names}

    # ---------------- data ----------------

    def set_data(self, audio_excerpts, audio_features, selected_features=None):
        """
        Swaps in new excerpts/features (new file, new excerpt length/offset or
        newly computed features). Keeps method/count/normalize settings. Does
        NOT re-cluster: call create_clusters() afterwards.
        """
        with self._lock:
            self.audio_excerpts = audio_excerpts
            self.audio_features = audio_features
            self._scaler_cache = {}
            self.cluster_labels = None

            if selected_features is None:
                selected_features = self.feature_names
            self.set_selected_features_silent(selected_features)

    # ---------------- parameters ----------------

    def set_cluster_method(self, method):
        if method in ("kmeans", "minibatch_kmeans"):
            self.cluster_method = method
            self.create_clusters()

    def set_cluster_count(self, cluster_count):
        self.cluster_count = max(int(cluster_count), 1)
        self.create_clusters()

    def set_random_state(self, random_state):
        self.random_state = int(random_state)
        self.create_clusters()

    def set_normalize(self, normalize):
        self.normalize = bool(normalize)
        self.create_clusters()

    # ---------------- clustering ----------------

    def _get_single_feature_matrix(self, name):
        matrix = np.asarray(self.audio_features[name], dtype=np.float64)
        matrix = matrix.reshape(matrix.shape[0], -1)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            if name not in self._scaler_cache:
                self._scaler_cache[name] = StandardScaler().fit(matrix)
            matrix = self._scaler_cache[name].transform(matrix)

        if self.equal_feature_weight:
            matrix = matrix / np.sqrt(matrix.shape[1])

        return matrix

    def _get_feature_matrix(self):
        """Concatenation of all selected (normalized, weighted) features, shape N x sum(D)."""
        blocks = [self._get_single_feature_matrix(name) for name in self.feature_names]
        return np.concatenate(blocks, axis=1)

    def create_clusters(self):
        """(Re)clusters using the current features/method/count/random_state."""
        with self._lock:
            if not self.feature_names:
                self.cluster_labels = None
                return

            matrix = self._get_feature_matrix()
            n_clusters = max(1, min(self.cluster_count, matrix.shape[0]))

            if self.cluster_method == "minibatch_kmeans":
                km = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=self.random_state)
            else:
                km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=self.random_state)

            self.cluster_labels = km.fit_predict(matrix)

    def create_cluster_kmeans(self, cluster_count, random_state):
        self.cluster_method = "kmeans"
        self.cluster_count = cluster_count
        self.random_state = random_state
        self.create_clusters()

    def get_label_count(self):
        labels = self.cluster_labels
        return len(set(labels)) if labels is not None else 0

    def get_cluster_sizes(self):
        labels = self.cluster_labels
        if labels is None:
            return {}
        values, counts = np.unique(labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def get_cluster_audio_excerpts(self, label):
        labels = self.cluster_labels
        if labels is None:
            return None
        mask = (labels == label)
        if not np.any(mask) or mask.shape[0] != self.audio_excerpts.shape[0]:
            return None
        return self.audio_excerpts[mask]


def createModel(config):
    clustering = Clustering(
        config["audio_excerpts"],
        config["audio_features"],
        normalize=config.get("normalize", True),
        selected_features=config.get("selected_features"),
        equal_feature_weight=config.get("equal_feature_weight", True),
        feature_provider=config.get("feature_provider"),
    )
    clustering.create_cluster_kmeans(config["cluster_count"], config["cluster_random_state"])
    return clustering
