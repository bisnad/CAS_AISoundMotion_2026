import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

"""
model architecture

Revisions vs. the original audio_model.py:

1. Feature normalization: raw feature magnitudes vary wildly in scale
   (e.g. mel spectrogram energy vs. spectral centroid in Hz vs. a [0,1]
   flatness value). K-means uses Euclidean distance, so without
   normalization, whichever feature happens to have the largest raw
   magnitude dominates the clustering regardless of how informative it
   actually is. Every feature is now z-score normalized (StandardScaler)
   before fitting, and normalize=False lets you opt back into the
   old (unnormalized) behavior for comparison.

2. Fixed the select_audio_feature()/create_clusters() inconsistency: the
   original selected a feature and called create_clusters(), which only
   ever reads self.cluster_method to decide which private _create_clusters_*
   method to call, but never re-applied the cluster_count/random_state the
   caller might have intended. create_clusters() now always re-clusters
   using whatever parameters (cluster_count, random_state, method) are
   currently stored on the instance, so switching features or calling
   create_clusters() directly is consistent regardless of call path.

3. Added MiniBatchKMeans as a second cluster_method option, useful when the
   excerpt count is large (many minutes of audio at a small excerpt
   offset can easily produce tens of thousands of excerpts) and full
   KMeans becomes slow.

4. Added get_cluster_sizes() / get_feature_names() / get_current_feature()
   for the GUI to display cluster population counts and available
   descriptors without reaching into private state.

5. Added set_data(), letting a NEW audio file's excerpts/features be
   swapped into an EXISTING Clustering instance (used by the "load audio
   file" GUI workflow), instead of requiring the whole model/GUI/synthesis
   stack to be torn down and rebuilt whenever a different file is loaded.
   Clears the per-feature StandardScaler cache since scalers fit on the
   old data are no longer valid for the new one.
"""

config = {
    "audio_excerpts": None,
    "audio_features": None,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170,
    "normalize": True,
}


class Clustering():
    def __init__(self, audio_excerpts, audio_features, normalize=True):

        self.audio_excerpts = audio_excerpts
        self.audio_features = audio_features
        self.normalize = normalize

        self.feature_name = self._default_feature_name()

        self.cluster_method = "kmeans"
        self.cluster_labels = None

        self.cluster_count = 20
        self.random_state = 170

        self._scaler_cache = {}

    def _default_feature_name(self):
        # skip the "waveform" pseudo-feature (raw audio samples) when
        # picking a default - clustering directly on raw waveform samples
        # is rarely meaningful and was never intended to be selected here
        feature_names = [name for name in self.audio_features.keys() if name != "waveform"]
        return feature_names[0] if feature_names else list(self.audio_features.keys())[0]

    def set_data(self, audio_excerpts, audio_features):
        """
        Swaps in a new set of audio excerpts/features (e.g. after loading a
        different audio file), keeping the current cluster_method,
        cluster_count, random_state and normalize settings. If the
        currently selected feature name does not exist in the new feature
        set, falls back to the new set's default feature. Clears cached
        StandardScalers, since they were fit on the previous data and do
        not apply to the new one. Does NOT re-cluster automatically - call
        create_clusters() (or select_audio_feature()) afterwards.
        """
        self.audio_excerpts = audio_excerpts
        self.audio_features = audio_features
        self._scaler_cache = {}

        if self.feature_name not in self.audio_features:
            self.feature_name = self._default_feature_name()

        self.cluster_labels = None

    def get_feature_names(self):
        return [name for name in self.audio_features.keys() if name != "waveform"]

    def get_current_feature(self):
        return self.feature_name

    def select_audio_feature(self, feature_name):

        if feature_name in self.audio_features:
            self.feature_name = feature_name
            self.create_clusters()

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

    def _get_feature_matrix(self):
        matrix = self.audio_features[self.feature_name]

        if not self.normalize:
            return matrix

        if self.feature_name not in self._scaler_cache:
            self._scaler_cache[self.feature_name] = StandardScaler()
            return self._scaler_cache[self.feature_name].fit_transform(matrix)

        return self._scaler_cache[self.feature_name].transform(matrix)

    def create_clusters(self):
        """
        (Re)runs clustering using whichever feature/method/cluster_count/
        random_state are currently set on this instance. This is the single
        entry point used by select_audio_feature(), set_cluster_method(),
        set_cluster_count(), set_random_state() and set_normalize(), so all
        of them behave consistently.
        """
        matrix = self._get_feature_matrix()

        if self.cluster_method == "minibatch_kmeans":
            model = MiniBatchKMeans(
                n_clusters=self.cluster_count, n_init="auto", random_state=self.random_state
            )
        else:
            model = KMeans(
                n_clusters=self.cluster_count, n_init="auto", random_state=self.random_state
            )

        self.cluster_labels = model.fit_predict(matrix)

    def create_cluster_kmeans(self, cluster_count, random_state):
        """
        Kept for backward compatibility with the original API - sets
        parameters and (re)clusters using k-means specifically.
        """
        self.cluster_method = "kmeans"
        self.cluster_count = cluster_count
        self.random_state = random_state
        self.create_clusters()

    def get_label_count(self):
        return len(set(self.cluster_labels)) if self.cluster_labels is not None else 0

    def get_cluster_sizes(self):
        """
        Returns a dict {label: count} for every cluster label currently
        present, useful for a GUI list showing how many excerpts fall into
        each cluster.
        """
        if self.cluster_labels is None:
            return {}
        labels, counts = np.unique(self.cluster_labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}

    def get_cluster_audio_excerpts(self, label):

        if self.cluster_labels is None:
            return None

        mask = (self.cluster_labels == label)

        if not np.any(mask):
            return None

        return self.audio_excerpts[mask]


def createModel(config):

    clustering = Clustering(
        config["audio_excerpts"],
        config["audio_features"],
        normalize=config.get("normalize", True),
    )
    clustering.create_cluster_kmeans(config["cluster_count"], config["cluster_random_state"])

    return clustering
