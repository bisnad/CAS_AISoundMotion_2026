import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server

"""
AudioControl: unchanged in overall shape from the original (threaded OSC
server dispatching to synthesis/model methods, same pattern as
motion_receiver.MotionReceiver), with two additions matching the new
capabilities in audio_model.py / audio_synthesis.py:

- /synth/clustercount   -> Clustering.set_cluster_count
- /synth/clustermethod  -> Clustering.set_cluster_method ("kmeans" or
                            "minibatch_kmeans")

These let an external OSC client (e.g. the mocap or audio-analysis tools,
or a separate controller) drive re-clustering remotely, not just cluster
selection and feature choice as in the original.
"""

config = {"synthesis": None,
          "model": None,
          "ip": "127.0.0.1",
          "port": 9004}


class AudioControl():

    def __init__(self, config):

        self.synthesis = config["synthesis"]
        self.model = config["model"]
        self.ip = config["ip"]
        self.port = config["port"]

        self.dispatcher = dispatcher.Dispatcher()

        self.dispatcher.map("/synth/clusterlabel", self.setClusterLabel)
        self.dispatcher.map("/synth/audiofeature", self.selectAudioFeature)
        self.dispatcher.map("/synth/clustercount", self.setClusterCount)
        self.dispatcher.map("/synth/clustermethod", self.setClusterMethod)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)

    def start_server(self):
        self.server.serve_forever()

    def start(self):

        self.th = threading.Thread(target=self.start_server, daemon=True)
        self.th.start()

    def stop(self):
        self.server.server_close()

    def setClusterLabel(self, address, *args):

        label = args[0]

        self.synthesis.setClusterLabel(label)

    def selectAudioFeature(self, address, *args):

        featureName = args[0]

        self.synthesis.selectAudioFeature(featureName)

    def setClusterCount(self, address, *args):

        cluster_count = int(args[0])

        self.model.set_cluster_count(cluster_count)
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())

    def setClusterMethod(self, address, *args):

        method = args[0]

        self.model.set_cluster_method(method)
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
