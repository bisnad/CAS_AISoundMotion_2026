import threading
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server

"""
AudioControl: threaded OSC server dispatching to synthesis/model methods.

OSC addresses:
/synth/clusterlabel  <int>
/synth/audiofeature  <name> [<name> ...]   one or several feature names; a single
                     string may also hold a comma separated list,
                     e.g. "mfcc,root mean square". Features that were not computed
                     yet are computed on demand (the OSC call blocks meanwhile).
/synth/clustercount  <int>
/synth/clustermethod <"kmeans" | "minibatch_kmeans">
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
        self.server.shutdown()
        self.server.server_close()

    def setClusterLabel(self, address, *args):
        self.synthesis.setClusterLabel(int(args[0]))

    def selectAudioFeature(self, address, *args):
        names = []
        for arg in args:
            names.extend([s.strip() for s in str(arg).split(",") if s.strip()])

        if not names:
            return

        try:
            self.synthesis.selectAudioFeatures(names)
        except Exception as e:
            print("[audio_control] could not select features {}: {}".format(names, e))

    def setClusterCount(self, address, *args):
        self.model.set_cluster_count(int(args[0]))
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())

    def setClusterMethod(self, address, *args):
        self.model.set_cluster_method(args[0])
        self.synthesis.setClusterLabel(self.synthesis.get_cluster_label())
