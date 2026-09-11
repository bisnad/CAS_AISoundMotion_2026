import threading
import numpy as np
from pythonosc import dispatcher
from pythonosc import osc_server

config = {
    "ip": "0.0.0.0",
    "port": 9003, # Separate port for high-frequency motion data
    "synthesis": None
}

class OscReceiver():
    def __init__(self, config):
        self.ip = config["ip"]
        self.port = config["port"]
        self.synthesis = config["synthesis"]

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/0/joint/pos_local", self.receive_pos_local)
        self.dispatcher.map("/mocap/0/joint/rot_local", self.receive_rot_local)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.current_pos_local = None

    def start_server(self):
        self.server.serve_forever()

    def start(self):
        self.th = threading.Thread(target=self.start_server)
        self.th.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        if hasattr(self, 'th') and self.th.is_alive():
            self.th.join()

    def receive_pos_local(self, address, *args):
        self.current_pos_local = np.array(args).reshape(-1, 3)

    def receive_rot_local(self, address, *args):
        current_rot_local = np.array(args).reshape(-1, 4)
        if self.synthesis:
            self.synthesis.update_live_buffer(self.current_pos_local, current_rot_local)