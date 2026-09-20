import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "latent_dim": 32,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.latent_dim = config["latent_dim"]
        self.ip = config["ip"]
        self.port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        # Shutdown signals serve_forever() to stop and blocks until it does.
        self.server.shutdown()
        # Once the loop stops, we can safely close the server and port.
        self.server.server_close()
        # Wait for the thread to cleanly exit
        if hasattr(self, 'th') and self.th.is_alive():
            self.th.join()

