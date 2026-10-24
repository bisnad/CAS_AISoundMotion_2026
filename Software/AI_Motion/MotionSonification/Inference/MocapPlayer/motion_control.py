import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server

import motion_gui


config = {"player": None,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.gui = config["gui"]
        self.ip = config["ip"]
        self.port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/player/load", self.player_load)
        self.dispatcher.map("/player/start", self.player_start)
        self.dispatcher.map("/player/stop", self.player_stop)
        self.dispatcher.map("/player/fps", self.player_set_fps)
        self.dispatcher.map("/player/frame", self.player_set_frame)
        self.dispatcher.map("/player/start_frame", self.player_set_start_frame)
        self.dispatcher.map("/player/end_frame", self.player_set_end_frame)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def player_load(self, address, *args):
        
        filename = args[0]
        
        self.gui.stop()
        self.gui.load_file(filename)
        
    def player_start(self, address, *args):
        
        self.gui.start()
        
    def player_stop(self, address, *args):
        
        self.gui.stop()
        
    def player_set_fps(self, address, *args):
        
        fps = args[0]

        self.gui.change_fps(fps)        
        
    def player_set_frame(self, address, *args):
        
        frame = args[0]

        self.gui.change_play_frame(frame)
        
    def player_set_start_frame(self, address, *args):
        
        frame = args[0]

        self.gui.change_start_play_frame(frame)
        
    def player_set_end_frame(self, address, *args):
        
        frame = args[0]

        self.gui.change_end_play_frame(frame)