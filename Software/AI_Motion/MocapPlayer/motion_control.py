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
        self.dispatcher.map("/player/time", self.player_set_time)
        self.dispatcher.map("/player/start_time", self.player_set_start_time)
        self.dispatcher.map("/player/end_time", self.player_set_end_time)
        
        # Changed from frames to time (percentage 0-1000 to match GUI slider, or exact seconds)
        # Using the exact seconds is better for OSC control
        self.dispatcher.map("/player/time", self.player_set_time)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        # Politely tell the serve_forever() loop to stop
        self.server.shutdown()
        # Wait for the background thread to finish exiting
        if hasattr(self, 'th') and self.th.is_alive():
            self.th.join()
        # Now it is safe to close the socket
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
        
    def player_set_time(self, address, *args):
        # Allow passing time in seconds directly
        play_time_seconds = args[0]
        end_t = getattr(self.gui.player, 'end_time', 0.0)
        
        if end_t > 0:
            # Map time back to the GUI's slider scale (0-1000)
            val = int((play_time_seconds / end_t) * 1000)
            self.gui.change_play_time(val)

    def player_set_start_time(self, address, *args):
        start_time_seconds = args[0]
        max_t = getattr(self.gui.player, 'max_time', 0.0)
        if max_t > 0:
            val = int((start_time_seconds / max_t) * 1000)
            self.gui.change_start_time(val)

    def player_set_end_time(self, address, *args):
        end_time_seconds = args[0]
        max_t = getattr(self.gui.player, 'max_time', 0.0)
        if max_t > 0:
            val = int((end_time_seconds / max_t) * 1000)
            self.gui.change_end_time(val)