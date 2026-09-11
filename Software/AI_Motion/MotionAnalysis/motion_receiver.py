import threading
import numpy as np
import re
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server

config = {"messages": [],
          "data": [],
          "ip": "127.0.0.1",
          "port": 9007}

class MotionReceiver():
    
    def __init__(self, config):
        
        self.messages = config["messages"]
        self.data = config["data"]
        
        self.ip = config["ip"]
        self.port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        self.motion_data = {}
        
        for message in self.messages:
            
            self.dispatcher.map(message, self.receive)
            self.motion_data["message"] = None
            
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)

    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()            
            
    def receive(self, address, *args):

        #print("receive address ", address)

        # Convert OSC patterns to regex (replace * with .*)
        def osc_pattern_to_regex(pattern):
            return re.compile('^' + re.escape(pattern).replace('\\*', '.*') + '$')

        matching_index = None
        for idx, pattern in enumerate(self.messages):
            regex = osc_pattern_to_regex(pattern)
            if regex.match(address):
                matching_index = idx
                break

        if matching_index is None:
            # If no pattern matches, ignore the message or handle error
            return
        
        values = np.array(args)
        data_shape = self.data[matching_index].shape
        values = np.reshape(values, data_shape)
        np.copyto(self.data[matching_index], values)