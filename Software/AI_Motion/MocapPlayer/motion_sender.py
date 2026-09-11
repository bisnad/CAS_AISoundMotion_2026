import numpy as np

from pythonosc.udp_client import SimpleUDPClient

config = {
    "ip": "127.0.0.1",
    "port": 9005
    }

class OscSender():
    def __init__(self, config):
        
        self.ip = config["ip"]
        self.port = config["port"]
        self.active = True
        
        self._create_client()
    
    def _create_client(self):
        self.osc_sender = SimpleUDPClient(self.ip, self.port)
        
    def get_address(self):
        return self.ip, self.port
    
    def set_address(self, ip, port):
        self.ip = ip
        self.port = port
        
        self._create_client()
    
    def get_active(self):
        return self.active
     
    def set_active(self, active):
        self.active = active


    def send(self, address, values):
        
        if self.active == False:
            return

        osc_values = np.reshape(values, (-1)).tolist()
        
        self.osc_sender.send_message(address, osc_values) 
    