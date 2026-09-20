import numpy as np

from pythonosc.udp_client import SimpleUDPClient

config = {
    "ip": "127.0.0.1",
    "port": 9005
    }

class OscSender():
    def __init__(self, config):
        self.osc_sender = SimpleUDPClient(config["ip"], config["port"])
        
    def send(self, address, values):

        osc_values = np.reshape(values, (-1)).tolist()
        
        self.osc_sender.send_message(address, osc_values) 
    