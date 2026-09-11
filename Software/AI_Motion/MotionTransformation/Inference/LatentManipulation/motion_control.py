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
        
        self.dispatcher.map("/mocap/seq1index", self.setSeq1Index)
        self.dispatcher.map("/mocap/seq2index", self.setSeq2Index)
        
        self.dispatcher.map("/mocap/seq1frameindex", self.setSeq1FrameIndex)
        self.dispatcher.map("/mocap/seq2frameindex", self.setSeq2FrameIndex)

        self.dispatcher.map("/mocap/seq1time", self.setSeq1Time)
        self.dispatcher.map("/mocap/seq2time", self.setSeq2Time)
        
        self.dispatcher.map("/mocap/seq1framerange", self.setSeq1FrameRange)
        self.dispatcher.map("/mocap/seq2framerange", self.setSeq2FrameRange)

        self.dispatcher.map("/mocap/seq1timerange", self.setSeq1TimeRange)
        self.dispatcher.map("/mocap/seq2timerange", self.setSeq2TimeRange)
        
        self.dispatcher.map("/mocap/seq1frameincr", self.setSeq1FrameIncrement)
        self.dispatcher.map("/mocap/seq2frameincr", self.setSeq2FrameIncrement)        
        
        self.dispatcher.map("/synth/encodingmix", self.setEncodingMix)
        self.dispatcher.map("/synth/encodingoffset", self.setEncodingOffset)  

        self.dispatcher.map("/mocap/0/joint/pos_local", self.receive_pos_local)
        self.dispatcher.map("/mocap/0/joint/rot_local", self.receive_rot_local)
        self.dispatcher.map("/mocap/0/joint/pos_world", self.receive_pos_world)
        self.dispatcher.map("/mocap/0/joint/rot_world", self.receive_rot_world)  
    
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
        
    def setSeq1Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq1Index(index)
        
    def setSeq2Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq2Index(index)
        
    def setSeq1FrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeq1FrameIndex(index)

    def setSeq2FrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeq2FrameIndex(index)

    def setSeq1Time(self, address, *args):
        
        seqTime = args[0]
        seqFrame = int(seqTime * self.synthesis.mocap_fps)
        self.synthesis.setSeq1FrameIndex(seqFrame)

    def setSeq2Time(self, address, *args):
        
        seqTime = args[0]
        seqFrame = int(seqTime * self.synthesis.mocap_fps)
        self.synthesis.setSeq2FrameIndex(seqFrame)
        
    def setSeq1FrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeq1FrameRange(startFrame, endFrame)

    def setSeq2FrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeq2FrameRange(startFrame, endFrame)    

    def setSeq1TimeRange(self, address, *args):
        
        seqStartTime = args[0]
        seqEndTime = args[1]
        seqStartFrame = int(seqStartTime * self.synthesis.mocap_fps)
        seqEndFrame = int(seqEndTime * self.synthesis.mocap_fps)
        
        self.synthesis.setSeq1FrameRange(seqStartFrame, seqEndFrame)

    def setSeq2TimeRange(self, address, *args):
        
        seqStartTime = args[0]
        seqEndTime = args[1]
        seqStartFrame = int(seqStartTime * self.synthesis.mocap_fps)
        seqEndFrame = int(seqEndTime * self.synthesis.mocap_fps)
        
        self.synthesis.setSeq2FrameRange(seqStartFrame, seqEndFrame)  

    def setSeq1FrameIncrement(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setSeq1FrameIncrement(incr)

    def setSeq2FrameIncrement(self, address, *args):
        
        incr = args[0]
        
        self.synthesis.setSeq2FrameIncrement(incr)
        
    def setEncodingMix(self, address, *args):
        
        mix = []
        
        for d in range(self.latent_dim):
        
            mix.append(args[d])
        
        self.synthesis.setEncodingMix(mix)
        
    def setEncodingOffset(self, address, *args):
        
        offset = []
        
        for d in range(self.latent_dim):
        
            offset.append(args[d])
        
        self.synthesis.setEncodingOffset(offset)

    def receive_pos_local(self, address, *args):
        # Cache position until rotation arrives (assuming they are sent together)
        self.current_pos_local = np.array(args).reshape(-1, 3)

    def receive_rot_local(self, address, *args):
        # When rotation arrives, update the synthesis live buffer
        current_rot_local = np.array(args).reshape(-1, 4)
        if self.synthesis:
            self.synthesis.update_live_motion(self.current_pos_local, current_rot_local)
            
    def receive_pos_world(self, address, *args):
        pass # Available for extended usage
        
    def receive_rot_world(self, address, *args):
        pass # Available for extended usage