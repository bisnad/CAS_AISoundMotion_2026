import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "input_length": 64,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.input_length = config["input_length"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/seqindex", self.setSequenceIndex)
        self.dispatcher.map("/mocap/frameindex", self.setFrameIndex)
        self.dispatcher.map("/mocap/seqtime", self.setSequenceTime)
        self.dispatcher.map("/mocap/seqblend", self.setSequenceBlend)
        self.dispatcher.map("/mocap/rand", self.setRand)
        self.dispatcher.map("/mocap/setjointrot", self.setJointRotation)
        self.dispatcher.map("/mocap/changejointrot", self.changeJointRotation)
        self.dispatcher.map("/mocap/setsmoothcutoff", self.setSmoothCutoff)
        self.dispatcher.map("/mocap/setsmoothbeta", self.setSmoothBeta)
        self.dispatcher.map("/synth/setgaussiantemp", self.setGaussianTemp)
        self.dispatcher.map("/synth/setpitemp", self.setPiTemp)
        self.dispatcher.map("/synth/settopp", self.setTopP)
        self.dispatcher.map("/mocap/live/enable", self.setUseLiveMotion)
        self.dispatcher.map("/mocap/live/trigger", self.triggerContinuation)
        self.dispatcher.map("/mocap/live/return", self.returnToLive)

        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()
        
    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        
        self.th2 = threading.Thread(target=self.stop_server)
        self.th2.start()
        
    def setSequenceIndex(self, address, *args):
        
        seq_index = args[0]
        self.synthesis.setOrigSeqIndex(seq_index)
        
    def setFrameIndex(self, address, *args):
        
        if len(args) == 1: # start frame index
        
            seq_start_index = args[0]
            self.synthesis.setOrigSeqStartFrameIndex(seq_start_index)
            
        elif len(args) == 2: # frame index, frame count
        
            seq_start_index = args[0]
            seq_frame_count = args[1]
            
            self.synthesis.setOrigSeqStartFrameIndex(seq_start_index)
            self.synthesis.setOrigSeqFrameCount(seq_frame_count)
            
    def setSequenceTime(self, address, *args):

        if len(args) == 1: # start frame time

            seq_start_time = args[0]
            seq_start_index = int(seq_start_time * self.synthesis.mocap_fps)
            self.synthesis.setOrigSeqStartFrameIndex(seq_start_index)

    def setSequenceBlend(self, address, *args):
        
        blend = args[0]
        
        self.synthesis.setOrigSeqBlend(blend)
        
    def setRand(self, address, *args):
        
        rand = args[0]
        
        self.synthesis.setRandRange(rand)
        
    def setJointRotation(self, address, *args):
        
        if len(args) == 5: # start frame index, rotation axis, rotation angle
        
            joint_index = args[0]
            rot_axis = np.array([args[1], args[2], args[3]])
            rot_angle = args[4]
            
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            self.synthesis.setJointRotation(joint_index, rot_quat, 1)
        elif len(args) > 5:
            
            rot_axis = np.array([args[-4], args[-3], args[-2]])
            rot_angle = args[-1]
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            for aI in range(len(args) - 4):
                joint_index = args[aI]
                
                self.synthesis.setJointRotation(joint_index, rot_quat, 1)
            
        """
        elif len(args) == 6: # start frame index, rotation axis, rotation angle, frame_count
        
            joint_index = args[0]
            rot_axis = np.array([args[1], args[2], args[3]])
            rot_angle = args[4]
            frame_count = args[5]
            
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            self.synthesis.setJointRotation(joint_index, rot_quat, frame_count)
        """

                
            
    def changeJointRotation(self, address, *args):
        
        if len(args) == 5: # start frame index, rotation axis, rotation angle
        
            joint_index = args[0]
            rot_axis = np.array([args[1], args[2], args[3]])
            rot_angle = args[4]
            
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            self.synthesis.changeJointRotation(joint_index, rot_quat, 1)
        elif len(args) > 5:
            
            rot_axis = np.array([args[-4], args[-3], args[-2]])
            rot_angle = args[-1]
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            for aI in range(len(args) - 4):
                joint_index = args[aI]
                
                self.synthesis.changeJointRotation(joint_index, rot_quat, 1)

            
        """
        elif len(args) == 6: # start frame index, rotation axis, rotation angle, frame_count
        
            joint_index = args[0]
            rot_axis = np.array([args[1], args[2], args[3]])
            rot_angle = args[4]
            frame_count = args[5]
            
            rot_quat = t3d.quaternions.axangle2quat(rot_axis, rot_angle)
            
            self.synthesis.changeJointRotation(joint_index, rot_quat, frame_count)
        """

    def setGaussianTemp(self, address, *args):
        gaussian_temp = args[0]
        self.synthesis.setGaussianTemp(gaussian_temp)

    def setPiTemp(self, address, *args):
        pi_temperature = args[0]
        self.synthesis.setPiTemp(pi_temperature)

    def setTopP(self, address, *args):
        top_p = args[0]
        self.synthesis.setTopP(top_p)

    def setSmoothCutoff(self, address, *args):
        cutoff = args[0]
        self.synthesis.one_euro_filter.setMinCutOff(cutoff)

    def setSmoothBeta(self, address, *args):
        beta = args[0]
        self.synthesis.one_euro_filter.setBeta(beta)

    def setUseLiveMotion(self, address, *args):
        self.synthesis.use_live_motion = bool(args[0])

    def triggerContinuation(self, address, *args):
        self.synthesis.trigger_continuation()
        
    def returnToLive(self, address, *args):
        self.synthesis.return_to_live()
