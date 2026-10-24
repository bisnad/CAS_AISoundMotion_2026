import numpy as np
import pathlib
import pickle

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, q_conj_np, qmul_np, slerp, qfix

config = { 
    "file_name": "data/mocap/accumulation_fullbody_take1.bvh"
    }

class MotionPlayer():
    def __init__(self, config):

        self.load(config)
        
    def load(self, config):

        self.file_name = config["file_name"]
        self.root_joint_index = config["root_joint_index"]
        
        file_suffix = pathlib.Path(self.file_name).suffix

        if(file_suffix == ".bvh" or file_suffix == ".BVH"):
            self.load_bvh(self.file_name)
        elif(file_suffix == ".fbx" or file_suffix == ".FBX"):
            self.load_fbx(self.file_name)
        elif(file_suffix == ".pkl"):
            self.load_pkl(self.file_name, config)

        # scale positions
        pos_scale = config["pos_scale"]
        self.mocap_data["motion"]["pos_local"] *= pos_scale
        self.mocap_data["motion"]["pos_world"] *= pos_scale

        # remove root pos
        self.remove_root_position(self.mocap_data, self.root_joint_index)
        self.remove_root_rotation(self.mocap_data, self.root_joint_index)
        
        self.fps = config["fps"]

        # update start, end, and play position
        self.play_frame = 0
        self.start_play_frame = 0
        self.end_play_frame = self.mocap_data["motion"]["pos_world"].shape[0] - 1

    def load_bvh(self, file_name):
        # load mocap data
        bvh_tools = bvh.BVH_Tools()
        mocap_tools = mocap.Mocap_Tools()

        bvh_data = bvh_tools.load(file_name)
        self.mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
        self.mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(self.mocap_data["motion"]["rot_local_euler"] , self.mocap_data["rot_sequence"])
        self.mocap_data["motion"]["pos_world"], self.mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(self.mocap_data["motion"]["rot_local"], self.mocap_data["motion"]["pos_local"], self.mocap_data["skeleton"])

    def load_fbx(self, file_name):
        # load mocap data
        fbx_tools = fbx.FBX_Tools()
        mocap_tools = mocap.Mocap_Tools()
        
        fbx_data = fbx_tools.load(file_name)

        all_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)
        self.mocap_data = all_mocap_data[0] # use only mocap data of first skeleton
        self.mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(self.mocap_data["motion"]["rot_local_euler"] , self.mocap_data["rot_sequence"])
        self.mocap_data["motion"]["pos_world"], self.mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(self.mocap_data["motion"]["rot_local"], self.mocap_data["motion"]["pos_local"], self.mocap_data["skeleton"])

    def load_pkl(self, file_name, config):
        with open(file_name, "rb") as f:
            pkl_data = pickle.load(f)

        self.mocap_data = self.pkl_to_mocap(pkl_data)
        self.mocap_data["skeleton"] = {}
        self.mocap_data["skeleton"]["parents"] = config["parents"]
        self.mocap_data["skeleton"]["children"] = config["children"]

        print(self.mocap_data["motion"]["pos_local"].shape)
        print(self.mocap_data["motion"]["pos_world"].shape)


    @staticmethod
    def pkl_to_mocap(pkl_data):
        
        mocap_data = {}
        mocap_data["motion"] = {}
        
        # unique osc message addresses
        sensor_ids = pkl_data["sensor_ids"]
        sensor_values = pkl_data["sensor_values"]
        
        # get unique sensor ids
        unique_sensor_ids = list(set(sensor_ids))
        #print("unique_sensor_ids ", unique_sensor_ids)
        
        # get rid of all unique sensor ids that don't correspond to skeleton 0
        unique_sensor_ids = [ unique_sensor_id for unique_sensor_id in unique_sensor_ids if "/mocap/0" in unique_sensor_id ]
        #print("unique_sensor_ids ", unique_sensor_ids)
        
        # get indices of unique sensor ids into sensor values list
        unique_sensor_indices = {}
        for unique_sensor_id in unique_sensor_ids:
            unique_indices = [i for i, s in enumerate(sensor_ids) if s == unique_sensor_id]
            unique_sensor_indices[unique_sensor_id] = unique_indices
        
        # create motion data
        for unique_sensor_id in unique_sensor_ids:
            
            indices = unique_sensor_indices[unique_sensor_id]
            values = [sensor_values[i] for i in indices]
            
            mocap_data["motion"][unique_sensor_id.replace("/mocap/0/joint/", "")] = np.array(values)


        return mocap_data

    @staticmethod
    def remove_root_position(mocap_data, root_joint_index):
        
        pos_local = mocap_data["motion"]["pos_local"]
        pos_world = mocap_data["motion"]["pos_world"]
        
        #print("pos_local s ", pos_local.shape)
        #print("pos_world s ", pos_world.shape)

        mocap_joint_count = len(mocap_data["skeleton"]["parents"])
        
        pos_local = pos_local.reshape(-1, mocap_joint_count, 3)
        pos_world = pos_world.reshape(-1, mocap_joint_count, 3)
        
        #print("pos_local 2 s ", pos_local.shape)
        #print("pos_world 2 s ", pos_world.shape)
        
        pos_local[:, root_joint_index, :] = 0.0
        pos_world_root = np.expand_dims(pos_world[:, root_joint_index, :], 1)
        
        #print("pos_world_root s ", pos_world_root.shape)
        
        pos_world -= pos_world_root
        
        mocap_data["motion"]["pos_local"] = pos_local
        mocap_data["motion"]["pos_world"] = pos_world

    @staticmethod
    def quat_relative(q1, q2, normalize_inputs=True):
        """
        Relative rotation that takes q1 to q2, batched.
        q1, q2: shape (N, 4) (or any broadcastable (..., 4))
        Returns q_rel: shape (N, 4), with q_rel * q1 = q2.
        """
        if normalize_inputs:
            q1 = qnormalize_np(q1)
            q2 = qnormalize_np(q2)

        q1_inv = q_conj_np(q1)   # for unit quats, inverse == conjugate
        return qmul_np(q2, q1_inv)

    @staticmethod
    def remove_root_rotation(mocap_data, root_joint_index):
        rot_local = mocap_data["motion"]["rot_local"]
        rot_world = mocap_data["motion"]["rot_world"]

        mocap_joint_count = len(mocap_data["skeleton"]["parents"])

        rot_local = rot_local.reshape(-1, mocap_joint_count, 4)
        rot_world = rot_world.reshape(-1, mocap_joint_count, 4)

        # set root local rotation to identity
        rot_local[:, root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))

        # extract root world rotation (F,4)
        rot_world_root = rot_world[:, root_joint_index, :]

        # inverse root rotation (F,4)
        rot_world_root_inv = q_conj_np(qnormalize_np(rot_world_root))

        # broadcast to (F,J,4)
        rot_world_root_inv = np.expand_dims(rot_world_root_inv, 1)
        rot_world_root_inv = np.repeat(rot_world_root_inv, repeats=mocap_joint_count, axis=1)

        # apply: q_new = q_root_inv * q_world
        rot_world = qmul_np(rot_world_root_inv, rot_world)

        rot_local = rot_local.reshape(-1, mocap_joint_count * 4)
        rot_world = rot_world.reshape(-1, mocap_joint_count * 4)

        mocap_data["motion"]["rot_local"] = rot_local
        mocap_data["motion"]["rot_world"] = rot_world
        
    def update(self):

        if self.play_frame >= self.end_play_frame:
            self.play_frame = self.start_play_frame
        
        self.play_frame += 1
        
    def get_file_name(self):
        return self.file_name
        
    def get_fps(self):
        return self.fps
    
    def set_fps(self, fps):
        self.fps = fps
        
    def get_play_frame(self):
        return self.play_frame
        
    def set_play_frame(self, frame):

        if frame >= self.end_play_frame:
            frame = self.end_play_frame - 1
            
        if frame <= self.start_play_frame:
             frame = self.start_play_frame + 1
             
        self.play_frame = frame
        
    def get_start_play_frame(self):
        return self.start_play_frame
        
    def set_start_play_frame(self, frame):
        
        if frame >= self.end_play_frame:
            frame = self.end_play_frame - 1

        self.start_play_frame = frame
        
        if self.play_frame < self.start_play_frame:
            self.play_frame = self.start_play_frame
        
        #self.play_frame  = self.start_play_frame
    
    def get_end_play_frame(self):
        return self.end_play_frame
        
    def set_end_play_frame(self, frame):
        
        if frame <= self.start_play_frame:
             frame = self.start_play_frame + 1

        self.end_play_frame = frame
        
        if self.play_frame > self.end_play_frame:
            self.play_frame = self.end_play_frame
        
        #self.play_frame  = self.end_play_frame

    def get_skeleton(self):
        return self.mocap_data["skeleton"]

    def get_pose(self, pose_feature):
        
        return self.mocap_data["motion"][pose_feature][self.play_frame, ...]
