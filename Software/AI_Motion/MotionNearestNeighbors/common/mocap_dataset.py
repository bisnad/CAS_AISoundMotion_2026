# dataset for storing and processing motion capture data
# code adapted from original pyTorch implementation of Quaternet 
# quaternet / common / mocap_dataset.py

import numpy as np
import torch
from common.skeleton import Skeleton
from common.quaternion import qeuler_np, qfix

class MocapDataset:
    def __init__(self, path, fps):
        skeleton, data = self._load(path)
        
        self._skeleton = skeleton
        self._data = data
        self._fps = fps
        self._use_gpu = False

    def cuda(self):
        self._use_gpu = True
        self._skeleton.cuda()
        return self
        
    # assumes that all subjects in the dataset possess the same skeleton
    # also assumes that only one action is stored in the dataset
    # mandatoy content in dataset:
    # offsets: relative position offsets of joints with respect to parent joints
    # parents: parent joints in hiearchical skeleton topology
    # rot_local: relative rotations (as quaternions) of joints with respect to parent joints
    # pos_world: absolute 3D positions of joints
    # non-mandatory content in dataset:
    # everyhing else: which will be just copied into the output dataset
    def _load(self, path):
        data = np.load(path, 'r', allow_pickle=True)
        
        non_copy_keys = ["offsets", "parents", "children", "rot_local", "rot_world", "pos_local", "names"]
        
        # create skeleton
        subject = list(data.keys())[0]
        skeleton_offsets = data[subject]["offsets"]
        skeleton_parents = data[subject]["parents"]

        root_joint_index = skeleton_parents.index(-1)

        skeleton = Skeleton(offsets=skeleton_offsets, parents=skeleton_parents)

        # create mocap_data
        mocap_data = {}
        action_name = "A1"

        for subject in data.keys():
            subject_data = data[subject]
    
            rotations = np.copy(subject_data["rot_local"])
            positions = subject_data["pos_world"]
            trajectory = np.copy(positions[:, root_joint_index, :])
            
            mocap_data[subject] = {}    
            mocap_data[subject][action_name] = {
                "rotations": rotations,
                "trajectory": trajectory
                }
        
        # add non-mandatory content to mocap data
        for subject in data.keys():
            subject_data = data[subject]
            
            for key in list(subject_data.keys()):
                
                if key in non_copy_keys:
                    continue
                
                mocap_data[subject][action_name][key] =  subject_data[key]
        
        return skeleton, mocap_data
        
    def downsample(self, factor, keep_strides=True):
        """
        Downsample this dataset by an integer factor, keeping all strides of the data
        if keep_strides is True.
        The frame rate must be divisible by the given factor.
        The sequences will be replaced by their downsampled versions, whose actions
        will have '_d0', ... '_dn' appended to their names.
        """
        assert self._fps % factor == 0
        
        for subject in self._data.keys():
            new_actions = {}
            for action in list(self._data[subject].keys()):
                for idx in range(factor):
                    tup = {}
                    for k in self._data[subject][action].keys():
                        tup[k] = self._data[subject][action][k][idx::factor]
                    new_actions[action + '_d' + str(idx)] = tup
                    if not keep_strides:
                        break
            self._data[subject] = new_actions
            
        self._fps //= factor

    def compute_euler_angles(self, order):
        for subject in self._data.values():
            for action in subject.values():
                action['rotations_euler'] = qeuler_np(action['rotations'], order)
                
    def compute_positions(self):
        
        """
        TODO: since tensorflow doesn't permit the assignment of values to tensors, I'm converting back and forth between numpy arrays and tensors. This is very slow. Maybe there is a better alternative?
        """
        
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0)
                
                if self._use_gpu:
                    rotations = rotations.cuda()
                    trajectory = trajectory.cuda()
                
                action['positions_world'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()
                
                # set root position to zero for calculating local joint positions
                trajectory[:, :, :] = 0   
                action['positions_local'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(0).cpu().numpy()

    def compute_standardized_values(self, value_key):
        
        for subject in self._data.values():
            for action in subject.values():
                values = action[value_key]
                
                std = np.std(values, axis=0) + 1e-10
                mean = np.mean(values, axis=0)
                std_values = (values - mean) / std 

                action[value_key + "_std"] = std
                action[value_key + "_mean"] = mean
                action[value_key + "_standardized"] = std_values
                
                
    def __getitem__(self, key):
        return self._data[key]
    
        
    def subjects(self):
        return self._data.keys()
    
        
    def subject_actions(self, subject):
        return self._data[subject].keys()
        
        
    def all_actions(self):
        result = []
        for subject, actions in self._data.items():
            for action in actions.keys():
                result.append((subject, action))
        return result
    
    
    def fps(self):
        return self._fps
    
    
    def skeleton(self):
        return self._skeleton