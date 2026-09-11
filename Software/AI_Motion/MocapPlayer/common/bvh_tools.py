import numpy as np
import pandas as pd
import re

class BVHScanner:
    def __init__(self):
        self.rules = [
            ('ROOT', r'ROOT'),
            ('JOINT', r'JOINT'),
            ('End_Site', r'End Site'),
            ('HIERARCHY', r'HIERARCHY'),
            ('MOTION', r'MOTION'),
            ('Frames', r'Frames:'),
            ('Frame', r'Frame'),
            ('Time', r'Time:'),
            ('OFFSET', r'OFFSET'),
            ('CHANNELS', r'CHANNELS'),
            ('LBRACK', r'\{'),
            ('RBRACK', r'\}'),
            ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            # Broadened FLOAT to catch edge cases like -.05, .05, etc.
            ('FLOAT', r'-*[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'),
            ('INTEGER', r'-?[0-9]+'),
            # Use \s+ to safely consume Windows \r\n line endings
            ('WS', r'\s+'), 
        ]
        self.rules = [(type, re.compile(regex)) for type, regex in self.rules]

    def scan(self, text):
        tokens = []
        pos = 0
        length = len(text)
        
        while pos < length:
            match = None
            for type, regex in self.rules:
                # Use the 'pos' parameter to search without slicing the string
                match = regex.match(text, pos)
                if match:
                    if type != 'WS':
                        tokens.append((type, match.group(0)))
                    pos = match.end()
                    break
                    
            if not match:
                raise SyntaxError(f'Unexpected token at {text[pos:pos+20]}')
                
        return tokens

class BVHData:
    def __init__(self):
        self.skeleton = {}
        self.channel_names = []
        self.values = pd.DataFrame()
        self.root_name = ''
        self.framerate = 0.0
        self.times_per_joint = {} # ADDED: holds keyframe times per joint name

class BVH_Tools:

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.scanner = BVHScanner()
        self.data = BVHData()
        self._skeleton = {}
        self._motion_channels = []
        self._motions = []
        self.current_token = 0
        self.root_name = ''
        self.framerate = 0.0

    def load(self, filename):
        self.reset()

        with open(filename, 'r') as bvh_file:
            raw_contents = bvh_file.read()

        tokens = self.scanner.scan(raw_contents)
        self._parse_hierarchy(tokens)
        self._parse_motion(tokens)
        
        self.data.skeleton = self._skeleton
        self.data.channel_names = self._motion_channels
        self.data.values = self._to_DataFrame()
        self.data.root_name = self.root_name
        self.data.framerate = self.framerate

        return self.data
    
    def _parse_hierarchy(self, bvh):
        self.current_token = 0
        if bvh[self.current_token][0] != 'HIERARCHY':
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][0] != 'ROOT':
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][0] != 'IDENT':
            return None
        self.root_name = bvh[self.current_token][1]
        self._parse_joint(bvh)

    def _parse_joint(self, bvh, is_end_site=False, parent_name=""):
        # If it's an End Site, there is no name token in the file. 
        # We auto-generate one using the parent's name.
        if is_end_site:
            joint_name = parent_name + '_EndSite'
        else:
            joint_name = bvh[self.current_token][1]
            self.current_token = self.current_token + 1

        joint_data = {'name': joint_name, 'offset': [], 'channels': [], 'children': []}

        if bvh[self.current_token][0] != 'LBRACK':
            return None
        self.current_token = self.current_token + 1
        if bvh[self.current_token][0] != 'OFFSET':
            return None
        self.current_token = self.current_token + 1
        
        for i in range(3):
            if bvh[self.current_token][0] not in ['FLOAT', 'INTEGER']:
                return None
            joint_data['offset'].append(float(bvh[self.current_token][1]))
            self.current_token = self.current_token + 1
            
        if bvh[self.current_token][0] == 'CHANNELS':
            self.current_token = self.current_token + 1
            num_channels = int(bvh[self.current_token][1])
            self.current_token = self.current_token + 1
            for i in range(num_channels):
                channel_name = bvh[self.current_token][1]
                joint_data['channels'].append(channel_name)
                self._motion_channels.append((joint_name, channel_name))
                self.current_token = self.current_token + 1
                
        # FIX: Add the joint to the dictionary BEFORE parsing its children,
        # so that parents always appear before children in the skeleton keys!
        self._skeleton[joint_name] = joint_data
                
        while bvh[self.current_token][0] in ['JOINT', 'End_Site']:
            is_child_end_site = (bvh[self.current_token][0] == 'End_Site')
            self.current_token = self.current_token + 1
            
            child_data = self._parse_joint(bvh, is_end_site=is_child_end_site, parent_name=joint_name)
            
            if child_data is None:
                print(f"Error parsing child of {joint_name}")
                return None
                
            joint_data['children'].append(child_data['name'])

        if bvh[self.current_token][0] != 'RBRACK':
            return None
        
        self.current_token = self.current_token + 1

        return joint_data

    def _parse_motion(self, bvh):
        
        # Check by token type (index 0) to avoid mismatch issues
        if bvh[self.current_token][0] != 'MOTION':
            print('Unexpected text, expected MOTION')
            return None
        self.current_token = self.current_token + 1
        
        if bvh[self.current_token][0] != 'Frames':
            print('Unexpected text, expected Frames:')
            return None
        self.current_token = self.current_token + 1
        frame_count = int(bvh[self.current_token][1])
        self.current_token = self.current_token + 1
        
        if bvh[self.current_token][0] != 'Frame':
            print('Unexpected text, expected Frame Time:')
            return None
        self.current_token = self.current_token + 1
        
        if bvh[self.current_token][0] != 'Time':
            print('Unexpected text, expected Time:')
            return None
        self.current_token = self.current_token + 1
        frame_rate = float(bvh[self.current_token][1])

        self.framerate = frame_rate
       
        self.current_token = self.current_token + 1
       
        frame_time = 0.0
        self._motions = [()] * frame_count
        for i in range(frame_count):
            channel_values = []
            for channel in self._motion_channels:
                channel_values.append((channel[0], channel[1], float(bvh[self.current_token][1])))
                self.current_token = self.current_token + 1
            self._motions[i] = (frame_time, channel_values)
            frame_time = frame_time + frame_rate
            
        joint_names = set(channel[0] for channel in self._motion_channels)
        time_array = np.arange(frame_count) * frame_rate
        self.data.times_per_joint = {j_name: time_array for j_name in joint_names}

    def _to_DataFrame(self):
        time = [m[0] for m in self._motions]
        cols = [c[0] + '_' + c[1] for c in self._motion_channels]
        data = np.array([[c[2] for c in m[1]] for m in self._motions])
        df = pd.DataFrame(data, index=time, columns=cols)
        return df


class bvhWriter:
    """
    A class for writing BVH files from mocap data.
    """
    
    def __init__(self, mocap_data):
        self.mocap_data = mocap_data
        
    def write(self, filename):
        
        skeleton_data = self.mocap_data["skeleton"]
        motion_data = self.mocap_data["motion"]
        
        # open file
        bvh_file = open(filename, 'w')
        
        # write hierarchy
        bvh_file.write("HIERARCHY\n")
        
        joints = skeleton_data["joints"]
        parents = skeleton_data["parents"]
        children = skeleton_data["children"]
        offsets = skeleton_data["offsets"]
        
        self.write_hierarchy(bvh_file, 0, joints, parents, children, offsets, 0)
        
        # write motion
        bvh_file.write("MOTION\n")
        
        pos_local = motion_data["pos_local"]
        rot_local_euler = motion_data["rot_local_euler"]

        rot_sequence = self.mocap_data["rot_sequence"]
        frame_rate = self.mocap_data["frame_rate"]
        
        # check number of frames
        if pos_local.shape[0] != rot_local_euler.shape[0]:
            print("ERROR: pos_local and rot_local_euler have different number of frames")
            
        frame_count = pos_local.shape[0]
        
        bvh_file.write("Frames: " + str(frame_count) + "\n")
        bvh_file.write("Frame Time: " + str(frame_rate) + "\n")
        
        self.write_motion(bvh_file, pos_local, rot_local_euler, rot_sequence)

        # close file
        bvh_file.close()
        
    def write_hierarchy(self, bvh_file, joint_index, joints, parents, children, offsets, level):

        indent = "    " * level
        
        # write joint name
        if level == 0:
            bvh_file.write(indent + "ROOT " + joints[joint_index] + "\n")
        else:
            bvh_file.write(indent + "JOINT " + joints[joint_index] + "\n")
            
        bvh_file.write(indent + "{\n")
        
        # write offset
        offset = offsets[joint_index]
        bvh_file.write(indent + "    OFFSET " + str(offset[0]) + " " + str(offset[1]) + " " + str(offset[2]) + "\n")
        
        # write channels
        if level == 0:
            bvh_file.write(indent + "    CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n")
        else:
            bvh_file.write(indent + "    CHANNELS 3 Xrotation Yrotation Zrotation\n")
            
        # write children
        joint_children = children[joint_index]
        
        if len(joint_children) == 0:
            bvh_file.write(indent + "    End Site\n")
            bvh_file.write(indent + "    {\n")
            bvh_file.write(indent + "        OFFSET 0.0 0.0 0.0\n")
            bvh_file.write(indent + "    }\n")
        else:
            for child_index in joint_children:
                self.write_hierarchy(bvh_file, child_index, joints, parents, children, offsets, level + 1)
                
        bvh_file.write(indent + "}\n")

    def write_motion(self, bvh_file, pos_local, rot_local_euler, rot_sequence):

        # loop through frames
        for frame_index in range(pos_local.shape[0]):
            
            # loop through joints
            for joint_index in range(pos_local.shape[1]):
                
                # get joint data
                pos = pos_local[frame_index, joint_index, :]
                rot = rot_local_euler[frame_index, joint_index, :]
                
                # write joint data
                if joint_index == 0:
                    bvh_file.write(str(pos[0]) + " " + str(pos[1]) + " " + str(pos[2]) + " ")
                    
                bvh_file.write(str(rot[0]) + " " + str(rot[1]) + " " + str(rot[2]) + " ")
                
            bvh_file.write("\n")