import numpy as np
import pathlib

from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap

config = { 
    "file_name": "",
    "topology_file_name": "",
    "fps": 50
    }

class MotionPlayer():
    def __init__(self, config):
        self.file_name = config.get("file_name", "")
        self.topology_file_name = config.get("topology_file_name", "")
        self.base_fps = config.get("fps", 50)

        self.skeletons_data = [] 
        
        self.fps = self.base_fps 
        self.play_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        self.max_time = 0.0
        
        if self.file_name:
            if "topology_file_name" in config:
                self.load(self.file_name, config["topology_file_name"])
            else:
                self.load(self.file_name)
        
    def load(self, file_name, skel_parents = None):
        self.file_name = file_name
        file_suffix = pathlib.Path(file_name).suffix.lower()

        if file_suffix == ".bvh":
            raw_data = self.load_bvh(file_name)
        elif file_suffix == ".fbx":
            raw_data = self.load_fbx(file_name)
        elif file_suffix == ".npz":
            raw_data = self.load_npz(file_name, skel_parents)
        else:
            return
            
        self.skeletons_data = raw_data
        
        if len(self.skeletons_data) > 0:
            self.start_time = 0.0
            max_t = 0.0
            
            for skel in self.skeletons_data:
                fps = skel.get("frame_rate", 50.0) 
                
                # Scan the actual exact timestamps
                if "times" in skel["motion"]:
                    for joint_name, times in skel["motion"]["times"].items():
                        if len(times) > 0:
                            local_max = np.max(times)
                            if local_max > max_t:
                                max_t = float(local_max)
                
                # Check maximum lengths based on list/array sizes as fallback
                if "pos_local" in skel["motion"]:
                    pos_local = skel["motion"]["pos_local"]
                    is_dense = not isinstance(pos_local, list)
                    if is_dense:
                        num_frames = pos_local.shape[0]
                    else:
                        num_frames = max([len(p) for p in pos_local] + [0])
                        
                    duration = max(0.0, (num_frames - 1) / fps)
                    if duration > max_t:
                        max_t = float(duration)
                        
            self.max_time = float(max_t)
            if self.max_time <= 0.001:
                self.max_time = 1.0 
                
            self.end_time = self.max_time 
            self.play_time = 0.0
            self.base_fps = self.skeletons_data[0].get("frame_rate", 50.0)
            if self.base_fps <= 0.0: self.base_fps = 50.0
            self.fps = self.base_fps

    def load_bvh(self, file_name):
        bvh_tools_inst = bvh.BVH_Tools()
        mocap_tools_inst = mocap.Mocap_Tools()

        bvh_data = bvh_tools_inst.load(file_name)
        mocap_data = mocap_tools_inst.bvh_to_mocap(bvh_data)

        mocap_data["motion"]["rot_local"] = mocap_tools_inst.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]
            
        return mocap_data

    def load_fbx(self, file_name):
        fbx_tools_inst = fbx.FBX_Tools()
        mocap_tools_inst = mocap.Mocap_Tools()
        
        fbx_data = fbx_tools_inst.load(file_name)
        mocap_data = mocap_tools_inst.fbx_to_mocap(fbx_data)
        
        for skel_data in mocap_data:
            skel_data["motion"]["rot_local"] = mocap_tools_inst.euler_to_quat(skel_data["motion"]["rot_local_euler"], skel_data["rot_sequence"])
        
        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]

        return mocap_data

    def load_npz(self, file_name, topology_file_name):
        mocap_tools_inst = mocap.Mocap_Tools()
        npz_data, topo_data = npz.NPZ_Tools.load(file_name, topology_file_name)
        mocap_data = mocap_tools_inst.npz_to_mocap(npz_data, topo_data, self.base_fps)

        # Ensure it returns a list of skeletons
        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]
            
        return mocap_data

    def get_fps(self):
        return self.fps
        
    def set_fps(self, fps):
        self.fps = fps

    def get_play_time(self):
        return self.play_time

    def set_play_time(self, play_time):
        if play_time >= self.end_time:
            self.play_time = self.end_time - 0.001
        elif play_time <= self.start_time:
            self.play_time = self.start_time
        else:
            self.play_time = play_time

    def get_start_time(self):
        return self.start_time
        
    def set_start_time(self, t):
        if t >= self.end_time:
            t = self.end_time - 0.001
        self.start_time = max(0.0, t)
        if self.play_time < self.start_time:
            self.play_time = self.start_time
            
    def get_end_time(self):
        return self.end_time
        
    def set_end_time(self, t):
        if t <= self.start_time:
             t = self.start_time + 0.001
        self.end_time = min(self.max_time, t)
        if self.play_time > self.end_time:
            self.play_time = self.end_time

    def update(self, dt):
        speed_factor = self.fps / self.base_fps if self.base_fps > 0 else 1.0
        self.play_time += dt * speed_factor
        if self.play_time > self.end_time:
            self.play_time = self.start_time

    def get_frame(self):
        current_poses = []
        if not self.skeletons_data:
            return current_poses
            
        mocap_tools_inst = mocap.Mocap_Tools()
            
        for skel_idx, skel_data in enumerate(self.skeletons_data):
            pos_local_list = skel_data["motion"]["pos_local"]
            rot_local_list = skel_data["motion"]["rot_local"]
            
            is_dense = not isinstance(pos_local_list, list)
            has_variable_times = "times" in skel_data["motion"]
            fps = skel_data.get("frame_rate", self.fps)
            if fps <= 0.0: fps = 50.0
            
            curr_pos_local = []
            curr_rot_local = []
            
            for joint_idx, joint_name in enumerate(skel_data["skeleton"]["joints"]):
                if is_dense:
                    positions = pos_local_list[:, joint_idx, :]
                    rotations = rot_local_list[:, joint_idx, :]
                else:
                    positions = pos_local_list[joint_idx]
                    rotations = rot_local_list[joint_idx]
                
                if has_variable_times and joint_name in skel_data["motion"]["times"]:
                    times = skel_data["motion"]["times"][joint_name]
                    if len(times) == 0:
                        curr_pos_local.append(np.zeros(3))
                        curr_rot_local.append(np.array([1.0, 0.0, 0.0, 0.0])) # Identity
                        continue
                        
                    idx = np.searchsorted(times, self.play_time)
                    if idx == 0:
                        curr_pos_local.append(positions[0])
                        curr_rot_local.append(rotations[0])
                    elif idx == len(times):
                        curr_pos_local.append(positions[-1])
                        curr_rot_local.append(rotations[-1])
                    else:
                        t0, t1 = float(times[idx-1]), float(times[idx])
                        p0, p1 = positions[idx-1], positions[idx]
                        r0, r1 = rotations[idx-1], rotations[idx]
                        
                        if t1 <= t0:
                            curr_pos_local.append(p0)
                            curr_rot_local.append(r0)
                        else:
                            alpha = float(self.play_time - t0) / (t1 - t0)
                            curr_pos_local.append(p0 + alpha * (p1 - p0))
                            # Quick normalized lerp for quaternions
                            dot = np.sum(r0 * r1)
                            if dot < 0.0: r1 = -r1
                            rq = r0 + alpha * (r1 - r0)
                            curr_rot_local.append(rq / np.linalg.norm(rq))
                else:
                    exact_frame = self.play_time * fps
                    num_frames = positions.shape[0]
                    frame_0 = max(0, min(num_frames - 1, int(np.floor(exact_frame))))
                    frame_1 = max(0, min(num_frames - 1, int(np.ceil(exact_frame))))
                    p0, p1 = positions[frame_0], positions[frame_1]
                    r0, r1 = rotations[frame_0], rotations[frame_1]
                    
                    if frame_0 == frame_1:
                        curr_pos_local.append(p0)
                        curr_rot_local.append(r0)
                    else:
                        alpha = exact_frame - frame_0
                        curr_pos_local.append(p0 + alpha * (p1 - p0))
                        dot = np.sum(r0 * r1)
                        if dot < 0.0: r1 = -r1
                        rq = r0 + alpha * (r1 - r0)
                        curr_rot_local.append(rq / np.linalg.norm(rq))
            
            # Format inputs correctly for real-time local_to_world
            arr_pos = np.array(curr_pos_local)[np.newaxis, ...]
            arr_rot = np.array(curr_rot_local)[np.newaxis, ...]
            
            pos_world, rot_world = mocap_tools_inst.local_to_world(arr_rot, arr_pos, skel_data["skeleton"])
            
            # Format inputs correctly for real-time local_to_world
            arr_pos = np.array(curr_pos_local)[np.newaxis, ...]
            arr_rot = np.array(curr_rot_local)[np.newaxis, ...]

            pos_world, rot_world = mocap_tools_inst.local_to_world(arr_rot, arr_pos, skel_data["skeleton"])

            current_poses.append({
                "id": skel_idx,
                "skeleton": skel_data["skeleton"],
                "pos_local": arr_pos[0],          # ADDED
                "rot_local": arr_rot[0],          # ADDED
                "pos_world": pos_world[0],
                "rot_world": rot_world[0]         # ADDED
            })
            
        return current_poses