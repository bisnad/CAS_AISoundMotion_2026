import pandas
import math
import numpy as np
import transforms3d as t3d
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from common import bvh_tools as bvh
from common import fbx_tools as fbx
import copy

class Mocap_Tools:

    def bvh_to_mocap(self, bvh_data):
        
        mocap_data = {}
        
        mocap_data["frame_rate"] = 1.0 / bvh_data.framerate
        
        rot_channels = list(bvh_data.skeleton.values())[0]["channels"][3:]
        rot_channel_names = ["Xrotation", "Yrotation", "Zrotation"]
        mocap_data["rot_sequence"]  = [ rot_channel_names.index(rot_channel) for rot_channel in rot_channels ]
        
        self._create_skeleton_data(bvh_data, mocap_data)
        self._create_motion_data(bvh_data, mocap_data)
        
        return mocap_data
    
    def fbx_to_mocap(self, fbx_data):
        
        all_motion_data = []

        for fbx_per_skel_data in fbx_data:
            
            skeleton = {}
            skeleton["root"] = fbx_per_skel_data.skeleton_root
            skeleton["joints"] = fbx_per_skel_data.skeleton_joints
            skeleton["parents"] = fbx_per_skel_data.skeleton_parents
            skeleton["children"] = fbx_per_skel_data.skeleton_children
            skeleton["offsets"] = fbx_per_skel_data.skeleton_joint_offsets
            
            motion = {}
            motion["pos_local"] = fbx_per_skel_data.motion_pos_local
            motion["rot_local_euler"] = fbx_per_skel_data.motion_rot_local_euler
            
            motion_data = {}
            motion_data["frame_rate"] = fbx_per_skel_data.motion_frame_rate 
            motion_data["rot_sequence"] = fbx_per_skel_data.motion_rot_sequence
            motion_data["skeleton"] = skeleton
            motion_data["motion"] = motion
            
            all_motion_data.append(motion_data)
    
        return all_motion_data

    def npz_to_mocap(self, npz_data, topology_data, fps):
        
        # 1. Discover all unique skeleton indices from the keys
        skeleton_indices = set()
        for key in npz_data.keys():
            parts = key.split('/')
            if len(parts) > 2 and parts[1] == 'mocap' and parts[2].isdigit():
                skeleton_indices.add(int(parts[2]))
                
        skeleton_indices = sorted(list(skeleton_indices))
        all_mocap_data = []
        
        for skel_idx in skeleton_indices:
            print(f"--- Processing Skeleton {skel_idx} ---")
            
            # 2. Extract data using dynamically formatted strings
            pos_world = npz_data[f'/mocap/{skel_idx}/joint/pos_world_values']
            pos_world_time  = npz_data[f'/mocap/{skel_idx}/joint/pos_world_timestamps']
            pos_local = npz_data[f'/mocap/{skel_idx}/joint/pos_local_values']
            pos_local_time  = npz_data[f'/mocap/{skel_idx}/joint/pos_local_timestamps']
            rot_world = npz_data[f'/mocap/{skel_idx}/joint/rot_world_values']
            rot_world_time  = npz_data[f'/mocap/{skel_idx}/joint/rot_world_timestamps']
            rot_local = npz_data[f'/mocap/{skel_idx}/joint/rot_local_values']
            rot_local_time  = npz_data[f'/mocap/{skel_idx}/joint/rot_local_timestamps']

            pos_world = np.reshape(pos_world, (pos_world.shape[0], pos_world.shape[1] // 3, 3))
            pos_local = np.reshape(pos_local, (pos_local.shape[0], pos_local.shape[1] // 3, 3))
            rot_world = np.reshape(rot_world, (rot_world.shape[0], rot_world.shape[1] // 4, 4))
            rot_local = np.reshape(rot_local, (rot_local.shape[0], rot_local.shape[1] // 4, 4))

            # resample
            # remove non-incremental times from the time_arrays
            valid_indices_pos_world_time = np.concatenate(([True], np.diff(pos_world_time) > 0))
            valid_indices_pos_local_time = np.concatenate(([True], np.diff(pos_local_time) > 0))
            valid_indices_rot_world_time = np.concatenate(([True], np.diff(rot_world_time) > 0))
            valid_indices_rot_local_time = np.concatenate(([True], np.diff(rot_local_time) > 0))
            
            pos_world = pos_world[valid_indices_pos_world_time, :, :]
            pos_world_time = pos_world_time[valid_indices_pos_world_time]
            pos_local = pos_local[valid_indices_pos_local_time, :, :]
            pos_local_time = pos_local_time[valid_indices_pos_local_time]
            rot_world = rot_world[valid_indices_rot_world_time, :, :]
            rot_world_time = rot_world_time[valid_indices_rot_world_time]
            rot_local = rot_local[valid_indices_rot_local_time, :, :]
            rot_local_time = rot_local_time[valid_indices_rot_local_time]

            # Find the overlapping valid time window
            start_time = max([pos_world_time[0], pos_local_time[0], rot_world_time[0], rot_local_time[0]])
            end_time = min([pos_world_time[-1], pos_local_time[-1], rot_world_time[-1], rot_local_time[-1]])
            time_target = np.arange(start_time, end_time, 1.0 / fps)
            frame_count = len(time_target)
            joint_count = pos_local.shape[1]

            # Linear interpolation for positions
            interp_pos_world = interp1d(pos_world_time, pos_world, axis=0, kind='linear')
            resampled_pos_world = interp_pos_world(time_target)
            
            interp_pos_local = interp1d(pos_local_time, pos_local, axis=0, kind='linear')
            resampled_pos_local = interp_pos_local(time_target)

            # Spherical Linear Interpolation (Slerp) for quaternions
            resampled_rot_world = np.zeros((frame_count, joint_count, 4))
            resampled_rot_local = np.zeros((frame_count, joint_count, 4))
            for j in range(joint_count):
                slerp_rot_world = Slerp(rot_world_time, R.from_quat(rot_world[:, j, :]))
                resampled_rot_world[:, j, :] = slerp_rot_world(time_target).as_quat()
                
                slerp_rot_local = Slerp(rot_local_time, R.from_quat(rot_local[:, j, :]))
                resampled_rot_local[:, j, :] = slerp_rot_local(time_target).as_quat()

            # fill motion dictionary
            mocap_joints = topology_data[2]
            mocap_parents = topology_data[0]
            mocap_children = topology_data[1]
            
            # --- SIMPLIFIED OFFSET CALCULATION ---
            # Because pos_local is now properly computed by the sender (as true local space),
            # the offsets are simply the mean of pos_local across time for each joint.
            offsets = np.zeros((joint_count, 3))
            
            for j_idx in range(joint_count):
                parent_idx = mocap_parents[j_idx]
                if parent_idx == -1:
                    if len(resampled_pos_local) > 0:
                        offsets[j_idx] = resampled_pos_local[0, j_idx].copy()
                        # root is grounded in X/Z
                        offsets[j_idx, 0] = 0.0
                        offsets[j_idx, 2] = 0.0
                else:
                    if len(resampled_pos_local) > 0:
                        # Direct average since resampled_pos_local is already in parent-relative space
                        offsets[j_idx] = np.mean(resampled_pos_local[:, j_idx, :], axis=0)
            # --------------------------------

            # Find root dynamically for downstream components
            root_name = mocap_joints[0]
            for i, p in enumerate(mocap_parents):
                if p == -1:
                    root_name = mocap_joints[i]
                    break

            mocap_data = {}
            mocap_data["frame_rate"] = fps 
            
            mocap_data["skeleton"] = {}
            mocap_data["skeleton"]["root"] = root_name
            mocap_data["skeleton"]["joints"] = mocap_joints
            mocap_data["skeleton"]["parents"] = mocap_parents
            mocap_data["skeleton"]["children"] = mocap_children
            mocap_data["skeleton"]["offsets"] = offsets
            
            mocap_data["motion"] = {}
            mocap_data["motion"]["pos_world"] = resampled_pos_world
            mocap_data["motion"]["pos_local"] = resampled_pos_local
            mocap_data["motion"]["rot_world"] = resampled_rot_world
            mocap_data["motion"]["rot_local"] = resampled_rot_local

            all_mocap_data.append(mocap_data)

        return all_mocap_data
    
    @staticmethod
    def resample_euler_mocap_data(mocap_data, target_fps, time_ranges):
        times_dict = mocap_data["motion"].get("times", {})
        joints = mocap_data["skeleton"]["joints"]
        num_joints = len(joints)

        pos_local = mocap_data["motion"]["pos_local"]
        rot_local_euler = mocap_data["motion"]["rot_local_euler"]

        def get_joint_data(data, j_idx):
            if isinstance(data, list):
                return data[j_idx]
            else:
                return data[:, j_idx, :]

        joint_times_list = []
        for j_idx, j_name in enumerate(joints):
            if j_name in times_dict:
                j_times = times_dict[j_name]
            else:
                j_frames = len(get_joint_data(pos_local, j_idx))
                orig_fps = mocap_data.get("frame_rate", target_fps)
                j_times = np.arange(j_frames) / orig_fps
            joint_times_list.append(j_times)

        if time_ranges is None:
            max_t = 0.0
            for jt in joint_times_list:
                if len(jt) > 0:
                    max_t = max(max_t, jt[-1])
            time_ranges = [[0.0, max_t]]

        resampled_segments = []

        for t_range in time_ranges:
            start_time, end_time = t_range[0], t_range[1]
            target_times = np.arange(start_time, end_time, 1.0 / target_fps)
            num_frames = len(target_times)

            if num_frames == 0:
                continue

            new_pos_local = np.zeros((num_frames, num_joints, 3))
            new_rot_local_euler = np.zeros((num_frames, num_joints, 3))

            for j_idx in range(num_joints):
                j_times = joint_times_list[j_idx]
                j_pos = get_joint_data(pos_local, j_idx)
                j_rot = get_joint_data(rot_local_euler, j_idx)

                if len(j_times) == 0:
                    continue

                if len(j_times) == 1:
                    new_pos_local[:, j_idx, :] = j_pos[0]
                    new_rot_local_euler[:, j_idx, :] = j_rot[0]
                    continue

                for i in range(3):
                    new_pos_local[:, j_idx, i] = np.interp(target_times, j_times, j_pos[:, i])

                j_rot_rad = np.deg2rad(j_rot)
                j_rot_rad_unwrapped = np.unwrap(j_rot_rad, axis=0)
                j_rot_deg_unwrapped = np.rad2deg(j_rot_rad_unwrapped)

                for i in range(3):
                    new_rot_local_euler[:, j_idx, i] = np.interp(target_times, j_times, j_rot_deg_unwrapped[:, i])

            segment_data = copy.deepcopy(mocap_data)
            segment_data["motion"]["pos_local"] = new_pos_local
            segment_data["motion"]["rot_local_euler"] = new_rot_local_euler
            segment_data["frame_rate"] = target_fps

            if "times" in segment_data["motion"]:
                del segment_data["motion"]["times"]

            resampled_segments.append(segment_data)

        return resampled_segments

    def mocap_to_bvh(self, mocap_data):
        
        bvh_data = bvh.BVH_Data()
        bvh_data.framerate = 1.0 / mocap_data["frame_rate"]
        bvh_data.root_name = mocap_data["skeleton"]["root"]
        
        bvh_channel_names, bvh_channels = self._create_bvh_channel_names(mocap_data)
        bvh_data.channel_names = bvh_channel_names
        
        bvh_skeleton = self._create_bvh_skeleton(mocap_data, bvh_channels)
        bvh_data.skeleton = bvh_skeleton
        
        bvh_frames = self._create_bvh_frames(mocap_data, bvh_channels)
        bvh_data.values = bvh_frames
        
        return bvh_data
    
    # warning: this doesn't create any fbx nodes
    # the nodes are only created when writing the fmx_data to a file using the FBX_Tools 
    def mocap_to_fbx(self, all_motion_data):
        
        fbx_data = []
        
        for motion_per_skel_data in all_motion_data:
        
            fbx_per_skel_data = fbx.FBX_Mocap_Data()
            
            motion_skeleton = motion_per_skel_data["skeleton"]
            motion_motion = motion_per_skel_data["motion"]
            
            fbx_per_skel_data.motion_frame_rate = motion_per_skel_data["frame_rate"]
            fbx_per_skel_data.motion_rot_sequence = motion_per_skel_data["rot_sequence"]
            fbx_per_skel_data.skeleton_root = motion_skeleton["root"]
            fbx_per_skel_data.skeleton_joints = motion_skeleton["joints"]
            fbx_per_skel_data.skeleton_children = motion_skeleton["children"]
            fbx_per_skel_data.skeleton_parents = motion_skeleton["parents"]
            fbx_per_skel_data.skeleton_joint_offsets = motion_skeleton["offsets"]
            fbx_per_skel_data.motion_pos_local = motion_motion["pos_local"]
            fbx_per_skel_data.motion_rot_local_euler = motion_motion["rot_local_euler"]
            fbx_per_skel_data.motion_frame_count = fbx_per_skel_data.motion_rot_local_euler.shape[0]

            fbx_data.append(fbx_per_skel_data)
            
        return fbx_data
    
    def local_to_world(self, rot_local, pos_local, skeleton):
        
        root_name = skeleton["root"]
        joint_names = skeleton["joints"]
        root_index = joint_names.index(root_name)
        parents = skeleton["parents"]
        children = skeleton["children"]
        offsets = skeleton["offsets"]

        root_positions = pos_local[:, root_index, :]

        frame_count = rot_local.shape[0]
        joint_count = rot_local.shape[1]
        
        # 1. Compute Topological Execution Order dynamically
        # This ensures parents are always processed before their children
        execution_order = []
        _visited = set()
        
        def _visit(j_idx):
            if j_idx in _visited: return
            p_idx = parents[j_idx]
            if p_idx != -1 and p_idx not in _visited:
                _visit(p_idx)
            execution_order.append(j_idx)
            _visited.add(j_idx)
            
        for jI in range(joint_count):
            _visit(jI)
        
        positions_world = []
        rotations_world = []
        
        for fI in range(frame_count):
            
            # 2. Pre-allocate lists so we can assign to specific indices
            frame_positions_world = [None] * joint_count
            frame_rotations_world = [None] * joint_count
            
            # 3. Iterate over the safe execution order instead of sequentially
            for jI in execution_order:
                
                if parents[jI] == -1:
                    frame_positions_world[jI] = root_positions[fI]
                    frame_rotations_world[jI] = rot_local[fI, jI]
                else:
                    p_idx = parents[jI]
                    parent_pos = frame_positions_world[p_idx]
                    parent_rot = frame_rotations_world[p_idx]
                    
                    # Ensure quaternions are safely handled using scipy (x,y,z,w expected)
                    try:
                        r_parent = R.from_quat(parent_rot)
                    except ValueError:
                        # Fallback to identity if invalid
                        r_parent = R.from_quat([0.0, 0.0, 0.0, 1.0])
                    
                    # Rotate the offset by the parent's global rotation and add parent's global pos
                    rotated_offset = r_parent.apply(offsets[jI])
                    frame_positions_world[jI] = rotated_offset + parent_pos
                    
                    if len(children[jI]) > 0:
                        try:
                            r_local = R.from_quat(rot_local[fI, jI])
                            # Global rotation = Parent global * Local rotation
                            frame_rotations_world[jI] = (r_parent * r_local).as_quat()
                        except ValueError:
                            frame_rotations_world[jI] = parent_rot
                    else:
                        # End effectors/nubs inherit parent rotation
                        frame_rotations_world[jI] = parent_rot
                        
            frame_positions_world = np.stack(frame_positions_world, axis=0)
            frame_rotations_world = np.stack(frame_rotations_world, axis=0)
            
            positions_world.append(frame_positions_world)
            rotations_world.append(frame_rotations_world)
            
        positions_world = np.stack(positions_world, axis=0)
        rotations_world = np.stack(rotations_world, axis=0)
        
        return positions_world, rotations_world

    def euler_to_quat(self, rot_euler, rot_sequence):
        # In FBX, RotationOrder is an enum/int where:
        # 0 = XYZ, 1 = XZY, 2 = YZX, 3 = YXZ, 4 = ZXY, 5 = ZYX, 6 = SphericXYZ
        if hasattr(rot_sequence, 'value'):
            rot_val = rot_sequence.value
        elif isinstance(rot_sequence, int):
            rot_val = rot_sequence
        else:
            rot_val = 0
            
        rot_mapping = {
            0: "xyz", 1: "xzy", 2: "yzx", 3: "yxz", 4: "zxy", 5: "zyx", 6: "xyz" 
        }
        rot_seq_string = rot_mapping.get(rot_val, "xyz")

        if isinstance(rot_euler, list):
            rot_quat = []
            for joint_euler in rot_euler:
                if len(joint_euler) == 0:
                    rot_quat.append(np.zeros((0, 4)))
                    continue
                rot_scipy = R.from_euler(rot_seq_string, joint_euler, degrees=True)
                # Keep natively as [x, y, z, w]
                rot_quat.append(rot_scipy.as_quat())
            return rot_quat
        else:
            frame_count = rot_euler.shape[0]
            joint_count = rot_euler.shape[1]
            rot_quat = np.zeros((frame_count, joint_count, 4))
            for jI in range(joint_count):
                rot_scipy = R.from_euler(rot_seq_string, rot_euler[:, jI, :], degrees=True)
                # Keep natively as [x, y, z, w]
                rot_quat[:, jI, :] = rot_scipy.as_quat()
            return rot_quat

    def euler_to_quat_bvh(self, rot_euler, rot_sequence):
        frame_count = rot_euler.shape[0]
        joint_count = rot_euler.shape[1]
        rot_quat = np.zeros((frame_count, joint_count, 4))
        
        for fI in range(frame_count):
            for jI in range(joint_count):
                rx = math.radians(rot_euler[fI, jI, 0])
                ry = math.radians(rot_euler[fI, jI, 1])
                rz = math.radians(rot_euler[fI, jI, 2])
                
                qx = [math.cos(rx/2.0), math.sin(rx/2.0), 0.0, 0.0]
                qy = [math.cos(ry/2.0), 0.0, math.sin(ry/2.0), 0.0]
                qz = [math.cos(rz/2.0), 0.0, 0.0, math.sin(rz/2.0)]
                
                qs = [qx, qy, qz]
                
                qr = t3d.quaternions.qmult(qs[rot_sequence[0]], qs[rot_sequence[1]])
                qr = t3d.quaternions.qmult(qr, qs[rot_sequence[2]])
                
                # BUGFIX: t3d generates [w, x, y, z]. We convert it to SciPy's [x, y, z, w]
                rot_quat[fI, jI, 0] = qr[1] # x
                rot_quat[fI, jI, 1] = qr[2] # y
                rot_quat[fI, jI, 2] = qr[3] # z
                rot_quat[fI, jI, 3] = qr[0] # w

        return rot_quat

    def quat_to_euler(self, rotations_quat, rot_sequence):
        rot_string = "".join([ "xyz"[i] for i in rot_sequence ])
        seq_length = rotations_quat.shape[0]
        
        rotations_quat = np.reshape(rotations_quat, (-1, 4))
        # Removed `scalar_first=True` since rotations_quat is now cleanly [x, y, z, w]
        rotations_euler = R.from_quat(rotations_quat).as_euler(rot_string, degrees=True)
        rotations_euler = np.reshape(rotations_euler, (seq_length, -1, 3))
                
        return rotations_euler

    def quat_to_euler_bvh(self, rotations_quat, rot_sequence):
        frame_count = rotations_quat.shape[0]
        joint_count = rotations_quat.shape[1]
        rotations_euler = []
        
        for fI in range(frame_count):
            joint_rotations_euler = []
            for jI in range(joint_count):
                rotation_quat = rotations_quat[fI, jI]
                
                # t3d expects [w, x, y, z], so we temporarily map our [x, y, z, w] backward
                t3d_quat = [rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]]
                
                rotation_euler = np.array(t3d.euler.quat2euler(t3d_quat, axes="syxz"))
                rotation_euler  *= 180.0 / math.pi
                rotation_euler = np.array((rotation_euler[1], rotation_euler[0], rotation_euler[2]))

                joint_rotations_euler.append(rotation_euler)

            rotations_euler.append(joint_rotations_euler)
                
        rotations_euler = np.stack(rotations_euler, axis=0)
                
        return rotations_euler
    
    def remove_joints(self, mocap_data, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        
        Important: assumes that the root joint is not removed
        """
        
        skeleton = mocap_data["skeleton"]
        
        parents = skeleton["parents"]
        children = skeleton["children"]
        joints = skeleton["joints"]

        # gather joints that are not removed
        valid_parents = []
        valid_children = []
        valid_joints = []
        
        for parent in range(len(parents)):
            if parent not in joints_to_remove:
                valid_parents.append(parent)
                valid_children.append(children[parent])
                valid_joints.append(joints[parent])
                
        #print("parents ", parents)
        #print("valid_parents ", valid_parents)
        #print("children ", children)
        #print("valid_children ", valid_children)
        #print("joints ", joints)
        #print("valid_joints ", valid_joints)
        
        # remove offsets / pos_local / rot_local_euler of non-valid joints
        motion = mocap_data["motion"]
        
        offsets = skeleton["offsets"]
        pos_local = motion["pos_local"]
        rot_local_euler = motion["rot_local_euler"]
        
        new_offsets = offsets[valid_parents, :]
        new_pos_local = pos_local[:, valid_parents, :]
        new_rot_local_euler = rot_local_euler[:, valid_parents, :]
        
        #print("offsets s ", offsets.shape)
        #print("new_offsets s ", new_offsets.shape)
        #print("pos_local s ", pos_local.shape)
        #print("new_pos_local s ", new_pos_local.shape)
        #print("rot_local_euler s ", rot_local_euler.shape)
        #print("new_rot_local_euler s ", new_rot_local_euler.shape)


                
        # renumber parents
        index_offsets = np.zeros(len(parents), dtype=int)
        
        new_parents = []
        for i, parent in enumerate(parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
                
        #print("valid_parents ", valid_parents)
        #print("new_parents ", new_parents)
                
        valid_to_new_parent_map = { valid_parents[i] : new_parents[i] for i in range(len(new_parents)) }
        
        #print("valid_to_new_parent_map ", valid_to_new_parent_map)
        
        # renumber children
        new_children = []
        for i, parent in enumerate(new_parents):
            new_children.append([])
        for i, parent in enumerate(new_parents):
            if parent != -1:
                new_children[parent].append(i)
        

        #print("valid_children ", valid_children)
        #print("new_children ", new_children)
        
        # new joint names
        new_joints = valid_joints
        
        # create new mocap data
        
        new_skeleton = {}
        new_skeleton["children"] = new_children
        new_skeleton["joints"] = new_joints
        new_skeleton["offsets"] = new_offsets
        new_skeleton["parents"] = new_parents
        new_skeleton["root"] = skeleton["root"]
        
        new_motion = {}
        new_motion["pos_local"] = new_pos_local
        new_motion["rot_local_euler"] = new_rot_local_euler
        
        new_mocap_data = {}
        
        new_mocap_data["frame_rate"] = mocap_data["frame_rate"]
        new_mocap_data["motion"] = new_motion
        new_mocap_data["rot_sequence"] = mocap_data["rot_sequence"]
        new_mocap_data["skeleton"] = new_skeleton
                
        return new_mocap_data
    
    def mocap_excerpt(self, mocap_data, start_frame=-1, end_frame=-1):
        
        mocap_data_excerpt = copy.deepcopy(mocap_data)
        
        motion_data = mocap_data_excerpt["motion"]

        full_frame_count = motion_data[list(motion_data.keys())[0]].shape[0]

        if start_frame == -1:
            start_frame = 0
    
        if end_frame == -1 or end_frame > full_frame_count:
            end_frame = full_frame_count
            
        for key in motion_data.keys():
            
            motion_data[key] = motion_data[key][start_frame:end_frame, ...]
            
            #values = values[start_frame:end_frame, ...]
            
        return mocap_data_excerpt
            
 
    def _create_skeleton_data(self, bvh_data, mocap_data):
        
        skeleton_data = {}
        
        joint_names = []
        joint_child_lists = []
        joint_offsets = []
        
        # 1. First pass: Collect names, children, and offsets
        for joint_name, joint_info in bvh_data.skeleton.items():
            joint_names.append(joint_name)
            
            # Extract children securely (ignoring capitalization just in case)
            children = []
            for key, val in joint_info.items():
                if key.lower() == 'children':
                    children = val
                    break
            joint_child_lists.append(children)
            
            offset = [0.0, 0.0, 0.0]
            for key, val in joint_info.items():
                if 'offset' in key.lower():
                    try:
                        offset = [float(x) for x in val]
                    except (ValueError, TypeError):
                        pass
                    break
            joint_offsets.append(offset)
            
        # 2. Reconstruct parent mapping entirely from the children lists
        # This guarantees we find the parent even if the 'parent' key is completely missing
        parent_map = {name: None for name in joint_names}
        for parent_name, children in zip(joint_names, joint_child_lists):
            for child_name in children:
                if child_name in parent_map:
                    parent_map[child_name] = parent_name
                    
        # 3. Create joint_parent_indices safely based on our reconstructed map
        joint_parent_indices = []
        for name in joint_names:
            p_name = parent_map[name]
            if p_name in joint_names:
                joint_parent_indices.append(joint_names.index(p_name))
            else:
                joint_parent_indices.append(-1)
        
        # 4. Map child names to indices securely
        joint_child_indices = []
        for children in joint_child_lists:
            c_indices = [joint_names.index(c) for c in children if c in joint_names]
            joint_child_indices.append(c_indices)
            
        joint_offsets = np.stack(joint_offsets, axis=0)
        
        skeleton_data["root"] = bvh_data.root_name
        skeleton_data["joints"] = joint_names
        skeleton_data["parents"] = joint_parent_indices
        skeleton_data["children"] = joint_child_indices
        skeleton_data["offsets"] = joint_offsets

        mocap_data["skeleton"] = skeleton_data
        
        return mocap_data
    
    def _create_motion_data(self, bvh_data, mocap_data):
        
        motion = {}
        
        joint_names = mocap_data["skeleton"]["joints"]
        joint_offsets = mocap_data["skeleton"]["offsets"]
        joint_count = len(joint_names)
        rot_sequence = mocap_data["rot_sequence"]
        
        bvh_frames = bvh_data.values
        frame_count = bvh_frames.shape[0]
        
        bvh_frames_column_names = [ column for column in bvh_data.values.columns ]
        bvh_channels = set(bvh_data.channel_names)
        bvh_channel_joint_names = set([channel[0] for channel in bvh_channels])
        bvh_channel_value_names = ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"]
        
        motion_translation = []
        motion_euler_rotation = []

        for jI, joint_name in enumerate(joint_names):

            if joint_name in bvh_channel_joint_names:
                joint_frames_combined = []      
            
                for i, value_name in enumerate(bvh_channel_value_names):
                    column_name = joint_name + "_" + value_name
                
                    if column_name in bvh_frames_column_names:
                        joint_frames_combined.append(np.array(bvh_frames[column_name]))
                    else:
                        # If a position channel doesn't exist dynamically (like for most child joints),
                        # fallback to the static skeleton offset for that axis!
                        if i < 3:
                            static_val = np.full(frame_count, joint_offsets[jI][i])
                            joint_frames_combined.append(static_val)
                        else:
                            joint_frames_combined.append(np.zeros(frame_count))

                joint_translations = joint_frames_combined[:3]
                joint_rotations = joint_frames_combined[3:]
                
                joint_translations = np.array(joint_translations)
                joint_rotations = np.array(joint_rotations)

                joint_translations = np.transpose(joint_translations)
                joint_rotations = np.transpose(joint_rotations)
                
                motion_translation.append(joint_translations)
                motion_euler_rotation.append(joint_rotations)
            else:
                # If the joint isn't in the animation channels at all (like an End Site / nub),
                # its local position is STRICTLY its skeletal offset, repeated across all frames.
                joint_translations = np.tile(joint_offsets[jI], (frame_count, 1))
                joint_rotations = np.zeros((frame_count, 3))
                
                motion_translation.append(joint_translations)
                motion_euler_rotation.append(joint_rotations)
                
        motion_translation = np.stack(motion_translation, axis=1)
        motion_euler_rotation = np.stack(motion_euler_rotation, axis=1)
        
        motion["pos_local"] = motion_translation
        motion["rot_local_euler"] = motion_euler_rotation
        
        mocap_data["motion"] = motion
        
        return mocap_data

    def _create_bvh_channel_names(self, mocap_data):
        
        joints = mocap_data["skeleton"]["joints"]
        children = mocap_data["skeleton"]["children"]
        rot_sequence = mocap_data["rot_sequence"]
        
        pos_channel_names = ["Xposition", "Yposition", "Zposition"]
        rot_channel_names = ["Xrotation", "Yrotation", "Zrotation"]

        bvh_channels = pos_channel_names + [ rot_channel_names[i] for i in rot_sequence ]
        
        bvh_channel_names = []
        
        for jI, joint_name in enumerate(joints):
            
            # ignore nub joints
            if len(children[jI]) == 0:
                continue

            for channel_name in bvh_channels:
                
                bvh_channel_names.append((joint_name, channel_name))
            
        
        return bvh_channel_names, bvh_channels

    
    def _create_bvh_skeleton(self, mocap_data, bvh_channels):
        
        joints = mocap_data["skeleton"]["joints"]
        parents  = mocap_data["skeleton"]["parents"]
        children  = mocap_data["skeleton"]["children"]
        offsets = mocap_data["skeleton"]["offsets"]
        
        bvh_skeleton = {}
        
        for jI, joint_name in enumerate(joints):
            
            bvh_joint = {}
            
            # add name of parent joint
            if parents[jI] != -1:
                bvh_joint["parent"]  = joints[parents[jI]]
            else:
                bvh_joint["parent"] = None
            
            # add names of child joints
            bvh_joint["children"] = [ joints[child] for child in children[jI] ]
            
            # add joint offset
            bvh_joint["offsets"] = offsets[jI].tolist()
            
            # add joint channels
            if len(children[jI]) > 0:
                bvh_joint["channels"] = bvh_channels
            else: 
                bvh_joint["channels"] = []
            
                    
            bvh_skeleton[joint_name] = bvh_joint
            
        
        return bvh_skeleton
    
    def _create_bvh_frames(self, mocap_data, bvh_channels):
        
        joints = mocap_data["skeleton"]["joints"]
        children = mocap_data["skeleton"]["children"]
        pos_local = mocap_data["motion"]["pos_local"]
        rot_euler = mocap_data["motion"]["rot_local_euler"]
        rot_sequence = mocap_data["rot_sequence"]
        
        pos_channels = ["Xposition", "Yposition", "Zposition"]
        rot_channels = ["Xrotation", "Yrotation", "Zrotation"]
        
        bvh_frames = {}
        
        for jI, joint_name in enumerate(joints):
            
            if len(children[jI]) == 0:
                continue
            
            for channel_name in bvh_channels:
                
                col_name = joint_name + "_" + channel_name
                channel_values = []
                
                if channel_name in pos_channels:
                    channel_values = pos_local[:, jI, pos_channels.index(channel_name)].tolist()
                    
                elif channel_name in rot_channels:
                    channel_values = rot_euler[:, jI, rot_channels.index(channel_name)].tolist()

                else:
                    continue
                
                bvh_frames[col_name] = channel_values
                
        dataFrame = pandas.DataFrame(bvh_frames)
        
        return dataFrame