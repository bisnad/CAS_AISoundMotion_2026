"""
Motion Loader - loads a mocap recording (bvh, fbx or npz) and returns world-space joint positions
"""

import pathlib
import numpy as np

from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap

config = {
    "file_name": "",
    "topology_file_name": "",     # only needed for npz files
    "fps": 50                     # only used for npz files, other formats carry their own frame rate
    }

class MotionLoader():
    def __init__(self, config):
        self.file_name = config.get("file_name", "")
        self.topology_file_name = config.get("topology_file_name", "")
        self.base_fps = config.get("fps", 50)

        self.skeletons_data = []
        self.max_time = 0.0

        if self.file_name:
            self.load(self.file_name, self.topology_file_name)

    # ---------------- loading ----------------

    def load(self, file_name, topology_file_name=""):
        self.file_name = file_name
        file_suffix = pathlib.Path(file_name).suffix.lower()

        if file_suffix == ".bvh":
            raw_data = self.load_bvh(file_name)
        elif file_suffix == ".fbx":
            raw_data = self.load_fbx(file_name)
        elif file_suffix == ".npz":
            raw_data = self.load_npz(file_name, topology_file_name)
        else:
            raise ValueError("unsupported file type '{}', use .bvh, .fbx or .npz".format(file_suffix))

        self.skeletons_data = raw_data

        if len(self.skeletons_data) > 0:
            self.base_fps = self._frame_rate(self.skeletons_data[0])
            self.max_time = max(self._skeleton_duration(skel) for skel in self.skeletons_data)

    def load_bvh(self, file_name):
        bvh_tools_inst = bvh.BVH_Tools()
        mocap_tools_inst = mocap.Mocap_Tools()

        bvh_data = bvh_tools_inst.load(file_name)
        mocap_data = mocap_tools_inst.bvh_to_mocap(bvh_data)

        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]

        for skel_data in mocap_data:
            skel_data["motion"]["rot_local"] = mocap_tools_inst.euler_to_quat_bvh(skel_data["motion"]["rot_local_euler"], skel_data["rot_sequence"])

        return mocap_data

    def load_fbx(self, file_name):
        fbx_tools_inst = fbx.FBX_Tools()
        mocap_tools_inst = mocap.Mocap_Tools()

        fbx_data = fbx_tools_inst.load(file_name)
        mocap_data = mocap_tools_inst.fbx_to_mocap(fbx_data)

        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]

        for skel_data in mocap_data:
            skel_data["motion"]["rot_local"] = mocap_tools_inst.euler_to_quat(skel_data["motion"]["rot_local_euler"], skel_data["rot_sequence"])

        return mocap_data

    def load_npz(self, file_name, topology_file_name):
        mocap_tools_inst = mocap.Mocap_Tools()
        npz_data, topo_data = npz.NPZ_Tools.load(file_name, topology_file_name)
        mocap_data = mocap_tools_inst.npz_to_mocap(npz_data, topo_data, self.base_fps)

        if not isinstance(mocap_data, list):
            mocap_data = [mocap_data]

        return mocap_data

    # ---------------- information ----------------

    def get_skeleton_count(self):
        return len(self.skeletons_data)

    def get_joint_names(self, skeleton_index=0):
        return list(self.skeletons_data[skeleton_index]["skeleton"]["joints"])

    def get_frame_rate(self):
        return self.base_fps

    def get_duration(self):
        return self.max_time

    @staticmethod
    def _frame_rate(skel_data):
        fps = skel_data.get("frame_rate", 50.0)
        return fps if fps > 0.0 else 50.0

    def _track_length(self, skel_data, joint_index):
        pos_local = skel_data["motion"]["pos_local"]
        return pos_local.shape[0] if not isinstance(pos_local, list) else len(pos_local[joint_index])

    def _skeleton_duration(self, skel_data):
        motion = skel_data["motion"]
        fps = self._frame_rate(skel_data)
        duration = 0.0
        for joint_index, joint_name in enumerate(skel_data["skeleton"]["joints"]):
            if "times" in motion and joint_name in motion["times"]:
                times = motion["times"][joint_name]
                if len(times) > 0:
                    duration = max(duration, float(np.max(times)))
            else:
                duration = max(duration, (self._track_length(skel_data, joint_index) - 1) / fps)
        return duration

    # ---------------- sampling ----------------

    @staticmethod
    def _sample_track(track_times, positions, rotations, t):
        """
        Samples one joint track at the time stamps t: linear interpolation for positions,
        normalised linear interpolation for quaternions (w, x, y, z)
        """
        count = len(track_times)
        if count == 0:
            return np.zeros((len(t), 3)), np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(t), 1))
        if count == 1:
            return np.tile(positions[0], (len(t), 1)), np.tile(rotations[0], (len(t), 1))

        idx = np.clip(np.searchsorted(track_times, t), 1, count - 1)
        t0, t1 = track_times[idx - 1], track_times[idx]
        span = np.where(t1 > t0, t1 - t0, 1.0)
        alpha = np.clip((t - t0) / span, 0.0, 1.0)[:, np.newaxis]

        p0, p1 = positions[idx - 1], positions[idx]
        r0, r1 = rotations[idx - 1], rotations[idx]
        r1 = np.where(np.sum(r0 * r1, axis=1, keepdims=True) < 0.0, -r1, r1)

        pos = p0 + alpha * (p1 - p0)
        rot = r0 + alpha * (r1 - r0)
        rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
        return pos, rot

    def get_positions(self, skeleton_index=0, joint_names=None, fps=None, start_sec=0.0, end_sec=None):
        """
        Returns world-space joint positions [time, joints, 3] in the units of the file, and the frame rate used.
        joint_names: joints (and their order) to return, None = all joints of the skeleton.
        fps: sampling rate, None = the frame rate of the recording.
        """
        mocap_tools_inst = mocap.Mocap_Tools()
        skel_data = self.skeletons_data[skeleton_index]
        motion = skel_data["motion"]

        file_fps = self._frame_rate(skel_data)
        fps = file_fps if fps is None else fps
        duration = self._skeleton_duration(skel_data)
        end_sec = duration if end_sec is None else min(end_sec, duration)
        t = start_sec + np.arange(int(round((end_sec - start_sec) * fps)) + 1) / fps

        file_joint_names = list(skel_data["skeleton"]["joints"])

        pos_tracks, rot_tracks = [], []
        for joint_index, joint_name in enumerate(file_joint_names):
            if isinstance(motion["pos_local"], list):
                positions, rotations = motion["pos_local"][joint_index], motion["rot_local"][joint_index]
            else:
                positions, rotations = motion["pos_local"][:, joint_index, :], motion["rot_local"][:, joint_index, :]

            if "times" in motion and joint_name in motion["times"]:
                track_times = np.asarray(motion["times"][joint_name], dtype=np.float64)
            else:
                track_times = np.arange(len(positions)) / file_fps

            pos, rot = self._sample_track(track_times, np.asarray(positions), np.asarray(rotations), t)
            pos_tracks.append(pos)
            rot_tracks.append(rot)

        pos_local = np.stack(pos_tracks, axis=1)      # [time, joints, 3]
        rot_local = np.stack(rot_tracks, axis=1)      # [time, joints, 4]

        pos_world, _ = mocap_tools_inst.local_to_world(rot_local, pos_local, skel_data["skeleton"])
        pos_world = np.asarray(pos_world)

        if joint_names is not None:
            pos_world = pos_world[:, self._select_joints(file_joint_names, joint_names), :]

        return pos_world, fps

    @staticmethod
    def _select_joints(available_names, wanted_names):
        exact = { name: i for i, name in enumerate(available_names) }
        lower = { name.split(":")[-1].lower(): i for i, name in enumerate(available_names) }    # ignores case and namespaces

        indices, missing = [], []
        for name in wanted_names:
            if name in exact:
                indices.append(exact[name])
            elif name.lower() in lower:
                indices.append(lower[name.lower()])
            else:
                missing.append(name)

        if missing:
            raise ValueError("the recording does not contain these joints: {}\njoints found in the recording ({}):\n{}".format(
                missing, len(available_names), available_names))
        return indices
