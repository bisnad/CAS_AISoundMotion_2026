import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import cv2
import argparse
import mediapipe as mp
import numpy as np
import time
import datetime
import sys
import urllib.request
import math
import threading
from scipy.spatial.transform import Rotation as R
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSizePolicy,
    QSlider, QDoubleSpinBox, QLineEdit, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSlot, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QVector3D
from PyQt5 import QtGui
import pyqtgraph.opengl as gl

import motion_sender

try:
    import fbx
except ImportError:
    print("WARNING: Autodesk 'fbx' module not found. FBX export will not work. Install with 'pip install fbx'.")


# ==============================================================================
# 1. CONSTANTS & TOPOLOGY
# ==============================================================================

DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 9007

VISUAL_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

_base_pose_kinematic = [
    edge for edge in VISUAL_POSE_CONNECTIONS
    if edge not in [(11, 23), (12, 24), (23, 24)]
]
KINEMATIC_POSE_CONNECTIONS = _base_pose_kinematic + [
    (0, 9), (0, 10),
    (0, 11), (0, 12),
    (0, 23), (0, 24)
]

VISUAL_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

KINEMATIC_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def build_parents_from_edges(num_nodes, edges, root_node=0):
    parents = [-1] * num_nodes
    adj = {i: [] for i in range(num_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = {root_node}
    queue = [root_node]

    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                parents[neighbor] = curr
                queue.append(neighbor)
    return parents


def configure_topology(enable_pose, enable_hands):
    if enable_pose and enable_hands:
        num_nodes = 75

        hand_plane_edges = [(15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20)]
        kp_edges = [e for e in KINEMATIC_POSE_CONNECTIONS if e not in hand_plane_edges]

        parents = [-2] * num_nodes

        p_pose = build_parents_from_edges(33, kp_edges, 0)
        for i in range(33):
            if i in [17, 18, 19, 20, 21, 22]:
                parents[i] = -2
            else:
                parents[i] = p_pose[i]

        p_hand = build_parents_from_edges(21, KINEMATIC_HAND_CONNECTIONS, 0)
        for i in range(21):
            parents[33 + i] = 15 if p_hand[i] == -1 else p_hand[i] + 33
            parents[54 + i] = 16 if p_hand[i] == -1 else p_hand[i] + 54

        slices = {'pose': (0, 33), 'Left': (33, 54), 'Right': (54, 75)}
        attachments = {'Left': 15, 'Right': 16}

    elif enable_pose:
        num_nodes = 33
        parents = build_parents_from_edges(33, KINEMATIC_POSE_CONNECTIONS, 0)
        slices = {'pose': (0, 33)}
        attachments = {}

    elif enable_hands:
        num_nodes = 42
        parents = [-2] * num_nodes
        p_hand = build_parents_from_edges(21, KINEMATIC_HAND_CONNECTIONS, 0)
        for i in range(21):
            parents[i] = -1 if p_hand[i] == -1 else p_hand[i]
            parents[21 + i] = -1 if p_hand[i] == -1 else p_hand[i] + 21
        slices = {'Left': (0, 21), 'Right': (21, 42)}
        attachments = {'Left': -1, 'Right': -1}

    return np.array(parents), num_nodes, slices, attachments


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class PoseFilter:
    def __init__(self, num_nodes, min_cutoff=0.5, beta=1.0):
        self.num_nodes = num_nodes
        self.filters = None
        self.min_cutoff = min_cutoff
        self.beta = beta

    def update_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        if self.filters is not None:
            for i in range(self.num_nodes):
                for j in range(3):
                    self.filters[i][j].min_cutoff = min_cutoff
                    self.filters[i][j].beta = beta

    def __call__(self, t, pos_world):
        if self.filters is None:
            self.filters = [
                [OneEuroFilter(t, pos_world[i, j], min_cutoff=self.min_cutoff, beta=self.beta) for j in range(3)]
                for i in range(self.num_nodes)
            ]
            return pos_world

        filtered_pos = np.zeros_like(pos_world)
        for i in range(self.num_nodes):
            for j in range(3):
                filtered_pos[i, j] = self.filters[i][j](t, pos_world[i, j])
        return filtered_pos


# ==============================================================================
# 2. FBX EXPORT & KINEMATICS
# ==============================================================================

def export_fbx(filename, frames_data, parents, fps=30.0):
    if 'fbx' not in globals():
        return

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "MocapScene")

    num_joints = len(parents)
    nodes = []

    try:
        skel_root_enum = fbx.FbxSkeleton.EType.eRoot
        skel_limb_enum = fbx.FbxSkeleton.EType.eLimbNode
    except AttributeError:
        skel_root_enum = fbx.FbxSkeleton.eRoot
        skel_limb_enum = fbx.FbxSkeleton.eLimbNode

    try:
        interp_linear_enum = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    except AttributeError:
        try:
            interp_linear_enum = fbx.FbxAnimCurveDef.eInterpolationLinear
        except AttributeError:
            interp_linear_enum = 4

    for i in range(num_joints):
        if parents[i] == -2:
            nodes.append(None)
            continue

        node = fbx.FbxNode.Create(scene, f"Joint_{i}")
        skeleton = fbx.FbxSkeleton.Create(scene, f"Skel_{i}")
        skeleton.SetSkeletonType(skel_root_enum if parents[i] == -1 else skel_limb_enum)
        node.SetNodeAttribute(skeleton)

        tx, ty, tz = frames_data[0]['pos_rel'][i] * 100.0
        node.LclTranslation.Set(fbx.FbxDouble3(float(tx), float(ty), float(tz)))
        nodes.append(node)

    for i in range(num_joints):
        if parents[i] == -2:
            continue
        if parents[i] != -1 and nodes[parents[i]] is not None:
            nodes[parents[i]].AddChild(nodes[i])
        else:
            scene.GetRootNode().AddChild(nodes[i])

    anim_stack = fbx.FbxAnimStack.Create(scene, "MotionStack")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "MotionLayer")
    anim_stack.AddMember(anim_layer)

    curves = []
    for i in range(num_joints):
        if parents[i] == -2:
            curves.append(None)
            continue
        n = nodes[i]
        curves_i = []
        for prop in [n.LclTranslation, n.LclRotation]:
            for axis in ["X", "Y", "Z"]:
                c = prop.GetCurve(anim_layer, axis, True)
                c.KeyModifyBegin()
                curves_i.append(c)
        curves.append(curves_i)

    fbx_time = fbx.FbxTime()
    for frame_idx, frame in enumerate(frames_data):
        fbx_time.SetSecondDouble(frame_idx / fps)
        pos_rel, quat_rel = frame['pos_rel'], frame['quat_rel']
        for i in range(num_joints):
            if parents[i] == -2:
                continue
            tx, ty, tz = pos_rel[i] * 100.0
            rx, ry, rz = R.from_quat(quat_rel[i]).as_euler('xyz', degrees=True)
            for curve, val in zip(curves[i], [tx, ty, tz, rx, ry, rz]):
                k_idx = curve.KeyAdd(fbx_time)[0]
                curve.KeySet(k_idx, fbx_time, float(val), interp_linear_enum)

    for i in range(num_joints):
        if parents[i] == -2:
            continue
        for c in curves[i]:
            c.KeyModifyEnd()

    unroll_filter = fbx.FbxAnimCurveFilterUnroll()
    for i in range(num_joints):
        if parents[i] == -2:
            continue
        rot_curve_node = nodes[i].LclRotation.GetCurveNode(anim_layer, False)
        if rot_curve_node:
            try:
                unroll_filter.Apply(rot_curve_node)
            except Exception:
                status = fbx.FbxStatus()
                unroll_filter.Apply(rot_curve_node, status)

    exporter = fbx.FbxExporter.Create(manager, "")
    if exporter.Initialize(filename, -1, manager.GetIOSettings()):
        exporter.Export(scene)

    print(f"\n---> Successfully exported FBX to: {filename}")


def extract_landmarks(landmark_list):
    return np.array([[lm.x, -lm.y, -lm.z] for lm in landmark_list])


def get_shortest_quat(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    elif dot < -0.999999:
        ortho = np.cross(np.array([1.0, 0.0, 0.0]), v1)
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(np.array([0.0, 1.0, 0.0]), v1)
        ortho = ortho / np.linalg.norm(ortho)
        return np.array([ortho[0], ortho[1], ortho[2], 0.0])

    cross = np.cross(v1, v2)
    q = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    return q / np.linalg.norm(q)


def compute_kinematics(pos_world, parents, offsets):
    num_nodes = len(pos_world)
    pos_rel = np.zeros_like(pos_world)
    quat_world = np.zeros((num_nodes, 4))
    quat_rel = np.zeros((num_nodes, 4))

    # Initialize quaternions to identity (w=1.0)
    quat_world[:, 3] = 1.0
    quat_rel[:, 3] = 1.0

    children = {i: [] for i in range(num_nodes)}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    # Breadth-first topological sort to ensure parents are computed before children
    topo_order = []
    queue = [i for i, p in enumerate(parents) if p == -1]
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        queue.extend(children[curr])

    for i in topo_order:
        p = parents[i]
        child_list = children[i]

        if p == -1:
            if len(child_list) >= 2:
                v_rest = np.array([offsets[c] for c in child_list])
                v_curr = np.array([pos_world[c] - pos_world[i] for c in child_list])
                valid_idx = (np.linalg.norm(v_rest, axis=1) > 1e-6) & (np.linalg.norm(v_curr, axis=1) > 1e-6)

                if sum(valid_idx) >= 2:
                    try:
                        rot, _ = R.align_vectors(v_curr[valid_idx], v_rest[valid_idx])
                        quat_world[i] = rot.as_quat()
                    except Exception:
                        quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
                elif sum(valid_idx) == 1:
                    idx = np.where(valid_idx)[0][0]
                    quat_world[i] = get_shortest_quat(v_rest[idx], v_curr[idx])
                else:
                    quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])
            elif len(child_list) == 1:
                v_rest = offsets[child_list[0]]
                v_curr = pos_world[child_list[0]] - pos_world[i]
                quat_world[i] = get_shortest_quat(v_rest, v_curr)
            else:
                quat_world[i] = np.array([0.0, 0.0, 0.0, 1.0])

        else:
            if len(child_list) == 0:
                # End-effector (nub) joints just inherit parent's global rotation
                quat_world[i] = quat_world[p]
            else:
                v_rest = np.array([offsets[c] for c in child_list])
                v_curr = np.array([pos_world[c] - pos_world[i] for c in child_list])
                valid_idx = (np.linalg.norm(v_rest, axis=1) > 1e-6) & (np.linalg.norm(v_curr, axis=1) > 1e-6)
                
                if sum(valid_idx) >= 2:
                    try:
                        # Fix: Handle branching joints correctly by aligning to ALL children
                        rot, _ = R.align_vectors(v_curr[valid_idx], v_rest[valid_idx])
                        quat_world[i] = rot.as_quat()
                    except Exception:
                        quat_world[i] = quat_world[p]
                elif sum(valid_idx) == 1:
                    idx = np.where(valid_idx)[0][0]
                    v_rest_single = v_rest[idx]
                    v_curr_single = v_curr[idx]
                    try:
                        v_expected = R.from_quat(quat_world[p]).apply(v_rest_single)
                        q_swing = get_shortest_quat(v_expected, v_curr_single)
                        r_swing = R.from_quat(q_swing)
                        r_parent = R.from_quat(quat_world[p])
                        quat_world[i] = (r_swing * r_parent).as_quat()
                    except Exception:
                        quat_world[i] = quat_world[p]
                else:
                    quat_world[i] = quat_world[p]

    for i in range(num_nodes):
        if parents[i] == -2:
            continue
        p = parents[i]
        if p != -1:
            try:
                # Calculate correct local rotation: R_parent^-1 * R_world
                r_p_inv = R.from_quat(quat_world[p]).inv()
                quat_rel[i] = (r_p_inv * R.from_quat(quat_world[i])).as_quat()
                
                # Fix: Calculate standard mocap pos_rel by un-rotating the world-space delta
                pos_rel[i] = r_p_inv.apply(pos_world[i] - pos_world[p])
            except Exception:
                quat_rel[i] = np.array([0.0, 0.0, 0.0, 1.0])
                pos_rel[i] = offsets[i]
        else:
            quat_rel[i] = quat_world[i]
            pos_rel[i] = pos_world[i]

    return pos_rel, quat_world, quat_rel


def fallback_missing_joints(current_pos, last_pos, start_idx, end_idx, attach_idx=-1):
    if attach_idx != -1:
        offset = current_pos[attach_idx] - last_pos[attach_idx]
        current_pos[start_idx:end_idx] = last_pos[start_idx:end_idx] + offset
    else:
        current_pos[start_idx:end_idx] = last_pos[start_idx:end_idx]


def download_models(model_type):
    pose_model = f"pose_landmarker_{model_type}.task"
    models = {
        pose_model: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{model_type}/float16/latest/{pose_model}",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    }
    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

def get_joint_color(j1, j2, is_2d=False):
    # Left indices: Pose odds 11..31, Hand 33..53
    def is_left(n):
        return (11 <= n <= 31 and n % 2 == 1) or (33 <= n <= 53)
    # Right indices: Pose evens 12..32, Hand 54..74
    def is_right(n):
        return (12 <= n <= 32 and n % 2 == 0) or (54 <= n <= 74)
    def is_face(n):
        return 0 <= n <= 10

    if is_left(j1) and is_left(j2):
        # Orange
        return (255, 128, 0) if is_2d else (1.0, 0.5, 0.0, 1.0)
    elif is_right(j1) and is_right(j2):
        # Cyan
        return (0, 204, 255) if is_2d else (0.0, 0.8, 1.0, 1.0)
    elif is_face(j1) and is_face(j2):
        # Yellow
        return (255, 255, 0) if is_2d else (1.0, 1.0, 0.0, 1.0)
    else:
        # White for cross-body/torso connections
        return (255, 255, 255) if is_2d else (1.0, 1.0, 1.0, 1.0)

def draw_landmarks_custom(image, landmarks, connections, offset=0):
    h, w, _ = image.shape
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            color = get_joint_color(start_idx + offset, end_idx + offset, is_2d=True)
            cv2.line(image, pt1, pt2, color, 2)
            
    # Draw points
    for i, lm in enumerate(landmarks):
        pt = (int(lm.x * w), int(lm.y * h))
        color = get_joint_color(i + offset, i + offset, is_2d=True)
        cv2.circle(image, pt, 3, color, -1)


# ==============================================================================
# 3. Latest-frame reader for live cameras
# ==============================================================================

class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.latest_frame = None
        self.latest_id = 0
        self.latest_ts_ms = 0
        self.read_fail = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                with self.lock:
                    self.read_fail = True
                    self.cond.notify_all()
                break

            ts_ms = int(time.perf_counter() * 1000)

            with self.lock:
                self.latest_frame = frame
                self.latest_id += 1
                self.latest_ts_ms = ts_ms
                self.cond.notify_all()

    def get_latest(self, last_id=None, block=True, timeout=1.0):
        with self.lock:
            if block:
                end_time = time.perf_counter() + timeout
                while self.running and not self.read_fail:
                    if self.latest_frame is not None and (last_id is None or self.latest_id != last_id):
                        break
                    remaining = end_time - time.perf_counter()
                    if remaining <= 0:
                        return False, None, last_id, 0
                    self.cond.wait(timeout=remaining)

            if self.latest_frame is None:
                return False, None, last_id, 0

            if last_id is not None and self.latest_id == last_id:
                return False, None, last_id, 0

            return True, self.latest_frame.copy(), self.latest_id, self.latest_ts_ms

    def stop(self):
        self.running = False
        with self.lock:
            self.cond.notify_all()
        self.thread.join(timeout=1.0)


# ==============================================================================
# 4. GUI & THREADING
# ==============================================================================

class VideoThread(QThread):
    def __init__(self, args, parents, num_nodes, slices, attachments):
        super().__init__()
        self.args = args
        self.parents = parents
        self.num_nodes = num_nodes
        self.slices = slices
        self.attachments = attachments
        self._run_flag = True
        self._new_osc_settings = None

        self.is_recording = False
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.recorded_frames = []

        self.rest_offsets = np.zeros((self.num_nodes, 3))
        self.offsets_captured = np.array([p == -2 for p in self.parents], dtype=bool)

        self.pose_landmarker = None
        self.hand_landmarker = None
        self.tracking_confidence = 0.7
        self.recreate_landmarker = False
        self.pose_filter = None
        self.filter_min_cutoff = 0.1
        self.filter_beta = 5.0

        self.max_visual_fps = max(1.0, float(args.display_fps))
        self.frame_mode = args.frame_mode

        self._frame_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._latest_frame_rgb = None
        self._latest_frame_seq = 0
        self._latest_pos_world = None
        self._processing_fps = 0.0
        self._source_fps = 0.0
        self._skipped_frames = 0

        motion_sender.config["ip"] = DEFAULT_OSC_IP
        motion_sender.config["port"] = DEFAULT_OSC_PORT
        self.osc_sender = motion_sender.OscSender(motion_sender.config)

        self.init_pose_landmarker()

        if self.args.hands:
            options_hand = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.2,
                min_tracking_confidence=0.2
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

    def init_pose_landmarker(self):
        if self.pose_landmarker:
            self.pose_landmarker.close()

        options_pose = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=f'pose_landmarker_{self.args.model}.task'),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=self.tracking_confidence
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
        self.recreate_landmarker = False

    def update_osc_settings(self, ip, port):
        self._new_osc_settings = (ip, port)

    def get_latest_frame(self, last_seq=-1):
        with self._frame_lock:
            if self._latest_frame_rgb is None or self._latest_frame_seq == last_seq:
                return None, last_seq
            return self._latest_frame_rgb.copy(), self._latest_frame_seq

    def get_latest_pos_world(self):
        with self._frame_lock:
            if self._latest_pos_world is None:
                return None
            return self._latest_pos_world.copy()

    def get_stats(self):
        with self._stats_lock:
            return self._processing_fps, self._source_fps, self._skipped_frames

    def _store_latest_frame(self, frame_rgb):
        with self._frame_lock:
            self._latest_frame_rgb = frame_rgb.copy()
            self._latest_frame_seq += 1

    def _set_stats(self, processing_fps=None, source_fps=None, skipped_inc=0):
        with self._stats_lock:
            if processing_fps is not None:
                self._processing_fps = processing_fps
            if source_fps is not None:
                self._source_fps = source_fps
            self._skipped_frames += skipped_inc

    def osc_send_tracked_data(self, tracked_data):
        if not tracked_data:
            return
        pose_data = tracked_data["skeleton_0"]["pose"]
        self.osc_sender.send("/mocap/0/joint/pos_world", pose_data["pos_world"])
        self.osc_sender.send("/mocap/0/joint/pos_local", pose_data["pos_rel"])
        self.osc_sender.send("/mocap/0/joint/rot_world", pose_data["quat_world"])
        self.osc_sender.send("/mocap/0/joint/rot_local", pose_data["quat_rel"])

    def set_tracking_confidence(self, conf):
        self.tracking_confidence = conf
        self.recreate_landmarker = True

    def set_filter_params(self, min_cutoff, beta):
        self.filter_min_cutoff = min_cutoff
        self.filter_beta = beta
        if self.pose_filter is not None:
            self.pose_filter.update_params(min_cutoff, beta)

    def reset_calibration(self):
        self.offsets_captured = np.array([p == -2 for p in self.parents], dtype=bool)
        print("Rest pose calibration reset! Please hold a flat hand/T-Pose.")

    def _safe_timestamp_ms(self, ts_ms, last_timestamp_ms):
        if ts_ms <= last_timestamp_ms:
            return last_timestamp_ms + 1
        return ts_ms

    def _get_file_timestamp_ms(self, cap, last_timestamp_ms):
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        return self._safe_timestamp_ms(ts_ms, last_timestamp_ms)

    def _process_frame(self, frame_rgb, timestamp_ms, last_pos_world):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms) if self.pose_landmarker else None
        hand_res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms) if self.hand_landmarker else None

        current_pos_world = np.copy(last_pos_world)

        left_wrist_pose = None
        right_wrist_pose = None
        updated_hands = set()

        if not self.args.pose and pose_res and pose_res.pose_world_landmarks:
            pose_world_lms = extract_landmarks(pose_res.pose_world_landmarks[0])
            left_wrist_pose = pose_world_lms[15]
            right_wrist_pose = pose_world_lms[16]

        if self.args.pose:
            start, end = self.slices['pose']
            if pose_res and pose_res.pose_world_landmarks:
                current_pos_world[start:end] = extract_landmarks(pose_res.pose_world_landmarks[0])
                if pose_res.pose_landmarks:
                    draw_landmarks_custom(frame_rgb, pose_res.pose_landmarks[0], VISUAL_POSE_CONNECTIONS, offset=0)
            else:
                fallback_missing_joints(current_pos_world, last_pos_world, start, end, -1)

        if self.args.hands:
            if hand_res and hand_res.hand_world_landmarks:
                for i, hand_lms in enumerate(hand_res.hand_world_landmarks):
                    handedness = hand_res.handedness[i][0].category_name
                    if handedness in self.slices:
                        start, end = self.slices[handedness]
                        hand_arr = extract_landmarks(hand_lms)
                        attach_idx = self.attachments[handedness]

                        if attach_idx != -1:
                            hand_arr += (current_pos_world[attach_idx] - hand_arr[0])
                        elif handedness == 'Left' and left_wrist_pose is not None:
                            hand_arr += (left_wrist_pose - hand_arr[0])
                        elif handedness == 'Right' and right_wrist_pose is not None:
                            hand_arr += (right_wrist_pose - hand_arr[0])

                        current_pos_world[start:end] = hand_arr
                        updated_hands.add(handedness)

                        if hand_res.hand_landmarks:
                            draw_landmarks_custom(frame_rgb, hand_res.hand_landmarks[i], VISUAL_HAND_CONNECTIONS, offset=start)

            for handedness in ['Left', 'Right']:
                if handedness in self.slices and handedness not in updated_hands:
                    start, end = self.slices[handedness]
                    if not self.args.pose and ((handedness == 'Left' and left_wrist_pose is not None) or (handedness == 'Right' and right_wrist_pose is not None)):
                        offset = (left_wrist_pose if handedness == 'Left' else right_wrist_pose) - last_pos_world[start]
                        current_pos_world[start:end] = last_pos_world[start:end] + offset
                    else:
                        fallback_missing_joints(current_pos_world, last_pos_world, start, end, self.attachments[handedness])

        if self.pose_filter is None:
            self.pose_filter = PoseFilter(self.num_nodes, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)

        t_sec = timestamp_ms / 1000.0
        filtered_pos_world = self.pose_filter(t_sec, current_pos_world)
        
        with self._frame_lock:
            self._latest_pos_world = filtered_pos_world.copy()

        if not np.all(self.offsets_captured):
            for i in range(self.num_nodes):
                if not self.offsets_captured[i]:
                    active = False
                    for slice_name, (start, end) in self.slices.items():
                        if start <= i < end:
                            if slice_name == 'pose' and pose_res and pose_res.pose_world_landmarks:
                                active = True
                            elif slice_name in ['Left', 'Right'] and slice_name in updated_hands:
                                dist = np.linalg.norm(current_pos_world[start + 9] - current_pos_world[start])
                                if dist > 0.04:
                                    active = True
                            break

                    if active:
                        p = self.parents[i]
                        if p == -1:
                            self.rest_offsets[i] = current_pos_world[i].copy()
                            self.rest_offsets[i, [0, 2]] = 0.0
                            self.offsets_captured[i] = True
                        else:
                            self.rest_offsets[i] = current_pos_world[i] - current_pos_world[p]
                            self.offsets_captured[i] = True

        if np.any(self.offsets_captured):
            pos_rel, quat_world, quat_rel = compute_kinematics(filtered_pos_world, self.parents, self.rest_offsets)
        else:
            pos_rel = np.zeros_like(filtered_pos_world)
            quat_world = np.zeros((self.num_nodes, 4))
            quat_rel = np.zeros((self.num_nodes, 4))
            quat_world[:, 3] = 1.0
            quat_rel[:, 3] = 1.0

        tracked_data = {"skeleton_0": {"pose": {
            "pos_world": current_pos_world,
            "pos_rel": pos_rel,
            "quat_world": quat_world,
            "quat_rel": quat_rel,
            "parents": self.parents
        }}}
        self.osc_send_tracked_data(tracked_data)

        if self.is_counting_down:
            remaining = 5 - int(time.time() - self.countdown_start_time)
            if remaining > 0:
                cv2.putText(frame_rgb, f"RECORDING IN {remaining}...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                self.is_counting_down = False
                self.is_recording = True
                self.recorded_frames = []
        elif self.is_recording:
            self.recorded_frames.append({'pos_rel': pos_rel, 'quat_rel': quat_rel})
            cv2.putText(frame_rgb, "RECORDING", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        return frame_rgb, current_pos_world

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open source: {self.args.source}")
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        is_file = not str(self.args.source).isdigit()
        source_fps = 0.0

        if is_file:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 30.0
        else:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 1e-6:
                source_fps = 0.0

        self._set_stats(source_fps=source_fps)

        frame_period = 1.0 / source_fps if source_fps > 1e-6 else None
        next_frame_time = time.perf_counter() if (is_file and frame_period is not None) else None

        last_timestamp_ms = -1
        last_pos_world = np.zeros((self.num_nodes, 3))

        prev_loop_time = time.perf_counter()
        fps_filter = 0.0
        last_visual_time = 0.0
        visual_period = 1.0 / self.max_visual_fps

        latest_reader = None
        last_camera_frame_id = None

        if (not is_file) and self.frame_mode == "realtime":
            latest_reader = LatestFrameReader(cap)

        try:
            while self._run_flag:
                if self.recreate_landmarker:
                    self.init_pose_landmarker()

                if self._new_osc_settings is not None:
                    new_ip, new_port = self._new_osc_settings
                    print(f"Updating OSC Sender to {new_ip}:{new_port}")
                    motion_sender.config["ip"] = new_ip
                    motion_sender.config["port"] = new_port
                    self.osc_sender = motion_sender.OscSender(motion_sender.config)
                    self._new_osc_settings = None

                if is_file:
                    success, frame = cap.read()
                    if not success:
                        break
                    timestamp_ms = self._get_file_timestamp_ms(cap, last_timestamp_ms)
                else:
                    if self.frame_mode == "all":
                        success, frame = cap.read()
                        if not success:
                            break
                        timestamp_ms = self._safe_timestamp_ms(int(time.perf_counter() * 1000), last_timestamp_ms)
                    else:
                        success, frame, new_id, timestamp_ms = latest_reader.get_latest(
                            last_id=last_camera_frame_id,
                            block=True,
                            timeout=1.0
                        )
                        if not success:
                            if latest_reader.read_fail:
                                break
                            continue

                        if last_camera_frame_id is not None and new_id > last_camera_frame_id + 1:
                            self._set_stats(skipped_inc=(new_id - last_camera_frame_id - 1))

                        last_camera_frame_id = new_id
                        timestamp_ms = self._safe_timestamp_ms(timestamp_ms, last_timestamp_ms)

                last_timestamp_ms = timestamp_ms

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb, last_pos_world = self._process_frame(frame_rgb, timestamp_ms, last_pos_world)

                now = time.perf_counter()
                if now - last_visual_time >= visual_period:
                    self._store_latest_frame(frame_rgb)
                    last_visual_time = now

                loop_end = time.perf_counter()
                inst_fps = 1.0 / max(loop_end - prev_loop_time, 1e-6)
                prev_loop_time = loop_end
                fps_filter = inst_fps if fps_filter == 0.0 else (0.9 * fps_filter + 0.1 * inst_fps)
                self._set_stats(processing_fps=fps_filter)

                if is_file and frame_period is not None:
                    if self.frame_mode == "all":
                        pass
                    else:
                        next_frame_time += frame_period
                        delay = next_frame_time - time.perf_counter()

                        if delay > 0:
                            time.sleep(delay)
                        else:
                            frames_behind = int((-delay) / frame_period)
                            skipped_now = 0

                            for _ in range(frames_behind):
                                if cap.grab():
                                    skipped_now += 1
                                    next_frame_time += frame_period
                                else:
                                    break

                            if skipped_now > 0:
                                self._set_stats(skipped_inc=skipped_now)

                            if next_frame_time < time.perf_counter() - frame_period:
                                next_frame_time = time.perf_counter()

        finally:
            if latest_reader is not None:
                latest_reader.stop()
            cap.release()
            if self.pose_landmarker:
                self.pose_landmarker.close()
            if self.hand_landmarker:
                self.hand_landmarker.close()

    def stop(self):
        self._run_flag = False
        self.wait()

    def start_recording(self):
        self.recorded_frames = []
        if self.args.source.isdigit():
            self.is_counting_down = True
            self.countdown_start_time = time.time()
        else:
            self.is_counting_down = False
            self.is_recording = True

    def stop_recording(self):
        self.is_counting_down = False
        self.is_recording = False
        if len(self.recorded_frames) > 0:
            filename = datetime.datetime.now().strftime("mocap_%Y%m%d_%H%M%S.fbx")
            export_fbx(filename, self.recorded_frames, self.parents)
            self.recorded_frames = []


class MainWindow(QMainWindow):
    def __init__(self, args, parents, num_nodes, slices, attachments):
        super().__init__()
        self.setWindowTitle("MediaPipe Pose Estimation")
        self.resize(1200, 680)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create a horizontal split for Video and 3D
        self.visuals_layout = QHBoxLayout()
        
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: black;")
        self.visuals_layout.addWidget(self.image_label, stretch=1)

        # Setup GL View Widget
        self.gl_widget = gl.GLViewWidget()
        self.visuals_layout.addWidget(self.gl_widget, stretch=1)
        self.gl_widget.setCameraPosition(distance=5, elevation=15, azimuth=45)
        
        # Setup Grid
        grid = gl.GLGridItem(size=QtGui.QVector3D(10, 10, 0))
        grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)

        # Setup Scatter
        self.point_colors = np.zeros((num_nodes, 4))
        for i in range(num_nodes):
            self.point_colors[i] = get_joint_color(i, i, is_2d=False)
            
        self.scatter = gl.GLScatterPlotItem(color=self.point_colors, size=10)
        self.gl_widget.addItem(self.scatter)
        
        # Dynamically create lines based on the parsed hierarchy
        self.lines = []
        for i in range(1, num_nodes):
            p = parents[i]
            if p >= 0:
                edge_color = get_joint_color(i, p, is_2d=False)
                line = gl.GLLinePlotItem(color=edge_color, width=4, antialias=True)
                self.lines.append((i, p, line))
                self.gl_widget.addItem(line)

        # Add visual layout to main vertical layout
        self.layout.addLayout(self.visuals_layout, stretch=4)

        self.filter_layout = QHBoxLayout()
        self.filter_layout.setContentsMargins(10, 5, 10, 0)

        self.filter_layout.addWidget(QLabel("One-Euro Filter - Min Cutoff:"))
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.001, 5.0)
        self.cutoff_spin.setSingleStep(0.05)
        self.cutoff_spin.setDecimals(3)
        self.cutoff_spin.setValue(0.1)
        self.cutoff_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.cutoff_spin)

        self.filter_layout.addWidget(QLabel("Beta (Speed):"))
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 50.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setValue(5.0)
        self.beta_spin.valueChanged.connect(self.on_filter_changed)
        self.filter_layout.addWidget(self.beta_spin)

        self.filter_layout.addStretch(1)
        self.layout.addLayout(self.filter_layout)

        self.osc_layout = QHBoxLayout()
        self.osc_layout.setContentsMargins(10, 5, 10, 0)

        self.osc_layout.addWidget(QLabel("OSC IP:"))
        self.ip_input = QLineEdit(DEFAULT_OSC_IP)
        self.ip_input.setFixedWidth(120)
        self.osc_layout.addWidget(self.ip_input)

        self.osc_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(DEFAULT_OSC_PORT)
        self.port_spin.setFixedWidth(80)
        self.osc_layout.addWidget(self.port_spin)

        self.btn_apply_osc = QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.on_apply_osc)
        self.osc_layout.addWidget(self.btn_apply_osc)

        self.osc_layout.addStretch(1)
        self.layout.addLayout(self.osc_layout)

        self.controls_layout = QHBoxLayout()
        self.controls_layout.setContentsMargins(10, 5, 10, 10)

        self.btn_reset_calib = QPushButton("Reset Calibration")
        self.controls_layout.addWidget(self.btn_reset_calib)

        self.btn_start_record = QPushButton("Start Recording")
        self.btn_start_record.clicked.connect(self.start_recording)
        self.controls_layout.addWidget(self.btn_start_record)

        self.btn_stop_record = QPushButton("Stop Recording")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        self.controls_layout.addWidget(self.btn_stop_record)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.controls_layout.addWidget(self.btn_exit)

        self.conf_label = QLabel("Tracking Conf: 0.70")
        self.controls_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(70)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.controls_layout.addWidget(self.conf_slider)

        self.fps_label = QLabel("Proc FPS: 0.0 | View FPS: 0.0 | Src FPS: 0.0 | Skipped: 0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controls_layout.addWidget(self.fps_label)

        self.layout.addLayout(self.controls_layout)

        self.thread = VideoThread(args, parents, num_nodes, slices, attachments)
        self.btn_reset_calib.clicked.connect(self.thread.reset_calibration)
        self.thread.start()

        self.last_frame_seq = -1
        self.display_fps = 0.0
        self.prev_display_time = None

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(max(1, int(1000 / max(1.0, float(args.ui_fps)))))
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start()

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.thread.stop()
        event.accept()

    @pyqtSlot()
    def refresh_ui(self):
        frame_rgb, seq = self.thread.get_latest_frame(self.last_frame_seq)

        if frame_rgb is not None:
            self.last_frame_seq = seq

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.image_label.setPixmap(pixmap)

            # --- 3D RENDERING ---
            pos_world = self.thread.get_latest_pos_world()
            if pos_world is not None:
                # Map coordinate space. 
                # Note: extract_landmarks already negates Y and Z. 
                # So pos_world Y is UP, and Z is CLOSER to camera.
                # Pyqtgraph: X right, Y forward (depth), Z up
                plot_pts = np.zeros_like(pos_world)
                plot_pts[:, 0] = pos_world[:, 0]
                plot_pts[:, 1] = -pos_world[:, 2] # Push Z 'into' screen 
                plot_pts[:, 2] = pos_world[:, 1]  # Y is already UP

                # Move the character up so feet touch the grid (roughly)
                # MediaPipe root is pelvis, usually around 0.9m above ground
                plot_pts[:, 2] += 0.9 

                # Update scatter
                self.scatter.setData(pos=plot_pts)

                # Update lines based on topology dynamically
                for i, p, line in self.lines:
                    if i < len(plot_pts) and p < len(plot_pts):
                        line.setData(pos=np.vstack([plot_pts[i], plot_pts[p]]))

            now = time.perf_counter()
            if self.prev_display_time is not None:
                inst_display_fps = 1.0 / max(now - self.prev_display_time, 1e-6)
                self.display_fps = (
                    inst_display_fps if self.display_fps == 0.0
                    else 0.9 * self.display_fps + 0.1 * inst_display_fps
                )
            self.prev_display_time = now

        proc_fps, src_fps, skipped = self.thread.get_stats()
        self.fps_label.setText(
            f"Proc FPS: {proc_fps:.1f} | View FPS: {self.display_fps:.1f} | "
            f"Src FPS: {src_fps:.1f} | Skipped: {skipped}"
        )

    def on_conf_changed(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"Tracking Conf: {conf:.2f}")
        self.thread.set_tracking_confidence(conf)

    def on_filter_changed(self):
        cutoff = self.cutoff_spin.value()
        beta = self.beta_spin.value()
        self.thread.set_filter_params(cutoff, beta)

    def on_apply_osc(self):
        ip = self.ip_input.text().strip()
        port = self.port_spin.value()
        self.thread.update_osc_settings(ip, port)

    def start_recording(self):
        self.thread.start_recording()
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)

    def stop_recording(self):
        self.thread.stop_recording()
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--hands", action="store_true")
    parser.add_argument("--model", type=str, choices=["lite", "full", "heavy"], default="heavy",
                        help="Pose model type to use")
    parser.add_argument("--display_fps", type=float, default=20.0,
                        help="Maximum rate for generating display frames")
    parser.add_argument("--ui_fps", type=float, default=30.0,
                        help="GUI refresh rate used to pull and display the latest frame")
    parser.add_argument("--frame_mode", type=str, choices=["all", "realtime"], default="realtime",
                        help="all = process every frame in order; realtime = drop stale frames to stay current")
    args = parser.parse_args()

    if not (args.pose or args.hands):
        args.pose = args.hands = True

    download_models(args.model)
    PARENTS, NUM_NODES, SLICES, ATTACHMENTS = configure_topology(args.pose, args.hands)

    app = QApplication(sys.argv)
    window = MainWindow(args, PARENTS, NUM_NODES, SLICES, ATTACHMENTS)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()