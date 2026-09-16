import os
# Force pyqtgraph to use PyQt5 to prevent binding conflicts
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import sys
import torch
import cv2
import numpy as np
import argparse
import math
from collections import deque
import datetime
import time
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSlot
import pyqtgraph.opengl as gl

# Ultralytics
from ultralytics import YOLO

# MotionBERT
sys.path.append('./MotionBERT')
from lib.model.DSTformer import DSTformer

# SciPy Rotations for FBX/Kinematics
from scipy.spatial.transform import Rotation as R

# FBX
try:
    import fbx
except ImportError:
    print("WARNING: Autodesk fbx module not found. FBX export will not work. Install with 'pip install fbx'.")

# OSC
try:
    from pythonosc import udp_client
except ImportError:
    print("WARNING: python-osc not found. OSC will not work. Install with 'pip install python-osc'.")


# ==========================================
# Default T-Pose for Initial Inactive Tracks
# ==========================================
T_POSE_WORLD = np.array([
    [0.0, 0.0, 0.0],       # 0: Pelvis
    [-0.15, 0.0, 0.0],     # 1: R Hip 
    [-0.15, -0.4, 0.0],    # 2: R Knee
    [-0.15, -0.8, 0.0],    # 3: R Ankle
    [0.15, 0.0, 0.0],      # 4: L Hip
    [0.15, -0.4, 0.0],     # 5: L Knee
    [0.15, -0.8, 0.0],     # 6: L Ankle
    [0.0, 0.2, 0.0],       # 7: Spine
    [0.0, 0.4, 0.0],       # 8: Neck/Thorax
    [0.0, 0.5, 0.0],       # 9: Nose/Jaw
    [0.0, 0.6, 0.0],       # 10: Head
    [0.2, 0.4, 0.0],       # 11: L Shoulder
    [0.45, 0.4, 0.0],      # 12: L Elbow
    [0.7, 0.4, 0.0],       # 13: L Wrist
    [-0.2, 0.4, 0.0],      # 14: R Shoulder
    [-0.45, 0.4, 0.0],     # 15: R Elbow
    [-0.7, 0.4, 0.0],      # 16: R Wrist
], dtype=np.float32)

# ==========================================
# LatestFrameReader (For Webcam Realtime Mode)
# ==========================================
class LatestFrameReader(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame = None
        self.read_fail = False
        self.running = True
        self.start()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                with self.lock:
                    self.read_fail = True
                break
            with self.lock:
                self.latest_frame = frame

    def get_latest(self):
        with self.lock:
            if self.read_fail: return False, None
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                self.latest_frame = None
                return True, frame
            return None, None

    def stop(self):
        self.running = False
        self.join()

def plot_skeleton_kpts(im, kpts, conf_thresh=0.5):
    skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12),
        (7, 13), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3),
        (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)
    ]
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), 
        (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), 
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), 
        (255, 0, 255), (255, 0, 170), (255, 0, 85)
    ]
    for i, (p1, p2) in enumerate(skeleton):
        x1, y1, conf1 = kpts[p1-1]
        x2, y2, conf2 = kpts[p2-1]
        if conf1 > conf_thresh and conf2 > conf_thresh:
            cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), colors[i % len(colors)], thickness=2)
            cv2.circle(im, (int(x1), int(y1)), 4, colors[i % len(colors)], thickness=-1)
            cv2.circle(im, (int(x2), int(y2)), 4, colors[i % len(colors)], thickness=-1)

# ==========================================
# Persistent Tracker (FIXED)
# ==========================================
class SimpleTracker:
    def __init__(self, max_lost=15, dist_thresh=150, max_tracks=-1):
        self.next_id = 1
        self.tracks = {}  
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh
        self.max_tracks = max_tracks
        
        # Pre-allocate specific slots if max_tracks is provided
        if self.max_tracks > 0:
            for i in range(self.max_tracks):
                self.tracks[i] = {'centroid': None, 'lost': 0}

    def update(self, centroids):
        active_matches = []
        used_tracks = set()
        used_centroids = set()
        
        # 1. Match active tracks first
        for tid, track in self.tracks.items():
            if track['centroid'] is None: continue
            best_c_idx, best_dist = None, self.dist_thresh
            for c_idx, (cx, cy) in enumerate(centroids):
                if c_idx in used_centroids: continue
                dist = math.hypot(track['centroid'][0] - cx, track['centroid'][1] - cy)
                if dist < best_dist:
                    best_dist, best_c_idx = dist, c_idx
            if best_c_idx is not None:
                self.tracks[tid]['centroid'] = centroids[best_c_idx]
                self.tracks[tid]['lost'] = 0
                active_matches.append((tid, best_c_idx))
                used_tracks.add(tid)
                used_centroids.add(best_c_idx)

        # 2. Assign remaining centroids to empty slots or new tracks
        for c_idx, (cx, cy) in enumerate(centroids):
            if c_idx in used_centroids: continue
            
            spawn_tid = None
            if self.max_tracks > 0:
                for tid in range(self.max_tracks):
                    if self.tracks[tid]['centroid'] is None:
                        spawn_tid = tid
                        break
            else:
                spawn_tid = self.next_id
                self.next_id += 1
                
            if spawn_tid is not None:
                self.tracks[spawn_tid] = {'centroid': (cx, cy), 'lost': 0}
                active_matches.append((spawn_tid, c_idx))
                used_tracks.add(spawn_tid)
                used_centroids.add(c_idx)

        # 3. Handle lost tracks
        lost_ids = []
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                if self.tracks[tid].get('centroid') is not None:
                    self.tracks[tid]['lost'] += 1
                    if self.tracks[tid]['lost'] > self.max_lost:
                        if self.max_tracks > 0:
                            # Just clear the centroid to make slot available, but keep the ID alive
                            self.tracks[tid]['centroid'] = None
                            self.tracks[tid]['lost'] = 0
                            lost_ids.append(tid)
                        else:
                            # Fully delete dynamic track
                            lost_ids.append(tid)
                            del self.tracks[tid]

        return active_matches, lost_ids

# ==========================================
# OSC Sender Class
# ==========================================
class OscSender:
    def __init__(self, ip="127.0.0.1", port=9007):
        self.ip = ip
        self.port = port
        try:
            self.client = udp_client.SimpleUDPClient(self.ip, self.port)
            self.active = True
            print(f"OSC Client Initialized: {self.ip}:{self.port}")
        except Exception as e:
            print(f"OSC Error: {e}")
            self.active = False

    def send_mocap(self, track_id, address_suffix, data):
        if not self.active: return
        flat_data = data.flatten().tolist()
        address = f"/mocap/{track_id}/joint/{address_suffix}"
        self.client.send_message(address, flat_data)


# ==========================================
# Euler Unrolling & FBX Exporter
# ==========================================
def unroll_euler(prev_euler, curr_euler):
    """ Prevents 360-degree rotation flips in animation curves """
    diff = curr_euler - prev_euler
    diff = (diff + 180.0) % 360.0 - 180.0
    return prev_euler + diff


# Standard FBX time modes and their fps, used to pick the closest matching
# preset (falls back to eCustom when nothing matches closely enough). Setting
# only the per-key FbxTime values is NOT enough for importers like
# MotionBuilder to report the correct frame rate — the scene's global time
# mode / custom frame rate must also be set explicitly.
_FBX_STANDARD_FPS_MODES = [
    (120.0, "eFrames120"),
    (100.0, "eFrames100"),
    (60.0, "eFrames60"),
    (59.94, "eFrames59dot94"),
    (50.0, "eFrames50"),
    (48.0, "eFrames48"),
    (30.0, "eFrames30"),
    (29.97, "eNTSCDropFrame"),
    (25.0, "ePAL"),
    (24.0, "eFrames24"),
    (96.0, "eFrames96"),
    (72.0, "eFrames72"),
]


def _resolve_fbx_time_mode(fps):
    """
    Map an arbitrary fps value to the closest standard FbxTime::EMode, or to
    eCustom if nothing matches within tolerance. Returns (time_mode_enum, is_custom).
    """
    def _get_enum(name):
        try:
            return getattr(fbx.FbxTime.EMode, name)
        except AttributeError:
            return getattr(fbx.FbxTime, name)

    for standard_fps, name in _FBX_STANDARD_FPS_MODES:
        if abs(fps - standard_fps) < 0.01:
            try:
                return _get_enum(name), False
            except AttributeError:
                break

    try:
        return _get_enum("eCustom"), True
    except AttributeError:
        return _get_enum("eDefaultMode"), False


def export_fbx(filename, recorded_tracks, parents, fps=30.0):
    if 'fbx' not in globals(): return
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "MocapScene")
    num_joints = len(parents)

    # --- Set the scene's global frame rate so importers (e.g. MotionBuilder)
    # report the correct fps instead of defaulting to 30. ---
    time_mode, is_custom = _resolve_fbx_time_mode(fps)
    try:
        if is_custom:
            fbx.FbxTime.SetGlobalTimeMode(time_mode, float(fps))
        else:
            fbx.FbxTime.SetGlobalTimeMode(time_mode)
    except Exception as e:
        print(f"WARNING: Could not set FbxTime global time mode: {e}")

    global_settings = scene.GetGlobalSettings()
    try:
        global_settings.SetTimeMode(time_mode)
        if is_custom:
            global_settings.SetCustomFrameRate(float(fps))
    except Exception as e:
        print(f"WARNING: Could not set scene global settings time mode: {e}")

    try:
        skel_root_enum = fbx.FbxSkeleton.EType.eRoot
        skel_limb_enum = fbx.FbxSkeleton.EType.eLimbNode
        interp_linear_enum = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear
    except AttributeError:
        skel_root_enum = fbx.FbxSkeleton.eRoot
        skel_limb_enum = fbx.FbxSkeleton.eLimbNode
        interp_linear_enum = 4

    anim_stack = fbx.FbxAnimStack.Create(scene, "MotionStack")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "MotionLayer")
    anim_stack.AddMember(anim_layer)
    fbx_time = fbx.FbxTime()

    max_frame_idx = 0

    for tid, frames in recorded_tracks.items():
        if not frames: continue

        nodes = []
        for i in range(num_joints):
            node = fbx.FbxNode.Create(scene, f"T{tid}_Joint_{i}")
            skeleton = fbx.FbxSkeleton.Create(scene, f"T{tid}_Skel_{i}")
            skeleton.SetSkeletonType(skel_root_enum if parents[i] == -1 else skel_limb_enum)
            node.SetNodeAttribute(skeleton)
            
            # Initial Setup Transform
            tx, ty, tz = frames[0]['pos_rel'][i] * 100.0 
            node.LclTranslation.Set(fbx.FbxDouble3(float(tx), float(ty), float(tz)))
            nodes.append(node)

        for i in range(num_joints):
            if parents[i] != -1 and nodes[parents[i]] is not None:
                nodes[parents[i]].AddChild(nodes[i])
            elif parents[i] == -1:
                scene.GetRootNode().AddChild(nodes[i])

        curves = []
        for i in range(num_joints):
            n = nodes[i]
            curves_i = []
            for prop in [n.LclTranslation, n.LclRotation]:
                for axis in ['X', 'Y', 'Z']:
                    c = prop.GetCurve(anim_layer, axis, True)
                    c.KeyModifyBegin()
                    curves_i.append(c)
            curves.append(curves_i)

        prev_eulers = np.zeros((num_joints, 3))
        
        # Write Data
        for frame_idx_loop, frame in enumerate(frames):
            frame_idx = frame.get('frame_idx', 0)
            max_frame_idx = max(max_frame_idx, frame_idx)
            fbx_time.SetSecondDouble(frame_idx / fps)
            pos_rel, quat_rel = frame['pos_rel'], frame['quat_rel']
            
            for i in range(num_joints):
                tx, ty, tz = pos_rel[i] * 100.0
                rx, ry, rz = R.from_quat(quat_rel[i]).as_euler('xyz', degrees=True)
                
                if frame_idx_loop > 0:
                    rx, ry, rz = unroll_euler(prev_eulers[i], np.array([rx, ry, rz]))
                
                prev_eulers[i] = [rx, ry, rz]
                
                for curve, val in zip(curves[i], [tx, ty, tz, rx, ry, rz]):
                    kidx = curve.KeyAdd(fbx_time)[0]
                    curve.KeySet(kidx, fbx_time, float(val), interp_linear_enum)

        for i in range(num_joints):
            for c in curves[i]: c.KeyModifyEnd()

    # Set the timeline (start/stop) span to match the recorded animation, at
    # the resolved frame rate, so players reflect the correct duration/fps.
    try:
        time_span = fbx.FbxTimeSpan()
        start_time = fbx.FbxTime()
        start_time.SetSecondDouble(0.0)
        stop_time = fbx.FbxTime()
        stop_time.SetSecondDouble(max_frame_idx / fps)
        time_span.SetStart(start_time)
        time_span.SetStop(stop_time)
        global_settings.SetTimelineDefaultTimeSpan(time_span)
    except Exception as e:
        print(f"WARNING: Could not set FBX timeline span: {e}")

    exporter = fbx.FbxExporter.Create(manager, "")
    if exporter.Initialize(filename, -1, manager.GetIOSettings()):
        exporter.Export(scene)
        print(f"--- Successfully exported FBX to {filename} (fps={fps:.3f}, custom_mode={is_custom})")
        
    manager.Destroy()

# ==========================================
# OneEuro Filter & Kinematics Preprocessing
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = x0, dx0, t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

class PoseFilter:
    def __init__(self, num_nodes, min_cutoff=0.5, beta=1.0):
        self.num_nodes = num_nodes
        self.filters = None
        self.min_cutoff = min_cutoff
        self.beta = beta

    def __call__(self, t, pos_world):
        if self.filters is None:
            self.filters = [[OneEuroFilter(t, pos_world[i, j], min_cutoff=self.min_cutoff, beta=self.beta) 
                             for j in range(3)] for i in range(self.num_nodes)]
            return pos_world
        filtered_pos = np.zeros_like(pos_world)
        for i in range(self.num_nodes):
            for j in range(3):
                filtered_pos[i, j] = self.filters[i][j](t, pos_world[i, j])
        return filtered_pos

def coco2h36m(coco_kpts):
    h36m_kpts = np.zeros((17, 3), dtype=np.float32)
    h36m_kpts[0] = (coco_kpts[11] + coco_kpts[12]) / 2.0
    h36m_kpts[1], h36m_kpts[2], h36m_kpts[3] = coco_kpts[12], coco_kpts[14], coco_kpts[16]
    h36m_kpts[4], h36m_kpts[5], h36m_kpts[6] = coco_kpts[11], coco_kpts[13], coco_kpts[15]
    neck = (coco_kpts[5] + coco_kpts[6]) / 2.0
    h36m_kpts[7] = (h36m_kpts[0] + neck) / 2.0
    h36m_kpts[8] = neck
    h36m_kpts[9] = coco_kpts[0]
    head_center = (coco_kpts[3] + coco_kpts[4]) / 2.0 
    h36m_kpts[10] = head_center + (head_center - neck) * 0.5 
    h36m_kpts[11], h36m_kpts[12], h36m_kpts[13] = coco_kpts[5], coco_kpts[7], coco_kpts[9]
    h36m_kpts[14], h36m_kpts[15], h36m_kpts[16] = coco_kpts[6], coco_kpts[8], coco_kpts[10]
    return h36m_kpts

def crop_scale_sequence(kpts_seq):
    seq_norm = kpts_seq.copy()
    min_x, max_x = np.min(seq_norm[:, :, 0]), np.max(seq_norm[:, :, 0])
    min_y, max_y = np.min(seq_norm[:, :, 1]), np.max(seq_norm[:, :, 1])
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    scale = max(max_x - min_x, max_y - min_y) / 2.0
    if scale < 1e-4: scale = 1.0 
    seq_norm[:, :, 0] = (seq_norm[:, :, 0] - center_x) / scale
    seq_norm[:, :, 1] = (seq_norm[:, :, 1] - center_y) / scale
    return seq_norm, center_x, center_y, scale

def get_h36m_parents():
    return [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

def get_shortest_quat(v1, v2):
    v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
    if v1_norm < 1e-6 or v2_norm < 1e-6: return np.array([0.0, 0.0, 0.0, 1.0])
    v1, v2 = v1 / v1_norm, v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if dot > 0.999999: return np.array([0.0, 0.0, 0.0, 1.0])
    elif dot < -0.999999:
        ortho = np.cross(np.array([1.0, 0.0, 0.0]), v1)
        if np.linalg.norm(ortho) < 1e-6: ortho = np.cross(np.array([0.0, 1.0, 0.0]), v1)
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


# ==========================================
# VideoThread — owns capture, tracking, inference, recording, playback
# ==========================================
class VideoThread(QThread):
    def __init__(self, args, yolo_model, mbert_model, device):
        super().__init__()
        self.args = args
        self.device = device
        self.yolo_model = yolo_model
        self.mbert_model = mbert_model
        self.max_tracks = int(self.args.max_tracks)

        self._run_flag = True

        self.window_size = 243
        self.tracker = SimpleTracker(max_tracks=self.max_tracks)
        self.pose_queues = {}
        self.filters = {}
        self.track_states = {}

        self.is_recording = False
        self.recorded_frames = {}
        self.recording_frame_idx = 0

        self.parents = get_h36m_parents()
        self.rest_offsets = np.zeros((17, 3))
        self.offsets_captured = False

        self.filter_min_cutoff = 0.1
        self.filter_beta = 5.0
        self.yolo_conf = 0.5

        # --- Export FPS mode: 'auto' derives fps from frame_mode; 'manual'
        # uses an explicit user-specified value. ---
        self.export_fps_mode = 'auto'
        self.export_fps_manual_value = 30.0

        self.is_webcam = str(self.args.source).isdigit()

        # --- Thread-safety primitives ---
        self._frame_lock = threading.Lock()
        self._render_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._playback_lock = threading.Lock()

        self._latest_frame_rgb = None
        self._latest_frame_seq = 0
        self._latest_render_state = {}  # {tid: {pose, pos_rel, quat_world, quat_rel}}
        self._removed_tids = set()      # tids removed since last GUI poll (dynamic mode)

        self.proc_fps = 0.0
        self.skipped_count = 0
        self.src_fps = 0.0

        # --- Playback control (file sources only) ---
        self._play_event = threading.Event()
        self._play_event.set()
        self._is_paused = False
        self._seek_request_sec = None
        self._seek_wake_event = threading.Event()
        self._video_duration_sec = 0.0
        self._current_pos_sec = 0.0

        self.osc = OscSender("127.0.0.1", 9007)

        # Pre-populate slots with default T-poses spread apart horizontally
        if self.max_tracks > 0:
            default_offsets = np.zeros((17, 3))
            for j in range(17):
                p = self.parents[j]
                if p != -1: default_offsets[j] = T_POSE_WORLD[j] - T_POSE_WORLD[p]

            pos_rel, quat_world, quat_rel = compute_kinematics(T_POSE_WORLD, self.parents, default_offsets)

            for tid in range(self.max_tracks):
                tid_offset = (tid - (self.max_tracks - 1) / 2.0) * 1.0
                pose_t = T_POSE_WORLD.copy()
                pose_t[:, 0] += tid_offset

                pos_rel_t = pos_rel.copy()
                pos_rel_t[0][0] += tid_offset

                self.track_states[tid] = {
                    'pose': pose_t,
                    'pos_rel': pos_rel_t,
                    'quat_world': quat_world.copy(),
                    'quat_rel': quat_rel.copy()
                }
            self._latest_render_state = {tid: dict(s) for tid, s in self.track_states.items()}

    # ------------------------------------------------------------------
    # Thread-safe getters used by the GUI thread
    # ------------------------------------------------------------------

    def get_latest_frame(self, last_seq=-1):
        with self._frame_lock:
            if self._latest_frame_rgb is None or self._latest_frame_seq == last_seq:
                return None, last_seq
            return self._latest_frame_rgb.copy(), self._latest_frame_seq

    def get_latest_render_state(self):
        with self._render_lock:
            removed = self._removed_tids
            self._removed_tids = set()
            return {tid: dict(s) for tid, s in self._latest_render_state.items()}, removed

    def get_stats(self):
        with self._stats_lock:
            return self.proc_fps, self.src_fps, self.skipped_count

    def get_playback_state(self):
        with self._playback_lock:
            return self._is_paused, self._current_pos_sec, self._video_duration_sec

    def is_playback_controllable(self):
        return not self.is_webcam

    # ------------------------------------------------------------------
    # Control methods (called from the GUI thread)
    # ------------------------------------------------------------------

    def pause_playback(self):
        if self.is_webcam: return
        self._is_paused = True
        self._play_event.clear()

    def resume_playback(self):
        if self.is_webcam: return
        self._is_paused = False
        self._play_event.set()

    def seek_to_seconds(self, seconds):
        if self.is_webcam: return
        with self._playback_lock:
            self._seek_request_sec = max(0.0, float(seconds))
        # Wake a paused loop WITHOUT touching _play_event, so pause state
        # is preserved across seeks.
        self._seek_wake_event.set()

    def set_filter_params(self, min_cutoff, beta):
        self.filter_min_cutoff = min_cutoff
        self.filter_beta = beta
        for pfilter in self.filters.values():
            pfilter.min_cutoff = min_cutoff
            pfilter.beta = beta
            if pfilter.filters is not None:
                for i in range(pfilter.num_nodes):
                    for j in range(3):
                        pfilter.filters[i][j].min_cutoff = min_cutoff
                        pfilter.filters[i][j].beta = beta

    def set_yolo_conf(self, conf):
        self.yolo_conf = conf

    def update_osc_settings(self, ip, port):
        self.osc = OscSender(ip, port)

    def set_export_fps_mode(self, mode, manual_value=None):
        self.export_fps_mode = mode
        if manual_value is not None:
            self.export_fps_manual_value = manual_value

    def _get_auto_export_fps(self):
        if self.args.frame_mode == 'all':
            if self.src_fps and self.src_fps > 1e-3:
                return self.src_fps
            if self.proc_fps and self.proc_fps > 1e-3:
                return self.proc_fps
            return 30.0
        if self.proc_fps and self.proc_fps > 1e-3:
            return self.proc_fps
        if self.src_fps and self.src_fps > 1e-3:
            return self.src_fps
        return 30.0

    def get_export_fps(self):
        if self.export_fps_mode == 'manual':
            return max(1.0, float(self.export_fps_manual_value))
        return self._get_auto_export_fps()

    def start_recording(self):
        self.is_recording = True
        self.recording_frame_idx = 0
        self.recorded_frames = {tid: [] for tid in self.track_states.keys()}
        print("Recording Started...")

    def stop_recording(self):
        self.is_recording = False
        print("Recording Stopped. Exporting FBX...")

        has_data = any(len(frames) > 0 for frames in self.recorded_frames.values())
        if has_data:
            export_fps = self.get_export_fps()
            print(f"Exporting FBX with fps={export_fps:.3f} (mode={self.export_fps_mode}, frame_mode={self.args.frame_mode})")
            filename = datetime.datetime.now().strftime("mocap_all_%Y%m%d_%H%M%S.fbx")
            export_fbx(filename, self.recorded_frames, self.parents, fps=export_fps)

        self.recorded_frames = {}

    def stop(self):
        self._run_flag = False
        self._play_event.set()
        self._seek_wake_event.set()
        self.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_latest_frame(self, frame_rgb):
        with self._frame_lock:
            self._latest_frame_rgb = frame_rgb.copy()
            self._latest_frame_seq += 1

    def _store_render_state(self, removed_tids=None):
        with self._render_lock:
            self._latest_render_state = {tid: dict(s) for tid, s in self.track_states.items()}
            if removed_tids:
                self._removed_tids.update(removed_tids)

    def _set_stats(self, proc_fps=None, skipped_inc=0):
        with self._stats_lock:
            if proc_fps is not None:
                self.proc_fps = proc_fps
            self.skipped_count += skipped_inc

    def _set_playback_position(self, pos_sec, duration_sec=None):
        with self._playback_lock:
            self._current_pos_sec = pos_sec
            if duration_sec is not None:
                self._video_duration_sec = duration_sec

    def _consume_seek_request(self):
        with self._playback_lock:
            req = self._seek_request_sec
            self._seek_request_sec = None
            return req

    def remove_track(self, tid):
        if tid in self.pose_queues: del self.pose_queues[tid]

        if self.max_tracks == -1:
            if tid in self.filters: del self.filters[tid]
            if tid in self.track_states: del self.track_states[tid]
            return tid
        return None

    def _apply_seek(self, cap, target_sec):
        """
        Move the capture's read position and reset temporal state that
        depends on frame continuity. Unlike MediaPipe's VIDEO-mode
        landmarkers, YOLO+MotionBERT have no monotonic-timestamp
        requirement, but MotionBERT's sliding window is invalidated by a
        jump, so the pose queues must be cleared.
        """
        cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
        self.pose_queues.clear()
        self.offsets_captured = False

    def _process_frame(self, frame):
        """Runs YOLO + tracker + MotionBERT + kinematics + OSC + recording
        for a single frame. Returns the annotated BGR frame for preview."""
        orig_img = frame.copy()

        if self.is_recording:
            self.recording_frame_idx += 1

        results = self.yolo_model(orig_img, verbose=False, device=self.device)
        current_centroids, current_kpts = [], []

        if len(results[0]) > 0 and results[0].keypoints is not None:
            all_kpts = results[0].keypoints.data.cpu().numpy()
            for kpts in all_kpts:
                valid_pts = kpts[kpts[:, 2] > self.yolo_conf]
                if len(valid_pts) == 0: continue
                cx, cy = np.mean(valid_pts[:, 0]), np.mean(valid_pts[:, 1])
                current_centroids.append((cx, cy))
                current_kpts.append(kpts)

        active_matches, lost_ids = self.tracker.update(current_centroids)
        removed_tids = set()
        for tid in lost_ids:
            removed = self.remove_track(tid)
            if removed is not None:
                removed_tids.add(removed)

        active_ids = []
        for tid, c_idx in active_matches:
            active_ids.append(tid)
            kpts_2d_h36m = coco2h36m(current_kpts[c_idx])

            if tid not in self.pose_queues:
                self.pose_queues[tid] = deque(maxlen=self.window_size)
                self.filters[tid] = PoseFilter(17, min_cutoff=self.filter_min_cutoff, beta=self.filter_beta)
                if self.is_recording and tid not in self.recorded_frames:
                    self.recorded_frames[tid] = []

            self.pose_queues[tid].append(kpts_2d_h36m)

            cx, cy = current_centroids[c_idx]
            colors = [(1,0.2,0.2,1), (0.2,1,0.2,1), (0.2,0.4,1,1), (1,1,0.2,1), (1,0.2,1,1), (0.2,1,1,1)]
            color_bgr = tuple(int(c*255) for c in colors[tid % len(colors)][:3][::-1])
            cv2.putText(orig_img, f"ID: {tid}", (int(cx), int(cy)-50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 3)
            plot_skeleton_kpts(orig_img, current_kpts[c_idx], self.yolo_conf)

        for tid in list(self.pose_queues.keys()):
            if tid not in active_ids and len(self.pose_queues[tid]) > 0:
                self.pose_queues[tid].append(self.pose_queues[tid][-1])

        if len(self.pose_queues) > 0:
            batch_seqs, batch_ids, batch_offsets = [], [], []
            for tid, seq_queue in self.pose_queues.items():
                if len(seq_queue) == 0: continue
                seq = list(seq_queue)
                while len(seq) < self.window_size: seq.insert(0, seq[0])
                seq_norm, cx, cy, scale = crop_scale_sequence(np.array(seq, dtype=np.float32))
                batch_seqs.append(seq_norm)
                batch_ids.append(tid)
                batch_offsets.append((cx, cy, scale))

            seq_tensor = torch.tensor(np.array(batch_seqs)).to(self.device)
            with torch.no_grad():
                pred_3d = self.mbert_model(seq_tensor)

            t_sec = time.perf_counter()

            for i, tid in enumerate(batch_ids):
                pose = pred_3d[i, -1].cpu().numpy()
                pose = pose - pose[0].copy()
                pose = self.filters[tid](t_sec, pose)

                if not self.offsets_captured:
                    for j in range(17):
                        p = self.parents[j]
                        if p != -1: self.rest_offsets[j] = pose[j] - pose[p]
                    self.offsets_captured = True

                pos_rel, quat_world, quat_rel = compute_kinematics(pose, self.parents, self.rest_offsets)

                cx, cy, scale = batch_offsets[i]
                vid_h = orig_img.shape[0]
                vid_w = orig_img.shape[1]
                dz = (1.7 * vid_h) / (2.0 * scale) if scale > 0.1 else 0.0
                dx = (cx - vid_w / 2.0) * (dz / vid_h)
                dy = (cy - vid_h / 2.0) * (dz / vid_h)

                pos_rel_out = pos_rel.copy()
                pos_rel_out[0][0] += dx
                pos_rel_out[0][1] += dy
                pos_rel_out[0][2] += dz

                pose_out = pose.copy()
                pose_out[:, 0] += dx
                pose_out[:, 1] += dy
                pose_out[:, 2] += dz

                self.track_states[tid] = {
                    'pose': pose_out,
                    'pos_rel': pos_rel_out,
                    'quat_world': quat_world,
                    'quat_rel': quat_rel
                }

        for tid in list(self.track_states.keys()):
            state = self.track_states[tid]

            self.osc.send_mocap(tid, "pos_world", state['pose'])
            self.osc.send_mocap(tid, "pos_local", state['pos_rel'])
            self.osc.send_mocap(tid, "rot_world", state['quat_world'])
            self.osc.send_mocap(tid, "rot_local", state['quat_rel'])

            if self.is_recording and tid in self.recorded_frames:
                self.recorded_frames[tid].append({
                    'pos_rel': state['pos_rel'],
                    'quat_rel': state['quat_rel'],
                    'frame_idx': self.recording_frame_idx
                })

        if self.is_recording:
            cv2.putText(orig_img, f"RECORDING ({self.recording_frame_idx})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        self._store_render_state(removed_tids=removed_tids)

        return orig_img

    def run(self):
        source = int(self.args.source) if self.args.source.isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Failed to open source: {self.args.source}")
            return

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps < 1e-6: src_fps = 30.0
        self.src_fps = src_fps
        frame_period = 1.0 / src_fps

        is_file = not self.is_webcam
        if is_file:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_sec = frame_count / src_fps if (frame_count and frame_count > 0 and src_fps > 1e-6) else 0.0
            self._set_playback_position(0.0, duration_sec=duration_sec)

        reader = None
        if self.is_webcam and self.args.frame_mode == 'realtime':
            reader = LatestFrameReader(cap)

        next_frame_time = None
        prev_loop_time = time.perf_counter()
        fps_filter = 0.0

        try:
            while self._run_flag:
                if is_file:
                    seek_target = self._consume_seek_request()
                    if seek_target is not None:
                        self._apply_seek(cap, seek_target)
                        next_frame_time = None
                        prev_loop_time = time.perf_counter()

                    while self._run_flag and not self._play_event.is_set():
                        self._seek_wake_event.clear()
                        woke = self._play_event.wait(timeout=0.1)
                        if not woke:
                            self._seek_wake_event.wait(timeout=0.0)

                        pending_seek = self._consume_seek_request()
                        if pending_seek is not None:
                            self._apply_seek(cap, pending_seek)
                            success, frame = cap.read()
                            if success:
                                pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                                annotated = self._process_frame(frame)
                                self._store_latest_frame(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                                self._set_playback_position(pos_sec)
                    if not self._run_flag:
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    self._set_playback_position(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

                    if self.args.frame_mode == 'realtime':
                        now = time.perf_counter()
                        if next_frame_time is None:
                            next_frame_time = now
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
                else:
                    if self.args.frame_mode == 'realtime' and reader is not None:
                        ret, frame = reader.get_latest()
                        if ret is False:
                            break
                        if ret is None:
                            time.sleep(0.001)
                            continue
                    else:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                annotated = self._process_frame(frame)
                self._store_latest_frame(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                loop_end = time.perf_counter()
                inst_fps = 1.0 / max(loop_end - prev_loop_time, 1e-6)
                prev_loop_time = loop_end
                fps_filter = inst_fps if fps_filter == 0.0 else (0.9 * fps_filter + 0.1 * inst_fps)
                self._set_stats(proc_fps=fps_filter)

        finally:
            if reader is not None:
                reader.stop()
            cap.release()


# ==========================================
# Main GUI Application
# ==========================================
class PosePipelineApp(QtWidgets.QMainWindow):
    def __init__(self, args, yolo_model, mbert_model, device):
        super().__init__()
        self.args = args
        self.render_items = {}
        self.skeleton_h36m = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (9, 10),
            (8, 14), (14, 15), (15, 16), (8, 11), (11, 12), (12, 13)
        ]
        self.colors = [(1,0.2,0.2,1), (0.2,1,0.2,1), (0.2,0.4,1,1), (1,1,0.2,1), (1,0.2,1,1), (0.2,1,1,1)]

        self.view_fps = 0.0
        self.last_view_time = time.perf_counter()
        self.last_frame_seq = -1
        self._user_is_seeking = False
        self._slider_scale = 1000.0

        self.initUI()

        self.thread = VideoThread(args, yolo_model, mbert_model, device)
        self.thread.start()

        playback_enabled = self.thread.is_playback_controllable()
        self.btn_play.setEnabled(playback_enabled)
        self.btn_pause.setEnabled(playback_enabled)
        self.seek_slider.setEnabled(playback_enabled)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setInterval(10)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start()

    def initUI(self):
        self.setWindowTitle('MotionBERT Tracker (OSC, FBX, 1EuroFilter) — Threaded')
        self.resize(1400, 800)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        visuals_layout = QtWidgets.QHBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        visuals_layout.addWidget(self.video_label, stretch=1)

        self.gl_widget = gl.GLViewWidget()
        visuals_layout.addWidget(self.gl_widget, stretch=1)
        self.gl_widget.setCameraPosition(distance=10, elevation=15, azimuth=45)
        grid = gl.GLGridItem(size=QtGui.QVector3D(20, 20, 0)); grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addLayout(visuals_layout, stretch=1)

        # --- Playback controls (Play/Pause + seek slider, in seconds) ---
        playback_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.on_play_clicked)
        playback_layout.addWidget(self.btn_play)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.on_pause_clicked)
        playback_layout.addWidget(self.btn_pause)

        self.playback_time_label = QtWidgets.QLabel("0.0 s / 0.0 s")
        playback_layout.addWidget(self.playback_time_label)

        self.seek_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self.on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self.on_seek_released)
        playback_layout.addWidget(self.seek_slider, stretch=1)

        left_panel.addLayout(playback_layout)
        main_layout.addLayout(left_panel, stretch=4)

        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 10)

        filter_group = QtWidgets.QGroupBox("One-Euro Filter")
        f_layout = QtWidgets.QFormLayout()
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.001, 5.0); self.cutoff_spin.setSingleStep(0.05); self.cutoff_spin.setValue(0.1)
        self.cutoff_spin.valueChanged.connect(self.on_filter_changed)
        self.beta_spin = QtWidgets.QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 50.0); self.beta_spin.setSingleStep(0.5); self.beta_spin.setValue(5.0)
        self.beta_spin.valueChanged.connect(self.on_filter_changed)
        f_layout.addRow("Min Cutoff:", self.cutoff_spin)
        f_layout.addRow("Beta Speed:", self.beta_spin)
        filter_group.setLayout(f_layout)
        controls_layout.addWidget(filter_group)

        osc_group = QtWidgets.QGroupBox("OSC Export")
        o_layout = QtWidgets.QFormLayout()
        self.ip_input = QtWidgets.QLineEdit("127.0.0.1")
        self.port_input = QtWidgets.QSpinBox(); self.port_input.setRange(1024, 65535); self.port_input.setValue(9007)
        self.btn_apply_osc = QtWidgets.QPushButton("Apply OSC")
        self.btn_apply_osc.clicked.connect(self.apply_osc)
        o_layout.addRow("IP Address:", self.ip_input)
        o_layout.addRow("Port:", self.port_input)
        o_layout.addRow(self.btn_apply_osc)
        osc_group.setLayout(o_layout)
        controls_layout.addWidget(osc_group)

        track_group = QtWidgets.QGroupBox("Tracking")
        t_layout = QtWidgets.QVBoxLayout()
        self.conf_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100); self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        t_layout.addWidget(QtWidgets.QLabel("Keypoint Confidence Threshold"))
        t_layout.addWidget(self.conf_slider)
        track_group.setLayout(t_layout)
        controls_layout.addWidget(track_group)

        fps_group = QtWidgets.QGroupBox("FBX Export FPS")
        fps_layout = QtWidgets.QFormLayout()
        self.export_fps_mode_combo = QtWidgets.QComboBox()
        self.export_fps_mode_combo.addItems(["Auto (source / achieved)", "Manual"])
        self.export_fps_mode_combo.currentIndexChanged.connect(self.on_export_fps_mode_changed)
        fps_layout.addRow("Mode:", self.export_fps_mode_combo)

        self.export_fps_spin = QtWidgets.QDoubleSpinBox()
        self.export_fps_spin.setRange(1.0, 240.0)
        self.export_fps_spin.setSingleStep(1.0)
        self.export_fps_spin.setDecimals(3)
        self.export_fps_spin.setValue(30.0)
        self.export_fps_spin.setEnabled(False)
        self.export_fps_spin.valueChanged.connect(self.on_export_fps_value_changed)
        fps_layout.addRow("Manual FPS:", self.export_fps_spin)

        self.export_fps_preview_label = QtWidgets.QLabel("...")
        fps_layout.addRow("Will export at:", self.export_fps_preview_label)
        fps_group.setLayout(fps_layout)
        controls_layout.addWidget(fps_group)

        rec_group = QtWidgets.QGroupBox("Recording")
        r_layout = QtWidgets.QVBoxLayout()
        self.btn_start_rec = QtWidgets.QPushButton("Start Recording")
        self.btn_start_rec.clicked.connect(self.start_recording)
        self.btn_stop_rec = QtWidgets.QPushButton("Stop Recording")
        self.btn_stop_rec.clicked.connect(self.stop_recording)
        r_layout.addWidget(self.btn_start_rec)
        r_layout.addWidget(self.btn_stop_rec)
        rec_group.setLayout(r_layout)
        controls_layout.addWidget(rec_group)

        stats_group = QtWidgets.QGroupBox("Statistics")
        s_layout = QtWidgets.QVBoxLayout()
        self.lbl_proc_fps = QtWidgets.QLabel("Proc FPS: 0.0")
        self.lbl_view_fps = QtWidgets.QLabel("View FPS: 0.0")
        self.lbl_src_fps = QtWidgets.QLabel("Src FPS: 0.0")
        self.lbl_skipped = QtWidgets.QLabel("Skipped: 0")
        s_layout.addWidget(self.lbl_proc_fps); s_layout.addWidget(self.lbl_view_fps)
        s_layout.addWidget(self.lbl_src_fps); s_layout.addWidget(self.lbl_skipped)
        stats_group.setLayout(s_layout)
        controls_layout.addWidget(stats_group)

        controls_layout.addStretch(1)
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        controls_layout.addWidget(self.btn_exit)

        main_layout.addLayout(controls_layout, stretch=1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            if not self.thread.is_recording:
                self.start_recording()
            else:
                self.stop_recording()

    def on_filter_changed(self):
        self.thread.set_filter_params(self.cutoff_spin.value(), self.beta_spin.value())

    def on_conf_changed(self, value):
        self.thread.set_yolo_conf(value / 100.0)

    def apply_osc(self):
        self.thread.update_osc_settings(self.ip_input.text(), self.port_input.value())

    def on_export_fps_mode_changed(self, index):
        mode = 'manual' if index == 1 else 'auto'
        self.export_fps_spin.setEnabled(mode == 'manual')
        self.thread.set_export_fps_mode(mode, self.export_fps_spin.value() if mode == 'manual' else None)

    def on_export_fps_value_changed(self, value):
        if self.thread.export_fps_mode == 'manual':
            self.thread.set_export_fps_mode('manual', value)

    def on_play_clicked(self):
        self.thread.resume_playback()

    def on_pause_clicked(self):
        self.thread.pause_playback()

    def on_seek_pressed(self):
        self._user_is_seeking = True

    def on_seek_released(self):
        seconds = self.seek_slider.value() / self._slider_scale
        self.thread.seek_to_seconds(seconds)
        self._user_is_seeking = False

    def start_recording(self):
        self.thread.start_recording()
        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)

    def stop_recording(self):
        self.thread.stop_recording()
        self.btn_start_rec.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)

    @pyqtSlot()
    def refresh_ui(self):
        frame_rgb, seq = self.thread.get_latest_frame(self.last_frame_seq)
        if frame_rgb is not None:
            self.last_frame_seq = seq
            h, w, ch = frame_rgb.shape
            q_img = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888).copy()
            pixmap = QtGui.QPixmap.fromImage(q_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
            self.video_label.setPixmap(pixmap)

            now = time.perf_counter()
            view_time = now - self.last_view_time
            inst_view_fps = 1.0 / view_time if view_time > 0 else 0
            self.view_fps = 0.9 * self.view_fps + 0.1 * inst_view_fps
            self.last_view_time = now

        render_state, removed_tids = self.thread.get_latest_render_state()

        for tid in removed_tids:
            if tid in self.render_items:
                self.gl_widget.removeItem(self.render_items[tid]['scatter'])
                for line in self.render_items[tid]['lines']:
                    self.gl_widget.removeItem(line)
                del self.render_items[tid]

        for tid, state in render_state.items():
            pose = state['pose']
            plot_pts = np.zeros_like(pose)
            plot_pts[:, 0] = pose[:, 0]
            plot_pts[:, 1] = pose[:, 2]
            plot_pts[:, 2] = -pose[:, 1]

            if tid not in self.render_items:
                c = self.colors[tid % len(self.colors)]
                scatter = gl.GLScatterPlotItem(color=c, size=10)
                lines = [gl.GLLinePlotItem(color=c, width=4, antialias=True) for _ in self.skeleton_h36m]
                self.gl_widget.addItem(scatter)
                for line in lines: self.gl_widget.addItem(line)
                self.render_items[tid] = {'scatter': scatter, 'lines': lines}

            self.render_items[tid]['scatter'].setData(pos=plot_pts)
            for idx, (j1, j2) in enumerate(self.skeleton_h36m):
                self.render_items[tid]['lines'][idx].setData(pos=np.vstack([plot_pts[j1], plot_pts[j2]]))

        proc_fps, src_fps, skipped = self.thread.get_stats()
        self.lbl_proc_fps.setText(f"Proc FPS: {proc_fps:.1f}")
        self.lbl_view_fps.setText(f"View FPS: {self.view_fps:.1f}")
        self.lbl_src_fps.setText(f"Src FPS: {src_fps:.1f}")
        self.lbl_skipped.setText(f"Skipped: {skipped}")

        self.update_playback_ui()
        self.export_fps_preview_label.setText(
            f"{self.thread.get_export_fps():.2f} fps ({self.thread.export_fps_mode}/{self.args.frame_mode})"
        )

    def update_playback_ui(self):
        if not self.thread.is_playback_controllable():
            return

        is_paused, pos_sec, duration_sec = self.thread.get_playback_state()

        max_ms = int(round(duration_sec * self._slider_scale))
        if self.seek_slider.maximum() != max_ms:
            self.seek_slider.setRange(0, max(0, max_ms))

        if not self._user_is_seeking:
            pos_ms = int(round(pos_sec * self._slider_scale))
            if self.seek_slider.value() != pos_ms:
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(min(pos_ms, self.seek_slider.maximum()))
                self.seek_slider.blockSignals(False)

        self.playback_time_label.setText(f"{pos_sec:.1f} s / {duration_sec:.1f} s")
        self.btn_play.setEnabled(is_paused)
        self.btn_pause.setEnabled(not is_paused)

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.thread.stop()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video file path or camera ID')
    parser.add_argument("--yolo_model", type=str, choices=["n", "s", "m", "l", "x"], default="s",
                        help="YOLO pose model size")
    parser.add_argument('--mbert_model', type=str, default='MotionBERT/ft_h36m.bin', help='Path to MotionBERT checkpoint')
    parser.add_argument('--frame_mode', type=str, choices=['all', 'realtime'], default='realtime', help='Process all frames vs sync to realtime')
    parser.add_argument('--max_tracks', type=int, default=-1, help='Fix exactly N tracks (default -1 for dynamic). Extra people ignored. Missing people start in T-pose.')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("Loading Models...")
    model_dir = "models"
    model_name = f"yolov8{args.yolo_model}-pose.pt"
    model_path = os.path.join(model_dir, model_name)
    yolo_model = YOLO(model_path)

    motionbert = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2, num_joints=17, maxlen=243, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=torch.nn.LayerNorm).to(device)

    checkpoint = torch.load(args.mbert_model, map_location=device)
    state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint['model'] if 'model' in checkpoint else checkpoint
    from collections import OrderedDict
    motionbert.load_state_dict(OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()]), strict=False)
    motionbert.eval()

    window = PosePipelineApp(args, yolo_model, motionbert, device)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
