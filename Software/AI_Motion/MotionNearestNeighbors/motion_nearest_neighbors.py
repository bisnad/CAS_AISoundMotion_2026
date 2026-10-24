"""
Motion Nearest Neighbors (pyqtgraph) - standalone
- builds a new motion by chaining motion excerpts via nearest-neighbor search in feature space
- the generated motion is played back (and sent via OSC) while the search is still running
"""

import os, sys, json, time, socket, traceback, pathlib, threading
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import npz_tools as npz
from common import mocap_tools as mocap
import motion_analysis as ma

try:
    import motion_sender
except Exception:
    motion_sender = None

"""
Settings
"""

config = {
    "mocap_dir": "data/mocap",
    "file_name": "polytopia_fullbody_take2.fbx",
    "topology_file_name": "",                       # needed for npz
    "joint_weight_file": "configs/joint_weights_qualisys_fbx.json",
    "body_weight": 60,
    "fps": 50,                                      # used for npz, otherwise data["frame_rate"] if present
    "pos_scale": 1.0,
    "up_axis": 1,                                   # 1 = Y up (FBX/BVH), 2 = Z up
    "view_scale": 1.0,                              # world data is assumed to be in cm
    "excerpt_length_s": 1.6,
    "excerpt_overlap_s": 0.8,                       # hop = length - overlap
    "valid_start_s": 4.0,                           # only use this part of the recording
    "valid_end_s": 128.0,                           # 0 = until the end
    "features": ["bsphere", "space_effort"],
    "start_excerpt": 0,
    "step_interval_s": 0.05,                        # time between nearest neighbor steps
    "osc_ip": "127.0.0.1",
    "osc_port": 9008,
}

ALL_FEATURES = [
    "space_effort", "flow_effort", "time_effort", "weight_effort", "space_effort_v1",
    "qom", "bbox", "bsphere",
    "pos_scalar", "vel_scalar", "accel_scalar", "jerk_scalar",
    "vel_world", "accel_world", "jerk_world",
    "curvature", "travel_distance", "windowed_travel_distance", "volume",
]
SLOW_FEATURES = {"volume"}
MAX_EXCERPTS = 5000
SM = 25  # smoothing / analysis window in frames


"""
Quaternion helper (w, x, y, z)
"""

def slerp(q0, q1, w):
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    s_safe = np.where(small, 1.0, s)
    a = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / s_safe)
    b = np.where(small, w, np.sin(w * theta) / s_safe)
    q = a * q0 + b * q1
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


"""
Mocap source: loading, analysis (lazy features), world transforms
"""

class MotionSource:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mocap_data = None
        self.file_path = ""
        self.file_type = ""
        self.fps = cfg["fps"]
        self.root = 0
        self.ctx = None
        self.feature_matrices = {}
        self.tools_mocap = mocap.Mocap_Tools()

    # ---------- loading ----------
    def load(self, path, topology_file=""):
        ext = pathlib.Path(path).suffix.lower()
        if ext == ".bvh":
            d = bvh.BVH_Tools().load(path)
            md = self.tools_mocap.bvh_to_mocap(d)
            if isinstance(md, list): md = md[0]
            md["motion"]["rot_local"] = self.tools_mocap.euler_to_quat_bvh(
                md["motion"]["rot_local_euler"], md["rot_sequence"])
        elif ext == ".fbx":
            d = fbx.FBX_Tools().load(path)
            md = self.tools_mocap.fbx_to_mocap(d)[0]
            md["motion"]["rot_local"] = self.tools_mocap.euler_to_quat(
                md["motion"]["rot_local_euler"], md["rot_sequence"])
        elif ext == ".npz":
            if not topology_file:
                raise ValueError("npz files need a topology file")
            nd, topo = npz.NPZ_Tools.load(path, topology_file)
            md = self.tools_mocap.npz_to_mocap(nd, topo, self.cfg["fps"])
            if isinstance(md, list): md = md[0]
        else:
            raise ValueError("unsupported file type: " + ext)

        if isinstance(md["motion"]["pos_local"], list):
            raise ValueError("variable-timestamp motion data is not supported")

        md["skeleton"]["offsets"] = md["skeleton"]["offsets"] * self.cfg["pos_scale"]
        md["motion"]["pos_local"] = md["motion"]["pos_local"] * self.cfg["pos_scale"]
        md["skeleton"]["offsets"][0, 0] = 0.0
        md["skeleton"]["offsets"][0, 2] = 0.0

        pos_world, rot_world = self.tools_mocap.local_to_world(
            md["motion"]["rot_local"], md["motion"]["pos_local"], md["skeleton"])
        md["motion"]["pos_world"] = pos_world
        md["motion"]["rot_world"] = rot_world

        self.mocap_data = md
        self.file_path = path
        self.file_type = ext[1:]
        fr = md.get("frame_rate", None)
        self.fps = float(fr) if fr and fr > 0 else float(self.cfg["fps"])

        parents = np.asarray(md["skeleton"]["parents"]).astype(int)
        roots = np.where(parents < 0)[0]
        self.root = int(roots[0]) if len(roots) > 0 else 0

        self._analyse()

    def frame_count(self):
        return self.mocap_data["motion"]["rot_local"].shape[0]

    def _joint_weights(self, joint_count):
        try:
            with open(self.cfg["joint_weight_file"]) as fh:
                w = np.array(json.load(fh)["jointWeights"], dtype=np.float64)
            if len(w) == joint_count:
                return w * self.cfg["body_weight"] / 100.0
        except Exception:
            pass
        return np.full(joint_count, self.cfg["body_weight"] / joint_count)

    # ---------- analysis (lazy features) ----------
    def _analyse(self):
        m = self.mocap_data["motion"]
        jw = self._joint_weights(m["rot_local"].shape[1])
        dt = 1.0 / self.fps
        p = ma.smooth(m["pos_world"] / 100.0, SM)
        v = ma.smooth(ma.derivative(p, dt), SM)
        a = ma.smooth(ma.derivative(v, dt), SM)
        j = ma.smooth(ma.derivative(a, dt), SM)
        self.ctx = dict(
            jw=jw, pos_m=m["pos_world"] / 100.0, p=p, v=v, a=a, j=j,
            ps=ma.scalar(p, "norm"), vs=ma.scalar(v, "norm"),
            as_=ma.scalar(a, "norm"), js=ma.scalar(j, "norm"))
        self.feature_matrices = {}

    def _compute_feature(self, name):
        c = self.ctx
        T = self.frame_count()

        def curvature():
            k = ma.curvature(c["a"], c["v"], c["vs"] + 1e-8)
            return np.reshape(k, (T, -1))

        def travel():
            d = ma.joint_travel_distance(c["p"])
            return np.concatenate([np.zeros_like(d[:1]), d], axis=0)

        def space_v1():
            with np.errstate(divide="ignore", invalid="ignore"):
                return ma.space_effort(c["ps"], c["jw"], SM)

        def volume():
            out = np.zeros((T, 1))
            for t in range(T):
                try:
                    out[t, 0] = ma.joint_volumes(c["pos_m"][t])[0, 0]
                except Exception:
                    pass
            return out

        table = {
            "weight_effort": lambda: ma.weight_effort(c["vs"], c["jw"], SM),
            "space_effort": lambda: ma.space_effort_v2(c["pos_m"], c["jw"], SM),
            "time_effort": lambda: ma.time_effort(c["as_"], c["jw"], SM),
            "flow_effort": lambda: ma.flow_effort(c["js"], c["jw"], SM),
            "space_effort_v1": space_v1,
            "qom": lambda: ma.quantity_of_motion(c["vs"], c["jw"]),
            "bbox": lambda: ma.bounding_box(c["pos_m"]),
            "bsphere": lambda: ma.bounding_sphere(c["pos_m"]),
            "pos_scalar": lambda: c["ps"],
            "vel_scalar": lambda: c["vs"],
            "accel_scalar": lambda: c["as_"],
            "jerk_scalar": lambda: c["js"],
            "vel_world": lambda: c["v"],
            "accel_world": lambda: c["a"],
            "jerk_world": lambda: c["j"],
            "curvature": curvature,
            "travel_distance": travel,
            "windowed_travel_distance": lambda: ma.windowed_joint_travel_distance(c["p"], SM),
            "volume": volume,
        }
        f = np.asarray(table[name](), dtype=np.float64)
        f = np.reshape(f, (f.shape[0], -1))
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        if f.shape[0] < T:
            f = np.concatenate([f, np.repeat(f[-1:], T - f.shape[0], axis=0)])
        return f[:T]

    def get_feature(self, name):
        if name not in self.feature_matrices:
            try:
                self.feature_matrices[name] = self._compute_feature(name)
            except Exception as e:
                raise RuntimeError("feature '{}' failed: {}".format(name, e))
        return self.feature_matrices[name]

    # ---------- world transforms ----------
    def to_world(self, rot_local, pos_local):
        return self.tools_mocap.local_to_world(rot_local, pos_local, self.mocap_data["skeleton"])

    def fix_root_at_origin(self, pos_local, pos_world):
        """Remove the horizontal root translation (all axes except up_axis)."""
        pos_local, pos_world = pos_local.copy(), pos_world.copy()
        delta = pos_world[:, self.root, :].copy()
        delta[:, self.cfg["up_axis"]] = 0.0
        pos_world -= delta[:, None, :]
        pos_local[:, self.root, :] -= delta
        return pos_local, pos_world


"""
Export (same format as the loaded file)
"""

def write_mocap(source, md, base_path):
    """base_path has no extension. Returns the list of files written."""
    mocap_t = source.tools_mocap
    t = source.file_type
    if t == "fbx":
        scene = mocap_t.mocap_to_fbx([md])
        fbx.FBX_Tools().write(scene, base_path + ".fbx")
        return [base_path + ".fbx"]
    if t == "bvh":
        bvh_data = mocap_t.mocap_to_bvh(md)
        bvh.BVH_Tools().write(bvh_data, base_path + ".bvh")
        return [base_path + ".bvh"]
    if t == "npz":
        npz_dict = mocap_t.mocap_to_npz([md])
        np.savez_compressed(base_path + ".npz", **npz_dict)
        sk = md["skeleton"]
        topo = {
            "parents": [int(p) for p in sk.get("parents", [])],
            "children": [[int(c) for c in ch] for ch in sk.get("children", [])],
            "joints": [str(j) for j in sk.get("joints", [])],
        }
        with open(base_path + "_topology.json", "w") as fh:
            json.dump(topo, fh, indent=2)
        return [base_path + ".npz", base_path + "_topology.json"]
    raise RuntimeError("unknown format " + t)


"""
Model: excerpt features, nearest neighbor chaining, motion blending
"""

class NearestNeighborModel:

    def __init__(self, cfg):
        self.cfg = cfg
        self.base = MotionSource(cfg)
        self.lock = threading.RLock()
        self.ready = False
        self.done = False
        self.count = 0
        self.final_count = 0
        self.sequence = []
        self.starts = np.zeros(0, dtype=int)
        self.L = self.ov = self.hop = 0
        self.total_frames = 0

    # ---------- excerpts + features ----------
    def prepare(self, length_s, overlap_s, names, normalize, equal_weight,
                valid_start_s, valid_end_s, start_index):
        b = self.base
        fps, N = b.fps, b.frame_count()
        L = max(2, int(round(length_s * fps)))
        ov = min(max(0, int(round(overlap_s * fps))), L - 1)
        hop = L - ov
        a = max(0, int(round(valid_start_s * fps)))
        e = N if valid_end_s <= 0 else min(N, int(round(valid_end_s * fps)))
        if e - a < L + hop:
            raise ValueError("valid frame range is too short for these excerpts")
        starts = np.arange(a, e - L + 1, hop)
        if len(starts) > MAX_EXCERPTS:
            raise ValueError("{} excerpts exceed the limit of {}".format(len(starts), MAX_EXCERPTS))
        if len(starts) < 3:
            raise ValueError("fewer than 3 excerpts")

        blocks = []
        for n in names:
            X = b.get_feature(n)
            if normalize:
                X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
            if equal_weight:
                X = X / np.sqrt(X.shape[1])
            blocks.append(X)
        X = np.concatenate(blocks, axis=1)
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        feats = (cs[starts + L] - cs[starts]) / L

        ramp = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, ov)) if ov > 0 else np.zeros(0)
        env = np.concatenate([ramp, np.ones(L - ov)])

        with self.lock:
            self.starts, self.feats = starts, feats
            self.L, self.ov, self.hop, self.env = L, ov, hop, env
            self.total_frames = (len(starts) - 1) * hop + L
            self.ready = True
            self.reset(start_index)
        return len(starts)

    # ---------- generation state ----------
    def reset(self, start_index=None):
        with self.lock:
            n = len(self.starts)
            if start_index is not None:
                self.start_index = int(np.clip(start_index, 0, max(0, n - 1)))
            J = self.base.mocap_data["motion"]["rot_local"].shape[1]
            cap = min(self.total_frames, 4 * self.L)
            self.rot_buf = np.zeros((cap, J, 4), np.float32)
            self.rot_buf[..., 0] = 1.0
            self.pos_buf = np.zeros((cap, J, 3), np.float32)
            self.used = np.zeros(n, dtype=bool)
            self.sequence = []
            self.cur = -1
            self.count = 0
            self.final_count = 0
            self.done = False
            self.next_base_h = np.zeros(3)

    def _ensure(self, needed):
        cap = self.rot_buf.shape[0]
        if needed <= cap: return
        new_cap = min(max(needed, 2 * cap), self.total_frames)
        J = self.rot_buf.shape[1]
        rot = np.zeros((new_cap, J, 4), np.float32)
        rot[..., 0] = 1.0
        rot[:cap] = self.rot_buf
        pos = np.zeros((new_cap, J, 3), np.float32)
        pos[:cap] = self.pos_buf
        self.rot_buf, self.pos_buf = rot, pos

    def step(self):
        """Pick the nearest unused excerpt and blend it into the generated motion."""
        with self.lock:
            if not self.ready or self.done:
                return
            if self.count == 0:
                idx = self.start_index
            else:
                d = np.linalg.norm(self.feats - self.feats[self.cur], axis=1)
                d[self.used] = np.inf
                idx = int(np.argmin(d))

            m = self.base.mocap_data["motion"]
            root, up = self.base.root, self.cfg["up_axis"]
            L, hop = self.L, self.hop
            s = int(self.starts[idx])
            p = self.count * hop
            self._ensure(p + L)

            exc_rot = m["rot_local"][s:s + L]
            exc_pos = m["pos_local"][s:s + L].copy()
            for a in range(3):
                if a != up:
                    exc_pos[:, root, a] += self.next_base_h[a] - exc_pos[0, root, a]
            self.next_base_h = exc_pos[min(hop, L - 1), root, :].copy()

            w = (np.ones(L) if self.count == 0 else self.env)[:, None, None]
            self.rot_buf[p:p + L] = slerp(self.rot_buf[p:p + L], exc_rot, w)
            self.pos_buf[p:p + L] = (1.0 - w) * self.pos_buf[p:p + L] + w * exc_pos

            self.used[idx] = True
            self.cur = idx
            self.sequence.append(idx)
            self.count += 1
            if self.used.all():
                self.done = True
                self.final_count = p + L
            else:
                self.final_count = p + hop

    def progress(self):
        n = len(self.starts)
        return self.count / n if n else 0.0

    def excerpt_at(self, frame):
        seq = self.sequence
        if not seq: return -1
        return seq[min(frame // max(1, self.hop), len(seq) - 1)]

    # ---------- frame / export access ----------
    def get_frame(self, i, fix_root):
        rot = self.rot_buf[i:i + 1].astype(np.float64)
        pos = self.pos_buf[i:i + 1].astype(np.float64)
        pw, rw = self.base.to_world(rot, pos)
        if fix_root:
            pos, pw = self.base.fix_root_at_origin(pos, pw)
        return rot[0], pos[0], pw[0], rw[0]

    def export_arrays(self):
        with self.lock:
            f = self.final_count
            return (self.rot_buf[:f].astype(np.float64), self.pos_buf[:f].astype(np.float64))


def build_sequence_export(model, fix_root):
    base = model.base
    rot, pos = model.export_arrays()
    if rot.shape[0] < 2:
        raise ValueError("nothing generated yet")
    pw, rw = base.to_world(rot, pos)
    if fix_root:
        pos, pw = base.fix_root_at_origin(pos, pw)
    rot_sequence = base.mocap_data.get("rot_sequence", [0, 1, 2])
    return {
        "frame_rate": float(base.fps),
        "rot_sequence": rot_sequence,
        "skeleton": base.mocap_data["skeleton"],
        "motion": {
            "pos_local": pos.astype(np.float32),
            "rot_local": rot.astype(np.float32),
            "pos_world": pw.astype(np.float32),
            "rot_world": rw.astype(np.float32),
            "rot_local_euler": base.tools_mocap.quat_to_euler(rot.astype(np.float32), rot_sequence),
        },
    }


"""
Background task helper
"""

class TaskWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.finished.emit(self.fn())
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


"""
GUI
"""

class NearestNeighborGui(QtWidgets.QWidget):

    def __init__(self, model, cfg, sender=None):
        super().__init__()
        self.model, self.cfg = model, cfg
        self.osc_out = sender
        self.osc_address = (cfg["osc_ip"], int(cfg["osc_port"]))
        self._thread = self._worker = None
        self.gen_thread = None
        self.gen_event = None
        self.topology_file = cfg["topology_file_name"]
        self.playhead = 0.0
        self.clock = None
        self.vs = cfg["view_scale"]
        self.edge_pairs = []

        # ---- 3D view ----
        self.canvas = gl.GLViewWidget()
        self.canvas.setCameraPosition(distance=250 * self.vs, elevation=15, azimuth=-90)
        self.canvas.opts["center"] = Vector(0, 0, 90 * self.vs)
        grid = gl.GLGridItem()
        grid.setSize(400 * self.vs, 400 * self.vs)
        grid.setSpacing(25 * self.vs, 25 * self.vs)
        self.canvas.addItem(grid)
        self.bones = gl.GLLinePlotItem(mode="lines", width=3.0, color=(1, 1, 1, 0.9))
        self.joints = gl.GLScatterPlotItem(size=9.0, color=(1, 1, 1, 0.6))
        self.canvas.addItem(self.bones)
        self.canvas.addItem(self.joints)
        self.info_label = QtWidgets.QLabel("")
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.canvas, 1)
        left.addWidget(self.info_label)

        # ---- file ----
        self.q_load = QtWidgets.QPushButton("load mocap file...")
        self.q_load.clicked.connect(self.load_file)
        self.q_topo = QtWidgets.QPushButton("topology file (npz)...")
        self.q_topo.clicked.connect(self.choose_topology)
        self.q_status = QtWidgets.QLabel("")
        self.q_status.setWordWrap(True)
        file_box = QtWidgets.QGroupBox("mocap file")
        fl = QtWidgets.QFormLayout(file_box)
        fl.addRow(self.q_load)
        fl.addRow(self.q_topo)
        fl.addRow(self.q_status)

        # ---- excerpts ----
        self.q_length = self._dspin(0.1, 60.0, cfg["excerpt_length_s"])
        self.q_overlap = self._dspin(0.0, 59.9, cfg["excerpt_overlap_s"])
        self.q_range_start = self._dspin(0.0, 100000.0, cfg["valid_start_s"])
        self.q_range_end = self._dspin(0.0, 100000.0, cfg["valid_end_s"])
        self.q_length.valueChanged.connect(self.update_excerpt_info)
        self.q_overlap.valueChanged.connect(self.update_excerpt_info)
        self.q_excerpt_info = QtWidgets.QLabel("")
        self.q_excerpt_info.setWordWrap(True)
        ex_box = QtWidgets.QGroupBox("excerpts")
        el = QtWidgets.QFormLayout(ex_box)
        el.addRow("excerpt length:", self.q_length)
        el.addRow("overlap (= blend):", self.q_overlap)
        el.addRow("use from:", self.q_range_start)
        el.addRow("use until (0 = end):", self.q_range_end)
        el.addRow(self.q_excerpt_info)

        # ---- features ----
        self.q_features = QtWidgets.QListWidget()
        self.q_features.setMaximumHeight(200)
        for name in ALL_FEATURES:
            it = QtWidgets.QListWidgetItem(name + ("  (slow)" if name in SLOW_FEATURES else ""))
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in cfg["features"] else Qt.Unchecked)
            self.q_features.addItem(it)
        feat_box = QtWidgets.QGroupBox("motion features used for search")
        QtWidgets.QVBoxLayout(feat_box).addWidget(self.q_features)

        # ---- search settings ----
        self.q_norm = QtWidgets.QCheckBox("normalize features (z-score)")
        self.q_norm.setChecked(True)
        self.q_equal = QtWidgets.QCheckBox("equal weight per feature")
        self.q_equal.setChecked(True)
        self.q_start_excerpt = QtWidgets.QSpinBox()
        self.q_start_excerpt.setRange(0, 100000)
        self.q_start_excerpt.setValue(cfg["start_excerpt"])
        self.q_step_interval = self._dspin(0.0, 5.0, cfg["step_interval_s"])
        self.q_apply = QtWidgets.QPushButton("apply analysis settings")
        self.q_apply.clicked.connect(lambda: self.start_analysis(None))
        search_box = QtWidgets.QGroupBox("search")
        sl = QtWidgets.QFormLayout(search_box)
        sl.addRow(self.q_norm)
        sl.addRow(self.q_equal)
        sl.addRow("start excerpt:", self.q_start_excerpt)
        sl.addRow("step interval:", self.q_step_interval)
        sl.addRow(self.q_apply)

        # ---- playback ----
        self.q_start = QtWidgets.QPushButton("start")
        self.q_start.clicked.connect(self.start)
        self.q_stop = QtWidgets.QPushButton("stop")
        self.q_stop.clicked.connect(self.stop)
        self.q_reset = QtWidgets.QPushButton("reset")
        self.q_reset.clicked.connect(self.reset_generation)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.q_start)
        row.addWidget(self.q_stop)
        row.addWidget(self.q_reset)
        self.q_speed = self._dspin(0.1, 4.0, 1.0)
        self.q_speed.setSuffix(" x")
        self.q_loop = QtWidgets.QCheckBox("loop when search is complete")
        self.q_loop.setChecked(True)
        self.q_fix_root = QtWidgets.QCheckBox("fix root at origin (playback, osc, export)")
        self.q_save = QtWidgets.QPushButton("save generated motion...")
        self.q_save.clicked.connect(self.save_motion)
        self.q_progress = QtWidgets.QProgressBar()
        self.q_progress.setRange(0, 100)
        self.q_gen_status = QtWidgets.QLabel("")
        self.q_gen_status.setWordWrap(True)

        # ---- osc output ----
        self.q_osc = QtWidgets.QCheckBox("send osc to {}:{}".format(*self.osc_address))
        self.q_osc.setEnabled(self.osc_out is not None)
        self.q_osc_ip = QtWidgets.QLineEdit(self.osc_address[0])
        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1, 65535)
        self.q_osc_port.setKeyboardTracking(False)
        self.q_osc_port.setValue(self.osc_address[1])
        self.q_osc_ip.editingFinished.connect(self.change_osc_address)
        self.q_osc_port.editingFinished.connect(self.change_osc_address)
        osc_box = QtWidgets.QGroupBox("osc output")
        ol = QtWidgets.QFormLayout(osc_box)
        ol.addRow(self.q_osc)
        ol.addRow("ip / host:", self.q_osc_ip)
        ol.addRow("port:", self.q_osc_port)

        right = QtWidgets.QVBoxLayout()
        for w in (file_box, ex_box, feat_box, search_box):
            right.addWidget(w)
        right.addLayout(row)
        right.addWidget(self.q_speed)
        right.addWidget(self.q_loop)
        right.addWidget(self.q_fix_root)
        right.addWidget(self.q_progress)
        right.addWidget(self.q_gen_status)
        right.addWidget(osc_box)
        right.addWidget(self.q_save)
        right.addStretch(1)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(410)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addWidget(scroll)
        self.setGeometry(50, 50, 1300, 950)
        self.setWindowTitle("Motion Nearest Neighbors")

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(200)
        self.progress_timer.timeout.connect(self.refresh_progress)
        self.progress_timer.start()
        self.update_excerpt_info()

    # ---------- helpers ----------
    def _dspin(self, lo, hi, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(3)
        s.setSingleStep(0.05)
        s.setRange(lo, hi)
        s.setSuffix(" s")
        s.setValue(val)
        return s

    def checked_features(self):
        return [self.q_features.item(i).data(Qt.UserRole) for i in range(self.q_features.count())
                if self.q_features.item(i).checkState() == Qt.Checked]

    def update_excerpt_info(self, *a):
        self.q_overlap.setMaximum(max(0.0, self.q_length.value() - 0.01))
        hop = max(0.0, self.q_length.value() - self.q_overlap.value())
        self.q_excerpt_info.setText("hop {:.3f} s".format(hop))

    def set_busy(self, busy, msg=""):
        for w in (self.q_load, self.q_topo, self.q_apply, self.q_save, self.q_features,
                  self.q_length, self.q_overlap, self.q_range_start, self.q_range_end,
                  self.q_norm, self.q_equal, self.q_start_excerpt, self.q_start, self.q_reset):
            w.setEnabled(not busy)
        self.q_status.setText(msg)

    def run_task(self, fn, on_done, on_fail):
        self._thread = QThread(self)
        self._worker = TaskWorker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(on_fail)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    # ---------- osc address ----------
    def change_osc_address(self, *a):
        host = self.q_osc_ip.text().strip()
        port = int(self.q_osc_port.value())
        if (host, port) == self.osc_address:
            return
        if motion_sender is None:
            self.q_status.setText("motion_sender module not available.")
            return
        try:
            socket.gethostbyname(host)
            new_out = motion_sender.OscSender({"ip": host, "port": port})
        except Exception as e:
            self.q_status.setText("invalid osc address '{}:{}': {}".format(host, port, e))
            return
        self.osc_out = new_out
        self.osc_address = (host, port)
        self.q_osc.setEnabled(True)
        self.q_osc.setText("send osc to {}:{}".format(host, port))
        self.q_status.setText("osc output: {}:{}".format(host, port))

    # ---------- file + analysis ----------
    def choose_topology(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Topology file", "", "Topology (*.json *.yaml *.txt);;All (*.*)")
        if f:
            self.topology_file = f
            self.q_status.setText("topology: " + pathlib.Path(f).name)

    def load_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load mocap file", self.cfg["mocap_dir"], "Mocap files (*.bvh *.fbx *.npz)")
        if f:
            if f.lower().endswith(".npz") and not self.topology_file:
                self.choose_topology()
            self.start_analysis(f)

    def start_analysis(self, file_path):
        names = self.checked_features()
        if not names:
            self.q_status.setText("select at least one motion feature.")
            return
        if file_path is None and self.model.base.mocap_data is None:
            self.q_status.setText("load a mocap file first.")
            return
        self.stop()
        params = dict(length_s=self.q_length.value(), overlap_s=self.q_overlap.value(), names=names,
                      normalize=self.q_norm.isChecked(), equal_weight=self.q_equal.isChecked(),
                      valid_start_s=self.q_range_start.value(), valid_end_s=self.q_range_end.value(),
                      start_index=self.q_start_excerpt.value())
        topo = self.topology_file

        def fn():
            if file_path:
                self.model.base.load(file_path, topo)
            return self.model.prepare(**params)

        self.set_busy(True, "loading and analysing...")
        self.run_task(fn, self.on_analysis_done, self.on_analysis_failed)

    def on_analysis_done(self, n):
        b = self.model.base
        parents = np.asarray(b.mocap_data["skeleton"]["parents"]).astype(int)
        self.edge_pairs = [(i, int(p)) for i, p in enumerate(parents) if p >= 0]
        self.q_start_excerpt.setMaximum(max(0, n - 1))
        self.playhead = 0.0
        self.set_busy(False, "{} excerpts of {:.2f} s ({}), {:.1f} fps.".format(
            n, self.q_length.value(), b.file_type, b.fps))
        self.start()

    def on_analysis_failed(self, msg):
        self.set_busy(False, "failed: " + msg)

    # ---------- generation + playback ----------
    def start(self):
        if not self.model.ready:
            self.q_gen_status.setText("load a mocap file first.")
            return
        if (self.gen_thread is None or not self.gen_thread.is_alive()) and not self.model.done:
            self.gen_event = threading.Event()
            self.gen_thread = threading.Thread(target=self.generation_loop, daemon=True)
            self.gen_thread.start()
        self.clock = time.time()
        self.timer.start()

    def stop(self):
        if self.gen_event is not None:
            self.gen_event.set()
        if self.gen_thread is not None:
            self.gen_thread.join(timeout=1.0)
            self.gen_thread = None
        self.timer.stop()

    def reset_generation(self):
        self.stop()
        if self.model.ready:
            self.model.reset(self.q_start_excerpt.value())
        self.playhead = 0.0
        self.q_gen_status.setText("reset - press start.")

    def generation_loop(self):
        while not self.gen_event.is_set() and not self.model.done:
            try:
                self.model.step()
            except Exception:
                traceback.print_exc()
                break
            time.sleep(self.q_step_interval.value())

    def tick(self):
        now = time.time()
        dt, self.clock = now - self.clock, now
        final = self.model.final_count
        if final < 2:
            return
        self.playhead += dt * self.model.base.fps * self.q_speed.value()
        if self.playhead >= final - 1:
            if self.model.done and self.q_loop.isChecked():
                self.playhead %= (final - 1)
            else:
                self.playhead = float(final - 1)
        self.draw(int(self.playhead))

    def to_display(self, p):
        p = p * self.vs
        if self.cfg["up_axis"] == 1:
            return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
        return p

    def draw(self, f):
        m = self.model
        f = max(0, min(f, m.final_count - 1))
        rot, pos, pw, rw = m.get_frame(f, self.q_fix_root.isChecked())
        p = self.to_display(pw)
        lines = np.array([p[i] if j == 0 else p[par] for i, par in self.edge_pairs for j in (0, 1)])
        self.bones.setData(pos=lines, color=(1.0, 0.85, 0.3, 0.95))
        self.joints.setData(pos=p, color=(1.0, 0.85, 0.3, 0.6))

        exc = m.excerpt_at(f)
        self.info_label.setText("frame {} / {} generated | source excerpt {} ({:.2f} s)".format(
            f, m.final_count, exc, (m.starts[exc] / m.base.fps) if exc >= 0 else 0.0))

        if self.osc_out is not None and self.q_osc.isChecked():
            out = self.osc_out
            out.send("/nn/excerpt", int(exc))
            out.send("/mocap/0/joint/pos_local", pos)
            out.send("/mocap/0/joint/rot_local", rot)
            out.send("/mocap/0/joint/pos_world", pw)
            out.send("/mocap/0/joint/rot_world", rw)

    def refresh_progress(self):
        m = self.model
        if not m.ready:
            return
        self.q_progress.setValue(int(m.progress() * 100))
        fps = m.base.fps
        text = "generated {:.1f} s, played {:.1f} s".format(m.final_count / fps, self.playhead / fps)
        if m.done:
            text += " (search complete)"
        elif self.timer.isActive() and self.playhead >= m.final_count - 1:
            text += " (waiting for more motion...)"
        self.q_gen_status.setText(text)

    # ---------- save ----------
    def save_motion(self):
        b = self.model.base
        if not self.model.ready or self.model.final_count < 2:
            self.q_status.setText("nothing generated yet.")
            return
        ext = b.file_type
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save generated motion", "results/nearest_neighbors." + ext, "{} files (*.{})".format(ext, ext))
        if not path:
            return
        base_path = os.path.splitext(path)[0]
        fix = self.q_fix_root.isChecked()

        def fn():
            md = build_sequence_export(self.model, fix)
            return write_mocap(b, md, base_path)

        self.set_busy(True, "saving...")
        self.run_task(fn,
                      lambda files: self.set_busy(False, "saved: " + ", ".join(files)),
                      self.on_save_failed)

    def on_save_failed(self, msg):
        self.set_busy(False, "saving failed: " + msg)
        QtWidgets.QMessageBox.critical(self, "Saving failed", msg)

    def closeEvent(self, ev):
        self.stop()
        super().closeEvent(ev)


"""
Main
"""

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    model = NearestNeighborModel(config)
    sender = None
    if motion_sender is not None:
        sender = motion_sender.OscSender({"ip": config["osc_ip"], "port": config["osc_port"]})
    gui = NearestNeighborGui(model, config, sender)
    gui.show()
    first = os.path.join(config["mocap_dir"], config["file_name"])
    if config["file_name"] and os.path.exists(first):
        QTimer.singleShot(200, lambda: gui.start_analysis(first))
    app.exec_()