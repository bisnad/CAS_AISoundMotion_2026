"""
Motion Clustering GUI (pyqtgraph)
- loads bvh/fbx/npz mocap, computes motion descriptors, clusters excerpts
- plays the excerpts of the selected cluster as a looped, crossfaded skeleton animation
- saves each cluster as fbx/bvh/npz (format of the loaded file)
"""

import os, sys, json, time, socket, traceback, colorsys, pathlib
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

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
    "file_name": "Muriel_Take4_Blumen_Baile.fbx",
    "topology_file_name": "",                       # needed for npz
    "joint_weight_file": "configs/joint_weights_xsens_fbx.json",
    "body_weight": 60,
    "fps": 50,                                      # used for npz, otherwise data["frame_rate"] if present
    "pos_scale": 1.0,
    "up_axis": 1,                                   # 1 = Y up (FBX/BVH), 2 = Z up
    "view_scale": 1.0,                              # world data is assumed to be in cm
    "excerpt_length_s": 1.8,
    "excerpt_overlap_s": 0.9,
    "crossfade_s": 0.3,
    "cluster_count": 10,
    "cluster_method": "kmeans",
    "random_state": 170,
    "features": ["space_effort", "flow_effort", "time_effort", "weight_effort"],
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
CLUSTER_METHODS = ["kmeans", "minibatch_kmeans", "agglomerative", "gmm"]
MAX_EXCERPTS = 5000
SM = 25  # smoothing / analysis window in frames


def cluster_color(label, count):
    r, g, b = colorsys.hsv_to_rgb((label / max(1, count)) % 1.0, 0.65, 1.0)
    return (r, g, b)


"""
Quaternion helpers (w, x, y, z)
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
Model: loading, analysis, excerpts, clustering, crossfaded timeline rendering
"""

class MotionClusterModel:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mocap_data = None
        self.file_path = ""
        self.file_type = ""
        self.fps = cfg["fps"]
        self.root = 0
        self.ctx = None
        self.feature_matrices = {}
        self.cluster_method = cfg["cluster_method"]
        self.cluster_count = cfg["cluster_count"]
        self.starts = np.zeros(0, dtype=int)
        self.length_frames = 0
        self.hop_frames = 0
        self.labels = np.zeros(0, dtype=int)
        self.features = None
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
            raise ValueError("variable-timestamp motion data is not supported by the clustering tool")

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

    # ---------- excerpts + clustering ----------
    def frame_count(self):
        return self.mocap_data["motion"]["rot_local"].shape[0]

    def count_excerpts(self, length_s, overlap_s):
        if self.mocap_data is None: return 0
        L = max(2, int(round(length_s * self.fps)))
        hop = max(1, L - int(round(overlap_s * self.fps)))
        return max(0, (self.frame_count() - L) // hop + 1)

    def cluster(self, length_s, overlap_s, names, method, count, normalize, equal_weight):
        N = self.frame_count()
        L = max(2, int(round(length_s * self.fps)))
        hop = max(1, L - int(round(overlap_s * self.fps)))
        if N < L: raise ValueError("excerpt is longer than the recording")
        starts = np.arange(0, N - L + 1, hop)
        if len(starts) > MAX_EXCERPTS:
            raise ValueError("{} excerpts exceed the limit of {}".format(len(starts), MAX_EXCERPTS))
        if len(starts) < max(2, count):
            raise ValueError("only {} excerpts for {} clusters".format(len(starts), count))

        blocks = []
        for n in names:
            X = self.get_feature(n)
            if normalize:
                X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
            if equal_weight:
                X = X / np.sqrt(X.shape[1])
            blocks.append(X)
        X = np.concatenate(blocks, axis=1)
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        feats = (cs[starts + L] - cs[starts]) / L

        rs = self.cfg["random_state"]
        if method == "kmeans":
            labels = KMeans(n_clusters=count, n_init="auto", random_state=rs).fit_predict(feats)
        elif method == "minibatch_kmeans":
            labels = MiniBatchKMeans(n_clusters=count, n_init="auto", random_state=rs).fit_predict(feats)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(feats)
        elif method == "gmm":
            labels = GaussianMixture(n_components=count, random_state=rs).fit(feats).predict(feats)
        else:
            raise ValueError("unknown method " + method)

        self.starts, self.length_frames, self.hop_frames = starts, L, hop
        self.labels, self.features = labels.astype(int), feats
        self.cluster_method, self.cluster_count = method, count

    def cluster_sizes(self):
        u, c = np.unique(self.labels, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}

    def cluster_excerpt_indices(self, label):
        return np.where(self.labels == label)[0]   # ascending = original recording order

    # ---------- crossfaded timeline ----------
    def render_timeline(self, starts, crossfade_s):
        """Concatenate excerpts (original order, looped) with slerp envelopes.
        Returns rot_local (F,J,4), pos_local (F,J,3), excerpt_number per frame (F,)"""
        m = self.mocap_data["motion"]
        rot, pos = m["rot_local"], m["pos_local"]
        L = self.length_frames
        C = int(min(round(crossfade_s * self.fps), L // 2))
        P = L - C
        n = len(starts)
        F = n * P
        j = np.arange(F)
        k = j // P
        loc = j % P
        starts = np.asarray(starts)
        N = rot.shape[0]

        cur_idx = np.minimum(starts[k] + loc, N - 1)
        prev_k = (k - 1) % n
        prev_idx = np.minimum(starts[prev_k] + loc + P, N - 1)

        if C > 0:
            x = np.clip(loc / float(C), 0.0, 1.0)
            w = np.where(loc < C, 0.5 - 0.5 * np.cos(np.pi * x), 1.0)
        else:
            w = np.ones(F)

        h = [a for a in range(3) if a != self.cfg["up_axis"]]
        pc, pp = pos[cur_idx].copy(), pos[prev_idx].copy()
        for a in h:
            pc[:, self.root, a] -= pos[starts[k], self.root, a]
            pp[:, self.root, a] -= pos[starts[prev_k], self.root, a]

        w3 = w[:, None, None]
        rot_out = slerp(rot[prev_idx], rot[cur_idx], w3)
        pos_out = (1.0 - w3) * pp + w3 * pc
        return rot_out, pos_out, k

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
Export (one file per cluster, same format as the loaded file)
"""

def build_export_data(model, label, crossfade_s, fix_root=False):
    idx = model.cluster_excerpt_indices(label)
    rot, pos, _ = model.render_timeline(model.starts[idx], crossfade_s)
    pw, rw = model.to_world(rot, pos)
    if fix_root:
        pos, pw = model.fix_root_at_origin(pos, pw)
    src = model.mocap_data
    rot_sequence = src.get("rot_sequence", [0, 1, 2])
    rot = rot.astype(np.float32)
    return {
        "frame_rate": float(model.fps),
        "rot_sequence": rot_sequence,
        "skeleton": src["skeleton"],
        "motion": {
            "pos_local": pos.astype(np.float32),
            "rot_local": rot,
            "pos_world": pw.astype(np.float32),
            "rot_world": rw.astype(np.float32),
            "rot_local_euler": model.tools_mocap.quat_to_euler(rot, rot_sequence),
        },
    }


def write_mocap(model, md, base_path):
    """base_path has no extension. Returns the list of files written."""
    mocap_t = model.tools_mocap
    t = model.file_type
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
Workers
"""

class AnalysisWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, file_path, topo, params):
        super().__init__()
        self.model, self.file_path, self.topo, self.params = model, file_path, topo, params

    def run(self):
        try:
            if self.file_path:
                self.model.load(self.file_path, self.topo)
            self.model.cluster(**self.params)
            self.finished.emit({"excerpts": len(self.model.starts), "fps": self.model.fps})
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class ExportWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, model, folder, crossfade_s, fix_root=False):
        super().__init__()
        self.model, self.folder = model, folder
        self.crossfade_s, self.fix_root = crossfade_s, fix_root

    def run(self):
        try:
            stem = pathlib.Path(self.model.file_path).stem
            sizes = self.model.cluster_sizes()
            written = []
            for i, label in enumerate(sorted(sizes)):
                md = build_export_data(self.model, label, self.crossfade_s, self.fix_root)
                base = os.path.join(self.folder, "{}_cluster{:02d}".format(stem, label))
                written.extend(write_mocap(self.model, md, base))
                self.progress.emit(i + 1, len(sizes))
            self.finished.emit({"folder": self.folder, "files": written})
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


"""
GUI
"""

class ClusterGui(QtWidgets.QWidget):

    def __init__(self, model, cfg, sender=None):
        super().__init__()
        self.model, self.cfg = model, cfg
        self.osc_out = sender
        self.osc_address = (cfg["osc_ip"], int(cfg["osc_port"]))
        self._thread = self._worker = None
        self.topology_file = cfg["topology_file_name"]
        self.current_label = 0
        self.frames_world = None
        self.frame_excerpt = None
        self.rot_frames = self.pos_frames = self.rot_world_frames = None
        self.cluster_excerpts = []
        self.playhead = 0.0
        self.period = 1
        self.playing = False
        self.last_excerpt_row = -1
        self.vs = cfg["view_scale"]

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
        self.q_crossfade = self._dspin(0.0, 30.0, cfg["crossfade_s"])
        self.q_length.valueChanged.connect(self.update_excerpt_info)
        self.q_overlap.valueChanged.connect(self.update_excerpt_info)
        self.q_crossfade.valueChanged.connect(self.rebuild_timeline_keep)
        self.q_excerpt_info = QtWidgets.QLabel("")
        self.q_excerpt_info.setWordWrap(True)
        ex_box = QtWidgets.QGroupBox("excerpts")
        el = QtWidgets.QFormLayout(ex_box)
        el.addRow("excerpt length:", self.q_length)
        el.addRow("overlap (clustering):", self.q_overlap)
        el.addRow("crossfade (playback):", self.q_crossfade)
        el.addRow(self.q_excerpt_info)

        # ---- features ----
        self.q_features = QtWidgets.QListWidget()
        self.q_features.setMaximumHeight(220)
        for name in ALL_FEATURES:
            it = QtWidgets.QListWidgetItem(name + ("  (slow)" if name in SLOW_FEATURES else ""))
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in cfg["features"] else Qt.Unchecked)
            self.q_features.addItem(it)
        feat_box = QtWidgets.QGroupBox("motion features")
        QtWidgets.QVBoxLayout(feat_box).addWidget(self.q_features)

        # ---- clustering ----
        self.q_method = QtWidgets.QComboBox()
        self.q_method.addItems(CLUSTER_METHODS)
        self.q_method.setCurrentText(cfg["cluster_method"])
        self.q_count = QtWidgets.QSpinBox()
        self.q_count.setRange(2, 200)
        self.q_count.setValue(cfg["cluster_count"])
        self.q_norm = QtWidgets.QCheckBox("normalize features (z-score)")
        self.q_norm.setChecked(True)
        self.q_equal = QtWidgets.QCheckBox("equal weight per feature")
        self.q_equal.setChecked(True)
        self.q_apply = QtWidgets.QPushButton("apply analysis settings")
        self.q_apply.clicked.connect(lambda: self.start_analysis(None))
        cl_box = QtWidgets.QGroupBox("clustering")
        cl = QtWidgets.QFormLayout(cl_box)
        cl.addRow("cluster method:", self.q_method)
        cl.addRow("cluster count:", self.q_count)
        cl.addRow(self.q_norm)
        cl.addRow(self.q_equal)
        cl.addRow(self.q_apply)

        # ---- playback ----
        self.q_start = QtWidgets.QPushButton("start")
        self.q_start.clicked.connect(self.start)
        self.q_stop = QtWidgets.QPushButton("stop")
        self.q_stop.clicked.connect(self.stop)
        self.q_speed = self._dspin(0.1, 4.0, 1.0)
        self.q_speed.setSuffix(" x")
        self.q_solo = QtWidgets.QCheckBox("loop selected excerpt only")
        self.q_solo.stateChanged.connect(self.rebuild_timeline_keep)
        self.q_fix_root = QtWidgets.QCheckBox("fix root at origin (playback, osc, export)")
        self.q_fix_root.setChecked(False)
        self.q_fix_root.stateChanged.connect(self.rebuild_timeline_keep)
        self.q_save = QtWidgets.QPushButton("save clusters as files...")
        self.q_save.clicked.connect(self.save_clusters)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.q_start)
        row.addWidget(self.q_stop)

        # ---- osc output ----
        self.q_osc = QtWidgets.QCheckBox("send osc to {}:{}".format(*self.osc_address))
        self.q_osc.setEnabled(self.osc_out is not None)
        self.q_osc_ip = QtWidgets.QLineEdit(self.osc_address[0])
        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1, 65535)
        self.q_osc_port.setKeyboardTracking(False)   # apply on Enter / focus change
        self.q_osc_port.setValue(self.osc_address[1])
        self.q_osc_ip.editingFinished.connect(self.change_osc_address)
        self.q_osc_port.editingFinished.connect(self.change_osc_address)
        osc_box = QtWidgets.QGroupBox("osc output")
        ol = QtWidgets.QFormLayout(osc_box)
        ol.addRow(self.q_osc)
        ol.addRow("ip / host:", self.q_osc_ip)
        ol.addRow("port:", self.q_osc_port)

        # ---- lists ----
        self.q_cluster_list = QtWidgets.QListWidget()
        self.q_cluster_list.currentRowChanged.connect(self.change_cluster)
        self.q_excerpt_list = QtWidgets.QListWidget()
        self.q_excerpt_list.currentRowChanged.connect(self.change_excerpt)

        right = QtWidgets.QVBoxLayout()
        for w in (file_box, ex_box, feat_box, cl_box):
            right.addWidget(w)
        right.addLayout(row)
        right.addWidget(self.q_speed)
        right.addWidget(self.q_solo)
        right.addWidget(self.q_fix_root)
        right.addWidget(osc_box)
        right.addWidget(self.q_save)
        right.addWidget(QtWidgets.QLabel("clusters (select to play):"))
        right.addWidget(self.q_cluster_list, 1)
        right.addWidget(QtWidgets.QLabel("excerpts in cluster (select to jump):"))
        right.addWidget(self.q_excerpt_list, 1)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)

        # scrollable control panel: window height no longer depends on its content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFixedWidth(400)          # 380 panel + scrollbar

        # keep the lists usable inside the scroll area
        self.q_cluster_list.setMinimumHeight(120)
        self.q_excerpt_list.setMinimumHeight(120)

        # let the 3D view and info label shrink freely
        self.canvas.setMinimumSize(200, 150)
        self.info_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                      QtWidgets.QSizePolicy.Fixed)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addWidget(scroll)

        # small minimum size + start size clamped to the available screen
        self.setMinimumSize(700, 250)
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w = min(1280, screen.width() - 100)
        h = min(1000, screen.height() - 100)
        self.resize(w, h)
        self.move(screen.x() + 50, screen.y() + 50)
        self.setWindowTitle("Motion Clustering")

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.clock = None
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
        if self.model.mocap_data is None:
            self.q_excerpt_info.setText("no file loaded")
            return
        n = self.model.count_excerpts(self.q_length.value(), self.q_overlap.value())
        hop = max(0.0, self.q_length.value() - self.q_overlap.value())
        t = "hop {:.3f} s, about {} excerpts".format(hop, n)
        if n > MAX_EXCERPTS: t += " - above the limit of {}".format(MAX_EXCERPTS)
        self.q_excerpt_info.setText(t)

    def set_busy(self, busy, msg=""):
        for w in (self.q_load, self.q_topo, self.q_apply, self.q_save, self.q_features,
                  self.q_length, self.q_overlap, self.q_method, self.q_count,
                  self.q_norm, self.q_equal, self.q_cluster_list):
            w.setEnabled(not busy)
        self.q_status.setText(msg)

    def run_worker(self, worker):
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.finished.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
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
        print("osc output now sends to {}:{}".format(host, port))

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
        if file_path is None and self.model.mocap_data is None:
            self.q_status.setText("load a mocap file first.")
            return
        self.stop()
        params = dict(length_s=self.q_length.value(), overlap_s=self.q_overlap.value(), names=names,
                      method=self.q_method.currentText(), count=self.q_count.value(),
                      normalize=self.q_norm.isChecked(), equal_weight=self.q_equal.isChecked())
        self.set_busy(True, "loading and clustering...")
        w = AnalysisWorker(self.model, file_path, self.topology_file, params)
        w.finished.connect(self.on_analysis_done)
        w.failed.connect(self.on_analysis_failed)
        self.run_worker(w)

    def on_analysis_done(self, info):
        self.set_busy(False, "{} excerpts of {:.2f} s, {:.1f} fps ({}).".format(
            info["excerpts"], self.q_length.value(), info["fps"], self.model.file_type))
        self.refresh_cluster_list()
        self.update_excerpt_info()
        self.q_cluster_list.setCurrentRow(0)
        self.rebuild_timeline()
        self.start()

    def on_analysis_failed(self, msg):
        self.set_busy(False, "failed: " + msg)

    # ---------- cluster / excerpt selection ----------
    def refresh_cluster_list(self):
        sizes = self.model.cluster_sizes()
        self.q_cluster_list.blockSignals(True)
        self.q_cluster_list.clear()
        for label, n in sorted(sizes.items()):
            it = QtWidgets.QListWidgetItem("cluster {} ({} excerpts)".format(label, n))
            it.setData(Qt.UserRole, label)
            r, g, b = cluster_color(label, len(sizes))
            it.setForeground(QBrush(QColor(int(r * 255), int(g * 255), int(b * 255))))
            self.q_cluster_list.addItem(it)
        self.q_cluster_list.blockSignals(False)

    def change_cluster(self, row):
        it = self.q_cluster_list.item(row)
        if it is None: return
        self.current_label = int(it.data(Qt.UserRole))
        idx = self.model.cluster_excerpt_indices(self.current_label)
        self.cluster_excerpts = idx
        self.q_excerpt_list.blockSignals(True)
        self.q_excerpt_list.clear()
        for i in idx:
            s = self.model.starts[i] / self.model.fps
            e = (self.model.starts[i] + self.model.length_frames) / self.model.fps
            self.q_excerpt_list.addItem("excerpt {}: {:.2f} - {:.2f} s".format(i, s, e))
        self.q_excerpt_list.blockSignals(False)
        self.last_excerpt_row = -1
        self.rebuild_timeline()

    def change_excerpt(self, row):
        if row < 0 or self.model.mocap_data is None: return
        if self.q_solo.isChecked():
            self.rebuild_timeline()
        else:
            self.playhead = float(row * self.period)
            self.last_excerpt_row = row

    def rebuild_timeline_keep(self, *a):
        self.rebuild_timeline(keep_playhead=True)

    def rebuild_timeline(self, *a, keep_playhead=False):
        m = self.model
        if m.mocap_data is None or len(self.cluster_excerpts) == 0: return
        if self.q_solo.isChecked() and self.q_excerpt_list.currentRow() >= 0:
            sel = [m.starts[self.cluster_excerpts[self.q_excerpt_list.currentRow()]]]
        else:
            sel = [m.starts[i] for i in self.cluster_excerpts]
        rot, pos, k = m.render_timeline(sel, self.q_crossfade.value())
        pw, rw = m.to_world(rot, pos)
        if self.q_fix_root.isChecked():
            pos, pw = m.fix_root_at_origin(pos, pw)
        old = self.playhead
        self.frames_world, self.frame_excerpt = pw, k
        self.rot_frames, self.pos_frames, self.rot_world_frames = rot, pos, rw
        self.period = max(1, len(pw) // len(sel))
        self.playhead = min(old, len(pw) - 1) if keep_playhead else 0.0
        if not keep_playhead:
            self.last_excerpt_row = -1
        self.draw(int(self.playhead))

    # ---------- playback ----------
    def start(self):
        if self.frames_world is None: return
        self.playing = True
        self.clock = time.time()
        self.timer.start()

    def stop(self):
        self.playing = False
        self.timer.stop()

    def tick(self):
        now = time.time()
        dt, self.clock = now - self.clock, now
        if self.frames_world is None: return
        F = len(self.frames_world)
        self.playhead = (self.playhead + dt * self.model.fps * self.q_speed.value()) % F
        self.draw(int(self.playhead))

    def to_display(self, p):
        p = p * self.vs
        if self.cfg["up_axis"] == 1:
            return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
        return p

    def draw(self, f):
        if self.frames_world is None: return
        f = min(f, len(self.frames_world) - 1)
        p = self.to_display(self.frames_world[f])
        parents = list(self.model.mocap_data["skeleton"]["parents"])
        pairs = [(i, par) for i, par in enumerate(parents) if par != -1]
        lines = np.array([p[i] if j == 0 else p[par] for i, par in pairs for j in (0, 1)])
        r, g, b = cluster_color(self.current_label, self.model.cluster_count)
        self.bones.setData(pos=lines, color=(r, g, b, 0.95))
        self.joints.setData(pos=p, color=(r, g, b, 0.6))

        k = int(self.frame_excerpt[f])
        if not self.q_solo.isChecked() and k != self.last_excerpt_row:
            self.last_excerpt_row = k
            self.q_excerpt_list.blockSignals(True)
            self.q_excerpt_list.setCurrentRow(k)
            self.q_excerpt_list.blockSignals(False)
        self.info_label.setText("cluster {} | excerpt {}/{} | frame {}/{}".format(
            self.current_label, k + 1, len(self.cluster_excerpts), f, len(self.frames_world)))

        if self.osc_out is not None and self.q_osc.isChecked():
            out = self.osc_out
            out.send("/cluster/label", self.current_label)
            out.send("/cluster/excerpt", int(self.cluster_excerpts[min(k, len(self.cluster_excerpts) - 1)]))
            out.send("/mocap/0/joint/pos_local", self.pos_frames[f])
            out.send("/mocap/0/joint/rot_local", self.rot_frames[f])
            out.send("/mocap/0/joint/pos_world", self.frames_world[f])
            out.send("/mocap/0/joint/rot_world", self.rot_world_frames[f])

    # ---------- save ----------
    def save_clusters(self):
        if self.model.mocap_data is None or len(self.model.labels) == 0:
            self.q_status.setText("nothing to save.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for cluster files", "results")
        if not folder: return
        self.set_busy(True, "saving clusters as .{}...".format(self.model.file_type))
        w = ExportWorker(self.model, folder, self.q_crossfade.value(), self.q_fix_root.isChecked())
        w.progress.connect(self.on_save_progress)
        w.finished.connect(self.on_save_finished)
        w.failed.connect(self.on_save_failed)
        self.run_worker(w)

    def on_save_progress(self, done, total):
        self.q_status.setText("saving cluster {} of {}...".format(done, total))

    def on_save_finished(self, info):
        self.set_busy(False, "saved {} files to '{}'.".format(len(info["files"]), info["folder"]))

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
    model = MotionClusterModel(config)
    sender = None
    if motion_sender is not None:
        sender = motion_sender.OscSender({"ip": config["osc_ip"], "port": config["osc_port"]})
    gui = ClusterGui(model, config, sender)
    gui.show()
    first = os.path.join(config["mocap_dir"], config["file_name"])
    if config["file_name"] and os.path.exists(first):
        QTimer.singleShot(200, lambda: gui.start_analysis(first))
    app.exec_()