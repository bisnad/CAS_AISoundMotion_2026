"""
Motion Nearest Neighbors - Introductory Code Example
"""

"""
Imports
"""

import os, json
import numpy as np

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer
import motion_analysis as ma

"""
Mocap Settings
"""

mocap_file = "data/mocap/polytopia_fullbody_take2.fbx"   # .fbx or .bvh
mocap_fps = 50
mocap_frame_range = [200, 6400]       # frames used for the search
mocap_joint_weight_file = "configs/joint_weights_qualisys_fbx.json"
mocap_body_weight = 60

"""
Analysis Settings
"""

smooth_length = 25                    # frames
excerpt_length = 80                   # frames
excerpt_offset = 40                   # frames between excerpt starts
motion_feature_names = ["bsphere", "space_effort"]

"""
Export Settings
"""

export_formats = ["gif", "fbx", "bvh", "npz"]   # any of these
export_path = "results/mocap"
gif_max_frames = 300                  # rendering a long gif takes a while

"""
Load Mocap
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

if mocap_file.lower().endswith(".bvh"):
    mocap_data = mocap_tools.bvh_to_mocap(bvh_tools.load(mocap_file))
    mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(
        mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
else:
    mocap_data = mocap_tools.fbx_to_mocap(fbx_tools.load(mocap_file))[0]   # first skeleton only
    mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(
        mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

# the root joint will stay in place in the exported motion
mocap_data["skeleton"]["offsets"][0, 0] = 0.0
mocap_data["skeleton"]["offsets"][0, 2] = 0.0

offsets = mocap_data["skeleton"]["offsets"]
children = mocap_data["skeleton"]["children"]

pos_world, _ = mocap_tools.local_to_world(
    mocap_data["motion"]["rot_local"], mocap_data["motion"]["pos_local"], mocap_data["skeleton"])
pos_world = pos_world / 100.0         # centimeters -> meters
rot_local = mocap_data["motion"]["rot_local"]
frame_count = rot_local.shape[0]
joint_count = rot_local.shape[1]

with open(mocap_joint_weight_file) as fh:
    joint_weights = np.array(json.load(fh)["jointWeights"]) * mocap_body_weight / 100.0

"""
Calculate Motion Features
"""

dt = 1.0 / mocap_fps

pos_smooth = ma.smooth(pos_world, smooth_length)
vel = ma.smooth(ma.derivative(pos_smooth, dt), smooth_length)
vel_scalar = ma.scalar(vel, "norm")
accel = ma.smooth(ma.derivative(vel, dt), smooth_length)
accel_scalar = ma.scalar(accel, "norm")
jerk = ma.smooth(ma.derivative(accel, dt), smooth_length)
jerk_scalar = ma.scalar(jerk, "norm")

motion_features = {}

if "qom" in motion_feature_names:
    motion_features["qom"] = ma.quantity_of_motion(vel_scalar, joint_weights)
if "bsphere" in motion_feature_names:
    motion_features["bsphere"] = ma.bounding_sphere(pos_world)
if "weight_effort" in motion_feature_names:
    motion_features["weight_effort"] = ma.weight_effort(vel_scalar, joint_weights, smooth_length)
if "space_effort" in motion_feature_names:
    motion_features["space_effort"] = ma.space_effort_v2(pos_world, joint_weights, smooth_length)
if "time_effort" in motion_feature_names:
    motion_features["time_effort"] = ma.time_effort(accel_scalar, joint_weights, smooth_length)
if "flow_effort" in motion_feature_names:
    motion_features["flow_effort"] = ma.flow_effort(jerk_scalar, joint_weights, smooth_length)

"""
Normalise and Combine Motion Features
"""

features_proc = []
for name in motion_feature_names:
    feature = np.reshape(motion_features[name], (frame_count, -1))
    feature_norm = (feature - np.mean(feature, axis=0)) / (np.std(feature, axis=0) + 1e-8)   # z-score
    features_proc.append(feature_norm)

features_proc = np.concatenate(features_proc, axis=1)

"""
Create Mocap Excerpts
"""

frame_end = min(mocap_frame_range[1], frame_count)
excerpt_starts = np.arange(mocap_frame_range[0], frame_end - excerpt_length, excerpt_offset)

# one feature vector per excerpt: the average of the frame-wise features
excerpt_features = np.stack([np.mean(features_proc[s:s + excerpt_length], axis=0) for s in excerpt_starts])
excerpt_rotations = np.stack([rot_local[s:s + excerpt_length] for s in excerpt_starts])

excerpt_count = excerpt_features.shape[0]
print("excerpts ", excerpt_features.shape)

"""
Prepare Motion Blending
"""

def slerp(q0, q1, w):
    """spherical interpolation between quaternions (w, x, y, z), works on whole arrays"""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sin_theta = np.sin(theta)
    safe = sin_theta > 1e-6          # nearly identical rotations: fall back to linear interpolation
    denom = np.where(safe, sin_theta, 1.0)
    s0 = np.where(safe, np.sin((1.0 - w) * theta) / denom, 1.0 - w)
    s1 = np.where(safe, np.sin(w * theta) / denom, w)
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

# envelope: blend in over the overlap, then keep the new excerpt at full weight
overlap = excerpt_length - excerpt_offset
envelope = np.concatenate([np.linspace(0.0, 1.0, overlap), np.ones(excerpt_offset)])
envelope = envelope[:, None, None]

# empty motion (identity rotations) that the excerpts are blended into
gen_frame_count = excerpt_length + (excerpt_count - 1) * excerpt_offset
gen_motion = np.zeros((gen_frame_count, joint_count, 4))
gen_motion[..., 0] = 1.0

"""
Find Nearest Neighbors
"""

excerpt_used = np.zeros(excerpt_count, dtype=bool)
nn_current_index = 0                  # begin the search with the first excerpt

for step in range(excerpt_count):

    if step > 0:
        # distance from the current excerpt to all others, already used excerpts are excluded
        nn_distances = np.linalg.norm(excerpt_features - excerpt_features[nn_current_index], axis=1)
        nn_distances[excerpt_used] = np.inf
        nn_current_index = int(np.argmin(nn_distances))

    excerpt_used[nn_current_index] = True

    # blend the nearest excerpt into the generated motion
    s = step * excerpt_offset
    w = np.ones_like(envelope) if step == 0 else envelope
    gen_motion[s:s + excerpt_length] = slerp(gen_motion[s:s + excerpt_length],
                                             excerpt_rotations[nn_current_index], w)

    print("step ", step + 1, " of ", excerpt_count, " excerpt ", nn_current_index)

"""
Export Generated Motion
"""

def export_motion(rot_local, file_base):
    frame_count = rot_local.shape[0]
    pos_local = np.repeat(offsets[None], frame_count, axis=0)       # root stays in place
    pos_world, rot_world = mocap_tools.local_to_world(rot_local, pos_local, mocap_data["skeleton"])

    out = {
        "frame_rate": mocap_fps,
        "rot_sequence": mocap_data["rot_sequence"],
        "skeleton": mocap_data["skeleton"],
        "motion": {
            "pos_local": pos_local, "rot_local": rot_local,
            "pos_world": pos_world, "rot_world": rot_world,
            "rot_local_euler": mocap_tools.quat_to_euler(rot_local, mocap_data["rot_sequence"]),
        },
    }

    if "gif" in export_formats:
        edge_list = [[p, c] for p in range(len(children)) for c in children[p]]
        frames = pos_world[:gif_max_frames]
        view_min, view_max = utils.get_equal_mix_max_positions(frames)
        images = PoseRenderer(edge_list).create_pose_images(frames, view_min, view_max, 90.0, -90.0, 1.0, 4.0, 4.0)
        images[0].save(file_base + ".gif", save_all=True, append_images=images[1:],
                       optimize=False, duration=int(1000 / mocap_fps), loop=0)
    if "fbx" in export_formats:
        fbx_tools.write(mocap_tools.mocap_to_fbx([out]), file_base + ".fbx")
    if "npz" in export_formats:
        np.savez_compressed(file_base + ".npz", **mocap_tools.mocap_to_npz([out]))
        skeleton = mocap_data["skeleton"]
        topology = {"parents": [int(p) for p in skeleton["parents"]],
                    "children": [[int(c) for c in ch] for ch in skeleton["children"]],
                    "joints": [str(j) for j in skeleton["joints"]]}
        with open(file_base + "_topology.json", "w") as fh:
            json.dump(topology, fh, indent=2)
    if "bvh" in export_formats:
        out["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(rot_local, mocap_data["rot_sequence"])
        bvh_tools.write(mocap_tools.mocap_to_bvh(out), file_base + ".bvh")

os.makedirs(export_path, exist_ok=True)
export_motion(gen_motion, export_path + "/nearest_neighbors")
print("wrote ", export_path + "/nearest_neighbors", " (", gen_motion.shape[0] / mocap_fps, " s)")