"""
Motion Analysis - Introductory Code Example (loads a mocap recording and a config file)
"""

# ------------------------
# Imports
# ------------------------

import json
import numpy as np
import scipy
from scipy.spatial import QhullError
import matplotlib.pyplot as plt

import motion_loader
import analysis as ma

# ----------------------------
# Motion Settings
# ----------------------------

motion_file_path = "data/mocap/Muriel_Take4_Blumen_Baile.fbx"        # .bvh, .fbx or .npz
motion_config_path = "configs/xsens_fbx_config.json"   # joint names, topology, weights, fps, scale
motion_topology_path = None                             # skeleton topology file, only needed for .npz recordings
skeleton_index = 0                                      # which skeleton to analyse if the recording holds several
motion_excerpt_sec = [ 20.0, 24.0 ]                       # None = use the whole recording

# ----------------------------
# Load Config
# ----------------------------

with open(motion_config_path) as json_data:
    motion_config = json.load(json_data)

joint_names = motion_config["joint_names"]
joint_weights = np.array(motion_config["joint_weights"], dtype=np.float64)
joint_dim = motion_config["pos_dim"]
pos_scale = motion_config["pos_scale"]            # converts file units to metres (0.01 = centimetres)
ground_coordinates = motion_config["ground_coordinates"]   # axes that span the floor, e.g. [0, 2] = x and z (y is up)

def joint_indices(names):
    return [ joint_names.index(name) for name in names ]

hip_index = joint_names.index(motion_config["joint_name_hip"])

# body parts used for the separate hull volumes. Some configs spell the keys
# "joint_name_rightbody" (singular), others "joint_names_rightbody", both are accepted.
body_parts = {
    "full body":  list(range(len(joint_names))),
    "upper body": joint_indices(motion_config["joint_names_upperbody"]),
    "lower body": joint_indices(motion_config["joint_names_lowerbody"]),
    "right body": joint_indices(motion_config.get("joint_names_rightbody", motion_config.get("joint_name_rightbody"))),
    "left body":  joint_indices(motion_config.get("joint_names_leftbody", motion_config.get("joint_name_leftbody"))),
}

assert len(joint_weights) == len(joint_names), "joint_weights and joint_names must have the same length"

# ----------------------------
# Load Mocap Recording
# ----------------------------

# The loading itself (bvh, fbx, npz) is done by motion_loader.py. It returns world-space joint positions
# [time, joints, 3] in the units of the file, with the joints ordered like motion_config["joint_names"].

motion_loader.config["file_name"] = motion_file_path
motion_loader.config["topology_file_name"] = motion_topology_path if motion_topology_path else ""
motion_loader.config["fps"] = motion_config["fps"]

loader = motion_loader.MotionLoader(motion_loader.config)

if loader.get_skeleton_count() > 1:
    print("note: the recording contains {} skeletons, using skeleton {}".format(loader.get_skeleton_count(), skeleton_index))

if motion_excerpt_sec is not None:
    excerpt_start, excerpt_end = motion_excerpt_sec
else:
    excerpt_start, excerpt_end = 0.0, None

motion_positions, motion_fps = loader.get_positions(
    skeleton_index=skeleton_index, joint_names=joint_names, start_sec=excerpt_start, end_sec=excerpt_end
)

if abs(motion_fps - motion_config["fps"]) > 1e-3:
    print("note: the recording runs at {:.2f} fps but the config says {} fps. Using the recording's rate.".format(motion_fps, motion_config["fps"]))

motion_positions = motion_positions[:, :, :joint_dim] * pos_scale      # now in metres
frame_count = motion_positions.shape[0]
time_sec = excerpt_start + np.arange(frame_count) / motion_fps

print("loaded {} frames, {} joints, {:.1f} fps".format(frame_count, motion_positions.shape[1], motion_fps))

# ----------------------------
# Plot Motion
# ----------------------------

plot_joint_names = [ "Hips", "Head", "LeftHand", "RightHand" ]

plt.title('Joint Heights (y)')
for name in plot_joint_names:
    plt.plot(time_sec, motion_positions[:, joint_names.index(name), 1], label=name)
plt.xlabel('Time (s)')
plt.legend()
plt.show()

plt.title('Hip Trajectory on Ground Plane')
plt.plot(motion_positions[:, hip_index, ground_coordinates[0]], motion_positions[:, hip_index, ground_coordinates[1]])
plt.axis('equal')
plt.show()

# ----------------------------
# Extract Motion Buffer
# ----------------------------

# Real-time analysis works on a short sliding window of recent poses (a ring buffer),
# just like the audio buffer, here 25 frames.

motion_buffer_size = 25
motion_buffer = motion_positions[:motion_buffer_size]

plt.title('Motion Buffer (hip x, y, z)')
plt.plot(motion_buffer[:, hip_index, :])
plt.show()

# ----------------------------
# Smooth Positions
# ----------------------------

# ma.smooth convolves every coordinate with a Hanning window (an odd number of frames wide).
# Used to suppress sensor jitter before taking derivatives. It plays the role the amplitude envelope
# (window) plays for audio. (The real-time pipeline uses cheaper exponential smoothing instead.)

smooth_window_length = 9

motion_positions_smooth = ma.smooth(motion_positions, smooth_window_length, "hanning")

plt.title('Right Hand Height: Raw vs Smoothed')
right_hand_index = joint_names.index("RightHand")
plt.plot(time_sec, motion_positions[:, right_hand_index, 1], label='raw')
plt.plot(time_sec, motion_positions_smooth[:, right_hand_index, 1], label='smoothed')
plt.legend()
plt.show()

# ----------------------------
# Calculate Motion Features
# ----------------------------

def plot_motion_and_feature(title, positions, feature, fps, joint_index=hip_index):
    positions = np.asarray(positions)
    feature = np.asarray(feature)

    duration = len(positions) / fps
    time_pos = np.linspace(0, duration, num=len(positions))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 7), sharex="col",
        gridspec_kw={"height_ratios": [1, 2], "width_ratios": [1, 0.03], "wspace": 0.02}
    )
    (ax_pos, ax_pos_spacer), (ax_feat, ax_cbar) = axes
    ax_pos_spacer.axis("off")
    fig.suptitle(title)

    ax_pos.plot(time_pos, positions[:, joint_index, :], linewidth=0.8)
    ax_pos.set(ylabel="Position (m), {}".format(joint_names[joint_index]))

    if feature.ndim == 1:
        time_feat = np.arange(len(feature)) / fps
        ax_feat.plot(time_feat, feature, color="darkorange")
        ax_feat.set(xlabel="Time (s)", ylabel="Feature value")
        ax_cbar.axis("off")
    else:
        img = ax_feat.imshow(
            feature.T, origin="lower", aspect="auto", cmap="magma",
            extent=[0, len(feature) / fps, 0, feature.shape[1]]
        )
        ax_feat.set(xlabel="Time (s)", ylabel="Bin")
        fig.colorbar(img, cax=ax_cbar)

    ax_pos.set_xlim(0, duration)
    fig.tight_layout()
    plt.show()

# derivatives
# ma.derivative: first differences divided by the time step (the first frame is set to 0).
# Each derivative is smoothed again before the next one is taken, to keep noise from growing.
# Velocity relates to speed of movement, acceleration to changes of speed, jerk to abruptness (smooth vs. jerky).

time_step = 1.0 / motion_fps

velocity = ma.derivative(motion_positions_smooth, time_step)
acceleration = ma.derivative(ma.smooth(velocity, smooth_window_length), time_step)
jerk = ma.derivative(ma.smooth(acceleration, smooth_window_length), time_step)

# scalar values
# ma.scalar with mode "norm": Euclidean norm over x, y, z. Direction is dropped, only magnitude remains,
# one value per joint [time, joints, 1].

velocity_scalar = ma.scalar(velocity, mode="norm")
acceleration_scalar = ma.scalar(acceleration, mode="norm")
jerk_scalar = ma.scalar(jerk, mode="norm")

plot_motion_and_feature("velocity (scalar per joint)", motion_positions_smooth, velocity_scalar[:, :, 0], motion_fps)
plot_motion_and_feature("acceleration (scalar per joint)", motion_positions_smooth, acceleration_scalar[:, :, 0], motion_fps)
plot_motion_and_feature("jerk (scalar per joint)", motion_positions_smooth, jerk_scalar[:, :, 0], motion_fps)

# quantity of motion (QoM)
# ma.quantity_of_motion: weighted average of the joint speeds, using the joint weights from the config.
# Relates to the overall amount of movement, similar to RMS for loudness.

qom = ma.quantity_of_motion(velocity_scalar, joint_weights).flatten()

plot_motion_and_feature("quantity of motion", motion_positions_smooth, qom, motion_fps)

# bounding box
# ma.bounding_box: minimum and maximum of all joint positions along each axis (min x, y, z, max x, y, z).
# Relates to the space the body occupies, e.g. expansion vs. contraction.

bbox = ma.bounding_box(motion_positions_smooth).reshape(frame_count, -1)

plot_motion_and_feature("bounding box", motion_positions_smooth, bbox, motion_fps)

# bounding sphere
# ma.bounding_sphere: centre of mass of the joints plus the distance to the furthest joint (centre x, y, z, radius).
# Relates to how far the limbs reach away from the body's centre.

bsphere = ma.bounding_sphere(motion_positions_smooth)

plot_motion_and_feature("bounding sphere", motion_positions_smooth, bsphere, motion_fps)

# convex hull volume
# ma.filter picks the joints of a body part, ma.joint_volumes returns the volume of the smallest convex shape
# that contains them, for every pose. Relates to the expansion of the body (or of one part of it) in 3D space.

hull_volumes = np.concatenate([
    ma.joint_volumes(ma.filter(motion_positions_smooth, indices)) for indices in body_parts.values()
], axis=1)

plot_motion_and_feature("convex hull volumes ({})".format(", ".join(body_parts.keys())), motion_positions_smooth, hull_volumes, motion_fps)

# hand distance and hand level
# ma.joint_distance: distance between two joints. ma.joint_level: 1 = joint above joint1, 2 = between joint1 and joint2,
# 3 = below joint2. Here: the distance between both hands, and the right hand relative to the head and the hips.
# Relate to open vs. closed gestures and to reaching up or down.

left_hand_index = joint_names.index("LeftHand")
head_index = joint_names.index("Head")

hand_distance = ma.joint_distance(motion_positions_smooth[:, left_hand_index], motion_positions_smooth[:, right_hand_index]).flatten()
hand_level = ma.joint_level(motion_positions_smooth[:, right_hand_index], motion_positions_smooth[:, head_index], motion_positions_smooth[:, hip_index]).flatten()

plot_motion_and_feature("hand distance", motion_positions_smooth, hand_distance, motion_fps, joint_index=left_hand_index)
plot_motion_and_feature("right hand level (1 above head, 2 between head and hips, 3 below hips)", motion_positions_smooth, hand_level, motion_fps, joint_index=right_hand_index)

# Laban effort factors (sliding window over the motion buffer)
# The real-time functions of motion_analysis (_rt) are applied to the last motion_buffer_size frames, exactly as in
# the real-time pipeline. Each returns a single value per window.
# weight effort: strongest weighted energy in the window (strong vs. light movement).
# time effort: weighted mean acceleration (sudden vs. sustained).
# flow effort: weighted mean jerk (bound vs. free flowing).
# space effort: how much the path deviates from a straight line (direct vs. indirect).

weight_effort, time_effort, flow_effort, space_effort = [], [], [], []

for t in range(motion_buffer_size, frame_count + 1):
    window = slice(t - motion_buffer_size, t)
    weight_effort.append(ma.weight_effort_rt(velocity_scalar[window], joint_weights)[0])
    time_effort.append(ma.time_effort_rt(acceleration_scalar[window], joint_weights)[0])
    flow_effort.append(ma.flow_effort_rt(jerk_scalar[window], joint_weights)[0])
    space_effort.append(ma.space_effort_v2_rt(motion_positions_smooth[window], joint_weights)[0])

efforts = np.nan_to_num(np.stack([ weight_effort, time_effort, flow_effort, space_effort ], axis=1))    # a joint that does not move at all gives NaN in space effort
efforts_positions = motion_positions_smooth[motion_buffer_size - 1:]

for i, name in enumerate([ "weight effort", "time effort", "flow effort", "space effort" ]):
    plot_motion_and_feature(name, efforts_positions, efforts[:, i], motion_fps)

# travel distance
# ma.windowed_joint_travel_distance: path length the hip travelled on the floor (ground_coordinates from the config)
# inside a sliding window. Relates to how much the person moves through the room.

ground_trajectory = motion_positions_smooth[:, hip_index, ground_coordinates]

travel_distance = ma.windowed_joint_travel_distance(ground_trajectory, motion_buffer_size).flatten()

plot_motion_and_feature("travel distance", motion_positions_smooth[:len(travel_distance)], travel_distance, motion_fps)

# area covered
# ma.joint_volume of the hip's floor trajectory inside a sliding window gives the area of its convex hull.
# Relates to how much floor space was used. A window in which the hip moves along a straight line (or not at all)
# has no area, ConvexHull raises an error then, so those windows are set to 0.

def window_area(points):
    try:
        return ma.joint_volume(points)[0, 0]
    except QhullError:
        return 0.0

area_covered = np.array([
    window_area(ground_trajectory[t - motion_buffer_size:t])
    for t in range(motion_buffer_size, len(ground_trajectory) + 1)
])

plot_motion_and_feature("area covered", efforts_positions, area_covered, motion_fps)
