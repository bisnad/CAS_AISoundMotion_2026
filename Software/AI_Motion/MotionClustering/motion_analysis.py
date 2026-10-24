import numpy as np


from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

"""
smooth data
"""

# code from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth_1d(data_1d, window_length, window_type="hanning"):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is padded with zeros at both ends
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """
    
    if data_1d.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if data_1d.size < window_length:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_length<3:
        return data_1d

    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #print("data_1d s ", data_1d.shape)
    
    pad_left = np.flip(data_1d[:(window_length - 1)//2])
    pad_right = np.flip(data_1d[-(window_length - 1)//2:])
    
    data_padded=np.concatenate((pad_left, data_1d, pad_right))
    
    #print("data_padded s ", data_padded.shape)
    
    #print(len(s))
    if window_type == 'flat': #moving average
        window=np.ones(window_length,'d')
    else:
        window=eval('np.'+window_type+'(window_length)')

    data_smooth=np.convolve(window/window.sum(),data_padded,mode='valid')
    
    #print("data_smooth s ", data_smooth.shape)
    
    return data_smooth

def smooth(data, window_length, window_type="hanning"):
    """
    helper function for multidimensional data
    """
    
    orig_shape = data.shape

    data = np.reshape(data, (orig_shape[0], -1))
    data_dim = data.shape[1]
    data_smooth = [ smooth_1d(data[:, d], window_length, window_type) for d in range(data_dim) ]
    data_smooth = np.stack(data_smooth, axis=1)
    data_smooth = np.reshape(data_smooth, orig_shape)
    
    return data_smooth

def derivative(data, time_interval):
    """
    compute first order differences
    """
    
    deriv = (data[1:,...] - data[:-1,...]) / time_interval
    
    deriv = np.concatenate((np.zeros_like(deriv[0:1,...]), deriv))
    
    return deriv

def scalar(data, mode="average"):
    """
    compute scalar value
    mode: "average", "norm", "min", "max"
    """
    
    if not mode in ['average', 'norm', 'min', 'max']:
        raise ValueError("mode is on of 'average', 'norm', 'min', 'max'")
    

    if mode == "average":
        data_scalar = np.mean(data, axis=-1, keepdims=True)
    if mode == "norm":
        data_scalar = np.linalg.norm(data, axis=-1, keepdims=True)
    elif mode == "min":
        data_scalar = np.min(data, axis=-1, keepdims=True)
    elif mode == "max":
        data_scalar = np.max(data, axis=-1, keepdims=True)
        
    return data_scalar

def curvature(accelerations, velocities, scalar_velocities):
    """
    input:
        accelerations: T X J X D
        velocities: T X J X D
        scalar velocities: T X J X 1
    output:
        curv: T X 1
    """
    
    joint_dim = accelerations.shape[-1]
    
    accelerations = np.reshape(accelerations, (-1, joint_dim))
    velocities = np.reshape(velocities, (-1, joint_dim))
    scalar_velocities = np.reshape(scalar_velocities, (-1, 1))
    
    cross = np.cross(accelerations, velocities)
    cross_lengths = np.linalg.norm(cross, axis=-1)
    curv = cross_lengths / np.reshape(scalar_velocities, (-1))
    
    curv = np.reshape(curv, (-1, 1))
    
    return curv

def quantity_of_motion(scalar_velocities, joint_weights):
    """
    input: 
        scalar velocities: T X J X 1
        joint_weights: J
    output:
        qom: T X 1
    """
    
    seq_length = scalar_velocities.shape[0]
    
    scalar_velocities = np.squeeze(scalar_velocities, axis=-1)
    
    div = 1.0 / np.sum(joint_weights)
    
    qom = np.zeros(seq_length)
    
    for i in range(0, seq_length):
        qom[i] = np.sum(scalar_velocities[i] * joint_weights) * div
    
    qom = np.reshape(qom, (-1, 1))
    
    return qom

def bounding_box(positions):
    """
    input:
        positions: T x J x D
    output:
        bbox: T x 2 x D
    """

    seq_length = positions.shape[0]
    
    bbox_min = np.min(positions, axis=1)
    bbox_max = np.max(positions, axis=1)
 
    bbox = np.concatenate([bbox_min, bbox_max], axis=1)
    bbox = np.reshape(bbox, (seq_length, 2, -1))

    return bbox

def bounding_box_rt(positions):
    """
    input:
        positions: J x D
    output:
        bbox: 2 x D
    """

    
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox = np.concatenate([bbox_min, bbox_max], axis=0)

    return bbox

def bounding_sphere(positions):
    """
    input:
        positions: T x J x D
    output:
        sphere: T x D + 1
    """
    
    #print("positions s ", positions.shape)
    
    seq_length = positions.shape[0]
    joint_count = positions.shape[1]

    positions_mean = np.mean(positions, axis=1)
    bsphere_center = positions_mean
    
    #print("bsphere_center s ", bsphere_center.shape)
    
    joint_distances = positions - np.expand_dims(positions_mean, axis=1)
    
    #print("joint_distances s ", joint_distances.shape)
    
    joint_distances = np.linalg.norm(joint_distances, axis=2)
    
    #print("joint_distances 2 s ", joint_distances.shape)
    
    max_distance = np.max(joint_distances, axis=1)
    bsphere_radius = max_distance
    
    #print("bsphere_radius s ", bsphere_radius.shape)
    
    bsphere_radius = np.expand_dims(bsphere_radius, axis=1)
    
    bsphere = np.concatenate([bsphere_center, bsphere_radius], axis=1)
    
    #print("bsphere s ", bsphere.shape)

    return bsphere


def bounding_sphere_rt(positions):
    """
    input:
        positions: J x D
    output:
        sphere: 4
    """

    joint_count = positions.shape[0]

    positions_mean = np.mean(positions, axis=0)
    bsphere_center = positions_mean
    
    joint_distances = positions - positions_mean  
    joint_distances = np.linalg.norm(joint_distances, axis=1)
    
    max_distance = np.max(joint_distances)
    bsphere_radius = max_distance
    
    bsphere = np.array([bsphere_center[0], bsphere_center[1], bsphere_center[2], bsphere_radius])

    return bsphere
    
def filter(data, indices):
    """
    filter data
    indices: joint indices to keep
    """
    
    return np.copy(data[:, indices, :])
    
# non realtime calculation of weight effort: calculate weight efforts for each position of a window that slides across the sequence
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
def weight_effort(scalar_velocities, joint_weights, window_length):
    
    seq_length = scalar_velocities.shape[0]
    
    #print("scalar_velocities1 s ", scalar_velocities.shape)
    
    scalar_velocities = np.reshape(scalar_velocities, (seq_length, -1))
    
    #print("scalar_velocities2 s ", scalar_velocities.shape)

    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("joint_weights s ", joint_weights.shape)
    
    efforts_sum = np.sum(scalar_velocities * scalar_velocities * joint_weights, axis=1)
    
    #print("efforts_sum s ", efforts_sum.shape)
    
    pad_left = np.flip(efforts_sum[:(window_length - 1)//2 + 1])
    pad_right = np.flip(efforts_sum[-(window_length - 1)//2:])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
        
    efforts_padded=np.concatenate((pad_left, efforts_sum, pad_right))
    
    #print("efforts_padded s ", efforts_padded.shape)
    
    efforts = np.zeros_like(efforts_sum)
    
    for i in range(0, seq_length - window_length):
        
        excerpt = efforts_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        effort_max = np.max(excerpt)
        
        #print("i ", i, " effort_max s ", effort_max.shape)
        
        efforts[i] = effort_max
    
    efforts = np.expand_dims(efforts, axis=1)
    
    return efforts

# real-time calculation of weight effort: calculate single weight effort for entire sequence, without applying any windowing
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
# expected parameters
# scalar_velocities: time x joints
# joint_weights: joints 
def weight_effort_rt(scalar_velocities, joint_weights):
    
    #print("scalar_velocities s ", scalar_velocities.shape)
    #print("joint_weights s ", joint_weights.shape)
    seq_length = scalar_velocities.shape[0]
    
    scalar_velocities = np.reshape(scalar_velocities, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("joint_weights s ", joint_weights.shape)
    
    efforts_sum = np.sum(scalar_velocities * scalar_velocities * joint_weights, axis=1)

    #print("efforts_sum s ", efforts_sum.shape)
    
    effort_max = np.max(efforts_sum)
    effort_max = np.array([effort_max])

    return effort_max


# non realtime calculation of time effort: calculate time efforts for each position of a window that slides across the sequence
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
def time_effort(scalar_accelerations, joint_weights, window_length):
    
    seq_length = scalar_accelerations.shape[0]
    scalar_accelerations = np.reshape(scalar_accelerations, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_accelerations s ", scalar_accelerations.shape)

    pad_left = np.flip(scalar_accelerations[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_accelerations[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    accelerations_padded=np.concatenate((pad_left, scalar_accelerations, pad_right))
    
    #print("accelerations_padded s ", accelerations_padded.shape)
    
    time = np.zeros_like(scalar_accelerations)
    
    #print("time s ", time.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = accelerations_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        time[i] = np.sum(excerpt, axis=0) / window_length
        
    efforts = np.sum(time * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    #print("efforts s ", efforts.shape)
    
    return efforts

# real-time calculation of time effort: calculate single time effort for entire sequence, without applying any windowing
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
# expected parameters
# scalar_accelerations: time x joints
# joint_weights: joints 
def time_effort_rt(scalar_accelerations, joint_weights):
    
    seq_length = scalar_accelerations.shape[0]
    scalar_accelerations = np.reshape(scalar_accelerations, (seq_length, -1))
    
    time = np.sum(scalar_accelerations, axis=0) / seq_length
    effort = np.sum(time * joint_weights)
    
    effort = np.array([effort])
    
    return effort

# non realtime calculation of flow effort: calculate flow efforts for each position of a window that slides across the sequence
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
def flow_effort(scalar_jerks, joint_weights, window_length):
    
    seq_length = scalar_jerks.shape[0]
    scalar_jerks = np.reshape(scalar_jerks, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_accelerations s ", scalar_accelerations.shape)

    pad_left = np.flip(scalar_jerks[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_jerks[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    jerks_padded=np.concatenate((pad_left, scalar_jerks, pad_right))
    
    #print("accelerations_padded s ", accelerations_padded.shape)
    
    flow = np.zeros_like(scalar_jerks)
    
    #print("time s ", time.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = jerks_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        flow[i] = np.sum(excerpt, axis=0) / window_length
        
    efforts = np.sum(flow * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    #print("efforts s ", efforts.shape)
    
    return efforts

# real-time calculation of flow effort: calculate single flow effort for entire sequence, without applying any windowing
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
# expected parameters
# scalar_jerks: time x joints
# joint_weights: joints 
def flow_effort_rt(scalar_jerks, joint_weights):

    seq_length = scalar_jerks.shape[0]
    scalar_jerks = np.reshape(scalar_jerks, (seq_length, -1))
    
    flow = np.sum(scalar_jerks, axis=0) / seq_length
    effort = np.sum(flow * joint_weights)
    
    effort = np.array([effort])
    
    return effort

# non realtime calculation of flow effort: calculate flow efforts for each position of a window that slides across the sequence
# implementation based on Larboulette and Gibet Paper: A review of computable expressive descriptors of human motion
def space_effort(scalar_positions, joint_weights, window_length):
    
    seq_length = scalar_positions.shape[0]
    scalar_positions = np.reshape(scalar_positions, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_positions s ", scalar_positions.shape)
    
    pad_left = np.flip(scalar_positions[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_positions[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    positions_padded=np.concatenate((pad_left, scalar_positions, pad_right))
    
    #print("positions_padded s ", positions_padded.shape)
    
    space = np.zeros_like(scalar_positions)
    
    for i in range(0, seq_length - window_length):
        excerpt = positions_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        start_end_dist = np.absolute(excerpt[0, ...] - excerpt[-1, ...])
        
        #print("start_end_dist s ", start_end_dist.shape)
        
        cum_dist = np.sum(np.absolute(excerpt[1:, ...] - excerpt[:-1, ...]), axis=0)
        
        #print("cum_dist s ", cum_dist.shape)
        
        space[i] = cum_dist / start_end_dist
        
    efforts = np.sum(space * joint_weights, axis=1)
    
    return efforts
    
# non realtime calculation of flow effort: calculate flow efforts for each position of a window that slides across the sequence
# own implementation that uses deviation between overall direction and individual direction for each time step as basis for space effort
def space_effort_v2(positions, joint_weights, window_length):
    
    seq_length = positions.shape[0]
    joint_dim = positions.shape[-1]
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("positions s ", positions.shape)
    
    pad_left = np.flip(positions[:(window_length - 1)//2 + 1, :, :])
    pad_right = np.flip(positions[-(window_length - 1)//2:, :, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    positions_padded=np.concatenate((pad_left, positions, pad_right))
    
    #print("positions_padded s ", positions_padded.shape)
    
    space = np.zeros(positions.shape[:-1])
    
    #print("space s ", space.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = positions_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        start_end_diff = excerpt[-1, ...] - excerpt[0, ...]
        start_end_length = np.linalg.norm(start_end_diff, axis=-1) 
        start_end_dir = start_end_diff / np.expand_dims(start_end_length + 1e-8, axis=1)
        start_end_dir = np.repeat(np.expand_dims(start_end_dir, axis=0), window_length - 1, axis=0)
        start_end_dir = np.reshape(start_end_dir, (-1, joint_dim))
        
        #print("start_end_diff s ", start_end_diff.shape)
        #print("start_end_length s ", start_end_length.shape)
        #print("start_end_dir s ", start_end_dir.shape)
        
        step_diffs = excerpt[1:, ...] - excerpt[:-1, ...]

        #print("step_diffs s ", step_diffs.shape)
        
        step_lengths = np.linalg.norm(step_diffs, axis=-1) 
        
        #print("step_lengths s ", step_lengths.shape)
        
        step_dirs = step_diffs / np.expand_dims(step_lengths + 0.0001, axis=2)
        
        #print("step_dirs s ", step_dirs.shape)
        
        step_dirs = np.reshape(step_dirs, (-1, joint_dim))
        
        #print("step_dirs s ", step_dirs.shape)
        
        step_dots = np.einsum('ij,ij->i',step_dirs, start_end_dir)
        
        #print("step_dots s ", step_dots.shape)
        
        step_dots = np.reshape(step_dots, (window_length  - 1, -1))
        
        #print("step_dots s ", step_dots.shape)
        
        #step_dots = np.dot(step_dirs, start_end_dir)
        step_deviations = (step_dots * -1.0 + 1.0) * 0.5
        step_deviations *= step_lengths
        
        #print("step_deviations s ", step_deviations.shape)
        
        cum_deviation = np.sum(step_deviations, axis=0)
        
        #print("cum_deviation s ", cum_deviation.shape)
        
        space[i] = cum_deviation
    
    #print("space s ", space.shape)
        
    efforts = np.sum(space * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    return efforts

# real-time calculation of space effort: calculate space flow effort for entire sequence, without applying any windowing
# own implementation that uses deviation between overall direction and individual direction for each time step as basis for space effort
# expected parameters
# positions: time x joints x D
# joint_weights: joints
def space_effort_v2_rt(positions, joint_weights):
    
    #print("positions s ", positions.shape)
    #print("joint_weights s ", joint_weights.shape)
    
    seq_length = positions.shape[0]
    joint_dim = positions.shape[-1]
    
    #joint_weights = np.reshape(joint_weights, (1, -1))
 
    #print("i ", i, " excerpt s ", excerpt.shape)
    
    start_end_diff = positions[-1, ...] - positions[0, ...]
    
    #print("start_end_diff s ", start_end_diff.shape)
    
    start_end_length = np.linalg.norm(start_end_diff, axis=-1) 
    
    #print("start_end_length s ", start_end_length.shape)
    
    start_end_dir = start_end_diff / np.expand_dims(start_end_length + 1e-8, axis=1)
    start_end_dir = np.repeat(np.expand_dims(start_end_dir, axis=0), seq_length - 1, axis=0)
    start_end_dir = np.reshape(start_end_dir, (-1, joint_dim))
    
    #print("start_end_diff s ", start_end_diff.shape)
    #print("start_end_length s ", start_end_length.shape)
    #print("start_end_dir s ", start_end_dir.shape)
    
    step_diffs = positions[1:, ...] - positions[:-1, ...]

    #print("step_diffs s ", step_diffs.shape)
    
    step_lengths = np.linalg.norm(step_diffs, axis=-1) 
    
    #print("step_lengths s ", step_lengths.shape)
    
    step_dirs = step_diffs / np.expand_dims(step_lengths + 0.0001, axis=2)
    
    #print("step_dirs s ", step_dirs.shape)
    
    step_dirs = np.reshape(step_dirs, (-1, joint_dim))
    
    #print("step_dirs s ", step_dirs.shape)
    
    step_dots = np.einsum('ij,ij->i',step_dirs, start_end_dir)
    
    #print("step_dots s ", step_dots.shape)
    
    step_dots = np.reshape(step_dots, (seq_length  - 1, -1))
    
    #print("step_dots s ", step_dots.shape)
    
    #step_dots = np.dot(step_dirs, start_end_dir)
    step_deviations = (step_dots * -1.0 + 1.0) * 0.5
    step_deviations *= step_lengths
    
    #print("step_deviations s ", step_deviations.shape)
    
    space = np.sum(step_deviations, axis=0)
    
    #print("space s ", space.shape)
    
    effort = np.sum(space * joint_weights)
    effort = np.array([effort])
    
    return effort

def joint_distance(joint1, joint2):
    """
    calculate distance between two joints (time x joint_dimension)
    joint1: time x joint_dimension
    joint2: time x joint_dimension
    """
    return np.linalg.norm(joint1 - joint2, axis=1, keepdims=True)
    
def joint_level(joint, joint1, joint2):
    """
    calculates joint level with regards to two other joints
    level 1: joint is above joint1
    level 2: joint is between joint2 and joint1
    level 3: joint is below joint2
    
    joint: time x joint_dimension
    joint1: time x joint_dimension
    joint2: time x joint_dimension
    """
    
    above = joint[:, 1] > joint1[:, 1]
    below = joint[:, 1] < joint2[:, 1]
    between = np.logical_not(np.logical_or(above, below))
    level = above.astype(int) * 1 + between.astype(int) * 2 + below.astype(int) * 3
    level = np.expand_dims(level, axis=1)
    
    return level

def joint_travel_distance(joints):
    """
    Calculates the distance travel by joints
    
    joints: time x joint_count x joint_dimension or time x joint_dimension
    
    TODO: change so it works with an analysis window
    """

    directions = joints[1:, ...] - joints[:-1, ...]
    lengths = np.linalg.norm(directions, axis = 2 if len(joints.shape) == 3 else 1, keepdims=True)
    
    return lengths

def windowed_joint_travel_distance(joints, window_size):
    """
    Calculates the distance travel by joints
    joints: time x joint_count x joint_dimension or time x joint_dimension
    
    the distances are averaged within each window
    """
    
    # calc travel lengths for each time-step and joint
    directions = joints[1:, ...] - joints[:-1, ...]
    lengths = np.linalg.norm(directions, axis = 2 if len(joints.shape) == 3 else 1, keepdims=True)
    
    # prepend zero length
    lengths = np.concatenate([np.zeros([1] + list(lengths.shape[1:])), lengths], axis=0)
    
    windowed_lengths = []
    
    for tI in range(lengths.shape[0] - window_size):
        length_excerpt = lengths[tI:tI + window_size]
        windowed_value = np.mean(length_excerpt, axis=0, keepdims=True)
        windowed_lengths.append(windowed_value)
        
    windowed_lengths = np.concatenate(windowed_lengths, axis=0)

    return windowed_lengths

def joint_volume(joints):
    """
    Calculates the volume occupied by joints
    first by deriving the convex hull and then obtaining the volume from the hull
    joints can be provided either as a time series of poses or as a single pose
    
    joints: either time x joint_count x joint_dimension or joint_count x joint_dimension
    """
    
    joints = np.reshape(joints, [-1, joints.shape[-1]])
    
    volumes = []
    volumes.append(ConvexHull(joints).volume)
    
    volumes = np.array(volumes, dtype=np.float32)
    volumes = np.expand_dims(volumes, axis=1)
    
    return volumes

def windowed_joint_volume(joints, window_size):
    """
    same as joint_volume but calculates the volume for each time window
    """
    
    joints = np.reshape(joints, [-1, joints.shape[-1]])
    
    windowed_volumes = []
    
    for tI in range(joints.shape[0] - window_size):
        joint_excerpt = joints[tI:tI + window_size]
        
        windowed_volumes.append(ConvexHull(joint_excerpt).volume)

    windowed_volumes = np.array(windowed_volumes, dtype=np.float32)
    windowed_volumes = np.expand_dims(windowed_volumes, axis=1)

    return windowed_volumes

def joint_volumes(joints):
    """
    Calculates the volume occupied by joints
    differs from joint_volume in that this function calculates multiple volumes, one for each pose
    in a time series
    joints: either time x joint_count x joint_dimension or joint_count x joint_dimension
    """
    
    volumes = []
    
    if len(joints.shape) == 2: # single pose
        volumes.append(ConvexHull(joints).volume)
    else:
        for tI in range(joints.shape[0]):
            volumes.append(ConvexHull(joints[tI]).volume)
            
    volumes = np.array(volumes, dtype=np.float32)
    volumes = np.expand_dims(volumes, axis=1)
    
    return volumes

def windowed_min(time_series, window_size):
    
    time_series_shape = time_series.shape
    time_series = np.reshape(time_series, [time_series_shape[0], -1])
    
    values = []
    
    for tI in range(time_series_shape[0] - window_size):
        time_series_window = time_series[tI:tI + window_size]
        value = np.min(time_series_window, axis=0, keepdims=True)
        #value = np.reshape(value, time_series_shape[1:])
        values.append(value)
        
    values = np.concatenate(values, axis=0)

    return values

def windowed_max(time_series, window_size):
    
    time_series_shape = time_series.shape
    time_series = np.reshape(time_series, [time_series_shape[0], -1])

    values = []
    
    for tI in range(time_series_shape[0] - window_size):
        time_series_window = time_series[tI:tI + window_size]
        value = np.max(time_series_window, axis=0, keepdims=True)
        #value = np.reshape(value, time_series_shape[1:])
        values.append(value)
        
    values = np.concatenate(values, axis=0)

    return values

def windowed_mean(time_series, window_size):
    
    time_series_shape = time_series.shape
    time_series = np.reshape(time_series, [time_series_shape[0], -1])
    
    values = []
    
    for tI in range(time_series_shape[0] - window_size):
        time_series_window = time_series[tI:tI + window_size]
        value = np.mean(time_series_window, axis=0, keepdims=True)
        #value = np.reshape(value, time_series_shape[1:])
        values.append(value)
        
    values = np.concatenate(values, axis=0)
    
    return values

def windowed_std(time_series, window_size):
    
    time_series_shape = time_series.shape
    time_series = np.reshape(time_series, [time_series_shape[0], -1])

    values = []
    
    for tI in range(time_series_shape[0] - window_size):
        time_series_window = time_series[tI:tI + window_size]
        value = np.std(time_series_window, axis=0, keepdims=True)
        #value = np.reshape(value, time_series_shape[1:])
        values.append(value)
        
    values = np.concatenate(values, axis=0)

    return values
    