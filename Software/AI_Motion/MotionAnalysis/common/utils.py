import torch
import numpy as np
from matplotlib import pyplot as plt
import csv

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(image_file_name)

def get_skeleton_edge_list(skeleton):
    skel_edge_list = []

    skeleton_children = skeleton.children()
    for parent_joint_index in range(len(skeleton_children)):
        for child_joint_index in skeleton_children[parent_joint_index]:
            skel_edge_list.append([parent_joint_index, child_joint_index])
    
    return skel_edge_list

def get_equal_mix_max_positions(poses):

    min_pos = np.min(np.reshape(poses, (-1, 3)), axis=0)
    max_pos = np.max(np.reshape(poses, (-1, 3)), axis=0)
    min_pos = np.min(min_pos, axis=0)
    max_pos = np.max(max_pos, axis=0)
    
    _min_pos = [min_pos, min_pos, min_pos]
    _max_pos = [max_pos, max_pos, max_pos]

    min_pos = _min_pos
    max_pos = _max_pos

    return min_pos, max_pos

def create_ref_pose_sequence(ref_poses, start_frame, frame_count):
    _ref_poses = ref_poses[start_frame:start_frame + frame_count]

    return _ref_poses

def create_pred_pose_sequence(ref_poses, start_frame, frame_count, encoder, decoder, batch_size = 32):
    pred_poses = []
    
    for i in range(start_frame, start_frame + frame_count, batch_size):
        target_poses = []
    
        for bI in range(batch_size):
            target_poses.append(ref_poses[i + bI])
    
        target_poses = torch.stack(target_poses)
        _pred_poses = decoder.predict(encoder.predict(target_poses))
        pred_poses.append(_pred_poses)
    
    pred_poses = np.array(pred_poses)

    pred_poses = np.reshape(pred_poses, (-1, pred_poses.shape[-1]))
    pred_poses = pred_poses[:frame_count, :]

    return pred_poses

def create_2_pose_interpolation(ref_poses, frame1, frame2, interpolation_count, encoder, decoder):
    start_pose = ref_poses[frame1]
    end_pose = ref_poses[frame2]
    
    start_pose = np.expand_dims(start_pose, axis=0)
    end_pose = np.expand_dims(end_pose, axis=0)
    
    start_enc = encoder.predict(start_pose)
    end_enc = encoder.predict(end_pose)
    
    inter_poses = []

    for i in range(interpolation_count):
        inter_enc = start_enc + (end_enc - start_enc) * i / (interpolation_count - 1.0)
        inter_pose = decoder.predict(inter_enc)
        inter_poses.append(torch.squeeze(inter_pose, 0))

    inter_poses = np.array(inter_poses)

    return inter_poses

def create_3_pose_interpolation(ref_poses, frame1, frame2, frame3, interpolation_count, encoder, decoder):
    inter_poses = []

    ref_pose1 = ref_poses[frame1]
    ref_pose2 = ref_poses[frame2]
    ref_pose3 = ref_poses[frame3]

    ref_pose1 = np.expand_dims(ref_pose1, axis=0)
    ref_pose2 = np.expand_dims(ref_pose2, axis=0)
    ref_pose3 = np.expand_dims(ref_pose3, axis=0)

    ref_enc1 = encoder.predict(ref_pose1)
    ref_enc2 = encoder.predict(ref_pose2)
    ref_enc3 = encoder(ref_pose3, training=False)

    for hI in range(interpolation_count[0]):
        h_mix = hI / (interpolation_count[0] - 1)
        h_mix_enc12 = ref_enc1 * (1.0 - h_mix) + ref_enc2 * h_mix
    
        for vI in range(interpolation_count[1]):
            v_mix = vI / (interpolation_count[1] - 1)
            v_mix_enc13 = ref_enc1 * (1.0 - v_mix) + ref_enc3 * v_mix
            f_mix_enc = h_mix_enc12 + v_mix_enc13 - ref_enc1
        
            f_mix_pose = decoder.predict(f_mix_enc)
        
            inter_poses.append(f_mix_pose)

    inter_poses = np.array(inter_poses)
    
    return inter_poses

def create_pose_deviation(ref_poses, frame, latent_dim, deviation_range, deviation_count, encoder, decoder):
    deviation_poses = []

    ref_pose = ref_poses[frame]
    ref_pose = np.expand_dims(ref_pose, axis=0)
    ref_enc = encoder.predict(ref_pose)
    
    for lI in range(latent_dim):
        
        deviation_vec = np.zeros(shape=ref_enc.shape)
        
        for dI in range(-deviation_count, deviation_count + 1):

            deviation_vec[0, lI] = deviation_range * dI / (deviation_count - 1)
            deviation_pose = decoder.predict(ref_enc + deviation_vec)
            
            deviation_poses.append(deviation_pose)
    
    deviation_poses = np.array(deviation_poses)
    
    return deviation_poses