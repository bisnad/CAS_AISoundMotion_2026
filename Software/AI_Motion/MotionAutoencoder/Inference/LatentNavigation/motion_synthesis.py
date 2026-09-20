import torch
import numpy as np
import torch.nn.functional as nnF
from scipy.spatial.transform import Rotation as R

from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

config = {
    "skeleton": None,
    "model": None,
    "device": "cuda",
    "seq_window_length": 64,
    "seq_window_offset": 1,
    "root_trajectory": True,
    "root_pos_mean": None,
    "root_pos_std": None
}

class MotionSynthesis:
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model"]
        self.device = torch.device(config["device"])
        self.seq_window_length = int(config["seq_window_length"])
        self.seq_window_offset = int(config["seq_window_offset"])
        self.root_trajectory = bool(config["root_trajectory"])
        self.root_pos_mean = config["root_pos_mean"]
        self.root_pos_std = config["root_pos_std"]

        if self.skeleton is None:
            raise ValueError("config['skeleton'] must not be None")
        if self.model is None:
            raise ValueError("config['model'] must not be None")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.seq_window_offset = max(1, min(self.seq_window_offset, self.seq_window_length))
        self.seq_window_overlap = self.seq_window_length - self.seq_window_offset

        self.joint_offsets = np.asarray(self.skeleton["offsets"], dtype=np.float32)
        self.joint_parents = self.skeleton["parents"]
        self.joint_children = self.skeleton["children"]

        self.joint_count = self.joint_offsets.shape[0]
        self.joint_dim = 6
        self.pose_dim = self.joint_count * self.joint_dim
        self.input_dim = self.pose_dim + (3 if self.root_trajectory else 0)

        if self.root_trajectory:
            if self.root_pos_mean is None:
                self.root_pos_mean = np.zeros((1, 1, 3), dtype=np.float32)
            if self.root_pos_std is None:
                self.root_pos_std = np.ones((1, 1, 3), dtype=np.float32)

            self.root_pos_mean = torch.as_tensor(self.root_pos_mean, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
            self.root_pos_std = torch.as_tensor(self.root_pos_std, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
            self.root_pos_std = torch.where(self.root_pos_std == 0.0, torch.ones_like(self.root_pos_std), self.root_pos_std)

        self._create_edge_list()

        self.encodings = []
        self.encoding_index = 0

        self.latent_dim, self.latent_steps = self._infer_latent_shape()

        # Initialize with zeros
        self.gen_seq = torch.zeros((self.seq_window_length, self.input_dim), dtype=torch.float32, device=self.device)
        
        # Calculate where rotation dimensions start
        rot_start_idx = 3 if self.root_trajectory else 0
        
        # Initialize all joints to an identity 6D rotation: [1, 0, 0, 0, 1, 0]
        for j in range(self.joint_count):
            idx = rot_start_idx + j * 6
            self.gen_seq[:, idx] = 1.0      # X-axis x-component
            self.gen_seq[:, idx + 4] = 1.0  # Y-axis y-component

        self.gen_seq_window = None

        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        self.synth_pose_lrot = None

        self.seq_update_index = 0

    def _create_edge_list(self):
        self.edge_list = []
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])

    def _infer_latent_shape(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_dim, self.seq_window_length, device=self.device)
            mu, _ = self.model.encode(dummy)
            return int(mu.shape[1]), int(mu.shape[2])

    def addEncoding(self, encoding):
        """Adds an encoding directly mapped from the UI."""
        self.encodings.append(encoding)

    def removeEncoding(self, idx):
        if idx < len(self.encodings):
            self.encodings.pop(idx)

    def clearEncodings(self):
        self.encodings.clear()
        self.encoding_index = 0

    def _postprocess_decoded_window(self, decoded_window):
        decoded_window = decoded_window.permute(0, 2, 1).squeeze(0)

        if self.root_trajectory:
            root = decoded_window[:, :3]
            root = (root * self.root_pos_std.squeeze(0)) + self.root_pos_mean.squeeze(0)
            rot = decoded_window[:, 3:].reshape(self.seq_window_length, self.joint_count, 6)
            rot = rot_to.orthogonalize_r6d(rot).reshape(self.seq_window_length, -1)
            decoded_window = torch.cat((root, rot), dim=-1)
        else:
            rot = decoded_window.reshape(self.seq_window_length, self.joint_count, 6)
            rot = rot_to.orthogonalize_r6d(rot).reshape(self.seq_window_length, -1)
            decoded_window = rot

        return decoded_window

    def getEdgeList(self):
        return self.edge_list

    def _generate_window(self):
        self.gen_seq_window = None

        if len(self.encodings) == 0:
            return
        
        self.encoding_index = min(self.encoding_index, len(self.encodings) - 1)
        encoding = self.encodings[self.encoding_index]

        # The legacy GUI adds a batch dimension, so encoding might already be [1, C, L].
        # We ensure it is exactly 3D [batch, channels, length] for the CNN decoder.
        while encoding.ndim < 3:
            encoding = encoding.unsqueeze(0)
        while encoding.ndim > 3:
            encoding = encoding.squeeze(0)

        # Decode the tensor
        with torch.no_grad():
            decoded = self.model.decode(encoding)

        self.gen_seq_window = self._postprocess_decoded_window(decoded)

        # Increment encoding index cyclically
        self.encoding_index += 1
        if self.encoding_index >= len(self.encodings):
            self.encoding_index = 0

    def _blend(self):
        if self.gen_seq_window is None:
            return
            
        self.gen_seq = torch.roll(self.gen_seq, shifts=-self.seq_window_offset, dims=0)
        
        if self.seq_window_overlap <= 0:
            self.gen_seq = self.gen_seq_window.clone()
            return
            
        # Match training script: Hanning window overlapping 
        # using matrix interpolation matching decode_sequence_encodings
        
        old_rot = self.gen_seq[:self.seq_window_overlap].reshape(self.seq_window_overlap, self.joint_count, 6)
        new_rot = self.gen_seq_window[:self.seq_window_overlap].reshape(self.seq_window_overlap, self.joint_count, 6)
        
        old_mat = rot_to.r6d_to_mat(old_rot)
        new_mat = rot_to.r6d_to_mat(new_rot)
        
        alpha = torch.linspace(0.0, 1.0, self.seq_window_overlap, device=self.device).view(-1, 1, 1, 1)
        
        blend_mat = old_mat * (1.0 - alpha) + new_mat * alpha
        
        x_avg = blend_mat[..., 0]
        y_avg = blend_mat[..., 1]
        blend_rot = torch.cat([x_avg, y_avg], dim=-1).reshape(self.seq_window_overlap, -1)
        
        if self.root_trajectory:
            old_root = self.gen_seq[:self.seq_window_overlap, :3]
            new_root = self.gen_seq_window[:self.seq_window_overlap, :3]
            alpha_root = alpha.view(-1, 1)
            blend_root = old_root * (1.0 - alpha_root) + new_root * alpha_root
            blend_seq = torch.cat([blend_root, blend_rot], dim=-1)
        else:
            blend_seq = blend_rot
            
        self.gen_seq[:self.seq_window_overlap] = blend_seq
        self.gen_seq[self.seq_window_overlap:] = self.gen_seq_window[self.seq_window_overlap:].clone()

    def update(self):
        pred_pose = self.gen_seq[self.seq_update_index]

        if self.root_trajectory:
            root_trajectory = pred_pose[:3].reshape(1, 1, 3)
            joint_rot_6d = pred_pose[3:].reshape(1, 1, self.joint_count, 6)
        else:
            root_trajectory = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.device)
            joint_rot_6d = pred_pose.reshape(1, 1, self.joint_count, 6)

        joint_pos_world, joint_rot_6d_world = self._forward_kinematics(joint_rot_6d, root_trajectory)

        # Process joint world positions
        self.synth_pose_wpos = joint_pos_world.detach().cpu().numpy().reshape(self.joint_count, 3)

        # Process joint world rotations (Matrix -> Quaternion) using NumPy utils
        self.synth_pose_wrot = rot_np.r6d_to_quat(
            joint_rot_6d_world.detach().cpu().numpy()
        ).reshape(self.joint_count, 4)

        # Process local rotations
        self.synth_pose_lrot = rot_np.r6d_to_quat(joint_rot_6d.squeeze().detach().cpu().numpy()).reshape(self.joint_count, 4)

        self.seq_update_index += 1

        if self.seq_update_index >= self.seq_window_offset:
            self._generate_window()
            self._blend()
            self.seq_update_index = 0

    def _forward_kinematics(self, rotations_6d, root_positions):
        rotation_matrices = rot_to.r6d_to_mat(rotations_6d)

        offsets = torch.as_tensor(self.joint_offsets, dtype=torch.float32, device=self.device)
        # Reshape to (1, 1, Joints, 3, 1) and expand
        expanded_offsets = offsets.view(1, 1, self.joint_count, 3, 1).expand(
            rotations_6d.shape[0], rotations_6d.shape[1], self.joint_count, 3, 1
        )

        positions_world = [None] * self.joint_count
        rotations_world = [None] * self.joint_count

        # Lazily calculate and cache safe execution order for NPZ topologies
        if not hasattr(self, 'execution_order'):
            self.execution_order = []
            for jI in range(self.joint_count):
                if self.joint_parents[jI] == -1:
                    self.execution_order.append(jI)
            for jI in self.execution_order:
                for cI in range(self.joint_count):
                    if self.joint_parents[cI] == jI:
                        self.execution_order.append(cI)

        # Iterate using the safe topological order
        for jI in self.execution_order:
            if self.joint_parents[jI] == -1:
                # Root Joint
                positions_world[jI] = root_positions
                rotations_world[jI] = rotation_matrices[:, :, jI]
            else:
                # Child Joints
                parent_idx = self.joint_parents[jI]
                parent_rot = rotations_world[parent_idx]
                parent_pos = positions_world[parent_idx]

                local_offset = expanded_offsets[:, :, jI]
                
                # Restored Column-Major Matrix Multiplication for Vectors
                rotated_offset = torch.matmul(parent_rot, local_offset).squeeze(-1)
                positions_world[jI] = rotated_offset + parent_pos

                if len(self.joint_children[jI]) > 0:
                    local_rot = rotation_matrices[:, :, jI]
                    # Restored Column-Major Matrix Multiplication for Rotations
                    new_world_rot = torch.matmul(parent_rot, local_rot)
                    rotations_world[jI] = new_world_rot
                else:
                    rotations_world[jI] = parent_rot

        # Stack into (Batch, Time, Joints, 3)
        pos_world_tensor = torch.stack(positions_world, dim=2)
        
        # Stack into (Batch, Time, Joints, 3, 3)
        rot_world_matrix_tensor = torch.stack(rotations_world, dim=2)
        
        # Extract the 6D representation for world rotations
        x_world = rot_world_matrix_tensor[..., :, 0]
        y_world = rot_world_matrix_tensor[..., :, 1]
        rot_world_6d = torch.cat((x_world, y_world), dim=-1)

        return pos_world_tensor, rot_world_6d
