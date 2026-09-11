import torch
from torch import nn
import torch.nn.functional as nnF
import numpy as np
import math

from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

config = {
    "skeleton": None,
    "rot_sequence": [0, 1, 2],
    "source_mocap_format": "fbx",
    "model": None,
    "seq_length": 64,
    "orig_sequences": [],
    "orig_seq_index": 0,
    "use_live_motion": False,
    "device": "cuda",
    "train_root_trajectory": True,
    "root_pos_mean": None,
    "root_pos_std": None,
    "joint_count": 0,
    "mocap_fps": 50,
    "smooth_cutoff": 0.5,
    "smooth_beta": 0.01,
    "gaussian_temp": 0.1,
    "pi_temperature": 2.0,
    "top_p": 0.9
}

class OneEuroFilterTorch:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def setMinCutOff(self, mincutoff):
        self.mincutoff = mincutoff

    def setBeta(self, beta):
        self.beta = beta

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x.clone()
            self.dx_prev = torch.zeros_like(x)
            return x.clone()

        dx = (x - self.x_prev) * self.freq
        a_dcutoff = self._alpha(self.dcutoff)
        edx = a_dcutoff * dx + (1.0 - a_dcutoff) * self.dx_prev
        cutoff = self.mincutoff + self.beta * torch.abs(edx)
        a = self._alpha(cutoff)
        x_filtered = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_filtered.clone()
        self.dx_prev = edx.clone()
        return x_filtered

class MotionSynthesis():
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.rot_sequence = config.get("rot_sequence", [0, 1, 2])
        self.source_mocap_format = config.get("source_mocap_format", "fbx")
        self.model = config["model"]
        self.seq_length = config["seq_length"]
        self.device = config["device"]
        self.orig_sequences = config["orig_sequences"]
        self.orig_seq_index = config["orig_seq_index"]
        self.use_live_motion = config.get("use_live_motion", False)
        self.orig_seq_start_frame_index = 0
        self.orig_seq_frame_count = self.seq_length
        self.orig_seq_blend_factor = 1.0
        self.seq_rand_range = 0.00
        
        self.train_root_trajectory = config.get("train_root_trajectory", True)
        self.root_pos_mean = config.get("root_pos_mean")
        self.root_pos_std = config.get("root_pos_std")

        self.joint_count = config.get("joint_count", 0)
        self.mocap_fps = config.get("mocap_fps", 0)

        self.motion_seq = np.copy(self.orig_sequences[self.orig_seq_index][self.orig_seq_start_frame_index:self.orig_seq_start_frame_index + self.orig_seq_frame_count, ...])

        if self.train_root_trajectory:
            self.motion_seq[:, :3] = (self.motion_seq[:, :3] - self.root_pos_mean.flatten()) / self.root_pos_std.flatten()

        self.motion_seq = torch.from_numpy(self.motion_seq).to(self.device)
        self.orig_seq_changed = False

        self.is_continuing = False 
        self.blend_frames_max = 15 # Frames to blend over during transition
        self.blend_frame_current = 0
        self.transition_start_pose = None

        self.input_dim = self.motion_seq.shape[1]

        self.joint_offsets = self.skeleton["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton["parents"]
        self.joint_children = self.skeleton["children"]

        # Initialize live buffer matching input_dim size
        self.live_seq_buffer = np.zeros((self.seq_length, self.input_dim), dtype=np.float32)

        self._create_edge_list()
        self._build_execution_order()

        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        self.synth_pose_lrot = None

        smooth_freq = self.mocap_fps
        smooth_cutoff = config["smooth_cutoff"]
        smooth_beta = config["smooth_beta"]
        self.one_euro_filter = OneEuroFilterTorch(freq=smooth_freq, mincutoff=smooth_cutoff, beta=smooth_beta)

        self.gaussian_temp = config["gaussian_temp"]
        self.pi_temperature = config["pi_temperature"]
        self.top_p = config["top_p"]

    def _create_edge_list(self):
        self.edge_list = []
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])

    def _build_execution_order(self):
        self.execution_order = []
        visited = set()
        def visit(jI):
            if jI in visited: return
            p = self.joint_parents[jI]
            if p != -1 and p not in visited: visit(p)
            self.execution_order.append(jI)
            visited.add(jI)
        for jI in range(self.joint_count): visit(jI)

    def setOrigSeqIndex(self, index):
        self.orig_seq_index = min(index, len(self.orig_sequences) - 1 )
        self.orig_seq_changed = True

    def setOrigSeqStartFrameIndex(self, index):
        self.orig_seq_start_frame_index = min(index, self.orig_sequences[self.orig_seq_index].shape[0] - self.seq_length )
        self.orig_seq_changed = True

    def setOrigSeqFrameCount(self, count):
        self.orig_seq_frame_count = min(count, self.seq_length)
        self.orig_seq_changed = True

    def setOrigSeqBlend(self, blend):
        self.orig_seq_blend_factor = max(min(1.0, blend), 0.0)
        self.orig_seq_changed = True

    def setRandRange(self, rand):
        self.seq_rand_range = rand

    def changeSequence(self):
        orig_seq = np.copy(self.orig_sequences[self.orig_seq_index][self.orig_seq_start_frame_index:self.orig_seq_start_frame_index + self.orig_seq_frame_count, ...])

        if self.train_root_trajectory:
            orig_seq[:, :3] = (orig_seq[:, :3] - self.root_pos_mean.flatten()) / self.root_pos_std.flatten()
            orig_seq = torch.from_numpy(orig_seq).to(self.device)

        if self.orig_seq_frame_count < self.seq_length:
            self.motion_seq = torch.cat((orig_seq, self.motion_seq[:self.seq_length - self.orig_seq_frame_count, ...]), dim=0)
        else:
            self.motion_seq = orig_seq
        self.orig_seq_changed = False

    def setGaussianTemp(self, gaussian_temp):
        self.gaussian_temp = gaussian_temp

    def setPiTemp(self, pi_temperature):
        self.pi_temperature = pi_temperature

    def setTopP(self, top_p):
        self.top_p = top_p

    def _orthogonalize_pose(self, pose):
        pose_shape = pose.shape
        if self.train_root_trajectory:
            root_pos = pose[..., :3]
            rot_6d = pose[..., 3:]
        else:
            rot_6d = pose

        rot_6d = rot_6d.view(*pose_shape[:-1], self.joint_count, 6)
        rot_mat = rot_to.r6d_to_mat(rot_6d)
        rot_6d_norm = rot_to.mat_to_r6d(rot_mat)
        rot_6d_norm = rot_6d_norm.view(*pose_shape[:-1], -1)

        if self.train_root_trajectory:
            return torch.cat((root_pos, rot_6d_norm), dim=-1)
        else:
            return rot_6d_norm

    def update_live_buffer(self, pos_local, rot_local):
        """Processes incoming OSC frame into the 6D + Root format and shifts the buffer."""
        if rot_local is None: return

        # FIX: MediaPipe sends quaternions in [x, y, z, w] format.
        # The model, npz files, and rot_np utilities expect [w, x, y, z].
        # We must explicitly reorder the quaternion components before conversion.
        rot_local_wxyz = np.empty_like(rot_local)
        rot_local_wxyz[..., 0] = rot_local[..., 3]  # W
        rot_local_wxyz[..., 1] = rot_local[..., 0]  # X
        rot_local_wxyz[..., 2] = rot_local[..., 1]  # Y
        rot_local_wxyz[..., 3] = rot_local[..., 2]  # Z

        rot_local_wxyz = rot_np.normalize(rot_local_wxyz.astype(np.float32))
        
        # Convert the correctly ordered quaternion to 6D
        rot_6d = rot_np.quat_to_r6d(np.expand_dims(rot_local_wxyz, axis=0))
        rot_flat = rot_6d.reshape(1, -1)
        
        if self.train_root_trajectory and pos_local is not None:
            root_pos = pos_local[0:1, :]
            root_pos_norm = (root_pos - self.root_pos_mean.flatten()) / self.root_pos_std.flatten()
            frame = np.concatenate((root_pos_norm, rot_flat), axis=1).astype(np.float32)
        else:
            frame = rot_flat.astype(np.float32)
            
        # Shift buffer and append new frame
        self.live_seq_buffer = np.roll(self.live_seq_buffer, -1, axis=0)
        self.live_seq_buffer[-1] = frame[0]

    def _transpose_6d(self, pose_6d):
            """Transposes the rotation matrices embedded within the 6D representation to align coordinate spaces."""
            shape = pose_6d.shape
            rot_6d = pose_6d.view(-1, 6)
            rot_mat = rot_to.r6d_to_mat(rot_6d)
            rot_mat_T = rot_mat.transpose(-1, -2) # Flip the rotation axes
            rot_6d_T = rot_to.mat_to_r6d(rot_mat_T)
            return rot_6d_T.view(shape)

    def update(self):
        """State machine for tracking live frames, blending transitions, and updating kinematics."""
        
        # --- 1. Update Pose Sequence Based on Mode ---
        if getattr(self, "use_live_motion", False) == False:
            if getattr(self, "orig_seq_changed", False):
                self.changeSequence()
            self._generate_next_frame()
            
        else:
            if getattr(self, "is_continuing", False) == False:
                # Live OSC Tracking (Bypass generation)
                current_frame_np = self.live_seq_buffer[-1:]
                self.pred_pose = torch.from_numpy(current_frame_np).unsqueeze(0).to(self.device)
                
                # Removed: self.pred_pose = self._orthogonalize_pose(self.pred_pose)
                # (No longer needed now that the quaternion [w, x, y, z] is correct!)
                
                if getattr(self, "transition_start_pose", None) is not None and self.blend_frame_current < self.blend_frames_max:
                    alpha = self.blend_frame_current / self.blend_frames_max
                    self.pred_pose = self._blend_poses(self.transition_start_pose, self.pred_pose, alpha)
                    self.blend_frame_current += 1
                elif getattr(self, "transition_start_pose", None) is not None:
                    self.transition_start_pose = None 
                    
            else:
                # Autoregressive Continuation
                self._generate_next_frame()
                
                if getattr(self, "transition_start_pose", None) is not None and self.blend_frame_current < self.blend_frames_max:
                    alpha = self.blend_frame_current / self.blend_frames_max
                    self.pred_pose = self._blend_poses(self.transition_start_pose, self.pred_pose, alpha)
                    self.blend_frame_current += 1
                    
                    self.motion_seq[-1:] = self.pred_pose.squeeze(0)
                elif getattr(self, "transition_start_pose", None) is not None:
                    self.transition_start_pose = None

        # --- 2. Reconstruct World Kinematics for Output ---
        # Safe extraction via flatten ensures 1D array regardless of source
        pred_numpy = self.pred_pose.detach().cpu().numpy().flatten()

        if self.train_root_trajectory:
            root_traj = pred_numpy[:3]
            root_traj = (root_traj * self.root_pos_std.flatten()) + self.root_pos_mean.flatten()
            rot_seq = pred_numpy[3:]
        else:
            root_traj = np.zeros(3, dtype=np.float32)
            rot_seq = pred_numpy

        # Format exactly as the autoencoder expects it
        joint_rot_6d = torch.tensor(rot_seq, dtype=torch.float32, device=self.device).reshape(1, 1, self.joint_count, 6)
        root_trajectory = torch.tensor(root_traj, dtype=torch.float32, device=self.device).reshape(1, 1, 3)

        # Calculate kinematics using the original _forward_kinematics function
        joint_pos_world, joint_rot_6d_world = self._forward_kinematics(joint_rot_6d, root_trajectory)

        # Process kinematics with the identical r6d_to_quat utilities from the working autoencoder
        self.synth_pose_wpos = joint_pos_world.detach().cpu().numpy().reshape(self.joint_count, 3)
        self.synth_pose_wrot = rot_np.r6d_to_quat(joint_rot_6d_world.detach().cpu().numpy()).reshape(self.joint_count, 4)
        self.synth_pose_lrot = rot_np.r6d_to_quat(joint_rot_6d.squeeze().detach().cpu().numpy()).reshape(self.joint_count, 4)
    def _generate_next_frame(self):
            """Autoregressively predicts the next frame and updates the sequence buffer."""
            with torch.no_grad():
                log_pi, mu, sigma = self.model(self.motion_seq.unsqueeze(0), return_sequence=False)
                pred_pose_norm = self._sample_mdn(log_pi, mu, sigma, gaussian_temp=self.gaussian_temp, pi_temperature=self.pi_temperature, top_p=self.top_p)
                
                # Apply 1 Euro Filter and enforce orthogonality
                filtered_pose = self.one_euro_filter(pred_pose_norm)
                self.pred_pose = self._orthogonalize_pose(filtered_pose)
                
                # Shift autoregressive history and append the newly generated pose
                self.motion_seq = torch.cat([self.motion_seq[1:, :], self.pred_pose.squeeze(0)], axis=0)

    def _blend_poses(self, pose_a, pose_b, alpha):
        """
        Linearly interpolates the root translation and uses Batched Slerp 
        for the joint rotations to ensure natural skeletal transitions.
        """
        # Flatten safely avoids squeeze dimension errors regardless of input shape
        pose_a_np = pose_a.detach().cpu().numpy().flatten()
        pose_b_np = pose_b.detach().cpu().numpy().flatten()
        
        # Separate root and rotation data
        if self.train_root_trajectory:
            root_a, rot_a = pose_a_np[:3], pose_a_np[3:]
            root_b, rot_b = pose_b_np[:3], pose_b_np[3:]
            root_interp = root_a * (1.0 - alpha) + root_b * alpha
        else:
            rot_a, rot_b = pose_a_np, pose_b_np
            root_interp = None
            
        rot_a_6d = rot_a.reshape(self.joint_count, 6)
        rot_b_6d = rot_b.reshape(self.joint_count, 6)
        
        # Convert 6D -> Quaternions for proper spherical interpolation
        quat_a = rot_np.r6d_to_quat(rot_a_6d)
        quat_b = rot_np.r6d_to_quat(rot_b_6d)
        
        # Batched Quaternion Slerp
        dot = np.sum(quat_a * quat_b, axis=-1, keepdims=True)
        quat_b_aligned = np.where(dot < 0, -quat_b, quat_b) # Shortest path
        dot = np.clip(np.abs(dot), -1.0, 1.0)
        
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        # Guard against divide-by-zero for parallel vectors (use lerp instead)
        mask = sin_theta > 1e-6
        s0 = np.where(mask, np.sin((1.0 - alpha) * theta) / (sin_theta + 1e-8), 1.0 - alpha)
        s1 = np.where(mask, np.sin(alpha * theta) / (sin_theta + 1e-8), alpha)
        
        quat_interp = s0 * quat_a + s1 * quat_b_aligned
        quat_interp = quat_interp / np.linalg.norm(quat_interp, axis=-1, keepdims=True)
        
        # Convert Quaternions -> 6D -> Flat array
        rot_interp_6d = rot_np.quat_to_r6d(quat_interp)
        rot_interp_flat = rot_interp_6d.reshape(-1)
        
        if self.train_root_trajectory:
            pose_interp_np = np.concatenate((root_interp, rot_interp_flat), axis=0)
        else:
            pose_interp_np = rot_interp_flat
            
        # Wrap back into a consistent device tensor shape (1, 1, input_dim)
        return torch.from_numpy(pose_interp_np).view(1, 1, -1).to(self.device)

    def trigger_continuation(self):
        """Triggers the model to start generating based on the current live buffer."""
        self.is_continuing = True
        
        # Load the live buffer as the sequence seed
        self.motion_seq = torch.from_numpy(self.live_seq_buffer).to(self.device)
        
        # Save the last actual live pose to blend from, ensuring shape (1, 1, input_dim)
        self.transition_start_pose = self.motion_seq[-1:].unsqueeze(0).clone()
        self.blend_frame_current = 0

    def return_to_live(self):
        """Snaps back to mirroring the live OSC stream without generation."""
        self.is_continuing = False
        
        # Prepare to blend back to live stream, shape is already (1, 1, input_dim)
        self.transition_start_pose = self.pred_pose.clone() if hasattr(self, 'pred_pose') else None
        self.blend_frame_current = 0

    def _sample_mdn(self, log_pi, mu, sigma, gaussian_temp=0.1, pi_temperature=1.5, top_p=0.9):
        batch_size, seq_len, K = log_pi.shape

        scaled_log_pi = log_pi / pi_temperature
        pi_probs = nnF.softmax(scaled_log_pi, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(pi_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(pi_probs, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            pi_probs[indices_to_remove] = 0.0
            pi_probs = pi_probs / pi_probs.sum(dim=-1, keepdim=True)

        pi_probs_flat = pi_probs.reshape(-1, K)
        chosen_idx = torch.multinomial(pi_probs_flat, 1)
        chosen_idx = chosen_idx.view(batch_size, seq_len, 1, 1)

        idx_expanded = chosen_idx.expand(batch_size, seq_len, 1, mu.size(-1))

        chosen_mu = torch.gather(mu, 2, idx_expanded).squeeze(2)
        chosen_sigma = torch.gather(sigma, 2, idx_expanded).squeeze(2)

        epsilon = torch.randn_like(chosen_mu) * gaussian_temp
        sampled_pose = chosen_mu + chosen_sigma * epsilon
        
        return self._orthogonalize_pose(sampled_pose)

    def _forward_kinematics(self, rotations_6d, root_positions):
        # Convert 6D to 3x3 matrices internally
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
                
                # Column-Major Matrix Multiplication for Vectors
                rotated_offset = torch.matmul(parent_rot, local_offset).squeeze(-1)
                positions_world[jI] = rotated_offset + parent_pos

                if len(self.joint_children[jI]) > 0:
                    local_rot = rotation_matrices[:, :, jI]
                    # Column-Major Matrix Multiplication for Rotations
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