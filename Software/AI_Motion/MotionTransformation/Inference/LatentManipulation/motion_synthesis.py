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
    "root_pos_std": None,
    "mocap_fps": None,
    "orig_sequences": [],
    "orig_seq1_index": 0,
    "orig_seq2_index": 1,
    "use_live_seq1": False,
    "use_live_seq2": False
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
        self.mocap_fps = config["mocap_fps"]
        self.orig_sequences = config["orig_sequences"]
        self.orig_seq1_index = int(config["orig_seq1_index"])
        self.orig_seq2_index = int(config["orig_seq2_index"])
        self.use_live_seq1 = config.get("use_live_seq1", False)
        self.use_live_seq2 = config.get("use_live_seq2", False)

        if self.skeleton is None:
            raise ValueError("config['skeleton'] must not be None")
        if self.model is None:
            raise ValueError("config['model'] must not be None")
        if len(self.orig_sequences) == 0:
            raise ValueError("config['orig_sequences'] must contain at least one sequence")

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

        if hasattr(self.model, "input_dim") and self.model.input_dim != self.input_dim:
            raise ValueError(
                f"Model input_dim ({self.model.input_dim}) does not match synthesis input_dim ({self.input_dim})."
            )

        if self.root_trajectory:
            if self.root_pos_mean is None:
                self.root_pos_mean = np.zeros((1, 1, 3), dtype=np.float32)
            if self.root_pos_std is None:
                self.root_pos_std = np.ones((1, 1, 3), dtype=np.float32)

            self.root_pos_mean = torch.as_tensor(self.root_pos_mean, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
            self.root_pos_std = torch.as_tensor(self.root_pos_std, dtype=torch.float32, device=self.device).reshape(1, 1, 3)
            self.root_pos_std = torch.where(self.root_pos_std == 0.0, torch.ones_like(self.root_pos_std), self.root_pos_std)

        self._create_edge_list()

        self.orig_seq1 = self._prepare_sequence(self.orig_sequences[self.orig_seq1_index])
        self.orig_seq2 = self._prepare_sequence(self.orig_sequences[self.orig_seq2_index])

        self.orig_seq1_changed = False
        self.orig_seq2_changed = False

        self.seq1_length = self.orig_seq1.shape[0]
        self.seq2_length = self.orig_seq2.shape[0]

        self.orig_seq1_frame_index = 0
        self.orig_seq2_frame_index = 0

        self.orig_seq1_frame_incr = self.seq_window_offset
        self.orig_seq2_frame_incr = self.seq_window_offset

        self.orig_seq1_frame_range = [0, max(0, self.seq1_length - self.seq_window_length)]
        self.orig_seq2_frame_range = [0, max(0, self.seq2_length - self.seq_window_length)]

        self.live_buffer_pos = np.zeros((self.seq_window_length, self.joint_count, 3), dtype=np.float32)
        self.live_buffer_rot = np.zeros((self.seq_window_length, self.joint_count, 4), dtype=np.float32)
        self.live_buffer_rot[:, :, 0] = 1.0  # Set default quaternion (W=1)

        self.latent_dim, self.latent_steps = self._infer_latent_shape()

        self.encoding_mix = torch.zeros((1, self.latent_dim, self.latent_steps), dtype=torch.float32, device=self.device)
        self.encoding_offset = torch.zeros((1, self.latent_dim, self.latent_steps), dtype=torch.float32, device=self.device)

        self.gen_seq = self._generate_window()
        self.gen_seq_window = self.gen_seq.clone()

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

    def _prepare_sequence(self, sequence):
        if isinstance(sequence, dict):
            if "motion" in sequence:
                motion = sequence["motion"]
                rot_local = motion["rot_local"]
                pos_local = motion.get("pos_local", None)
            else:
                rot_local = sequence.get("rot_local", None)
                pos_local = sequence.get("pos_local", None)
                if rot_local is None:
                    raise ValueError("Sequence dict must contain 'motion.rot_local' or 'rot_local'")

            rot_local = np.asarray(rot_local, dtype=np.float32)
            if rot_local.ndim != 3:
                raise ValueError("Sequence rotations must have shape [frames, joints, 4] or [frames, joints, 6]")

            if rot_local.shape[-1] == 4:
                rot_local = rot_np.quat_to_6d(rot_local)
            elif rot_local.shape[-1] != 6:
                raise ValueError("Rotation representation must be quaternion (4) or 6D (6)")

            rot_flat = rot_local.reshape(rot_local.shape[0], -1)

            if self.root_trajectory:
                if pos_local is not None:
                    pos_local = np.asarray(pos_local, dtype=np.float32)
                    root_pos = pos_local[:, 0, :]
                else:
                    root_pos = np.zeros((rot_flat.shape[0], 3), dtype=np.float32)
                return np.concatenate((root_pos, rot_flat), axis=1).astype(np.float32)

            return rot_flat.astype(np.float32)

        seq = np.asarray(sequence, dtype=np.float32)

        if seq.ndim == 3:
            if seq.shape[-1] == 4:
                seq = rot_np.quat_to_6d(seq)
            elif seq.shape[-1] != 6:
                raise ValueError("3D sequence arrays must end in 4 (quat) or 6 (6D)")

            rot_flat = seq.reshape(seq.shape[0], -1)
            if self.root_trajectory:
                root_pos = np.zeros((seq.shape[0], 3), dtype=np.float32)
                return np.concatenate((root_pos, rot_flat), axis=1).astype(np.float32)
            return rot_flat.astype(np.float32)

        if seq.ndim == 2:
            if seq.shape[1] == self.input_dim:
                return seq.astype(np.float32)
            if self.root_trajectory and seq.shape[1] == self.pose_dim:
                root_pos = np.zeros((seq.shape[0], 3), dtype=np.float32)
                return np.concatenate((root_pos, seq), axis=1).astype(np.float32)
            if (not self.root_trajectory) and seq.shape[1] == self.pose_dim:
                return seq.astype(np.float32)

        raise ValueError("Unsupported sequence format")

    def _slice_window(self, sequence, start_index):
        max_start = max(0, sequence.shape[0] - self.seq_window_length)
        start_index = int(np.clip(start_index, 0, max_start))
        end_index = start_index + self.seq_window_length

        if end_index <= sequence.shape[0]:
            return sequence[start_index:end_index].copy()

        available = sequence[start_index:].copy()
        pad_count = end_index - sequence.shape[0]
        pad = np.repeat(sequence[-1:, :], pad_count, axis=0)
        return np.concatenate((available, pad), axis=0)

    def _prepare_window_tensor(self, window_np):
        window = torch.from_numpy(window_np).to(self.device).unsqueeze(0)

        if self.root_trajectory:
            root = window[:, :, :3]
            root = (root - self.root_pos_mean) / self.root_pos_std
            window = torch.cat((root, window[:, :, 3:]), dim=-1)

        return window.permute(0, 2, 1)

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

    def _advance_frame_index(self, frame_index, incr, frame_range):
        frame_index += incr
        if frame_index < frame_range[0] or frame_index > frame_range[1]:
            frame_index = frame_range[0]
        return frame_index

    def _expand_latent_control(self, value):
        if np.isscalar(value):
            return torch.full(
                (1, self.latent_dim, self.latent_steps),
                float(value),
                dtype=torch.float32,
                device=self.device,
            )

        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        if value.ndim == 0:
            return value.view(1, 1, 1).expand(1, self.latent_dim, self.latent_steps)

        if value.ndim == 1:
            if value.shape[0] == self.latent_dim:
                return value.view(1, self.latent_dim, 1).expand(1, self.latent_dim, self.latent_steps)
            if value.shape[0] == self.latent_steps:
                return value.view(1, 1, self.latent_steps).expand(1, self.latent_dim, self.latent_steps)
            if value.shape[0] == 1:
                return value.view(1, 1, 1).expand(1, self.latent_dim, self.latent_steps)

        if value.ndim == 2:
            if value.shape == (self.latent_dim, self.latent_steps):
                return value.unsqueeze(0)
            if value.shape == (1, self.latent_dim):
                return value.unsqueeze(-1).expand(1, self.latent_dim, self.latent_steps)
            if value.shape == (1, self.latent_steps):
                return value.unsqueeze(1).expand(1, self.latent_dim, self.latent_steps)

        if value.ndim == 3 and value.shape == (1, self.latent_dim, self.latent_steps):
            return value

        raise ValueError(
            f"Could not broadcast latent control with shape {tuple(value.shape)} "
            f"to (1, {self.latent_dim}, {self.latent_steps})"
        )

    def setSeq1Index(self, index):
        self.orig_seq1_index = int(np.clip(index, 0, len(self.orig_sequences) - 1))
        self.orig_seq1_changed = True

    def setSeq2Index(self, index):
        self.orig_seq2_index = int(np.clip(index, 0, len(self.orig_sequences) - 1))
        self.orig_seq2_changed = True

    def changeSeq1(self):
        self.orig_seq1 = self._prepare_sequence(self.orig_sequences[self.orig_seq1_index])
        self.seq1_length = self.orig_seq1.shape[0]
        self.orig_seq1_frame_index = 0
        self.orig_seq1_frame_range = [0, max(0, self.seq1_length - self.seq_window_length)]
        self.orig_seq1_changed = False

    def changeSeq2(self):
        self.orig_seq2 = self._prepare_sequence(self.orig_sequences[self.orig_seq2_index])
        self.seq2_length = self.orig_seq2.shape[0]
        self.orig_seq2_frame_index = 0
        self.orig_seq2_frame_range = [0, max(0, self.seq2_length - self.seq_window_length)]
        self.orig_seq2_changed = False

    def setSeq1FrameIndex(self, index):
        self.orig_seq1_frame_index = int(np.clip(index, 0, max(0, self.seq1_length - self.seq_window_length)))

    def setSeq2FrameIndex(self, index):
        self.orig_seq2_frame_index = int(np.clip(index, 0, max(0, self.seq2_length - self.seq_window_length)))

    def setSeq1FrameRange(self, startFrame, endFrame):
        max_index = max(0, self.seq1_length - self.seq_window_length)
        self.orig_seq1_frame_range[0] = int(np.clip(startFrame, 0, max_index))
        self.orig_seq1_frame_range[1] = int(np.clip(endFrame, 0, max_index))
        if self.orig_seq1_frame_range[1] < self.orig_seq1_frame_range[0]:
            self.orig_seq1_frame_range[1] = self.orig_seq1_frame_range[0]

    def setSeq2FrameRange(self, startFrame, endFrame):
        max_index = max(0, self.seq2_length - self.seq_window_length)
        self.orig_seq2_frame_range[0] = int(np.clip(startFrame, 0, max_index))
        self.orig_seq2_frame_range[1] = int(np.clip(endFrame, 0, max_index))
        if self.orig_seq2_frame_range[1] < self.orig_seq2_frame_range[0]:
            self.orig_seq2_frame_range[1] = self.orig_seq2_frame_range[0]

    def setSeq1FrameIncrement(self, incr):
        self.orig_seq1_frame_incr = int(max(1, incr))

    def setSeq2FrameIncrement(self, incr):
        self.orig_seq2_frame_incr = int(max(1, incr))

    def setEncodingMix(self, mix):
        self.encoding_mix = self._expand_latent_control(mix)

    def setEncodingOffset(self, offset):
        self.encoding_offset = self._expand_latent_control(offset)

    def getEdgeList(self):
        return self.edge_list

    def getSynthPoseWorldPositions(self):
        return self.synth_pose_wpos

    def getSynthPoseWorldRotations(self):
        return self.synth_pose_wrot

    def _generate_window(self):
        if self.use_live_seq1:
            orig_seq1_window = self._get_live_window()
        else:
            orig_seq1_window = self._slice_window(self.orig_seq1, self.orig_seq1_frame_index)
        
        if self.use_live_seq2:
            orig_seq2_window = self._get_live_window()
        else:
            orig_seq2_window = self._slice_window(self.orig_seq2, self.orig_seq2_frame_index)

        orig_seq1_window = self._prepare_window_tensor(orig_seq1_window)
        orig_seq2_window = self._prepare_window_tensor(orig_seq2_window)

        with torch.no_grad():
            mu1, logvar1 = self.model.encode(orig_seq1_window)
            z1 = self.model.reparameterize(mu1, logvar1)

            mu2, logvar2 = self.model.encode(orig_seq2_window)
            z2 = self.model.reparameterize(mu2, logvar2)

            z = z1 * (1.0 - self.encoding_mix) + z2 * self.encoding_mix
            z = z + self.encoding_offset

            decoded = self.model.decode(z)

        gen_seq_window = self._postprocess_decoded_window(decoded)

        self.orig_seq1_frame_index = self._advance_frame_index(
            self.orig_seq1_frame_index,
            self.orig_seq1_frame_incr,
            self.orig_seq1_frame_range,
        )
        self.orig_seq2_frame_index = self._advance_frame_index(
            self.orig_seq2_frame_index,
            self.orig_seq2_frame_incr,
            self.orig_seq2_frame_range,
        )

        return gen_seq_window

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

    def update_live_motion(self, pos_local, rot_local):
        """Shifts the live buffers and adds the newest frame."""
        self.live_buffer_pos = np.roll(self.live_buffer_pos, -1, axis=0)
        self.live_buffer_rot = np.roll(self.live_buffer_rot, -1, axis=0)
        
        if pos_local is not None:
            self.live_buffer_pos[-1] = pos_local
        if rot_local is not None:
            self.live_buffer_rot[-1] = rot_local

    def _get_live_window(self):
        """Converts the live quaternion buffer into the 6D representation expected by the network."""
        rot_local = self.live_buffer_rot.copy()
        pos_local = self.live_buffer_pos.copy()
        
        rot_6d = rot_np.quat_to_r6d(rot_local)
        rot_flat = rot_6d.reshape(self.seq_window_length, -1)
        
        if self.root_trajectory:
            root_pos = pos_local[:, 0, :]
            window = np.concatenate((root_pos, rot_flat), axis=1).astype(np.float32)
        else:
            window = rot_flat.astype(np.float32)
        
        return window

    def update(self):
        if self.orig_seq1_changed:
            self.changeSeq1()
            self.gen_seq = self._generate_window()
            self.seq_update_index = 0

        if self.orig_seq2_changed:
            self.changeSeq2()
            self.gen_seq = self._generate_window()
            self.seq_update_index = 0

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
            self.gen_seq_window = self._generate_window()
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
