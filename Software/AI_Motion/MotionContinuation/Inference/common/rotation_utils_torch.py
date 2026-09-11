
"""
Rotational Representations and Utilities (PyTorch)
Contains 6D representation, Quaternions, and Angle-Axis functions utilizing PyTorch tensors.
"""

import torch
import torch.nn.functional as nnF
import math

class RotationUtilsTorch:

    # ==============================
    # 6D Rotations
    # ==============================

    @staticmethod
    def orthogonalize_r6d(rot_6d):
        matrices = RotationUtilsTorch.r6d_to_mat(rot_6d)
        
        # Extract the X and Y column vectors [..., 3]
        x = matrices[..., :, 0]
        y = matrices[..., :, 1]
        
        # Concatenate to form proper [..., 6] format
        return torch.cat((x, y), dim=-1)
        
    @staticmethod
    def r6d_to_mat(poses):
        """Convert 6D rotation to 3x3 rotation matrix."""
        x_raw = poses[..., 0:3]
        y_raw = poses[..., 3:6]
        x = nnF.normalize(x_raw, dim=-1)
        z = torch.cross(x, y_raw, dim=-1)
        z = nnF.normalize(z, dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack((x, y, z), dim=-1)

    @staticmethod
    def mat_to_r6d(mat):
        """Convert 3x3 rotation matrix to 6D rotation."""
        x = mat[..., :, 0]
        y = mat[..., :, 1]
        return torch.cat((x, y), dim=-1)

    @staticmethod
    def r6d_to_quat(poses_6d):
        mat = RotationUtilsTorch.r6d_to_mat(poses_6d)
        return RotationUtilsTorch.mat_to_quat(mat)

    @staticmethod
    def quat_to_r6d(quats):
        mat = RotationUtilsTorch.quat_to_mat(quats)
        return RotationUtilsTorch.mat_to_r6d(mat)

    @staticmethod
    def interpolate_r6d(input_tensor, size):
        batch = input_tensor.shape[0]
        length = input_tensor.shape[-1]

        x = input_tensor.reshape(batch, -1, 6, length).permute(0, 1, 3, 2)
        input_q = RotationUtilsTorch.r6d_to_quat(x)

        idx = torch.linspace(0, length - 1, size, device=input_tensor.device)
        idx_l = torch.floor(idx).long()
        idx_r = torch.clamp(idx_l + 1, max=length - 1)
        t = (idx - idx_l).view(1, 1, -1, 1)

        q0 = input_q[:, :, idx_l, :]
        q1 = input_q[:, :, idx_r, :]
        res_q = RotationUtilsTorch.slerp(q0, q1, t)

        res_6d = RotationUtilsTorch.quat_to_r6d(res_q)
        res_6d = res_6d.permute(0, 1, 3, 2).reshape(batch, -1, size)
        return res_6d

    # ==============================
    # Quaternions
    # ==============================

    @staticmethod
    def mag(q):
        """Return magnitude of quaternion"""
        return torch.linalg.norm(q, dim=-1, keepdim=True)

    @staticmethod
    def conj(q):
        """Returns conjugate of quaternion"""
        return torch.cat((q[..., :1], q[..., -3:] * -1), dim=-1)

    @staticmethod
    def inv(q):
        """Returns inverse of quaternion"""
        return RotationUtilsTorch.conj(q) / RotationUtilsTorch.mag(q)

    @staticmethod
    def normalize(q):
        """Returns normalized quaternion"""
        return nnF.normalize(q, dim=-1)

    @staticmethod
    def mul(q, r):
        """Multiply quaternion(s) q with quaternion(s) r"""
        original_shape = q.shape
        terms = torch.bmm(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    @staticmethod
    def rot(q, v):
        """Rotate vector(s) v about the rotation described by quaternion(s) q"""
        original_shape = list(v.shape)
        q = q.reshape(-1, 4)
        v = v.reshape(-1, 3)
        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

    @staticmethod
    def quat_to_mat(quats):
        """Convert [w, x, y, z] quaternions to 3x3 rotation matrices."""
        quats = nnF.normalize(quats, dim=-1)
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        row0 = torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1)
        row1 = torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1)
        row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)

        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def mat_to_quat(mat):
        """Convert 3x3 rotation matrices to [w, x, y, z] quaternions."""
        m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
        m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
        m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

        trace = m00 + m11 + m22

        def safe_sqrt(x): return torch.sqrt(torch.clamp(x, min=0.0))

        q_w = 0.5 * safe_sqrt(1.0 + trace)
        q_x = 0.5 * safe_sqrt(1.0 + m00 - m11 - m22) * torch.sign(m21 - m12)
        q_y = 0.5 * safe_sqrt(1.0 - m00 + m11 - m22) * torch.sign(m02 - m20)
        q_z = 0.5 * safe_sqrt(1.0 - m00 - m11 + m22) * torch.sign(m10 - m01)

        quats = torch.stack([q_w, q_x, q_y, q_z], dim=-1)
        return nnF.normalize(quats, dim=-1)

    @staticmethod
    def quat_to_euler(q, order='xyz', degrees=True):
        """Convert (w, x, y, z) quaternions to euler angles."""
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        es = torch.empty(q0.shape + (3,), device=q.device, dtype=q.dtype)

        if order == 'xyz':
            es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
            es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        else:
            raise NotImplementedError(f'Cannot convert to ordering {order}')

        if degrees:
            es = es * 180 / math.pi
        return es

    @staticmethod
    def slerp(q0, q1, t):
        """Spherical Linear Interpolation between quaternions."""
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.clamp(torch.abs(dot), -1.0, 1.0)

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        mask = sin_theta > 1e-6
        w0 = torch.where(mask, torch.sin((1 - t) * theta) / sin_theta, 1.0 - t)
        w1 = torch.where(mask, torch.sin(t * theta) / sin_theta, t)

        res = w0 * q0 + w1 * q1
        return nnF.normalize(res, dim=-1)

    # ==============================
    # Angle Axis & Euler
    # ==============================

    @staticmethod
    def aa_to_quat(rots, form='wxyz', unified_orient=True):
        """Convert angle-axis representation to quaternion"""
        angles = rots.norm(dim=-1, keepdim=True)
        norm = angles.clone()
        norm[norm < 1e-8] = 1
        axis = rots / norm
        quats = torch.empty(rots.shape[:-1] + (4,), device=rots.device, dtype=rots.dtype)
        angles = angles * 0.5

        if form == 'wxyz':
            quats[..., 0] = torch.cos(angles.squeeze(-1))
            quats[..., 1:] = torch.sin(angles) * axis
        elif form == 'xyzw':
            quats[..., :3] = torch.sin(angles) * axis
            quats[..., 3] = torch.cos(angles.squeeze(-1))

        if unified_orient:
            idx = quats[..., 0] < 0
            quats[idx, :] *= -1
        return quats

    @staticmethod
    def quat_to_aa(quats):
        """Convert quaternions to angle-axis representation"""
        _cos = quats[..., 0]
        xyz = quats[..., 1:]
        _sin = xyz.norm(dim=-1)
        norm = _sin.clone()
        norm[norm < 1e-7] = 1
        axis = xyz / norm.unsqueeze(-1)
        angle = torch.atan2(_sin, _cos) * 2
        return axis * angle.unsqueeze(-1)
