"""
Rotational Representations and Utilities (PyTorch)
Contains 6D representation, Quaternions, and Angle-Axis functions utilizing PyTorch tensors.
Aligned to [x, y, z, w] quaternion format.
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
        x = matrices[..., :, 0]
        y = matrices[..., :, 1]
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
    # Quaternions [x, y, z, w]
    # ==============================

    @staticmethod
    def mag(q):
        return torch.linalg.norm(q, dim=-1, keepdim=True)

    @staticmethod
    def conj(q):
        """Returns conjugate of quaternion [x, y, z, w] -> [-x, -y, -z, w]"""
        return torch.cat((q[..., :3] * -1, q[..., 3:]), dim=-1)

    @staticmethod
    def inv(q):
        return RotationUtilsTorch.conj(q) / RotationUtilsTorch.mag(q)

    @staticmethod
    def normalize(q):
        return nnF.normalize(q, dim=-1)

    @staticmethod
    def mul(q, r):
        """Multiply quaternion(s) q with quaternion(s) r"""
        original_shape = q.shape
        q = q.reshape(-1, 4)
        r = r.reshape(-1, 4)
        
        x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack((x, y, z, w), dim=1).view(original_shape)

    @staticmethod
    def rot(q, v):
        """Rotate vector(s) v about the rotation described by quaternion(s) q"""
        original_shape = list(v.shape)
        q = q.reshape(-1, 4)
        v = v.reshape(-1, 3)
        
        qvec = q[:, :3] # x, y, z
        qw = q[:, 3:4]  # w
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (qw * uv + uuv)).view(original_shape)

    @staticmethod
    def quat_to_mat(quats):
        """Convert [x, y, z, w] quaternions to 3x3 rotation matrices."""
        quats = nnF.normalize(quats, dim=-1)
        x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        row0 = torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1)
        row1 = torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1)
        row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)

        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def mat_to_quat(mat):
        """Convert 3x3 rotation matrices to [x, y, z, w] quaternions."""
        m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
        m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
        m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

        trace = m00 + m11 + m22

        def safe_sqrt(x): return torch.sqrt(torch.clamp(x, min=0.0))

        q_w = 0.5 * safe_sqrt(1.0 + trace)
        q_x = 0.5 * safe_sqrt(1.0 + m00 - m11 - m22) * torch.sign(m21 - m12)
        q_y = 0.5 * safe_sqrt(1.0 - m00 + m11 - m22) * torch.sign(m02 - m20)
        q_z = 0.5 * safe_sqrt(1.0 - m00 - m11 + m22) * torch.sign(m10 - m01)

        quats = torch.stack([q_x, q_y, q_z, q_w], dim=-1)
        return nnF.normalize(quats, dim=-1)

    @staticmethod
    def quat_to_euler(q, order='xyz', degrees=True):
        """Convert [x, y, z, w] quaternions to euler angles."""
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        es = torch.empty(x.shape + (3,), device=q.device, dtype=q.dtype)

        if order == 'xyz':
            es[..., 2] = torch.atan2(2 * (w * z - x * y), w * w + x * x - y * y - z * z)
            es[..., 1] = torch.asin((2 * (x * z + w * y)).clip(-1, 1))
            es[..., 0] = torch.atan2(2 * (w * x - y * z), w * w - x * x - y * y + z * z)
        else:
            raise NotImplementedError(f'Cannot convert to ordering {order}')

        if degrees:
            es = es * 180 / math.pi
        return es

    @staticmethod
    def slerp(q0, q1, t):
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
    def aa_to_quat(rots, form='xyzw', unified_orient=True):
        angles = rots.norm(dim=-1, keepdim=True)
        norm = angles.clone()
        norm[norm < 1e-8] = 1
        axis = rots / norm
        quats = torch.empty(rots.shape[:-1] + (4,), device=rots.device, dtype=rots.dtype)
        angles = angles * 0.5

        # Hardcode to xyzw target mapping regardless of input
        quats[..., :3] = torch.sin(angles) * axis
        quats[..., 3] = torch.cos(angles.squeeze(-1))

        if unified_orient:
            idx = quats[..., 3] < 0
            quats[idx, :] *= -1
        return quats

    @staticmethod
    def quat_to_aa(quats):
        xyz = quats[..., :3]
        _cos = quats[..., 3]
        _sin = xyz.norm(dim=-1)
        norm = _sin.clone()
        norm[norm < 1e-7] = 1
        axis = xyz / norm.unsqueeze(-1)
        angle = torch.atan2(_sin, _cos) * 2
        return axis * angle.unsqueeze(-1)