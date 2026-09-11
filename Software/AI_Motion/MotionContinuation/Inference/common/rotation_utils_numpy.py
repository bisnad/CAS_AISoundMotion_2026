
"""
Rotational Representations and Utilities (NumPy)
Contains 6D representation, Quaternions, and Angle-Axis functions utilizing NumPy and SciPy.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class RotationUtilsNumpy:

    # ==============================
    # 6D Rotations
    # ==============================

    @staticmethod
    def r6d_to_mat(poses):
        orig_shape = poses.shape
        poses = poses.reshape(-1, 6)
        x = poses[:, 0:3]
        y_raw = poses[:, 3:6]

        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        y = np.cross(z, x)

        matrices = np.stack((x, y, z), axis=-1)
        return matrices.reshape(orig_shape[:-1] + (3, 3))

    @staticmethod
    def mat_to_r6d(mat):
        x = mat[..., :, 0]
        y = mat[..., :, 1]
        return np.concatenate((x, y), axis=-1).astype(np.float32)

    @staticmethod
    def quat_to_r6d(quats):
        orig_shape = quats.shape
        quats_flat = quats.reshape(-1, 4)
        matrices = R.from_quat(quats_flat, scalar_first=True).as_matrix()
        matrices = matrices.reshape(orig_shape[:-1] + (3, 3))
        return RotationUtilsNumpy.mat_to_r6d(matrices)

    @staticmethod
    def r6d_to_quat(poses_6d):
        matrices = RotationUtilsNumpy.r6d_to_mat(poses_6d)
        orig_shape = matrices.shape
        quats = R.from_matrix(matrices.reshape(-1, 3, 3)).as_quat(scalar_first=True)
        return quats.reshape(orig_shape[:-2] + (4,)).astype(np.float32)

    @staticmethod
    def interpolate_r6d(input_array, size):
        batch = input_array.shape[0]
        length = input_array.shape[-1]

        x = input_array.reshape(batch, -1, 6, length).transpose(0, 1, 3, 2)
        joint_count = x.shape[1]

        times_orig = np.arange(length)
        times_new = np.linspace(0, length - 1, size)

        res_6d = np.zeros((batch, joint_count, size, 6), dtype=np.float32)

        for b in range(batch):
            for j in range(joint_count):
                mats = RotationUtilsNumpy.r6d_to_mat(x[b, j])
                rotations = R.from_matrix(mats)
                slerp = Slerp(times_orig, rotations)
                interp_rotations = slerp(times_new)
                res_6d[b, j] = RotationUtilsNumpy.mat_to_r6d(interp_rotations.as_matrix())

        res_6d = res_6d.transpose(0, 1, 3, 2).reshape(batch, -1, size)
        return res_6d

    # ==============================
    # Quaternions
    # ==============================

    @staticmethod
    def mag(q):
        """Return magnitude of quaternion"""
        return np.linalg.norm(q, axis=-1, keepdims=True)

    @staticmethod
    def conj(q):
        """Returns conjugate of quaternion"""
        return np.concatenate((q[..., :1], q[..., -3:] * -1), axis=-1)

    @staticmethod
    def inv(q):
        """Returns inverse of quaternion"""
        return RotationUtilsNumpy.conj(q) / RotationUtilsNumpy.mag(q)

    @staticmethod
    def normalize(q):
        """Returns normalized quaternion"""
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    @staticmethod
    def mul(q, r):
        """Multiply quaternion(s) q with quaternion(s) r"""
        q_orig = q.shape
        q = q.reshape(-1, 4)
        r = r.reshape(-1, 4)

        w = q[:,0]*r[:,0] - q[:,1]*r[:,1] - q[:,2]*r[:,2] - q[:,3]*r[:,3]
        x = q[:,0]*r[:,1] + q[:,1]*r[:,0] + q[:,2]*r[:,3] - q[:,3]*r[:,2]
        y = q[:,0]*r[:,2] - q[:,1]*r[:,3] + q[:,2]*r[:,0] + q[:,3]*r[:,1]
        z = q[:,0]*r[:,3] + q[:,1]*r[:,2] - q[:,2]*r[:,1] + q[:,3]*r[:,0]

        return np.stack((w, x, y, z), axis=-1).reshape(q_orig)

    @staticmethod
    def rot(q, v):
        """Rotate vector(s) v about the rotation described by quaternion(s) q"""
        q_orig = q.shape
        v_orig = v.shape
        q = q.reshape(-1, 4)
        v = v.reshape(-1, 3)

        qvec = q[:, 1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return (v + 2 * (q[:, :1] * uv + uuv)).reshape(v_orig)

    @staticmethod
    def quat_to_mat(q):
        """Convert (w, x, y, z) quaternions to 3x3 rotation matrix"""
        return R.from_quat(q.reshape(-1, 4), scalar_first=True).as_matrix().reshape(q.shape[:-1] + (3, 3))

    @staticmethod
    def mat_to_quat(mat):
        """Convert 3x3 rotation matrices to (w, x, y, z) quaternions."""
        orig_shape = mat.shape
        mat_flat = mat.reshape(-1, 3, 3)
        quats = R.from_matrix(mat_flat).as_quat(scalar_first=True)
        return quats.reshape(orig_shape[:-2] + (4,)).astype(np.float32)

    @staticmethod
    def quat_to_euler(q, order='xyz', degrees=True):
        """Convert (w, x, y, z) quaternions to xyz euler angles."""
        return R.from_quat(q.reshape(-1, 4), scalar_first=True).as_euler(order, degrees=degrees).reshape(q.shape[:-1] + (3,))

    @staticmethod
    def slerp(q0, q1, t):
        """Spherical Linear Interpolation between quaternions."""
        orig_shape = q0.shape
        q0 = q0.reshape(-1, 4)
        q1 = q1.reshape(-1, 4)
        t = np.atleast_1d(t).reshape(-1)

        if len(t) == 1:
            t = np.repeat(t, len(q0))

        q0_n = RotationUtilsNumpy.normalize(q0)
        q1_n = RotationUtilsNumpy.normalize(q1)

        dot = np.sum(q0_n * q1_n, axis=-1)
        q1_n[dot < 0] *= -1
        dot = np.clip(np.abs(dot), -1.0, 1.0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        res = np.zeros_like(q0_n)
        mask = sin_theta > 1e-6

        # Linear fallback for small angles
        res[~mask] = (1 - t[~mask, None]) * q0_n[~mask] + t[~mask, None] * q1_n[~mask]

        # Slerp for larger angles
        w0 = np.sin((1 - t[mask]) * theta[mask]) / sin_theta[mask]
        w1 = np.sin(t[mask] * theta[mask]) / sin_theta[mask]
        res[mask] = w0[:, None] * q0_n[mask] + w1[:, None] * q1_n[mask]

        return RotationUtilsNumpy.normalize(res).reshape(orig_shape)

    @staticmethod
    def fix_continuity(q):
        """Enforce quaternion continuity across the time dimension."""
        result = q.copy()
        dot_products = np.sum(q[1:] * q[:-1], axis=-1)
        mask = dot_products < 0
        mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
        result[1:][mask] *= -1
        return result

    # ==============================
    # Angle Axis & Exponential Map
    # ==============================

    @staticmethod
    def expmap_to_quat(e):
        """Convert axis-angle rotations (aka exponential maps) to quaternions."""
        original_shape = list(e.shape)
        original_shape[-1] = 4
        e = e.reshape(-1, 3)

        theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
        w = np.cos(0.5 * theta).reshape(-1, 1)
        xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
        return np.concatenate((w, xyz), axis=1).reshape(original_shape)
