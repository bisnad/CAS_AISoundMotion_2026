"""
quaternion calculations and conversions that operate on torch tensors
representation: w, x, y, z

code partially adopted from the following publications:
    Modeling Human Motion with Quaternion-Based Neural Networks
    GANimator: Neural Motion Synthesis from a Single Sequence
"""

import torch
import numpy as np

def mag(q):
    """
    return magnitude of quaternion (tested)
    """

    assert q.shape[-1] == 4
    
    return torch.linalg.norm(q, dim=-1, keepdim=True)

def conj(q):
    """
    returns conjugate of quaternion (tested)
    """
    assert q.shape[-1] == 4
    
    return torch.concat((q[..., :1], q[..., -3:] * -1), dim=-1)

def inv(q):
    """ 
    returns inverse of quaternion (tested)
    """
    
    assert q.shape[-1] == 4
    
    return conj(q) / mag(q)

def normalize(q):
    """
    returns normalised quaternion (tested)
    """
    assert q.shape[-1] == 4
    
    return torch.nn.functional.normalize(q, dim=-1)

def mul(q, r):
    """
    from paper: Quaternet 
    
    Multiply quaternion(s) q with quaternion(s) s
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions
    Returns q*r as a tensor of shape (*, 4)
    """

    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def rot(q, v):
    """
    from paper: Quaternet(tested)
    
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def aa2quat(rots, form='wxyz', unified_orient=True):
    """
    from paper: Ganimator (tested)
    
    Convert angle-axis representation to wxyz quaternion and to the half plan (w >= 0)
    @param rots: angle-axis rotations, (*, 3)
    @param form: quaternion format, either 'wxyz' or 'xyzw'
    @param unified_orient: Use unified orientation for quaternion (quaternion is dual cover of SO3)
    :return:
    """
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

def quat2aa(quats):
    """
    from paper: Ganimator (tested)
    
    Convert wxyz quaternions to angle-axis representation
    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)


def quat2euler(q, order='xyz', degrees=True):
    """
    from paper: Ganimator
    
    Convert (w, x, y, z) quaternions to xyz euler angles. This is  used for bvh output.
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.empty(q0.shape + (3,), device=q.device, dtype=q.dtype)

    if order == 'xyz':
        es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)

    if degrees:
        es = es * 180 / np.pi

    return es

"""
TODO: implement euler2quat
"""

def quat2mat(quats: torch.Tensor):
    """
    from paper: Ganimator (tested)
    
    Convert (w, x, y, z) quaternions to 3x3 rotation matrix
    :param quats: quaternions of shape (..., 4)
    :return:  rotation matrices of shape (..., 3, 3)
    """
    qw = quats[..., 0]
    qx = quats[..., 1]
    qy = quats[..., 2]
    qz = quats[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quats.shape[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m

def mat2quat(R):
    """
    from paper: Ganimator (tested)
    
    https://github.com/duolu/pyrotation/blob/master/pyrotation/pyrotation.py
    Convert a rotation matrix to a unit quaternion.

    This uses the Shepperd’s method for numerical stability.
    """

    # The rotation matrix must be orthonormal

    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    # if R[..., 2, 2] < 0:
    #
    #     if R[..., 0, 0] > R[..., 1, 1]:
    #
    #         x = torch.sqrt(x2)
    #         w = wx / x
    #         y = xy / x
    #         z = xz / x
    #
    #     else:
    #
    #         y = torch.sqrt(y2)
    #         w = wy / y
    #         x = xy / y
    #         z = yz / y
    #
    # else:
    #
    #     if R[..., 0, 0] < -R[..., 1, 1]:
    #
    #         z = torch.sqrt(z2)
    #         w = wz / z
    #         x = xz / z
    #         y = yz / z
    #
    #     else:
    #
    #         w = torch.sqrt(w2)
    #         x = wx / w
    #         y = wy / w
    #         z = wz / w

    res = [w, x, y, z]
    res = [z.unsqueeze(-1) for z in res]

    return torch.cat(res, dim=-1) / 2

def slerp(q0, q1, t=0.5, unit=True):
    """
    from paper: Ganimator (tested)
    
    :param q0: shape = (*, n)
    :param q1: shape = (*, n)
    :param t: shape = (*)
    :param unit: If q0 and q1 are unit vectors
    :return: res: shape = (*, n)
    """
    eps = 1e-8
    if not unit:
        q0_n = q0 / torch.norm(q0, dim=-1, keepdim=True)
        q1_n = q1 / torch.norm(q1, dim=-1, keepdim=True)
    else:
        q0_n = q0
        q1_n = q1
    omega = torch.acos((q0_n * q1_n).sum(dim=-1).clamp(-1, 1))
    dom = torch.sin(omega)

    flag = dom < eps

    res = torch.empty_like(q0_n)
    t_t = t[flag].unsqueeze(-1)
    res[flag] = (1 - t_t) * q0_n[flag] + t_t * q1_n[flag]

    flag = ~ flag

    t_t = t[flag]
    d_t = dom[flag]
    va = torch.sin((1 - t_t) * omega[flag]) / d_t
    vb = torch.sin(t_t * omega[flag]) / d_t
    res[flag] = (va.unsqueeze(-1) * q0_n[flag] + vb.unsqueeze(-1) * q1_n[flag])
    return res