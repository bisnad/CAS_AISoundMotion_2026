"""
calculations and conversions with 6d representation of rotations, operates on torch tensors
"""

import torch
import common.quaternion_torch as tquat

def quat2repr6d(quat):
    """
    from paper: GANimator (tested)
    """
    
    mat = tquat.quat2mat(quat)
    res = mat[..., :2, :]
    res = res.reshape(res.shape[:-2] + (6, ))
    return res

def repr6d2mat(repr):
    """
    from paper: GANimator (tested)
    """
    
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def repr6d2quat(repr):
    """
    from paper: GANimator (tested)
    """
    
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return tquat.mat2quat(mat)

def interpolate_6d(input, size):
    """
    from paper: GANimator
    
    :param input: (batch_size, n_channels, length)
    :param size: required output size for temporal axis
    :return:
    """
    batch = input.shape[0]
    length = input.shape[-1]
    input = input.reshape((batch, -1, 6, length))
    input = input.permute(0, 1, 3, 2)     # (batch_size, n_joint, length, 6)
    input_q = repr6d2quat(input)
    idx = torch.tensor(list(range(size)), device=input_q.device, dtype=torch.float) / size * (length - 1)
    idx_l = torch.floor(idx)
    t = idx - idx_l
    idx_l = idx_l.long()
    idx_r = idx_l + 1
    t = t.reshape((1, 1, -1))
    res_q = tquat.slerp(input_q[..., idx_l, :], input_q[..., idx_r, :], t, unit=True)
    res = quat2repr6d(res_q)  # shape = (batch_size, n_joint, t, 6)
    res = res.permute(0, 1, 3, 2)
    res = res.reshape((batch, -1, size))
    return res

    