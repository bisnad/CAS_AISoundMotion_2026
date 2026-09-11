# quaternion math implemented using tensorflow tensors
# code adapted from original pyTorch implementation of Quaternet 
# quaternet / common / quaternion.py

import torch
import numpy as np

def qmul(q, r):
    """
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

def qrot(q, v):
    """
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

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

# Numpy-backed implementations

def qnormalize_np(q):
    q_orig_shape = q.shape
    q = np.reshape(q, (-1, 4))
    norm = np.linalg.norm(q, axis=1) + 0.000001
    norm = np.reshape(norm, (-1, 1))

    q_norm = q / norm    
    q_norm = np.reshape(q_norm, q_orig_shape)
    
    return q_norm

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)

def slerp(q0, q1, amount=0.5):
    # TODO: modify this code so it works for arrays of quaternions
    
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    q0 = qnormalize_np(q0)
    q1 = qnormalize_np(q1)

    amount = np.clip(amount, 0, 1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        qr = qnormalize_np(qr)
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)
        
    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr = qnormalize_np(qr)
    return qr

def slerp_np(q0, q1, amount=0.5):
    orig_shape = q0.shape

    q0 = np.reshape(q0, (-1, 4))
    q1 = np.reshape(q1, (-1, 4))

    
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    q0 = qnormalize_np(q0)
    q1 = qnormalize_np(q1)

    amount = np.clip(amount, 0, 1)
    
    dot = np.sum(np.multiply(q0, q1), axis=1)


    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    negdot = dot > 0.0
    negdot = negdot.astype(np.float32)
    negdot = negdot * 2.0 - 1.0
    dot = dot * negdot
    
    negdot = np.expand_dims(negdot, axis=1)

    
    negdot = np.repeat(negdot, 4, axis=1)

    
    q0 = q0 * negdot

    # sin_theta_0 can not be zero
    bigdot = dot > 0.9995
    bigdot = bigdot.astype(np.float32)
    bigqr = q0 + (q1 - q0) * np.repeat(np.reshape(amount, (-1, 1)), 4, axis=1)
    bigqr = qnormalize_np(bigqr)
    dot = np.clip(dot, float('-inf'), 0.9995)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * amount
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s0 = np.repeat(np.reshape(s0, (-1, 1)), 4, axis=1)
    
    s1 = sin_theta / sin_theta_0
    s1 = np.repeat(np.reshape(s1, (-1, 1)), 4, axis=1)
    
    smallqr = (s0 * q0) + (s1 * q1)
    invbigdot = np.ones_like(bigdot) - bigdot
    
    qr = bigqr * np.repeat(np.reshape(bigdot, (-1, 1)), 4, axis=1) + smallqr * np.repeat(np.reshape(invbigdot, (-1, 1)), 4, axis=1)
    qr = qnormalize_np(qr)
    
    qr = np.reshape(qr, orig_shape)
    
    return qr

def slerp2(q0, q1, amount):
    
    orig_shape = q0.shape
    
    q0 = torch.reshape(q0, (-1, 4))
    q1 = torch.reshape(q1, (-1, 4))
    amount = torch.clamp(amount, 0.0, 1.0)
    
    #Ensure quaternion inputs are unit quaternions and 0 <= amount <= 1	
    q0 = torch.nn.functional.normalize(q0)
    q1 = torch.nn.functional.normalize(q1)
    
    dot = torch.bmm(torch.reshape(q0, (-1, 1, 4)), torch.reshape(q1, (-1, 4, 1)))
    dot = torch.squeeze(dot)
    
    negdot = dot > 0.0
    negdot = negdot.to(torch.float32)
    negdot = negdot * 2.0 - 1.0
    
    dot = dot * negdot
    
    negdot = torch.unsqueeze(negdot, 1)
    negdot = negdot.repeat((1, 4))
    
    q0 = q0 * negdot
    
    bigdot = dot > 0.9995
    bigdot = bigdot.to(torch.float32)
    
    #calculate qr as if dot was bigger than 0.9995
    bigqr = q0 + (q1 - q0) * torch.reshape(amount, (-1, 1)).repeat((1, 4))
    
    #calculate qr as if dot was smaller than 0.9995
    dot = torch.clamp_max(dot, 0.9995)
    theta_0 = torch.arccos(dot)
    
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * amount
    
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s0 = torch.reshape(s0, (-1, 1)).repeat((1, 4))
    
    s1 = sin_theta / sin_theta_0
    s1 = torch.reshape(s1, (-1, 1)).repeat((1, 4))
    
    smallqr = (s0 * q0) + (s1 * q1)
    
    invbigdot = torch.ones_like(bigdot) - bigdot
    
    qr = bigqr * torch.reshape(bigdot,(-1, 1)).repeat((1, 4)) + smallqr * torch.reshape(invbigdot, (-1, 1)).repeat((1, 4))
    qr = torch.nn.functional.normalize(qr)
    qr = torch.reshape(qr, orig_shape)
    
    return qr