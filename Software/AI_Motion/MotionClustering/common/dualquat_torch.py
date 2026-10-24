"""
conversions and calculations of dual quaternion representation of positions and rotations, operate on torch tensors

code from paper: Pose Representations for Deep Skeletal Animaion
"""

import torch
import torch.nn.functional as nnF
import common.quaternion_torch as tquat

def conj(dq):
    """
    Returns conjugate of dual quaternion
    inputs
    -------
    q: torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,8), conjugate dual quaternion """
    return torch.cat((dq[:,0][:,None],-dq[:,1][:,None],-dq[:,2][:,None],-dq[:,3][:,None],-dq[:,4][:,None],dq[:,5][:,None],dq[:,6][:,None],dq[:,7][:,None]),dim=1)

def inv(dq):
    """
    Returns inverse of dual quaternion
    inputs
    -------
    q: torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,8), inverse dual quaternion """
    return torch.cat((dq[:,0][:,None],-dq[:,1][:,None],-dq[:,2][:,None],-dq[:,3][:,None],dq[:,4][:,None],-dq[:,5][:,None],-dq[:,6][:,None],-dq[:,7][:,None]),dim=1)

def mul(dq1, dq2):  # accepts normalized quaternions (*,8)
    """
    Multiplies two normalized dual quaternions. 
    inputs
    -------
    dq1: dual quaternions, torch.tensor, shape: (*,8)
    dq2: dual quaternions, torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor, shape: (*,8), multiplication result
    """
    q = dq1[:, :4]
    r = dq2[:, :4]

    d_q = dq1[:, 4:]
    d_r = dq2[:, 4:]

    q_ = tquat.mul(q, r)
    d_ = tquat.mul(q, d_r) + tquat.mul(d_q, r)

    assert (q_ == tquat.mul(q, r)).all()
    return torch.cat((q_, d_), 1)
    

def rot(dq_trans,dq_point):
    """
    Rotates a point by a dual quaternion
    inputs
    -------
    dq_trans: transformation, torch.tensor, shape: (*,8)
    dq_point: point, torch.tensor, shape: (*,8)
    
    outputs
    -------
    torch.tensor (*,3), rotated point """
    return mul(mul(dq_trans,dq_point),conj(dq_trans))[:,5:]

def is_unit(dq,stop=True):
    """
    Checks if dual quaternion is unit.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)
    stop: bool, stop if not unit
    
    outputs
    -------
    """
    if stop==True:
        assert (torch.norm(dq[:,:4], dim=1)-1<1e-4).all() and (torch.sum(dq[:,:4]*dq[:,4:],dim=1)<1e-3).all()
    return (torch.norm(dq[:,:4], dim=1)-1<1e-4).all() and (torch.sum(dq[:,:4]*dq[:,4:],dim=1)<1e-3).all()

def normalize(dq,force=False):
    """
    Normalizes dual quaternion.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)
    force: bool, force normalize (used for network outputs)
    
    outputs
    -------
    torch.tensor, shape: (*,8), normalized dual quaternions
    """
    quats = dq[:,:4]
    dualquats = dq[:,4:]
    quats_normalized = nnF.normalize(quats,dim=1)
    norm = torch.norm(quats,dim=1)
    norm =torch.stack((norm,norm,norm,norm),dim=1)
    dualquats_normalized = torch.div(dualquats,norm)
    if force:
        if is_unit(dq,stop=False)==False:

            q = dq[:, :4]
            d = dq[:, 4:]

            qnorm = torch.norm(q, dim=1)
            quats_normalized = nnF.normalize(q,dim=1) #torch.div(q, qnorm[:, None])

            qd = torch.sum(q * d, dim=1)
            dualquats_normalized = torch.div(d, qnorm[:, None]) - quats_normalized * torch.div(qd, qnorm ** 2)[:, None]
            if is_unit(torch.cat((quats_normalized,dualquats_normalized),1))==False:
                print(q,d)
                print(dq)
    return torch.cat((quats_normalized,dualquats_normalized),1)


def translation(dq):
    """
    Extract translation component.
    inputs
    -------
    dq: dual quaternions, torch.tensor, shape: (*,8)

    outputs
    -------
    torch.tensor, shape: (*,3), translation
    """
    dq = normalize(dq)
    dualquats_normalized = dq[:,4:]
    qt = tquat.conj(dq[:,:4])
    return tquat.mul(torch.mul(2,dualquats_normalized),qt)[:,1:]

