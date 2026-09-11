"""
conversions and calculations of dual quaternion representation of positions and rotations, operate on numpy arrays
representation per quaternion: w, x, y, z
"""

import numpy as np
import common.quaternion_np as nquat

def normalize(q_r, q_d):
    return nquat.normalize(q_r), nquat.normalize(q_d)

def dconj(q_r, q_d):
    """
    Return the dual number conjugate (qr, qd)* = (qr, -qd) (tested)
    This form of conjugate is seldom used.
    """
    
    return q_r, -q_d

def cconj(q_r, q_d):
    """
    Return the combination of the quaternion conjugate and dual number conjugate (tested)
    (qr, qd)* = (qr*, -qd*)
    """
    
    return nquat.conj(q_r), -nquat.conj(q_d)

def inv(q_r, q_d):
    """
    Return the dual quaternion inverse (tested)
    """
    
    q_r_inv = nquat.inv(q_r)
    q_d_inv = nquat.mul(nquat.mul(-q_r_inv, q_d), q_r_inv)
    
    return q_r_inv, q_d_inv

def translation(q_r, q_d):
    """
    Get the translation component of the dual quaternion in vector form (tested)
    """
    
    mult = nquat.mul((2.0 * q_d), nquat.conj(nquat.normalize(q_r)))
    return mult[1:]
    

def mul(q1_r, q1_d, q2_r, q2_d):
    """
    Dual quaternion multiplication (tested)

    :return product: DualQuaternion object. Math:
        dq1 * dq2 = q1_r * q2_r + (q1_r * q2_d + q1_d * q2_r) * eps
    """
    
    q_r_prod = nquat.mul(q1_r, q2_r)
    q_d_prod = nquat.mul(q1_r, q2_d) + nquat.mul(q1_d, q2_r)
    
    return q_r_prod, q_d_prod

def smul(q_r, q_d, sc):
    """
    Multiplication with a scalar (tested)
    """
    
    return q_r * sc, q_d * sc

def div(q1_r, q1_d, q2_r, q2_d):
    """
    Dual quaternion division. (tested)
    """
    
    q2_r_sq = nquat.mul(q2_r, q2_r)
    div_r = nquat.mul(nquat.mul(q1_r, q2_r), nquat.inv(q2_r_sq))
    div_d = nquat.mul((nquat.mul(q2_r, q1_d) - nquat.mul(q1_r, q2_d)), nquat.inv(q2_r_sq))
    
    return div_r, div_d

def add(q1_r, q1_d, q2_r, q2_d):
    """
    Dual quaternion addition. (tested)
    """
    
    return q1_r + q2_r, q1_d + q2_d
    
  
def eq(q1_r, q1_d, q2_r, q2_d):

    return (q1_r == q2_r or q1_r == -q2_r) and (q1_d == q2_d or q1_d == -q2_d)

def transform_point(q1_r, q1_d, pt):
        """
        Convenience function to apply the transformation to a given vector. (tested)
        """
        
        qv_r = np.array([1.0, 0.0, 0.0, 0.0])
        qv_d = np.hstack((0, pt))
        resq_r, resq_d = mul(*mul(q1_r, q1_d, qv_r, qv_d), *cconj(q1_r, q1_d))
        
        return resq_d[1:]
    
def identity():
    
    return np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])
    
    
def hmat2dq(matrix):
    """
    Create dual quaternion from a 4 by 4 homogeneous transformation matrix (tested)
    """

    q_r = nquat.normalize(nquat.mat2quat(matrix[:3, :3]))
    v_t = matrix[:3, 3]
    
    q_d = nquat.mul(0.5 * np.array([0, v_t[0], v_t[1], v_t[2]]), q_r)
    
    return q_r, q_d

def dq2hmat(q_r, q_d):
    """
    Homogeneous 4x4 transformation matrix from the dual quaternion (tested)
    """
    
    hmat = np.eye(4)
    hmat[:3, :3] = nquat.quat2mat(q_r)
    hmat[:3, 3] = translation(q_r, q_d)
    
    return hmat
    

def qtvec2dq(q_r, v_t):
    """
    Create a dual quaternion from a quaternion q_r and translation v_t (tested)
    """
    
    q_d = nquat.mul(0.5 * np.array([0.0, v_t[0], v_t[1], v_t[2]]), q_r)
    
    return q_r, q_d

def dq2qtvec(q_r, q_t):
    """
    return dual quaternion as a quaternion followed by the translation vector (tested)
    given a dual quaternion p + eq, the rotation in quaternion form is p and the translation in
    quaternion form is 2qp*
    """

    return q_r, translation(q_r, q_t)


def tvec2dq(v_t):
    """
    Create dual quaternion from a cartesian point (tested)
    """
    
    q_r = np.array([1.0, 0.0, 0.0, 0.0])
    q_d = nquat.mul(0.5 * np.array([0, v_t[0], v_t[1], v_t[2]]), q_r)
    
    return q_r, q_d
    

def normalize(q_r, q_d):
    """
    Normalize dual quaternion (tested)
    """
    
    norm_qr = nquat.mag(q_r)
    
    return q_r / norm_qr, q_d / norm_qr

def pow(q_r, q_d, exp):
    """
    exponent (semi-tested)
    """
    
    #print("pow q_r ", q_r, " q_d ", q_d)

    exp = float(exp)
    
    #print("q_r[0] ", q_r[0])
    
    theta = 2 * np.arccos(q_r[0])
    
    #print("theta ", theta)
    
    if np.isclose(theta, 0):
        t_v = exp * translation(q_r, q_d)
        return tvec2dq(t_v)
    else:
        s0 = q_r[1:] / np.sin(theta / 2)
        
        #print("s0 ", s0)
        
        d = -2. * q_d[0] / np.sin(theta / 2)
        
        #print("d ", d)
        
        #print("q_d[1:] ", q_d[1:])
        
        se = (q_d[1:] - s0 * d / 2 * np.cos(theta / 2)) / np.sin(theta / 2)
        
        #print("se ", se)
        
    powq_r = np.hstack((np.cos(exp * theta / 2), np.sin(exp * theta / 2) * s0))
    
    powq_d_s = -exp * d / 2 * np.sin(exp * theta / 2)
    powq_d_v = exp * d / 2 * np.cos(exp * theta / 2) * s0 + np.sin(exp * theta / 2) * se
    powq_d = np.hstack((powq_d_s, powq_d_v))
    
    return powq_r, powq_d

    
def sclerp(q1_r, q1_d, q2_r, q2_d, t):
    """
    Screw Linear Interpolation (semi-tested)

    Generalization of Quaternion slerp (Shoemake et al.) for rigid body motions
    ScLERP guarantees both shortest path (on the manifold) and constant speed
    interpolation and is independent of the choice of coordinate system.
    ScLERP(dq1, dq2, t) = dq1 * dq12^t where dq12 = dq1^-1 * dq2
    """
    
    # ensure we always find closest solution. See Kavan and Zara 2005
    if nquat.mul(q1_r, q2_r)[0] < 0:
        q1_r *= -1
        
    #print("start r", q1_r, " d ", q1_d)
    #print("start inv ", inv(q1_r, q1_d))
    #print("stop r", q2_r, " d ", q2_d)
    #print("start * (start.inverse() * stop) ", mul(*mul(q1_r, q1_d, *inv(q1_r, q1_d)), q2_r, q2_d))
    
    return mul(q1_r, q1_d, *pow( *(mul(*inv(q1_r, q1_d), q2_r, q2_d)), t))



def localquats2currentdq(lq, offsets, parh,joints_num = 32):
    """takes in local quaternion, offsets, and parents to produce hierarchy-aware dual quaternions

    inputs
    ------
    lq: array of local quaternions, size: (T,J*4) (#frames x (number of joints used*4))
    offsets: array, size: #joints used x 3
    parh: parents list , for us size = 31 (I think for MotioNet - the joints that the network predicts)


    outputs
    -------
    allcq: current dual quaternions for each joint, size: #frames x (#joints used *8)
    """
    allcq = []
    # print(parh)
    for ff in range(lq.shape[0]):
        cq = {}
        for i in range(joints_num):
            if i == 0:
                cq[i] = DualQuaternion.from_quat_pose_array(
                    list(lq[ff, i * 4:i * 4 + 4]) + list(offsets[0])).normalized().dq_array()
            else:
                cq[i] = (DualQuaternion.from_dq_array((cq[parh[i]])) * DualQuaternion.from_quat_pose_array(
                    list(lq[ff, i * 4:i * 4 + 4]) + list(offsets[i]))).normalized().dq_array()
        temp = []
        for i in cq.items():
            temp.append(i[1][0])
            temp.append(i[1][1])
            temp.append(i[1][2])
            temp.append(i[1][3])
            temp.append(i[1][4])
            temp.append(i[1][5])
            temp.append(i[1][6])
            temp.append(i[1][7])
        allcq.append(temp)
    allcq = np.array(allcq)
    return allcq
