"""
conversions and calculations of dual quaternion representation of positions and rotations, operate on numpy arrays
representation per quaternion: w, x, y, z
"""

import numpy as np
import common.quaternion_np as nquat

def is_normalized(q_r, q_d):
    """Check if the dual quaternion is normalized"""
    if np.isclose(np.linalg.norm(q_r), 0):
        return True

    rot_normalized = np.isclose(np.linalg.norm(q_r), 1)
    trans_normalized = (q_d / np.linalg.norm(q_r)) == q_d
    
    rot_normalized = (rot_normalized == True).all()
    trans_normalized = (trans_normalized == True).all()
    
    return rot_normalized and trans_normalized

def dconj(q_r, q_d):
    """
    Return the dual number conjugate (qr, qd)* = (qr, -qd) (tested)
    This form of conjugate is seldom used.
    """
    
    return q_r, -q_d

def cconj(q_r, q_d):
    """
    Return the combination of the quaternion conjugate and dual number conjugate
    (qr, qd)* = (qr*, -qd*)

    This form is commonly used to transform a point
    See also DualQuaternion.dual_number_conjugate() and DualQuaternion.quaternion_conjugate().
    """
    
    return nquat.conj(q_r), nquat.conj(-q_d)

def conj(q_r, q_d):
    """
    Return the individual quaternion conjugates (qr, qd)* = (qr*, qd*)

    This is equivalent to inverse of a homogeneous matrix. It is used in applying
    a transformation to a line expressed in Plucker coordinates.
    See also DualQuaternion.dual_conjugate() and DualQuaternion.combined_conjugate().
    """
    
    return nquat.conj(q_r), nquat.conj(q_d)

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
    
    #mult = nquat.mul((2.0 * q_d), nquat.conj(nquat.normalize(q_r)))
    mult = nquat.mul((2.0 * q_d), nquat.conj(q_r))
    return mult[1:]
    

def mul(q1_r, q1_d, q2_r, q2_d):
    """
    Dual quaternion multiplication (tested)

    :return product: DualQuaternion object. Math:
        dq1 * dq2 = q1_r * q2_r + (q1_r * q2_d + q1_d * q2_r) * eps
    """
    
    # ugly: fix later
    q1_r = q1_r.astype(np.float32)
    q1_d = q1_d.astype(np.float32)
    q2_r = q2_r.astype(np.float32)
    q2_d = q2_d.astype(np.float32)
    
    #print("q1_r d ", q1_r.dtype, " q1_d d ", q1_d.dtype, " q2_r d ", q2_r.dtype, " q2_d d ", q2_d.dtype)

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
    
    #q_r = nquat.mat2quat(matrix[:3, :3])
    q_r = nquat.normalize(nquat.mat2quat(matrix[:3, :3]))
    
    v_t = matrix[:3, 3]
    
    q_d = nquat.mul(np.array([0.0, v_t[0], v_t[1], v_t[2]]) * 0.5, q_r)
 
    
    #q_d = nquat.mul(0.5 * np.array([0, v_t[0], v_t[1], v_t[2]]), q_r)
    
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
    q_r = nquat.normalize(q_r)
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
    exponent (tested)
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
    
    powq_r = powq_r.astype(q_r.dtype)
    powq_d = powq_r.astype(q_d.dtype)
    
    #print("pow q_r d ", q_r.dtype, " q_d d ", q_d.dtype, " powq_r d ", powq_r.dtype, " powq_d d ", powq_d.dtype)
    
    return powq_r, powq_d

# from Achllle (https://github.com/Achllle/dual_quaternions)
def screw(q_r, q_d):
    # tested
    """
    Get the screw parameters for this dual quaternion.
    Chasles' theorem (Mozzi, screw theorem) states that any rigid displacement is equivalent to a rotation about
    some line and a translation in the direction of the line. This line does not go through the origin!
    This function returns the Plucker coordinates for the screw axis (l, m) as well as the amount of rotation
    and translation, theta and d.
    If the dual quaternion represents a pure translation, theta will be zero and the screw moment m will be at
    infinity.

    :return: l (unit length), m, theta, d
    :rtype np.array(3), np.array(3), float, float
    """
    
    # start by extracting theta and l directly from the real part of the dual quaternion
    theta = nquat.angle(q_r)
    theta_close_to_zero = np.isclose(theta, 0)
    t = translation(q_r, q_d)

    if not theta_close_to_zero:
        vec = nquat.vector(q_r)
        l = vec / np.sin(theta/2)  # since q_r is normalized, l should be normalized too

        # displacement d along the line is the projection of the translation onto the line l
        d = np.dot(t, l)

        # m is a bit more complicated. Derivation see K. Daniliidis, Hand-eye calibration using Dual Quaternions
        m = 0.5 * (np.cross(t, l) + np.cross(l, np.cross(t, l) / np.tan(theta / 2)))
    else:
        # l points along the translation axis
        d = np.linalg.norm(t)

        if not np.isclose(d, 0):  # unit transformation
            l = t / d
        else:
            l = (0, 0, 0)
            
        m = np.array([np.inf, np.inf, np.inf])

    return l, m, theta, d

# from Achllle (https://github.com/Achllle/dual_quaternions)
def from_screw(l, m, theta, d):
        """
        Create a DualQuaternion from screw parameters

        :param l: unit vector defining screw axis direction
        :param m: screw axis moment, perpendicular to l and through the origin
        :param theta: screw angle; rotation around the screw axis
        :param d: displacement along the screw axis
        """
        l = np.array(l)
        m = np.array(m)
        if not np.isclose(np.linalg.norm(l), 1):
            raise AttributeError("Expected l to be a unit vector, received {} with norm {} instead"
                                 .format(l, np.linalg.norm(l)))
        theta = float(theta)
        d = float(d)
        
        q_r = np.hstack((np.array([np.cos(theta/2)]), np.sin(theta/2) * l))
        q_d = np.hstack((np.array([-d/2 * np.sin(theta/2)]), np.sin(theta/2) * m + d/2 * np.cos(theta/2) * l))
        
        return q_r, q_d

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
    
    #print("sclerp q1_r.d ", q1_r.dtype, " q1_d d ", q1_d.dtype, " q2_r d ", q2_r.dtype, " q2_d d ", q2_d.dtype, " t d ", t.dtype )
    
    return mul(q1_r, q1_d, *pow( *(mul(*inv(q1_r, q1_d), q2_r, q2_d)), t))
