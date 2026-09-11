import os
from unittest import TestCase
import common.dualquat_np as ndquat
import numpy as np

class TestDualQuaternion(TestCase):

    def setUp(self):
        self.identity_dq_r,self.identity_dq_d  = ndquat.identity()
        self.random_dq_r, self.random_dq_d = ndquat.qtvec2dq(np.array([1.0,2.0,3.0,4.0]), np.array([5.0,6.0,7.0]))
        self.random_dq_r, self.random_dq_d = ndquat.normalize(self.random_dq_r, self.random_dq_d)
        self.other_random_dq_r, self.other_random_dq_d = ndquat.qtvec2dq(np.array([0.2,0.1,0.3,0.07]), np.array([1.2,0.9,0.2]))
        self.other_random_dq_r, self.other_random_dq_d = ndquat.normalize(self.other_random_dq_r, self.other_random_dq_d)
        self.normalized_dq_r, self.normalized_dq_d = ndquat.normalize(self.random_dq_r, self.random_dq_d)

        """
        print("self.identity_dq r ", self.identity_dq_r, " d ", self.identity_dq_d, " t ", ndquat.translation(self.identity_dq_r,self.identity_dq_d))
        print("self.random_dq r ", self.random_dq_r, " d ", self.random_dq_d, " t ", ndquat.translation(self.random_dq_r, self.random_dq_d))
        print("self.other_random_dq r ", self.other_random_dq_r, " d ", self.other_random_dq_d, " t ", ndquat.translation(self.other_random_dq_r, self.other_random_dq_d))
        print("self.normalized_dq r ", self.normalized_dq_r, " d ", self.normalized_dq_d, " t ", ndquat.translation(self.normalized_dq_r, self.normalized_dq_d))
        """
        
    def equal(self, dq1_r, dq1_d, dq2_r, dq2_d):
        return (dq1_r == dq2_r).all() and (dq1_d == dq2_d).all()

    def test_creation(self):
        
        # from dual quaternion array: careful, need to supply a normalized DQ
        dql_r = np.array([0.7071067811, 0.7071067811, 0, 0])
        dql_d = np.array([-3.535533905, 3.535533905, 1.767766952, -1.767766952])
        
        dq1_r = np.copy(dql_r)
        dq1_d = np.copy(dql_d)
        dq2_r = np.copy(dql_r)
        dq2_d = np.copy(dql_d)
        
        """
        print("dq1_r ", dq1_r, " d ", dq1_d, " t ", ndquat.translation(dq1_r, dq1_d))    
        print("dq2_r ", dq2_r, " d ", dq2_d, " t ", ndquat.translation(dq2_r, dq2_d))    
        """
            
        assert((dq1_r == dq2_r).all() and (dq1_d == dq2_d).all())
        
        # from quaternion + translation array
        dq3_r, dq3_d = ndquat.qtvec2dq(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0]))
        dq3_r, dq3_d = ndquat.normalize(dq3_r, dq3_d)
        dq4_r, dq4_d = ndquat.qtvec2dq(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0]))
        dq4_r, dq4_d = ndquat.normalize(dq4_r, dq4_d)
        
        """
        print("dq3_r ", dq3_r, " d ", dq3_d, " t ", ndquat.translation(dq3_r, dq3_d))    
        print("dq4_r ", dq4_r, " d ", dq4_d, " t ", ndquat.translation(dq4_r, dq4_d))    
        """
        
        self.equal(dq3_r, dq3_d, dq4_r, dq4_d)
        
        # from homogeneous transformation matrix
        
        T = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        dq7_r, dq7_d = ndquat.hmat2dq(T)
        
        """
        print("dq7_r ", dq7_r, " d ", dq7_d, " t ", ndquat.translation(dq7_r, dq7_d)) 
        """
        
        assert((dq7_r == np.array([1.0, 0.0, 0.0, 0.0])).all())
        assert((ndquat.translation(dq7_r, dq7_d) == np.array([2.0, 3.0, 1.0])).all())
        
        try:
            np.testing.assert_array_almost_equal(ndquat.dq2hmat(dq7_r, dq7_d), T)
        except AssertionError as e:
            self.fail(e)
        
        # from a point
        
        dq8_r, dq8_d = ndquat.tvec2dq(np.array([4.0, 6.0, 8.0]))
        
        """
        print("dq8_r ", dq8_r, " d ", dq8_d, " t ", ndquat.translation(dq8_r, dq8_d)) 
        """
        
        assert((ndquat.translation(dq8_r, dq8_d) == np.array([4.0, 6.0, 8.0])).all())


    def test_unit(self):
        q_r_unit = np.array([1.0, 0.0, 0.0, 0.0])
        q_d_zero = np.array([0.0, 0.0, 0.0, 0.0])
        q_mul_r, q_mul_d = ndquat.mul(q_r_unit, q_d_zero, q_r_unit, q_d_zero)
        
        """
        print("q_mul_r ", q_mul_r, " d ", q_mul_d)
        """
                            
        assert((q_r_unit == q_mul_r).all() and (q_d_zero == q_mul_d).all())

    def test_add(self):
        
        dq1_r, dq1_d = ndquat.tvec2dq(np.array([4.0, 6.0, 8.0]))
        dq2_r, dq2_d = ndquat.tvec2dq(np.array([1.0, 2.0, 3.0]))
        
        """
        print("dq1_r ", dq1_r, " d ", dq1_d, " t ", ndquat.translation(dq1_r, dq1_d)) 
        print("dq8_r ", dq2_r, " d ", dq2_d, " t ", ndquat.translation(dq2_r, dq2_d)) 
        """
        
        dq_sum_r, dq_sum_d = ndquat.add(dq1_r, dq1_d, dq2_r, dq2_d)
        
        """
        print("dq_sum_r ", dq_sum_r, " d ", dq_sum_d, " t ", ndquat.translation(dq_sum_r, dq_sum_d)) 
        """
        
        assert((dq_sum_d == np.array([0., 2.5, 4., 5.5])).all())

    def test_mult(self):
        # quaternion multiplication. Compare with homogeneous transformation matrices
        theta1 = np.pi / 180 * 20  # 20 deg
        T_pure_rot = np.array([[1., 0., 0., 0.],
                               [0., np.cos(theta1), -np.sin(theta1), 0.],
                               [0., np.sin(theta1), np.cos(theta1), 0.],
                               [0., 0., 0., 1.]])
        
        dq_pure_rot_r, dq_pure_rot_d = ndquat.hmat2dq(T_pure_rot)
        
        """
        print("dq_pure_rot_r ", dq_pure_rot_r, " d ", dq_pure_rot_d, " t ", ndquat.translation(dq_pure_rot_r, dq_pure_rot_d))
        """
        
        T_pure_trans = np.array([[1., 0., 0., 1.],
                                 [0., 1., 0., 2.],
                                 [0., 0., 1., 3.],
                                 [0., 0., 0., 1.]])
        
        dq_pure_trans_r, dq_pure_trans_d = ndquat.hmat2dq(T_pure_trans)

        """
        print("dq_pure_trans_r ", dq_pure_trans_r, " d ", dq_pure_trans_d, " t ", ndquat.translation(dq_pure_trans_r, dq_pure_trans_d))
        """
        
        T_double_rot = np.dot(T_pure_rot, T_pure_rot)
        
        dq_double_rot_r, dq_double_rot_d = ndquat.mul(dq_pure_rot_r, dq_pure_rot_d, dq_pure_rot_r, dq_pure_rot_d)
        
        """
        print("dq_double_rot_r ", dq_double_rot_r, " d ", dq_double_rot_d, " t ", ndquat.translation(dq_double_rot_r, dq_double_rot_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(T_double_rot, ndquat.dq2hmat(dq_double_rot_r, dq_double_rot_d))
        except AssertionError as e:
            self.fail(e)

        T_double_trans = np.dot(T_pure_trans, T_pure_trans)
        
        dq_double_trans_r, dq_double_trans_d = ndquat.mul(dq_pure_trans_r, dq_pure_trans_d, dq_pure_trans_r, dq_pure_trans_d)
        
        """
        print("dq_double_trans_r ", dq_double_trans_r, " d ", dq_double_trans_d, " t ", ndquat.translation(dq_double_trans_r, dq_double_trans_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(T_double_trans, ndquat.dq2hmat(dq_double_trans_r, dq_double_trans_d))
        except AssertionError as e:
            self.fail(e)

        # composed: trans and rot
        T_composed = np.dot(T_pure_rot, T_pure_trans)
    
        dq_composed_r, dq_composed_d = ndquat.mul(dq_pure_rot_r, dq_pure_rot_d, dq_pure_trans_r, dq_pure_trans_d)
        
        """
        print("dq_composed_r ", dq_composed_r, " d ", dq_composed_d, " t ", ndquat.translation(dq_composed_r, dq_composed_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(T_composed, ndquat.dq2hmat(dq_composed_r, dq_composed_d))
        except AssertionError as e:
            self.fail(e)

    def test_div(self):
        
        dq_div1_r, dq_div1_d = ndquat.div(self.random_dq_r, self.random_dq_d, self.random_dq_r, self.random_dq_d)
        
        """
        print("dq_div1_r ", dq_div1_r, " d ", dq_div1_d, " t ", ndquat.translation(dq_div1_r, dq_div1_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(dq_div1_r, self.identity_dq_r)
            np.testing.assert_array_almost_equal(dq_div1_d, self.identity_dq_d)
        except AssertionError as e:
            self.fail(e)
        
        dq_div2_r, dq_div2_d = ndquat.div(self.random_dq_r, self.random_dq_d, self.identity_dq_r, self.identity_dq_d)
        
        """
        print("dq_div2_r ", dq_div2_r, " d ", dq_div2_d, " t ", ndquat.translation(dq_div2_r, dq_div2_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(dq_div2_r, self.random_dq_r)
            np.testing.assert_array_almost_equal(dq_div2_d, self.random_dq_d)
        except AssertionError as e:
            self.fail(e)

    def test_inverse(self):
        # use known matrix inversion
        T_1_2 = np.array([[0, 1, 0, 2], [-1, 0, 0, 4], [0, 0, 1, 6], [0, 0, 0, 1]])
        T_2_1 = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, -6], [0, 0, 0, 1]])
        
        dq_1_2_r, dq_1_2_d = ndquat.hmat2dq(T_1_2)
        dq_2_1_r, dq_2_1_d = ndquat.hmat2dq(T_2_1)
        
        dq_1_2_r, dq_1_2_d = ndquat.normalize(dq_1_2_r, dq_1_2_d)
        
        """
        print("dq_1_2_r ", dq_1_2_r, " d ", dq_1_2_d, " t ", ndquat.translation(dq_1_2_r, dq_1_2_d))
        print("dq_2_1_r ", dq_1_2_r, " d ", dq_2_1_d, " t ", ndquat.translation(dq_2_1_r, dq_2_1_d))
        """
        
        dq_1_2_inverse_r, dq_1_2_inverse_d = ndquat.inv(dq_1_2_r, dq_1_2_d)
        
        """
        print("dq_1_2_inverse_r ", dq_1_2_r, " d ", dq_1_2_inverse_d, " t ", ndquat.translation(dq_1_2_inverse_r, dq_1_2_inverse_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(ndquat.dq2hmat(dq_2_1_r, dq_2_1_d), ndquat.dq2hmat(dq_1_2_inverse_r, dq_1_2_inverse_d))
        except AssertionError as e:
            self.fail(e)

    def test_equal(self):
        
        """
        print("identity_dq_r ", self.identity_dq_r, " d ", self.identity_dq_d, " t ", ndquat.translation(self.identity_dq_r, self.identity_dq_d))
        """
        
        id_r, id_d = ndquat.identity()
        
        assert((self.identity_dq_r == id_r).all() and (self.identity_dq_d == id_d).all())
        assert((self.identity_dq_r == np.array([1.0, 0.0, 0.0, 0.0])).all() and (self.identity_dq_d == id_d).all())


        theta1 = np.pi / 180 * 20  # 20 deg
        T_pure_rot = np.array([[1., 0., 0., 0.],
                               [0., np.cos(theta1), -np.sin(theta1), 0.],
                               [0., np.sin(theta1), np.cos(theta1), 0.],
                               [0., 0., 0., 1.]])
        dq_pure_rot_r, dq_pure_rot_d = ndquat.hmat2dq(T_pure_rot)
        
        """
        print("dq_pure_rot_r ", dq_pure_rot_r, " d ", dq_pure_rot_d, " t ", ndquat.translation(dq_pure_rot_r, dq_pure_rot_d))
        """

        # manually flip sign on terms
        dq_pure_rot_r = - dq_pure_rot_r
        dq_pure_rot_d = - dq_pure_rot_d
        try:
            np.testing.assert_array_almost_equal(ndquat.dq2hmat(dq_pure_rot_r, dq_pure_rot_d), T_pure_rot)
        except AssertionError as e:
            self.fail(e)
        dq_pure_rot_d = - dq_pure_rot_d
        try:
            np.testing.assert_array_almost_equal(ndquat.dq2hmat(dq_pure_rot_r, dq_pure_rot_d), T_pure_rot)
        except AssertionError as e:
            self.fail(e)
        dq_pure_rot_r = - dq_pure_rot_r
        try:
            np.testing.assert_array_almost_equal(ndquat.dq2hmat(dq_pure_rot_r, dq_pure_rot_d), T_pure_rot)
        except AssertionError as e:
            self.fail(e)

    def test_quaternion_conjugate(self):
        
        dq_conj_r, dq_conj_d = ndquat.conj(self.normalized_dq_r, self.normalized_dq_d)
        
        """
        print("dq_conj_r ", dq_conj_r, " d ", dq_conj_d, " t ", ndquat.translation(dq_conj_r, dq_conj_d))
        print("self.normalized_dq_r ", self.normalized_dq_r, " d ", self.normalized_dq_d, " t ", ndquat.translation(self.normalized_dq_r, self.normalized_dq_d))
        """
        
        dq2_conj_r, dq2_conj_d = ndquat.mul(self.normalized_dq_r, self.normalized_dq_d, dq_conj_r, dq_conj_d)
        
        """
        print("dq2_conj_r ", dq2_conj_r, " d ", dq2_conj_d, " t ", ndquat.translation(dq2_conj_r, dq2_conj_d))
        """
        
        # a normalized quaternion multiplied with its quaternion conjugate should yield unit dual quaternion
        
        try:
            np.testing.assert_array_almost_equal(dq2_conj_r, self.identity_dq_r)
            np.testing.assert_array_almost_equal(dq2_conj_d, self.identity_dq_d)
        except AssertionError as e:
            self.fail(e)

        # test that the conjugate corresponds to the inverse of it's matrix representation
        matr = ndquat.dq2hmat(self.normalized_dq_r, self.normalized_dq_d)
        
        """
        print("matr ", matr)
        """

        inv = np.linalg.inv(matr)
        
        """
        print("inv ", inv)
        """
        
        dq_inv_r, dq_inv_d = ndquat.hmat2dq(inv)

        """
        print("dq_inv_r ", dq_inv_r, " d ", dq_inv_d, " t ", ndquat.translation(dq_inv_r, dq_inv_d))
        """

        # (dq1 @ dq2)* ?= dq2* @ dq1*
        res1_r, res1_d = ndquat.mul(self.random_dq_r, self.random_dq_d, self.other_random_dq_r, self.other_random_dq_d)
        res1_r, res1_d = ndquat.conj(res1_r, res1_d)
        
        """
        print("res1_r ", res1_r, " d ", res1_d, " t ", ndquat.translation(res1_r, res1_d))
        """
        
        r_conj_r, r_conj_d = ndquat.conj(self.random_dq_r, self.random_dq_d)
        or_conj_r, or_conj_d  = ndquat.conj(self.other_random_dq_r, self.other_random_dq_d)
        
        """
        print("r_conj_r ", r_conj_r, " d ", r_conj_d, " t ", ndquat.translation(r_conj_r, r_conj_d))
        print("or_conj_r ", or_conj_r, " d ", or_conj_d, " t ", ndquat.translation(or_conj_r, or_conj_d))
        """
        
        res2_r, res2_d = ndquat.mul(or_conj_r, or_conj_d, r_conj_r, r_conj_d)
        
        """
        print("res2_r ", res2_r, " d ", res2_d, " t ", ndquat.translation(res2_r, res2_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(res1_r, res2_r)
            np.testing.assert_array_almost_equal(res1_d, res2_d)
        except AssertionError as e:
            self.fail(e)

    def test_homogeneous_conversion(self):
        # 1. starting from a homogeneous matrix
        theta1 = np.pi/2  # 90 deg
        trans = [10., 5., 0.]
        H1 = np.array([[1., 0., 0., trans[0]],
                     [0., np.cos(theta1), -np.sin(theta1), trans[1]],
                     [0., np.sin(theta1), np.cos(theta1), trans[2]],
                     [0., 0., 0., 1.]])
        # check that if we convert to DQ and back to homogeneous matrix, we get the same result
        dq_conv1_r, dq_conv1_d = ndquat.hmat2dq(H1)
        double_conv1 = ndquat.dq2hmat(dq_conv1_r, dq_conv1_d)
        
        """
        print("double_conv1 ", double_conv1)
        """
        
        try:
            np.testing.assert_array_almost_equal(H1, double_conv1)
        except AssertionError as e:
            self.fail(e)
            
        # check that dual quaternions are also equal
        dq1_r, dq1_d = ndquat.hmat2dq(H1)
        dq_double1_r, dq_double1_d = ndquat.hmat2dq(double_conv1)
        
        """
        print("dq1_r ", dq1_r, " d ", dq1_d, " t ", ndquat.translation(dq1_r, dq1_d))
        print("dq_double1_r ", dq_double1_r, " d ", dq_double1_d, " t ", ndquat.translation(dq_double1_r, dq_double1_d))
        """
        
        assert((dq1_r == dq_double1_r).all() and (dq1_d == dq_double1_d).all())

        # 2. starting from a DQ
        dq_trans_r, dq_trans_d = ndquat.tvec2dq(np.array([10.0, 5.0, 0.0]))
        
        """
        print("dq_trans_r ", dq_trans_r, " d ", dq_trans_d, " t ", ndquat.translation(dq_trans_r, dq_trans_d))
        """
        
        dq_rot_r = np.array([np.cos(theta1 / 2), np.sin(theta1 / 2), 0.0, 0.0])
        dq_rot_d = np.array([0.0, 0.0, 0.0, 0.0])

        """
        print("dq_rot_r ", dq_rot_r, " d ", dq_rot_d, " t ", ndquat.translation(dq_rot_r, dq_rot_d))
        """
        
        dq2_r, dq2_d = ndquat.mul(dq_trans_r, dq_trans_d, dq_rot_r, dq_rot_d)

        """
        print("dq2_r ", dq2_r, " d ", dq2_d, " t ", ndquat.translation(dq2_r, dq2_d))
        """
        
        # check that this is the same as the previous DQ
        try:
            np.testing.assert_array_almost_equal(dq2_r, dq1_r)
            np.testing.assert_array_almost_equal(dq2_d, dq1_d)
        except AssertionError as e:
            self.fail(e)

        
        # check that if we convert to homogeneous matrix and back, we get the same result
        mat2 = ndquat.dq2hmat(dq2_r, dq2_d)
        
        """
        print("mat2 ", mat2)
        """
        
        double_conv2_r, double_conv2_d = ndquat.hmat2dq(mat2)
        
        """
        print("double_conv2_r ", double_conv2_r, " d ", double_conv2_d, " t ", ndquat.translation(double_conv2_r, double_conv2_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(dq2_r, double_conv2_r)
            np.testing.assert_array_almost_equal(dq2_d, double_conv2_d)
        except AssertionError as e:
            self.fail(e)

    def test_dual_number_conjugate(self):
        # dual number conjugate doesn't behave as you would expect given its special definition
        # (dq1 @ dq2)* ?= dq1* @ dq2*  This is a different order than the other conjugates!
        
        mul1_r, mul1_d = ndquat.mul(self.random_dq_r, self.random_dq_d, self.other_random_dq_r, self.other_random_dq_d)
        
        res1_r, res1_d = ndquat.dconj(mul1_r, mul1_d)
        
        """
        print("res1_r ", res1_r, " d ", res1_d, " t ", ndquat.translation(res1_r, res1_d))
        """
        
        dq_conj_r, dq_conj_d = ndquat.dconj(self.random_dq_r, self.random_dq_d)
        dq2_conj_r, dq2_conj_d = ndquat.dconj(self.other_random_dq_r, self.other_random_dq_d)
        res2_r, res2_d = ndquat.mul(dq_conj_r, dq_conj_d, dq2_conj_r, dq2_conj_d)
        
        """
        print("res2_r ", res2_r, " d ", res2_d, " t ", ndquat.translation(res2_r, res2_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(res1_r, res2_r)
            np.testing.assert_array_almost_equal(res1_d, res2_d)
        except AssertionError as e:
            self.fail(e)

    def test_combined_conjugate(self):
        dq_conj_r, dq_conj_d = ndquat.cconj(self.normalized_dq_r, self.normalized_dq_d)
        
        """
        print("dq_conj_r ", dq_conj_r, " d ", dq_conj_d, " t ", ndquat.translation(dq_conj_r, dq_conj_d))
        """
        
        dq_r, dq_d = ndquat.mul(self.normalized_dq_r, self.normalized_dq_d, dq_conj_r, dq_conj_d)
        
        """
        print("dq_r ", dq_r, " d ", dq_d, " t ", ndquat.translation(dq_r, dq_d))
        """
    
        # a normalized quaternion multiplied with its combined conjugate should yield unit rotation
        try:
            np.testing.assert_array_almost_equal(dq_r, np.array([1.0, 0.0, 0.0, 0.0]))
        except AssertionError as e:
            self.fail(e)

        # (dq1 @ dq2)* ?= dq2* @ dq1*
        mul1_r, mul1_d = ndquat.mul(self.random_dq_r, self.random_dq_d, self.other_random_dq_r, self.other_random_dq_d)
        res1_r, res1_d = ndquat.cconj(mul1_r, mul1_d)
        
        """
        print("res1_r ", dq_r, " d ", res1_d, " t ", ndquat.translation(res1_r, res1_d))
        """
        
        cc1_r, cc1_d = ndquat.cconj(self.other_random_dq_r, self.other_random_dq_d)
        cc2_r, cc2_d = ndquat.cconj(self.random_dq_r, self.random_dq_d)
        res2_r, res2_d = ndquat.mul(cc1_r, cc1_d, cc2_r, cc2_d)
        
        """
        print("res2_r ", res2_r, " d ", res2_d, " t ", ndquat.translation(res2_r, res2_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(res1_r, res2_r)
            np.testing.assert_array_almost_equal(res1_d, res2_d)
        except AssertionError as e:
            self.fail(e)

    def test_normalize(self):
        """
        print("norm ", ndquat.is_normalized(self.identity_dq_r, self.identity_dq_d))
        """
        
        assert(ndquat.is_normalized(self.identity_dq_r, self.identity_dq_d))
        
        dq1_norm_r, dq1_norm_d = ndquat.normalize(self.identity_dq_r, self.identity_dq_d)
        assert((dq1_norm_r == self.identity_dq_r).all() and (dq1_norm_d == self.identity_dq_d).all())
        
        unnormalized_dq_r, unnormalized_dq_d = ndquat.qtvec2dq(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0]))
        
        """
        print("unnormalized_dq_r ", unnormalized_dq_r, " d ", unnormalized_dq_d, " t ", ndquat.translation(unnormalized_dq_r, unnormalized_dq_d))
        """
        
        normalized_dq_r, normalized_dq_d = ndquat.normalize(unnormalized_dq_r, unnormalized_dq_d)
        
        """
        print("normalized_dq_r ", normalized_dq_r, " d ", normalized_dq_d, " t ", ndquat.translation(normalized_dq_r, normalized_dq_d))
        """
        
        assert(ndquat.is_normalized(self.normalized_dq_r, self.normalized_dq_d))

    def test_transform(self):
        # transform a point from one frame (f2) to another (f1)
        point_f2 = [1.0, 1.0, 0.0]
        
        t_point = ndquat.transform_point(self.identity_dq_r, self.identity_dq_d, point_f2)
        
        """
        print("t_point ", t_point)
        """
        
        assert((t_point == point_f2).all())

        # test that quaternion transform and matrix transform yield the same result
        T_f1_f2 = np.array([[1, 0, 0, 2],
                            [0, 0.54028748, -0.8414805, 3],
                            [0, 0.8414805, 0.54028748, 1],
                            [0, 0, 0, 1]])
        
        dq_f1_f2_r, dq_f1_f2_d = ndquat.hmat2dq(T_f1_f2)
        
        """
        print("dq_f1_f2_r ", dq_f1_f2_r, " d ", dq_f1_f2_d, " t ", ndquat.translation(dq_f1_f2_r, dq_f1_f2_d))
        """
        
        # point is in f2, transformation will express it in f1
        point_f1_matrix = np.dot(T_f1_f2, np.expand_dims(np.array(point_f2 + [1]), 1))
        
        point_f1_dq = ndquat.transform_point(dq_f1_f2_r, dq_f1_f2_d, point_f2)
        
        """
        print("point_f1_dq ", point_f1_dq)
        """
        
        try:
            np.testing.assert_array_almost_equal(point_f1_matrix[:3].T.flatten(), point_f1_dq.flatten(), decimal=3)
        except AssertionError as e:
            self.fail(e)

    def test_screw(self):
        # test unit
        l, m, theta, d = ndquat.screw(self.identity_dq_r, self.identity_dq_d)
        
        """
        print("1 l ", l, " m ", m, " theta ", theta, " d ", d)
        """
        
        assert(d == 0)
        assert(theta == 0)

        # test pure translation
        trans = [10.0, 5.0, 0.0]
        dq_trans_r, dq_trans_d = ndquat.tvec2dq(trans)
        
        """
        print("dq_trans_r ", dq_trans_r, " d ", dq_trans_d, " t ", ndquat.translation(dq_trans_r, dq_trans_d))
        """
        
        l, m, theta, d = ndquat.screw(dq_trans_r, dq_trans_d)

        """
        print("2 l ", l, " m ", m, " theta ", theta, " d ", d)
        """

        self.assertAlmostEqual(d, np.linalg.norm(trans), 2)
        self.assertAlmostEqual(theta, 0)

        # test pure rotation
        theta1 = np.pi/2
        dq_rot_r = np.array([np.cos(theta1 / 2), np.sin(theta1 / 2), 0, 0])
        dq_rot_d = np.array([0.0, 0.0, 0.0, 0.0])
        
        """
        print("dq_rot_r ", dq_rot_r, " d ", dq_rot_d, " t ", ndquat.translation(dq_rot_r, dq_rot_d))
        """
        
        l, m, theta, d = ndquat.screw(dq_rot_r, dq_rot_d)
        
        """
        print("3 l ", l, " m ", m, " theta ", theta, " d ", d)
        """
        
        self.assertAlmostEqual(theta, theta1)

        # test simple rotation and translation: rotate in the plane of a coordinate system with the screw axis offset
        # along +y. Rotate around z axis so that the coordinate system stays in the plane. Translate along z-axis
        theta2 = np.pi/2
        
        dq_rot2_r = np.array([np.cos(theta2 / 2), 0, 0, np.sin(theta2 / 2)])
        dq_rot2_d = np.array([0.0, 0.0, 0.0, 0.0])
        
        """
        print("dq_rot2_r ", dq_rot2_r, " d ", dq_rot2_d, " t ", ndquat.translation(dq_rot2_r, dq_rot2_d))
        """
        
        dist_axis = 5.
        displacement_z = 3.
        
        dq_trans_r, dq_trans_d = ndquat.tvec2dq(np.array([dist_axis*np.sin(theta2), dist_axis*(1.-np.cos(theta2)), displacement_z]))

        """
        print("dq_trans_r ", dq_trans_r, " d ", dq_trans_d, " t ", ndquat.translation(dq_trans_r, dq_trans_d))
        """
        dq_comb_r, dq_comb_d = ndquat.mul(dq_trans_r, dq_trans_d, dq_rot2_r, dq_rot2_d)
        
        """
        print("dq_comb_r ", dq_comb_r, " d ", dq_comb_d, " t ", ndquat.translation(dq_comb_r, dq_comb_d))
        """
        
        l, m, theta, d = ndquat.screw(dq_comb_r, dq_comb_d)
        
        """
        print("4 l ", l, " m ", m, " theta ", theta, " d ", d)
        """
        
        try:
            # the direction of the axis should align with the z axis of the origin
            np.testing.assert_array_almost_equal(l, np.array([0, 0, 1]), decimal=3)
            # m = p x l with p any point on the line
            np.testing.assert_array_almost_equal(np.cross(np.array([[0, dist_axis, 0]]), l).flatten(), m)
        except AssertionError as e:
            self.fail(e)
        self.assertAlmostEqual(d, displacement_z)  # only displacement along z should exist here
        self.assertAlmostEqual(theta, theta2)  # the angles should be the same

    def test_from_screw(self):
        # construct an axis along the positive z-axis
        l = np.array([0, 0, 1])
        # pick a point on the axis that defines it's location
        p = np.array([-1, 0, 0])
        # moment vector
        m = np.cross(p, l)
        theta = np.pi/2
        d = 3.
        # this corresponds to a rotation around the axis parallel with the origin's z-axis through the point p
        # the resulting transformation should move the origin to a DQ with elements:
            
        desired_dq_rot_r, desired_dq_rot_d = ndquat.qtvec2dq(np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)]), np.array([0.0, 0.0, 0.0]))

        """
        print("desired_dq_rot_r ", desired_dq_rot_r, " d ", desired_dq_rot_d, " t ", ndquat.translation(desired_dq_rot_r, desired_dq_rot_d))
        """
        
        desired_dq_trans_r, desired_dq_trans_d = ndquat.tvec2dq(np.array([-1.0, 1.0, d]))

        """
        print("desired_dq_trans_r ", desired_dq_trans_r, " d ", desired_dq_trans_d, " t ", ndquat.translation(desired_dq_trans_r, desired_dq_trans_d))
        """
        
        desired_dq_r, desired_dq_d = ndquat.mul(desired_dq_trans_r, desired_dq_trans_d, desired_dq_rot_r, desired_dq_rot_d)
        
        """
        print("desired_dq_r ", desired_dq_r, " d ", desired_dq_d, " t ", ndquat.translation(desired_dq_r, desired_dq_d))
        """
        
        dq_r, dq_d = ndquat.from_screw(l, m, theta, d)
        
        """
        print("dq_r ", dq_r, " d ", dq_d, " t ", ndquat.translation(dq_r, dq_d))
        """
        
        assert((dq_r == desired_dq_r).all() and (dq_d == desired_dq_d).all())

    def test_from_screw_and_back(self):
        # start with a random valid dual quaternion
        
        dq_r, dq_d = ndquat.qtvec2dq(np.array([0.5, 0.3, 0.1, 0.4]), np.array([2.0, 5.0, -2.0]))
        
        """
        print("dq_r ", dq_r, " d ", dq_d, " t ", ndquat.translation(dq_r, dq_d))
        """
        
        lr, mr, thetar, dr = ndquat.screw(dq_r, dq_d)
        
        """
        print("lr ", lr, " mr ", mr, " thetar ", thetar, " dr ", dr)
        """
        
        dq_reconstructed_r, dq_reconstructed_d = ndquat.from_screw(lr, mr, thetar, dr)
        
        """
        print("dq_reconstructed_r ", dq_reconstructed_r, " d ", dq_reconstructed_d, " t ", ndquat.translation(dq_reconstructed_r, dq_reconstructed_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(dq_r, dq_reconstructed_r)
            np.testing.assert_array_almost_equal(dq_d, dq_reconstructed_d)
        except AssertionError as e:
            self.fail(e)
            
        # start with some screw parameters
        l1 = np.array([0.4, 0.2, 0.5])
        l1 /= np.linalg.norm(l1)  # make sure l1 is normalized
        # pick some point away from the origin
        p1 = np.array([2.3, 0.9, 1.1])
        m1 = np.cross(p1, l1)
        d1 = 4.32
        theta1 = 1.94
        
        dq1_r, dq1_d = ndquat.from_screw(l1, m1, theta1, d1)
        
        """
        print("dq1_r ", dq1_r, " d ", dq1_d, " t ", ndquat.translation(dq1_r, dq1_d))
        """
        
        l2, m2, theta2, d2 = ndquat.screw(dq1_r, dq1_d)
        
        """
        print("l2 ", l2, " m2 ", m2, " theta2 ", theta2, " d2 ", d2)
        """
        
        try:
            np.testing.assert_array_almost_equal(l1, l2, decimal=3)
            np.testing.assert_array_almost_equal(l1, l2, decimal=3)
        except AssertionError as e:
            self.fail(e)
            
        self.assertAlmostEqual(theta1, theta2)
        self.assertAlmostEqual(d1, d2)

    def test_sclerp_position(self):
        """test Screw Linear Interpolation for diff position, same orientation"""
        
        dq1_r, dq1_d = ndquat.tvec2dq(np.array([2.0, 2.0, 2.0]))
        
        """
        print("dq1_r ", dq1_r, " d ", dq1_d, " t ", ndquat.translation(dq1_r, dq1_d))
        """
        
        dq2_r, dq2_d = ndquat.tvec2dq(np.array([3.0, 4.0, -2.0]))
        
        """
        print("dq2_r ", dq2_r, " d ", dq2_d, " t ", ndquat.translation(dq2_r, dq2_d))
        """
        
        interpolated1_r, interpolated1_d = ndquat.sclerp(dq1_r, dq1_d, dq2_r, dq2_d, 0.5)

        """
        print("interpolated1_r ", interpolated1_r, " d ", interpolated1_d, " t ", ndquat.translation(interpolated1_r, interpolated1_d))
        """
        
        expected1_r, expected1_d = ndquat.tvec2dq(np.array([2.5, 3.0, 0.0]))
        
        """
        print("expected1_r ", expected1_r, " d ", expected1_d, " t ", ndquat.translation(expected1_r, interpolated1_d))
        """
        
        assert((interpolated1_r == expected1_r).all() and (interpolated1_d == expected1_d).all())


        interpolated2_r, interpolated2_d = ndquat.sclerp(dq1_r, dq1_d, dq2_r, dq2_d, 0.1)
        
        """
        print("interpolated2_r ", interpolated2_r, " d ", interpolated2_d, " t ", ndquat.translation(interpolated2_r, interpolated2_d))
        """
        
        expected2_r, expected2_d = ndquat.tvec2dq(np.array([2.1, 2.2, 1.6]))

        """
        print("expected2_r ", expected2_r, " d ", expected2_d, " t ", ndquat.translation(expected2_r, expected2_d))
        """
        
        assert((interpolated2_r == expected2_r).all() and (interpolated2_d == expected2_d).all())

    def test_sclerp_orientation(self):
        """test Screw Linear Interpolation for diff orientation, same position"""
        
        id_r = np.array([1., 0., 0., 0.])
        id_d = np.array([0., 0., 0., 0.])
        
        """
        print("id_r ", id_r, " d ", id_d, " t ", ndquat.translation(id_r, id_d))
        """
        
        T_id = ndquat.dq2hmat(id_r, id_d)
        T_id[0:2, 0:2] = np.array([[0, -1], [1, 0]])  # rotate 90 around z
        
        """
        print("T_id ", T_id)
        """
        
        dq2_r, dq2_d = ndquat.hmat2dq(T_id)
        
        """
        print("dq2_r ", dq2_r, " d ", dq2_d, " t ", ndquat.translation(dq2_r, dq2_d))
        """

        interpolated1_r, interpolated1_d = ndquat.sclerp(self.identity_dq_r, self.identity_dq_d, dq2_r, dq2_d, 0.5)
        
        """
        print("interpolated1_r ", interpolated1_r, " d ", interpolated1_d, " t ", ndquat.translation(interpolated1_r, interpolated1_d))
        """
        
        T_exp = ndquat.dq2hmat(id_r, id_d)
        sq22 = np.sqrt(2)/2
        T_exp[0:2, 0:2] = np.array([[sq22, -sq22], [sq22, sq22]])  # rotate 45 around z
        
        """
        print("T_exp ", T_exp)
        """
        
        expected1_r, expected1_d = ndquat.hmat2dq(T_exp)
        
        """
        print("expected1_r ", expected1_r, " d ", expected1_d, " t ", ndquat.translation(expected1_r, expected1_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(interpolated1_r, expected1_r, decimal=3)
            np.testing.assert_array_almost_equal(interpolated1_d, expected1_d, decimal=3)
        except AssertionError as e:
            self.fail(e)
        
        interpolated2_r, interpolated2_d = ndquat.sclerp(self.identity_dq_r, self.identity_dq_d, dq2_r, dq2_d, 0.0)
        
        """
        print("interpolated2_r ", interpolated2_r, " d ", interpolated2_d, " t ", ndquat.translation(interpolated2_r, interpolated2_d))
        """
        
        interpolated3_r, interpolated3_d = ndquat.sclerp(self.identity_dq_r, self.identity_dq_d, dq2_r, dq2_d, 1.0)
        
        """
        print("interpolated3_r ", interpolated3_r, " d ", interpolated3_d, " t ", ndquat.translation(interpolated3_r, interpolated3_d))
        """

        assert((interpolated2_r == self.identity_dq_r).all() and (interpolated2_d == self.identity_dq_d).all())
        assert((interpolated3_r == dq2_r).all() and (interpolated3_d == dq2_d).all())

    def test_sclerp_screw(self):
        """Interpolating with ScLERP should yield same result as interpolating with screw parameters
        ScLERP is a screw motion interpolation with constant rotation and translation speeds. We can
        simply interpolate screw parameters theta and d and we should get the same result.
        """
        taus = [0., 0.23, 0.6, 1.0]
        
        l, m, theta, d= ndquat.screw(self.normalized_dq_r, self.normalized_dq_d)
        
        """
        print("l ", l, " m ", m, " theta ", theta, " d ", d)
        """
        
        for tau in taus:
            # interpolate using sclerp
            interpolated_dq_r, interpolated_dq_d = ndquat.sclerp(self.identity_dq_r, self.identity_dq_d, self.normalized_dq_r, self.normalized_dq_d, tau)

            """
            print("interpolated_dq_r ", interpolated_dq_r, " d ", interpolated_dq_d, " t ", ndquat.translation(interpolated_dq_r, interpolated_dq_d))
            """
            
            # interpolate using screw: l and m stay the same, theta and d vary with tau
            interpolated_dq_screw_r, interpolated_dq_screw_d = ndquat.from_screw(l, m, tau*theta, tau*d)
            
            """
            print("interpolated_dq_screw_r ", interpolated_dq_screw_r, " d ", interpolated_dq_screw_d, " t ", ndquat.translation(interpolated_dq_screw_r, interpolated_dq_screw_d))
            """
            
            try:
                np.testing.assert_array_almost_equal(interpolated_dq_r, interpolated_dq_screw_r, decimal=3)
                np.testing.assert_array_almost_equal(interpolated_dq_d, interpolated_dq_screw_d, decimal=3)
            except AssertionError as e:
                self.fail(e)

    def test_pow(self):
        
        expected_result_r, expected_result_d = ndquat.mul(self.normalized_dq_r, self.normalized_dq_d, self.normalized_dq_r, self.normalized_dq_d)
        
        """
        print("expected_result_r ", expected_result_r, " d ", expected_result_d, " t ", ndquat.translation(expected_result_r, expected_result_d))
        """
        
        received_result_r, received_result_d = ndquat.pow(self.normalized_dq_r, self.normalized_dq_d, 2.0)
        
        """
        print("received_result_r ", received_result_r, " d ", received_result_d, " t ", ndquat.translation(received_result_r, received_result_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(received_result_r, expected_result_r, decimal=3)
            np.testing.assert_array_almost_equal(received_result_d, expected_result_d, decimal=3)
        except AssertionError as e:
            self.fail(e)

        expected_result_r, expected_result_d = ndquat.mul(self.random_dq_r, self.random_dq_d, self.random_dq_r, self.random_dq_d)
         
        """
        print("expected_result_r ", expected_result_r, " d ", expected_result_d, " t ", ndquat.translation(expected_result_r, expected_result_d))
        """
        
        received_result_r, received_result_d = ndquat.pow(self.random_dq_r, self.random_dq_d, 2.0)
        
        """
        print("received_result_r ", received_result_r, " d ", received_result_d, " t ", ndquat.translation(received_result_r, received_result_d))
        """
        
        try:
            np.testing.assert_array_almost_equal(received_result_r, expected_result_r, decimal=3)
            np.testing.assert_array_almost_equal(received_result_d, expected_result_d, decimal=3)
        except AssertionError as e:
            self.fail(e)

tests = TestDualQuaternion()
tests.setUp()
tests.test_creation()
tests.test_unit()
tests.test_add()
tests.test_mult()
tests.test_div()
tests.test_inverse()
tests.test_equal()
tests.test_quaternion_conjugate()
tests.test_homogeneous_conversion()
tests.test_dual_number_conjugate()
tests.test_combined_conjugate()
tests.test_normalize()
tests.test_transform()
tests.test_screw()
tests.test_from_screw()
tests.test_from_screw_and_back()
tests.test_sclerp_position()
tests.test_sclerp_orientation()
tests.test_sclerp_screw()

tests.test_pow()

T_1_2 = np.array([[0, 1, 0, 2], [-1, 0, 0, 4], [0, 0, 1, 6], [0, 0, 0, 1]])
T_2_1 = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, -6], [0, 0, 0, 1]])

dq_1_2_r, dq_1_2_d = ndquat.hmat2dq(T_1_2)
dq_2_1_r, dq_2_1_d = ndquat.hmat2dq(T_2_1)


