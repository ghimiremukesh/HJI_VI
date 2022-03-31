
import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate


class BRT:  # from *** paper ***
    def __init__(self, N, T, rho):
        """
        :param N: number of time steps = N - 1
        :param T: final time
        :param rho: shape parameter for differentiable approx. of min(Array)
        """
        self.v_e = 0.75
        self.v_p = 0.75
        self.N = N
        self.dt = T / N
        self.rho = rho

    # dynamic function
    # TODO: use v_e and v_p
    def f(self, x_, u_, d_): return ca.vertcat(
        -0.75 + 0.75 * ca.cos(x_[2]) + u_ * x_[1],
        0.75 * ca.sin(x_[2]) - u_ * x_[0],
        d_ - u_)

    def l(self, x_): return ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9) - 0.25

    def l_batch(self, x_): return ca.sqrt(x_[0, :] ** 2 + x_[1, :] ** 2 + 1e-9) - 0.25

    # gradient l(x_i) w.r.t x_i
    def dldx(self, x_): return ca.vertcat(
        x_[0] / ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9),
        x_[1] / ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9),
        0)

    # gradient f(x_i) w.r.t x_i, I + df/dx * dt
    # TODO: use v_e and v_p
    def dfdx(self, x_, u_):
        row1 = ca.horzcat(1, -u_ * self.dt, 0)
        row2 = ca.horzcat(u_ * self.dt, 1, 0)
        row3 = ca.horzcat(-0.75 * ca.sin(x_[2]) * self.dt, 0.75 * ca.cos(x_[2]) * self.dt, 1)
        return ca.vertcat(row1,
                          row2,
                          row3)

    # differentiable objective for min(Array)
    def F(self, x_):
        return - 1 / self.rho * ca.log(ca.sum2(ca.exp(- self.rho * self.l_batch(x_))))

    def dFdx(self, x_, id):  # x_: state trj, i: time step
            e = ca.exp(-self.rho * self.l_batch(x_))
            if isinstance(x_, np.ndarray):  # this is for post analysis
                softmax = e[id]/ca.sum1(e)
            else:
                softmax = e[id]/ca.sum2(e)  # for casadi formulation
            return softmax * self.dldx(x_[:, id])

    def u_star(self, lam, x_, dt):
        return 3 * ca.sign((lam[0] * x_[1] - lam[1] * x_[0] - lam[2]) * dt)

    def d_star(self, lam): return 3 * ca.sign(lam[2])


# for debug only
class Test:
    def F(self, x_):
        return x_[0]**2 + x_[1]**2

    def dFdx(self, x_):
        return np.array([2*x_[0], 2*x_[1]])


# another test problem for casadi
class TestFEA:
    def F(self, x):  # to minimize strain energy
        K = self.K(x)  # stiffness matrix
        f = np.array([20000, 20000, 0])  # external force
        K_ = K[np.ix_([2, 4, 5], [2, 4, 5])]
        u = ca.mtimes(ca.inv(K_), f)  # deflection
        e = ca.mtimes(ca.mtimes(ca.transpose(u), K_), u)  # strain energy
        return e

    def K(self, x):  # global stiffness matrix
        l = [40, 30, 50, 40]
        theta = [0, np.pi/2, 0.643501109, 0]
        k = [self.k(l[i], x[i], theta[i]) for i in range(4)]  # element-wise stiffness matrix

        # for a specific case:
        # K = ca.MX.zeros((3, 3))
        # K[0, 0] += k[0][2, 2]  # from element 1
        # K += k[1][np.ix_([0, 2, 3], [0, 2, 3])]  # from element 2
        # K[np.ix_([1, 2], [1, 2])] += k[2][np.ix_([2, 3], [2, 3])]  # from element 3
        # K[np.ix_([1, 2], [1, 2])] += k[3][np.ix_([2, 3], [2, 3])]  # from element 4

        # more general cases:
        K = ca.MX.zeros((8, 8))
        K[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])] += k[0]
        K[np.ix_([2, 3, 4, 5], [2, 3, 4, 5])] += k[1]
        K[np.ix_([0, 1, 4, 5], [0, 1, 4, 5])] += k[2]
        K[np.ix_([6, 7, 4, 5], [6, 7, 4, 5])] += k[3]

        return K

    def k(self, l, A, theta):  # element stiffness matrix (spring)
        E = 29.5e6
        # E = 600
        c = np.cos(theta)
        s = np.sin(theta)
        k = np.array([[c**2, c*s, -c**2, -c*s],
                      [c*s, s**2, -c*s, -s**2],
                      [-c**2, -c*s, c**2, c*s],
                      [-c*s, -s**2, c*s, s**2]])
        k *= A*E/l
        return k

    def h(self, x):  # constraint
        l = [40, 30, 50, 40]
        v = 0
        for i in range(4):
            v += x[i] * l[i]
        v0 = np.sum(l)
        return v - v0

# for debug only
class TestDynamics:
    def __init__(self, N):
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([[0.], [1.]])
        self.N = N
        self.dt = 10.

    def F(self, x_):
        return 0.5 * ca.sum2(x_[0, :]**2 + x_[1, :]**2)

    def dFdx(self, x_, id):
        return x_[:, id]

    def f(self, x_, u_):
        return ca.mtimes(self.A, x_) + ca.mtimes(self.B, u_)

    def dfdx(self):
        return ca.transpose(self.A) * self.dt + np.eye(2)
