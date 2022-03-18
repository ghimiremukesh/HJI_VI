
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
            softmax = e[id]/ca.sum1(e)
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
