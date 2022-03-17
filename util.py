# Test file to simulate trajectory given u and d

import numpy as np
import time
import math
import scipy.io


class Simulate:

    def __init__(self, N, u, d, initial_state):
        self.v_e = 0.75
        self.v_p = 0.75
        self.N = N
        self.dt = 1 / self.N
        self.initial_state = initial_state

        # control variable
        self.u_static = u
        self.d_static = d

    # dynamic function
    def f(self, x_, u_, d_): return [-0.75 + 0.75 * np.cos(x_[2]) + u_ * x_[1],
                               0.75 * np.sin(x_[2]) - u_ * x_[0],
                               d_ - u_]

    def l(self, x_): return np.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9) - 0.25

    def rollout(self):
        states = np.empty((3, self.N))
        # ini_state = np.array([0.5, -0.5, 1.586])
        states[:, 0] = self.initial_state

        # dynamic constraint
        for i in range(self.N - 1):
            states[:, i + 1] = states[:, i] + \
                               np.array(self.f(states[:, i], self.u_static[i], self.d_static[i])) * self.dt
        return states

    def loss(self, states):
        return [self.l(states[:, i]) for i in range(self.N)]

# print([l(states[:, i]) for i in range(N)])

