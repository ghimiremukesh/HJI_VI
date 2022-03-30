# Test file to simulate trajectory given u and d

import numpy as np
import time
import math
import scipy.io
import matplotlib.pyplot as plt

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


class Utility:
    def brt_plot(self, u, d, x0, prob):  # visualize the evade-pursue game
        """
        :param u: evader input
        :param d: pursuer input
        :param x0: initial relative states
        :param prob: problem definition
        :return: na
        """
        x_e = [np.array([0, 0, 0])]
        x_p = [x0]
        for i in range(prob.N-1):
            x_e_new = np.array([x_e[i][0] + prob.v_e * np.cos(x_e[i][2]) * prob.dt,
                                x_e[i][1] + prob.v_e * np.sin(x_e[i][2]) * prob.dt,
                                x_e[i][2] + u[i] * prob.dt])

            x_p_new = np.array([x_p[i][0] + prob.v_p * np.cos(x_p[i][2]) * prob.dt,
                                x_p[i][1] + prob.v_p * np.sin(x_p[i][2]) * prob.dt,
                                x_p[i][2] + d[i] * prob.dt])
            x_e.append(x_e_new)
            x_p.append(x_p_new)

        x_e = np.asarray(x_e)
        x_p = np.asarray(x_p)
        plt.plot(x_e[:, 0], x_e[:, 1], 'r', x_p[:, 0], x_p[:, 1], 'b')
        plt.show()

        return {"evader states": x_e,
                "pursuer states": x_p}

