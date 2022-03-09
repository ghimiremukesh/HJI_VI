# Test file to simulate trajectory given u and d

import numpy as np
import time
import math
import scipy.io


v_e = 0.75
v_p = 0.75
N = 20
dt = 0.1 / N

# control variable
u_static = [3] * N
d_static = [3] * N


# dynamic function
def f(x_, u_, d_): return [-0.75 + 0.75 * np.cos(x_[2]) + u_ * x_[1],
                           0.75 * np.sin(x_[2]) - u_ * x_[0],
                           d_ - u_]


def l(x_): return np.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9) - 0.25


states = np.empty((3, N))
ini_state = np.array([0.5, -0.5, 1.59])
states[:, 0] = ini_state

# dynamic constraint
for i in range(N - 1):
    states[:, i + 1] = states[:, i] + np.array(f(states[:, i], u_static[i], d_static[i])) * dt

print([l(states[:, i]) for i in range(N)])

