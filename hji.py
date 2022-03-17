# Solve the PMP formulation for a single trajectory of HJI BRT

import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate

v_e = 0.75
v_p = 0.75
N = 20
dt = 1 / N
rho = 10

opti = ca.Opti()

# constraint lambda, mu
lam = opti.variable(3, N - 1)

# control variable
u = opti.variable(N - 1)
# d = opti.variable(N)

# state
states = opti.variable(3, N)
x1 = states[0, :]
x2 = states[1, :]
x3 = states[2, :]

# parameters
opt_x0 = opti.parameter(3)


# dynamic function
def f(x_, u_, d_): return ca.vertcat(
    -0.75 + 0.75 * ca.cos(x_[2]) + u_ * x_[1],
    0.75 * ca.sin(x_[2]) - u_ * x_[0],
    d_ - u_)


def l(x_): return ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9) - 0.25


def l_batch(x_): return ca.sqrt(x_[0, :] ** 2 + x_[1, :] ** 2 + 1e-9) - 0.25


# gradient l(x_i) w.r.t x_i
def dldx(x_): return ca.vertcat(
    x_[0] / ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9),
    x_[1] / ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9),
    0)


# gradient f(x_i) w.r.t x_i, I + df/dx * dt
def dfdx(x_, u_, dt):
    row1 = ca.horzcat(1, -u_ * dt, 0)
    row2 = ca.horzcat(u_ * dt, 1, 0)
    row3 = ca.horzcat(-0.75 * ca.sin(x_[2]) * dt, 0.75 * ca.cos(x_[2]) * dt, 1)
    return ca.vertcat(row1,
                      row2,
                      row3)


def F(x_):
    return - 1 / rho * ca.log(ca.sum2(ca.exp(- rho * l_batch(x_))))


def dFdx(x_, id):  # x_: state trj, i: time step
        e = ca.exp(-rho * l_batch(x_))
        softmax = e[id]/ca.sum2(e)
        return softmax * dldx(x_[:, id])


def u_star(lam, x_, dt):
    return 3 * ca.sign((lam[0] * x_[1] - lam[1] * x_[0] - lam[2]) * dt)


def d_star(lam): return 3 * ca.sign(lam[2])


# init_condition
opti.subject_to(states[:, 0] == opt_x0)

# dynamic constraint
for i in range(N - 1):
    #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
    # d[:, i] = d_star(lam[:, i])
    x_next = states[:, i] + f(states[:, i], u[i], d_star(lam[:, i])) * dt
    opti.subject_to(states[:, i + 1] == x_next)

# costate constraint
for i in range(1, N - 1):
    opti.subject_to(dFdx(states, i) + lam[:, i - 1] -
                    ca.mtimes(dfdx(states[:, i], u[i], dt), lam[:, i]) == 0)
opti.subject_to(dFdx(states,  N - 1) + lam[:, N - 2] == 0)


# INITIAL GUESSES
ini_state = np.array([0.5, -0.5, 1.586])
d_static = [3] * (N - 1)
u_static = [3] * (N - 1)
s = Simulate(N, u_static, d_static, ini_state)
ss = s.rollout()

opti.set_initial(x1, ss[0, :].tolist())
opti.set_initial(x2, ss[1, :].tolist())
opti.set_initial(x3, ss[2, :].tolist())
# opti.set_initial(d, [3] * N)
opti.set_initial(u, [3] * (N - 1))

# objective function
obj = -F(states)
opti.minimize(obj)

# boundary and control conditions
opti.subject_to(opti.bounded(-1.0, x1, 1.0))
opti.subject_to(opti.bounded(-1.0, x2, 1.0))
opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

# opti.subject_to(opti.bounded(-3, d, 3))
opti.subject_to(opti.bounded(-3, u, 3))

ini_state = np.array([[0.5], [-0.5], [1.586]])
opti.set_value(opt_x0, ini_state)

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 50000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-4,
                'ipopt.inf_pr_output': "original"}

opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(states)
res_u = sol.value(u)
res_lambda = sol.value(lam)
res_d = [d_star(res_lambda[:, i]) for i in range(N-1)]
res_obj = np.min(l_batch(res_x))
# res_d = sol.value(d)
# res_epsilon = sol.value(epsilon)

print(res_x)
print(res_u)
print(res_d)
print(res_lambda)
print(res_obj)

ini_state = np.array([0.5, -0.5, 1.586])
d_static = [3] * N
u_static = [3] * N
s = Simulate(N, u_static, d_static, ini_state)
states = s.rollout()
loss = s.loss(states)
print(np.min(loss))
print()