# try solving evade-pursue using iterative optimal control

import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util_new import Simulate, Utility
from problem import BRT

N = 20
rho = 100
prob = BRT(N=N, T=1.0, rho=rho)
opti = ca.Opti()


def solve_evade(d, u_last, x0):
    # control variable
    d_static = d
    u = opti.variable(1, N - 1)

    # state
    x = opti.variable(3, N - 1)
    x1 = x[0, :]
    x2 = x[1, :]
    x3 = x[2, :]

    # parameters
    opt_x0 = opti.parameter(3)

    con = [1] * (N - 1)
    x_next = opt_x0 + prob.f(opt_x0, u[:, 0], d_static[0]) * prob.dt
    con[0] = x[:, 0] == x_next
    opti.subject_to(con[0])
    # dynamic constraint
    for i in range(1, N - 1):
        #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
        # d[:, i] = d_star(lam[:, i])
        x_next = x[:, i - 1] + prob.f(x[:, i - 1], u[:, i], d_static[i]) * prob.dt
        con[i] = x[:, i] == x_next
        opti.subject_to(con[i])

    # INITIAL GUESSES
    # opti.set_initial(x1, x_last[0, :].tolist())
    # opti.set_initial(x2, x_last[1, :].tolist())
    # opti.set_initial(x3, x_last[2, :].tolist())
    # opti.set_initial(u, u_last)

    # adv. obj to minimize the min of l(x)
    opti.minimize(-prob.F(x))

    # boundary and control conditions
    opti.subject_to(opti.bounded(-1.0, x1, 1.0))
    opti.subject_to(opti.bounded(-1.0, x2, 1.0))
    opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

    opti.subject_to(opti.bounded(-3, u, 3))

    opti.set_value(opt_x0, x0)

    opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-4,
                    'ipopt.inf_pr_output': "original"}

    opti.solver('ipopt', opts_setting)
    sol = opti.solve()

    res_x = sol.value(x)
    res_u = sol.value(u)
    res_obj = np.min(prob.l_batch(res_x))

    return res_u, res_x


def solve_pursue(u, d_last, x0):
    # control variable
    u_static = u
    d = opti.variable(1, N - 1)

    # state
    x = opti.variable(3, N - 1)
    x1 = x[0, :]
    x2 = x[1, :]
    x3 = x[2, :]

    # parameters
    opt_x0 = opti.parameter(3)

    con = [1] * (N - 1)
    x_next = opt_x0 + prob.f(opt_x0, u_static[0], d[:, 0]) * prob.dt
    con[0] = x[:, 0] == x_next
    opti.subject_to(con[0])
    # dynamic constraint
    for i in range(1, N - 1):
        #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
        # d[:, i] = d_star(lam[:, i])
        x_next = x[:, i - 1] + prob.f(x[:, i - 1], u_static[i], d[:, i]) * prob.dt
        con[i] = x[:, i] == x_next
        opti.subject_to(con[i])

    # INITIAL GUESSES
    # opti.set_initial(x1, x_last[0, :].tolist())
    # opti.set_initial(x2, x_last[1, :].tolist())
    # opti.set_initial(x3, x_last[2, :].tolist())
    # opti.set_initial(d, d_last)

    # adv. obj to minimize the min of l(x)
    opti.minimize(prob.F(x))

    # boundary and control conditions
    opti.subject_to(opti.bounded(-1.0, x1, 1.0))
    opti.subject_to(opti.bounded(-1.0, x2, 1.0))
    opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

    opti.subject_to(opti.bounded(-3, d, 3))

    opti.set_value(opt_x0, x0)

    opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-4,
                    'ipopt.inf_pr_output': "original"}

    opti.solver('ipopt', opts_setting)
    sol = opti.solve()

    res_x = sol.value(x)
    res_d = sol.value(d)
    res_obj = np.min(prob.l_batch(res_x))

    return res_d, res_x


util = Utility()
itr = 3
x0 = np.array([[0.5], [-0.5], [1.586]])
x0 = np.array([[-0.52070], [0.88401], [-1.96348]])
samples = 10
x0s = np.random.uniform(-1, 1, (samples, 2, 1))
x0s = np.concatenate((x0s, np.random.uniform(-math.pi, math.pi, (samples, 1, 1))), 1)

errors = []
dataground = {"evader state": [],
              "pursuer state": []}

# x0 = np.array([[0.1], [0.], [-3.14]])
d = [0] * (N - 1)
u = [0] * (N - 1)
fail = False
for x0 in x0s:
    for i in range(itr):
        try:
            u, _ = solve_evade(d, u, x0)
        except:
            errors.append(x0)
            fail = True
            break
        util.brt_plot(u, d, x0.flatten(), prob)

        # d, _ = solve_evade(u, d, x0)
        try:
            d, _ = solve_pursue(u, d, x0)
        except:
            errors.append(x0)
            fail = True
            break

    if not fail:
        gt = util.brt_plot(u, d, x0.flatten(), prob)  # returns a dictionary of the ground truth

        # add to the ground truth data
        dataground["evader state"].append(gt["evader states"])
        dataground["pursuer state"].append(gt["pursuer states"])

print()
