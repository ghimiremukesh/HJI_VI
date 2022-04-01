"""
Generate Data for Supervised Learning

Format:
x1, x2, x3, V, dual

Test 1: 1. generate value function for the states from true data
        2. interpolate and plot the value slice at x3 = pi/2 at final time (0.9s)
        3. compare the plot with the deepreach's valfunc.

Use states from the BRT_Raw_valfuncs. next
"""

import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate, Utility
from problem import BRT
import matplotlib.pyplot as plt

N = 10
rho = 100
prob = BRT(N=N, T=1.0, rho=rho)
opti = ca.Opti()

# brt_raw = scipy.io.loadmat("Air3D_raw_valfuncs.mat") # raw value functions
true_data = scipy.io.loadmat("analytical_BRT_air3D.mat")
states = true_data['gmat']


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
    opti.set_initial(u, u_last)

    # adv. obj to minimize the min of l(x)
    opti.minimize(-prob.F(x))

    # boundary and control conditions
    # opti.subject_to(opti.bounded(-1.0, x1, 1.0))
    # opti.subject_to(opti.bounded(-1.0, x2, 1.0))
    opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

    opti.subject_to(opti.bounded(-3, u, 3))

    opti.set_value(opt_x0, x0)

    opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 1,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-4,
                    'ipopt.inf_pr_output': "original"}

    opti.solver('ipopt', opts_setting)
    sol = opti.solve()

    res_x = sol.value(x)
    res_u = sol.value(u)
    res_obj = np.min(prob.l_batch(res_x))

    # return all costates
    costates = [sol.value(opti.dual(con[i])) for i in range(N - 1)]

    return res_u, res_x, costates


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
    opti.set_initial(d, d_last)

    # adv. obj to minimize the min of l(x)
    opti.minimize(prob.F(x))

    # boundary and control conditions
    # opti.subject_to(opti.bounded(-1.0, x1, 1.0))
    # opti.subject_to(opti.bounded(-1.0, x2, 1.0))
    opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

    opti.subject_to(opti.bounded(-3, d, 3))

    opti.set_value(opt_x0, x0)

    opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 1,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-4, 'ipopt.acceptable_obj_change_tol': 1e-4,
                    'ipopt.inf_pr_output': "original"}

    opti.solver('ipopt', opts_setting)
    sol = opti.solve()

    res_x = sol.value(x)
    res_d = sol.value(d)
    res_obj = np.min(prob.l_batch(res_x))

    # return all costates
    costates = [sol.value(opti.dual(con[i])) for i in range(N - 1)]

    return res_d, res_x, costates


# calculate BRT value
def get_value(x, idx, T):
    """
    @param x: states (x1, x2, x3)
    @param idx: index after which the values are constant
    @param T: total timesteps
    @return: value array for T timesteps
    """
    value = np.empty((1, T))
    collision_r = 0.25  # collision radius as defined in BRT
    # val = math.inf  # to get the min
    l = [np.linalg.norm(x[:2, k]) - 0.25 for k in range(T)]
    for i in range(idx+1):
        val = np.min(l[i:]) # scan the min in the future
        value[i] = val
    if idx+1 < T:
        value[idx+1:T] = value[idx]
    return value


util = Utility()
d = [0] * (N - 1)
u = [0] * (N - 1)

# set termination criterion
u_prev = u
d_prev = d
change = 1
tol = 1e-3  # Tolerance for norm

# generate ground truth data for #samples
samples = 101
# data_ground = {"evader_states": [],
#                "pursuer_states": []}

# save the value functions
val_funcs = np.empty((samples, samples, samples, N))
count = 0 # for debugging
for i in range(101):
    for j in range(101):
        for k in range(101):
            x0 = states[i, j, k, :]
            x0 = np.reshape(x0, (3, 1)) # reshape into PMP format
            while change > tol:
                u, x_u, costate_u = solve_evade(d, u, x0)
                # util.brt_plot(u, d, x0.flatten(), prob)

                d, x_d, costate_d = solve_pursue(u, d, x0)
                util.brt_plot(u, d, x0.flatten(), prob)

                # find the point after which the co-states are almost zero
                # and check the convergence of control before that
                co_norm = np.linalg.norm(costate_d, ord=-math.inf, axis=1)
                idx = np.where(co_norm < tol)[0][0]  # get the start index from where the control doesn't matter
                # calculate the change in control[:idx]
                change = (np.linalg.norm(d[:idx] - d_prev[:idx]) > np.linalg.norm(u[:idx] - u_prev[:idx])) * np.linalg.norm(
                    d[:idx] - d_prev[:idx]) + \
                         (np.linalg.norm(d[:idx] - d_prev[:idx]) < np.linalg.norm(u[:idx] - u_prev[:idx])) * np.linalg.norm(
                    u[:idx] - u_prev[:idx])  # pick whichever is max

                u_prev = u
                d_prev = d

            # add to the list of ground truth data
            gt = util.brt_plot(u, d, x0.flatten(), prob)  # returns a dictionary of the ground truth
            val_funcs[i, j, k, :] = get_value(np.concatenate((x0, x_d), axis=1), idx, N)

            count += 1
            print("%d data generated\n" % count)

        # add to the ground truth data
#     data_ground["evader_states"].append(gt["evader_states"])
#     data_ground["pursuer_states"].append(gt["pursuer_states"])


# save the value function from PMP
val_functions = {"PMP": val_funcs}
scipy.io.savemat("Air3D_valfuncs_pmp.mat", val_functions)
