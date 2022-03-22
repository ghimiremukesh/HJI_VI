# Solve the PMP formulation for a single trajectory of HJI BRT

import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate, Utility
from problem import BRT

N = 20
rho = 100
prob = BRT(N=N, T=1.0, rho=rho)
opti = ca.Opti()

# constraint lambda, mu
lam = opti.variable(3, N - 1)

# control variable
u = opti.variable(N - 1)
d = opti.variable(N - 1)

# state
x = opti.variable(3, N - 1)

# parameters
opt_x0 = opti.parameter(3)

# dynamic constraint
s_con = [1] * (N - 1)
x_next = opt_x0 + prob.f(opt_x0, u[0], d[0]) * prob.dt
s_con[0] = x[:, 0] == x_next
opti.subject_to(s_con[0])
for i in range(1, N - 1):
    #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
    # d[:, i] = d_star(lam[:, i])
    x_next = x[:, i - 1] + prob.f(x[:, i - 1], u[i], d[i]) * prob.dt
    s_con[i] = x[:, i] == x_next
    opti.subject_to(s_con[i])

# Optimality conditions for the inner pursuer problem
# costates
co_con = [1] * (N - 2)
for i in range(1, N - 2):
    co_con[i - 1] = prob.dFdx(x, i - 1) + lam[:, i - 1] - ca.mtimes(prob.dfdx(x[:, i - 1], u[i]), lam[:, i]) == 0
    opti.subject_to(co_con[i - 1])
co_con[N - 3] = prob.dFdx(x,  N - 2) + lam[:, N - 2] == 0
opti.subject_to(co_con[N - 3])
# pursuer inputs maximize hamiltonian unless singularity
d_con = [1] * (N - 1)
for i in range(N - 1):
    d_con[i] = d[i] == 3 * ca.sign(ca.floor(lam[2, i]*1000))
    opti.subject_to(d_con[i])

# INITIAL GUESSES
# ini_state = np.array([0.5, -0.5, 1.586])
# d_static = [3] * (N - 1)
# u_static = [3] * (N - 1)
# s = Simulate(N, u_static, d_static, ini_state)
# ss = s.rollout()
#
# opti.set_initial(x1, ss[0, :].tolist())
# opti.set_initial(x2, ss[1, :].tolist())
# opti.set_initial(x3, ss[2, :].tolist())
# # opti.set_initial(d, [3] * N)
# opti.set_initial(u, [3] * (N - 1))

# objective function
obj = -prob.F(x)
opti.minimize(obj)

# boundary and control conditions
# opti.subject_to(opti.bounded(-1.0, x1, 1.0))
# opti.subject_to(opti.bounded(-1.0, x2, 1.0))
# opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

opti.subject_to(opti.bounded(-3, d, 3))
opti.subject_to(opti.bounded(-3, u, 3))

ini_state = np.array([[0.5], [-0.5], [1.586]])
opti.set_value(opt_x0, ini_state)

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 50000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-6, 'ipopt.acceptable_obj_change_tol': 1e-6,
                'ipopt.inf_pr_output': "original"}

opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(x)
res_u = sol.value(u)
res_lambda = sol.value(lam)
res_d = sol.value(d)
res_obj = np.min(prob.l_batch(res_x))

print(res_x)
print(res_u)
print(res_d)
print(res_lambda)
print(res_obj)

util = Utility()
util.brt_plot(res_u, res_d, ini_state.flatten(), prob)

ini_state = np.array([0.5, -0.5, 1.586])
d_static = [3] * N
u_static = [-3] * N
s = Simulate(N, u_static, d_static, ini_state)
states = s.rollout()
loss = s.loss(states)
print(np.min(loss))
print()