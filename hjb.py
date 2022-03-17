import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate
from problem import BRT

N = 20
rho = 100
prob = BRT(N=N, T=1.0, rho=rho)
opti = ca.Opti()

# control variable
u_static = [3] * N
d = opti.variable(1, N)

# state
states = opti.variable(3, N)
x1 = states[0, :]
x2 = states[1, :]
x3 = states[2, :]

# parameters
opt_x0 = opti.parameter(3)

# init_condition
opti.subject_to(states[:, 0] == opt_x0)
con = [1] * (N - 1)
# dynamic constraint
for i in range(N - 1):
    #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
    # d[:, i] = d_star(lam[:, i])
    x_next = states[:, i] + prob.f(states[:, i], u_static[i], d[:, i]) * prob.dt
    con[i] = states[:, i + 1] == x_next
    opti.subject_to(con[i])

# INITIAL GUESSES
# ini_state = np.array([0.5, -0.5, 1.586])
# d_static = [3] * N
# u_static = [3] * N
# s = Simulate(N, u_static, d_static, ini_state)
# ss = s.rollout()
#
# opti.set_initial(x1, ss[0, :].tolist())
# opti.set_initial(x2, ss[1, :].tolist())
# opti.set_initial(x3, ss[2, :].tolist())
# opti.set_initial(d, [3] * N)

# adv. obj to minimize the min of l(x)
opti.minimize(prob.F(states))

# boundary and control conditions
# opti.subject_to(opti.bounded(-1.0, x1, 1.0))
# opti.subject_to(opti.bounded(-1.0, x2, 1.0))
# opti.subject_to(opti.bounded(-math.pi, x3, math.pi))
#
# opti.subject_to(opti.bounded(-3, d, 3))

ini_state = np.array([[0.5], [-0.5], [1.586]])
opti.set_value(opt_x0, ini_state)

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-8,
                'ipopt.inf_pr_output': "original"}

opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(states)
res_d = sol.value(d)
res_obj = np.min(prob.l_batch(res_x))

# print(res_x)
# print(res_d)
# print(res_obj)

# check Lagrangian
# NOTE: Casadi flips the sign of lagrangian multipliers!
dLdx = [prob.dFdx(res_x, i) - sol.value(opti.dual(con[i - 1])) +
        np.matmul(prob.dfdx(res_x[:, i], res_d[i]), sol.value(opti.dual(con[i])))
        for i in range(1, N - 1)]

print(dLdx)

ini_state = np.array([0.5, -0.5, 1.586])
d_static = [3] * N
u_static = [3] * N
s = Simulate(N, u_static, d_static, ini_state)
states = s.rollout()
loss = s.loss(states)
print(np.min(loss))
