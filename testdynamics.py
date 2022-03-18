# for debug only
import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate
from problem import TestDynamics

N = 3
prob = TestDynamics(N=N)
opti = ca.Opti()
x = opti.variable(2, N - 1)
u = opti.variable(N - 1)
opt_x0 = opti.parameter(2)

con = [1] * (N - 1)
# dynamic constraint
x_next = opt_x0 + prob.f(opt_x0, u[0]) * prob.dt
con[0] = x[:, 0] == x_next
opti.subject_to(con[0])
for i in range(1, N - 1):
    #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
    # d[:, i] = d_star(lam[:, i])
    x_next = x[:, i - 1] + prob.f(x[:, i - 1], u[i]) * prob.dt
    con[i] = x[:, i] == x_next
    opti.subject_to(con[i])

ini_state = np.array([[0.0], [-0.1]])
opti.set_value(opt_x0, ini_state)
opti.subject_to(opti.bounded(-1.0, u, 1.0))
opti.minimize(prob.F(x))

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-8,
                'ipopt.inf_pr_output': "original"}
opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(x)
res_u = sol.value(u)
res_obj = prob.F(res_x)

# check Lagrangian
dLdx = [prob.dFdx(res_x, i - 1) + sol.value(opti.dual(con[i - 1])) -
        np.matmul(prob.dfdx(), sol.value(opti.dual(con[i])))
        for i in range(1, N - 1)]
print(dLdx)
print()

