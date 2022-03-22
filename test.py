# for debug only
import casadi as ca
import numpy as np
import time
import math
import scipy.io
from util import Simulate
from problem import TestFEA

prob = TestFEA()
opti = ca.Opti()
x = opti.variable(4)

# prob.F(np.ones(4))

con = prob.h(x) == 0
opti.subject_to(con)
opti.subject_to(opti.bounded(1e-3, x, 1000))
opti.minimize(prob.F(x))

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-8,
                'ipopt.inf_pr_output': "original"}
opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(x)
res_obj = prob.F(res_x)

# check Lagrangian
# dLdx = prob.dFdx(res_x) - sol.value(opti.dual(con)) * np.array([1, 1])
# print(dLdx)

