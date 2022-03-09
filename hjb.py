import casadi as ca
import numpy as np
import time
import math
import scipy.io


v_e = 0.75
v_p = 0.75
N = 20
dt = 0.1 / N

opti = ca.Opti()

# constraint lambda, mu
epsilon = opti.variable(1, N)
C = opti.variable(1, 1)

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


# dynamic function
def f(x_, u_, d_): return ca.vertcat(
    -0.75 + 0.75 * ca.cos(x_[2]) + u_ * x_[1],
    0.75 * ca.sin(x_[2]) - u_ * x_[0],
    d_ - u_)


def l(x_): return ca.sqrt(x_[0] ** 2 + x_[1] ** 2 + 1e-9) - 0.25


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


def u_star(lam, x_, dt):
    return 3 * ca.sign((lam[0] * x_[1] - lam[1] * x_[0] - lam[2]) * dt)


def d_star(lam): return 3 * ca.sign(lam[2])


# init_condition
opti.subject_to(states[:, 0] == opt_x0)

# dynamic constraint
for i in range(N - 1):
    #     u[:, i] = u_star(lam[:, i + 1], states[:, i], dt)
    # d[:, i] = d_star(lam[:, i])
    x_next = states[:, i] + f(states[:, i], u_static[i], d[:, i]) * dt
    opti.subject_to(states[:, i + 1] == x_next)

# multiplier constraint
for i in range(N):
    opti.subject_to(epsilon[:, i] >= 0)
    opti.subject_to(C <= l(states[:, i]))
    opti.subject_to(C >= l(states[:, i]) - epsilon[:, i])

# INITIAL GUESSES
data = scipy.io.loadmat('ground_truth_data_20.mat')
x1_gus = data['x1'].flatten().tolist()
x2_gus = data['x2'].flatten().tolist()
x3_gus = data['x3'].flatten().tolist()

opti.set_initial(x1, x1_gus)
opti.set_initial(x2, x2_gus)
opti.set_initial(x3, x3_gus)

# objective function
obj = C
for i in range(N):
    obj = obj + epsilon[:, i]

opti.minimize(obj)

# boundary and control conditions
opti.subject_to(opti.bounded(-1.0, x1, 1.0))
opti.subject_to(opti.bounded(-1.0, x2, 1.0))
opti.subject_to(opti.bounded(-math.pi, x3, math.pi))

opti.subject_to(opti.bounded(-3, d, 3))

ini_state = np.array([[0.5], [-0.5], [1.59]])
opti.set_value(opt_x0, ini_state)

opts_setting = {'ipopt.hessian_approximation': "limited-memory", 'ipopt.max_iter': 20000, 'ipopt.print_level': 5,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6,
                'ipopt.inf_pr_output': "original"}

opti.solver('ipopt', opts_setting)
sol = opti.solve()

res_x = sol.value(states)
res_d = sol.value(d)
res_epsilon = sol.value(epsilon)

print(res_x)
print(res_d)
print()



