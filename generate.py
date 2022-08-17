'''
This script generates data from scratch using time-marching.
'''

import numpy as np
from utilities.BVP_solver import solve_bvp
# from scipy.integrate import solve_bvp
from utilities.BVP_solver1 import solve_bvp1
from utilities.BVP_solver2 import solve_bvp2
import scipy.io
import time
import warnings
import copy
import matplotlib.pyplot as plt

from utilities.other import int_input

from examples.choose_problem import system, problem, config

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')
np.random.seed(config.random_seeds['generate'])
N_states = problem.N_states
alpha = problem.alpha

tags = ['na_na', 'na_a', 'a_a', 'a_na']
idx = 0  # change the case you choose

tag = tags[idx]
if idx == 0:
    problem.theta1 = 5
    problem.theta2 = 5
elif idx == 1:
    problem.theta1 = 5
    problem.theta2 = 1
elif idx == 2:
    problem.theta1 = 1
    problem.theta2 = 1
else:
    problem.theta1 = 1
    problem.theta2 = 5

print(problem.theta1, problem.theta2)

# Validation or training data?
data_type = int_input('What kind of data? Enter 0 for validation, 1 for training:')
if data_type:
    # data_type = 'train'
    data_type = 'test'
else:
    data_type = 'val'

save_path1 = 'examples/' + system + '/data_' + data_type + f'_{tag}_2.mat'
# save_path1 = 'examples/' + system + '/data_' + data_type + f'_{tag}_1.mat'
# save_path2 = 'examples/' + system + '/data_' + data_type + f'_{tag}_2.mat'
# save_path3 = 'examples/' + system + '/data_' + data_type + f'_{tag}_3.mat'
print(save_path1)
# print(save_path2)
# print(save_path3)
Ns = config.Ns[data_type]
X0_pool = problem.sample_X0(Ns)


# path = 'examples/' + system + '/data_train_a_na_1.mat'
# data = scipy.io.loadmat(path)
# t = data['t']  # time is from train_data
# X = data['X']
# data.update({'t0': data['t']})
# idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
# X0 = X[:, idx0][:, 1]
#
# X0_pool = X0

'''
If we want to separate the data into upper and lower triangle, we should use
'''
t_OUT1 = np.empty((1, 0))
X_OUT1 = np.empty((2 * N_states, 0))
A_OUT1 = np.empty((4 * N_states, 0))
V_OUT1 = np.empty((2, 0))

t_OUT2 = np.empty((1, 0))
X_OUT2 = np.empty((2 * N_states, 0))
A_OUT2 = np.empty((4 * N_states, 0))
V_OUT2 = np.empty((2, 0))

t_OUT3 = np.empty((1, 0))
X_OUT3 = np.empty((2 * N_states, 0))
A_OUT3 = np.empty((4 * N_states, 0))
V_OUT3 = np.empty((2, 0))

N_sol = 0
sol_time = []
step = 0
# X0 = X0_pool
N_opt1 = 0
N_opt2 = 0
X0 = X0_pool[:, 0]
X0_orignal = copy.deepcopy(X0)

N_converge = 0

time_horizon = problem.t1
print(time_horizon, problem.t1)

# ---------------------------------------------------------------------------- #
while N_sol < Ns:
    print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end=' ')

    step += 1
    X0 = X0_pool[:, N_sol]
    print(step)
    print(X0)
    bc = problem.make_bc(X0)

    start_time = time.time()
    tol = 5e-3  # 1e-01

    # Initial guess setting --> #Todo: what is this??
    X_guess = np.vstack((X0.reshape(-1, 1),
                         np.array([[alpha],
                                   [alpha * time_horizon],  # alpha * 3.
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [alpha],
                                   [alpha * time_horizon],  # alpha * 3.
                                   [0.],
                                   [0.]])))

    # Without time marching for BVP_solver
    collision_lower1 = problem.R1 / 2 - problem.theta1 * problem.W1 / 2
    collision_upper1 = problem.R1 / 2 + problem.W1 / 2 + problem.L1
    collision_lower2 = problem.R2 / 2 - problem.theta2 * problem.W2 / 2
    collision_upper2 = problem.R2 / 2 + problem.W2 / 2 + problem.L2
    X_guess1 = X_guess

    '''
    boundary line is based on lower boundary of the collision box, which is used to verify the trajectory for both cars
    position line is the ration of the initial position of car2 and car1, which is used to verify the trajectory for both cars
    '''
    boundary_line = collision_lower2 / collision_lower1
    position_line = X_guess1[2, 0] / X_guess1[0, 0]

    '''
    New guessing for X and t, go through all the actions and find the global solutions
    '''
    V_list = np.empty((1, 0))
    actions = [(10, -5), (-5, 10), (-5, -5), (10, 10)]
    X_sol = []
    Y_sol = []
    rms_sol = []
    '''
    time horizon is 3 seconds
    '''
    t_guess1 = np.linspace(0, time_horizon, 4)

    for i in range(3):
        X_guess1 = np.hstack((X_guess1, X_guess))
    for action in actions:
        a1, a2 = action
        for i in range(int(time_horizon)):
            X_guess1[0, i + 1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i + 1] + (a1 / 2) * t_guess1[i + 1] ** 2
            X_guess1[1, i + 1] = X_guess1[1, 0] + a1 * t_guess1[i + 1]
            X_guess1[2, i + 1] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i + 1] + (a2 / 2) * t_guess1[i + 1] ** 2
            X_guess1[3, i + 1] = X_guess1[3, 0] + a2 * t_guess1[i + 1]
            X_guess1[5, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[11, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)

        SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                        verbose=2, tol=tol, max_nodes=1500)
        # try:
        #     SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
        #                     verbose=2, tol=tol, max_nodes=1500)
        # except Exception as e:
        #     continue

        max_rms = np.max(SOL.rms_residuals)
        if max_rms < tol:
            V_tmp = (-SOL.y[-2:-1, 0:1]) + (-SOL.y[-1:, 0:1])
            V_list = np.hstack((V_list, V_tmp))
            X_sol.append(SOL.x)
            Y_sol.append(SOL.y)
            rms_sol.append(SOL.rms_residuals)
        else:
            pass

    if X_sol == []:
        pass
    else:
        index = np.argmax(V_list)

        t = X_sol[index]
        X = Y_sol[index][:2 * N_states]
        A = Y_sol[index][2 * N_states:6 * N_states]
        V1 = -Y_sol[index][-2:-1]
        V2 = -Y_sol[index][-1:]
        V = np.vstack((V1, V2))
        rms = np.max(rms_sol[index])

        sol_time.append(time.time() - start_time)

        t_OUT1 = np.hstack((t_OUT1, t.reshape(1, -1)))
        X_OUT1 = np.hstack((X_OUT1, X))
        A_OUT1 = np.hstack((A_OUT1, A))
        V_OUT1 = np.hstack((V_OUT1, V))

        N_converge += 1

        # # depends on the lambda sign to separate the data
        # if A[0, 0:1] > 0 and A[6, 0:1] < 0:
        #     t_OUT1 = np.hstack((t_OUT1, t.reshape(1, -1)))
        #     X_OUT1 = np.hstack((X_OUT1, X))
        #     A_OUT1 = np.hstack((A_OUT1, A))
        #     V_OUT1 = np.hstack((V_OUT1, V))
        # if A[0, 0:1] < 0 and A[6, 0:1] > 0:
        #     t_OUT2 = np.hstack((t_OUT2, t.reshape(1, -1)))
        #     X_OUT2 = np.hstack((X_OUT2, X))
        #     A_OUT2 = np.hstack((A_OUT2, A))
        #     V_OUT2 = np.hstack((V_OUT2, V))
        # if A[0, 0:1] > 0 and A[6, 0:1] > 0:
        #     t_OUT3 = np.hstack((t_OUT3, t.reshape(1, -1)))
        #     X_OUT3 = np.hstack((X_OUT3, X))
        #     A_OUT3 = np.hstack((A_OUT3, A))
        #     V_OUT3 = np.hstack((V_OUT3, V))

    N_sol += 1

# for m in range(6):  # segment for speed x2, separate into 11 segment
#     for n in range(6):
#         print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end=' ')
#
#         step += 1
#         # X0 = X0_pool[:, N_sol]
#         print(step)
#         print(X0)
#         bc = problem.make_bc(X0)
#
#         start_time = time.time()
#         tol = 5e-3  # 1e-01
#
#         # Initial guess setting --> #Todo: what is this??
#         X_guess = np.vstack((X0.reshape(-1, 1),
#                              np.array([[alpha],
#                                        [alpha * time_horizon],  # alpha * 3.
#                                        [0.],
#                                        [0.],
#                                        [0.],
#                                        [0.],
#                                        [alpha],
#                                        [alpha * time_horizon],  # alpha * 3.
#                                        [0.],
#                                        [0.]])))
#
#         # Without time marching for BVP_solver
#         collision_lower1 = problem.R1 / 2 - problem.theta1 * problem.W1 / 2
#         collision_upper1 = problem.R1 / 2 + problem.W1 / 2 + problem.L1
#         collision_lower2 = problem.R2 / 2 - problem.theta2 * problem.W2 / 2
#         collision_upper2 = problem.R2 / 2 + problem.W2 / 2 + problem.L2
#         X_guess1 = X_guess
#
#         '''
#         boundary line is based on lower boundary of the collision box, which is used to verify the trajectory for both cars
#         position line is the ration of the initial position of car2 and car1, which is used to verify the trajectory for both cars
#         '''
#         boundary_line = collision_lower2 / collision_lower1
#         position_line = X_guess1[2, 0] / X_guess1[0, 0]
#
#         '''
#         New guessing for X and t, go through all the actions and find the global solutions
#         '''
#         V_list = np.empty((1, 0))
#         actions = [(10, -5), (-5, 10), (-5, -5), (10, 10)]
#         X_sol = []
#         Y_sol = []
#         rms_sol = []
#         '''
#         time horizon is 3 seconds
#         '''
#         t_guess1 = np.linspace(0, time_horizon, 4)
#
#         for i in range(3):
#             X_guess1 = np.hstack((X_guess1, X_guess))
#         for action in actions:
#             a1, a2 = action
#             for i in range(int(time_horizon)):
#                 X_guess1[0, i + 1] = X_guess1[0, 0] + X_guess1[1, 0] * t_guess1[i + 1] + (a1 / 2) * t_guess1[i + 1] ** 2
#                 X_guess1[1, i + 1] = X_guess1[1, 0] + a1 * t_guess1[i + 1]
#                 X_guess1[2, i + 1] = X_guess1[2, 0] + X_guess1[3, 0] * t_guess1[i + 1] + (a2 / 2) * t_guess1[i + 1] ** 2
#                 X_guess1[3, i + 1] = X_guess1[3, 0] + a2 * t_guess1[i + 1]
#                 X_guess1[5, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
#                 X_guess1[11, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
#
#             SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
#                             verbose=2, tol=tol, max_nodes=1500)
#             # try:
#             #     SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
#             #                     verbose=2, tol=tol, max_nodes=1500)
#             # except Exception as e:
#             #     continue
#
#             max_rms = np.max(SOL.rms_residuals)
#             if max_rms < tol:
#                 V_tmp = (-SOL.y[-2:-1, 0:1]) + (-SOL.y[-1:, 0:1])
#                 V_list = np.hstack((V_list, V_tmp))
#                 X_sol.append(SOL.x)
#                 Y_sol.append(SOL.y)
#                 rms_sol.append(SOL.rms_residuals)
#             else:
#                 pass
#
#         if X_sol == []:
#             pass
#         else:
#             index = np.argmax(V_list)
#
#             t = X_sol[index]
#             X = Y_sol[index][:2 * N_states]
#             A = Y_sol[index][2 * N_states:6 * N_states]
#             V1 = -Y_sol[index][-2:-1]
#             V2 = -Y_sol[index][-1:]
#             V = np.vstack((V1, V2))
#             rms = np.max(rms_sol[index])
#
#             sol_time.append(time.time() - start_time)
#
#             t_OUT1 = np.hstack((t_OUT1, t.reshape(1, -1)))
#             X_OUT1 = np.hstack((X_OUT1, X))
#             A_OUT1 = np.hstack((A_OUT1, A))
#             V_OUT1 = np.hstack((V_OUT1, V))
#
#             N_converge += 1
#
#         N_sol += 1
#
#         X0[0] = X0[0] + 1  # step for x1 is 0.5 m
#         # if abs(X0[0] - 17.5) < 1e-6:
#         #     X0[0] = X0[0] + 0.1
#     X0[2] = X0[2] + 1  # step for x2 is 0.5 m
#     X0[0] = X0_orignal[0]

# ---------------------------------------------------------------------------- #

sol_time = np.sum(sol_time)

print('')
print(step, '/', step, 'successful solution attempts:')
print('Average solution time: %1.1f' % (sol_time / step), 'sec')
print('Total solution time: %1.1f' % (sol_time), 'sec')

print('')
print('Total data generated:', X_OUT1.shape[1] + X_OUT2.shape[1] + X_OUT3.shape[1])
print('Converge Number:', N_converge)
print('')

# ---------------------------------------------------------------------------- #
save_data = int_input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    try:
        save_dict1 = scipy.io.loadmat(save_path1)
        # save_dict2 = scipy.io.loadmat(save_path2)
        # save_dict3 = scipy.io.loadmat(save_path3)

        overwrite_data = int_input('Overwrite existing data? Enter 0 for no, 1 for yes:')

        if overwrite_data:
            raise RuntimeWarning

        save_dict1.update({'t': np.hstack((save_dict1['t'], t_OUT1)),
                           'X': np.hstack((save_dict1['X'], X_OUT1)),
                           'A': np.hstack((save_dict1['A'], A_OUT1)),
                           'V': np.hstack((save_dict1['V'], V_OUT1))})

        # save_dict2.update({'t': np.hstack((save_dict2['t'], t_OUT2)),
        #                    'X': np.hstack((save_dict2['X'], X_OUT2)),
        #                    'A': np.hstack((save_dict2['A'], A_OUT2)),
        #                    'V': np.hstack((save_dict2['V'], V_OUT2))})
        #
        # save_dict3.update({'t': np.hstack((save_dict3['t'], t_OUT3)),
        #                    'X': np.hstack((save_dict3['X'], X_OUT3)),
        #                    'A': np.hstack((save_dict3['A'], A_OUT3)),
        #                    'V': np.hstack((save_dict3['V'], V_OUT3))})

    except:
        U1, U2 = problem.U_star(np.vstack((X_OUT1, A_OUT1)))
        U = np.vstack((U1, U2))

        save_dict1 = {'lb_1': np.min(X_OUT1[:N_states], axis=1, keepdims=True),
                      'ub_1': np.max(X_OUT1[:N_states], axis=1, keepdims=True),
                      'lb_2': np.min(X_OUT1[N_states:2 * N_states], axis=1, keepdims=True),
                      'ub_2': np.max(X_OUT1[N_states:2 * N_states], axis=1, keepdims=True),
                      'A_lb_11': np.min(A_OUT1[:N_states], axis=1, keepdims=True),
                      'A_ub_11': np.max(A_OUT1[:N_states], axis=1, keepdims=True),
                      'A_lb_12': np.min(A_OUT1[N_states:2 * N_states], axis=1, keepdims=True),
                      'A_ub_12': np.max(A_OUT1[N_states:2 * N_states], axis=1, keepdims=True),
                      'A_lb_21': np.min(A_OUT1[2 * N_states:3 * N_states], axis=1, keepdims=True),
                      'A_ub_21': np.max(A_OUT1[2 * N_states:3 * N_states], axis=1, keepdims=True),
                      'A_lb_22': np.min(A_OUT1[3 * N_states:4 * N_states], axis=1, keepdims=True),
                      'A_ub_22': np.max(A_OUT1[3 * N_states:4 * N_states], axis=1, keepdims=True),
                      'U_lb_1': np.min(U1, axis=1, keepdims=True),
                      'U_ub_1': np.max(U1, axis=1, keepdims=True),
                      'U_lb_2': np.min(U2, axis=1, keepdims=True),
                      'U_ub_2': np.max(U2, axis=1, keepdims=True),
                      'V_min_1': np.min(V_OUT1[-2:-1, :]), 'V_max_1': np.max(V_OUT1[-2:-1, :]),
                      'V_min_2': np.min(V_OUT1[-1, :]), 'V_max_2': np.max(V_OUT1[-1, :]),
                      't': t_OUT1, 'X': X_OUT1, 'A': A_OUT1, 'V': V_OUT1}
        scipy.io.savemat(save_path1, save_dict1)

        # U1, U2 = problem.U_star(np.vstack((X_OUT2, A_OUT2)))
        # U = np.vstack((U1, U2))
        #
        # save_dict2 = {'lb_1': np.min(X_OUT2[:N_states], axis=1, keepdims=True),
        #               'ub_1': np.max(X_OUT2[:N_states], axis=1, keepdims=True),
        #               'lb_2': np.min(X_OUT2[N_states:2 * N_states], axis=1, keepdims=True),
        #               'ub_2': np.max(X_OUT2[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_lb_11': np.min(A_OUT2[:N_states], axis=1, keepdims=True),
        #               'A_ub_11': np.max(A_OUT2[:N_states], axis=1, keepdims=True),
        #               'A_lb_12': np.min(A_OUT2[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_ub_12': np.max(A_OUT2[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_lb_21': np.min(A_OUT2[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #               'A_ub_21': np.max(A_OUT2[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #               'A_lb_22': np.min(A_OUT2[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #               'A_ub_22': np.max(A_OUT2[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #               'U_lb_1': np.min(U1, axis=1, keepdims=True),
        #               'U_ub_1': np.max(U1, axis=1, keepdims=True),
        #               'U_lb_2': np.min(U2, axis=1, keepdims=True),
        #               'U_ub_2': np.max(U2, axis=1, keepdims=True),
        #               'V_min_1': np.min(V_OUT2[-2:-1, :]), 'V_max_1': np.max(V_OUT2[-2:-1, :]),
        #               'V_min_2': np.min(V_OUT2[-1, :]), 'V_max_2': np.max(V_OUT2[-1, :]),
        #               't': t_OUT2, 'X': X_OUT2, 'A': A_OUT2, 'V': V_OUT2}
        # scipy.io.savemat(save_path2, save_dict2)
        #
        # U1, U2 = problem.U_star(np.vstack((X_OUT3, A_OUT3)))
        # U = np.vstack((U1, U2))
        #
        # save_dict3 = {'lb_1': np.min(X_OUT3[:N_states], axis=1, keepdims=True),
        #               'ub_1': np.max(X_OUT3[:N_states], axis=1, keepdims=True),
        #               'lb_2': np.min(X_OUT3[N_states:2 * N_states], axis=1, keepdims=True),
        #               'ub_2': np.max(X_OUT3[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_lb_11': np.min(A_OUT3[:N_states], axis=1, keepdims=True),
        #               'A_ub_11': np.max(A_OUT3[:N_states], axis=1, keepdims=True),
        #               'A_lb_12': np.min(A_OUT3[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_ub_12': np.max(A_OUT3[N_states:2 * N_states], axis=1, keepdims=True),
        #               'A_lb_21': np.min(A_OUT3[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #               'A_ub_21': np.max(A_OUT3[2 * N_states:3 * N_states], axis=1, keepdims=True),
        #               'A_lb_22': np.min(A_OUT3[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #               'A_ub_22': np.max(A_OUT3[3 * N_states:4 * N_states], axis=1, keepdims=True),
        #               'U_lb_1': np.min(U1, axis=1, keepdims=True),
        #               'U_ub_1': np.max(U1, axis=1, keepdims=True),
        #               'U_lb_2': np.min(U2, axis=1, keepdims=True),
        #               'U_ub_2': np.max(U2, axis=1, keepdims=True),
        #               'V_min_1': np.min(V_OUT3[-2:-1, :]), 'V_max_1': np.max(V_OUT3[-2:-1, :]),
        #               'V_min_2': np.min(V_OUT3[-1, :]), 'V_max_2': np.max(V_OUT3[-1, :]),
        #               't': t_OUT3, 'X': X_OUT3, 'A': A_OUT3, 'V': V_OUT3}
        # scipy.io.savemat(save_path3, save_dict3)