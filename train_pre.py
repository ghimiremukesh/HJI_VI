'''
Run this script to train the NN. Loads the problem and configuration
according to examples/choose_problem.py.
'''

import numpy as np
import torch
import scipy.io
import time
import matplotlib.pyplot as plt
from matplotlib import patches

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent
from utilities.neural_networks_pre import HJI_network

c = np.random.seed(config.random_seeds['train'])
# torch.manual_seed(config.random_seeds['train'])
# tf.set_random_seed(config.random_seeds['train'])

# ---------------------------------------------------------------------------- #

##### Loads data sets #####
train_data1 = scipy.io.loadmat('examples/' + system + '/data_train_a_a_1.mat')
train_data2 = scipy.io.loadmat('examples/' + system + '/data_train_a_a_2.mat')
val_data1 = scipy.io.loadmat('examples/' + system + '/data_val_a_a_1.mat')
val_data2 = scipy.io.loadmat('examples/' + system + '/data_val_a_a_2.mat')
test_data1 = scipy.io.loadmat('examples/' + system + '/data_test_a_a_1.mat')
test_data2 = scipy.io.loadmat('examples/' + system + '/data_test_a_a_2.mat')
N_states = problem.N_states

# when lambda 11 > 0 and lambda 22 < 0
def preprocess1(data):
    t = data['t']
    X = data['X']
    A = data['A']
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    A11 = np.zeros((len(idx0), 3))
    A12 = np.zeros((len(idx0), 3))
    A21 = np.zeros((len(idx0), 3))
    A22 = np.zeros((len(idx0), 3))
    for n in range(1, len(idx0)+1):
        if n == len(idx0):
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A11[n - 1, :] = [max(A[1, idx0[n - 1]:]),
                            (min(A[1, idx0[n - 1]:]) - max(A[1, idx0[n - 1]:])) / (-max(A[0, idx0[n - 1]:])),
                            min(A[1, idx0[n - 1]:])]
            A12[n - 1, :] = [min(A[3, idx0[n - 1]:]),
                             (max(A[3, idx0[n - 1]:]) - min(A[3, idx0[n - 1]:])) / (-min(A[2, idx0[n - 1]:])),
                             max(A[3, idx0[n - 1]:])]
            A21[n - 1, :] = [max(A[5, idx0[n - 1]:]),
                             (min(A[5, idx0[n - 1]:]) - max(A[5, idx0[n - 1]:])) / (-max(A[4, idx0[n - 1]:])),
                             min(A[5, idx0[n - 1]:])]
            A22[n - 1, :] = [min(A[7, idx0[n - 1]:]),
                            (max(A[7, idx0[n - 1]:]) - min(A[7, idx0[n - 1]:])) / (-min(A[6, idx0[n - 1]:])),
                            max(A[7, idx0[n - 1]:])]

        else:
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A11[n - 1, :] = [max(A[1, idx0[n - 1]:idx0[n]]),
                            (min(A[1, idx0[n - 1]:idx0[n]]) - max(A[1, idx0[n - 1]:idx0[n]])) / (-max(A[0, idx0[n - 1]:idx0[n]])),
                            min(A[1, idx0[n - 1]:idx0[n]])]
            A12[n - 1, :] = [min(A[3, idx0[n - 1]:idx0[n]]),
                             (max(A[3, idx0[n - 1]:idx0[n]]) - min(A[3, idx0[n - 1]:idx0[n]])) / (-min(A[2, idx0[n - 1]:idx0[n]])),
                             max(A[3, idx0[n - 1]:idx0[n]])]
            A21[n - 1, :] = [max(A[5, idx0[n - 1]:idx0[n]]),
                             (min(A[5, idx0[n - 1]:idx0[n]]) - max(A[5, idx0[n - 1]:idx0[n]])) / (-max(A[4, idx0[n - 1]:idx0[n]])),
                             min(A[5, idx0[n - 1]:idx0[n]])]
            A22[n - 1, :] = [min(A[7, idx0[n - 1]:idx0[n]]),
                            (max(A[7, idx0[n - 1]:idx0[n]]) - min(A[7, idx0[n - 1]:idx0[n]])) / (-min(A[6, idx0[n - 1]:idx0[n]])),
                            max(A[7, idx0[n - 1]:idx0[n]])]
    return X0, A11, A12, A21, A22

# when lambda 11 < 0 and lambda 22 > 0
def preprocess2(data):
    t = data['t']
    X = data['X']
    A = data['A']
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    A11 = np.zeros((len(idx0), 3))
    A12 = np.zeros((len(idx0), 3))
    A21 = np.zeros((len(idx0), 3))
    A22 = np.zeros((len(idx0), 3))
    for n in range(1, len(idx0)+1):
        if n == len(idx0):
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A11[n - 1, :] = [min(A[1, idx0[n - 1]:]),
                            (max(A[1, idx0[n - 1]:]) - min(A[1, idx0[n - 1]:])) / (-min(A[0, idx0[n - 1]:])),
                            max(A[1, idx0[n - 1]:])]
            A12[n - 1, :] = [max(A[3, idx0[n - 1]:]),
                             (min(A[3, idx0[n - 1]:]) - max(A[3, idx0[n - 1]:])) / (-max(A[2, idx0[n - 1]:])),
                             min(A[3, idx0[n - 1]:])]
            A21[n - 1, :] = [min(A[5, idx0[n - 1]:]),
                             (max(A[5, idx0[n - 1]:]) - min(A[5, idx0[n - 1]:])) / (-min(A[4, idx0[n - 1]:])),
                             max(A[5, idx0[n - 1]:])]
            A22[n - 1, :] = [max(A[7, idx0[n - 1]:]),
                            (min(A[7, idx0[n - 1]:]) - max(A[7, idx0[n - 1]:])) / (-max(A[6, idx0[n - 1]:])),
                            min(A[7, idx0[n - 1]:])]

        else:
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A11[n - 1, :] = [min(A[1, idx0[n - 1]:idx0[n]]),
                            (max(A[1, idx0[n - 1]:idx0[n]]) - min(A[1, idx0[n - 1]:idx0[n]])) / (-min(A[0, idx0[n - 1]:idx0[n]])),
                            max(A[1, idx0[n - 1]:idx0[n]])]
            A12[n - 1, :] = [max(A[3, idx0[n - 1]:idx0[n]]),
                             (min(A[3, idx0[n - 1]:idx0[n]]) - max(A[3, idx0[n - 1]:idx0[n]])) / (-max(A[2, idx0[n - 1]:idx0[n]])),
                             min(A[3, idx0[n - 1]:idx0[n]])]
            A21[n - 1, :] = [min(A[5, idx0[n - 1]:idx0[n]]),
                             (max(A[5, idx0[n - 1]:idx0[n]]) - min(A[5, idx0[n - 1]:idx0[n]])) / (-min(A[4, idx0[n - 1]:idx0[n]])),
                             max(A[5, idx0[n - 1]:idx0[n]])]
            A22[n - 1, :] = [max(A[7, idx0[n - 1]:idx0[n]]),
                            (min(A[7, idx0[n - 1]:idx0[n]]) - max(A[7, idx0[n - 1]:idx0[n]])) / (-max(A[6, idx0[n - 1]:idx0[n]])),
                            min(A[7, idx0[n - 1]:idx0[n]])]
    return X0, A11, A12, A21, A22

def postprocess(A, problem):
    A11 = A[:3, :]
    A12 = A[3:6, :]
    A21 = A[6:9, :]
    A22 = A[9:, :]

    time = np.linspace(0, problem.t1, num=1000)
    lambda11_1 = np.zeros((len(A11[0, :]), len(time)))
    lambda12_1 = np.zeros((len(A12[0, :]), len(time)))
    lambda21_1 = np.zeros((len(A21[0, :]), len(time)))
    lambda22_1 = np.zeros((len(A22[0, :]), len(time)))

    lambda11_2 = np.zeros((len(A11[0, :]), len(time)))
    lambda12_2 = np.zeros((len(A12[0, :]), len(time)))
    lambda21_2 = np.zeros((len(A21[0, :]), len(time)))
    lambda22_2 = np.zeros((len(A22[0, :]), len(time)))
    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
    for i in range(len(A11[0, :])):
        for n in range(len(time)):
            if time[n] < A11[1, i]:
                lambda11_1[i, n] = -(A11[2, i] - A11[0, i]) / A11[1, i]
                lambda11_2[i, n] = (A11[2, i] - A11[0, i]) / A11[1, i] * time[n] + A11[0, i]
            else:
                lambda11_1[i, n] = problem.alpha
                lambda11_2[i, n] = A11[2, i]
            if time[n] < A12[1, i]:
                lambda12_1[i, n] = -(A12[2, i] - A12[0, i]) / A12[1, i]
                lambda12_2[i, n] = (A12[2, i] - A12[0, i]) / A12[1, i] * time[n] + A12[0, i]
            else:
                lambda12_1[i, n] = 0
                lambda12_2[i, n] = A12[2, i]
            if time[n] < A21[1, i]:
                lambda21_1[i, n] = -(A21[2, i] - A21[0, i]) / A21[1, i]
                lambda21_2[i, n] = (A21[2, i] - A21[0, i]) / A21[1, i] * time[n] + A21[0, i]
            else:
                lambda21_1[i, n] = 0
                lambda21_2[i, n] = A21[2, i]
            if time[n] < A22[1, i]:
                lambda22_1[i, n] = -(A22[2, i] - A22[0, i]) / A22[1, i]
                lambda22_2[i, n] = (A22[2, i] - A22[0, i]) / A22[1, i] * time[n] + A22[0, i]
            else:
                lambda22_1[i, n] = problem.alpha
                lambda22_2[i, n] = A22[2, i]
    #     axs[0].plot(time, lambda1[i, :])
    #     axs[1].plot(time, lambda2[i, :])
    # axs[0].set_title('$\lambda_{11}(2)$')
    # axs[1].set_title('$\lambda_{22}(2)$')
    # plt.show()
    A_data = {'A1': lambda11_1,
              'A2': lambda11_2,
              'A3': lambda12_1,
              'A4': lambda12_2,
              'A5': lambda21_1,
              'A6': lambda21_2,
              'A7': lambda22_1,
              'A8': lambda22_2}

    return time, A_data

def discrete_data(time, problem, X0, A_data):
    U1 = A_data['A2']/2
    U2 = A_data['A8']/2
    t_step = time[1] - time[0]

    max_acc = 10
    min_acc = -5
    U1[np.where(U1 > max_acc)] = max_acc
    U1[np.where(U1 < min_acc)] = min_acc
    U2[np.where(U2 > max_acc)] = max_acc
    U2[np.where(U2 < min_acc)] = min_acc

    U1 = torch.tensor(U1, requires_grad=True, dtype=torch.float32)
    U2 = torch.tensor(U2, requires_grad=True, dtype=torch.float32)

    d1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    v1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    V1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    L1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    L1_tmp = np.zeros((len(U1[:, 0]), len(U1[0])))
    d2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    v2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    V2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    L2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    L2_tmp = np.zeros((len(U2[:, 0]), len(U2[0])))

    t = np.zeros((len(U1[:, 0]), len(U1[0])))

    for i in range(len(U1[:, 0])):
        for j in range(len(time)):
            if j == 0:
                d1[i][j] = X0.T[0, i]
                v1[i][j] = X0.T[1, i]
                d2[i][j] = X0.T[2, i]
                v2[i][j] = X0.T[3, i]
            else:
                v1[i][j] = v1[i][j - 1] + (time[j] - time[j - 1]) * (U1[i][j - 1] + U1[i][j]) / 2
                v2[i][j] = v2[i][j - 1] + (time[j] - time[j - 1]) * (U2[i][j - 1] + U2[i][j]) / 2
                d1[i][j] = d1[i][j - 1] + (time[j] - time[j - 1]) * (v1[i][j - 1] + v1[i][j]) / 2
                d2[i][j] = d2[i][j - 1] + (time[j] - time[j - 1]) * (v2[i][j - 1] + v2[i][j]) / 2

    for i in range(len(U1[:, 0])):
        for j in range(len(time)):
            x1 = torch.tensor(d1[i][j], requires_grad=True, dtype=torch.float32)
            x2 = torch.tensor(d2[i][j], requires_grad=True, dtype=torch.float32)
            x1_in = (x1 - problem.R1 / 2 + problem.theta1 * problem.W2 / 2) * 10
            x1_out = -(x1 - problem.R1 / 2 - problem.W2 / 2 - problem.L1) * 10
            x2_in = (x2 - problem.R2 / 2 + problem.theta2 * problem.W1 / 2) * 10
            x2_out = -(x2 - problem.R2 / 2 - problem.W1 / 2 - problem.L2) * 10
            L1_tmp[i][j] = (U1[i][j] ** 2 + problem.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                x2_in) * torch.sigmoid(x2_out)) * t_step
            L2_tmp[i][j] = (U2[i][j] ** 2 + problem.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                x2_in) * torch.sigmoid(x2_out)) * t_step
            t[i][j] = time[j]

    for i in range(len(U1[:, 0])):
        for j in range(len(time)):
            L1[i][j] = np.sum(L1_tmp[i][j:])
            L2[i][j] = np.sum(L2_tmp[i][j:])

    for i in range(len(U1[:, 0])):
        for j in range(len(time)):
            V1[i][j] = problem.alpha * d1[i][-1] - (v1[i][-1] - v1[i][0]) ** 2 - L1[i][j]
            V2[i][j] = problem.alpha * d2[i][-1] - (v2[i][-1] - v2[i][0]) ** 2 - L2[i][j]

    data = {'t': t.reshape(1, -1),
            'X': np.vstack((d1.reshape(1, -1),
                            v1.reshape(1, -1),
                            d2.reshape(1, -1),
                            v2.reshape(1, -1))),
            'A': np.vstack((A_data['A1'].reshape(1, -1),
                            A_data['A2'].reshape(1, -1),
                            A_data['A3'].reshape(1, -1),
                            A_data['A4'].reshape(1, -1),
                            A_data['A5'].reshape(1, -1),
                            A_data['A6'].reshape(1, -1),
                            A_data['A7'].reshape(1, -1),
                            A_data['A8'].reshape(1, -1))),
            'V': np.vstack((V1.reshape(1, -1),
                            V2.reshape(1, -1)))}

    return data

def input_data(X0_train, A11_train, A12_train, A21_train, A22_train, X0_val, A11_val, A12_val, A21_val, A22_val, X0_test, A11_test, A12_test, A21_test, A22_test):
    ub = np.max(X0_train.T, axis=1)
    lb = np.min(X0_train.T, axis=1)

    A11ub = np.max(A11_train.T, axis=1)
    A11lb = np.min(A11_train.T, axis=1)

    A12ub = np.max(A12_train.T, axis=1)
    A12lb = np.min(A12_train.T, axis=1)

    A21ub = np.max(A21_train.T, axis=1)
    A21lb = np.min(A21_train.T, axis=1)

    A22ub = np.max(A22_train.T, axis=1)
    A22lb = np.min(A22_train.T, axis=1)

    Aub = np.vstack((A11ub, A12ub, A21ub, A22ub)).reshape(1, -1)
    Alb = np.vstack((A11lb, A12lb, A21lb, A22lb)).reshape(1, -1)

    train_data = {'X0': X0_train.T,
                  'A': np.vstack((A11_train.T, A12_train.T, A21_train.T, A22_train.T))}

    val_data = {'X0': X0_val.T,
                'A': np.vstack((A11_val.T, A12_val.T, A21_val.T, A22_val.T))}

    test_data = {'X0': X0_test.T,
                 'A': np.vstack((A11_test.T, A12_test.T, A21_test.T, A22_test.T))}

    scaling = {'lb': lb.reshape(4, 1), 'ub': ub.reshape(4, 1),
               'A_lb': Alb.reshape(12, 1),
               'A_ub': Aub.reshape(12, 1)}

    return train_data, val_data, test_data, scaling

start_time = time.time()
X0_train, A11_train, A12_train, A21_train, A22_train = preprocess1(train_data1)
X0_val, A11_val, A12_val, A21_val, A22_val = preprocess1(val_data1)
X0_test, A11_test, A12_test, A21_test, A22_test = preprocess1(test_data1)

TrainData, ValData, TestData, Scaling = input_data(X0_train, A11_train, A12_train, A21_train, A22_train, X0_val, A11_val, A12_val, A21_val, A22_val,
                                         X0_test, A11_test, A12_test, A21_test, A22_test)
model = HJI_network(problem, Scaling, config, parameters=None)
A_train1, A_val1, A_test1 = model.train(TrainData, ValData, TestData, EPISODE=10000, LR=0.01)
train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

Time, A_data = postprocess(A_train1, problem)
train_data1 = discrete_data(Time, problem, X0_train, A_data)
Time, A_data = postprocess(A_val1, problem)
val_data1 = discrete_data(Time, problem, X0_val, A_data)
Time, A_data = postprocess(A_test1, problem)
test_data1 = discrete_data(Time, problem, X0_test, A_data)

start_time = time.time()
X0_train, A11_train, A12_train, A21_train, A22_train = preprocess2(train_data2)
X0_val, A11_val, A12_val, A21_val, A22_val = preprocess2(val_data2)
X0_test, A11_test, A12_test, A21_test, A22_test = preprocess2(test_data2)

TrainData, ValData, TestData, Scaling = input_data(X0_train, A11_train, A12_train, A21_train, A22_train, X0_val, A11_val, A12_val, A21_val, A22_val,
                                         X0_test, A11_test, A12_test, A21_test, A22_test)
model = HJI_network(problem, Scaling, config, parameters=None)
A_train2, A_val2, A_test2 = model.train(TrainData, ValData, TestData, EPISODE=10000, LR=0.01)
train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

Time, A_data = postprocess(A_train2, problem)
train_data2 = discrete_data(Time, problem, X0_train, A_data)
Time, A_data = postprocess(A_val2, problem)
val_data2 = discrete_data(Time, problem, X0_val, A_data)
Time, A_data = postprocess(A_test2, problem)
test_data2 = discrete_data(Time, problem, X0_test, A_data)


train_data = {'t': np.hstack((train_data1['t'], train_data2['t'])),
              'X': np.hstack((train_data1['X'], train_data2['X'])),
              'A': np.hstack((train_data1['A'], train_data2['A'])),
              'V': np.hstack((train_data1['V'], train_data2['V']))}

val_data = {'t': np.hstack((val_data1['t'], val_data2['t'])),
            'X': np.hstack((val_data1['X'], val_data2['X'])),
            'A': np.hstack((val_data1['A'], val_data2['A'])),
            'V': np.hstack((val_data1['V'], val_data2['V']))}

test_data = {'t': np.hstack((test_data1['t'], test_data2['t'])),
             'X': np.hstack((test_data1['X'], test_data2['X'])),
             'A': np.hstack((test_data1['A'], test_data2['A'])),
             'V': np.hstack((test_data1['V'], test_data2['V']))}


save_data = int_input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path_train = 'examples/' + system + '/data_discrete_' + 'train' + '.mat'
    save_path_val = 'examples/' + system + '/data_discrete_' + 'val' + '.mat'
    save_path_test = 'examples/' + system + '/data_discrete_' + 'test' + '.mat'

    U1, U2 = problem.U_star(np.vstack((train_data['X'], train_data['A'])))
    U = np.vstack((U1, U2))

    save_dict_train = {'lb_1': np.min(train_data['X'][:N_states], axis=1, keepdims=True),
                       'ub_1': np.max(train_data['X'][:N_states], axis=1, keepdims=True),
                       'lb_2': np.min(train_data['X'][N_states:2*N_states], axis=1, keepdims=True),
                       'ub_2': np.max(train_data['X'][N_states:2*N_states], axis=1, keepdims=True),
                       'A_lb_11': np.min(train_data['A'][:N_states], axis=1, keepdims=True),
                       'A_ub_11': np.max(train_data['A'][:N_states], axis=1, keepdims=True),
                       'A_lb_12': np.min(train_data['A'][N_states:2*N_states], axis=1, keepdims=True),
                       'A_ub_12': np.max(train_data['A'][N_states:2*N_states], axis=1, keepdims=True),
                       'A_lb_21': np.min(train_data['A'][2*N_states:3*N_states], axis=1, keepdims=True),
                       'A_ub_21': np.max(train_data['A'][2*N_states:3*N_states], axis=1, keepdims=True),
                       'A_lb_22': np.min(train_data['A'][3*N_states:4*N_states], axis=1, keepdims=True),
                       'A_ub_22': np.max(train_data['A'][3*N_states:4*N_states], axis=1, keepdims=True),
                       'U_lb_1': np.min(U1, axis=1, keepdims=True),
                       'U_ub_1': np.max(U1, axis=1, keepdims=True),
                       'U_lb_2': np.min(U2, axis=1, keepdims=True),
                       'U_ub_2': np.max(U2, axis=1, keepdims=True),
                       'V_min_1': np.min(train_data['V'][-2:-1,:]), 'V_max_1': np.max(train_data['V'][-2:-1,:]),
                       'V_min_2': np.min(train_data['V'][-1,:]), 'V_max_2': np.max(train_data['V'][-1,:]),
                       't': train_data['t'], 'X': train_data['X'], 'A': train_data['A'], 'V': train_data['V'],
                       'U': U}
    scipy.io.savemat(save_path_train, save_dict_train)

    U1, U2 = problem.U_star(np.vstack((val_data['X'], val_data['A'])))
    U = np.vstack((U1, U2))

    save_dict_val = {'lb_1': np.min(val_data['X'][:N_states], axis=1, keepdims=True),
                     'ub_1': np.max(val_data['X'][:N_states], axis=1, keepdims=True),
                     'lb_2': np.min(val_data['X'][N_states:2 * N_states], axis=1, keepdims=True),
                     'ub_2': np.max(val_data['X'][N_states:2 * N_states], axis=1, keepdims=True),
                     'A_lb_11': np.min(val_data['A'][:N_states], axis=1, keepdims=True),
                     'A_ub_11': np.max(val_data['A'][:N_states], axis=1, keepdims=True),
                     'A_lb_12': np.min(val_data['A'][N_states:2 * N_states], axis=1, keepdims=True),
                     'A_ub_12': np.max(val_data['A'][N_states:2 * N_states], axis=1, keepdims=True),
                     'A_lb_21': np.min(val_data['A'][2 * N_states:3 * N_states], axis=1, keepdims=True),
                     'A_ub_21': np.max(val_data['A'][2 * N_states:3 * N_states], axis=1, keepdims=True),
                     'A_lb_22': np.min(val_data['A'][3 * N_states:4 * N_states], axis=1, keepdims=True),
                     'A_ub_22': np.max(val_data['A'][3 * N_states:4 * N_states], axis=1, keepdims=True),
                     'U_lb_1': np.min(U1, axis=1, keepdims=True),
                     'U_ub_1': np.max(U1, axis=1, keepdims=True),
                     'U_lb_2': np.min(U2, axis=1, keepdims=True),
                     'U_ub_2': np.max(U2, axis=1, keepdims=True),
                     'V_min_1': np.min(val_data['V'][-2:-1, :]), 'V_max_1': np.max(val_data['V'][-2:-1, :]),
                     'V_min_2': np.min(val_data['V'][-1, :]), 'V_max_2': np.max(val_data['V'][-1, :]),
                     't': val_data['t'], 'X': val_data['X'], 'A': val_data['A'], 'V': val_data['V'],
                     'U': U}
    scipy.io.savemat(save_path_val, save_dict_val)

    U1, U2 = problem.U_star(np.vstack((test_data['X'], test_data['A'])))
    U = np.vstack((U1, U2))

    save_dict_test = {'lb_1': np.min(test_data['X'][:N_states], axis=1, keepdims=True),
                      'ub_1': np.max(test_data['X'][:N_states], axis=1, keepdims=True),
                      'lb_2': np.min(test_data['X'][N_states:2 * N_states], axis=1, keepdims=True),
                      'ub_2': np.max(test_data['X'][N_states:2 * N_states], axis=1, keepdims=True),
                      'A_lb_11': np.min(test_data['A'][:N_states], axis=1, keepdims=True),
                      'A_ub_11': np.max(test_data['A'][:N_states], axis=1, keepdims=True),
                      'A_lb_12': np.min(test_data['A'][N_states:2 * N_states], axis=1, keepdims=True),
                      'A_ub_12': np.max(test_data['A'][N_states:2 * N_states], axis=1, keepdims=True),
                      'A_lb_21': np.min(test_data['A'][2 * N_states:3 * N_states], axis=1, keepdims=True),
                      'A_ub_21': np.max(test_data['A'][2 * N_states:3 * N_states], axis=1, keepdims=True),
                      'A_lb_22': np.min(test_data['A'][3 * N_states:4 * N_states], axis=1, keepdims=True),
                      'A_ub_22': np.max(test_data['A'][3 * N_states:4 * N_states], axis=1, keepdims=True),
                      'U_lb_1': np.min(U1, axis=1, keepdims=True),
                      'U_ub_1': np.max(U1, axis=1, keepdims=True),
                      'U_lb_2': np.min(U2, axis=1, keepdims=True),
                      'U_ub_2': np.max(U2, axis=1, keepdims=True),
                      'V_min_1': np.min(test_data['V'][-2:-1, :]), 'V_max_1': np.max(test_data['V'][-2:-1, :]),
                      'V_min_2': np.min(test_data['V'][-1, :]), 'V_max_2': np.max(test_data['V'][-1, :]),
                      't': test_data['t'], 'X': test_data['X'], 'A': test_data['A'], 'V': test_data['V'],
                      'U': U}
    scipy.io.savemat(save_path_test, save_dict_test)


# fig, axs = plt.subplots(1, constrained_layout=True)
#
# for n in range(1, len(oidx0)+1):
#     if n == len(oidx0):
#         axs.plot(origX[0,oidx0[n - 1]:],origX[2,oidx0[n - 1]:], color='Blue', linewidth=3)
#     else:
#         axs.plot(origX[0,oidx0[n - 1]:oidx0[n]],origX[2,oidx0[n - 1]:oidx0[n]], color='Blue', linewidth=3)
#
# for i in range(len(U1[:,0])):
#     axs.plot(X1[i, :], X2[i, :], color='Red')
#     axs.plot()
#
# plt.show()
#
# intersection = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
# psuedoIntersection = patches.Rectangle((31.25, 31.25), 7.5, 7.5, linewidth=1, edgecolor='black', facecolor='white')
# axs.add_patch(psuedoIntersection)
# axs.add_patch(intersection)
# start = patches.Polygon(np.array([[15, 15], [20, 15], [20, 20]]), linewidth=1, edgecolor='black', facecolor='white')
# axs.add_patch(start)
# axs.set_xlim(15, 40)
# axs.set_xlabel('d1')
# axs.set_ylim(15, 40)
# axs.set_ylabel('d2')

# # ---------------------------------------------------------------------------- #
# # save_dict = {'train_time': train_time,
# #              'train_err': errors[0],
# #              'val_err': errors[1]
# #              }
#
# # scipy.io.savemat('examples/' + system + '/results/train_results.mat', save_dict)
#
# # Saves model parameters
# # save_me = int_input('Save model parameters? Enter 0 for no, 1 for yes:')
# #
# # save_data = int_input('Save data? Enter 0 for no, 1 for yes:')
# #
# # if save_me:
# #     # weights1, biases1, weights2, biases2 = model.export_model()
# #     # save_dict = scaling
# #     # save_dict.update({'weights1': weights1,
# #     #                   'biases1': biases1,
# #     #                   'weights2': weights2,
# #     #                   'biases2': biases2,
# #     #                   'train_time': train_time})
# #     weights, biases = model.export_model()
# #     save_dict = scaling
# #     save_dict.update({'weights': weights,
# #                       'biases': biases,
# #                       'train_time': train_time})
# #
# #     scipy.io.savemat(model_path, save_dict)
# #
# # if save_data:
# #     scipy.io.savemat(data_path, model_data)



# def discrete_data(time, problem, *args):
#     U1 = lambda1/2
#     U2 = lambda2/2
#     t_step = time[1] - time[0]
#
#     time = torch.tensor(time, requires_grad=True, dtype=torch.float32)
#
#     max_acc = 10
#     min_acc = -5
#     U1[np.where(U1 > max_acc)] = max_acc
#     U1[np.where(U1 < min_acc)] = min_acc
#     U2[np.where(U2 > max_acc)] = max_acc
#     U2[np.where(U2 < min_acc)] = min_acc
#
#     U1 = torch.tensor(U1, requires_grad=True, dtype=torch.float32)
#     U2 = torch.tensor(U2, requires_grad=True, dtype=torch.float32)
#
#     d1 = []
#     v1 = []
#     V1 = []
#     L1 = []
#     A1 = np.zeros((len(U1[:, 0]), len(U1[0])))
#     A2 = np.zeros((len(U1[:, 0]), len(U1[0])))
#     A3 = np.zeros((len(U1[:, 0]), len(U1[0])))
#     A4 = np.zeros((len(U1[:, 0]), len(U1[0])))
#     L1_tmp = []
#     d2 = []
#     v2 = []
#     V2 = []
#     L2 = []
#     L2_tmp = []
#     A5 = np.zeros((len(U2[:, 0]), len(U2[0])))
#     A6 = np.zeros((len(U2[:, 0]), len(U2[0])))
#     A7 = np.zeros((len(U2[:, 0]), len(U2[0])))
#     A8 = np.zeros((len(U2[:, 0]), len(U2[0])))
#
#     t = np.zeros((len(U1[:, 0]), len(U1[0])))
#
#     for i in range(len(U1[:, 0])):
#         d1.append([])
#         v1.append([])
#         d2.append([])
#         v2.append([])
#         for j in range(len(time)):
#             if j == 0:
#                 d1[i].append(torch.tensor(X0_train[j, 0], requires_grad=True, dtype=torch.float32))
#                 v1[i].append(torch.tensor(X0_train[j, 1], requires_grad=True, dtype=torch.float32))
#                 d2[i].append(torch.tensor(X0_train[j, 2], requires_grad=True, dtype=torch.float32))
#                 v2[i].append(torch.tensor(X0_train[j, 3], requires_grad=True, dtype=torch.float32))
#             else:
#                 v1[i].append((v1[i][j - 1] + (time[j] - time[j - 1]) * (U1[i][j - 1] + U1[i][j]) / 2).clone().detach().requires_grad_(True))
#                 v2[i].append((v2[i][j - 1] + (time[j] - time[j - 1]) * (U2[i][j - 1] + U2[i][j]) / 2).clone().detach().requires_grad_(True))
#                 d1[i].append((d1[i][j - 1] + (time[j] - time[j - 1]) * (v1[i][j - 1] + v1[i][j]) / 2).clone().detach().requires_grad_(True))
#                 d2[i].append((d2[i][j - 1] + (time[j] - time[j - 1]) * (v2[i][j - 1] + v2[i][j]) / 2).clone().detach().requires_grad_(True))
#
#     for i in range(len(U1[:, 0])):
#         L1_tmp.append([])
#         L2_tmp.append([])
#         for j in range(len(time)):
#             x1_in = (d1[i][j] - problem.R1 / 2 + problem.theta1 * problem.W2 / 2) * 10
#             x1_out = -(d1[i][j] - problem.R1 / 2 - problem.W2 / 2 - problem.L1) * 10
#             x2_in = (d2[i][j] - problem.R2 / 2 + problem.theta2 * problem.W1 / 2) * 10
#             x2_out = -(d2[i][j] - problem.R2 / 2 - problem.W1 / 2 - problem.L2) * 10
#             L1_tmp[i].append((U1[i][j] ** 2 + problem.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
#                 x2_in) * torch.sigmoid(x2_out)) * t_step)
#             L2_tmp[i].append((U2[i][j] ** 2 + problem.beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
#                 x2_in) * torch.sigmoid(x2_out)) * t_step)
#             t[i][j] = time[j]
#
#     # xx = 2 * d1[-1,-1]
#     x1_out = -(d1[-1][:] - problem.R1 / 2 - problem.W2 / 2 - problem.L1) * 10
#     x1_sum = torch.sum(x1_out)
#     x1_sum.requires_grad_()
#     dL1dx1 = torch.autograd.grad(x1_sum, d1[-1][:], create_graph=True)[0].detach().numpy()
#
#     for i in range(len(U1[:, 0])):
#         L1_sum = torch.sum(sum(L1_tmp[i][:]))
#         dL1dx1 = torch.autograd.grad(L1_tmp[i][1], d1[i][1], retain_graph=True, create_graph=True)[0].detach().numpy()
#         print('')
#
#     result = len(L1_tmp[1])
#
#     for i in range(len(U1[:, 0])):
#         L1.append([])
#         L2.append([])
#         for j in range(len(time)):
#             L1[i].append(sum(L1_tmp[i][j:]))
#             L2[i].append(sum(L2_tmp[i][j:]))
#
#     for i in range(len(U1[:, 0])):
#         V1.append([])
#         V2.append([])
#         for j in range(len(time)):
#             V1[i].append(problem.alpha * d1[i][-1] - (v1[i][-1] - v1[i][0]) ** 2 - L1[i][j])
#             V2[i].append(problem.alpha * d2[i][-1] - (v2[i][-1] - v2[i][0]) ** 2 - L2[i][j])
#
#     for i in range(len(U1[:, 0])):
#         result = sum(V1[i][:])
#         V_sum_1 = torch.sum(sum(V1[i][:]))
#         V_sum_1.requires_grad_()
#         V_sum_2 = torch.sum(sum(V2[i][:]))
#         V_sum_2.requires_grad_()
#         for j in range(len(time)):
#             result = torch.autograd.grad(V_sum_1, d1[i][j], create_graph=True)[0].detach().numpy()
#             A1[i][j] = torch.autograd.grad(V_sum_1, d1[i][j], create_graph=True)[0].detach().numpy()
#             A2[i][j] = torch.autograd.grad(V_sum_1, v1[i][j], create_graph=True)[0].detach().numpy()
#             A3[i][j] = torch.autograd.grad(V_sum_1, d2[i][j], create_graph=True)[0].detach().numpy()
#             A4[i][j] = torch.autograd.grad(V_sum_1, v2[i][j], create_graph=True)[0].detach().numpy()
#             A5[i][j] = torch.autograd.grad(V_sum_2, d1[i][j], create_graph=True)[0].detach().numpy()
#             A6[i][j] = torch.autograd.grad(V_sum_2, v1[i][j], create_graph=True)[0].detach().numpy()
#             A7[i][j] = torch.autograd.grad(V_sum_2, d2[i][j], create_graph=True)[0].detach().numpy()
#             A8[i][j] = torch.autograd.grad(V_sum_2, v2[i][j], create_graph=True)[0].detach().numpy()
#
#     data = {'t': t.reshape(1, -1),
#             'X': np.vstack((d1.detach().numpy(),
#                             v1.detach().numpy(),
#                             d2.detach().numpy(),
#                             v2.detach().numpy())),
#             'A': np.vstack((A1.detach().numpy(),
#                             A2.detach().numpy(),
#                             A3.detach().numpy(),
#                             A4.detach().numpy(),
#                             A5.detach().numpy(),
#                             A6.detach().numpy(),
#                             A7.detach().numpy(),
#                             A8.detach().numpy())),
#             'v': np.vstack((V1.detach().numpy(),
#                             V2.detach().numpy()))}
#
#     return data
