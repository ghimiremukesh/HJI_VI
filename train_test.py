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
from utilities.neural_networks_test import HJB_network


c = np.random.seed(config.random_seeds['train'])
# torch.manual_seed(config.random_seeds['train'])
# tf.set_random_seed(config.random_seeds['train'])

# ---------------------------------------------------------------------------- #

##### Loads data sets #####

# train_data = scipy.io.loadmat('examples/' + system + '/data_train_a_a_1.mat')
# val_data = scipy.io.loadmat('examples/' + system + '/data_val_a_a_1.mat')
# test_data = scipy.io.loadmat('examples/' + system + '/data_test_a_a_1.mat')

train_data = scipy.io.loadmat('examples/' + system + '/data_train_na_na_1.mat')
val_data = scipy.io.loadmat('examples/' + system + '/data_val_na_na_1.mat')
test_data = scipy.io.loadmat('examples/' + system + '/data_test_na_na_1.mat')

# if time_dependent:
#     system += '/tspan'
# else:
#     system += '/t0'
#     for data in [train_data, val_data, test_data]:
#         idx0 = np.nonzero(np.equal(data.pop('t'), 0.))[1]
#         data.update({'X': data['X'][:, idx0],
#                      'A': data['A'][:, idx0],
#                      'V': data['V'][:, idx0]})

N_train = train_data['X'].shape[1]
N_val = val_data['X'].shape[1]
N_test = test_data['X'].shape[1]

origA = test_data['A']
origX = test_data['X']
origt = test_data['t']

def originalTestTrajIDX0(data):
    t = data['t']
    X = data['X']
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]

    return idx0

oidx0 = originalTestTrajIDX0(test_data)

fig, axs = plt.subplots(1, 2, constrained_layout=True)

for n in range(1, len(oidx0)+1):
    if n == len(oidx0):
        axs[0].plot(origt[0,oidx0[n - 1]:], origA[1,oidx0[n - 1]:])
        axs[1].plot(origt[0,oidx0[n - 1]:], origA[7, oidx0[n - 1]:])
    else:
        axs[0].plot(origt[0,oidx0[n - 1]:oidx0[n]], origA[1,oidx0[n - 1]:oidx0[n]])
        axs[1].plot(origt[0,oidx0[n - 1]:oidx0[n]], origA[7, oidx0[n - 1]:oidx0[n]])
axs[0].set_title('$\lambda_{11}(2)$')
axs[1].set_title('$\lambda_{22}(2)$')
plt.show()

def preprocess(data):
    t = data['t']
    X = data['X']
    A = data['A']
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    A1 = np.zeros((len(idx0), 3))
    A2 = np.zeros((len(idx0), 3))
    for n in range(1, len(idx0)+1):
        if n == len(idx0):
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A1[n - 1, :] = [max(A[1, idx0[n - 1]:]),
                            (min(A[1, idx0[n - 1]:]) - max(A[1, idx0[n - 1]:])) / (-max(A[0, idx0[n - 1]:])),
                            min(A[1, idx0[n - 1]:])]
            A2[n - 1, :] = [min(A[7, idx0[n - 1]:]),
                            (max(A[7, idx0[n - 1]:]) - min(A[7, idx0[n - 1]:])) / (-min(A[6, idx0[n - 1]:])),
                            max(A[7, idx0[n - 1]:])]

        else:
            X0[n - 1, :] = X[:, idx0[n - 1]]
            A1[n - 1, :] = [max(A[1, idx0[n - 1]:idx0[n]]),
                            (min(A[1, idx0[n - 1]:idx0[n]]) - max(A[1, idx0[n - 1]:idx0[n]])) / (
                                -max(A[0, idx0[n - 1]:idx0[n]])), min(A[1, idx0[n - 1]:idx0[n]])]
            A2[n - 1, :] = [min(A[7, idx0[n - 1]:idx0[n]]),
                            (max(A[7, idx0[n - 1]:idx0[n]]) - min(A[7, idx0[n - 1]:idx0[n]])) / (
                                -min(A[6, idx0[n - 1]:idx0[n]])), max(A[7, idx0[n - 1]:idx0[n]])]
    return X0, A1, A2

def postprocess(A1, A2):
    time = np.linspace(0, 3, num = 100)
    lambda1 = np.zeros((len(A1[:,0]), len(time)))
    lambda2 = np.zeros((len(A2[:,0]), len(time)))
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    for i in range(len(A1[:,0])):
        for n in range(len(time)):
            if time[n] < A1[i, 1]:
                lambda1[i, n] = (A1[i,2]-A1[i,0])/A1[i,1]*time[n] + A1[i,0]
            else:
                lambda1[i, n] = A1[i,2]
            if time[n] < A2[i, 1]:
                lambda2[i, n] = (A2[i,2]-A2[i,0])/A2[i,1]*time[n] + A2[i,0]
            else:
                lambda2[i, n] = A2[i,2]
        axs[0].plot(time, lambda1[i, :])
        axs[1].plot(time, lambda2[i, :])
    axs[0].set_title('$\lambda_{11}(2)$')
    axs[1].set_title('$\lambda_{22}(2)$')
    plt.show()
    return time, lambda1, lambda2

X0_train, A1_train, A2_train = preprocess(train_data)
X0_val, A1_val, A2_val = preprocess(val_data)
X0_test, A1_test, A2_test = preprocess(test_data)

postprocess(A1_test, A2_test)

ub = np.max(X0_train.T, axis=1)
lb = np.min(X0_train.T, axis=1)

A1ub = np.max(A1_train.T, axis=1)
A1lb = np.min(A1_train.T, axis=1)

A2ub = np.max(A2_train.T, axis=1)
A2lb = np.min(A2_train.T, axis=1)

Aub = np.vstack((A1ub, A2ub)).reshape(1, -1)
Alb = np.vstack((A1lb, A2lb)).reshape(1, -1)

train_data = {
    'X0': X0_train.T,
    'A': np.vstack((A1_train.T, A2_train.T))
}

val_data = {
    'X0': X0_val.T,
    'A': np.vstack((A1_val.T, A2_val.T))
}

test_data = {
    'X0': X0_test.T,
    'A': np.vstack((A1_test.T, A2_test.T))
}

parameters = None
scaling = {
        'lb': lb.reshape(4, 1), 'ub': ub.reshape(4, 1),
        'A_lb': Alb.reshape(6, 1),
        'A_ub': Aub.reshape(6, 1),
}

start_time = time.time()

model = HJB_network(problem, scaling, config, parameters)

# use validation data to train the model and use train_data to verify
A_test = model.train(train_data, val_data, test_data, EPISODE=1000, LR=0.01)
train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

time, lambda1, lambda2 = postprocess(A_test.T[:, :3], A_test.T[:, 3:])

U1 = lambda1/2
U2 = lambda2/2

max_acc = 10
min_acc = -5
U1[np.where(U1 > max_acc)] = max_acc
U1[np.where(U1 < min_acc)] = min_acc
U2[np.where(U2 > max_acc)] = max_acc
U2[np.where(U2 < min_acc)] = min_acc

X1 = np.zeros((len(U1[:,0]),len(U1[0])))
V1 = np.zeros((len(U1[:,0]),len(U1[0])))
X2 = np.zeros((len(U2[:,0]),len(U2[0])))
V2 = np.zeros((len(U2[:,0]),len(U2[0])))

# X1[:,0] = X0_test[:,0]
# V1[:,0] = X0_test[:,1]
# X2[:,0] = X0_test[:,2]
# V2[:,0] = X0_test[:,3]

X1[:,0] = X0_train[:,0]
V1[:,0] = X0_train[:,1]
X2[:,0] = X0_train[:,2]
V2[:,0] = X0_train[:,3]


for i in range(len(U1[:,0])):
    for j in range(len(time)-1):
        V1[i][j + 1] = V1[i][j] + (-time[j] + time[j + 1]) * (U1[i][j] + U1[i][j + 1]) / 2
        V2[i][j + 1] = V2[i][j] + (-time[j] + time[j + 1]) * (U2[i][j] + U2[i][j + 1]) / 2
        X1[i][j + 1] = X1[i][j] + (-time[j] + time[j + 1]) * (V1[i][j] + V1[i][j + 1]) / 2
        X2[i][j + 1] = X2[i][j] + (-time[j] + time[j + 1]) * (V2[i][j] + V2[i][j + 1]) / 2

fig, axs = plt.subplots(1, constrained_layout=True)

for n in range(1, len(oidx0)+1):
    if n == len(oidx0):
        axs.plot(origX[0,oidx0[n - 1]:],origX[2,oidx0[n - 1]:], color='Blue', linewidth=3)
    else:
        axs.plot(origX[0,oidx0[n - 1]:oidx0[n]],origX[2,oidx0[n - 1]:oidx0[n]], color='Blue', linewidth=3)

for i in range(len(U1[:,0])):
    axs.plot(X1[i, :], X2[i, :], color='Red')
    axs.plot()

plt.show()

intersection = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
psuedoIntersection = patches.Rectangle((31.25, 31.25), 7.5, 7.5, linewidth=1, edgecolor='black', facecolor='white')
axs.add_patch(psuedoIntersection)
axs.add_patch(intersection)
start = patches.Polygon(np.array([[15, 15], [20, 15], [20, 20]]), linewidth=1, edgecolor='black', facecolor='white')
axs.add_patch(start)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')

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
