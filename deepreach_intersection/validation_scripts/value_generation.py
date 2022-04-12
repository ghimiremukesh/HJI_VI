# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

def value_function(X, t, model):
    d1 = 2.0 * (X[0, :] - 15) / (60 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (60 - 15) - 1.
    v2 = 2.0 * (X[3, :] - 15) / (32 - 15) - 1.
    X = np.vstack((d1, v1, d2, v2))

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X), dim=1)
    coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.cuda()}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']
    cut_index = x.shape[1] // 2
    y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = model_output['model_out'][:, cut_index:]  # agent 2's value

    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1, consider V = exp(u)
    lam11_1 = dvdx_1[:, :1] / ((60 - 15) / 2) / y1   # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2) / y1   # lambda_11

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam22_1 = dvdx_2[:, :1] / ((60 - 15) / 2) / y2   # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2) / y2   # lambda_22

    # y1 = torch.log(y1)
    # y2 = torch.log(y2)

    return y1, y2, lam11_1, lam11_2, lam22_1, lam22_2

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = './model_final_10k_pretrain_relu.pth'
    activation = 'relu'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=5)
    model.cuda()
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    test_data = scio.loadmat('data_test_a_a_18_18_cut.mat')

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]
    print(len(idx0))

    N = 76
    Time = np.linspace(0, 1.5, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    V1 = np.zeros((len(idx0), N))
    V2 = np.zeros((len(idx0), N))
    A11 = np.zeros((len(idx0), N))
    A12 = np.zeros((len(idx0), N))
    A21 = np.zeros((len(idx0), N))
    A22 = np.zeros((len(idx0), N))

    start_time = time.time()

    # value and costate generation
    for i in range(1, len(idx0) + 1):
        if i == len(idx0):
            for j in range(1, N + 1):
                d1 = X[0, idx0[i - 1]:][j - 1]
                v1 = X[1, idx0[i - 1]:][j - 1]
                d2 = X[2, idx0[i - 1]:][j - 1]
                v2 = X[3, idx0[i - 1]:][j - 1]
                X_nn = np.vstack((d1, v1, d2, v2))
                t_nn = np.array([[Time[j - 1]]])
                y1, y2, lam11, lam12, lam21, lam22 = value_function(X_nn, t_nn, model)
                V1[i - 1][j - 1] = y1
                V2[i - 1][j - 1] = y2
                A11[i - 1][j - 1] = lam11
                A12[i - 1][j - 1] = lam12
                A21[i - 1][j - 1] = lam21
                A22[i - 1][j - 1] = lam22

        else:
            for j in range(1, N + 1):
                d1 = X[0, idx0[i - 1]: idx0[i]][j - 1]
                v1 = X[1, idx0[i - 1]: idx0[i]][j - 1]
                d2 = X[2, idx0[i - 1]: idx0[i]][j - 1]
                v2 = X[3, idx0[i - 1]: idx0[i]][j - 1]
                X_nn = np.vstack((d1, v1, d2, v2))
                t_nn = np.array([[Time[j - 1]]])
                y1, y2, lam11, lam12, lam21, lam22 = value_function(X_nn, t_nn, model)
                V1[i - 1][j - 1] = y1
                V2[i - 1][j - 1] = y2
                A11[i - 1][j - 1] = lam11
                A12[i - 1][j - 1] = lam12
                A21[i - 1][j - 1] = lam21
                A22[i - 1][j - 1] = lam22

        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    V1 = V1.flatten()
    V2 = V2.flatten()
    A11 = A11.flatten()
    A12 = A12.flatten()
    A21 = A21.flatten()
    A22 = A22.flatten()

    X_OUT = X
    V_OUT = np.vstack((V1, V2))
    time = np.array([np.linspace(0, 3, num=100)])
    t_OUT = t

    data = {'X': X_OUT,
            't': t_OUT,
            'V': V_OUT,
            'A1': np.vstack((A11, A12)),
            'A2': np.vstack((A21, A22))}

    save_data = input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'value_generation.mat'
        scio.savemat(save_path, data)

