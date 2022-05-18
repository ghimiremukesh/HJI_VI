# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

def value_action(X, t, model, model_type):
    # normalize the state for agent 1, agent 2
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
    lam11_1 = dvdx_1[:, :1] / ((60 - 15) / 2) / y1    # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2) / y1   # lambda_11

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam22_1 = dvdx_2[:, :1] / ((60 - 15) / 2) / y2    # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2) / y2   # lambda_22

    max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
    min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

    if model_type == 'HJI':
        # action for agent 1
        U1 = 0.5 * lam11_2

        # action for agent 2
        U2 = 0.5 * lam22_2

        U1[torch.where(U1 > max_acc)] = max_acc
        U1[torch.where(U1 < min_acc)] = min_acc
        U2[torch.where(U2 > max_acc)] = max_acc
        U2[torch.where(U2 < min_acc)] = min_acc

    if model_type == 'BRAT':
        # action for agent 1
        U1 = lam11_2

        # action for agent 2
        U2 = lam22_2

        U1[torch.where(U1 > 0)] = max_acc
        U1[torch.where(U1 < 0)] = min_acc
        U2[torch.where(U2 > 0)] = max_acc
        U2[torch.where(U2 < 0)] = min_acc

    return U1, U2

def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + ((X_nn[1, :] + v1) / 2) * dt
    d2 = X_nn[2, :] + ((X_nn[3, :] + v2) / 2) * dt

    return d1, v1, d2, v2

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = './model_final.pth'
    activation = 'relu'

    model_type = 'BRAT'
    # model_type = 'HJI'

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
    X0 = np.zeros((len(idx0), 4))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    Time = np.linspace(0, 1.5, num=76)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = np.zeros((len(idx0), Time.shape[0]))
    v1 = np.zeros((len(idx0), Time.shape[0]))
    u1 = np.zeros((len(idx0), Time.shape[0]))
    d2 = np.zeros((len(idx0), Time.shape[0]))
    v2 = np.zeros((len(idx0), Time.shape[0]))
    u2 = np.zeros((len(idx0), Time.shape[0]))

    for n in range(len(idx0)):
        d1[n][0] = X0[0, n]
        v1[n][0] = X0[1, n]
        d2[n][0] = X0[2, n]
        v2[n][0] = X0[3, n]

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        last_action = (u1[i][0], u2[i][0])

        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[d1[i][j - 1]], [v1[i][j - 1]], [d2[i][j - 1]], [v2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1] = value_action(X_nn, t_nn, model, model_type)
            if j == Time.shape[0]:
                break
            else:
                d1[i][j], v1[i][j], d2[i][j], v2[i][j] = dynamic(X_nn, dt, (u1[i][j - 1], u2[i][j - 1]))
                last_action = (u1[i][j - 1], u2[i][j - 1])

        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    d1 = d1.flatten()
    v1 = v1.flatten()
    d2 = d2.flatten()
    v2 = v2.flatten()
    u1 = u1.flatten()
    u2 = u2.flatten()

    X_OUT = np.vstack((d1, v1, d2, v2))
    U_OUT = np.vstack((u1, u2))
    time = np.array([np.linspace(0, 1.5, num=76)])
    t_OUT = np.empty((1, 0))

    for _ in range(len(idx0)):
        t_OUT = np.hstack((t_OUT, time))

    data = {'X': X_OUT,
            't': t_OUT,
            'U': U_OUT}

    save_data = input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'closedloop_traj.mat'
        scio.savemat(save_path, data)

