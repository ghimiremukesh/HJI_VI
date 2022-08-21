# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def value_action(X, t, model, model_type):
    # normalize the state for agent 1, agent 2
    d1 = 2.0 * (X[0, :] - 15) / (105 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (105 - 15) - 1.
    v2 = 2.0 * (X[3, :] - 15) / (32 - 15) - 1.
    X = np.vstack((d1, v1, d2, v2))

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X), dim=1)
    coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.to(device)}
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
    lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
    lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

    max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

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

    return U1, U2, y1, y2

def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[2, :] + v2 * dt

    return d1, v1, d2, v2

if __name__ == '__main__':

    logging_root = './logs'
    N_neurons = 64

    policy = ['a_a', 'a_na', 'na_a', 'na_na']
    N_choice = 3

    # Setting to plot
    # ckpt_path = './model_supervised_' + str(N_neurons) + '.pth'
    # ckpt_path = './model_final_' + str(N_neurons) + '.pth'
    ckpt_path = '../experiment_scripts/logs/experiment_grow_gamma_tanh_try_gamma_5.0/checkpoints/model_final.pth'
    # ckpt_path = './model_supervised_' + str(policy[N_choice]) + '.pth'
    activation = 'tanh'

    # model_type = 'BRAT'
    model_type = 'HJI'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
    model.to(device)
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    test_data = scio.loadmat('data_test_a_a_18_18_cut.mat')
    # path = 'data_test_' + str(policy[N_choice]) + '_no collision_inter.mat'
    # test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]

    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = np.zeros((len(idx0), Time.shape[0]))
    v1 = np.zeros((len(idx0), Time.shape[0]))
    u1 = np.zeros((len(idx0), Time.shape[0]))
    d2 = np.zeros((len(idx0), Time.shape[0]))
    v2 = np.zeros((len(idx0), Time.shape[0]))
    u2 = np.zeros((len(idx0), Time.shape[0]))
    V1 = np.zeros((len(idx0), Time.shape[0]))
    V2 = np.zeros((len(idx0), Time.shape[0]))

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
            u1[i][j - 1], u2[i][j - 1], V1[i][j-1], V2[i][j-1] = value_action(X_nn, t_nn, model, model_type)
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
    V1 = V1.flatten()
    V2 = V2.flatten()

    X_OUT = np.vstack((d1, v1, d2, v2))
    U_OUT = np.vstack((u1, u2))
    t_OUT = t
    V_OUT = np.vstack((V1, V2))

    data = {'X': X_OUT,
            't': t_OUT,
            'U': U_OUT,
            'V': V_OUT}

    save_data = input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        # save_path = 'closedloop_traj_hybrid.mat'
        save_path = 'grow_gamma_traj.mat'
        # save_path = 'closedloop_traj_supervised_' + str(policy[N_choice]) + '.mat'
        scio.savemat(save_path, data)
