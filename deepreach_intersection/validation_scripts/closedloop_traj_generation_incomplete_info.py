# Enable import from parent package
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio


def value_action(X, t, model_a, model_na, model_type, true_param):  # should return actions and hamiltonians of both
    # agents for both aggressive and non-aggressive

    theta1, theta2 = true_param
    # normalize the state for agent 1, agent 2
    d1 = 2.0 * (X[0, :] - 15) / (60 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (60 - 15) - 1.
    v2 = 2.0 * (X[3, :] - 15) / (32 - 15) - 1.
    p1 = X[4, :]
    p2 = X[5, :]
    X = np.vstack((d1, v1, d2, v2, p1, p2))

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X), dim=1)
    coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.cuda()}
    model_output_a = model_a(model_in)
    model_output_na = model_na(model_in)

    x_a = model_output_a['model_in']
    x_na = model_output_na['model_in']
    y_a = model_output_a['model_out']
    y_na = model_output_na['model_out']
    cut_index = x_a.shape[1] // 2
    y1_a = model_output_a['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y1_na = model_output_na['model_out'][:, :cut_index]
    y2_a = model_output_a['model_out'][:, cut_index:]  # agent 2's value
    y2_na = model_output_na['model_out'][:, cut_index:]

    jac_a, _ = diff_operators.jacobian(y_a, x_a)
    jac_na, _ = diff_operators.jacobian(y_na, x_na)
    dv_1_a = jac_a[:, :cut_index, :]
    dv_1_na = jac_na[:, :cut_index, :]
    dv_2_a = jac_a[:, cut_index:, :]
    dv_2_na = jac_na[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdx_1_a = dv_1_a[..., 0, 1:].squeeze().reshape(1, dv_1_a.shape[-1] - 1)
    dvdx_1_na = dv_1_na[..., 0, 1:].squeeze().reshape(1, dv_1_na.shape[-1] - 1)

    # unnormalize the costate for agent 1
    lam11_1_a = dvdx_1_a[:, :1] / ((60 - 15) / 2)  # lambda_11  # agg
    lam11_2_a = dvdx_1_a[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1_a = dvdx_1_a[:, 2:3] / ((60 - 15) / 2)  # lambda_12
    lam12_2_a = dvdx_1_a[:, 3:4] / ((32 - 15) / 2)  # lambda_12

    lam11_1_na = dvdx_1_na[:, :1] / ((60 - 15) / 2)  # lambda_11  # n-agg
    lam11_2_na = dvdx_1_na[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1_na = dvdx_1_na[:, 2:3] / ((60 - 15) / 2)  # lambda_12
    lam12_2_na = dvdx_1_na[:, 3:4] / ((32 - 15) / 2)  # lambda_12
    # agent 2: partial gradient of V w.r.t. state
    dvdx_2_a = dv_2_a[..., 0, 1:].squeeze().reshape(1, dv_2_a.shape[-1] - 1)
    dvdx_2_na = dv_2_na[..., 0, 1:].squeeze().reshape(1, dv_2_na.shape[-1] - 1)

    # unnormalize the costate for agent 2,
    lam22_1_a = dvdx_2_a[:, :1] / ((60 - 15) / 2)  # lambda_22
    lam22_2_a = dvdx_2_a[:, 1:2] / ((32 - 15) / 2)  # lambda_22
    lam21_1_a = dvdx_2_a[:, 2:3] / ((60 - 15) / 2)  # lambda_21
    lam21_2_a = dvdx_2_a[:, 3:4] / ((32 - 15) / 2)  # lambda_21

    lam22_1_na = dvdx_2_na[:, :1] / ((60 - 15) / 2)  # lambda_22
    lam22_2_na = dvdx_2_na[:, 1:2] / ((32 - 15) / 2)  # lambda_22
    lam21_1_na = dvdx_2_na[:, 2:3] / ((60 - 15) / 2)  # lambda_21
    lam21_2_na = dvdx_2_na[:, 3:4] / ((32 - 15) / 2)  # lambda_21

    max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
    min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

    # Constants for Hamiltonian
    R1 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 1
    R2 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 2
    W1 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    W2 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    L1 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    L2 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    theta_a = torch.tensor([1.], dtype=torch.float32).cuda()  # behavior for agent 1
    theta_na = torch.tensor([5.], dtype=torch.float32).cuda()  # behavior for agent 2
    beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

    # calculate Hamiltonian for agent 1
    x1_in = ((d1 - R1 / 2 + theta_a * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
    x2_in = ((d2 - R2 / 2 + theta_a * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

    sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
    sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
    loss_instant = beta * sigmoid1 * sigmoid2

    # calculate the collision area lower and upper bounds (non-aggressive)
    x1_in_na = ((d1 - R1 / 2 + theta_na * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x2_in_na = ((d2 - R2 / 2 + theta_na * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()

    sigmoid1_na = torch.sigmoid(x1_in_na) * torch.sigmoid(x1_out)
    sigmoid2_na = torch.sigmoid(x2_in_na) * torch.sigmoid(x2_out)
    loss_instant_na = beta * sigmoid1_na * sigmoid2_na

    # Actions based on co-state
    u1_a = 0.5 * lam11_2_a
    u1_na = 0.5 * lam11_2_na
    u2_a = 0.5 * lam22_2_a
    u2_na = 0.5 * lam22_2_na


    # calculate ham for agents for belief dynamics:
    # H = \lambda^T * (-f) + L

    H_1_a = -lam11_1_a.squeeze() * v1.squeeze() - lam11_2_a.squeeze() * u1_a.squeeze() - \
            lam12_1_a.squeeze() * v2.squeeze() - lam12_2_a.squeeze() * u2_a.squeeze() + \
            (u1_a ** 2 + loss_instant.cuda())
    H_1_na = -lam11_1_na.squeeze() * v1.squeeze() - lam11_2_na.squeeze() * u1_na.squeeze() - \
             lam12_1_na.squeeze() * v2.squeeze() - lam12_2_na.squeeze() * u2_na.squeeze() + \
             (u1_na ** 2 + loss_instant.cuda())

    H_2_a = -lam21_1_a.squeeze() * v1.squeeze() - lam21_2_a.squeeze() * u1_a.squeeze() - \
            lam22_1_a.squeeze() * v2.squeeze() - lam22_2_a.squeeze() * u2_a.squeeze() + \
            (u2_a ** 2 + loss_instant.cuda())
    H_2_na = -lam21_1_na.squeeze() * v1.squeeze() - lam21_2_na.squeeze() * u1_na.squeeze() - \
             lam22_1_na.squeeze() * v2.squeeze() - lam22_2_na.squeeze() * u2_na.squeeze() + \
             (u2_na ** 2 + loss_instant_na.cuda())


    if model_type == 'HJI':
        # action for agent 1
        if theta1 == 'a':
            U1 = u1_a
        else:
            U1 = u1_na

        # action for agent 2
        if theta2 == 'a':
            U2 = u2_a
        else:
            U2 = u2_na

        U1[torch.where(U1 > max_acc)] = max_acc
        U1[torch.where(U1 < min_acc)] = min_acc
        U2[torch.where(U2 > max_acc)] = max_acc
        U2[torch.where(U2 < min_acc)] = min_acc

    return U1, U2, H_1_a, H_1_na, H_2_a, H_2_na


def dynamic(X_nn, dt, action, hams):
    alpha = 0.01
    u1, u2 = action
    h1_a, h1_na, h2_a, h2_na = hams
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + ((X_nn[1, :] + v1) / 2) * dt
    d2 = X_nn[2, :] + ((X_nn[3, :] + v2) / 2) * dt
    p1 = X_nn[4, :] + torch.sign(h2_a - h2_na) * alpha * dt
    p2 = X_nn[5, :] + torch.sign(h1_a - h1_na) * alpha * dt
    return d1, v1, d2, v2, p1, p2


if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path_a = '../experiment_scripts/logs/incomplete_info_try/a/checkpoints/model_final.pth'
    ckpt_path_na = '../experiment_scripts/logs/incomplete_info_try/na/checkpoints/model_final.pth'
    activation = 'tanh'

    # model_type = 'BRAT'
    model_type = 'HJI'

    # Initialize and load the model # two models
    model_a = modules.SingleBVPNet(in_features=7, out_features=1, type=activation, mode='mlp',
                                   final_layer_factor=1., hidden_features=64, num_hidden_layers=3)

    model_na = modules.SingleBVPNet(in_features=7, out_features=1, type=activation, mode='mlp',
                                    final_layer_factor=1., hidden_features=64, num_hidden_layers=3)

    model_a.cuda()
    model_na.cuda()

    checkpoint_a = torch.load(ckpt_path_a)
    checkpoint_na = torch.load(ckpt_path_na)

    try:
        model_weights_a = checkpoint_a['model']
        model_weights_na = checkpoint_na['model']
    except:
        model_weights_a = checkpoint_a
        model_weights_na = checkpoint_na

    model_a.load_state_dict(model_weights_a)
    model_na.load_state_dict(model_weights_na)
    model_a.eval()
    model_na.eval()

    # if no ground truth --> generate test data
    num_points = 300
    p_state_dim = 4
    b_state_dim = 2

    test_data = torch.zeros((num_points, p_state_dim)).uniform_(-1, 1)
    b_states = torch.zeros((num_points, b_state_dim)).uniform_(0, 1)
    ini_time = torch.zeros((num_points, 1))
    X = torch.cat((ini_time, test_data, b_states), dim=1)
    X0 = X.T

    Time = np.linspace(0, 1.5, num=76)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = np.zeros((num_points, Time.shape[0]))
    v1 = np.zeros((num_points, Time.shape[0]))
    u1 = np.zeros((num_points, Time.shape[0]))
    p1 = np.zeros((num_points, Time.shape[0]))
    d2 = np.zeros((num_points, Time.shape[0]))
    v2 = np.zeros((num_points, Time.shape[0]))
    u2 = np.zeros((num_points, Time.shape[0]))
    p2 = np.zeros((num_points, Time.shape[0]))
    h1_a = np.zeros((num_points, Time.shape[0]))
    h1_na = np.zeros((num_points, Time.shape[0]))
    h2_a = np.zeros((num_points, Time.shape[0]))
    h2_na = np.zeros((num_points, Time.shape[0]))

    for n in range(num_points):
        d1[n][0] = X0[1, n]
        v1[n][0] = X0[2, n]
        p1[n][0] = X0[5, n]
        d2[n][0] = X0[3, n]
        v2[n][0] = X0[4, n]
        p2[n][0] = X0[6, n]

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        last_action = (u1[i][0], u2[i][0])

        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array(
                [[d1[i][j - 1]], [v1[i][j - 1]], [d2[i][j - 1]], [v2[i][j - 1]], [p1[i][j - 1]], [p2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1], h1_a[i][j-1], h1_na[i][j-1], h2_a[i][j-1], h2_na[i][j-1] = \
                value_action(X_nn, t_nn, model_a, model_na, model_type)
            if j == Time.shape[0]:
                break
            else:
                d1[i][j], v1[i][j], d2[i][j], v2[i][j], p1[i][j], p2[i][j] = dynamic(X_nn, dt,
                                                                                     (u1[i][j - 1], u2[i][j - 1]),
                                                                                     (h1_a[i][j-1], h1_na[i][j-1],
                                                                                      h2_a[i][j-1], h2_na[i][j-1]))
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

    X_OUT = np.vstack((d1, v1, d2, v2, p1, p2))
    U_OUT = np.vstack((u1, u2))
    time = np.array([np.linspace(0, 1.5, num=76)])
    t_OUT = np.empty((1, 0))

    for _ in range(num_points):
        t_OUT = np.hstack((t_OUT, time))

    data = {'X': X_OUT,
            't': t_OUT,
            'U': U_OUT}

    save_data = input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'closedloop_traj_incomplete_info.mat'
        scio.savemat(save_path, data)
