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
    d1 = 2.0 * (X[0, :] - 15) / (100 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (100 - 15) - 1.
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
    y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdt_1 = (dv_1[..., 0, 0] / y1).squeeze()
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1
    lam11_1 = dvdx_1[:, :1] / ((100 - 15) / 2) / y1  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2) / y1  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((100 - 15) / 2) / y1  # lambda_12
    lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2) / y1  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdt_2 = (dv_2[..., 0, 0] / y2).squeeze()
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2
    lam21_1 = dvdx_2[:, 2:3] / ((100 - 15) / 2) / y2  # lambda_21
    lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2) / y2  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((100 - 15) / 2) / y2  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2) / y2  # lambda_22

    # calculate the collision area for aggressive-aggressive case
    R1 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 1
    R2 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 2
    W1 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    W2 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    L1 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    L2 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    theta1 = torch.tensor([1.], dtype=torch.float32).cuda()  # behavior for agent 1
    theta2 = torch.tensor([1.], dtype=torch.float32).cuda()  # behavior for agent 2
    beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

    # H = lambda^T * f - L
    # Agent 1's action
    u1 = (0.5 * dvdx_1[:, 1:2] / ((32 - 15) / 2) / y1).detach()

    # Agent 2's action
    u2 = (0.5 * dvdx_2[:, 1:2] / ((32 - 15) / 2) / y2).detach()

    # set up bounds for u1 and u2
    max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
    min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

    u1[torch.where(u1 > max_acc)] = max_acc
    u1[torch.where(u1 < min_acc)] = min_acc
    u2[torch.where(u2 > max_acc)] = max_acc
    u2[torch.where(u2 < min_acc)] = min_acc

    # detach and let the action as the number, not trainable variable
    u1.requires_grad = False
    u2.requires_grad = False

    # unnormalize the state for agent 1
    d1 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (100 - 15) / 2 + 15
    v1 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

    # unnormalize the state for agent 2
    d2 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (100 - 15) / 2 + 15
    v2 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

    # calculate the collision area lower and upper bounds
    x1_in = ((d1 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
    x2_in = ((d2 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

    sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
    sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
    loss_instant = beta * sigmoid1 * sigmoid2

    # calculate instantaneous loss
    loss_fun_1 = (u1 ** 2 + loss_instant).cuda()
    loss_fun_2 = (u2 ** 2 + loss_instant).cuda()

    # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
    ham_1 = lam11_1.squeeze() * v1.squeeze() + lam11_2.squeeze() * u1.squeeze() + \
            lam12_1.squeeze() * v2.squeeze() + lam12_2.squeeze() * u2.squeeze() - loss_fun_1.squeeze()
    ham_2 = lam21_1.squeeze() * v1.squeeze() + lam21_2.squeeze() * u1.squeeze() + \
            lam22_1.squeeze() * v2.squeeze() + lam22_2.squeeze() * u2.squeeze() - loss_fun_2.squeeze()

    hji1 = -dvdt_1 + ham_1
    hji2 = -dvdt_2 + ham_2

    return hji1, hji2, loss_fun_1.squeeze(), loss_fun_2.squeeze(), ham_1, ham_2

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    ckpt_path = './model_final_HJI_sine_trainable.pth'
    activation = 'sine'

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

    test_data = scio.loadmat('data_test_a_a_18_18.mat')

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]
    print(len(idx0))

    A = test_data['A']
    A1 = A[:, :151]
    t1 = t[:, :151]
    X1 = X[:, :151]
    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = X[0, :151][55]
    v1 = X[1, :151][55]
    d2 = X[2, :151][55]
    v2 = X[3, :151][55]
    X_nn = np.vstack((d1, v1, d2, v2))
    t_nn = np.array([[1.9]])

    y1, y2, loss1, loss2, ham1, ham2 = value_function(X_nn, t_nn, model)

    HJI1 = np.zeros((len(idx0), N))
    HJI2 = np.zeros((len(idx0), N))
    Loss1 = np.zeros((len(idx0), N))
    Loss2 = np.zeros((len(idx0), N))
    H1 = np.zeros((len(idx0), N))
    H2 = np.zeros((len(idx0), N))

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
                y1, y2, loss1, loss2, ham1, ham2 = value_function(X_nn, t_nn, model)
                HJI1[i - 1][j - 1] = y1
                HJI2[i - 1][j - 1] = y2
                Loss1[i - 1][j - 1] = loss1
                Loss2[i - 1][j - 1] = loss2
                H1[i - 1][j - 1] = ham1
                H2[i - 1][j - 1] = ham2

        else:
            for j in range(1, N + 1):
                d1 = X[0, idx0[i - 1]: idx0[i]][j - 1]
                v1 = X[1, idx0[i - 1]: idx0[i]][j - 1]
                d2 = X[2, idx0[i - 1]: idx0[i]][j - 1]
                v2 = X[3, idx0[i - 1]: idx0[i]][j - 1]
                X_nn = np.vstack((d1, v1, d2, v2))
                t_nn = np.array([[Time[j - 1]]])
                y1, y2, loss1, loss2, ham1, ham2 = value_function(X_nn, t_nn, model)
                HJI1[i - 1][j - 1] = y1
                HJI2[i - 1][j - 1] = y2
                Loss1[i - 1][j - 1] = loss1
                Loss2[i - 1][j - 1] = loss2
                H1[i - 1][j - 1] = ham1
                H2[i - 1][j - 1] = ham2

        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    HJI1 = HJI1.flatten()
    HJI2 = HJI2.flatten()
    Loss1 = Loss1.flatten()
    Loss2 = Loss2.flatten()
    H1 = H1.flatten()
    H2 = H2.flatten()

    X_OUT = X
    HJI_OUT = np.vstack((HJI1, HJI2))
    Loss_OUT = np.vstack((Loss1, Loss2))
    H_OUT = np.vstack((H1, H2))
    t_OUT = t

    data = {'X': X_OUT,
            't': t_OUT,
            'HJI': HJI_OUT,
            'Loss': Loss_OUT,
            'H': H_OUT}

    save_data = input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'HJI_generation_HJI.mat'
        scio.savemat(save_path, data)

