import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


def value_action(X_nn, t_nn, model):
    d1 = X_nn[0]
    v1 = X_nn [1]
    d2 = X_nn[2]
    v2 = X_nn[3]
    p = X_nn[4]

    uMax = 0.5
    dMax = 0.3


    X = np.vstack((d1, v1, d2, v2, p))
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t_nn, dtype=torch.float32, requires_grad=True)
    coords = torch.cat((t, X), dim=1)

    model_in = {'coords': coords.cuda()}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']

    jac, _ = diff_operators.jacobian(y, x)

    # partial gradient of V w.r.t. time and state
    dvdt = jac[..., 0, 0]
    dvdx = jac[..., 0, 1:]

    # unnormalize the costate for agent 1
    lam_1 = dvdx[:, :,  :1]
    lam_2 = dvdx[:, :, 1:2]
    lam_4 = dvdx[:, :, 2:3]
    lam_5 = dvdx[:, :, 3:4]
    lam_6 = dvdx[:, :, 4:]


    u = uMax  * -1 * torch.sign(lam_2)
    d = dMax * torch.sign(lam_5)

    return u, d, y



def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[2, :] + v2 * dt
    p = np.clip(X_nn[4, :] + np.sign(u1) * dt, 0, 1)

    return d1, v1, d2, v2, p


if __name__ == '__main__':
    logging_root = './logs'
    ckpt_path = '../experiment_scripts/logs/soccer_hji_exp_increased_a/checkpoints/model_final.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=6, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=32, num_hidden_layers=3)
    model.cuda()
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    num_physical = 4
    x0 = torch.zeros(1, num_physical).uniform_(-1, 1)
    x0[:, 0] = 0 # put them in the center
    x0[:, 2] = 0
    x0[:, 1] = 0
    x0[:, 3] = 0

    p = torch.zeros(1, 1).uniform_(0, 1)
    p = torch.Tensor([[1]]) # force type
    X0 = torch.cat((x0, p), dim=1)

    N = 151*3
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]


    d1 = np.zeros((N,))
    v1 = np.zeros((N,))
    u1 = np.zeros((N,))
    d2 = np.zeros((N,))
    v2 = np.zeros((N,))
    u2 = np.zeros((N,))
    p = np.zeros((N,))
    V = np.zeros((N,))


    theta = -1 # type right

    d1[0] = X0[:, 0]
    v1[0] = X0[:, 1]
    d2[0] = X0[:, 2]
    v2[0] = X0[:, 3]
    p[0] = X0[:, 4]

    start_time = time.time()

    for j in range(1, Time.shape[0] + 1):
        X_nn = np.array([[d1[j - 1]], [v1[j - 1]], [d2[j - 1]], [v2[j - 1]], [p[j-1]]])
        t_nn = np.array([[Time[j - 1]]])
        u1[j - 1], u2[j - 1], V[j - 1] = value_action(X_nn, t_nn, model)
        if j == Time.shape[0]:
            break
        else:
            d1[j], v1[j], d2[j], v2[j], p[j] = dynamic(X_nn, dt, (u1[j - 1], u2[j - 1]))
            last_action = (u1[j - 1], u2[j - 1])

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    fig, ax = plt.subplots(nrows=5, ncols=1)
    ax[0].plot(Time, d1)
    ax[0].set_ylabel('Player 1 (attacker)')
    ax[1].plot(Time, d2)
    ax[1].set_ylabel('Player 2 (defender)')
    ax[2].plot(Time, p)
    ax[2].set_ylabel('Belief over attacker\'s type')
    # plt.plot(p)
    ax[3].plot(Time, u1)
    ax[4].plot(Time, u2)
    ax[4].set_xlabel('Time')

    fig2, ax2= plt.subplots(1, 1)
    ax2.plot(d1,d2)
    plt.show()