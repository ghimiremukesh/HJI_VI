import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

if __name__ == '__main__':

    logging_root = './logs'

    # Setting to plot
    # ckpt_path = '../experiment_scripts/logs/experiment_check_toy_model_SSL/checkpoints/model_final.pth'
    ckpt_path = '../experiment_scripts/logs/experiment_check_toy_model_hy_1/checkpoints/model_final.pth'
    supervised = '../experiment_scripts/logs/experiment_check_toy_model_sup/checkpoints/model_final.pth'
    ssl = '../experiment_scripts/logs/experiment_check_toy_model_ssl_alpha_1/checkpoints/model_final.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=4, num_hidden_layers=1)

    model_sup = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=4, num_hidden_layers=1)

    model_ssl = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=4, num_hidden_layers=1)
    model_sup.cuda()
    model_ssl.cuda()
    model.cuda()
    checkpoint = torch.load(ckpt_path)
    chk_sup = torch.load(supervised)
    chk_ssl = torch.load(ssl)
    try:
        model_weights = checkpoint['model']
        model_w_sup = chk_sup['model']
        model_w_ssl = chk_ssl['model']
    except:
        model_weights = checkpoint
        model_w_sup = chk_sup
        model_w_ssl = chk_ssl

    model.load_state_dict(model_weights)
    model_sup.load_state_dict(model_w_sup)
    model_ssl.load_state_dict(model_w_ssl)
    model_sup.eval()
    model.eval()
    model_ssl.eval()

    test_coords = torch.linspace(-1, 1, 1000).reshape((-1, 1))
    model_in = {'coords': test_coords.cuda()}

    y = model(model_in)
    y_sup = model_sup(model_in)
    y_ssl = model_ssl(model_in)

    value_sup = y_sup['model_out']
    value_ssl = y_ssl['model_out']
    # scio.savemat({'X': test_coords,
    #               'V': y}, 'traj_1d.mat')
    value = y['model_out']

    v = np.zeros((1000, 1))
    x = np.zeros((1000, 1))

    v_sup = np.zeros((1000, 1))
    x_sup = np.zeros((1000, 1))

    v_ssl = np.zeros((1000, 1))
    x_ssl = np.zeros((1000, 1))

    v_sup = value_sup.detach().cpu()
    x_sup = test_coords.detach().cpu()

    v = value.detach().cpu()
    x = test_coords.detach().cpu()

    v_ssl = value_ssl.detach().cpu()
    x_ssl = test_coords.detach().cpu()

    import math

    font = {'family': 'Times New Roman', 'weight': 'heavy', 'size': 20}
    plt.rc('font', **font)


    alpha = 1
    N = 500
    C = -np.arctan(1 / alpha) * (1 / math.pi)
    X = np.linspace(-1, 1, num=N)
    V = np.zeros((1, N)).flatten()

    for i in range(len(X)):
        V[i] = np.arctan(X[i] / alpha) * (1 / math.pi) + C
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 8))
    axs2.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
    axs2.plot(x_ssl, v_ssl, label='SSL Value')
    axs2.plot(X, V, label='True Value')
    axs2.legend()
    # axs2.set_xlabel('X')
    # axs2.set_ylabel('Value')

    axs.plot(X, V,  c='black', label='True Value', linewidth=5)
    axs.plot(x, v, c='orange', label='Hybrid Value', linewidth=5)
    axs.plot(x_sup, v_sup, c='purple', label='Supervised Value', linewidth=5)
    axs.plot(x_ssl, v_ssl, c='grey', label='SSL Value', linewidth=5)
    # axs.set_xlabel('X', fontweight='bold')
    # axs.set_ylabel('Value', fontweight='bold')
    legend_prop = {'weight': 'bold'}
    # axs.legend(loc='best', fontsize='small', prop=legend_prop)
    fig.tight_layout()
    fig2.tight_layout()
    plt.xticks(X, weight='heavy')
    # fig.savefig('toy_case.png', dpi=600)
    # fig2.savefig('hybrid_value.png', dpi=600)
    plt.show()

    # ssl = {'value': v_ssl,
    #                     'x': x_ssl}
    # sl = {'value': v_sup,
    #                    'x': x_sup}
    # hy = {'value': v,
    #                    'x': x}
    #
    # scio.savemat('ssl.mat', ssl)
    # scio.savemat('sl.mat', sl)
    # scio.savemat('hy.mat', hy)

    # plt.savefig('toy_case.png', dpi=600)