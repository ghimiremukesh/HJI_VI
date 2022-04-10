import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io as scio
import modules
import time
import torch

def value_function(X, t, model):
    # normalize the state for agent 1, agent 2
    d1 = 2.0 * (X[0, :] - 15) / (60 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 14) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (60 - 15) - 1.
    v2 = 2.0 * (X[1, :] - 14) / (32 - 15) - 1.
    X = np.vstack((d1, v1, d2, v2))

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X), dim=1)
    coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.cuda()}
    model_output = model(model_in)

    x = model_output['model_in']
    cut_index = x.shape[1] // 2
    y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = model_output['model_out'][:, cut_index:]  # agent 2's value

    y1 = torch.log(y1)
    y2 = torch.log(y2)

    return y1, y2

ckpt_path = './model_final.pth'
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

N = 101

x1_axis = np.zeros(N)
x2_axis = np.zeros(N)

for i in range(N):
    x1_axis[i] = 15 + 0.25 * i
    x2_axis[i] = 15 + 0.25 * i

x1, x2 = np.meshgrid(x1_axis, x2_axis)

V1 = np.zeros((N, N))
V2 = np.zeros((N, N))

start_time = time.time()
count = 0

for i in range(N):
    for j in range(N):
        d1 = x1[i][j]
        v1 = 18.
        d2 = x2[i][j]
        v2 = 18.
        X_nn = np.vstack((d1, v1, d2, v2))
        t_nn = np.array([[1.5]])  # initial time
        y1, y2 = value_function(X_nn, t_nn, model)
        V1[i][j] = y1
        V2[i][j] = y2
        count += 1
        print(count)

print()
time_spend = time.time() - start_time
print('Total solution time: %1.1f' % (time_spend), 'sec')
print()

data = {'x1': x1,
        'x2': x2,
        'V1': V1,
        'V2': V2}

save_data = input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path = 'valuelandscape.mat'
    scio.savemat(save_path, data)
