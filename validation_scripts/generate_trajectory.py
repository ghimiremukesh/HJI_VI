# Enable import from parent package
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, utils, training, loss_functions, modules, diff_operators

import torch
import numpy as np
import math
import scipy.io as spio
from scipy.integrate import odeint
from matplotlib.collections import LineCollection
# logging_root = '/logs'
alpha_angle = 1.2 * math.pi

# Setting to plot
# ckpt_path = 'air3D_ckpt.pth'
ckpt_path = '/home/exx/PycharmProjects/deepreach/validation_scripts/air3D_ckpt.pth'
activation = 'sine'

# Initialize and load the model
model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

def dynamics(w, t, a, b):
    v_a = 0.75
    v_b = 0.75
    x, y, psi = w.tolist()

    return -v_a + v_b*(np.cos(psi * alpha_angle))+a*y, v_b*np.sin(psi * alpha_angle)-a*x, b-a

def dynamics_discrete(w, a, b):
    v_a = 0.75
    v_b = 0.75
    x, y, psi = w.tolist()

    return np.array([-v_a + v_b*(np.cos(psi * alpha_angle))+a*y,
                     v_b*np.sin(psi * alpha_angle)-a*x,
                     b-a])

#initial state
x0 = torch.tensor([0.5, -0.25, .4208], requires_grad=True)  # -0.25,0.25,.4208(final state)/(-0.1767, -0.3221, 0.4208)
v = 0.75
N_num = 20
t = torch.linspace(0, 1.0, N_num+1).flip(0)
t_ode = torch.linspace(0, 1.0, N_num+1)
# t = torch.linspace(0, 1.0, 21)
x = x0
xs = []
us = []
ds = []
values = []
hji_list = []
xs.append(x0.detach().cpu().numpy())
for i in range(N_num):
    x = torch.cat((torch.tensor(t[i], requires_grad=True).reshape(1), x)).reshape(1, 4).unsqueeze(0)
    #feed the model
    model_in = {'coords': x.cuda()}
    model_out = model(model_in)

    #get value from the model
    x = model_out['model_in']
    y = model_out['model_out']
    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0, 0]
    dudx = du[..., 0, 1:]
    x_theta = x[..., 3] * 1.0

    # Scale the costate for theta appropriately to align with the range of [-pi, pi]
    dudx[..., 2] = dudx[..., 2] / alpha_angle
    # Scale the coordinates
    x_theta = alpha_angle * x_theta

    #unnormalize since we normalized during training
    norm_to = 0.02
    mean = 0.25
    var = 0.5

    value = (y * var / norm_to) + mean

    values.append(value.detach().cpu().numpy().squeeze())

    omega = torch.linspace(-3, 3, 7)

    H = torch.zeros((len(omega), len(omega)))

    for d in range(len(omega)):
        for u in range(len(omega)):
            ham = omega[u] * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
            ham = ham - omega[d] * torch.abs(dudx[..., 2])  # Disturbance component
            ham = ham + (v * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (v * torch.sin(x_theta) * dudx[..., 1])  # Constant component
            #set the hamiltonian for given u and d
            H[d][u] = ham
    # find where is the max min H
    idx = torch.argmin(H)
    row = idx//len(omega)

    # u corresponds to max(u) min(d) H
    col = torch.argmax(H[row, :])

    hji = (dudt + H[row][col]).detach().cpu().numpy().flatten()
    hji_list.append(min(hji, np.array(np.linalg.norm(x[..., 1:3].detach().cpu().numpy())) - 0.25 - value.detach().cpu().numpy().flatten()))

    u = omega[col]
    d = omega[row]

    # get new state
    # x_new = odeint(dynamics, x[..., 1:].detach().cpu().numpy().squeeze(),
    #                t=np.arange(t_ode[i].detach().cpu().numpy(), t_ode[i+1].detach().numpy(), 0.001),
    #                args=(u.detach().cpu().numpy(), d.detach().cpu().numpy()))
    # x = x_new[-1]

    x_cur = x[..., 1:].detach().cpu().numpy().squeeze()
    x = x_cur + dynamics_discrete(x[..., 1:].detach().cpu().numpy().squeeze(),
                                  u.detach().cpu().numpy(), d.detach().cpu().numpy()) * 0.05
    xs.append(x)
    us.append(u.detach().cpu().numpy().tolist())
    ds.append(d.detach().cpu().numpy().tolist())
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)

x = torch.cat((torch.tensor(t[-1], requires_grad=True).reshape(1), x)).reshape(1, 4).unsqueeze(0)
print('action u', us)
print('action d', ds)
#feed the model
model_in = {'coords': x.cuda()}
model_out = model(model_in)

#get value from the model
x = model_out['model_in']
y = model_out['model_out']
# unnormalize since we normalized during training
norm_to = 0.02
mean = 0.25
var = 0.5
value = (y * var / norm_to) + mean
values.append(value.detach().cpu().numpy().squeeze())

du, status = diff_operators.jacobian(y, x)
dudt = du[..., 0, 0]
dudx = du[..., 0, 1:]
x_theta = x[..., 3] * 1.0

# Scale the costate for theta appropriately to align with the range of [-pi, pi]
dudx[..., 2] = dudx[..., 2] / alpha_angle
# Scale the coordinates
x_theta = alpha_angle * x_theta

omega = torch.linspace(-3, 3, 7)
H = torch.zeros((len(omega), len(omega)))

for d in range(len(omega)):
    for u in range(len(omega)):
        ham = omega[u] * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega[d] * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (v * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (v * torch.sin(x_theta) * dudx[..., 1])  # Constant component
        #set the hamiltonian for given u and d
        H[d][u] = ham
# find where is the max min H
idx = torch.argmin(H)
row = idx//len(omega)

# u corresponds to max(u) min(d) H
col = torch.argmax(H[row, :])

hji = (dudt + H[row][col]).detach().cpu().numpy().flatten()
hji_list.append(min(hji, np.array(np.linalg.norm(x[..., 1:3].detach().cpu().numpy())) - 0.25 - value.detach().cpu().numpy().flatten()))

x1 = list()
x2 = list()
x3 = list()

for i in range(len(xs) - 1):
   x1.append(xs[i][0])
   x2.append(xs[i][1])
   x3.append(xs[i][2] * alpha_angle)

# import scipy.io
# save_path = 'ground_truth_data_discrete_test_20.mat'
# save_dict = {'x1': x1,
#              'x2': x2,
#              'x3': x3,
#              'u': us,
#              'd': ds}
#
# scipy.io.savemat(save_path, save_dict)

print(x1)
print(x2)
print(x3)
print(values)

x1 = np.array(x1)
x2 = np.array(x2)
values = np.array(values)

angle_alpha = 1.2

# Setting to plot
times = [0.9]
time_indices_matlab = [int(time_to_plot / 0.1) + 1 for time_to_plot in times]
thetas = [1.5863]  # This theta is contained in the LS computation grid.

# Initialize and load the model
model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Load the ground truth BRS data
# true_BRT_path = 'analytical_BRT_air3D.mat'
check_BRT_path = '/home/exx/PycharmProjects/deepreach/validation_scripts/analytical_BRT_air3D_check.mat'

# test_data = spio.loadmat(check_BRT_path)
true_data = spio.loadmat(check_BRT_path)

# Save the value function arrays
val_functions = {}
val_functions['LS'] = []
val_functions['siren'] = []

num_times = len(times)
num_thetas = len(thetas)

# Find matlab indices for theta slices
theta_indices_matlab = []
theta_values = true_data['gmat'][0, 0, :, 2]
for i in range(num_thetas):
    theta_indices_matlab.append(np.argmin(abs(theta_values - thetas[i])))


def val_fn_BRS(model):
    num_times = len(times)
    num_thetas = len(thetas)

    # Find matlab indices for theta slices
    theta_indices_matlab = []
    theta_values = true_data['gmat'][0, 0, :, 2]
    for i in range(num_thetas):
        theta_indices_matlab.append(np.argmin(abs(theta_values - thetas[i])))

    # Create figures
    fig_brs = plt.figure(figsize=(5 * num_thetas, 5 * num_times))
    fig_valfunc_LS = plt.figure(figsize=(5 * num_thetas, 5 * num_times))
    fig_valfunc_siren = plt.figure(figsize=(5 * num_thetas, 5 * num_times))

    # Start plotting the results
    for i in range(num_times):
        for j in range(num_thetas):
            state_coords = torch.tensor(np.reshape(true_data['gmat'][:, :, theta_indices_matlab[j], :], (-1, 3)),
                                        dtype=torch.float32)
            state_coords[:, 2] = state_coords[:, 2] / (angle_alpha * math.pi)
            time_coords = torch.ones(state_coords.shape[0], 1) * times[i]
            coords = torch.cat((time_coords, state_coords), dim=1)[None]

            # Compute the value function
            model_in = {'coords': coords.cuda()}
            model_out = model(model_in)

            # Detatch outputs and reshape
            valfunc = model_out['model_out'].detach().cpu().numpy()
            valfunc_true = true_data['data'][:, :, theta_indices_matlab[j], time_indices_matlab[i]]
            valfunc = np.reshape(valfunc, valfunc_true.shape)

            # Unnormalize the value function and gradients
            norm_to = 0.02
            mean = 0.25
            var = 0.5
            valfunc = (valfunc * var / norm_to) + mean

            ## Plot the zero level set
            # Fetch the BRS
            brs_predicted = (valfunc <= 0.001) * 1.
            brs_actual = (valfunc_true <= 0.001) * 1.
            # Plot it
            ax = fig_brs.add_subplot(num_times, num_thetas, (j + 1) + i * num_thetas)
            ax.set_title('t = %0.1f, theta = %0.2f' % (times[i], thetas[j]))
            s1 = ax.imshow(brs_predicted.T, cmap='viridis', origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.),
                           interpolation='bilinear')
            s2 = ax.imshow(brs_actual.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1.,
                           extent=(-1., 1., -1., 1.), interpolation='bilinear')

            ax.plot(x1[0], x2[0], marker='*', color='black')
            # ax.plot(x1, x2, color='b')
            points = np.array([x1, x2]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-0.25, 1.2)
            line_segments = LineCollection(segments, linewidths=2, cmap='viridis', norm=norm)
            line_segments.set_array(values)
            valuebar = ax.add_collection(line_segments)
            fig_brs.colorbar(valuebar)

            ## Plot the predicted value function
            # ax = fig_valfunc_siren.add_subplot(num_times, num_thetas, (j + 1) + i * num_thetas)
            # ax.set_title('t = %0.1f, theta = %0.2f' % (1.1, thetas[j]))  # times[i]
            # s = ax.imshow(valfunc.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
            # fig_valfunc_siren.colorbar(s)

            ## Append the value functions
            val_functions['LS'].append(valfunc_true)
            val_functions['siren'].append(valfunc)

    # #sm = plt.cm.ScalarMappable(cmap=values, norm=plt.Normalize(-0.25, 1.25))
    # norm = plt.Normalize(-0.25, 1.2)
    # cmap = plt.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
    # cmap.set_array([])
    # ax.plot(x1[0], x2[0], marker='*', color='black')
    # ax.plot(x1, x2, c=cmap.to_rgba(values[0]))
    # fig_brs.colorbar(cmap)
    # plt.colorbar(sm)
    # ax.plot(x1, x2, color='b', c=values, cmap='seismic')
    # norm = plt.Normalize(-0.25, 1.2)
    # for i in range(len(values)):
    #     points = np.array([x1[i], x2[i]]).T.reshape(-1, 1, 2)
    #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #
    #     # Normalize/Plot Value 1 ColorBar
    #     lc = LineCollection(segments, cmap='PuOr', norm=norm)
    #     lc.set_array(values[i])
    #     lc.set_linewidth(2)
    #     line = ax.add_collection(lc)

    # fig_brs.colorbar(line, ax=ax)
    # ax.colorbar(line, ax=ax)

    return fig_brs, fig_valfunc_LS, fig_valfunc_siren, val_functions


# Run the validation of sets
fig_brs, fig_valfunc_LS, fig_valfunc_siren, val_functions = val_fn_BRS(model)

fig_brs.savefig('Air3D_BRS_comparison5.png')
fig_valfunc_siren.savefig('Air3D_Siren_valfunc.png')

matplotlib.use('TkAgg')
font = {'family': 'normal', 'weight': 'normal', 'size': 30}
plt.rc('font', **font)
step = np.linspace(0, 1.0, 21)
benchline = np.zeros(len(step))
plt.plot(step, hji_list, label='HJI VI')
plt.plot(step, benchline, label='HJI VI benchmark')
plt.xlabel('time step')
plt.ylabel('HJI VI value')

plt.legend(loc='lower right')
plt.show()




