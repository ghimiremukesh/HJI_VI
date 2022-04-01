"""
Compare the values between Level Set and PMP
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


true_data = scio.loadmat('analytical_BRT_air3D.mat')
val_funcs = scio.loadmat('Air3D_valfuncs_pmp.mat')

# create figures
times = [0.9, 0.5]
time_indices_matlab = [int(time_to_plot/0.1) + 1 for time_to_plot in times]
thetas = [1.5863]  # This theta is contained in the LS computation grid.

theta_indices_matlab = []
theta_values = true_data['gmat'][0, 0, :, 2]
num_thetas = len(thetas)
num_times = len(times)

for i in range(len(thetas)):
    theta_indices_matlab.append(np.argmin(abs(theta_values - thetas[i])))

# fig_brs = plt.figure(figsize=(5*num_thetas, 5*num_times))
fig_valfunc_LS = plt.figure(figsize=(5*num_thetas, 5*num_times))
fig_valfunc_pmp = plt.figure(figsize=(5*num_thetas, 5*num_times))

for i in range(len(times)):
    for j in range(len(thetas)):
        valfunc_true = true_data['data'][:, :, theta_indices_matlab[j], time_indices_matlab[i]] # LS val
        valfunc_pmp = np.reshape(val_funcs["PMP"][:, :, theta_indices_matlab[j], time_indices_matlab[i]-1], valfunc_true.shape)

        ## Plot the actual value function
        ax = fig_valfunc_LS.add_subplot(num_times, num_thetas, (j + 1) + i * num_thetas)
        ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
        s = ax.imshow(valfunc_true.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
        fig_valfunc_LS.colorbar(s)

        ## Plot the predicted value function
        ax = fig_valfunc_pmp.add_subplot(num_times, num_thetas, (j + 1) + i * num_thetas)
        ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
        s = ax.imshow(valfunc_pmp.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
        fig_valfunc_pmp.colorbar(s)

        fig_valfunc_LS.savefig('Air3D_LS_valfunc.png')
        fig_valfunc_pmp.savefig('Air3D_PMP_valfunc.png')