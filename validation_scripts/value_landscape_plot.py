import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.io as scio

font = {'family': 'normal', 'weight': 'normal', 'size': 16}
plt.rc('font', **font)
fig, axs = plt.subplots(1, 1, figsize=(7, 7))

data_landscape = scio.loadmat('valuelandscape.mat')
x1 = data_landscape['x1']
x2 = data_landscape['x2']
V1 = data_landscape['V1']
V2 = data_landscape['V2']

plt.figure(1)
theta1, theta2 = 1, 1
train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 3 + theta1 * 0.75 + 0.75,
                            3 + theta2 * 0.75 + 0.75, linewidth=2, edgecolor='k', facecolor='none')

# train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 1.5,
#                             1.5, linewidth=2, edgecolor='k', facecolor='none')
start1 = patches.Rectangle((15, 15), 5, 5, linewidth=2, edgecolor='k', facecolor='none')
axs.add_patch(train1)
axs.add_patch(start1)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')
axs.set_title('V1 contour under $\Theta^{*}=(a,a)$ in initial time')

a = plt.contourf(x1, x2, V1, 8, cmap=plt.cm.Spectral, fontsize=14)
b = plt.contour(x1, x2, V1, 8, colors='black', linewidths=1, linestyles='solid')
plt.colorbar(a)
plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
plt.show()

# plt.figure(1)
# theta1, theta2 = 1, 1
# train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 3 + theta1 * 0.75 + 0.75,
#                             3 + theta2 * 0.75 + 0.75, linewidth=2, edgecolor='k', facecolor='none')
#
# # train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 1.5,
# #                             1.5, linewidth=2, edgecolor='k', facecolor='none')
# start1 = patches.Rectangle((15, 15), 5, 5, linewidth=2, edgecolor='k', facecolor='none')
# axs.add_patch(train1)
# axs.add_patch(start1)
# axs.set_xlim(15, 40)
# axs.set_xlabel('d1')
# axs.set_ylim(15, 40)
# axs.set_ylabel('d2')
# axs.set_title('V2 contour under $\Theta^{*}=(a,a)$ in initial time')
#
# a = plt.contourf(x1, x2, V2, 8, cmap=plt.cm.Spectral, fontsize=14)
# b = plt.contour(x1, x2, V2, 8, colors='black', linewidths=1, linestyles='solid')
# plt.colorbar(a)
# plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
# plt.show()

