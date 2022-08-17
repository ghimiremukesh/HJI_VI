import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from examples.choose_problem import system
from NN_output import get_Q_value

font = {'family': 'normal', 'weight': 'normal', 'size': 24}
plt.rc('font', **font)

x1_axis = np.zeros(11)
x2_axis = np.zeros(11)

for i in range(11):
    x1_axis[i] = 15.1 + 0.5 * i
    x2_axis[i] = 15 + 0.5 * i

x1, x2 = np.meshgrid(x1_axis, x2_axis)

# data1 = scipy.io.loadmat('examples/' + system + '/data_train_a_a_lower.mat')
# data2 = scipy.io.loadmat('examples/' + system + '/data_train_a_a_upper.mat')
#
# t_OUT = np.hstack((data1['t'], data2['t']))
# X_OUT = np.hstack((data1['X'], data2['X']))
# A_OUT = np.hstack((data1['A'], data2['A']))
# V_OUT = np.hstack((data1['V'], data2['V']))
#
# data = {'t': t_OUT,
#         'X': X_OUT,
#         'A': A_OUT,
#         'V': V_OUT}

data = scipy.io.loadmat('examples/' + system + '/data_train_na_na.mat')

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0))[1]
print(idx0)

X = data['X'][:, idx0]
t = data['t'][:, idx0]
V = data['V'][:, idx0]
A = data['A'][:, idx0]
U = np.zeros((2, t.shape[1]))

V1 = V[0, :]
V2 = V[1, :]
A1 = A[1, :]
A2 = A[7, :]

# plt.figure(1)
# # ax = plt.axes(projection="3d")
# # ax.plot_surface(x1, x2, V1.reshape(11, 11))
# # ax.set_xlabel('Trajectory x1')
# # ax.set_ylabel('Trajectory x2')
# # ax.set_zlabel('V1')
# # ax.title.set_text('Value V1 Tendency under a-a policy pair(v1=v2=18m/s)')
# # ax.legend(loc='upper left')
# # plt.show()
#
# a = plt.contourf(x1, x2, V1.reshape(11, 11), 8, cmap=plt.cm.Spectral)
# b = plt.contour(x1, x2, V1.reshape(11, 11), 8, colors='black', linewidths=1, linestyles='solid')
# plt.colorbar(a)
# plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
# plt.xlabel('Trajectory d1')
# plt.ylabel('Trajectory d2')
# # plt.title('Value V1 Tendency under a-a policy pair(v1=v2=18m/s)')
# plt.title('Value V1 Tendency under (v1=v2=18m/s)')
#
# plt.legend(loc='upper left')
# plt.show()
#
# # fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# # for n in range(1, len(idx0) + 1):
# #     if n == len(idx0):
# #         V1 = data['V'][0, idx0[n - 1]:]
# #         T = data['t'][0, idx0[n - 1]:]
# #         axs.plot(T, V1)
# #     else:
# #         V1 = data['V'][0, idx0[n - 1]: idx0[n]]
# #         T = data['t'][0, idx0[n - 1]: idx0[n]]
# #         axs.plot(T, V1)
# #
# # title = 'Value V1 Tendency under a-a policy pair(v1=v2=18m/s)'
# # axs.set_xlabel('t')
# # axs.set_title(title)
# # plt.show()

# plt.figure(2)
# # ax = plt.axes(projection="3d")
# # ax.plot_surface(x1, x2, V2.reshape(11, 11))
# # ax.set_xlabel('Trajectory x1')
# # ax.set_ylabel('Trajectory x2')
# # ax.set_zlabel('V2')
# # ax.title.set_text('Value V2 Tendency under a-a policy pair(v1=v2=18m/s)')
# #
# # ax.legend(loc='upper left')
# # plt.show()
#
# a = plt.contourf(x1, x2, V2.reshape(11, 11), 8, cmap=plt.cm.Spectral, fontsize=14)
# b = plt.contour(x1, x2, V2.reshape(11, 11), 8, colors='black', linewidths=1, linestyles='solid')
# plt.colorbar(a)
# plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
# plt.xlabel('Trajectory d1')
# plt.ylabel('Trajectory d2')
# # plt.title('Value V2 Tendency under a-a policy pair(v1=v2=18m/s)')
# plt.title('Value V2 Tendency (v1=v2=18m/s)')
#
# plt.legend(loc='upper left')
# plt.show()
#
# # fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# # # for n in range(1, len(idx0) + 1):
# # for n in range(120, 121):
# #     if n == len(idx0):
# #         V2 = data['V'][1, idx0[n - 1]:]
# #         T = data['t'][0, idx0[n - 1]:]
# #         axs.plot(T, V2)
# #     else:
# #         V2 = data['V'][1, idx0[n - 1]: idx0[n]]
# #         T = data['t'][0, idx0[n - 1]: idx0[n]]
# #         axs.plot(T, V2)
# #
# # title = 'Value V2 Tendency under a-a policy pair(v1=v2=18m/s)'
# # axs.set_xlabel('t')
# # axs.set_title(title)
# # plt.show()

# plt.figure(3)
# # ax = plt.axes(projection="3d")
# # ax.plot_surface(x1, x2, A1.reshape(11, 11))
# # ax.set_xlabel('Trajectory x1')
# # ax.set_ylabel('Trajectory x2')
# # ax.set_zlabel('A1')
# # ax.title.set_text('Costate A1 Tendency under a-a policy pair(v1=v2=18m/s)')
# #
# # ax.legend(loc='upper left')
# # plt.show()
#
# a = plt.contourf(x1, x2, A1.reshape(11, 11), 6, cmap=plt.cm.Spectral, fontsize=14)
# b = plt.contour(x1, x2, A1.reshape(11, 11), 6, colors='black', linewidths=1, linestyles='solid')
# plt.colorbar(a)
# plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
# plt.xlabel('Trajectory d1')
# plt.ylabel('Trajectory d2')
# # plt.title('Costate A1 Tendency under a-a policy pair(v1=v2=18m/s)')
# plt.title('Costate $\lambda_{1}$ Tendency (v1=v2=18m/s)')
#
# plt.legend(loc='upper left')
# plt.show()
#
# # fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# # for n in range(1, len(idx0) + 1):
# #     if n == len(idx0):
# #         A1 = data['A'][1, idx0[n - 1]:]
# #         T = data['t'][0, idx0[n - 1]:]
# #         axs.plot(T, A1)
# #     else:
# #         A1 = data['A'][1, idx0[n - 1]: idx0[n]]
# #         T = data['t'][0, idx0[n - 1]: idx0[n]]
# #         axs.plot(T, A1)
# #
# # title = 'Costate A1 Tendency under a-a policy pair(v1=v2=18m/s)'
# # axs.set_xlabel('t')
# # axs.set_title(title)
# # plt.show()

plt.figure(4)
# ax = plt.axes(projection="3d")
# ax.plot_surface(x1, x2, A2.reshape(11, 11))
# ax.set_xlabel('Trajectory x1')
# ax.set_ylabel('Trajectory x2')
# ax.set_zlabel('A2')
# ax.title.set_text('Costate A2 Tendency under a-a policy pair(v1=v2=18m/s)')
#
# ax.legend(loc='upper left')
# plt.show()

a = plt.contourf(x1, x2, A2.reshape(11, 11), 6, cmap=plt.cm.Spectral, fontsize=14)
b = plt.contour(x1, x2, A2.reshape(11, 11), 6, colors='black', linewidths=1, linestyles='solid')
plt.colorbar(a)
plt.clabel(b, inline=True, fontsize=16, fmt='%1.2f', manual=True)
plt.xlabel('Trajectory d1')
plt.ylabel('Trajectory d2')
# plt.title('Costate A2 Tendency under a-a policy pair(v1=v2=18m/s)')
plt.title('Costate $\lambda_{2}$ Tendency (v1=v2=18m/s)')

plt.legend(loc='upper left')
plt.show()

# fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# for n in range(1, len(idx0) + 1):
#     if n == len(idx0):
#         A2 = data['A'][7, idx0[n - 1]:]
#         T = data['t'][0, idx0[n - 1]:]
#         axs.plot(T, A2)
#     else:
#         A2 = data['A'][7, idx0[n - 1]: idx0[n]]
#         T = data['t'][0, idx0[n - 1]: idx0[n]]
#         axs.plot(T, A2)
#
# title = 'Costate A2 Tendency under a-a policy pair(v1=v2=18m/s)'
# axs.set_xlabel('t')
# axs.set_title(title)
# plt.show()