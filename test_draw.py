import numpy as np
import matplotlib.pyplot as plt
import torch

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 22}

plt.rc('font', **font)

# Road length
R = 70
R = torch.tensor(R, dtype=torch.float32)
# Vehicle width
W = 1.5
W = torch.tensor(W, dtype=torch.float32)
# Vehicle length
L = 3
L = torch.tensor(L, dtype=torch.float32)

theta = 1
theta = torch.tensor(theta, dtype=torch.float32)

# weight for sigmoid function
beta = 10000

def sigmoid(x):
    x = torch.tensor(x, dtype=torch.float32)
    return beta * torch.sigmoid((x - R / 2 + theta * W / 2) * 5) * torch.sigmoid(-(x - R / 2 - W / 2 - L) * 5).detach().numpy()

x_axis = np.arange(20, 50, 0.1)
sigmoid_outputs = sigmoid(x_axis)

plt.plot(x_axis, sigmoid_outputs, linewidth=3)
# plt.xlabel("Player i's location d")
# ylabel = 'Collision Penalty $\Theta^{*}=(a,a)$'
# plt.ylabel(ylabel)
# my_x_ticks = np.arange(20, 51, 1)
# my_y_ticks = np.arange(0, 0.6, 0.05)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# title = 'Collision  when vehicle is non-aggressive'
# plt.title(title)
plt.show()
