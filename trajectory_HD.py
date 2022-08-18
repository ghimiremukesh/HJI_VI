import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from examples.choose_problem import system, problem

#____________________________________________________________________________________________________

bvpAA = True
bvpANA = False
bvpNAA = False
bvpNANA = False

#________________________________________________________________________________________
if bvpAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_train_a_a.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1
if bvpANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_train_a_na.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(a,na)$'
    special = 0
    theta1 = 1
    theta2 = 5
if bvpNAA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_train_na_a.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(na,a)$'
    special = 0
    theta1 = 5
    theta2 = 1
if bvpNANA is True:
    data = scipy.io.loadmat('examples/' + system + '/data_train_na_na.mat')
    index = 2
    title = 'BVP $\Theta^{*}=(na,na)$'
    special = 0
    theta1 = 5
    theta2 = 5

#____________________________________________________________________________________________________

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
plt.rc('font', **font)

X = data['X']
V = data['V']
t = data['t']
co_state = data['A']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

fig, axs = plt.subplots(1, 1, figsize=(7, 7))

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        X1 = X[:4, idx0[n - 1]:]
        X2 = X[4:, idx0[n - 1]:]

        dx1 = X1[1, :]
        dy1 = X1[0, :]
        dx2 = X2[0, :]
        dy2 = X2[1, :]

        dist = np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2)

        T = t[0, idx0[n - 1]:]
        axs.plot(T, dist)
    else:
        X1 = X[:4, idx0[n - 1]: idx0[n]]
        X2 = X[4:, idx0[n - 1]: idx0[n]]

        dx1 = X1[1, :]
        dy1 = X1[0, :]
        dx2 = X2[0, :]
        dy2 = X2[1, :]

        dist = np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2)

        T = t[0, idx0[n - 1]: idx0[n]]
        axs.plot(T, dist)

# train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 3 + theta1 * 0.75 + 0.75,
#                             3 + theta2 * 0.75 + 0.75, linewidth=1, edgecolor='k', facecolor='none')
# start1 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
# intersection1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
# axs.add_patch(intersection1)
# axs.add_patch(train1)
# axs.add_patch(start1)
# axs.set_xlim(15, 40)
# axs.set_xlabel('d1')
# axs.set_ylim(15, 40)
# axs.set_ylabel('d2')

# p1 = [0, 35 - theta1 * 0.75]
# p2 = [0, 35 - theta2 * 0.75]
# axs.plot(p1, p2, '-r')

axs.set_title(title)
axs.set_xlabel("time")
axs.set_ylabel('distance between cars')

plt.show()