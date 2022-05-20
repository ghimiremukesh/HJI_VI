import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

bvpAA = True
bvpNANA = False
bvpANA = False
bvpNAA = False

#________________________________________________________________________________________

if bvpAA is True:
    # file = 'data_test_a_a_no collision_inter.mat'
    # file = 'closedloop_traj_supervised_a_a.mat'
    file = 'closedloop_traj_hybrid_a_a.mat'
    file1 = 'data_train_a_a_ub.mat'
    file2 = 'data_train_a_a_lb.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(a,a)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1

if bvpANA is True:
    file = 'data_test_a_na_no collision_inter.mat'
    # file = 'closedloop_traj_supervised_a_na.mat'
    # file = 'closedloop_traj_hybrid_a_na.mat'
    file1 = 'data_train_a_na_ub.mat'
    file2 = 'data_train_a_na_lb.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(a,na)$'
    # title = 'Neural Network $\Theta^{*}=(a,na)$'
    special = 0
    theta1 = 1
    theta2 = 5

if bvpNAA is True:
    file = 'data_test_na_a_no collision_inter.mat'
    # file = 'closedloop_traj_supervised_na_a.mat'
    # file = 'closedloop_traj_hybrid_na_a.mat'
    file1 = 'data_train_na_a_ub.mat'
    file2 = 'data_train_na_a_lb.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(na,a)$'
    # title = 'Neural Network $\Theta^{*}=(na,a)$'
    special = 0
    theta1 = 5
    theta2 = 1

if bvpNANA is True:
    file = 'data_test_na_na_no collision_inter.mat'
    # file = 'closedloop_traj_supervised_na_na.mat'
    # file = 'closedloop_traj_hybrid_na_na.mat'
    file1 = 'data_train_na_na_ub.mat'
    file2 = 'data_train_na_na_lb.mat'
    index = 2
    title = 'BVP $\Theta^{*}=(na,na)$'
    # title = 'Neural Network $\Theta^{*}=(na,na)$'
    special = 0
    theta1 = 5
    theta2 = 5

#____________________________________________________________________________________________________

font = {'family': 'normal', 'weight': 'normal', 'size': 18}

plt.rc('font', **font)

data = scipy.io.loadmat(file)
X = data['X']
V = data['V']
T = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

data1 = scipy.io.loadmat(file1)
data2 = scipy.io.loadmat(file2)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

norm = plt.Normalize(np.min(V[0, :]), np.max(V[0, :]))
# norm = plt.Normalize(np.min(V[1, :]), np.max(V[1, :]))

for n in range(1, len(idx0) + 1):
    if n == len(idx0):
        d1 = X[0, idx0[n - 1]:]
        d2 = X[2, idx0[n - 1]:]
        V1 = V[0, idx0[n - 1]:]
        V2 = V[1, idx0[n - 1]:]
        t = T[0, idx0[n - 1]:]
    else:
        d1 = X[0, idx0[n - 1]: idx0[n]]
        d2 = X[2, idx0[n - 1]: idx0[n]]
        V1 = V[0, idx0[n - 1]: idx0[n]]
        V2 = V[1, idx0[n - 1]: idx0[n]]
        t = T[0, idx0[n - 1]: idx0[n]]

    points = np.array([d1, d2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize/Plot Value 1 ColorBar
    lc = LineCollection(segments, cmap='PuOr', norm=norm)  # PuOr, viridis
    lc.set_array(V1)
    # lc.set_array(V2)
    line = axs.add_collection(lc)

X1 = data1['X']
t1 = data1['t']
data1.update({'t0': data1['t']})
idx01 = np.nonzero(np.equal(data1.pop('t0'), 0.))[1]
#
X2 = data2['X']
t2 = data2['t']
data2.update({'t0': data2['t']})
idx02 = np.nonzero(np.equal(data2.pop('t0'), 0.))[1]

x11 = X1[0, idx01[0]:]
x12 = X1[index, idx01[0]:]
axs.plot(x11, x12, c='red')
#
x21 = X2[0, idx02[0]:]
x22 = X2[index, idx02[0]:]
axs.plot(x21, x22, c='red')

# Configure Plot
axs.set_title(title)
train2 = patches.Rectangle((35 - theta1*0.75, 35 - theta2*0.75), 3+theta1*0.75+0.75, 3+theta2*0.75+0.75, linewidth=1, edgecolor='k', facecolor='none')
start2 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
intersection2 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='grey', facecolor='grey')
axs.add_patch(intersection2)
axs.add_patch(train2)
axs.add_patch(start2)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')
fig.colorbar(line, ax=axs)

plt.show()