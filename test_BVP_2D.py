import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
from scipy import io
from examples.choose_problem import system

# train_data = scipy.io.loadmat('examples/' + system + '/data_E_a_a_belief_a_a.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_E_a_a_belief_na_na.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_E_na_na_belief_a_a.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_E_na_na_belief_na_na.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_NE_a_a_belief_a_a.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_NE_a_a_belief_na_na.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_NE_na_na_belief_a_a.mat')
train_data = scipy.io.loadmat('examples/' + system + '/data_NE_na_na_belief_na_na.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_baseline_a_a.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_baseline_na_na.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_BVP_a_a.mat')
# train_data = scipy.io.loadmat('examples/' + system + '/data_BVP_na_na.mat')

X = train_data['X']
V = train_data['V']
T = train_data['t']

train_data.update({'t0': train_data['t']})
idx0 = np.nonzero(np.equal(train_data.pop('t0'), 0.))[1]

# x1 = X[0, idx0[0]: idx0[1]]
# x2 = X[2, idx0[0]: idx0[1]]
# V1 = V[0, idx0[0]: idx0[1]]
# V2 = V[1, idx0[0]: idx0[1]]
# t = T[0, idx0[0]: idx0[1]]

fig, ax = plt.subplots()
# ax = fig.gca()
length = len(idx0)

for i in range(length):
    # There is 36 trajectories in data_BVP_na_na and data_baseline_na_na
    # The rest data contains 30 trajectories
    # For data_BVP_na_na and data_BVP_a_a, the index for x1 and x2 is 0 and 2 respectively,
    # For example, x1 = X[0, idx0[0]: idx0[1]], x2 = X[2, idx0[0]: idx0[1]]
    # For the rest data set, the index for x1 and x2 is 0 and 1 respectively,
    # For example, x1 = X[0, idx0[0]: idx0[1]], x2 = X[1, idx0[0]: idx0[1]]
    if i == 35:  # This number should be change to 35 when using data_BVP_na_na and data_Baseline_na_na
        plt.plot(X[0, idx0[i]:], X[2, idx0[i]:])
    else:
        plt.plot(X[0, idx0[i]: idx0[i + 1]], X[2, idx0[i]: idx0[i + 1]])
    # print(X[0, idx0[i]], X[2, idx0[i]])

rect1 = patches.Rectangle((34.25, 34.25), 4.5, 4.5, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(rect1)
rect = patches.Rectangle((31.25, 31.25), 7.5, 7.5, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.set_xlabel('Trajectory x1')
ax.set_ylabel('Trajectory x2')

ax.legend(loc='upper left')
ax.axis('equal')
plt.show()


