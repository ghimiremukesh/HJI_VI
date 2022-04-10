import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

#____________________________________________________________________________________________________

sym = False
baselineAA = False
baselineNANA = False
baselineANA = False
baselineNAA = False
bvpAA = True
bvpNANA = False
bvpANA = False
bvpNAA = False

bl = (34.25, 34.25)
tr = (38.75, 38.75)  #(collision area)

#________________________________________________________________________________________

def pointInRect(bl, tr, points):
    for p in points:
        if (p[0] >= bl[0] and p[0] <= tr[0] and p[1] >= bl[1] and p[1] <= tr[1]):
            return True
    return False

if bvpAA is True:
    file = 'data_test_a_a_18_18_inter.mat'
    data = scipy.io.loadmat(file)
    index = 2
    title = 'BVP $\Theta^{*}=(a,a)$'
    # title = 'Neural Network $\Theta^{*}=(a,a)$'
    special = 0
    theta1 = 1
    theta2 = 1

#____________________________________________________________________________________________________

font = {'family': 'normal', 'weight': 'normal', 'size': 18}

plt.rc('font', **font)

X = data['X']
V = data['V']
t = data['t']

data.update({'t0': data['t']})
idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

X0 = X[:, idx0]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
count = 0

for n in range(1, len(idx0) + 1):
# for n in range(53, 54):
    if n == len(idx0):
        x1 = X[0, idx0[n - 1]:]
        x2 = X[2, idx0[n - 1]:]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n)
        axs.plot(x1, x2)
    else:
        x1 = X[0, idx0[n - 1]: idx0[n]]
        x2 = X[2, idx0[n - 1]: idx0[n]]

        pairs = zip(x1, x2)
        if pointInRect(bl, tr, pairs):
            count += 1
            print(n)
        axs.plot(x1, x2)

train1 = patches.Rectangle((35 - theta1 * 0.75, 35 - theta2 * 0.75), 4.5,
                            4.5, linewidth=1, edgecolor='k', facecolor='grey')
start1 = patches.Rectangle((15, 15), 5, 5, linewidth=0.5, edgecolor='k', facecolor='none')
axs.add_patch(train1)
axs.add_patch(start1)
axs.set_xlim(15, 40)
axs.set_xlabel('d1')
axs.set_ylim(15, 40)
axs.set_ylabel('d2')

p1 = [0, 35 - theta1 * 0.75]
p2 = [0, 35 - theta2 * 0.75]
axs.plot(p1, p2, color='k', LineWidth=1)

print("Total Collision: %d" %count)
axs.set_title(title, fontsize=16)
plt.show()
