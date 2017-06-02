import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

dir = '/home/jujuman/Research/DataReductionMethods/models/train_test/learning_curves/'
files = os.listdir(dir)


cmap = mpl.cm.autumn
nsteps = len(files)
for i,f in enumerate(files):
    data = np.loadtxt(dir+f)
    print(data)
    plt.semilogy(np.arange(data[:, 1].size), data[:, 1], r'-', color=cmap(i / float(nsteps)), label='Train ' + str(i), linewidth=1)
    plt.semilogy(np.arange(data[:, 2].size), data[:, 2], r'--', color=cmap(i / float(nsteps)), label='Valid ' + str(i), linewidth=1)

plt.grid(True)
plt.legend(bbox_to_anchor=(0.6, 0.99), loc=2, borderaxespad=0.,fontsize=14)
plt.show()