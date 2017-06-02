import numpy as np
import matplotlib.pyplot as plt

dfile = '/home/jujuman/Research/SingleNetworkTest/train_05/diffs.dat'

f = open(dfile, 'r')
for l in f:
    diffs = np.array(l.split(','),dtype=np.float)
    plt.scatter(np.arange(diffs.size), diffs, color='black', label='DIFF', linewidth=1)
    plt.show()