import pyNeuroChem as pync
import hdnntools as hdn
import pyanitools as pyt
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

wkdir = '/home/jujuman/Research/DataReductionMethods/models/train_c08f/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

# Construct pyNeuroChem classes
nc = pync.conformers(cnstfile, saefile, nnfdir, 0, True)

h5file = '/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c01.h5'

# Declare loader
adl = pyt.anidataloader(h5file)

aevs = []
aevi = []
tspc = []
for i,data in enumerate(adl):
    # Extract molecule data from somewhere
    erg = data['energies']
    spc = data['species']
    xyz = data['coordinates'].reshape(erg.shape[0], len(spc), 3)

    Nm = erg.shape[0] # Number of molecules
    Na = len(spc) # number of atoms

    # Set the conformers
    nc.setConformers(confs=xyz, types=list(spc))

    # Compute Energies of Conformations this will produce the AEVs which we can load next
    Ec = nc.energy().copy()

    # Load AEVs for all atoms, store in aves
    for m in range(Nm):
        for a in range(Na):
            if spc[a] != 'H':
                aevi.append((i,Nm,Na))
                tspc.append(spc[a])
                aevs.append(nc.atomicenvironments(atom_idx=a,molec_idx=m).copy())

X = np.vstack(aevs)
print(X.shape)

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from matplotlib.pyplot import cm
Nc = 10

k_means = KMeans(n_clusters=Nc, random_state=0).fit(X)
labels = k_means.labels_
center = k_means.cluster_centers_

T = [[],[],[],[],[],[],[],[],[],[],]
D = [[],[],[],[],[],[],[],[],[],[],]

for i,(l,x) in enumerate(zip(labels,X)):
    d = np.linalg.norm(center[l] - x)
    Lc = np.linalg.norm(center[l])
    Lx = np.linalg.norm(x)
    t = np.dot(center[l],x)

    T[l].append(t)
    D[l].append(d)

color=cm.rainbow(np.linspace(0,1,Nc))
for t,d,c in zip(T,D,color):
    plt.scatter(t,d,color=c)

plt.show()