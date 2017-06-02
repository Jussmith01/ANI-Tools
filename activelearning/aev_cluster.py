import pyNeuroChem as pync
import hdnntools as hdn
import pyanitools as pyt
import numpy as np

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
tspc = []
for data in adl:
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
            tspc.append(spc[a])
            aevs.append(nc.atomicenvironments(atom_idx=a,molec_idx=m).copy())

X = np.vstack(aevs)
print(X.shape)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=30, random_state=0).fit(X)
print(kmeans.labels_.shape)

for s,l in zip(tspc, kmeans.labels_):
    print(l,s)