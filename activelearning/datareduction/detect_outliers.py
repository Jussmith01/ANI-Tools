import pyNeuroChem as pync
import hdnntools as hdn
import pyanitools as pyt
import numpy as np
import os

wkdir = '/home/jujuman/Scratch/Research/DataReductionMethods/models/cv/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

# Construct pyNeuroChem classes
print('Constructing CV network list...')
ncl =  [pync.conformers(cnstfile, saefile, wkdir + 'cv_train_' + str(l) + '/networks/', 0, False) for l in range(4)]
print('Complete.')

store_xyz = '/home/jujuman/Research/DataReductionMethods/models/cv/bad_xyz/'

svpath = '/home/jujuman/Scratch/Research/DataReductionMethods/models/ani_red_cnl_c08f.h5'
h5file = '/home/jujuman/Scratch/Research/DataReductionMethods/models/train_c08f/ani_red_c08f.h5'

# Remove file if exists
if os.path.exists(svpath):
    os.remove(svpath)

#open an HDF5 for compressed storage.
dpack = pyt.datapacker(svpath)

# Declare loader
adl = pyt.anidataloader(h5file)

Nd = 0
Nb = 0
for data in adl:
    # Extract the data
    Ea = data['energies']
    S = data['species']
    X = data['coordinates'].reshape(Ea.shape[0], len(S),3)

    El = []
    for nc in ncl:
        nc.setConformers(confs=X, types=list(S))
        El.append(np.abs(hdn.hatokcal*(nc.energy().copy() - Ea))/float(len(S)))

    El = np.vstack(El)
    #dEm = np.sum(El, axis=0) / 3.0

    bad_idx = []
    god_idx = []
    for  i in range(Ea.shape[0]):
        test = np.array([True if j > 1.0 else False for j in El[:,i]])
        if np.all(test):
            bad_idx.append(i)
        else:
            god_idx.append(i)

    Nd = Nd + len(god_idx)
    Nb = Nb + len(bad_idx)

    if len(god_idx) != 0:
        # print(gn)
        dpack.store_data(data['parent'] + "/" + data['name'], coordinates=X[god_idx], energies=Ea[god_idx], species=S)

    if len(bad_idx) > 0:
        print(data['parent'],data['name'],'- Bad:',len(bad_idx),'of',len(god_idx))
        for i in bad_idx:
            print('(', i, ')', ':' ,El[:,i])

        hdn.writexyzfile(store_xyz+data['parent'] + '_' + data['name']+'.xyz',X[bad_idx],S)

dpack.cleanup()
print('Bad:',Nb,'of',Nd+Nb)

