# pyneurochem
import pyNeuroChem as pync
import pyanitools as pyt
import pyaniasetools as aat
import hdnntools as hdn

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# Define test file
h5file = '/home/jujuman/Research/ForceNMPaper/polypeptide/tripeptide_full.h5'
h5file = '/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s07.h5'

# Define cross validation networks
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-06/cv3/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

# Define the conformer cross validator class
anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

# Declare data loader
adl = pyt.anidataloader(h5file)

means = []
sigms = []

# Iterate data set
for data in adl:
    # Extract the data
    X  = data['coordinates']
    S  = data['species']
    Ea = data['energies']

    minid = np.argmin(Ea)

    X  = X[minid].reshape(1,X.shape[1],3)
    Ea = np.array(Ea[minid])

    # Calculate std. dev. per atom for all conformers
    sigma = anicv.compute_stddev_conformations(X,S)

    # Calculate energy deltas
    delta = anicv.compute_energy_delta_conformations(X,Ea,S)

    # Print result
    #print('----------Result----------')
    #print(np.mean(delta,axis=0))
    #print(sigma)
    means.append(np.max(delta,axis=0)/float(len(S)))
    sigms.append(sigma)

means = np.concatenate(means)
sigms = np.concatenate(sigms)

plt.scatter(means,sigms)
plt.xlabel('Max Error (kcal/mol/atoms)')
plt.ylabel('Model std. dev. (kcal/mol/atoms)')
plt.show()