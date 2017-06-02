# Neuro Chem
import pyNeuroChem as pync
import numpy as np
import hdnntools as hdt
import pyanitools as pyt

#--------------Parameters------------------

path = "/home/jujuman/Research/ANI-DATASET/ANI-1_release/data/ani-1_data_c08.h5"

wkdir = '/home/jujuman/Research/CrossValidation/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

#-------------------------------------------
# Build networks
nc =  [pync.conformers(cnstfile, saefile, wkdir + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(5)]

# Build loader
adl = pyt.anidataloader(path)

# Load data
adl.load_node("/gdb11_s01/")

# Loop
for i in range(adl.size()):
    #print(i, ' of ', adl.size())
    data = adl.getdata(i)

    x = data[0]
    e = data[1]
    s = data[2]

    Nm = e.shape[0]
    Na = len(s)

    xyz_t = np.array(x, dtype=np.float32, order='C').reshape(Nm, Na, 3)
    spc_t = s

    energies = np.zeros((5,e.shape[0]), dtype=np.float64)
    for j,comp in enumerate(nc):
        comp.setConformers(confs=xyz_t, types=list(spc_t))
        energies[j] = comp.energy()

    print(s)
    for eA in energies:
        print(list(e))
        print(list(eA))
        print(hdt.hatokcal*(np.abs(eA-e).sum()/Nm))

adl.cleanup()