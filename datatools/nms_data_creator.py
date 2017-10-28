import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync

# Scipy
import scipy.spatial as scispc

# SimDivPicker
from rdkit.SimDivFilters import rdSimDivPickers

wkdir    = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

# reaction center atoms
aevsize = 384
Nk = 300
T = 2000
Ngen = 180
Nkep = 24

idir = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-10/inputs/'
cdir = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-10/confs/'
files = os.listdir(idir)
files.sort()

#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt

nc = pync.conformers(cnstfile, saefile, nnfdir, 0, False)
for f in files:
    data = hdn.read_rcdb_coordsandnm(idir+f)
    spc = data["species"]
    xyz = data["coordinates"]
    nmc = data["nmdisplacements"]
    frc = data["forceconstant"]

    nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=1.0E-2)

    conformers = []
    for i in range(Ngen):
        conformers.append(nms.get_random_structure())
    conformers = np.stack(conformers)

    nc.setConformers(confs=conformers, types=list(spc))
    Ecmp = nc.energy() # this generates AEVs

    atoms = [i for i,s in enumerate(spc) if s is not 'H']
    aevs = np.empty([Ngen, len(atoms) * aevsize])
    for m in range(Ngen):
        for j,a in enumerate(atoms):
            aevs[m, j * aevsize:(j + 1) * aevsize] = nc.atomicenvironments(a, m).copy()

    dm = scispc.distance.pdist(aevs, 'sqeuclidean')
    picker = rdSimDivPickers.MaxMinPicker()
    seed_list = [i for i in range(Ngen)]
    np.random.shuffle(seed_list)
    #print('seed:',seed_list)
    ids = list(picker.Pick(dm, Ngen, Nkep, firstPicks=list(seed_list[0:5])))
    ids.sort()
    print(f,len(ids),conformers.shape,dm.shape,":",ids)

    #plt.hist(hdn.hatokcal*(Ecmp-Ecmp.min()), 100, normed=1, facecolor='blue', alpha=0.5)
    #plt.hist(hdn.hatokcal*(Ecmp[ids]-Ecmp[ids].min()), 100, normed=1, facecolor='green', alpha=0.5)
    #plt.show()

    hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)
