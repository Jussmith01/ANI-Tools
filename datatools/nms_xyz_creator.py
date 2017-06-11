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
rcatoms = [0, 1, 2, 3]
inclist = [0,1,2,3,4,5,6,7,8,9]
aevsize = 384
Nk = 300
T = 600
Ngen = 300
Nkep = 30

idir = '/home/jujuman/Research/ReactionGeneration/DataGen/inputs/'
cdir = '/home/jujuman/Research/ReactionGeneration/DataGen/confs/'
files = os.listdir(idir)
files.sort()

nc = pync.conformers(cnstfile, saefile, nnfdir, 0, False)

aevs = np.empty([len(files),len(rcatoms)*aevsize])
l_dat = []
for m,f in enumerate(files):
    #print(f)
    data = hdn.read_rcdb_coordsandnm(idir+f)
    l_dat.append(data)

    spc = data["species"]
    xyz = data["coordinates"]

    nc.setConformers(confs=xyz.reshape(1,len(spc),3), types=list(spc))
    Ecmp = nc.energy()

    for i,a in enumerate(rcatoms):
        aevs[m, i * aevsize:(i + 1) * aevsize] = nc.atomicenvironments(a, 0).copy()

dm = scispc.distance.pdist(aevs, 'sqeuclidean')
picker = rdSimDivPickers.MaxMinPicker()
seed_list = [ i for i in range(aevs.shape[0])]
np.random.shuffle(seed_list)
print(seed_list)
ids = set(picker.Pick(dm, aevs.shape[0], Nk, firstPicks=list(seed_list[0:10])))
ids.update(set(inclist))
ids = list(ids)
print(ids)
ids.sort()

of = open(cdir+'kept_data.nfo','w')
for i in ids:
    data = l_dat[i]
    f = files[i]
    of.write(f+'\n')
    of.flush()
    #print (data)

    spc = data["species"]
    xyz = data["coordinates"]
    nmc = data["nmdisplacements"]
    frc = data["forceconstant"]

    nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)

    conformers = []
    for i in range(Ngen):
        conformers.append(nms.get_random_structure())
    conformers = np.stack(conformers)

    nc.setConformers(confs=conformers, types=list(spc))
    Ecmp = nc.energy() # this generates AEVs

    aevs = np.empty([Ngen, len(rcatoms) * aevsize])
    for m in range(Ngen):
        for j,a in enumerate(rcatoms):
            aevs[m, j * aevsize:(j + 1) * aevsize] = nc.atomicenvironments(a, m).copy()

    dm = scispc.distance.pdist(aevs, 'sqeuclidean')
    picker = rdSimDivPickers.MaxMinPicker()
    seed_list = [i for i in range(Ngen)]
    np.random.shuffle(seed_list)
    ids = list(picker.Pick(dm, Ngen, Nkep, firstPicks=list(seed_list[0:5])))
    ids.sort()
    print(f,len(ids),conformers.shape,dm.shape,":",ids)

    hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)
of.close()