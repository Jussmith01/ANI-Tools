import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync
<<<<<<< HEAD
import pyaniasetools as aat
=======

# Scipy
import scipy.spatial as scispc

# SimDivPicker
from rdkit.SimDivFilters import rdSimDivPickers
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808

wkdir    = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

# reaction center atoms
fpatoms = ['C', 'N', 'O']
aevsize = 384
<<<<<<< HEAD
T = 800
K = 10
P = 0.25
atmlist = [0, 1, 2, 3]

#idir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_05_red/inputs/'
#cdir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_05_red/confs/'

idir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/inputs/'
cdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/confs/'

=======
T = 1000
K = 175
P = 0.25

idir = '/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_03/inputs/'
cdir = '/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_03/confs/'
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
files = os.listdir(idir)
files.sort()

if not os.path.exists(cdir):
    os.mkdir(cdir)

<<<<<<< HEAD
dc = aat.diverseconformers(cnstfile, saefile, nnfdir, aevsize, 0, False)

for fi,f in enumerate(files):
    print(fi,'of',len(files),') Working on:', f)
=======
nc = pync.conformers(cnstfile, saefile, nnfdir, 0, False)

for f in files:
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
    data = hdn.read_rcdb_coordsandnm(idir+f)

    spc = data["species"]
    xyz = data["coordinates"]
    nmc = data["nmdisplacements"]
    frc = data["forceconstant"]

    Ngen = K*frc.size
    Nkep = int(Ngen*P)
<<<<<<< HEAD
    Ngen = 800
    Nkep = 400

    print('    -',Ngen,'of',Nkep)
=======

    print('Working on:', f,' - ',Ngen,'of',Nkep)
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808

    nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)

    conformers = []
    for i in range(Ngen):
        conformers.append(nms.get_random_structure())
    conformers = np.stack(conformers)

<<<<<<< HEAD
    ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, atmlist)
    print('    -',f,len(ids),conformers.shape)
=======
    nc.setConformers(confs=conformers, types=list(spc))
    Ecmp = nc.energy() # this generates AEVs

    fpa = [i for i,s in enumerate(spc) if s in fpatoms]

    aevs = np.empty([Ngen, len(fpa) * aevsize])
    for m in range(Ngen):
        for j,a in enumerate(fpa):
            aevs[m, j * aevsize:(j + 1) * aevsize] = nc.atomicenvironments(a, m).copy()

    dm = scispc.distance.pdist(aevs, 'sqeuclidean')
    picker = rdSimDivPickers.MaxMinPicker()
    seed_list = [i for i in range(Ngen)]
    np.random.shuffle(seed_list)
    ids = list(picker.Pick(dm, Ngen, Nkep, firstPicks=list(seed_list[0:5])))
    ids.sort()
    print(f,len(ids),conformers.shape,dm.shape)
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808

    hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)