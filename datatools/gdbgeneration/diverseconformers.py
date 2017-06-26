import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync
import pyaniasetools as aat

wkdir    = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

# reaction center atoms
fpatoms = ['C', 'N', 'O']
aevsize = 384
T = 800
K = 10
P = 0.25
atmlist = [0, 1, 2, 3]

#idir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_05_red/inputs/'
#cdir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_05_red/confs/'

idir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/inputs/'
cdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/confs/'

files = os.listdir(idir)
files.sort()

if not os.path.exists(cdir):
    os.mkdir(cdir)

dc = aat.diverseconformers(cnstfile, saefile, nnfdir, aevsize, 0, False)

for fi,f in enumerate(files):
    print(fi,'of',len(files),') Working on:', f)
    data = hdn.read_rcdb_coordsandnm(idir+f)

    spc = data["species"]
    xyz = data["coordinates"]
    nmc = data["nmdisplacements"]
    frc = data["forceconstant"]

    Ngen = K*frc.size
    Nkep = int(Ngen*P)
    Ngen = 800
    Nkep = 400

    print('    -',Ngen,'of',Nkep)

    nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)

    conformers = []
    for i in range(Ngen):
        conformers.append(nms.get_random_structure())
    conformers = np.stack(conformers)

    ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, atmlist)
    print('    -',f,len(ids),conformers.shape)

    hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)