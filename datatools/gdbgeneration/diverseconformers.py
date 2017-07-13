import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync
import pyaniasetools as aat
wkdir    = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08/cv1/train0/'
cnstfile = wkdir + '../rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + '../sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

# reaction center atoms
fpatoms = ['C', 'N', 'O']
aevsize = 384

T = 400
K = 10
P = 0.25
atmlist = []

idir = '/home/jujuman/Research/extensibility_test_sets/drugbank/inputs/'
cdir = '/home/jujuman/Research/extensibility_test_sets/drugbank/confs/'

#idir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/inputs/'
#cdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_h2ocluster/h2o_cluster/confs/'

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

    if set(['S','P','Cl','F','B','Br']).isdisjoint(set(spc)):

        Ngen = K*frc.size
        Nkep = int(Ngen*P)
        Ngen = 60
        Nkep = 16

        print('    -',Nkep,'of',Ngen)

        nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=5.0E-2)

        conformers = []
        for i in range(Ngen):
            conformers.append(nms.get_random_structure())
        conformers = np.stack(conformers)

        ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, atmlist)
        print('    -',f,len(ids),conformers.shape)

        hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)