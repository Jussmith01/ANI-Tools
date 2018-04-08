import hdnntools as hdn
import nmstools as nmt
import numpy as np
import os

import random

# pyneurochem
import pyNeuroChem as pync
import pyaniasetools as aat
wkdir    = '/home/jujuman/Research/DataReductionMethods/al_working_network/ANI-AL-0707.0001.0400/train3/'
cnstfile = wkdir + 'rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_wb97x-631gd.dat'
nnfdir   = wkdir + 'networks/'

# reaction center atoms
fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']
aevsize = 1008

T = 200
K = 16
#P = 0.25
atmlist = []

idir = '/home/jujuman/Research/extensibility_test_sets/COMP6v2/TripeptideS/inputs/'
cdir = '/home/jujuman/Research/extensibility_test_sets/COMP6v2/TripeptideS/confs/'

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

    if set(['P','B','Br']).isdisjoint(set(spc)):

        #Ngen = K*frc.size
        #Nkep = int(Ngen*P)
        Ngen = K*8
        Nkep = K

        print('    -',Nkep,'of',Ngen)

        nms = nmt.nmsgenerator(xyz,nmc,frc,spc,T,minfc=1.0E-3)


        conformers = []
        for i in range(Ngen):
            conformers.append(nms.get_random_structure())
        conformers = np.stack(conformers)

        ids = dc.get_divconfs_ids(conformers, spc, Ngen, Nkep, atmlist)
        print('    -',f,len(ids),conformers.shape)

        hdn.writexyzfile(cdir+f.split('.')[0]+'.xyz',conformers[ids],spc)
