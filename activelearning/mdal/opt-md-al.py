import pyaniasetools as pya
import gdbsearchtools as gdb

import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import random

import os

#--------------Parameters------------------
wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08_mdal01/cv2/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

At = ['C', 'O', 'N'] # Hydrogens added after check

dstore = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/h2o/mdal3/'

T = 800.0
dt = 0.5

idir = [(1.0,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/inputs/'),
        (0.1,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_1/inputs/'),
        (0.2,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s08/config_2/inputs/'),
        (0.15,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_1/inputs/'),
        (0.15,'/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s07/config_2/inputs/'),
        (1.0,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_01/inputs/'),
        (1.0,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_02/inputs/'),
        (1.0,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_03/inputs/'),
        (1.0,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_04/inputs/'),
        (1.0,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_05/inputs/'),
        (0.25,'/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_06/inputs/'),
        (1.0,'/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_03_red/inputs/'),
        (1.0,'/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_04_red/inputs/'),
        (1.0,'/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_05_red/inputs/'),
        (0.5,'/home/jujuman/Scratch/Research/GDB-11-AL-wB97x631gd/dnnts_red/dnntsgdb11_06_red/inputs/'),
        ]

#-------------------------------------------

activ = pya.moldynactivelearning(cnstfile, saefile, wkdir+'train', 5)

difo = open(dstore + 'data_info.nfo', 'w')
for di,id in enumerate(idir):
    files = os.listdir(id[1])
    random.shuffle(files)

    dnfo = str(di) + ' of ' + str(len(idir)) + ') dir: ' + str(id) + ' Selecting: '+str(id[0]*len(files))
    print(dnfo)
    difo.write(dnfo+'\n')

    for n,m in enumerate(files[0:int(id[0]*len(files))]):
        data = hdn.read_rcdb_coordsandnm(id[1]+m)
        S =  data["species"]
        print(n,') Working on',m,'...')

        # Set mols
        activ.setmol(data["coordinates"], S)

        # Generate conformations
        X = activ.generate_conformations(200, T, dt, 400, 5, dS = 0.08)

        nfo = activ._infostr_
        difo.write('  -'+m+': '+nfo+'\n')
        print(nfo)

        if X.size > 0:
            hdn.writexyzfile(dstore+m.split('.')[0]+'-'+str(di).zfill(3)+'.xyz',X,S)