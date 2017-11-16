import pyaniasetools as pya
import gdbsearchtools as gdb
import pymolfrag as pmf

import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np

import os
import time


#--------------Parameters------------------
# Molecule file
molfile = '/home/jujuman/Research/GDB_Dimer/test/input'

# Dynamics file
xyzfile = '/home/jujuman/Research/GDB_Dimer/test/mdcrd.xyz'

# Trajectory file
trajfile = '/home/jujuman/Research/GDB_Dimer/test/traj.dat'

# Optimized structure out
optfile = '/home/jujuman/Research/GDB_Dimer/test/optmol.xyz'

wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb09_1/cv5/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

At = ['C', 'O', 'N', 'S', 'F', 'Cl'] # Hydrogens added after check

dstore = '/home/jujuman/Research/GDB_Dimer/test/'

N = 20
T = 400.0
L = 20.0
V = 0.04
dt = 0.25
Nm = 220
Nr = 15

Ni = 3000
#Ns = 100


idir = [(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/inputs/'),
        (1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s02/inputs/'),
        (1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s03/inputs/'),
        (1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size1/inputs/'),
        (1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size2/inputs/'),
        (1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/gdb11_size3/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/water_input/'),
        #(5, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s04/inputs/'),
        #(2, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s05/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_1/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_2/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s06/config_3/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_1/inputs/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/chemmbl22/config_2/inputs/'),
        #(1, '/home/jujuman/Research/MD_TEST/WeirdThing/input/input/'),
        #(1, '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/C2_test/'),
        ]

#-------------------------------------------

mols = []
difo = open(dstore + 'info_data_mddimer.nfo', 'w')
for di,id in enumerate(idir):
    files = os.listdir(id[1])
    random.shuffle(files)

    dnfo = str(di) + ' of ' + str(len(idir)) + ') dir: ' + str(id) + ' Selecting: '+str(id[0]*len(files))
    print(dnfo)
    difo.write(dnfo+'\n')

    for i in range(id[0]):
    	for n,m in enumerate(files):
        	data = hdn.read_rcdb_coordsandnm(id[1]+m)
       		mols.append(data)

dgen = pmf.dimergenerator(cnstfilecv, saefilecv, nnfprefix, 5, mols)

difo.write('Beginning dimer generation...\n')

Nt = 0
Nd = 0
for i in range(Nr):
    dgen.init_dynamics(Nm, V, L, dt, T)

    dgen.run_dynamics(Ni, xyzfile, trajfile)
    dgen.__fragmentbox__(molfile+str(i).zfill(4)+'_')

    Nt += dgen.Nt
    Nd += dgen.Nd

    print('Step (',i,') [', str(dgen.Nd), '/', str(dgen.Nt),'] generated ',len(dgen.frag_list), 'dimers...')
    difo.write('Step ('+str(i)+') ['+ str(dgen.Nd)+ '/'+ str(dgen.Nt)+'] generated '+str(len(dgen.frag_list))+'dimers...\n')

difo.write('Generated '+str(Nd)+' of '+str(Nt)+' tested dimers. Percent: ' + "{:.2f}".format(100.0*Nd/float(Nt)))
difo.close()
