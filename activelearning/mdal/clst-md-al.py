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
# dir
dstore = '/home/jujuman/Research/Cluster_AL/water/'

# Molecule file
molfile = dstore + 'watercluster'

# Dynamics file
xyzfile = dstore + 'mdcrd.xyz'

# Trajectory file
trajfile = dstore + 'traj.dat'

# Optimized structure out
optfile = dstore + 'optmol.xyz'

wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb09_1/cv5/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

At = ['C', 'O', 'N', 'S', 'F', 'Cl'] # Hydrogens added after check

N = 20
T = 300.0
L = 30.0
V = 0.04
dt = 0.1
Nm = 902
Nr = 50

Ni = 5
Ns = 1


mfile1 = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_s01/water_input/gdb11_s01-2.ipt'

#-------------------------------------------

mol = [hdn.read_rcdb_coordsandnm(mfile)]

dgen = pmf.clustergenerator(cnstfilecv, saefilecv, nnfprefix, 5, mol)

dgen.edgepad = 0.8
dgen.mindist = 1.6

difo = open(dstore + 'info_data_mddimer.nfo', 'w')
difo.write('Beginning dimer generation...\n')

Nt = 0
Nd = 0
for i in range(Nr):
    dgen.init_dynamics(Nm, V, L, dt, T)

    for j in range(Ns):
        dgen.run_dynamics(Ni, xyzfile, trajfile)
        dgen.__fragmentbox__(molfile+str(i).zfill(4) + '-' + str(j).zfill(4) + '_')
        print('Step (',i,',',j,') [', str(dgen.Nd), '/', str(dgen.Nt),'] generated ',len(dgen.frag_list), 'dimers...')
        difo.write('Step (' + str(i) + ',' + str(i) + ') [' + str(dgen.Nd) + '/' + str(dgen.Nt) + '] generated ' + str(len(dgen.frag_list)) + 'dimers...\n')
        Nt += dgen.Nt
        Nd += dgen.Nd

difo.write('Generated '+str(Nd)+' of '+str(Nt)+' tested dimers. Percent: ' + "{:.2f}".format(100.0*Nd/float(Nt)))
difo.close()
