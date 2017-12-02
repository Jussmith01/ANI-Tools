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
dstore = '/home/jujuman/Research/cluster_testing/clusters/'

# Molecule file
molfile = dstore + 'watercluster'

# Dynamics file
xyzfile = dstore + 'mdcrd.xyz'

# Trajectory file
trajfile = dstore + 'traj.dat'

# Optimized structure out
optfile = dstore + 'optmol.xyz'

wkdircv = '/home/jujuman/Research/DataReductionMethods/al_working_network/ANI-AL-0707.0000.0411/'
cnstfilecv = wkdircv + 'train0/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'train0/sae_wb97x-631gd.dat'
nnfprefix  = wkdircv + 'train'

At = ['C', 'O', 'N', 'S', 'F', 'Cl'] # Hydrogens added after check

T = 600.0
L = 30.0
V = 0.04
dt = 0.25
#Nm = 902
Nm = 800
Nr = 50

Ni = 5
Ns = 100


solv_file = '/home/jujuman/Research/cluster_testing/solvents/gdb11_s01-2.ipt'
solu_dirs = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/aa_chains/aa1/inputs/'


gcmddict = {'edgepad': 0.8,
            'mindist': 1.6,
            'maxsig' : 0.7,
            'Nr': Nr,
            'Nm': Nm,
            'Ni': Ni,
            'Ns': Ns,
            'dt': dt,
            'V': V,
            'L': L,
            'T': T,
            'Nembed' : 3,
            'molfile' : molfile,
            'dstore' : dstore,
            }

#-------------------------------------------

#print(solu)

solv = [hdn.read_rcdb_coordsandnm(solv_file)]
solu = [hdn.read_rcdb_coordsandnm(solu_dirs+f) for f in os.listdir(solu_dirs)]

dgen = pmf.clustergenerator(cnstfilecv, saefilecv, nnfprefix, 5, solv, solu)

dgen.generate_clusters(gcmddict)


